"""The world: a toroidal grid where every occupied site is one cell.

Each step, every living cell
  1. perceives its own internal state, its energy and age, the food and chemical
     signals under it and their gradients, the mean internal state of its
     neighbours, and how many neighbours are kin / strangers / empty;
  2. runs one shared cell network whose hidden units are modulated by its lineage's
     phenotype (gamma, beta, action biases expressed by the translator from the
     genome tokens);
  3. acts: updates its internal state, eats, shares energy with kin, steals energy
     from strangers, emits signals, and may divide into a neighbouring empty site.

The physics is fixed and conserves energy except where stated:
  - energy enters only as food (regrowing faster under slowly drifting "suns");
  - eating moves food into a cell;
  - sharing moves energy between kin cells exactly;
  - stealing moves energy between strangers, losing (1 - attack_eff) of it;
  - living, signalling and attacking cost energy (lost);
  - dying returns part of the remaining energy to the food under the cell.

Kin = same body. A body is the set of cells descended from one founding cell through
births that stayed attached. Whether a birth stays attached is itself a decision of the
parent's genome (the "detach" action), so multicellular organisms are not defined into
the world: they appear only if lineages that hold together do better. A mutation inside
a body does not cut it apart (a somatic mutation); the body just becomes a mosaic.
"""
import math
import torch

from .genome import GenomePool
from .translator import Translator

DY = [-1, 0, 1, 0]   # N E S W
DX = [0, 1, 0, -1]
OPP = [2, 3, 0, 1]


def nb(x, d):
    """Value at the neighbour in direction d, for every site (toroidal)."""
    return torch.roll(x, shifts=(-DY[d], -DX[d]), dims=(-2, -1))


def blur(x):
    """3x3 box average (toroidal)."""
    s = x
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                s = s + torch.roll(x, shifts=(dy, dx), dims=(-2, -1))
    return s / 9.0


def grads(x):
    gx = 0.5 * (torch.roll(x, -1, -1) - torch.roll(x, 1, -1))
    gy = 0.5 * (torch.roll(x, -1, -2) - torch.roll(x, 1, -2))
    return gx, gy


class World:
    N_ACTIONS_EXTRA = 4 + 1 + 1 + 1 + 4 + 1  # emit, eat, share, divide, dir, attack

    def __init__(self, cfg):
        self.cfg = cfg
        dev = cfg.device
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(dev)
        self.gen = torch.Generator(device=self.device).manual_seed(cfg.seed)
        if cfg.max_genomes <= 0:
            # a big world holds far more distinct genomes; a full table silently
            # blocks mutation, so size it from the area (1 slot per 16 sites, min 32768)
            cfg.max_genomes = max(32768, (cfg.H * cfg.W) // 16)
        H, W, C, Q = cfg.H, cfg.W, cfg.n_state, cfg.n_signal

        self.n_actions = C + Q + 1 + 1 + 1 + 4 + 1 + 1
        self.n_percept = C + 2 + 3 + 3 * Q + C + 4 + 1 + 3   # +3: danger and its gradient
        self.state_dim = C + 4

        # ---- shared cell network (fixed, seeded) -------------------------------
        # The first 38 input columns and W2 are drawn exactly as in Emergence (level 1),
        # so a level-1 world loaded here keeps its cell physics; the 3 danger columns are
        # drawn afterwards and contribute nothing while no hunter has been near.
        g = torch.Generator().manual_seed(cfg.seed + 99)
        n_old = self.n_percept - 3
        W1_old = torch.randn(cfg.hidden, n_old, generator=g) * 1.5 / math.sqrt(n_old)
        self.W2 = (torch.randn(self.n_actions, cfg.hidden, generator=g) * 1.5
                   / math.sqrt(cfg.hidden)).to(self.device)
        W1_new = torch.randn(cfg.hidden, 3, generator=g) * 1.5 / math.sqrt(n_old)
        self.W1 = torch.cat([W1_old, W1_new], 1).to(self.device)

        # ---- genomes and their translator ------------------------------------
        self.pool = GenomePool(cfg, self.device, self.gen)
        self.tr = Translator(cfg, self.n_actions, self.state_dim, cfg.seed).to(self.device)
        self.tr.calibrate(2048, self.device, self.gen)
        G = cfg.max_genomes
        self.gamma = torch.zeros(G, cfg.hidden, device=self.device)
        self.beta = torch.zeros(G, cfg.hidden, device=self.device)
        self.obias = torch.zeros(G, self.n_actions, device=self.device)
        self.color = torch.zeros(G, 3, device=self.device)
        self.lin_state = torch.zeros(G, self.state_dim, device=self.device)

        # ---- grid fields ---------------------------------------------------------
        z = lambda *s: torch.zeros(*s, device=self.device)
        self.alive = torch.zeros(H, W, dtype=torch.bool, device=self.device)
        self.gid = torch.full((H, W), -1, dtype=torch.long, device=self.device)
        self.body = torch.full((H, W), -1, dtype=torch.long, device=self.device)
        self.next_body = 0
        self.energy = z(H, W)
        self.S = z(C, H, W)
        self.age = z(H, W)
        self.food = z(H, W) + 0.5
        # Level 2 writes here: a smell of hunters that cells can perceive (0 = none)
        self.danger = z(H, W)
        self.sig = z(Q, H, W)
        yy, xx = torch.meshgrid(torch.arange(H, device=self.device, dtype=torch.float32),
                                torch.arange(W, device=self.device, dtype=torch.float32),
                                indexing="ij")
        self.yy, self.xx = yy, xx
        self.sun_pos = torch.rand(cfg.n_suns, 2, generator=self.gen, device=self.device) \
            * torch.tensor([H, W], device=self.device, dtype=torch.float32)
        ang = torch.rand(cfg.n_suns, generator=self.gen, device=self.device) * 6.283
        self.sun_vel = torch.stack([torch.sin(ang), torch.cos(ang)], 1)
        self.sun = z(H, W)

        self.t = 0
        self.births = 0
        self.deaths = 0
        self.reseeds = 0
        self.history = []          # list of metric dicts
        self._update_sun()
        self.seed_life(cfg.seed_cells)

    # ------------------------------------------------------------------ utils
    def rand(self, *shape):
        return torch.rand(*shape, generator=self.gen, device=self.device)

    def _write_phenotype(self, rows, state):
        if rows.numel() == 0:
            return
        g, b, o, c = self.tr.express(self.pool.tokens[rows], self.pool.length[rows], state)
        self.gamma[rows], self.beta[rows], self.obias[rows], self.color[rows] = g, b, o, c
        self.lin_state[rows] = state

    def seed_life(self, n):
        """Abiogenesis: n random cells, each with its own random genome."""
        cfg = self.cfg
        empty = torch.nonzero(~self.alive.flatten()).flatten()
        if empty.numel() == 0:
            return
        pick = empty[torch.randperm(empty.numel(), generator=self.gen, device=self.device)[:n]]
        rows = self.pool.new_random(pick.numel(), self.t)
        pick = pick[:rows.numel()]
        self._write_phenotype(rows, torch.zeros(rows.numel(), self.state_dim, device=self.device))
        a, gd, e, ag = self.alive.view(-1), self.gid.view(-1), self.energy.view(-1), self.age.view(-1)
        a[pick] = True
        gd[pick] = rows
        self.body.view(-1)[pick] = torch.arange(self.next_body, self.next_body + pick.numel(),
                                                device=self.device)
        self.next_body += pick.numel()
        e[pick] = 0.8
        ag[pick] = 0
        S = self.S.view(cfg.n_state, -1)
        S[:, pick] = 0.1 * torch.randn(cfg.n_state, pick.numel(), generator=self.gen, device=self.device)
        self.pool.recount(self.gid[self.alive])

    def _update_sun(self):
        cfg = self.cfg
        H, W = cfg.H, cfg.W
        jitter = (torch.rand(cfg.n_suns, 2, generator=self.gen, device=self.device) - 0.5) * 0.15
        self.sun_vel = self.sun_vel + jitter
        self.sun_vel = self.sun_vel / self.sun_vel.norm(dim=1, keepdim=True)
        size = torch.tensor([H, W], device=self.device, dtype=torch.float32)
        self.sun_pos = (self.sun_pos + self.sun_vel * cfg.sun_speed * size) % size
        r = cfg.sun_radius * min(H, W)
        dy = (self.yy[None] - self.sun_pos[:, 0, None, None]).abs()
        dx = (self.xx[None] - self.sun_pos[:, 1, None, None]).abs()
        dy = torch.minimum(dy, H - dy)
        dx = torch.minimum(dx, W - dx)
        self.sun = torch.exp(-(dy ** 2 + dx ** 2) / (2 * r * r)).sum(0).clamp(max=1.0)

    # ------------------------------------------------------------------ step
    @torch.no_grad()
    def step(self):
        cfg = self.cfg
        C, Q = cfg.n_state, cfg.n_signal
        alive = self.alive
        aF = alive.float()
        gid = self.gid
        E = self.energy

        # ---------------- perception ---------------------------------------------
        nb_alive = torch.stack([nb(alive, d) for d in range(4)])
        nb_gid = torch.stack([nb(gid, d) for d in range(4)])
        body = self.body
        nb_body = torch.stack([nb(body, d) for d in range(4)])
        kin = nb_alive & (nb_body == body[None]) & alive[None]          # same organism
        alien = nb_alive & (nb_body != body[None]) & alive[None]        # any other organism
        twin = alien & (nb_gid == gid[None])                            # other organism, same genome
        n_occ = nb_alive.float().sum(0)
        S_nb = sum(nb(self.S, d) for d in range(4)) / n_occ.clamp(min=1)
        fgx, fgy = grads(self.food)
        sgx, sgy = grads(self.sig)
        dgx, dgy = grads(self.danger)
        P = torch.cat([
            self.S,
            (E / cfg.e_max)[None],
            torch.tanh(self.age / cfg.age_onset)[None],
            self.food[None], fgx[None] * 10, fgy[None] * 10,
            self.sig, sgx * 5, sgy * 5,
            S_nb,
            kin.float().sum(0)[None] / 4, alien.float().sum(0)[None] / 4,
            twin.float().sum(0)[None] / 4,
            (4 - n_occ)[None] / 4,
            torch.ones_like(E)[None],
            self.danger[None], dgx[None] * 5, dgy[None] * 5,
        ])

        # ---------------- the cell network (only for living cells) ----------------
        idx = torch.nonzero(alive.flatten()).flatten()
        A = self.n_actions
        out = torch.zeros(A, cfg.H * cfg.W, device=self.device)
        if idx.numel():
            g = gid.flatten()[idx]
            p = P.view(P.shape[0], -1)[:, idx].T
            h = p @ self.W1.T
            h = torch.tanh(h * (1 + self.gamma[g]) + self.beta[g])
            o = h @ self.W2.T + self.obias[g]
            out[:, idx] = o.T
        out = out.view(A, cfg.H, cfg.W)
        k = 0
        dS = out[k:k + C]; k += C
        emit = torch.sigmoid(out[k:k + Q]) * aF; k += Q
        eat = torch.sigmoid(out[k]) * aF; k += 1
        share = torch.sigmoid(out[k]) * aF; k += 1
        divide = torch.sigmoid(out[k]) * aF; k += 1
        dirlog = out[k:k + 4]; k += 4
        attack = torch.sigmoid(out[k]) * aF; k += 1
        detach = torch.sigmoid(out[k]) * aF; k += 1

        # ---------------- internal state ------------------------------------------
        self.S = (0.7 * self.S + 0.3 * torch.tanh(dS)) * aF

        # ---------------- eating ----------------------------------------------------
        up = eat * cfg.eat_max * self.food
        self.food = self.food - up
        E = E + up

        # ---------------- sharing with kin (exact) ---------------------------------
        out_d = (share * cfg.share_rate * E)[None] * kin.float()
        E = E - out_d.sum(0) + sum(nb(out_d[OPP[d]], d) for d in range(4))

        # ---------------- stealing from strangers -----------------------------------
        nbE = torch.stack([nb(E, d) for d in range(4)])
        req = (attack * cfg.attack_rate)[None] * nbE * alien.float()
        req_tot = sum(nb(req[OPP[d]], d) for d in range(4))
        scale = (E / (req_tot + 1e-9)).clamp(max=1.0)
        take = req * torch.stack([nb(scale, d) for d in range(4)])
        lost = sum(nb(take[OPP[d]], d) for d in range(4))
        E = E - lost + cfg.attack_eff * take.sum(0)

        # ---------------- costs ------------------------------------------------------
        E = E - aF * (cfg.metabolism + cfg.emit_cost * emit.sum(0) + cfg.attack_cost * attack)
        E = E.clamp(max=cfg.e_max) * aF

        # ---------------- death -------------------------------------------------------
        ramp = ((self.age - cfg.age_onset) / cfg.age_span).clamp(0, 1)
        hazard = cfg.hazard_base + (cfg.hazard_old - cfg.hazard_base) * ramp
        die = alive & ((E <= 0) | (self.rand(cfg.H, cfg.W) < hazard))
        self.food = (self.food + die.float() * E.clamp(min=0) * cfg.decompose).clamp(max=1.5)
        alive = alive & ~die
        aF = alive.float()
        E = E * aF
        self.S = self.S * aF
        gid = torch.where(alive, gid, torch.full_like(gid, -1))
        body = torch.where(alive, body, torch.full_like(body, -1))
        self.deaths += die.sum()

        # ---------------- division ---------------------------------------------------
        want = alive & (E > cfg.e_divide) & (self.rand(cfg.H, cfg.W) < divide)
        gum = -torch.log(-torch.log(self.rand(4, cfg.H, cfg.W).clamp(1e-9, 1 - 1e-9)))
        d_choice = (dirlog + gum).argmax(0)
        empty = ~alive
        pr = self.rand(cfg.H, cfg.W)
        claims = torch.stack([nb(want & (d_choice == OPP[d]), d) & empty for d in range(4)])
        prio = torch.stack([nb(pr, d) for d in range(4)])
        prio = torch.where(claims, prio, torch.full_like(prio, -1.0))
        win = prio.argmax(0)
        newborn = claims.any(0)
        won = torch.zeros_like(alive)
        for d in range(4):
            won |= want & (d_choice == d) & nb(newborn, d) & (nb(win, d) == OPP[d])
        # gather parent properties into the child site
        sel = win[None]
        pE = torch.gather(torch.stack([nb(E, d) for d in range(4)]), 0, sel)[0]
        pG = torch.gather(torch.stack([nb(gid, d) for d in range(4)]), 0, sel)[0]
        pB = torch.gather(torch.stack([nb(body, d) for d in range(4)]), 0, sel)[0]
        pD = torch.gather(torch.stack([nb(detach, d) for d in range(4)]), 0, sel)[0]
        # a child stays part of its parent's body unless the parent lets it go
        leaves = newborn & (self.rand(cfg.H, cfg.W) < pD)
        fresh = self.next_body + torch.arange(cfg.H * cfg.W, device=self.device).view(cfg.H, cfg.W)
        pB = torch.where(leaves, fresh, pB)
        self.next_body += cfg.H * cfg.W
        pS = torch.stack([nb(self.S, d) for d in range(4)])           # (4, C, H, W)
        pS = torch.gather(pS, 0, sel[:, None].expand(1, C, cfg.H, cfg.W))[0]
        childE = (0.5 * pE - cfg.birth_cost) * newborn.float()
        E = torch.where(won, 0.5 * E, E)
        E = torch.where(newborn, childE, E)
        gid = torch.where(newborn, pG, gid)
        body = torch.where(newborn, pB, body)
        self.S = torch.where(newborn[None], pS, self.S)
        self.age = torch.where(newborn, torch.zeros_like(self.age), self.age)
        alive = alive | newborn
        self.births += newborn.sum()

        # ---------------- mutation at birth --------------------------------------------
        mut = newborn & (self.rand(cfg.H, cfg.W) < cfg.mutation_rate)
        midx = torch.nonzero(mut.flatten()).flatten()
        if midx.numel():
            parents = gid.flatten()[midx]
            rows = self.pool.mutate_from(parents, self.t)
            n = rows.numel()
            if n:
                self._write_phenotype(rows, self.lin_state[parents[:n]])
                gflat = gid.view(-1)
                gflat[midx[:n]] = rows

        # ---------------- environment ----------------------------------------------------
        if self.t % 4 == 0:
            self._update_sun()
        rate = cfg.food_base + cfg.food_sun * self.sun
        self.food = self.food + rate * (1 - self.food)
        self.food = self.food + cfg.food_diffusion * (blur(self.food) - self.food)
        self.food = self.food.clamp(0, 1.5)
        sd = cfg.signal_diffusion
        self.sig = cfg.signal_decay * ((1 - sd) * self.sig + sd * blur(self.sig)) + 0.5 * emit
        self.sig = self.sig.clamp(0, 3)

        # ---------------- commit -------------------------------------------------------------
        self.alive, self.gid, self.energy, self.body = alive, gid, E, body
        self.age = (self.age + 1) * alive.float()
        self.pool.recount(gid[alive])
        self.t += 1

        if self.t % cfg.express_every == 0:
            self.regulate()
        if self.t % 10 == 0 and int(alive.sum()) < cfg.min_population:
            self.reseeds += 1
            self.seed_life(cfg.reseed_cells)

    # ------------------------------------------------------------- regulation
    @torch.no_grad()
    def regulate(self):
        """Feed each lineage's lived state back into the translator and re-express."""
        cfg = self.cfg
        G = cfg.max_genomes
        a = self.alive.flatten()
        if not a.any():
            return
        g = self.gid.flatten()[a]
        nbk = torch.stack([nb(self.alive, d) & (nb(self.body, d) == self.body) for d in range(4)])
        feats = torch.cat([
            self.S.view(cfg.n_state, -1)[:, a],
            (self.energy.flatten()[a] / cfg.e_max)[None],
            self.food.flatten()[a][None],
            (nbk.float().sum(0).flatten()[a] / 4)[None],
        ]).T                                                # (n, state_dim-1)
        sums = torch.zeros(G, feats.shape[1], device=self.device).index_add_(0, g, feats)
        cnt = self.pool.count.float()
        rows = torch.nonzero(cnt > 0).flatten()
        mean = sums[rows] / cnt[rows, None]
        size = torch.log1p(cnt[rows])[:, None] / 6
        self._write_phenotype(rows, torch.cat([mean, size], 1))

    # ------------------------------------------------------------- interaction
    def _disc(self, y, x, r):
        dy = (self.yy - y).abs(); dx = (self.xx - x).abs()
        dy = torch.minimum(dy, self.cfg.H - dy); dx = torch.minimum(dx, self.cfg.W - dx)
        return dy * dy + dx * dx <= r * r

    @torch.no_grad()
    def wipe(self, y, x, r):
        """Kill every cell in a disc (for damage / repair experiments)."""
        m = self._disc(y, x, r) & self.alive
        self.alive &= ~m
        self.energy *= self.alive.float()
        self.S *= self.alive.float()
        self.gid = torch.where(self.alive, self.gid, torch.full_like(self.gid, -1))
        self.body = torch.where(self.alive, self.body, torch.full_like(self.body, -1))
        self.pool.recount(self.gid[self.alive])

    @torch.no_grad()
    def feed(self, y, x, r, amount=1.0):
        self.food = torch.where(self._disc(y, x, r), torch.full_like(self.food, amount), self.food)

    # ------------------------------------------------------------- measurement
    @torch.no_grad()
    def metrics(self):
        cfg = self.cfg
        a = self.alive
        pop = int(a.sum())
        cnt = self.pool.count
        live = cnt > 0
        m = {"t": self.t, "population": pop, "genomes": int(live.sum()),
             "births": int(self.births), "deaths": int(self.deaths),
             "mutations": self.pool.total_mutations, "mutations_blocked": self.pool.blocked,
             "genome_table_used": float(live.sum()) / cfg.max_genomes,
             "reseeds": self.reseeds,
             "food": float(self.food.mean())}
        if pop:
            c = cnt[live].float()
            m["clades"] = int(torch.unique(self.pool.clade[live]).numel())
            m["mean_genome_len"] = float((self.pool.length[live].float() * c).sum() / c.sum())
            m["mean_depth"] = float((self.pool.depth[live].float() * c).sum() / c.sum())
            m["max_depth"] = int(self.pool.depth[live].max())
            m["top_genome_share"] = float(c.max() / c.sum())
            pnorm = c / c.sum()
            m["genome_entropy"] = float(-(pnorm * pnorm.log()).sum())
            occ = torch.stack([nb(a, d) for d in range(4)])
            kin = torch.stack([nb(a, d) & (nb(self.body, d) == self.body) for d in range(4)])
            k = (kin.float().sum(0))[a]
            o = (occ.float().sum(0))[a]
            m["kin_neighbours"] = float(k.mean())
            m["stranger_neighbours"] = float((o - k).mean())
            _, bsz = torch.unique(self.body[a], return_counts=True)
            bsz = bsz.float()
            m["bodies"] = int(bsz.numel())
            m["body_size_cellweighted"] = float((bsz * bsz).sum() / bsz.sum())
            m["max_body"] = int(bsz.max())
            m["multicell_fraction"] = float(bsz[bsz >= 8].sum() / bsz.sum())
            m["mean_energy"] = float(self.energy[a].mean())
            m["mean_age"] = float(self.age[a].mean())
        return m

    # ------------------------------------------------------------- persistence
    def state_dict(self):
        keys = ["alive", "gid", "body", "energy", "S", "age", "food", "sig", "danger", "sun_pos", "sun_vel",
                "gamma", "beta", "obias", "color", "lin_state"]
        d = {k: getattr(self, k).cpu() for k in keys}
        d["pool"] = {k: (v.cpu() if torch.is_tensor(v) else v)
                     for k, v in self.pool.state_dict().items()}
        d["next_body"] = self.next_body
        d["t"], d["births"], d["deaths"], d["reseeds"] = self.t, int(self.births), int(self.deaths), self.reseeds
        d["config"] = self.cfg.__dict__.copy()
        d["gen_state"] = self.gen.get_state()
        return d

    def load_state_dict(self, d):
        for k in ["alive", "gid", "body", "energy", "S", "age", "food", "sig", "sun_pos", "sun_vel"]:
            setattr(self, k, d[k].to(self.device))
        # a level-1 (Emergence) checkpoint has no danger field
        self.danger = d["danger"].to(self.device) if "danger" in d else torch.zeros_like(self.food)
        for k in ["gamma", "beta", "obias", "color", "lin_state"]:   # per-genome tables
            src = d[k].to(self.device)
            getattr(self, k)[:src.shape[0]] = src     # a resumed world may get a bigger table
        self.pool.load_state_dict({k: (v.to(self.device) if torch.is_tensor(v) else v)
                                   for k, v in d["pool"].items()})
        self.next_body = d.get("next_body", 0)
        self.t, self.births, self.deaths, self.reseeds = d["t"], d["births"], d["deaths"], d["reseeds"]
        try:
            self.gen.set_state(d["gen_state"])
        except Exception:
            pass
        self._update_sun()
