"""Level 2: hunters that live off the level-1 world.

A hunter is a body on the grid with an energy store. Its only food is the energy held
by living level-1 cells: each step it bites the cells in the 3x3 square around it. It
pays for living, for moving and for writing marks. At zero energy it dies. With enough
energy it splits, and the child starts with an empty memory.

All hunters share one brain, which learns while the world runs:

    patch of the world (15x15, 10 channels) + own energy, last action, last reward
        -> conv encoder -> GRU memory h -> policy (move N/E/S/W/stay x write mark or not)
                                        -> value (expected future energy)
                                        -> world model (change of the patch + reward)

The policy and value are trained by advantage actor-critic on the energy the hunter
actually gains. The world model is trained to predict what the hunter will see next.
Nothing tells the brain what an organism, a front, a ring or a sun is. Those exist in
the level-1 ground truth; probe.py asks whether h comes to encode them.

Two fields connect the levels:
  trace (involuntary) - every hunter leaves a fading smell where it is. Level-1 cells
                        perceive it as `danger`, so lineages can evolve responses.
  mark  (voluntary)   - a slow field hunters choose to write and can read back: a
                        secondary medium the hunters may use as shared memory.
"""
from dataclasses import dataclass, asdict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .world import blur

N_MOVES = 5                      # stay, N, E, S, W
N_ACT = N_MOVES * 2              # x (no mark, mark)
MDY = [0, -1, 0, 1, 0]
MDX = [0, 0, 1, 0, -1]
OBS_C = 10                       # alive, cell energy, food, 4 signals, trace, mark, hunters
WM_C = 3                         # world model predicts alive, cell energy, food


@dataclass
class HunterConfig:
    n_max: int = 512             # hunter slots
    n_start: int = 96
    n_min: int = 16              # if fewer live, respawn up to this (logged)
    patch: int = 15
    hidden: int = 256

    e_start: float = 2.0
    e_max: float = 8.0
    e_divide: float = 5.0
    metabolism: float = 0.20      # set so random-walking hunters barely break even
    move_cost: float = 0.02
    mark_cost: float = 0.01
    bite: float = 0.30           # fraction of each cell's energy taken per hunter per step
    eff: float = 0.7             # fraction of bitten energy the hunter keeps
    hazard: float = 0.0005
    disperse: float = 10.0       # a child is born this many cells (x 0.5..1) from its parent

    trace_decay: float = 0.90
    trace_diffusion: float = 0.3
    mark_decay: float = 0.997
    mark_deposit: float = 0.5

    learn: bool = True
    policy: str = "brain"        # "brain" or "random" (control)
    lr: float = 5e-4
    rollout: int = 16
    gamma: float = 0.99
    lam: float = 0.95
    ent: float = 0.03            # entropy bonus; at 0.01 the policy collapsed within ~100 updates
    vcoef: float = 0.5
    wm_coef: float = 1.0
    seed: int = 0

    @staticmethod
    def from_dict(d):
        c = HunterConfig()
        for k, v in d.items():
            if hasattr(c, k):
                setattr(c, k, v)
        return c


class Brain(nn.Module):
    def __init__(self, hc):
        super().__init__()
        P, H = hc.patch, hc.hidden
        self.enc = nn.Sequential(
            nn.Conv2d(OBS_C, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten())
        with torch.no_grad():
            n = self.enc(torch.zeros(1, OBS_C, P, P)).shape[1]
        self.proj = nn.Sequential(nn.Linear(n, 256), nn.ReLU())
        self.n_prop = 1 + N_ACT + 1
        self.gru = nn.GRUCell(256 + self.n_prop, H)
        self.pi = nn.Linear(H, N_ACT)
        self.v = nn.Linear(H, 1)
        self.wm = nn.Sequential(nn.Linear(H + N_ACT, 512), nn.ReLU(),
                                nn.Linear(512, WM_C * P * P + 1))
        # start the world model at "nothing changes", so its error starts equal to the
        # copy-last-frame baseline and any drop below it is something it has learned
        nn.init.zeros_(self.wm[-1].weight)
        nn.init.zeros_(self.wm[-1].bias)

    def core(self, obs, prop, h):
        z = self.proj(self.enc(obs))
        return self.gru(torch.cat([z, prop], 1), h)

    def heads(self, h):
        return self.pi(h), self.v(h).squeeze(1)

    def predict(self, h, a_onehot):
        return self.wm(torch.cat([h, a_onehot], 1))


class Hunters:
    def __init__(self, world, hc):
        self.w = world
        self.hc = hc
        self.dev = world.device
        cfg = world.cfg
        self.H, self.W = cfg.H, cfg.W
        self.gen = torch.Generator(device=self.dev).manual_seed(hc.seed + 777)
        torch.manual_seed(hc.seed + 778)
        N = hc.n_max
        z = lambda *s: torch.zeros(*s, device=self.dev)
        self.alive = torch.zeros(N, dtype=torch.bool, device=self.dev)
        self.y = torch.zeros(N, dtype=torch.long, device=self.dev)
        self.x = torch.zeros(N, dtype=torch.long, device=self.dev)
        self.E = z(N)
        self.age = z(N)
        self.h = z(N, hc.hidden)
        self.last_a = torch.zeros(N, dtype=torch.long, device=self.dev)
        self.last_r = z(N)
        self.trace = z(self.H, self.W)
        self.mark = z(self.H, self.W)
        self.brain = Brain(hc).to(self.dev)
        self.opt = torch.optim.Adam(self.brain.parameters(), lr=hc.lr)
        r = hc.patch // 2
        self.off = torch.arange(-r, r + 1, device=self.dev)
        self.buf = []
        self.h0 = None
        self._reset = torch.zeros(N, dtype=torch.bool, device=self.dev)
        self.stats = {"births": 0, "deaths": 0, "respawns": 0, "harvest": 0.0, "updates": 0,
                      "loss_pi": 0.0, "loss_v": 0.0, "wm_mse": 0.0, "copy_mse": 0.0,
                      "entropy": 0.0, "reward": 0.0}
        self._harvest_acc, self._steps_acc = 0.0, 0
        self.spawn(hc.n_start)

    # ------------------------------------------------------------- population
    def _rand(self, *s):
        return torch.rand(*s, generator=self.gen, device=self.dev)

    def spawn(self, n, at=None, energy=None):
        free = torch.nonzero(~self.alive).flatten()[:n]
        k = free.numel()
        if k == 0:
            return free
        if at is None:
            self.y[free] = (self._rand(k) * self.H).long().clamp(max=self.H - 1)
            self.x[free] = (self._rand(k) * self.W).long().clamp(max=self.W - 1)
        else:
            self.y[free], self.x[free] = at[0][:k], at[1][:k]
        self.E[free] = self.hc.e_start if energy is None else energy[:k]
        self.age[free] = 0
        self.h[free] = 0          # a new hunter remembers nothing
        self.last_a[free] = 0
        self.last_r[free] = 0
        self.alive[free] = True
        self._reset[free] = True
        return free

    # ------------------------------------------------------------- senses
    def hunter_map(self):
        m = torch.zeros(self.H * self.W, device=self.dev)
        idx = (self.y * self.W + self.x)[self.alive]
        m.index_add_(0, idx, torch.ones_like(idx, dtype=torch.float))
        return m.view(self.H, self.W)

    def fields(self):
        w = self.w
        return torch.cat([
            w.alive.float()[None],
            (w.energy / w.cfg.e_max)[None],
            w.food.clamp(0, 1.5)[None],
            (w.sig / 3.0),
            (self.trace / 3.0).clamp(0, 1)[None],
            self.mark[None],
            (self.hunter_map() / 3.0).clamp(0, 1)[None],
        ])                                                    # (OBS_C, H, W)

    def patches(self, F_, y=None, x=None):
        y = self.y if y is None else y
        x = self.x if x is None else x
        ys = (y[:, None] + self.off[None]) % self.H
        xs = (x[:, None] + self.off[None]) % self.W
        idx = ys[:, :, None] * self.W + xs[:, None, :]          # (N, P, P)
        C = F_.shape[0]
        return F_.view(C, -1)[:, idx].permute(1, 0, 2, 3)        # (N, C, P, P)

    def prop(self):
        return torch.cat([(self.E / self.hc.e_max)[:, None],
                          F.one_hot(self.last_a, N_ACT).float(),
                          self.last_r[:, None]], 1)

    # ------------------------------------------------------------- one step
    def step(self):
        hc, w = self.hc, self.w
        obs = self.patches(self.fields())
        prop = self.prop()
        h_in = self.h * (~self._reset).float()[:, None]
        with torch.no_grad():
            h = self.brain.core(obs, prop, h_in)
            logits, value = self.brain.heads(h)

        # learning happens on a full rollout, before acting on the new state
        if hc.learn and hc.policy == "brain" and len(self.buf) == hc.rollout:
            self.train(obs, value)

        if not self.buf:
            self.h0 = self.h.clone()
        if hc.policy == "random":
            a = (self._rand(hc.n_max) * N_ACT).long().clamp(max=N_ACT - 1)
        else:
            a = torch.distributions.Categorical(logits=logits).sample()
        move, write = a % N_MOVES, a // N_MOVES
        valid = self.alive.clone()
        reset = self._reset.clone()
        self._reset.zero_()

        # ---- move
        dy = torch.tensor(MDY, device=self.dev)[move]
        dx = torch.tensor(MDX, device=self.dev)[move]
        self.y = torch.where(self.alive, (self.y + dy) % self.H, self.y)
        self.x = torch.where(self.alive, (self.x + dx) % self.W, self.x)

        # ---- bite the cells around the new position (exactly conserving)
        gain = self.eat()

        # ---- write marks (the voluntary secondary field)
        wr = self.alive & (write == 1)
        if wr.any():
            idx = (self.y * self.W + self.x)[wr]
            mf = self.mark.view(-1)
            mf[idx] = (mf[idx] + hc.mark_deposit).clamp(max=1.0)

        # ---- costs, reward
        cost = hc.metabolism + hc.move_cost * (move > 0).float() + hc.mark_cost * write.float()
        cost = cost * self.alive.float()
        self.E = (self.E + gain - cost).clamp(max=hc.e_max)
        reward = (gain - cost) * valid.float()
        self.age += self.alive.float()

        # ---- death
        die = self.alive & ((self.E <= 0) | (self._rand(hc.n_max) < hc.hazard))
        self.alive &= ~die
        self.E = self.E * self.alive.float()
        self.stats["deaths"] += int(die.sum())

        # ---- record the transition (obs, action, reward ...) for learning
        if hc.learn and hc.policy == "brain":
            self.buf.append(dict(obs=obs, prop=prop, reset=reset, a=a, r=reward,
                                 done=die, valid=valid))

        # ---- births (child next to the parent, empty memory)
        par = torch.nonzero(self.alive & (self.E > hc.e_divide)).flatten()
        if par.numel():
            free = int((~self.alive).sum())
            par = par[:free]
            if par.numel():
                half = self.E[par] * 0.5
                self.E[par] = half
                # the child disperses: born a short random hop away, so it does not
                # starve on the patch its parent is already eating
                k = par.numel()
                ang = self._rand(k) * 2 * math.pi
                dist = hc.disperse * (0.5 + 0.5 * self._rand(k))
                cy = (self.y[par] + (dist * torch.sin(ang)).round().long()) % self.H
                cx = (self.x[par] + (dist * torch.cos(ang)).round().long()) % self.W
                kids = self.spawn(k, at=(cy, cx), energy=half)
                self.stats["births"] += kids.numel()
        n_alive = int(self.alive.sum())
        if n_alive < hc.n_min:
            self.stats["respawns"] += hc.n_min - n_alive
            self.spawn(hc.n_min - n_alive)

        self.h = h
        self.last_a = a
        self.last_r = reward

        # ---- the trace: an involuntary smell that level-1 cells perceive as danger
        self.trace = hc.trace_decay * ((1 - hc.trace_diffusion) * self.trace
                                       + hc.trace_diffusion * blur(self.trace))
        self.trace = self.trace + self.hunter_map()
        self.mark = self.mark * hc.mark_decay
        w.danger = (self.trace / 3.0).clamp(0, 1)

        self._harvest_acc += float(gain.sum())
        self._steps_acc += 1

    def eat(self):
        hc, w = self.hc, self.w
        H, W = self.H, self.W
        r = torch.tensor([-1, 0, 1], device=self.dev)
        ys = (self.y[:, None, None] + r[None, :, None]) % H
        xs = (self.x[:, None, None] + r[None, None, :]) % W
        idx = (ys * W + xs).reshape(self.y.numel(), 9)            # (N, 9)
        live = self.alive[:, None].expand_as(idx)
        mouths = torch.zeros(H * W, device=self.dev)
        mouths.index_add_(0, idx[live], torch.ones(int(live.sum()), device=self.dev))
        frac = 1 - (1 - hc.bite) ** mouths                        # fraction taken per site
        Ef = w.energy.view(-1)
        taken = frac * Ef * w.alive.view(-1).float()
        w.energy = (Ef - taken).view(H, W)
        # a cell bitten nearly empty dies on the world's next step
        w.energy = torch.where(w.energy < 0.01, torch.zeros_like(w.energy), w.energy)
        share = taken / mouths.clamp(min=1)
        gain = (share[idx] * live.float()).sum(1) * hc.eff
        return gain

    # ------------------------------------------------------------- learning
    def train(self, obs_next, v_boot):
        hc = self.hc
        T = len(self.buf)
        b = self.buf
        h = self.h0
        logits_l, v_l, pred_l = [], [], []
        for t in range(T):
            h = self.brain.core(b[t]["obs"], b[t]["prop"], h * (~b[t]["reset"]).float()[:, None])
            lg, v = self.brain.heads(h)
            logits_l.append(lg); v_l.append(v)
            pred_l.append(self.brain.predict(h, F.one_hot(b[t]["a"], N_ACT).float()))
        V = torch.stack(v_l)                                        # (T, N)
        r = torch.stack([x["r"] for x in b])
        done = torch.stack([x["done"] for x in b]).float()
        valid = torch.stack([x["valid"] for x in b]).float()
        with torch.no_grad():
            adv = torch.zeros_like(r)
            last = torch.zeros_like(v_boot)
            nxt = v_boot
            for t in reversed(range(T)):
                delta = r[t] + hc.gamma * nxt * (1 - done[t]) - V[t].detach()
                last = delta + hc.gamma * hc.lam * (1 - done[t]) * last
                adv[t] = last
                nxt = V[t].detach()
            ret = adv + V.detach()
            m = valid > 0
            if m.sum() > 1:
                a_n = (adv - adv[m].mean()) / (adv[m].std() + 1e-6)
            else:
                a_n = adv
        logits = torch.stack(logits_l)
        logp_all = F.log_softmax(logits, -1)
        a = torch.stack([x["a"] for x in b])
        logp = logp_all.gather(-1, a[..., None]).squeeze(-1)
        ent = -(logp_all.exp() * logp_all).sum(-1)
        nv = valid.sum().clamp(min=1)
        loss_pi = -(logp * a_n * valid).sum() / nv
        loss_v = (F.smooth_l1_loss(V, ret, reduction="none") * valid).sum() / nv
        loss_ent = -(ent * valid).sum() / nv

        # world model: predict the next patch's (alive, energy, food) and the reward
        P2 = hc.patch * hc.patch
        targets = [x["obs"][:, :WM_C] for x in b[1:]] + [obs_next[:, :WM_C]]
        wm_mse, copy_mse, n_wm = 0.0, 0.0, 0.0
        loss_wm = 0.0
        for t in range(T):
            ok = valid[t] * (1 - done[t])
            if ok.sum() == 0:
                continue
            tgt = targets[t].reshape(-1, WM_C * P2)
            cur = b[t]["obs"][:, :WM_C].reshape(-1, WM_C * P2)
            pred = pred_l[t]
            # the model predicts the CHANGE; "nothing changes" is its zero point
            nxt = cur + pred[:, :-1]
            e = ((nxt - tgt) ** 2).mean(1) + (pred[:, -1] - r[t]) ** 2
            loss_wm = loss_wm + (e * ok).sum()
            with torch.no_grad():
                wm_mse += float((((nxt - tgt) ** 2).mean(1) * ok).sum())
                copy_mse += float((((cur - tgt) ** 2).mean(1) * ok).sum())
                n_wm += float(ok.sum())
        loss_wm = loss_wm / max(n_wm, 1.0)
        loss = loss_pi + hc.vcoef * loss_v + hc.ent * loss_ent + hc.wm_coef * loss_wm
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.brain.parameters(), 1.0)
        self.opt.step()
        s = self.stats
        s["updates"] += 1
        k = 0.1 if s["updates"] > 1 else 1.0
        for key, val in [("loss_pi", loss_pi.item()), ("loss_v", loss_v.item()),
                         ("entropy", -loss_ent.item()), ("reward", float((r * valid).sum() / nv)),
                         ("wm_mse", wm_mse / max(n_wm, 1)), ("copy_mse", copy_mse / max(n_wm, 1))]:
            s[key] = (1 - k) * s[key] + k * val
        self.buf = []

    # ------------------------------------------------------------- measurement
    def metrics(self):
        a = self.alive
        n = int(a.sum())
        m = {"hunters": n,
             "hunter_energy": float(self.E[a].mean()) if n else 0.0,
             "hunter_age": float(self.age[a].mean()) if n else 0.0,
             "harvest_per_step": self._harvest_acc / max(self._steps_acc, 1),
             "harvest_per_hunter": self._harvest_acc / max(self._steps_acc, 1) / max(n, 1),
             "hunter_births": self.stats["births"], "hunter_deaths": self.stats["deaths"],
             "hunter_respawns": self.stats["respawns"], "updates": self.stats["updates"],
             "reward": self.stats["reward"], "entropy": self.stats["entropy"],
             "wm_mse": self.stats["wm_mse"], "copy_mse": self.stats["copy_mse"],
             "mark_mean": float(self.mark.mean())}
        self._harvest_acc, self._steps_acc = 0.0, 0
        return m

    # ------------------------------------------------------------- persistence
    def state_dict(self):
        return {"hc": asdict(self.hc), "alive": self.alive.cpu(), "y": self.y.cpu(), "x": self.x.cpu(),
                "E": self.E.cpu(), "age": self.age.cpu(), "h": self.h.cpu(),
                "last_a": self.last_a.cpu(), "last_r": self.last_r.cpu(),
                "trace": self.trace.cpu(), "mark": self.mark.cpu(),
                "brain": self.brain.state_dict(), "opt": self.opt.state_dict(),
                "stats": dict(self.stats)}

    def load_state_dict(self, d, load_opt=True):
        for k in ["alive", "y", "x", "E", "age", "h", "last_a", "last_r", "trace", "mark"]:
            setattr(self, k, d[k].to(self.dev))
        self.brain.load_state_dict(d["brain"])
        if load_opt:
            try:
                self.opt.load_state_dict(d["opt"])
            except Exception:
                pass
        self.stats.update(d.get("stats", {}))
        self._reset = torch.zeros(self.hc.n_max, dtype=torch.bool, device=self.dev)
        self.buf = []
