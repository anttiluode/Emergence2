"""Genomes: token sequences that live in a table on the GPU.

A genome is a row of integer tokens (0 = PAD). Cells do not carry tokens directly;
each cell carries the index of a row. When a birth mutates, a fresh row is written:
a copy of the parent's tokens with a point change, an insertion, a deletion or a
tandem duplication. Rows whose last cell died are reused.

Bookkeeping kept per row (it costs almost nothing and makes the history readable):
  clade  - which original seed genome this row descends from
  depth  - number of mutations since that seed
  parent - row index of the parent genome at the moment of birth
  born   - simulation step of the mutation
"""
import torch


class GenomePool:
    def __init__(self, cfg, device, gen):
        self.cfg = cfg
        self.device = device
        self.gen = gen
        G, L = cfg.max_genomes, cfg.genome_max
        self.tokens = torch.zeros(G, L, dtype=torch.long, device=device)
        self.length = torch.zeros(G, dtype=torch.long, device=device)
        self.count = torch.zeros(G, dtype=torch.long, device=device)   # live cells
        self.reserved = torch.zeros(G, dtype=torch.bool, device=device)
        self.clade = torch.full((G,), -1, dtype=torch.long, device=device)
        self.depth = torch.zeros(G, dtype=torch.long, device=device)
        self.parent = torch.full((G,), -1, dtype=torch.long, device=device)
        self.born = torch.zeros(G, dtype=torch.long, device=device)
        self.next_clade = 0
        self.total_mutations = 0
        self.blocked = 0          # mutations dropped because the table was full

    # ------------------------------------------------------------------ helpers
    def _rand(self, *shape):
        return torch.rand(*shape, generator=self.gen, device=self.device)

    def _randint(self, lo, hi, shape):
        return torch.randint(lo, hi, shape, generator=self.gen, device=self.device)

    def free_slots(self, n):
        free = torch.nonzero((self.count == 0) & (~self.reserved)).flatten()
        return free[:n]

    def live(self):
        return torch.nonzero(self.count > 0).flatten()

    # ------------------------------------------------------------- creation
    def new_random(self, n, step):
        """Write n fresh random genomes (abiogenesis). Returns their row indices."""
        cfg = self.cfg
        slots = self.free_slots(n)
        n = slots.numel()
        if n == 0:
            return slots
        L = cfg.genome_max
        toks = self._randint(1, cfg.vocab, (n, L))
        lens = torch.full((n,), cfg.genome_init_len, dtype=torch.long, device=self.device)
        pos = torch.arange(L, device=self.device)[None, :]
        toks = torch.where(pos < lens[:, None], toks, torch.zeros_like(toks))
        self.tokens[slots] = toks
        self.length[slots] = lens
        self.clade[slots] = torch.arange(self.next_clade, self.next_clade + n, device=self.device)
        self.next_clade += n
        self.depth[slots] = 0
        self.parent[slots] = -1
        self.born[slots] = step
        self.reserved[slots] = True
        return slots

    def mutate_from(self, parents, step):
        """Write one mutated copy of each parent row. Returns new rows (may be fewer
        than parents if the table is full; caller keeps the parent's row for the rest)."""
        cfg = self.cfg
        slots = self.free_slots(parents.numel())
        n = slots.numel()
        self.blocked += parents.numel() - n
        if n == 0:
            return slots
        parents = parents[:n]
        L = cfg.genome_max
        old = self.tokens[parents]                     # (n, L)
        ln = self.length[parents]                      # (n,)
        idx = torch.arange(L, device=self.device)[None, :].expand(n, L)
        op = self._rand(n)
        pos = (self._rand(n) * ln.float()).long().clamp(max=L - 1)[:, None]
        newtok = self._randint(1, cfg.vocab, (n, 1))

        # point substitution
        point = torch.where(idx == pos, newtok.expand(n, L), old)
        # insertion at pos (shift right, truncate)
        shifted_r = torch.cat([old[:, :1], old[:, :-1]], dim=1)
        insert = torch.where(idx < pos, old, torch.where(idx == pos, newtok.expand(n, L), shifted_r))
        # deletion at pos (shift left)
        shifted_l = torch.cat([old[:, 1:], torch.zeros_like(old[:, :1])], dim=1)
        delete = torch.where(idx < pos, old, shifted_l)
        # tandem duplication of a segment [a, a+k) inserted right after itself
        k = self._randint(2, 7, (n, 1))
        a = (self._rand(n, 1) * (ln[:, None] - k).clamp(min=0).float()).long()
        b = a + k
        src = torch.where(idx < b, idx, torch.where(idx < b + k, idx - k, idx - k))
        dup = torch.gather(old, 1, src.clamp(0, L - 1))
        dup = torch.where(idx < b, old, dup)

        is_point = op < 0.6
        is_ins = (op >= 0.6) & (op < 0.75)
        is_del = (op >= 0.75) & (op < 0.9)
        is_dup = op >= 0.9
        new = torch.where(is_point[:, None], point,
              torch.where(is_ins[:, None], insert,
              torch.where(is_del[:, None], delete, dup)))
        nl = ln.clone()
        nl = torch.where(is_ins, ln + 1, nl)
        nl = torch.where(is_del, ln - 1, nl)
        nl = torch.where(is_dup, ln + k[:, 0], nl)
        nl = nl.clamp(4, L)
        new = torch.where(idx < nl[:, None], new, torch.zeros_like(new))
        # a deletion that ran into the minimum length still must not leave PAD inside
        new = torch.where((idx < nl[:, None]) & (new == 0), newtok.expand(n, L), new)

        self.tokens[slots] = new
        self.length[slots] = nl
        self.clade[slots] = self.clade[parents]
        self.depth[slots] = self.depth[parents] + 1
        self.parent[slots] = parents
        self.born[slots] = step
        self.reserved[slots] = True
        self.total_mutations += n
        return slots

    def recount(self, gid_flat_alive):
        """Recompute live-cell counts from the grid; release reservations of rows that
        now have cells (they are protected by count>0 instead)."""
        self.count.zero_()
        if gid_flat_alive.numel():
            self.count.index_add_(0, gid_flat_alive, torch.ones_like(gid_flat_alive))
        self.reserved.zero_()  # rows written this step now hold cells; reservations end

    def as_text(self, row):
        """Readable token string for one genome row (for logs)."""
        n = int(self.length[row])
        return " ".join(f"{int(t):02d}" for t in self.tokens[row, :n].tolist())

    def state_dict(self):
        return {k: getattr(self, k) for k in
                ["tokens", "length", "count", "clade", "depth", "parent", "born"]} | {
                "next_clade": self.next_clade, "total_mutations": self.total_mutations,
                "blocked": self.blocked}

    def load_state_dict(self, d):
        for k in ["tokens", "length", "count", "clade", "depth", "parent", "born"]:
            src = d[k]
            getattr(self, k)[:src.shape[0]] = src      # table may have grown on resume
        self.next_clade = d["next_clade"]
        self.total_mutations = d["total_mutations"]
        self.blocked = d.get("blocked", 0)
        self.reserved.zero_()
