"""The translator: a transformer that reads a genome and expresses a phenotype.

This is the "AI" between the genes and the body. It never trains. Its weights are fixed
at birth of the world (seeded), like the genetic code and the ribosome: the same
reading machinery for every lineage. What evolves is the text it is given.

Input to the transformer, per genome:
    [ STATE ] tok_1 tok_2 ... tok_n  PAD ...
The STATE token is a projection of the lineage's current condition (mean internal state
of its cells, energy, food under them, how crowded with kin they are, population size).
That is the feedback path: what the body is living through changes how the genome is
read the next time. Its strength is cfg.regulation; 0 switches feedback off, which is the
control for any claim that regulation matters.

Output, per genome (all bounded):
    gamma  (hidden)   multiplicative gain on each hidden unit of the cell rule
    beta   (hidden)   bias on each hidden unit
    obias  (outputs)  bias on each action of the cell rule
    color  (3)        how the viewer paints this lineage (a readout of the same
                      phenotype vector, so similar phenotypes look similar)
"""
import torch
import torch.nn as nn


N_OUT_EXTRA = 3  # colour


class Translator(nn.Module):
    def __init__(self, cfg, n_actions, state_dim, seed):
        super().__init__()
        self.cfg = cfg
        self.n_actions = n_actions
        self.state_dim = state_dim
        g = torch.Generator().manual_seed(seed + 1234)
        d = cfg.d_model
        self.tok = nn.Embedding(cfg.vocab, d)
        self.pos = nn.Parameter(torch.zeros(cfg.genome_max + 1, d))
        self.state_proj = nn.Linear(state_dim, d)
        self.state_tag = nn.Parameter(torch.zeros(d))
        layer = nn.TransformerEncoderLayer(d, cfg.n_heads, 2 * d, dropout=0.0,
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, cfg.n_layers, enable_nested_tensor=False)
        self.out_dim = 2 * cfg.hidden + n_actions + N_OUT_EXTRA
        self.head = nn.Linear(d, self.out_dim)
        # deterministic random init from the world seed
        with torch.no_grad():
            for p in self.parameters():
                if p.dim() >= 2:
                    p.copy_(torch.randn(p.shape, generator=g) / (p.shape[-1] ** 0.5))
                else:
                    p.copy_(torch.randn(p.shape, generator=g) * 0.1)
            self.tok.weight.copy_(torch.randn(self.tok.weight.shape, generator=g))
            self.pos.copy_(torch.randn(self.pos.shape, generator=g) * 0.5)
            self.state_tag.copy_(torch.randn(d, generator=g))
        self.register_buffer("mu", torch.zeros(self.out_dim))
        self.register_buffer("sd", torch.ones(self.out_dim))
        for p in self.parameters():
            p.requires_grad_(False)

    def raw(self, tokens, lengths, state):
        cfg = self.cfg
        B, L = tokens.shape
        x = self.tok(tokens) + self.pos[1:L + 1]
        s = (self.state_tag + cfg.regulation * self.state_proj(state)) + self.pos[0]
        x = torch.cat([s[:, None, :], x], dim=1)
        pad = torch.arange(L, device=tokens.device)[None, :] >= lengths[:, None]
        mask = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=tokens.device), pad], 1)
        y = self.enc(x, src_key_padding_mask=mask)
        keep = (~mask).float()[..., None]
        pooled = (y * keep).sum(1) / keep.sum(1)
        return self.head(pooled)

    @torch.no_grad()
    def calibrate(self, n, device, gen):
        """Standardise each output over random genomes, so that a random transformer
        gives phenotypes that actually differ from genome to genome."""
        cfg = self.cfg
        toks = torch.randint(1, cfg.vocab, (n, cfg.genome_max), generator=gen, device=device)
        lens = torch.randint(8, cfg.genome_max + 1, (n,), generator=gen, device=device)
        pos = torch.arange(cfg.genome_max, device=device)[None]
        toks = torch.where(pos < lens[:, None], toks, torch.zeros_like(toks))
        st = torch.randn(n, self.state_dim, generator=gen, device=device) * 0.5
        z = self.raw(toks, lens, st)
        self.mu.copy_(z.mean(0))
        self.sd.copy_(z.std(0).clamp(min=1e-4))

    @torch.no_grad()
    def express(self, tokens, lengths, state, chunk=4096):
        outs = []
        for i in range(0, tokens.shape[0], chunk):
            z = self.raw(tokens[i:i + chunk], lengths[i:i + chunk], state[i:i + chunk])
            outs.append((z - self.mu) / self.sd)
        z = torch.cat(outs) if outs else torch.zeros(0, self.out_dim, device=tokens.device)
        h, a = self.cfg.hidden, self.n_actions
        gamma = 0.9 * torch.tanh(z[:, :h])
        beta = 1.5 * torch.tanh(z[:, h:2 * h])
        obias = 1.5 * torch.tanh(z[:, 2 * h:2 * h + a])
        color = torch.sigmoid(1.6 * z[:, 2 * h + a:2 * h + a + 3])
        return gamma, beta, obias, color
