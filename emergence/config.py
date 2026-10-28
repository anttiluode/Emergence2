"""All the knobs of the world in one place.

Everything that is a *law of the world* lives here. Everything that is a *choice of a
lineage* lives in its genome and is decided by evolution.
"""
from dataclasses import dataclass, asdict, field
import json


@dataclass
class Config:
    # ---- world -------------------------------------------------------------
    H: int = 384
    W: int = 384
    seed: int = 0
    device: str = "auto"          # "auto" -> cuda if available

    # ---- cell anatomy ----------------------------------------------------------
    n_state: int = 8               # private internal channels per cell
    n_signal: int = 4              # diffusing chemical channels (shared medium)
    hidden: int = 48               # hidden units of the cell's rule network

    # ---- genomes ------------------------------------------------------------
    vocab: int = 32                # token alphabet (0 = PAD)
    genome_max: int = 48           # max tokens per genome
    genome_init_len: int = 16
    max_genomes: int = 0           # genome table slots; 0 = auto (scales with world area)
    mutation_rate: float = 0.03    # probability a birth carries a mutated genome

    # ---- translator (the transformer that reads genomes) ----------------------
    d_model: int = 96
    n_heads: int = 4
    n_layers: int = 3
    regulation: float = 0.6        # strength of the lineage-state token (0 = no feedback)
    express_every: int = 25        # steps between re-expression of all live genomes

    # ---- energy economics ---------------------------------------------------
    e_max: float = 2.0
    e_divide: float = 0.6          # energy needed to divide
    birth_cost: float = 0.05
    metabolism: float = 0.006      # per cell per step
    eat_max: float = 0.30          # max fraction of local food eaten per step
    emit_cost: float = 0.004       # per unit of signal emitted
    attack_rate: float = 0.12      # fraction of a victim's energy one attack can take
    attack_cost: float = 0.004
    attack_eff: float = 0.7        # fraction of stolen energy the attacker keeps
    share_rate: float = 0.20       # max fraction of energy sent to each kin neighbour
    decompose: float = 0.6         # fraction of a dead cell's energy returned as food

    # ---- ageing -------------------------------------------------------------
    hazard_base: float = 0.0004
    age_onset: int = 300
    age_span: int = 1500           # hazard rises to hazard_old over this span
    hazard_old: float = 0.02

    # ---- environment --------------------------------------------------------
    food_base: float = 0.0015      # background regrowth rate
    food_sun: float = 0.03         # extra regrowth under moving "suns"
    n_suns: int = 7
    sun_radius: float = 0.09       # fraction of world size
    sun_speed: float = 0.0004      # fraction of world size per step
    food_diffusion: float = 0.05
    signal_decay: float = 0.92
    signal_diffusion: float = 0.5

    # ---- abiogenesis --------------------------------------------------------
    seed_cells: int = 3000         # random cells with random genomes at t=0
    min_population: int = 200      # if life falls below this, reseed (logged)
    reseed_cells: int = 1000

    def to_json(self):
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_dict(d):
        c = Config()
        for k, v in d.items():
            if hasattr(c, k):
                setattr(c, k, v)
        return c
