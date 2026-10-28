"""Read a saved world: who is alive, what their genomes say, how their bodies look.

    python inspect_world.py runs/night1/ckpt_0200000.pt --top 15

For each of the most common genomes: cell count, number of bodies it forms and their
size, founding clade, mutations since founding, genome tokens, and the action biases
the translator expressed for it. Then the ancestry chain of the most common genome,
followed back through parent rows for as long as those rows were not reused.
"""
import argparse

import torch

from emergence.config import Config
from emergence.world import World


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()
    d = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    if "world" in d:                 # an Emergence 2 checkpoint holds both levels
        d = d["world"]
    cfg = Config.from_dict(d["config"])
    cfg.device = "cpu"
    cfg.seed_cells = 0
    w = World(cfg)
    w.load_state_dict(d)
    m = w.metrics()
    print(f"t={w.t}  " + "  ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                   for k, v in m.items() if k != "t"))
    p = w.pool
    C, Q = cfg.n_state, cfg.n_signal
    names = {"eat": C + Q, "share": C + Q + 1, "divide": C + Q + 2, "attack": C + Q + 7,
             "detach": C + Q + 8}
    rows = torch.argsort(p.count, descending=True)[:a.top]
    print(f"\n{'genome':>7} {'cells':>6} {'bodies':>6} {'mean body':>9} {'clade':>6} "
          f"{'depth':>5} {'len':>4} " + " ".join(f"{n:>6}" for n in names) + "   tokens")
    for r in rows.tolist():
        if p.count[r] == 0:
            break
        cells = w.alive & (w.gid == r)
        b = torch.unique(w.body[cells], return_counts=True)[1].float()
        biases = " ".join(f"{float(w.obias[r, i]):+6.2f}" for i in names.values())
        print(f"{r:>7} {int(p.count[r]):>6} {b.numel():>6} {float(b.mean()):>9.1f} "
              f"{int(p.clade[r]):>6} {int(p.depth[r]):>5} {int(p.length[r]):>4} {biases}   "
              f"{p.as_text(r)}")
    r = int(rows[0])
    print(f"\nancestry of genome {r} (back through parent rows while they are intact):")
    chain = 0
    while r >= 0 and chain < 60:
        print(f"  #{r:<6} born t={int(p.born[r]):<8} depth {int(p.depth[r]):<4} {p.as_text(r)}")
        par = int(p.parent[r])
        if par < 0 or int(p.born[par]) > int(p.born[r]) or int(p.clade[par]) != int(p.clade[r]):
            if par >= 0:
                print("  (parent row was reused after this birth; chain ends here)")
            break
        r = par
        chain += 1


if __name__ == "__main__":
    main()
