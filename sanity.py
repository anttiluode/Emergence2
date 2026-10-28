"""Does the hunters' brain learn at all? A controlled check on a frozen world.

    python sanity.py runs/x/ckpt_0010000.pt --policy brain  --steps 4000
    python sanity.py runs/x/ckpt_0010000.pt --policy random --steps 4000

The level-1 world is frozen (no world steps, no regrowth); bitten-out cells disappear and
every 400 steps the prey is restored. Hunters never die or reproduce and pay nothing,
so the reward is exactly the energy they bite. A brain that learns should bite more per
step than random walkers on the same prey. Works with Emergence or Emergence 2
checkpoints.
"""
import argparse

import torch

from emergence.config import Config
from emergence.world import World
from emergence.hunters import Hunters, HunterConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--policy", default="brain", choices=["brain", "random"])
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--hunters", type=int, default=64)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    d = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    if "world" in d:
        d = d["world"]
    cfg = Config.from_dict(d["config"])
    cfg.seed_cells, cfg.device = 0, a.device
    w = World(cfg)
    w.load_state_dict(d)
    E0, A0 = w.energy.clone(), w.alive.clone()
    hc = HunterConfig(n_max=a.hunters, n_start=a.hunters, metabolism=0.0, move_cost=0.0,
                      mark_cost=0.0, hazard=0.0, e_max=1e9, e_divide=1e9, policy=a.policy)
    hu = Hunters(w, hc)
    for i in range(a.steps):
        if i % 400 == 0:
            w.energy.copy_(E0)
            w.alive.copy_(A0)
        hu.step()
        w.alive = w.alive & (w.energy > 0)
        if (i + 1) % 400 == 0:
            m = hu.metrics()
            print(f"{a.policy:>6} t={i + 1:<6} harvest/hunter/step={m['harvest_per_hunter']:.4f}  "
                  f"entropy={m['entropy']:.2f}  world-model={m['wm_mse']:.4f}  copy={m['copy_mse']:.4f}",
                  flush=True)


if __name__ == "__main__":
    main()
