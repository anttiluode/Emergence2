"""Open the hunters' brain and check it against the world's answer key.

    python probe.py runs/night1/ckpt_0500000.pt --steps 3000

Level 1 records everything that is hidden from the hunters: which body each cell belongs
to, which founding clade, how old it is, where the suns are. The hunters only ever see a
15x15 patch of cells, energy, food, signals and marks. So we can ask the Othello question
directly: does the hunter's memory h encode things it was never shown?

For each hidden quantity we fit a ridge-regression probe and report test R^2 (fit on the
first 70% of the run, tested on the last 30%) from three inputs:

  latent   the trained brain's memory h
  raw      the hunter's raw observation (the full 15x15x10 patch + own energy)
  random   the memory h of an untrained brain with the same architecture, run on
           exactly the same observations (does structure come from learning, or from
           any recurrent net integrating over time?)

A quantity the latent predicts better than both raw and random is information the brain
built for itself: integrated over time, or inferred from what it saw. A linear probe that
succeeds is evidence the brain carries the quantity, not proof that it uses it; the
intervention test (editing h and watching behaviour) is the stronger step and is not
implemented here yet.
"""
import argparse
import json

import torch
import torch.nn.functional as F

from emergence.config import Config
from emergence.world import World
from emergence.hunters import Hunters, HunterConfig, Brain

LOOKAHEAD = 20


def body_sizes(w):
    """Size of the body each site belongs to (0 for empty)."""
    a = w.alive.flatten()
    b = w.body.flatten()
    out = torch.zeros_like(b, dtype=torch.float)
    if a.any():
        u, inv, cnt = torch.unique(b[a], return_inverse=True, return_counts=True)
        out[a] = cnt[inv].float()
    return out.view(w.cfg.H, w.cfg.W)


def labels(w, hu, idx):
    """Hidden quantities for the hunters in idx."""
    H, W = w.cfg.H, w.cfg.W
    y, x = hu.y[idx], hu.x[idx]
    site = y * W + x
    bs = body_sizes(w).flatten()
    r = torch.tensor([-1, 0, 1], device=w.device)
    ys = (y[:, None, None] + r[None, :, None]) % H
    xs = (x[:, None, None] + r[None, None, :]) % W
    nb = (ys * W + xs).reshape(-1, 9)
    alive = w.alive.flatten()
    body = w.body.flatten()
    same = ((body[nb] == body[site][:, None]) & alive[nb]).float().sum(1) - alive[site].float()
    occ = alive[nb].float().sum(1) - alive[site].float()
    patch_alive = hu.patches(w.alive.float()[None], y, x)[:, 0]
    clade_map = torch.where(w.alive, w.pool.clade[w.gid.clamp(min=0)], torch.full_like(w.gid, -1))
    pc = hu.patches(clade_map.float()[None], y, x)[:, 0].reshape(len(idx), -1)
    n_clades = torch.tensor([float(torch.unique(row[row >= 0]).numel()) for row in pc], device=w.device)
    ages = hu.patches((w.age * w.alive.float())[None], y, x)[:, 0].reshape(len(idx), -1)
    mean_age = ages.sum(1) / patch_alive.reshape(len(idx), -1).sum(1).clamp(min=1)
    return {
        "sun_here": w.sun.flatten()[site],
        "log_body_size_here": torch.log1p(bs[site]),
        "same_body_neighbours": torch.where(occ > 0, same / occ.clamp(min=1), torch.zeros_like(occ)),
        "clades_in_view": n_clades,
        "mean_cell_age_in_view": mean_age,
    }


def ridge_r2(Xtr, ytr, Xte, yte):
    mu, sd = Xtr.mean(0), Xtr.std(0).clamp(min=1e-6)
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    ym = ytr.mean()
    n = Xtr.shape[0]
    nv = max(1, n // 5)
    Xa, ya, Xv, yv = Xtr[:-nv], ytr[:-nv], Xtr[-nv:], ytr[-nv:]
    best, best_a = -1e9, 1.0
    for alpha in [0.1, 1, 10, 100, 1000, 10000, 100000]:
        wgt = torch.linalg.solve(Xa.T @ Xa + alpha * torch.eye(Xa.shape[1]), Xa.T @ (ya - ym))
        r2 = 1 - ((Xv @ wgt + ym - yv) ** 2).sum() / ((yv - yv.mean()) ** 2).sum().clamp(min=1e-9)
        if r2 > best:
            best, best_a = float(r2), alpha
    wgt = torch.linalg.solve(Xtr.T @ Xtr + best_a * torch.eye(Xtr.shape[1]), Xtr.T @ (ytr - ym))
    r2 = 1 - ((Xte @ wgt + ym - yte) ** 2).sum() / ((yte - yte.mean()) ** 2).sum().clamp(min=1e-9)
    return float(r2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--every", type=int, default=5, help="sample hunters every k steps")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    d = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = Config.from_dict(d["world"]["config"])
    cfg.device, cfg.seed_cells = a.device, 0
    w = World(cfg)
    w.load_state_dict(d["world"])
    hc = HunterConfig.from_dict(d["hunters"]["hc"])
    hc.learn = False                                   # observe, don't train, while probing
    hu = Hunters(w, hc)
    hu.load_state_dict(d["hunters"], load_opt=False)
    fresh = Brain(hc).to(w.device)                     # untrained twin, same architecture
    h_fresh = torch.zeros_like(hu.h)

    feats = {"latent": [], "raw": [], "random": []}
    labs, pending = {}, []
    for t in range(a.steps):
        # the untrained twin sees exactly what the hunters see
        obs = hu.patches(hu.fields())
        prop = hu.prop()
        with torch.no_grad():
            h_fresh = fresh.core(obs, prop, h_fresh * (~hu._reset).float()[:, None])
        w.step()
        hu.step()
        h_fresh = h_fresh * hu.alive.float()[:, None]
        if t % a.every == 0:
            idx = torch.nonzero(hu.alive).flatten()
            if idx.numel():
                feats["latent"].append(hu.h[idx].cpu())
                o = hu.patches(hu.fields())[idx].reshape(idx.numel(), -1)
                feats["raw"].append(torch.cat([o, hu.E[idx, None]], 1).cpu())
                feats["random"].append(h_fresh[idx].cpu())
                for k, v in labels(w, hu, idx).items():
                    labs.setdefault(k, []).append(v.cpu())
                later = labs.setdefault("cells_in_view_later", [])
                later.append(None)
                pending.append((t + LOOKAHEAD, hu.y[idx].clone(), hu.x[idx].clone(), len(later) - 1))
        # fill in the future-prey label once its time has come
        while pending and pending[0][0] <= t:
            _, py, px, k = pending.pop(0)
            labs["cells_in_view_later"][k] = hu.patches(w.alive.float()[None], py, px)[:, 0] \
                .reshape(py.numel(), -1).sum(1).cpu()
    n_ok = sum(1 for v in labs["cells_in_view_later"] if v is not None)
    for k in feats:
        feats[k] = feats[k][:n_ok]
    for k in labs:
        labs[k] = labs[k][:n_ok]

    X = {k: torch.cat(v).float() for k, v in feats.items()}
    n = X["latent"].shape[0]
    cut = int(n * 0.7)
    print(f"{n} samples from {a.steps} steps; train on first {cut}, test on the rest\n")
    print(f"{'hidden quantity':<26}{'latent':>9}{'raw':>9}{'random':>9}")
    res = {}
    for k, v in labs.items():
        y = torch.cat(v).float()
        row = {}
        for name, Xf in X.items():
            row[name] = ridge_r2(Xf[:cut], y[:cut], Xf[cut:], y[cut:])
        res[k] = row
        print(f"{k:<26}{row['latent']:>9.3f}{row['raw']:>9.3f}{row['random']:>9.3f}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"samples": n, "steps": a.steps, "r2_test": res}, f, indent=2)


if __name__ == "__main__":
    main()
