"""Plot one or more runs' metrics.csv side by side.

    python plot.py runs/night1                     # one run
    python plot.py runs/base runs/noreg runs/nomut  # compare arms of an experiment

Writes <first run>/metrics.png (or --out). Every panel is a continuous measure,
on purpose: a threshold score can make a smooth change look like a sudden jump.
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PANELS = [
    ("population", "living cells"),
    ("bodies", "organisms (bodies)"),
    ("body_size_cellweighted", "body size a random cell lives in"),
    ("multicell_fraction", "cells in bodies of >= 8"),
    ("genomes", "distinct genomes alive"),
    ("clades", "founding lineages still alive"),
    ("mean_depth", "mutations since founding (mean)"),
    ("max_depth", "mutations since founding (max)"),
    ("mean_genome_len", "genome length (tokens)"),
    ("genome_entropy", "genome diversity (nats)"),
    ("stranger_neighbours", "strangers touching a cell"),
    ("mean_energy", "energy per cell"),
    # level 2
    ("hunters", "hunters alive"),
    ("harvest_per_hunter", "harvest per hunter per step"),
    ("reward", "hunter reward per step"),
    ("entropy", "policy entropy"),
    ("wm_mse", "world-model error"),
    ("copy_mse", "copy-last-frame error"),
]


def load(path):
    with open(os.path.join(path, "metrics.csv")) as f:
        rows = list(csv.DictReader(f))
    out = {}
    for k in rows[0]:
        try:
            out[k] = [float(r[k]) if r.get(k) not in ("", None) else float("nan") for r in rows]
        except ValueError:
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    data = [(os.path.basename(os.path.normpath(r)), load(r)) for r in a.runs]
    fig, axes = plt.subplots(3, 6, figsize=(26, 10))
    for ax, (k, title) in zip(axes.flat, PANELS):
        for name, d in data:
            if k in d:
                ax.plot(d["t"], d[k], label=name, lw=1.2)
        ax.set_title(title, fontsize=10)
        ax.tick_params(labelsize=8)
    axes.flat[0].legend(fontsize=8)
    fig.tight_layout()
    out = a.out or os.path.join(a.runs[0], "metrics.png")
    fig.savefig(out, dpi=110)
    print("wrote", out)


if __name__ == "__main__":
    main()
