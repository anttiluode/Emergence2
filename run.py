"""Emergence 2 - a world of cells (level 1) and hunters that live off it (level 2).

    python run.py                          # live viewer, 512x512; hunters arrive at t=1500
    python run.py --width 1024 --height 640
    python run.py --hunters-at 0           # release hunters immediately
    python run.py --load ../Emergence/runs/x/ckpt_0100000.pt   # bring hunters to an evolved level-1 world
    python run.py --headless --steps 300000 --out runs/night1 --video
    python run.py --set food_base=0.003 --hset metabolism=0.25 --hset learn=0
    python probe.py runs/night1/ckpt_0300000.pt                 # open the hunters' brain

Viewer keys
    1..9     view: life, bodies, clades, food, signals, state, energy, age, hunter fields
    SPACE    pause            + / -   steps per frame (or zoom-box size while choosing)
    Z        zoom box: aim it, + / - to size it, left click to zoom in (can repeat)
    BACKSPACE     zoom out one level      0   whole world      arrows  pan while zoomed
    left drag     wipe cells (damage)      mouse wheel  brush size
    right click   inspect the genome under the cursor
    F + left drag pour food
    J        release hunters now / drop 16 more under the mouse
    L        learning on/off        X  show/hide hunters
    R        abiogenesis: 500 random new cells
    S        save checkpoint           P   screenshot          V  start/stop video
    H        hide/show panel           Q / ESC  quit
"""
import argparse
import csv
import json
import os
import time

import numpy as np
import torch

from emergence.config import Config
from emergence.hunters import HunterConfig
from emergence.sim import Sim, VIEWS


def _parse(kvs, obj):
    out = {}
    for kv in kvs or []:
        k, v = kv.split("=", 1)
        old = getattr(obj, k)
        out[k] = (v.lower() in ("1", "true")) if isinstance(old, bool) else type(old)(v)
    return out


def build(args):
    cfg = Config()
    cfg.H = cfg.W = 512
    if args.size:
        cfg.H = cfg.W = args.size
    if args.width:
        cfg.W = args.width
    if args.height:
        cfg.H = args.height
    cfg.seed = args.seed
    wo = _parse(args.set, cfg)
    ho = _parse(args.hset, HunterConfig())
    if args.load:
        sim = Sim.load(args.load, cfg.device, wo, ho)
        if args.hunters_at >= 0 and sim.hu is None:
            sim.hunters_at = max(sim.w.t, args.hunters_at)
        print(f"loaded {args.load} at t={sim.w.t}"
              + ("" if sim.hu is None else f" with {int(sim.hu.alive.sum())} hunters"))
    else:
        for k, v in wo.items():
            setattr(cfg, k, v)
        hc = HunterConfig()
        hc.seed = args.seed
        for k, v in ho.items():
            setattr(hc, k, v)
        sim = Sim(cfg, hc, args.hunters_at if args.hunters_at >= 0 else 1500)
    w = sim.w
    cfg = w.cfg
    print(f"device={w.device}  world={cfg.H}x{cfg.W}  "
          f"cell net {w.n_percept}->{cfg.hidden}->{w.n_actions}  "
          f"translator d={cfg.d_model} x{cfg.n_layers} layers  hunters at t={sim.hunters_at}")
    return sim


def fmt(m):
    keys = ["t", "population", "bodies", "body_size_cellweighted", "genomes", "clades",
            "mean_depth", "hunters", "hunter_energy", "harvest_per_hunter",
            "reward", "wm_mse", "copy_mse", "updates"]
    return "  ".join(f"{k}={m[k]:.3g}" if isinstance(m.get(k), float) else f"{k}={m.get(k)}"
                     for k in keys if k in m)


HUNTER_KEYS = ["hunters", "hunter_energy", "hunter_age", "harvest_per_step", "harvest_per_hunter",
               "hunter_births", "hunter_deaths", "hunter_respawns", "updates", "reward",
               "entropy", "wm_mse", "copy_mse", "mark_mean"]


# ---------------------------------------------------------------------- headless
def headless(sim, args):
    w = sim.w
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        f.write(w.cfg.to_json())
    writer = None
    if args.video:
        import imageio
        writer = imageio.get_writer(os.path.join(args.out, "life.mp4"), fps=30, quality=8,
                                    macro_block_size=1)
    csv_path = os.path.join(args.out, "metrics.csv")
    fh = open(csv_path, "a", newline="")
    cw = None
    t0, s0 = time.time(), w.t
    end = w.t + args.steps
    while w.t < end:
        sim.step()
        if w.t % args.log_every == 0:
            m = sim.metrics()
            m["steps_per_sec"] = (w.t - s0) / (time.time() - t0)
            if cw is None:
                keys = list(w.metrics().keys()) + HUNTER_KEYS + ["steps_per_sec"]
                cw = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore", restval="")
                if fh.tell() == 0:
                    cw.writeheader()
            cw.writerow(m)
            fh.flush()
            print(fmt(m) + f"  ({m['steps_per_sec']:.0f} steps/s)", flush=True)
        if writer is not None and w.t % args.video_every == 0:
            writer.append_data(sim.render("life"))
        if w.t % args.snap_every == 0:
            from PIL import Image
            tiles = np.concatenate([sim.render(v) for v in ["life", "bodies", "food", "hunter fields"]], 1)
            Image.fromarray(tiles).save(os.path.join(args.out, f"snap_{w.t:07d}.png"))
        if w.t % args.ckpt_every == 0:
            torch.save(sim.state_dict(), os.path.join(args.out, f"ckpt_{w.t:07d}.pt"))
    torch.save(sim.state_dict(), os.path.join(args.out, f"ckpt_{w.t:07d}.pt"))
    if writer is not None:
        writer.close()
    fh.close()


# ---------------------------------------------------------------------- viewer
LABELS = [("population", "cells"), ("bodies", "organisms"),
          ("body_size_cellweighted", "body size (per cell)"), ("max_body", "largest body"),
          ("multicell_fraction", "in bodies >= 8"), ("genomes", "genomes alive"),
          ("clades", "founding clades"), ("mean_depth", "mutations (mean)"),
          ("max_depth", "mutations (max)"), ("mean_genome_len", "genome length"),
          ("kin_neighbours", "body-mate neighbours"), ("stranger_neighbours", "stranger neighbours"),
          ("top_genome_share", "top genome share"), ("mean_energy", "energy / cell"),
          ("mean_age", "age / cell"), ("mutations", "mutations total"),
          ("mutations_blocked", "mutations blocked"), ("genome_table_used", "genome table used"),
          ("reseeds", "reseeds"),
          (None, "-- level 2: hunters --"),
          ("hunters", "hunters"), ("hunter_energy", "energy / hunter"),
          ("harvest_per_hunter", "harvest / hunter"), ("hunter_age", "age / hunter"),
          ("hunter_births", "hunter births"), ("hunter_deaths", "hunter deaths"),
          ("hunter_respawns", "hunter respawns"), ("updates", "brain updates"),
          ("reward", "reward / step"), ("entropy", "policy entropy"),
          ("wm_mse", "world-model error"), ("copy_mse", "  vs copy-last-frame"),
          ("mark_mean", "marks written")]
def viewer(sim, args):
    w = sim.w
    import pygame
    pygame.init()
    cfg = w.cfg
    info = pygame.display.Info()
    # Fit everything on the screen: leave room for the taskbar and the title bar, use
    # a second panel column when the stats would not fit in one, and shrink the world
    # view below 1 pixel per cell when it is bigger than the screen (zoom shows detail).
    sw, sh = (info.current_w, info.current_h) if info.current_w > 0 else (1920, 1080)
    avail_w, avail_h = sw - 40, sh - 110
    n_lines = 7 + len(LABELS)
    extra_h = 5 * 40 + 140                      # sparklines + genome inspector
    # pick the largest line height (12..16) that lets everything fit in one column
    line_h = int(max(12, min(16, (avail_h - extra_h - 16) / n_lines)))
    text_h = n_lines * line_h + 16
    two_col = text_h + extra_h > avail_h
    panel = 760 if two_col else 380
    if args.scale:
        scale = float(args.scale)
    else:
        scale = min(avail_h / cfg.H, (avail_w - panel) / cfg.W)
        scale = float(int(scale)) if scale >= 1 else scale
    VW, VH = max(1, int(cfg.W * scale)), max(1, int(cfg.H * scale))   # viewport in pixels
    need_h = text_h if two_col else text_h + extra_h
    win_h = min(avail_h, max(VH, need_h))
    os.environ.setdefault("SDL_VIDEO_CENTERED", "1")
    screen = pygame.display.set_mode((VW + panel, win_h))
    pygame.display.set_caption("Emergence 2")
    font = pygame.font.SysFont("consolas,menlo,monospace", line_h - 2)
    small = pygame.font.SysFont("consolas,menlo,monospace", 12)
    clock = pygame.time.Clock()

    view, paused, spf, brush, show_panel = "life", False, 2, 10, True
    cams = [(0, 0, cfg.H, cfg.W)]                     # zoom stack: (y0, x0, h, w) in cells
    selecting = False                                 # zoom box follows the mouse
    box_w = max(16, cfg.W // 4)                       # zoom box width in cells
    feeding, inspect, rec = False, None, None
    hist = {k: [] for k in ["population", "body_size_cellweighted", "hunters",
                            "harvest_per_hunter", "wm_mse"]}
    last_m, t_rate, s_rate, rate = sim.metrics(), time.time(), w.t, 0.0
    t_metrics = 0.0
    os.makedirs(args.out, exist_ok=True)

    def to_world(pos):
        y0, x0, h, wd = cams[-1]
        return int(y0 + pos[1] * h / VH), int(x0 + pos[0] * wd / VW)

    def box_rect(pos):
        """Zoom box (y0, x0, h, w) in world cells, centred on the mouse, kept inside
        the current view and with the world's aspect ratio."""
        cy0, cx0, ch, cw = cams[-1]
        bw = int(box_w if box_w < cw else max(8, cw // 2))   # inside a zoom: half the view
        bh = max(4, int(round(bw * cfg.H / cfg.W)))
        bh = min(bh, ch)
        y, x = to_world(pos)
        y0 = int(np.clip(y - bh // 2, cy0, cy0 + ch - bh))
        x0 = int(np.clip(x - bw // 2, cx0, cx0 + cw - bw))
        return y0, x0, bh, bw

    def pan(dy, dx):
        y0, x0, h, wd = cams[-1]
        y0 = int(np.clip(y0 + dy * max(1, h // 8), 0, cfg.H - h))
        x0 = int(np.clip(x0 + dx * max(1, wd // 8), 0, cfg.W - wd))
        cams[-1] = (y0, x0, h, wd)

    zoom_click = False
    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                k = ev.key
                if k in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif pygame.K_1 <= k <= pygame.K_9:
                    view = VIEWS[k - pygame.K_1]
                elif k == pygame.K_SPACE:
                    paused = not paused
                elif k == pygame.K_z:
                    selecting = not selecting
                    if selecting and box_w >= cams[-1][3]:
                        box_w = max(8, cams[-1][3] // 2)
                elif k == pygame.K_BACKSPACE:
                    if len(cams) > 1:
                        cams.pop()
                    selecting = False
                elif k in (pygame.K_0, pygame.K_KP0):
                    del cams[1:]
                    selecting = False
                elif k in (pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN):
                    pan({pygame.K_UP: -1, pygame.K_DOWN: 1}.get(k, 0),
                        {pygame.K_LEFT: -1, pygame.K_RIGHT: 1}.get(k, 0))
                elif k in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    if selecting:
                        box_w = int(min(cfg.W, box_w * 1.25 + 1))
                    else:
                        spf = min(64, spf * 2)
                elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    if selecting:
                        box_w = int(max(8, box_w / 1.25))
                    else:
                        spf = max(1, spf // 2)
                elif k == pygame.K_r:
                    w.seed_life(500)
                elif k == pygame.K_j:
                    if sim.hu is None:
                        sim.release()
                    else:
                        my_, mx_ = to_world(pygame.mouse.get_pos())
                        n = 16
                        sim.hu.spawn(n, at=(torch.full((n,), my_ % cfg.H, device=w.device),
                                            torch.full((n,), mx_ % cfg.W, device=w.device)))
                elif k == pygame.K_l and sim.hu is not None:
                    sim.hu.hc.learn = not sim.hu.hc.learn
                    sim.hu.buf = []
                elif k == pygame.K_x:
                    sim.show_hunters = not sim.show_hunters
                elif k == pygame.K_s:
                    p = os.path.join(args.out, f"ckpt_{w.t:07d}.pt")
                    torch.save(sim.state_dict(), p)
                    print("saved", p)
                elif k == pygame.K_p:
                    p = os.path.join(args.out, f"shot_{w.t:07d}.png")
                    pygame.image.save(screen, p)
                    print("saved", p)
                elif k == pygame.K_f:
                    feeding = not feeding
                elif k == pygame.K_h:
                    show_panel = not show_panel
                elif k == pygame.K_v:
                    if rec is None:
                        import imageio
                        p = os.path.join(args.out, f"video_{w.t:07d}.mp4")
                        rec = imageio.get_writer(p, fps=30, quality=8, macro_block_size=1)
                        print("recording", p)
                    else:
                        rec.close(); rec = None
                        print("recording stopped")
            elif ev.type == pygame.MOUSEWHEEL:
                brush = int(np.clip(brush + ev.y * 2, 2, 80))
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and selecting:
                if ev.pos[0] < VW and ev.pos[1] < VH:
                    cams.append(box_rect(ev.pos))
                selecting = False
                zoom_click = True
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 3:
                y, x = to_world(ev.pos)
                if 0 <= y < cfg.H and 0 <= x < cfg.W and bool(w.alive[y, x]):
                    inspect = int(w.gid[y, x])
        buttons = pygame.mouse.get_pressed()
        if not buttons[0]:
            zoom_click = False          # the click that zoomed must not also wipe
        if buttons[0] and not selecting and not zoom_click and pygame.mouse.get_pos()[0] < VW:
            y, x = to_world(pygame.mouse.get_pos())
            if 0 <= y < cfg.H and 0 <= x < cfg.W:
                (w.feed if feeding else w.wipe)(y, x, brush)

        if not paused:
            for _ in range(spf):
                sim.step()
        if time.time() - t_metrics > 0.4:
            t_metrics = time.time()
            last_m = sim.metrics()
            for k in hist:
                hist[k].append(last_m.get(k, 0))
                hist[k] = hist[k][-300:]
        now = time.time()
        if now - t_rate > 1:
            rate = (w.t - s_rate) / (now - t_rate)
            t_rate, s_rate = now, w.t

        cy0, cx0, ch, cw = cams[-1]
        img = sim.render(view, region=(cy0, cy0 + ch, cx0, cx0 + cw))
        surf = pygame.surfarray.make_surface(np.ascontiguousarray(img.transpose(1, 0, 2)))
        if (cw, ch) != (VW, VH):
            if cw > VW or ch > VH:
                surf = pygame.transform.smoothscale(surf, (VW, VH))   # shrinking: average
            else:
                surf = pygame.transform.scale(surf, (VW, VH))         # enlarging: crisp cells
        if rec is not None:
            rec.append_data(pygame.surfarray.array3d(surf).transpose(1, 0, 2))
        screen.fill((14, 14, 18))
        screen.blit(surf, (0, 0))
        mx, my = pygame.mouse.get_pos()
        screen.set_clip((0, 0, VW, VH))
        if mx < VW and my < VH:
            if selecting:
                by0, bx0, bh, bw = box_rect((mx, my))
                rx = (bx0 - cx0) * VW / cw
                ry = (by0 - cy0) * VH / ch
                pygame.draw.rect(screen, (255, 230, 120),
                                 (rx, ry, bw * VW / cw, bh * VH / ch), 2)
            else:
                pygame.draw.circle(screen, (255, 255, 255) if not feeding else (120, 255, 120),
                                   (mx, my), max(2, int(brush * VW / cw)), 1)
        screen.set_clip(None)

        if show_panel:
            x0, y = VW + 14, 10
            hs = ("" if sim.hu is None else
                  ("   learning" if sim.hu.hc.learn else "   learning OFF"))
            lines = [f"EMERGENCE 2   t={w.t}" + hs, f"view [{VIEWS.index(view)+1}] {view}",
                     f"{rate:.0f} steps/s   x{spf}/frame" + ("   PAUSED" if paused else ""),
                     f"brush {brush}  {'FOOD' if feeding else 'wipe'}" + ("  REC" if rec else ""),
                     (f"zoom x{cfg.W / cw:.1f}  at ({cx0},{cy0})  level {len(cams) - 1}"
                      if len(cams) > 1 else "zoom x1")
                     + ("   BOX: click, +/- size" if selecting else ""), ""]
            for k, lab in LABELS:
                if k is None:
                    if sim.hu is not None:
                        lines.append(lab)
                    else:
                        lines.append(f"hunters arrive at t={sim.hunters_at} (J: now)")
                    continue
                v = last_m.get(k)
                if v is not None:
                    lines.append(f"{lab:<22}{v:.3g}" if isinstance(v, float) else f"{lab:<22}{v}")
            for ln in lines:
                screen.blit(font.render(ln, True, (220, 220, 225)), (x0, y)); y += line_h
            y += 6
            if two_col:                       # graphs and inspector go to the second column
                x0, y = VW + 14 + 380, 10
            for k, series in hist.items():
                if len(series) > 1:
                    lo, hi = min(series), max(series)
                    pts = [(x0 + i * 340 / 300, y + 24 - 24 * (v - lo) / (hi - lo + 1e-9))
                           for i, v in enumerate(series)]
                    pygame.draw.lines(screen, (120, 200, 255), False, pts, 1)
                    screen.blit(small.render(f"{k}  {lo:.3g}..{hi:.3g}", True, (150, 150, 160)),
                                (x0, y + 26))
                y += 40
            if inspect is not None and int(w.pool.count[inspect]) > 0:
                p = w.pool
                screen.blit(font.render(f"genome #{inspect}  cells={int(p.count[inspect])}",
                                        True, (255, 230, 150)), (x0, y)); y += 18
                screen.blit(small.render(f"clade {int(p.clade[inspect])}  depth {int(p.depth[inspect])}"
                                         f"  len {int(p.length[inspect])}  born t={int(p.born[inspect])}",
                                         True, (200, 200, 200)), (x0, y)); y += 16
                toks = p.tokens[inspect, :int(p.length[inspect])].tolist()
                for i, tk in enumerate(toks):
                    c = [int(255 * v) for v in torch.tensor(
                        [0.5 + 0.5 * np.cos(6.28 * (tk / 32 + o)) for o in (0, .33, .66)]).tolist()]
                    pygame.draw.rect(screen, c, (x0 + (i % 24) * 14, y + (i // 24) * 14, 12, 12))
                y += 34
                ob = w.obias[inspect].tolist()
                names = ["eat", "share", "divide", "attack", "detach"]
                C, Q = cfg.n_state, cfg.n_signal
                vals = [ob[C + Q], ob[C + Q + 1], ob[C + Q + 2], ob[C + Q + 7], ob[C + Q + 8]]
                for nm, v in zip(names, vals):
                    pygame.draw.rect(screen, (80, 80, 90), (x0 + 60, y + 3, 200, 8))
                    pygame.draw.rect(screen, (255, 180, 90), (x0 + 160, y + 3, int(v / 1.5 * 100), 8))
                    screen.blit(small.render(nm, True, (200, 200, 200)), (x0, y)); y += 14
        pygame.display.flip()
        clock.tick(120)
    if rec is not None:
        rec.close()
    pygame.quit()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=int, default=0)
    ap.add_argument("--width", type=int, default=0)
    ap.add_argument("--height", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", action="append", help="override a level-1 law, e.g. --set regulation=0")
    ap.add_argument("--hset", action="append", help="override a hunter law, e.g. --hset metabolism=0.25")
    ap.add_argument("--hunters-at", type=int, default=-1,
                    help="world step at which hunters are released (default 1500; 0 = at once)")
    ap.add_argument("--load", type=str, default="")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--out", type=str, default="runs/default")
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--snap-every", type=int, default=10000)
    ap.add_argument("--ckpt-every", type=int, default=50000)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--video-every", type=int, default=20)
    ap.add_argument("--scale", type=float, default=0, help="pixels per cell (default: fit the screen)")
    args = ap.parse_args()
    sim = build(args)
    if args.headless:
        headless(sim, args)
    else:
        viewer(sim, args)


if __name__ == "__main__":
    main()
