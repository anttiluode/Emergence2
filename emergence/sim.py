"""Both levels together: the cell world (level 1) and the hunters living off it (level 2)."""
import torch

from .config import Config
from .world import World
from .hunters import Hunters, HunterConfig
from .render import VIEWS as WORLD_VIEWS

VIEWS = WORLD_VIEWS + ["hunter fields"]


class Sim:
    def __init__(self, cfg, hc, hunters_at=1500):
        self.w = World(cfg)
        self.hc = hc
        self.hunters_at = hunters_at
        self.hu = None
        self.show_hunters = True

    @property
    def t(self):
        return self.w.t

    def release(self):
        if self.hu is None:
            self.hu = Hunters(self.w, self.hc)
            print(f"t={self.w.t}: {self.hc.n_start} hunters released")

    def step(self):
        self.w.step()
        if self.hu is None and self.w.t >= self.hunters_at:
            self.release()
        if self.hu is not None:
            self.hu.step()

    def metrics(self):
        m = self.w.metrics()
        if self.hu is not None:
            m.update(self.hu.metrics())
        return m

    # ------------------------------------------------------------- drawing
    @torch.no_grad()
    def image(self, view, region=None):
        """uint8 image tensor (H, W, 3) on the device, hunters drawn on top."""
        w = self.w
        if view == "hunter fields":
            from .render import _render
            life = _render(w, "life").float() / 255 * 0.35
            img = life.clone()
            if self.hu is not None:
                img[..., 0] += self.hu.trace.clamp(0, 3) / 3 * 0.9        # danger (red)
                img[..., 2] += self.hu.mark * 0.9                           # marks (blue)
                img[..., 1] += self.hu.mark * 0.5
            img = (img.clamp(0, 1) * 255).to(torch.uint8)
        else:
            from .render import _render
            img = _render(w, view)
        if self.hu is not None and self.show_hunters:
            hu = self.hu
            a = hu.alive
            if a.any():
                y, x, e = hu.y[a], hu.x[a], (hu.E[a] / hu.hc.e_max).clamp(0, 1)
                col = torch.stack([torch.ones_like(e), 0.35 + 0.65 * e, 0.2 * e], 1) * 255
                H, W = w.cfg.H, w.cfg.W
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        yy, xx = (y + dy) % H, (x + dx) % W
                        c = col if (dy == 0 and dx == 0) else col * 0.55
                        img[yy, xx] = c.to(torch.uint8)
        if region is not None:
            y0, y1, x0, x1 = region
            img = img[y0:y1, x0:x1]
        return img

    def render(self, view, region=None):
        return self.image(view, region).cpu().numpy()

    # ------------------------------------------------------------- persistence
    def state_dict(self):
        return {"world": self.w.state_dict(),
                "hunters": self.hu.state_dict() if self.hu is not None else None,
                "hunters_at": self.hunters_at, "hc": self.hc.__dict__.copy()}

    @staticmethod
    def load(path, device="auto", world_over=None, hunter_over=None):
        d = torch.load(path, map_location="cpu", weights_only=False)
        if "world" not in d:                       # a level-1 checkpoint (Emergence)
            d = {"world": d, "hunters": None, "hunters_at": d["t"], "hc": {}}
        cfg = Config.from_dict(d["world"]["config"])
        for k, v in (world_over or {}).items():
            setattr(cfg, k, v)
        cfg.device = device
        seed_cells, cfg.seed_cells = cfg.seed_cells, 0
        hc = HunterConfig.from_dict(d["hunters"]["hc"] if d["hunters"] else d.get("hc", {}))
        for k, v in (hunter_over or {}).items():
            setattr(hc, k, v)
        s = Sim(cfg, hc, d.get("hunters_at", 0))
        cfg.seed_cells = seed_cells
        s.w.load_state_dict(d["world"])
        if d["hunters"] is not None:
            s.hu = Hunters(s.w, hc)
            s.hu.load_state_dict(d["hunters"])
        return s
