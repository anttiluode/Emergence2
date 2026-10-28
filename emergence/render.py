"""Turn world tensors into RGB images (uint8, H x W x 3), computed on the device."""
import torch

VIEWS = ["life", "bodies", "clades", "food", "signals", "state", "energy", "age"]


def _hue(x):
    """Hash integers to bright colours."""
    x = ((x % 2147483647) * 2654435761) % 4294967296
    h = (x.float() / 4294967296.0)[..., None]
    k = torch.tensor([0.0, 2.0 / 3.0, 1.0 / 3.0], device=x.device)
    return (0.5 + 0.5 * torch.cos(6.2832 * (h + k))).clamp(0, 1)


@torch.no_grad()
def render(world, view="life", region=None):
    """region = (y0, y1, x0, x1) crops on the device before copying to the CPU."""
    img = _render(world, view)
    if region is not None:
        y0, y1, x0, x1 = region
        img = img[y0:y1, x0:x1]
    return img.cpu().numpy()


def _render(world, view):
    a = world.alive
    af = a.float()[..., None]
    food = world.food.clamp(0, 1)[..., None]
    ground = food * torch.tensor([0.10, 0.16, 0.08], device=food.device)
    if view == "life":
        g = world.gid.clamp(min=0)
        col = world.color[g]
        bright = (0.55 + 0.45 * (world.energy / world.cfg.e_max).clamp(0, 1))[..., None]
        img = af * col * bright + (1 - af) * ground
    elif view == "bodies":
        b = world.body.clamp(min=0)
        img = af * _hue(b) + (1 - af) * ground
        # outline organism borders: a cell whose neighbour belongs to another body
        edge = torch.zeros_like(world.alive)
        for sh in ((0, 1), (1, 0)):
            edge |= world.alive & (torch.roll(world.body, sh, (0, 1)) != world.body)
        img = img * (1 - 0.55 * edge.float()[..., None])
    elif view == "clades":
        cl = world.pool.clade[world.gid.clamp(min=0)].clamp(min=0)
        img = af * _hue(cl) + (1 - af) * ground
    elif view == "food":
        f = world.food.clamp(0, 1)
        img = torch.stack([0.2 * f + 0.6 * world.sun, 0.9 * f, 0.3 * f], -1)
        img = img * (1 - 0.5 * af)
    elif view == "signals":
        s = world.sig[:3]
        s = s / (s.amax(dim=(1, 2), keepdim=True) + 1e-6)
        img = s.permute(1, 2, 0)
    elif view == "state":
        img = af * (0.5 + 0.5 * world.S[:3].permute(1, 2, 0)) + (1 - af) * ground
    elif view == "energy":
        e = (world.energy / world.cfg.e_max).clamp(0, 1)
        img = torch.stack([e, e ** 2, 0.2 * af[..., 0]], -1)
    elif view == "age":
        ag = (world.age / (world.cfg.age_onset + world.cfg.age_span)).clamp(0, 1)
        img = af * torch.stack([ag, 0.4 * (1 - ag), 1 - ag], -1) + (1 - af) * ground
    else:
        raise ValueError(view)
    return (img.clamp(0, 1) * 255).to(torch.uint8)
