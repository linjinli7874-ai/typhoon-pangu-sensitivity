import numpy as np

from common.config import A_BOX, B_BOX, CP, LAT, LEVELS, LON, TR
from common.era5 import regional

AREA = (np.sin(np.deg2rad(LAT + .125)) - np.sin(np.deg2rad(LAT - .125)))[:, None] * np.ones((1, len(LON)))


def box_mask(box, taper=0.):
    n, w, s, e = box
    inside = (LAT[:, None] >= s) & (LAT[:, None] <= n) & (LON[None, :] >= w) & (LON[None, :] <= e)
    if not taper:
        return inside.astype(float)
    d = np.minimum(np.minimum(LAT[:, None] - s, n - LAT[:, None]), np.minimum(LON[None, :] - w, e - LON[None, :]))
    return inside * np.sin(.5 * np.pi * np.clip(d / taper, 0, 1)) ** 2


IN_A = box_mask(A_BOX) > 0
IN_B = box_mask(B_BOX) > 0


def layer_thickness(sp):
    valid = LEVELS[:, None, None] * 100 <= sp[None]
    dp = np.zeros((13, *sp.shape))
    for k in range(12):
        half = (LEVELS[k] - LEVELS[k + 1]) * 100 / 2
        keep = valid[k] & valid[k + 1]
        dp[k] += half * keep
        dp[k + 1] += half * keep
    return dp


def weights(sp, box):
    area = np.sin(np.deg2rad(LAT + .125)) - np.sin(np.deg2rad(LAT - .125))
    return layer_thickness(sp) * area[None, :, None] * box_mask(box)[None]


def energy(delta, w):
    t, u, v = np.asarray(delta, dtype=float)
    return float(np.sum(w * (u * u + v * v + CP / TR * t * t)) / (2 * w.sum()))


def column_energy(d, dp):
    t, u, v = np.asarray(d, dtype=float)
    return np.sum(dp * (u * u + v * v + CP / TR * t * t) / 2, axis=0)


def surface_pressure(t):
    return regional('regional_single', t, ['sp'])[('sp', 0)]


def construct(init, source_a, source_b):
    sp0 = surface_pressure(init)
    src = regional('regional_upper', source_a, ['t', 'u', 'v'])
    sub = regional('regional_upper', source_b, ['t', 'u', 'v'])
    sps = np.minimum(surface_pressure(source_a), surface_pressure(source_b))
    raw = np.stack([np.stack([src[n, int(p)] - sub[n, int(p)] for p in LEVELS]) for n in ['t', 'u', 'v']])
    margin = np.minimum(sp0, sps)[None] - LEVELS[:, None, None] * 100
    vertical = np.sin(.5 * np.pi * np.clip(margin / 2500, 0, 1)) ** 2
    raw *= box_mask(B_BOX, 2)[None, None] * vertical[None]
    w = weights(sp0, B_BOX)
    d = (raw * np.sqrt(1. / energy(raw, w))).astype(np.float32)
    return d, energy(d, w)
