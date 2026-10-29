import json

import numpy as np
from scipy import ndimage

from common.config import INIT, LAT, LON, OUT
from common.physics import AREA, IN_A, IN_B, column_energy, layer_thickness, surface_pressure

N_RC = 1600
OFFSETS = {'R0': (0, 0), 'R1': (0, 1), 'R2': (1, 1), 'R3': (1, 0), 'R4': (1, -1), 'R5': (0, -1), 'R6': (-1, -1), 'R7': (-1, 0), 'R8': (-1, 1)}


def connected_top(rho, n):
    peak = np.unravel_index(rho.argmax(), rho.shape)
    values = np.sort(rho[IN_B])[::-1]
    lo, hi = n - 1, len(values) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        lab, _ = ndimage.label(rho >= values[mid])
        if (lab == lab[peak]).sum() >= n:
            hi = mid
        else:
            lo = mid + 1
    lab, _ = ndimage.label(rho >= values[lo])
    piece = lab == lab[peak]
    mask = piece & (rho >= np.sort(rho[piece])[-n])
    if mask.sum() != n:
        raise ValueError('Ties at the threshold')
    return mask


def boxes(lon_c, lat_c, n=40):
    lon0 = round((lon_c - n * .125) / .25) * .25
    lat0 = round((lat_c - n * .125) / .25) * .25
    out = {}
    for name, (dx, dy) in OFFSETS.items():
        west, south = lon0 + dx * n * .25, lat0 + dy * n * .25
        m = ((LON[None, :] >= west - 1e-9) & (LON[None, :] < west + n * .25 - 1e-9) & (LAT[:, None] >= south - 1e-9) & (LAT[:, None] < south + n * .25 - 1e-9))
        out[name] = (m, {'west': west, 'east': west + (n - 1) * .25, 'south': south, 'north': south + (n - 1) * .25})
    return out


def main():
    members = [json.loads(p.read_text(encoding='utf-8')) for p in sorted((OUT / 'discovery').glob('*.json'))]
    if len(members) != 200:
        raise ValueError('Need 200 discovery members')
    members.sort(key=lambda r: (-r['response_primary_J_kg'], r['member']))
    dp = layer_thickness(surface_pressure(INIT))
    cache = {}
    for r in members[:20]:
        if r['direction'] not in cache:
            q = column_energy(np.load(OUT / 'library' / 'discovery' / f"{r['direction']}.npy"), dp) * IN_B
            cache[r['direction']] = q / np.sum(q * AREA)
    rho = np.mean([cache[r['direction']] for r in members[:20]], axis=0)
    rc = connected_top(rho, N_RC)
    w = rho * rc * AREA
    centre = (float(np.sum(LON[None, :] * w) / w.sum()), float(np.sum(LAT[:, None] * w) / w.sum()))
    jj, ii = np.where(rc)
    masks = {'RC': rc}
    info = {'RC': {'west': float(LON[ii].min()), 'east': float(LON[ii].max()), 'south': float(LAT[jj].min()), 'north': float(LAT[jj].max())}}
    for name, (m, geometry) in boxes(*centre).items():
        masks[name] = m
        info[name] = geometry
    for name, m in masks.items():
        info[name].update(grid_points=int(m.sum()), area_relative_to_RC=float(np.sum(m * AREA) / np.sum(rc * AREA)),
                          top20_energy_share=float(np.sum(rho * m * AREA)), fraction_inside_A=float((m & IN_A).sum() / m.sum()))
    folder = OUT / 'regions'
    folder.mkdir(parents=True, exist_ok=True)
    for name, m in masks.items():
        np.save(folder / f'{name}.npy', m.astype(np.float32))
    np.save(folder / 'top20_density.npy', rho)
    (folder / 'regions.json').write_text(json.dumps({'top20': [r['member'] for r in members[:20]], 'RC_energy_centre': centre, 'regions': info}, indent=1), encoding='utf-8')
    print(json.dumps(info, indent=1))


if __name__ == '__main__':
    main()
