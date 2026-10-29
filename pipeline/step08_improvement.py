import csv
import json

import numpy as np

from common.config import ETAS, HOLDOUT_LEAD, INIT, OUT, REGIONS

R_EARTH = 6371.0088
BOOT, SEED = 10000, 20261009
HOURS = list(range(0, HOLDOUT_LEAD + 1, 3))


def distance(lon1, lat1, lon2, lat2):
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    h = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.deg2rad(lon2 - lon1) / 2) ** 2
    return float(2 * R_EARTH * np.arcsin(np.sqrt(h)))


def series(track, control):
    t = {x['lead_hours']: x for x in track}
    c = {x['lead_hours']: x for x in control}
    ok = [t[h].get('quality') == 'ok' and c[h].get('quality') == 'ok' for h in HOURS]
    d = np.array([distance(t[h]['lon'], t[h]['lat'], c[h]['lon'], c[h]['lat']) if k else np.nan for h, k in zip(HOURS, ok)])
    p = np.array([abs(t[h]['pressure_hpa'] - c[h]['pressure_hpa']) if k else np.nan for h, k in zip(HOURS, ok)])
    return {'pressure': p, 'track': np.cumsum(d) / np.arange(1, len(HOURS) + 1)}


def main():
    control = [x for x in json.loads((OUT / 'starts' / f'{INIT}.json').read_text(encoding='utf-8'))['track'] if x['lead_hours'] <= HOLDOUT_LEAD]
    curves = {}
    for folder in sorted((OUT / 'holdout').iterdir()):
        curves[folder.name] = {}
        for path in folder.glob('*.json'):
            r = json.loads(path.read_text(encoding='utf-8'))
            curves[folder.name][(r['region'], float(r['eta']))] = series(r['track'], control)
    if len(curves) != 50 or any(len(v) != 41 for v in curves.values()):
        raise ValueError('Expected 50 members x 41 runs')
    rows = []
    for eta in ETAS:
        for region in REGIONS:
            for m in ('track', 'pressure'):
                keys = [('RAW', 0.0), ('RC', eta), (region, eta)]
                members = [x for x in sorted(curves) if all(np.isfinite(curves[x][k][m][-1]) for k in keys)]
                raw, f3 = (np.array([curves[x][k][m][-1] for x in members]) for k in (keys[0], keys[2]))
                pick = np.random.RandomState(SEED).randint(0, len(members), (BOOT, len(members)))
                low, high = np.quantile((1 - f3[pick].mean(axis=1) / raw[pick].mean(axis=1)) * 100, [.025, .975])
                rows.append({'eta': eta, 'region': region, 'metric': m, 'n': len(members), 'F2_mean': float(raw.mean()), 'F3_mean': float(f3.mean()),
                             'improvement_percent': float((1 - f3.mean() / raw.mean()) * 100), 'improvement_ci95_low': float(low), 'improvement_ci95_high': float(high)})
                print(eta, region, m, round(rows[-1]['improvement_percent'], 1), flush=True)
    with (OUT / 'improvement.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == '__main__':
    main()
