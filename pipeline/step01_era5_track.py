import csv
import json
from datetime import timedelta

from common.config import COAST, GSHHG, IBTRACS, OUT, SID, TRACK_END, TRACK_START, stamp, when
from common.era5 import regional
from common.tracker import Coast, locate, prepare_coast, subgrid


def observations():
    rows = []
    with IBTRACS.open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            if row['SID'].strip() != SID:
                continue
            try:
                rows.append({'valid_utc': when(row['ISO_TIME'].strip()).isoformat(), 'lon': float(row['USA_LON']), 'lat': float(row['USA_LAT']),
                             'flag': row['IFLAG'].strip()[0]})
            except ValueError:
                pass
    return sorted(rows, key=lambda r: r['valid_utc'])


def era5_fields(t):
    s = regional('regional_single', t, ['msl', '10u', '10v'])
    u = regional('regional_upper', t, ['u', 'v'])
    return {'msl': s['msl', 0], '10u': s['10u', 0], '10v': s['10v', 0], 'u850': u['u', 850], 'v850': u['v', 850]}


def main():
    if not COAST.exists():
        prepare_coast(GSHHG, COAST)
    coast = Coast(COAST)
    start, end = when(TRACK_START), when(TRACK_END)
    past = [p for p in observations() if when(p['valid_utc']) <= start and p['flag'] in ('O', 'V')]
    seed = past[-1]
    if (start - when(seed['valid_utc'])).total_seconds() > 6 * 3600:
        raise ValueError('Seed too old')
    grid, fine, prior, failed = [], [], seed, False
    t = start
    while t <= end:
        f = era5_fields(stamp(t))
        row = {'valid_utc': t.isoformat()}
        if failed:
            row['quality'] = 'tracking_failed'
        else:
            row.update(locate(f, prior, coast, initial=not grid))
            if row['quality'] == 'ok':
                prior = row
            else:
                failed = True
        grid.append(row)
        fine.append(subgrid(f['msl'], row))
        t += timedelta(hours=3)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'era5_track.json').write_text(json.dumps({'grid': grid, 'fine': fine}, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
