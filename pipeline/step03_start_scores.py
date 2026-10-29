import json
from datetime import timedelta

import numpy as np

from common.config import COAST, LANDFALL, OUT, STARTS, when
from common.tracker import Coast, compare, crossings, distance, linear


def main():
    coast = Coast(COAST)
    fine = json.loads((OUT / 'era5_track.json').read_text(encoding='utf-8'))['fine']
    fine_at = {r['valid_utc']: r for r in fine}
    results = {}
    for init, lead in sorted(STARTS.items()):
        t0 = when(init)
        era = [r for r in fine if t0 <= when(r['valid_utc']) <= t0 + timedelta(hours=72)]
        era_events = crossings(era, coast)
        track = [p for p in json.loads((OUT / 'starts' / f'{init}.json').read_text(encoding='utf-8'))['track'] if p['lead_hours'] <= lead]
        events = crossings(track, coast)
        ext = linear(fine, init)
        ext_events = crossings(ext, coast)
        errors = []
        for p, q in zip(track, ext):
            e = fine_at[p['valid_utc']]
            ok = p.get('quality') == 'ok'
            errors.append({'valid_utc': p['valid_utc'], 'lead_hours': p['lead_hours'],
                           'pangu_km': float(distance(p['lon'], p['lat'], e['lon'], e['lat'])) if ok else None,
                           'linear_km': float(distance(q['lon'], q['lat'], e['lon'], e['lat']))})
        land = when(era_events['first_mainland']['time_utc'])
        before = [x for x in errors if x['lead_hours'] > 0 and when(x['valid_utc']) < land and x['pangu_km'] is not None]
        results[init] = {'init': init, 'hours_before_landfall': int((LANDFALL - t0).total_seconds() // 3600),
                         'era5_track': era, 'era5_landfall': era_events, 'pangu_track': track, 'pangu_landfall': events,
                         'pangu_error': compare(events, era_events), 'linear_landfall': ext_events, 'linear_error': compare(ext_events, era_events),
                         'track_errors': errors,
                         'preland': {'pangu_mae_km': float(np.mean([x['pangu_km'] for x in before])), 'linear_mae_km': float(np.mean([x['linear_km'] for x in before]))}}
        print(init, results[init]['pangu_error'], results[init]['linear_error'], results[init]['preland'], flush=True)
    (OUT / 'start_results.json').write_text(json.dumps(results), encoding='utf-8')


if __name__ == '__main__':
    main()
