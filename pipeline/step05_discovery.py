import json

import numpy as np

from common.config import CONTROL_LEAD, OUT, WINDOW
from common.ensemble import Runner
from common.pangu import session
from common.tracker import crossings


def main():
    runner = Runner(session())
    folder = OUT / 'discovery'
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(1, 101):
        for sign in (1, -1):
            member = f'd{i:03d}_{"p" if sign == 1 else "m"}'
            target = folder / f'{member}.json'
            if target.exists():
                continue
            track, scores = runner.run(runner.perturbation('discovery', f'd{i:03d}', sign), CONTROL_LEAD)
            j = {x['lead_hours']: x['response_J_kg'] for x in scores}
            result = {'member': member, 'direction': f'd{i:03d}', 'sign': sign, 'response_primary_J_kg': float(np.mean([j[h] for h in WINDOW])),
                      'scores': scores, 'track': track, 'events': crossings(track, runner.coast)}
            target.write_text(json.dumps(result), encoding='utf-8')
            print(member, result['response_primary_J_kg'], flush=True)


if __name__ == '__main__':
    main()
