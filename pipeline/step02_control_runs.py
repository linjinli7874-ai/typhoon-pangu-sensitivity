import json

import numpy as np

from common.config import CONTROL_LEAD, COAST, INIT, OUT, STARTS, when
from common.era5 import initial_state
from common.pangu import rollout, session
from common.tracker import Coast, Follow


def main():
    coast = Coast(COAST)
    grid = {r['valid_utc']: r for r in json.loads((OUT / 'era5_track.json').read_text(encoding='utf-8'))['grid']}
    folder = OUT / 'starts'
    folder.mkdir(parents=True, exist_ok=True)
    s = session()
    for init, lead in sorted(STARTS.items()):
        target = folder / f'{init}.json'
        if target.exists():
            continue
        if init == INIT:
            lead = CONTROL_LEAD
            tuv = np.lib.format.open_memmap(OUT / 'control_tuv.part.npy', mode='w+', dtype=np.float32, shape=(lead // 3 + 1, 3, 13, 257, 417))
        follow = Follow(grid[when(init).isoformat()], coast)
        upper, surface = initial_state(init)
        for h, valid, regional, fields in rollout(s, upper, surface, init, lead):
            follow.add(h, valid, fields)
            if init == INIT:
                tuv[h // 3] = regional[2:5]
            print(init, h, flush=True)
        if init == INIT:
            tuv.flush()
            del tuv
            (OUT / 'control_tuv.part.npy').replace(OUT / 'control_tuv.npy')
        target.write_text(json.dumps({'init': init, 'lead_hours': lead, 'track': follow.track}, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
