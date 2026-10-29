import json

import numpy as np

from common.config import ETAS, HOLDOUT_LEAD, OUT, REGIONS
from common.ensemble import Runner
from common.pangu import session


def main():
    runner = Runner(session())
    masks = {r: np.load(OUT / 'regions' / f'{r}.npy') for r in REGIONS}
    runs = [('RAW', 0.0)] + [(r, eta) for eta in ETAS for r in REGIONS]
    for i in range(1, 26):
        for sign in (1, -1):
            member = f'h{i:03d}_{"p" if sign == 1 else "m"}'
            folder = OUT / 'holdout' / member
            folder.mkdir(parents=True, exist_ok=True)
            for region, eta in runs:
                target = folder / f'{region}_{eta}.json'
                if target.exists():
                    continue
                d = runner.perturbation('holdout', f'h{i:03d}', sign, None if region == 'RAW' else masks[region], eta)
                track, _ = runner.run(d, HOLDOUT_LEAD, response=False)
                target.write_text(json.dumps({'member': member, 'region': region, 'eta': eta, 'track': track}), encoding='utf-8')
                print(member, region, eta, flush=True)


if __name__ == '__main__':
    main()
