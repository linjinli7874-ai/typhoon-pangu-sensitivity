import json
import random
from datetime import timedelta

import numpy as np

from common.config import DISCOVERY_LAGS, HOLDOUT_LAGS, INIT, OUT, stamp, when
from common.physics import construct


def pairs(times, count, seed):
    candidates = [(a, b) for a in times for b in times if b - a in [72, 96, 120]]
    return sorted(random.Random(seed).sample(candidates, count))


def main():
    t0 = when(INIT)
    for split, lags, count, seed, prefix in (('discovery', DISCOVERY_LAGS, 100, 20261002, 'd'), ('holdout', HOLDOUT_LAGS, 25, 20261003, 'h')):
        folder = OUT / 'library' / split
        folder.mkdir(parents=True, exist_ok=True)
        for i, (a, b) in enumerate(pairs(lags, count, seed), 1):
            name = f'{prefix}{i:03d}'
            if (folder / f'{name}.json').exists():
                continue
            source_a, source_b = stamp(t0 + timedelta(hours=a)), stamp(t0 + timedelta(hours=b))
            d, e = construct(INIT, source_a, source_b)
            np.save(folder / f'{name}.npy', d)
            (folder / f'{name}.json').write_text(json.dumps({'direction': name, 'source_a': source_a, 'source_b': source_b, 'energy_J_kg': e}), encoding='utf-8')
            print(split, name, source_a, source_b, flush=True)


if __name__ == '__main__':
    main()
