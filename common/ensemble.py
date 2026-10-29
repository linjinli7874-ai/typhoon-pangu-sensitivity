import json

import numpy as np

from common.config import A_BOX, COAST, CROP, ENERGY, INIT, OUT, when
from common.era5 import initial_state
from common.pangu import rollout
from common.physics import energy, surface_pressure, weights
from common.tracker import Coast, Follow


class Runner:
    def __init__(self, s):
        self.s = s
        self.coast = Coast(COAST)
        self.prior = {r['valid_utc']: r for r in json.loads((OUT / 'era5_track.json').read_text(encoding='utf-8'))['grid']}[when(INIT).isoformat()]
        self.control = np.load(OUT / 'control_tuv.npy', mmap_mode='r')
        self.w = weights(surface_pressure(INIT), A_BOX)

    def perturbation(self, split, direction, sign, mask=None, eta=0.0):
        meta = json.loads((OUT / 'library' / split / f'{direction}.json').read_text(encoding='utf-8'))
        d = np.load(OUT / 'library' / split / f'{direction}.npy')
        d *= sign * np.sqrt(ENERGY / meta['energy_J_kg'])
        if mask is not None:
            d *= 1 - eta * mask[None, None]
        return d

    def run(self, d, lead, response=True):
        upper, surface = initial_state(INIT)
        upper[2:5, :, CROP[0], CROP[1]] += d
        follow = Follow(self.prior, self.coast)
        scores = []
        for h, valid, regional, fields in rollout(self.s, upper, surface, INIT, lead):
            follow.add(h, valid, fields)
            if response:
                scores.append({'lead_hours': h, 'response_J_kg': energy(regional[2:5] - self.control[h // 3], self.w)})
        return follow.track, scores
