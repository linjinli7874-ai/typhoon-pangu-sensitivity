from functools import lru_cache

import eccodes as ec
import numpy as np

from common.config import ERA5, INPUTS, LEVELS, when

UPPER = ('z', 'q', 't', 'u', 'v')
SURFACE = ('msl', '10u', '10v', '2t')


@lru_cache(maxsize=4)
def _day(kind, day, names):
    out = {}
    with (ERA5 / kind / f'{day}.grib').open('rb') as f:
        while (g := ec.codes_grib_new_from_file(f)) is not None:
            try:
                name = str(ec.codes_get(g, 'shortName'))
                if name not in names:
                    continue
                t = f"{int(ec.codes_get(g, 'dataDate'))}T{int(ec.codes_get(g, 'dataTime')) // 100:02d}Z"
                a = ec.codes_get_values(g).reshape(257, 417)
                if not np.isfinite(a).all():
                    raise ValueError(f'Nonfinite {name} {t}')
                out.setdefault(t, {})[(name, int(ec.codes_get(g, 'level')))] = a.astype(np.float32)
            finally:
                ec.codes_release(g)
    return out


def regional(kind, t, names):
    days = _day(kind, t[:8], tuple(names))
    if t not in days:
        raise ValueError(f'Missing {kind} {t}')
    return days[t]


def _global(kind, t, names, levels):
    moment = when(t)
    shape = (len(names), len(levels), 721, 1440) if levels else (len(names), 721, 1440)
    out = np.empty(shape, dtype=np.float32)
    seen = set()
    with (ERA5 / kind / f'{t[:8]}.grib').open('rb') as f:
        while (g := ec.codes_grib_new_from_file(f)) is not None:
            try:
                if (ec.codes_get(g, 'dataDate'), ec.codes_get(g, 'dataTime')) != (int(moment.strftime('%Y%m%d')), moment.hour * 100):
                    continue
                name = ec.codes_get(g, 'shortName')
                level = int(ec.codes_get(g, 'level')) if levels else 0
                if name not in names or (levels and level not in levels):
                    continue
                values = ec.codes_get_values(g).reshape(721, 1440)
                if levels:
                    out[names.index(name), levels.index(level)] = values
                else:
                    out[names.index(name)] = values
                seen.add((name, level))
            finally:
                ec.codes_release(g)
    if len(seen) != len(names) * (len(levels) if levels else 1):
        raise ValueError(f'Incomplete {kind} {t}')
    return out


def initial_state(t):
    folder = INPUTS / t
    if not (folder / 'input_surface.npy').exists():
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / 'input_upper.npy', _global('global_upper', t, UPPER, tuple(int(p) for p in LEVELS)))
        np.save(folder / 'input_surface.npy', _global('global_surface', t, SURFACE, None))
    return np.load(folder / 'input_upper.npy'), np.load(folder / 'input_surface.npy')
