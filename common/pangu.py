from datetime import timedelta

import numpy as np
import onnxruntime as ort

from common.config import CROP, MODEL, THREADS, when


def session(threads=THREADS):
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    options.enable_mem_reuse = False
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    s = ort.InferenceSession(str(MODEL), sess_options=options, providers=['CPUExecutionProvider'])
    s.disable_fallback()
    return s


def rollout(s, upper, surface, init, lead):
    t0 = when(init)
    upper, surface = np.ascontiguousarray(upper), np.ascontiguousarray(surface)
    for h in range(0, lead + 1, 3):
        if h:
            upper, surface = s.run(None, {'input': upper, 'input_surface': surface})
            if not (np.isfinite(upper).all() and np.isfinite(surface).all()):
                raise ValueError(f'Nonfinite forecast at +{h} h')
        regional = upper[:, :, CROP[0], CROP[1]]
        fields = {'msl': surface[0][CROP], '10u': surface[1][CROP], '10v': surface[2][CROP], 'u850': regional[3, 2], 'v850': regional[4, 2]}
        yield h, (t0 + timedelta(hours=h)).isoformat(), regional, fields
