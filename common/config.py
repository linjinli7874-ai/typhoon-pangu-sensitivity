import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
ERA5 = DATA / 'era5'
IBTRACS = DATA / 'ibtracs.last3years.list.v04r01.csv'
GSHHG = DATA / 'gshhg-bin-2.3.7.zip'
MODEL = ROOT / 'models' / 'pangu_weather_3.onnx'
OUT = ROOT / 'output'
IMG = ROOT / 'img'
COAST = OUT / 'coast_h.geojson'
INPUTS = OUT / 'inputs'
THREADS = int(os.environ.get('PANGU_THREADS', os.cpu_count() or 1))

SID = '2026182N09163'
INIT = '20260710T15Z'
LANDFALL = datetime(2026, 7, 11, 15, tzinfo=timezone.utc)
TRACK_START = '20260709T15Z'
TRACK_END = '20260714T03Z'
STARTS = {(LANDFALL - timedelta(hours=h)).strftime('%Y%m%dT%HZ'): h + 12 for h in range(6, 37, 3)}
CONTROL_LEAD = 72
HOLDOUT_LEAD = 36

LEVELS = np.array([1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50], dtype=float)
LAT = 62 - .25 * np.arange(257)
LON = 78 + .25 * np.arange(417)
CROP = (slice(112, 369), slice(312, 729))
AREA_NWSE = [62, 78, -2, 182]
CP, TR, RD = 1004., 270., 287.
B_BOX = [60, 80, 0, 180]
A_BOX = [45, 105, 20, 130]
ENERGY = 0.25
WINDOW = [18, 21, 24, 27, 30]
ETAS = [1.0, 0.7, 0.5, 0.3]
REGIONS = ['RC', 'R0', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8']
DISCOVERY_LAGS = list(range(-150, 0, 3)) + list(range(3, 151, 3))
HOLDOUT_LAGS = list(range(-300, -176, 3)) + list(range(177, 301, 3))


def when(s):
    if len(s) == 12 and s.endswith('Z'):
        return datetime.strptime(s, '%Y%m%dT%HZ').replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def stamp(t):
    return t.strftime('%Y%m%dT%HZ')
