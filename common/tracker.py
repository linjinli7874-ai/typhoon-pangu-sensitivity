import json
import struct
import zipfile
from datetime import timedelta

import numpy as np
from shapely import make_valid
from shapely.geometry import LineString, Point, Polygon, box, mapping, shape
from shapely.strtree import STRtree

from common.config import LAT, LON, when

R = 6371.0088


def distance(lon1, lat1, lon2, lat2):
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dp = p2 - p1
    dl = np.deg2rad((np.asarray(lon2) - lon1 + 180) % 360 - 180)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def extrema(a, maximum=False):
    out = np.ones(a.shape, dtype=bool)
    pad = np.pad(a, 1, mode='edge')
    for di in range(3):
        for dj in range(3):
            if (di, dj) == (1, 1):
                continue
            adj = pad[di:di + a.shape[0], dj:dj + a.shape[1]]
            out &= a >= adj if maximum else a <= adj
    out[[0, -1], :] = False
    out[:, [0, -1]] = False
    return out


def prepare_coast(archive, target):
    features = []
    cut = box(90, 0, 150, 50)
    with zipfile.ZipFile(archive) as z:
        name = next(n for n in z.namelist() if n.endswith('gshhs_h.b'))
        with z.open(name) as f:
            while header := f.read(44):
                pid, n, flag, west, east, south, north, area, full, container, ancestor = struct.unpack('>11i', header)
                raw = f.read(8 * n)
                if flag & 255 != 1 or north < 0 or south > 50e6:
                    continue
                xy = np.frombuffer(raw, dtype='>i4').astype(float).reshape(-1, 2) / 1e6
                xy[:, 0] = np.where(xy[:, 0] > 180, xy[:, 0] - 360, xy[:, 0])
                if xy[:, 0].max() < 90 or xy[:, 0].min() > 150:
                    continue
                p = make_valid(Polygon(xy)).intersection(cut)
                if p.is_empty:
                    continue

                def parts(g):
                    if g.geom_type == 'Polygon':
                        return [g]
                    if hasattr(g, 'geoms'):
                        return [q for c in g.geoms for q in parts(c)]
                    return []
                version = (flag >> 8) & 255
                scale = 10. ** (flag >> 26) if version >= 9 else 10.
                for part in parts(p):
                    features.append({'type': 'Feature', 'properties': {'id': pid, 'area_full_km2': full / scale}, 'geometry': mapping(part)})
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({'type': 'FeatureCollection', 'features': features}), encoding='utf-8')


class Coast:
    def __init__(self, geojson):
        data = json.loads(geojson.read_text(encoding='utf-8'))
        self.polys = [shape(f['geometry']) for f in data['features']]
        self.attrs = [f['properties'] for f in data['features']]
        self.tree = STRtree(self.polys)

    def land(self, lon, lat, min_area=0):
        p = Point(lon, lat)
        return any(self.attrs[int(i)]['area_full_km2'] >= min_area and self.polys[int(i)].contains(p) for i in self.tree.query(p))


def locate(f, prior, coast, radius=222.5, initial=False):
    p = f['msl']
    u = f['u850'].astype(float)
    v = f['v850'].astype(float)
    phi = np.deg2rad(LAT)[:, None]
    lam = np.deg2rad(LON)
    zeta = (np.gradient(v, lam, axis=1, edge_order=2) - np.gradient(u * np.cos(phi), phi[:, 0], axis=0, edge_order=2)) / (R * 1000 * np.cos(phi))
    vort_max = extrema(zeta, True) & (zeta > 5e-5)
    d = distance(prior['lon'], prior['lat'], LON[None, :], LAT[:, None])
    candidates = np.argwhere(extrema(p) & (d <= (445 if initial else radius)))
    good = []
    for i, j in candidates:
        lon, lat = float(LON[j]), float(LAT[i])
        around = distance(lon, lat, LON[None, :], LAT[:, None]) <= 278
        if not (around & vort_max).any():
            continue
        wind = float(np.hypot(f['10u'][around], f['10v'][around]).max())
        if coast.land(lon, lat) and wind <= 8:
            continue
        good.append({'lon': lon, 'lat': lat, 'pressure_hpa': float(p[i, j] / 100), 'displacement_km': float(d[i, j])})
    if not good:
        return {'quality': 'tracking_failed'}
    good.sort(key=lambda c: (c['pressure_hpa'], c['displacement_km'], c['lat'], c['lon']))
    return {**good[0], 'quality': 'ok'}


def subgrid(p, row):
    result = dict(row)
    if row.get('quality') != 'ok':
        return result
    j = int(np.argmin(abs(LON - row['lon'])))
    i = int(np.argmin(abs(LAT - row['lat'])))
    if min(i, j) < 1 or i >= len(LAT) - 1 or j >= len(LON) - 1:
        return result
    yy, xx = np.mgrid[-1:2, -1:2]
    design = np.column_stack([np.ones(9), xx.ravel(), yy.ravel(), .5 * xx.ravel() ** 2, xx.ravel() * yy.ravel(), .5 * yy.ravel() ** 2])
    z = p[i - 1:i + 2, j - 1:j + 2].astype(float).ravel() / 100
    b = np.linalg.lstsq(design, z, rcond=None)[0]
    H = np.array([[b[3], b[4]], [b[4], b[5]]])
    eig = np.linalg.eigvalsh(H)
    rms = float(np.sqrt(np.mean((design @ b - z) ** 2)))
    if eig[0] <= 1e-5 or eig[-1] / max(eig[0], 1e-15) > 1000:
        return result
    offset = -np.linalg.solve(H, b[1:3])
    if abs(offset).max() > .75 or rms > .25:
        return result
    result.update(lon=float(LON[j] + .25 * offset[0]), lat=float(LAT[i] - .25 * offset[1]))
    return result


class Follow:
    def __init__(self, prior, coast):
        self.prior = prior
        self.coast = coast
        self.failed = False
        self.track = []

    def add(self, h, valid, f):
        point = {'valid_utc': valid, 'lead_hours': h}
        if h == 0:
            point.update({k: v for k, v in self.prior.items() if k not in ('valid_utc', 'lead_hours')})
        elif self.failed:
            point['quality'] = 'tracking_failed'
        else:
            point.update(locate(f, self.prior, self.coast))
            if point['quality'] == 'ok':
                self.prior = point
            else:
                self.failed = True
        self.track.append(subgrid(f['msl'], point))


def linear(track, init, hours=12):
    t0 = when(init)
    hist = [p for p in track if t0 - timedelta(hours=hours) <= when(p['valid_utc']) <= t0]
    if len(hist) != hours // 3 + 1 or any(p['quality'] != 'ok' for p in hist):
        raise ValueError('Invalid history for linear extrapolation')
    lon0, lat0 = hist[-1]['lon'], hist[-1]['lat']
    t = np.array([(when(p['valid_utc']) - t0).total_seconds() / 3600 for p in hist])
    x = R * np.cos(np.deg2rad(lat0)) * np.deg2rad(np.array([p['lon'] for p in hist]) - lon0)
    y = R * np.deg2rad(np.array([p['lat'] for p in hist]) - lat0)
    design = np.column_stack([np.ones(len(t)), t])
    bx = np.linalg.lstsq(design, x, rcond=None)[0]
    by = np.linalg.lstsq(design, y, rcond=None)[0]
    return [{'valid_utc': (t0 + timedelta(hours=lead)).isoformat(), 'lead_hours': lead,
             'lon': float(lon0 + np.rad2deg(bx[1] * lead / (R * np.cos(np.deg2rad(lat0))))),
             'lat': float(lat0 + np.rad2deg(by[1] * lead / R)), 'quality': 'ok'} for lead in range(0, 73, 3)]


def _points(g):
    if g.is_empty:
        return []
    if g.geom_type == 'Point':
        return [g]
    if hasattr(g, 'geoms'):
        return [p for c in g.geoms for p in _points(c)]
    return []


def crossings(track, coast):
    events = []
    for seg, (a, b) in enumerate(zip(track[:-1], track[1:])):
        if a.get('quality') != 'ok' or b.get('quality') != 'ok':
            continue
        if not all(90 < p['lon'] < 150 and 0 < p['lat'] < 50 for p in (a, b)):
            continue
        line = LineString([(a['lon'], a['lat']), (b['lon'], b['lat'])])
        if line.length == 0:
            continue
        points = []
        for idx in coast.tree.query(line):
            idx = int(idx)
            inter = line.intersection(coast.polys[idx].boundary)
            if inter.geom_type in ('LineString', 'MultiLineString') and not inter.is_empty:
                continue
            for p in _points(inter):
                f = line.project(p, normalized=True)
                if 1e-10 < f < 1 - 1e-10:
                    points.append((f, idx, p))
        for f, p in ((0, Point(a['lon'], a['lat'])), (1, Point(b['lon'], b['lat']))):
            for idx in coast.tree.query(p):
                if coast.polys[int(idx)].boundary.distance(p) < 1e-10:
                    points.append((f, int(idx), p))
        points.sort(key=lambda v: v[0])
        fractions = sorted(set([0., 1.] + [v[0] for v in points]))
        for f, idx, p in points:
            if any(abs(f - q['fraction']) < 1e-9 and q['segment_start'] == a['valid_utc'] for q in events):
                continue
            left = max([q for q in fractions if q < f - 1e-10], default=f - .001)
            right = min([q for q in fractions if q > f + 1e-10], default=f + .001)
            eps = min(1e-5, (f - left) / 2, (right - f) / 2)
            before = line.interpolate(f - eps, normalized=True) if f > 0 else Point(a['lon'] - eps * (b['lon'] - a['lon']), a['lat'] - eps * (b['lat'] - a['lat']))
            after = line.interpolate(f + eps, normalized=True) if f < 1 else Point(b['lon'] + eps * (b['lon'] - a['lon']), b['lat'] + eps * (b['lat'] - a['lat']))
            unknown = False
            if f < 1e-10:
                previous = track[seg - 1] if seg > 0 else None
                if previous and previous.get('quality') == 'ok':
                    before = Point(a['lon'] + eps * (previous['lon'] - a['lon']), a['lat'] + eps * (previous['lat'] - a['lat']))
                else:
                    unknown = True
            if f > 1 - 1e-10:
                following = track[seg + 2] if seg + 2 < len(track) else None
                if following and following.get('quality') == 'ok':
                    after = Point(b['lon'] + eps * (following['lon'] - b['lon']), b['lat'] + eps * (following['lat'] - b['lat']))
                else:
                    unknown = True
            lb, la = coast.land(before.x, before.y), coast.land(after.x, after.y)
            direction = 'sea_to_land' if not lb and la else 'land_to_sea' if lb and not la else 'touch'
            if unknown:
                direction = 'unknown'
            t = when(a['valid_utc']) + (when(b['valid_utc']) - when(a['valid_utc'])) * f
            if any(abs((when(e['time_utc']) - t).total_seconds()) < .01 and distance(e['lon'], e['lat'], p.x, p.y) < .001 for e in events):
                continue
            events.append({'time_utc': t.isoformat(), 'lon': p.x, 'lat': p.y, 'direction': direction, 'segment_start': a['valid_utc'],
                           'fraction': float(f), 'mainland': coast.attrs[idx]['area_full_km2'] >= 1_000_000})
    events.sort(key=lambda e: e['time_utc'])
    return {'first_mainland': next((e for e in events if e['direction'] == 'sea_to_land' and e['mainland']), None), 'events': events}


def compare(events, reference):
    a, b = events['first_mainland'], reference['first_mainland']
    if a is None or b is None:
        return {}
    return {'landfall_distance_km': float(distance(a['lon'], a['lat'], b['lon'], b['lat'])),
            'landfall_time_bias_h': (when(a['time_utc']) - when(b['time_utc'])).total_seconds() / 3600}
