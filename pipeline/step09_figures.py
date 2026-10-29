import csv
import json
from datetime import datetime

import matplotlib

matplotlib.use('Agg')
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MultipleLocator

from common import style as ps
from common.config import A_BOX, ETAS, IMG, INIT, LAT, LON, OUT, REGIONS, when
from common.era5 import regional

WINDOW = (18, 30)
C_ERA5, C_PANGU = 'k', ps.BLUE
HALO = [pe.withStroke(linewidth=1.6, foreground='white')]
HALO_THIN = [pe.withStroke(linewidth=1.4, foreground='white')]
BOX = 1.15
NAMES = {'R0': '中心', 'R1': '北', 'R2': '东北', 'R3': '东', 'R4': '东南', 'R5': '南', 'R6': '西南', 'R7': '西', 'R8': '西北'}
LIGHT_RED, LIGHT_GREY, EDGE = '#d6604d', '#bdbdbd', '#404040'
METRIC = [('track', '0～36 h平均路径误差'), ('pressure', '36 h中心气压误差')]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(fig, name):
    IMG.mkdir(parents=True, exist_ok=True)
    fig.savefig(IMG / f'{name}.png')
    plt.close(fig)
    print('WROTE', name, flush=True)


def lonlat(track, last=72):
    ok = [p for p in track if p.get('quality', 'ok') == 'ok' and p.get('lead_hours', 0) <= last]
    return [p['lon'] for p in ok], [p['lat'] for p in ok]


def hours_of(s):
    return datetime.fromisoformat(s).timestamp() / 3600


def runs():
    return sorted(read(OUT / 'start_results.json').values(), key=lambda r: -r['hours_before_landfall'])


def start_colours(hours):
    cmap = plt.get_cmap('plasma')
    ordered = sorted(hours, reverse=True)
    return {h: cmap(0.02 + 0.86 * k / (len(ordered) - 1)) for k, h in enumerate(ordered)}


def lead_bar(fig, colour, hours, cax, loc='center'):
    bar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.BoundaryNorm(np.arange(4.5, 38, 3), len(hours)),
                                                    cmap=matplotlib.colors.ListedColormap([colour[h] for h in sorted(hours)])),
                       cax=cax, orientation='horizontal', ticks=[6, 12, 18, 24, 30, 36])
    bar.ax.tick_params(labelsize=ps.TINY, length=2, direction='in', top=False)
    bar.outline.set_linewidth(0.5)
    bar.ax.set_title('起报时刻距登陆的时间/h', fontsize=ps.TINY, pad=2.5, loc=loc)


def pair_legend(ax, **kwargs):
    ax.legend(handles=[Line2D([], [], color='k', lw=1.0, marker='o', ms=4, mfc='0.35', mec='k', mew=0.4, label='盘古预报'),
                       Line2D([], [], color='0.45', ls=(0, (4, 2)), lw=1.0, marker='^', ms=4.5, mfc='white', mec='0.35', mew=1.0, label='线性外推')],
              loc='upper right', ncols=2, **kwargs)


def fig2_2():
    rs = {r['init']: r for r in runs()}
    first = max(rs.values(), key=lambda r: r['hours_before_landfall'])
    track = [t for t in first['era5_track'] if t['valid_utc'] <= '2026-07-12T03']
    land = first['era5_landfall']['first_mainland']
    starts = [r['pangu_track'][0] for r in rs.values()]
    main = rs[INIT]['pangu_track'][0]
    fields = regional('regional_upper', INIT, ['z', 'u', 'v'])
    z500 = fields['z', 500] / 9.80665 / 10
    us = np.mean([fields['u', k] for k in (850, 700, 600, 500)], axis=0)
    vs = np.mean([fields['v', k] for k in (850, 700, 600, 500)], axis=0)
    views = [((114.5, 130.5), (19.5, 32.5)), ((100, 150), (8, 48))]
    ratio = [(x[1] - x[0]) * np.cos(np.deg2rad(np.mean(y))) / (y[1] - y[0]) for x, y in views]
    fig = ps.figure(ps.FULL, 8.2)
    a, b = fig.subplots(1, 2, width_ratios=ratio)
    ps.geo(a, ps.coast('h', 113, 132, 18, 34, thin=4), *views[0], 4, 3)
    a.plot([t['lon'] for t in track], [t['lat'] for t in track], color='k', lw=1.2, zorder=5)
    a.plot([t['lon'] for t in track], [t['lat'] for t in track], 'o', ms=2.6, mfc='white', mec='k', mew=0.6, zorder=6)
    a.plot([s['lon'] for s in starts], [s['lat'] for s in starts], 'o', ms=3.2, color=ps.BLUE, mec='k', mew=0.3, zorder=7)
    a.plot(main['lon'], main['lat'], 'o', ms=5.2, mfc='none', mec=ps.RED, mew=1.0, zorder=8)
    a.scatter(land['lon'], land['lat'], marker='*', s=70, facecolor='k', edgecolor='white', linewidth=0.5, zorder=9)
    for t in track:
        if t['valid_utc'][11:13] in ('00', '12'):
            a.annotate(f"{int(t['valid_utc'][8:10])}日{t['valid_utc'][11:13]}时", xy=(t['lon'], t['lat']), xytext=(5, -1), textcoords='offset points',
                       ha='left', va='center', fontsize=ps.TINY, path_effects=HALO, zorder=10)
    a.legend(handles=[Line2D([], [], color='k', lw=1.2, marker='o', ms=2.6, mfc='white', mec='k', mew=0.6, label='ERA5参照路径'),
                      Line2D([], [], color=ps.BLUE, ls='', marker='o', ms=3.2, mec='k', mew=0.3, label='控制预报起报位置'),
                      Line2D([], [], color=ps.RED, ls='', marker='o', ms=5.2, mfc='none', mew=1.0, label='扰动试验起报位置'),
                      Line2D([], [], color='k', ls='', marker='*', ms=7, mec='white', label='参照登陆点')],
             loc='upper right', handlelength=1.6, borderaxespad=0.5)
    ps.geo(b, ps.coast('i', 100, 150, 8, 48, thin=4), *views[1], 10, 10, lw=0.35)
    cs = b.contour(LON, LAT, z500, levels=np.arange(540, 596, 4), colors=ps.BLUE, linewidths=0.6, zorder=5)
    b.contour(LON, LAT, z500, levels=[588], colors=ps.BLUE, linewidths=1.5, zorder=5)
    for label in b.clabel(cs, [564, 572, 580, 584, 588, 592], fmt='%d', fontsize=ps.TINY, inline_spacing=2):
        label.set_zorder(9)
    u, v = us[::12, ::12], vs[::12, ::12]
    weak = np.hypot(u, v) < 3.0
    q = b.quiver(LON[::12], LAT[::12], np.where(weak, np.nan, u), np.where(weak, np.nan, v), scale=75, scale_units='inches',
                 width=0.0032, headwidth=3.6, headlength=4.2, headaxislength=3.8, color='0.25', zorder=6)
    b.quiverkey(q, 0.86, 1.045, 10, '10 m/s', labelpos='E', coordinates='axes', fontproperties={'size': ps.SMALL}, labelsep=0.05)
    b.scatter(main['lon'], main['lat'], marker='*', s=60, facecolor='white', edgecolor='k', linewidth=0.7, zorder=10)
    ps.panel(a, 'a')
    ps.panel(b, 'b')
    fig.legend(handles=[Line2D([], [], color=ps.BLUE, lw=0.6, label='500 hPa位势高度/dagpm'),
                        Line2D([], [], marker=r'$\rightarrow$', color='0.25', ls='', markersize=9, label='850～500 hPa平均风'),
                        Line2D([], [], marker='*', ls='', markersize=7.5, markerfacecolor='white', markeredgecolor='k', markeredgewidth=0.7, label='台风中心')],
               loc='outside lower right', ncols=3, columnspacing=1.4, handletextpad=0.5)
    save(fig, 'fig2_2')


def map_extent(lat0, lat1, lon_centre):
    half = BOX * (lat1 - lat0) / np.cos(np.deg2rad((lat0 + lat1) / 2)) / 2
    return (lon_centre - half, lon_centre + half), (lat0, lat1)


def fig4_1():
    rs = runs()
    hours = [r['hours_before_landfall'] for r in rs]
    cmap = plt.get_cmap('plasma')
    colour = start_colours(hours)
    views = [map_extent(20.2, 31.3, 122.6), map_extent(27.62, 29.72, 121.45)]
    fig = ps.figure(ps.FULL, 13.9)
    axs = fig.subplots(2, 2)
    for ax in axs.ravel():
        ax.set_box_aspect(1 / BOX)
    era = [t for t in max(rs, key=lambda r: r['hours_before_landfall'])['era5_track'] if t['valid_utc'] <= '2026-07-12T03']
    land = rs[0]['era5_landfall']['first_mainland']
    for ax, (xlim, ylim), step, segments, zoom in ((axs[0, 0], views[0], 3, ps.coast('h', 113, 132, 19, 33, thin=4), False),
                                                  (axs[0, 1], views[1], 0.5, ps.coast('h', 119.5, 123.5, 27, 30.5), True)):
        ps.geo(ax, segments, xlim, ylim, step, 2 if not zoom else 0.5, lw=0.5 if zoom else 0.4)
        ax.set_aspect('auto')
        for r in rs:
            c = colour[r['hours_before_landfall']]
            x, y = lonlat(r['pangu_track'])
            ax.plot(x, y, color=c, lw=0.6 if zoom else 0.9, alpha=0.75 if zoom else 1, zorder=5)
            if not zoom:
                ax.plot(x[0], y[0], 'o', ms=2.6, color=c, mec='k', mew=0.3, zorder=7)
        ax.plot([t['lon'] for t in era], [t['lat'] for t in era], color='k', lw=1.4, marker='o' if not zoom else None, ms=2.2, zorder=6)
        ax.scatter(land['lon'], land['lat'], marker='*', s=120 if zoom else 60, facecolor='k', edgecolor='white', linewidth=0.5, zorder=11)
    for r in rs:
        c = colour[r['hours_before_landfall']]
        p, q = r['pangu_landfall']['first_mainland'], r['linear_landfall']['first_mainland']
        axs[0, 1].scatter(q['lon'], q['lat'], marker='^', s=24, facecolor='white', edgecolor=c, linewidth=1.0, zorder=9)
        axs[0, 1].scatter(p['lon'], p['lat'], marker='o', s=20, facecolor=c, edgecolor='k', linewidth=0.4, zorder=10)
    (x0, x1), (y0, y1) = views[1]
    axs[0, 0].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec='0.3', lw=0.6, ls=(0, (3, 2)), zorder=8))
    axs[0, 0].legend(handles=[Line2D([], [], color='k', lw=1.4, marker='o', ms=2.2, label='ERA5参照路径'),
                              Line2D([], [], color=cmap(0.45), lw=0.9, marker='o', ms=2.6, mec='k', mew=0.3, label='盘古预报'),
                              Line2D([], [], color='k', ls='', marker='*', ms=7, mec='white', label='参照登陆点')],
                     loc='upper right', handlelength=1.6, borderaxespad=0.4)
    axs[0, 1].legend(handles=[Line2D([], [], color='0.35', ls='', marker='o', ms=4, mec='k', mew=0.4, label='盘古预报登陆点'),
                              Line2D([], [], color='0.35', ls='', marker='^', ms=4.5, mfc='white', mew=1.0, label='线性外推登陆点'),
                              Line2D([], [], color='k', ls='', marker='*', ms=7, mec='white', label='参照登陆点')],
                     loc='lower right', handletextpad=0.3, borderaxespad=0.4)
    lead_bar(fig, colour, hours, axs[0, 0].inset_axes([0.05, 0.075, 0.42, 0.032]))
    for ax, key, label in ((axs[1, 0], 'landfall_distance_km', '登陆点误差/km'), (axs[1, 1], 'landfall_time_bias_h', '登陆时间偏差/h')):
        ax.plot(hours, [r['linear_error'][key] for r in rs], color='0.45', ls=(0, (4, 2)), lw=1.0, zorder=3)
        ax.plot(hours, [r['pangu_error'][key] for r in rs], color='k', lw=1.0, zorder=4)
        for r in rs:
            c = colour[r['hours_before_landfall']]
            ax.plot(r['hours_before_landfall'], r['linear_error'][key], '^', ms=4.5, mfc='white', mec=c, mew=1.0, zorder=5)
            ax.plot(r['hours_before_landfall'], r['pangu_error'][key], 'o', ms=4, mfc=c, mec='k', mew=0.4, zorder=6)
        ax.set_xlim(37.5, 4.5)
        ax.set_xticks(range(36, 5, -6))
        ax.xaxis.set_minor_locator(MultipleLocator(3))
        ax.set_xlabel('起报时刻距登陆的时间/h')
        ax.set_ylabel(label)
    axs[1, 0].set_ylim(-8, 175)
    axs[1, 1].axhline(0, color='k', lw=0.6, zorder=1)
    axs[1, 1].set_ylim(-3, 18)
    axs[1, 1].yaxis.set_major_locator(MultipleLocator(3))
    pair_legend(axs[1, 0], columnspacing=1.2)
    for ax, letter in zip(axs.ravel(), 'abcd'):
        ps.panel(ax, letter)
    save(fig, 'fig4_1')


def fig4_2():
    rs = runs()
    hours = [r['hours_before_landfall'] for r in rs]
    colour = start_colours(hours)
    landfall = hours_of(rs[0]['era5_landfall']['first_mainland']['time_utc'])
    fig = ps.figure(ps.FULL, 5.9)
    axs = fig.subplots(1, 3)
    for ax, key in zip(axs[:2], ('pangu_km', 'linear_km')):
        ax.axvspan(0, 13, color='0.93', lw=0, zorder=0)
        ax.axvline(0, color='k', lw=0.7, ls=(0, (4, 2.5)), zorder=1)
        for r in rs:
            rows = [x for x in r['track_errors'] if x[key] is not None]
            ax.plot([hours_of(x['valid_utc']) - landfall for x in rows], [x[key] for x in rows], color=colour[r['hours_before_landfall']], lw=0.9,
                    marker='o', ms=2.0, zorder=3)
        ax.set_xlim(-37, 13)
        ax.set_xticks(range(-36, 13, 12))
        ax.xaxis.set_minor_locator(MultipleLocator(3))
        ax.set_ylim(0, 410)
        ax.set_xlabel('相对参照路径登陆的时间/h')
    axs[0].set_ylabel('路径误差/km')
    axs[1].tick_params(labelleft=False)
    axs[0].text(0.04, 0.95, '盘古预报', transform=axs[0].transAxes, ha='left', va='top')
    axs[1].text(0.04, 0.95, '线性外推', transform=axs[1].transAxes, ha='left', va='top')
    lead_bar(fig, colour, hours, axs[0].inset_axes([0.06, 0.66, 0.50, 0.04]), loc='left')
    c = axs[2]
    c.plot(hours, [r['preland']['linear_mae_km'] for r in rs], color='0.45', ls=(0, (4, 2)), lw=1.0, zorder=3)
    c.plot(hours, [r['preland']['pangu_mae_km'] for r in rs], color='k', lw=1.0, zorder=4)
    for r in rs:
        h = r['hours_before_landfall']
        c.plot(h, r['preland']['linear_mae_km'], '^', ms=4.5, mfc='white', mec=colour[h], mew=1.0, zorder=5)
        c.plot(h, r['preland']['pangu_mae_km'], 'o', ms=4, mfc=colour[h], mec='k', mew=0.4, zorder=6)
    c.set_xlim(37.5, 4.5)
    c.set_xticks(range(36, 5, -6))
    c.xaxis.set_minor_locator(MultipleLocator(3))
    c.set_ylim(0, 130)
    c.set_xlabel('起报时刻距登陆的时间/h')
    c.set_ylabel('登陆前路径误差MAE/km')
    pair_legend(c, columnspacing=1.0, handletextpad=0.4)
    for ax, letter in zip(axs, 'abc'):
        ps.panel(ax, letter)
    save(fig, 'fig4_2')


def members():
    out = [read(p) for p in (OUT / 'discovery').glob('*.json')]
    return sorted(out, key=lambda r: (-r['response_primary_J_kg'], r['member']))


def fig4_3():
    ms = members()
    hours = np.arange(0, 73, 3)
    j = np.array([[{x['lead_hours']: x['response_J_kg'] for x in m['scores']}[h] for h in hours] for m in ms])
    mean = np.array([m['response_primary_J_kg'] for m in ms])
    fig = ps.figure(ps.FULL, 5.6)
    a, b, c = fig.subplots(1, 3)
    for k, y in enumerate(j):
        a.plot(hours, y, color=plt.get_cmap('turbo')(1 - k / len(j)), alpha=0.35, lw=0.5, zorder=2)
    for y in j[:20]:
        b.plot(hours, y, color='#e8a0a0', lw=0.5, zorder=2)
    b.plot(hours, j[:20].mean(axis=0), color=ps.RED, lw=1.6, label='高响应成员平均', zorder=4)
    for ax in (a, b):
        ax.axvspan(*WINDOW, color='0.92', lw=0, zorder=0)
        ax.set_xlim(0, 72)
        ax.set_xticks(range(0, 73, 24))
        ax.xaxis.set_minor_locator(MultipleLocator(6))
        ax.set_ylim(0, 8.2)
        ax.set_xlabel('预报时效/h')
    a.set_ylabel('评价区响应/(J/kg)')
    b.legend(loc='upper left')
    rank = np.arange(1, len(mean) + 1)
    c.bar(rank[20:], mean[20:], width=1.0, color='0.72', lw=0, zorder=2)
    c.bar(rank[:20], mean[:20], width=1.0, color=ps.RED, lw=0, zorder=3)
    c.axhline(mean.mean(), color='k', lw=0.7, ls=(0, (4, 2.5)), zorder=4)
    c.text(198, mean.mean() + 0.04, '全部成员平均', ha='right', va='bottom', fontsize=ps.SMALL)
    c.set_xlim(0, 201)
    c.set_xticks([1, 50, 100, 150, 200])
    c.set_ylim(0, 2.4)
    c.set_xlabel('成员序号')
    c.set_ylabel('18～30 h平均响应/(J/kg)')
    for ax, letter in zip((a, b, c), 'abc'):
        ps.panel(ax, letter)
    save(fig, 'fig4_3')


def fig4_4():
    ms = members()
    ctl = {r['init']: r for r in runs()}[INIT]
    era = [dict(t, lead_hours=3 * i) for i, t in enumerate(ctl['era5_track'])]
    views = [((119.7, 125.9), (23.1, 31.6)), ((120.45, 121.75), (27.65, 28.65))]
    ratio = [(x[1] - x[0]) * np.cos(np.deg2rad(np.mean(y))) / (y[1] - y[0]) for x, y in views]
    fig = ps.figure(ps.FULL, 9.6)
    a, b = fig.subplots(1, 2, width_ratios=ratio)
    ps.geo(a, ps.coast('h', 118, 128, 22, 33, thin=3), *views[0], 2, 2)
    ps.geo(b, ps.coast('h', 119.5, 123, 26.8, 29.5), *views[1], 0.4, 0.2, lw=0.5)
    for group, colour, lw, z in ((ms[20:], '0.6', 0.3, 3), (ms[:20], ps.RED, 0.45, 4)):
        a.add_collection(LineCollection([np.column_stack(lonlat(m['track'], 36)) for m in group], colors=colour, linewidths=lw, alpha=0.6, zorder=z))
        pts = np.array([[m['events']['first_mainland']['lon'], m['events']['first_mainland']['lat']] for m in group])
        b.scatter(*pts.T, s=7 if z == 3 else 11, marker='o', facecolor='none' if z == 3 else colour, edgecolor=colour, linewidth=0.5, zorder=z + 2)
    for track, land, colour in ((era, ctl['era5_landfall'], C_ERA5), (ctl['pangu_track'], ctl['pangu_landfall'], C_PANGU)):
        for ax in (a, b):
            ax.plot(*lonlat(track, 36), color=colour, lw=1.3, zorder=7)
        p = land['first_mainland']
        for ax, size in ((a, 55), (b, 110)):
            ax.scatter(p['lon'], p['lat'], marker='*', s=size, facecolor=colour, edgecolor='white', linewidth=0.5, zorder=9)
    (x0, x1), (y0, y1) = views[1]
    a.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec='0.3', lw=0.6, ls=(0, (3, 2)), zorder=8))
    ps.panel(a, 'a')
    ps.panel(b, 'b')
    fig.legend(handles=[Line2D([], [], color=C_ERA5, lw=1.3, label='ERA5参照路径'), Line2D([], [], color=C_PANGU, lw=1.3, label='控制预报'),
                        Line2D([], [], color=ps.RED, lw=0.7, marker='o', markersize=3, label='高响应成员'),
                        Line2D([], [], color='0.6', lw=0.7, marker='o', markersize=3, markerfacecolor='none', label='其余成员'),
                        Line2D([], [], color='k', ls='', marker='*', markersize=7, markeredgecolor='white', label='首次大陆登陆点')],
               loc='outside lower center', ncols=5, columnspacing=1.3, handletextpad=0.4)
    save(fig, 'fig4_4')


def colour_bar(fig, cf, cax):
    bar = fig.colorbar(cf, cax=cax, ticks=np.arange(0.1, 1.01, 0.3))
    bar.set_label('归一化初始干能量')
    bar.ax.tick_params(direction='in', length=2)
    bar.outline.set_linewidth(0.6)


def centre_handle():
    return Line2D([], [], marker='*', ls='', markersize=7.5, markerfacecolor='white', markeredgecolor='k', markeredgewidth=0.7, label='起报时刻台风中心')


def fig4_5():
    rho = np.load(OUT / 'regions' / 'top20_density.npy')
    centre = {r['init']: r for r in runs()}[INIT]['era5_track'][0]
    levels, cmap, norm = ps.density_scale()
    extent = ((80, 180), (0, 60))
    fig, (ax,), cax, legend_y = ps.map_figure(ps.MID, [extent])
    cf = ax.contourf(LON, LAT, rho / rho.max(), levels=levels, cmap=cmap, norm=norm, zorder=1)
    ps.geo_axes(ax, ps.coast('i', 80, 180, 0, 60, thin=4), *extent, 20, 10)
    dash = (0, (4, 2.5))
    north, west, south, east = A_BOX
    ax.add_patch(plt.Rectangle((west, south), east - west, north - south, fill=False, ec='k', lw=0.9, ls=dash, zorder=6))
    ax.scatter(centre['lon'], centre['lat'], marker='*', s=42, facecolor='white', edgecolor='k', linewidth=0.7, zorder=8)
    colour_bar(fig, cf, cax)
    fig.legend(handles=[Line2D([], [], color='k', lw=0.9, ls=dash, label='评价区'), centre_handle()],
               loc='center', bbox_to_anchor=(0.5, legend_y), ncols=2, columnspacing=1.6, handletextpad=0.5)
    save(fig, 'fig4_5')


def fig5_1():
    reg = read(OUT / 'regions' / 'regions.json')['regions']
    rho = np.load(OUT / 'regions' / 'top20_density.npy')
    rho = np.minimum(rho / rho.max(), 1.0)
    rc = np.load(OUT / 'regions' / 'RC.npy')
    grid = {r['valid_utc']: r for r in read(OUT / 'era5_track.json')['grid']}[when(INIT).isoformat()]
    coast = ps.coast_domain()
    levels, cmap, norm = ps.density_scale()
    extents = [((80, 180), (0, 60)), ((102, 146), (10, 47))]
    fig, axs, cax, legend_y = ps.map_figure(ps.FULL, extents)
    dash = (0, (4, 2.5))
    north, west, south, east = A_BOX
    for ax, (xlim, ylim), step, letter in zip(axs, extents, [(20, 10), (10, 10)], 'ab'):
        cf = ax.contourf(LON, LAT, rho, levels=levels, cmap=cmap, norm=norm, zorder=1)
        ps.geo_axes(ax, coast, xlim, ylim, *step)
        for name in REGIONS[1:]:
            r = reg[name]
            ax.add_patch(Rectangle((r['west'] - .125, r['south'] - .125), r['east'] - r['west'] + .25, r['north'] - r['south'] + .25,
                                   fill=False, edgecolor=ps.BLUE, lw=0.6 if letter == 'a' else 0.8, zorder=6))
        ax.add_patch(Rectangle((west, south), east - west, north - south, fill=False, edgecolor='k', lw=0.9, ls=dash, zorder=7))
        ax.contour(LON, LAT, rc, levels=[.5], colors='k', linewidths=1.2 if letter == 'a' else 1.5, zorder=8)
        ax.scatter(grid['lon'], grid['lat'], marker='*', s=26 if letter == 'a' else 62, facecolor='white', edgecolor='k', linewidth=0.7, zorder=12)
        ps.panel(ax, letter)
    for name in REGIONS[1:]:
        r = reg[name]
        axs[1].text(r['west'] + .7, r['north'] - .7, name, color=ps.BLUE, ha='left', va='top', fontsize=ps.SMALL, zorder=11, path_effects=HALO_THIN)
    axs[1].text(127.4, 26.2, 'RC', color='k', ha='center', va='center', fontsize=ps.BASE, zorder=13, path_effects=HALO_THIN)
    colour_bar(fig, cf, cax)
    fig.legend(handles=[Line2D([], [], color='k', lw=1.5, label='敏感区RC'), Line2D([], [], color=ps.BLUE, lw=0.8, label='对照方框R0～R8'),
                        Line2D([], [], color='k', lw=0.9, ls=dash, label='评价区'), centre_handle()],
               loc='center', bbox_to_anchor=(0.5, legend_y), ncols=4, columnspacing=1.6, handletextpad=0.5)
    save(fig, 'fig5_1')


def improvement():
    with (OUT / 'improvement.csv').open(encoding='utf-8') as f:
        return [{**r, 'eta': float(r['eta'])} for r in csv.DictReader(f)]


def region_bars(ax, rows, eta, metric):
    cells = {r['region']: r for r in rows if r['eta'] == eta and r['metric'] == metric}
    mean, low, high = (np.array([float(cells[r][k]) for r in REGIONS]) for k in ('improvement_percent', 'improvement_ci95_low', 'improvement_ci95_high'))
    span = max(high.max(), 0) - min(low.min(), 0)
    for i, r in enumerate(REGIONS):
        kind = {'color': LIGHT_RED} if r == 'RC' else {'color': 'white', 'hatch': '////'} if r == 'R0' else {'color': LIGHT_GREY}
        ax.bar(i, mean[i], width=0.68, linewidth=0.5, zorder=3, edgecolor=LIGHT_RED if r == 'R0' else EDGE, **kind)
        ax.text(i, max(high[i], 0) + span * 0.02, f'{mean[i]:.1f}'.replace('-', '−'), ha='center', va='bottom', fontsize=7.5)
    x = np.arange(len(REGIONS))
    ax.errorbar(x, mean, yerr=[mean - low, high - mean], fmt='none', ecolor=EDGE, elinewidth=0.6, capsize=1.8, capthick=0.6, zorder=4)
    ax.axhline(0, color=EDGE, lw=0.6, zorder=2)
    ax.set_xlim(-0.6, len(x) - 0.4)
    ax.tick_params(axis='x', which='both', top=False, length=0)
    ax.set_xticks(x, ['RC\n' if r == 'RC' else f'{r}\n{NAMES[r]}' for r in REGIONS])


def region_legend(fig):
    fig.legend(handles=[Patch(facecolor=LIGHT_RED, edgecolor=EDGE, lw=0.5, label='敏感区RC'),
                        Patch(facecolor='white', edgecolor=LIGHT_RED, hatch='////', lw=0.5, label='同位置方框R0'),
                        Patch(facecolor=LIGHT_GREY, edgecolor=EDGE, lw=0.5, label='周围对照区R1～R8')],
               loc='outside lower center', ncols=3, columnspacing=1.8, handletextpad=0.5, handlelength=1.4)


def fig5_2():
    rows = improvement()
    fig = ps.figure(ps.FULL, 6.4)
    axs = fig.subplots(1, 2)
    for ax, (metric, title), letter in zip(axs, METRIC, 'ab'):
        region_bars(ax, rows, 1.0, metric)
        ax.set_ylim(-25, 52)
        ps.panel(ax, letter, title)
    axs[0].set_ylabel('预报改进率/%')
    region_legend(fig)
    save(fig, 'fig5_2')


def fig5_3():
    rows = improvement()
    fig = ps.figure(ps.FULL, 15.4)
    axs = fig.subplots(3, 2, sharex=True, sharey=True)
    for i, eta in enumerate(ETAS[1:]):
        for k, (metric, title) in enumerate(METRIC):
            region_bars(axs[i, k], rows, eta, metric)
            ps.panel(axs[i, k], 'abcdef'[2 * i + k], f'削减{eta:.0%}，{title}')
            axs[i, k].set_ylim(-16, 42)
        axs[i, 0].set_ylabel('预报改进率/%')
    region_legend(fig)
    save(fig, 'fig5_3')


if __name__ == '__main__':
    ps.configure()
    for f in (fig2_2, fig4_1, fig4_2, fig4_3, fig4_4, fig4_5, fig5_1, fig5_2, fig5_3):
        f()
