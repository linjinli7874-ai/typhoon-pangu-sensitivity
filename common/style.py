import struct
import zipfile
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.ticker import FuncFormatter, MultipleLocator

from common.config import GSHHG

CM = 1 / 2.54
FULL, MID = 16.0, 12.0
BASE, SMALL, TINY = 9.0, 8.0, 7.5
RED = '#b2182b'
BLUE = '#2166ac'
COAST_COLOUR = '#606060'


def configure():
    for name in ('times.ttf', 'timesi.ttf', 'timesbd.ttf', 'timesbi.ttf', 'simsun.ttc'):
        p = Path('C:/Windows/Fonts') / name
        if p.is_file():
            fm.fontManager.addfont(str(p))
    for family in ('Times New Roman', 'SimSun'):
        fm.findfont(fm.FontProperties(family=family), fallback_to_default=False)
    mpl.rcParams.update({
        'font.family': ['Times New Roman', 'SimSun'], 'font.serif': ['Times New Roman', 'SimSun'],
        'mathtext.fontset': 'custom', 'mathtext.rm': 'Times New Roman', 'mathtext.it': 'Times New Roman:italic',
        'mathtext.bf': 'Times New Roman:bold', 'mathtext.fallback': None,
        'font.size': BASE, 'axes.labelsize': BASE, 'axes.titlesize': BASE, 'xtick.labelsize': SMALL, 'ytick.labelsize': SMALL,
        'legend.fontsize': SMALL, 'legend.frameon': False, 'legend.handlelength': 1.8, 'legend.borderaxespad': 0.3,
        'axes.linewidth': 0.6, 'axes.spines.top': True, 'axes.spines.right': True, 'axes.grid': False, 'axes.unicode_minus': False,
        'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True,
        'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 3, 'ytick.major.size': 3,
        'xtick.minor.width': 0.4, 'ytick.minor.width': 0.4, 'xtick.minor.size': 1.6, 'ytick.minor.size': 1.6,
        'lines.linewidth': 1.1, 'lines.markersize': 3.5, 'patch.linewidth': 0.5, 'hatch.linewidth': 0.5,
        'figure.dpi': 150, 'savefig.dpi': 600, 'savefig.facecolor': 'white', 'figure.facecolor': 'white'})


def figure(width_cm, height_cm):
    return plt.figure(figsize=(width_cm * CM, height_cm * CM), layout='constrained')


def panel(ax, letter, text=''):
    ax.set_title(f'({letter})' + (' ' + text if text else ''), loc='left', fontsize=BASE, pad=3)


def density_scale():
    levels = np.arange(0.1, 1.01, 0.1)
    cmap = ListedColormap(plt.get_cmap('Reds')(np.linspace(0.12, 0.95, len(levels) - 1)))
    cmap.set_under('white')
    return levels, cmap, BoundaryNorm(levels, cmap.N)


def geo_axes(ax, coast, xlim, ylim, xstep, ystep):
    ax.add_collection(LineCollection(coast, colors=COAST_COLOUR, linewidths=0.35, zorder=3))
    ax.set(xlim=xlim, ylim=ylim)
    ax.xaxis.set_major_locator(MultipleLocator(xstep))
    ax.yaxis.set_major_locator(MultipleLocator(ystep))
    ax.xaxis.set_minor_locator(MultipleLocator(xstep / 2))
    ax.yaxis.set_minor_locator(MultipleLocator(ystep / 2))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}°E'))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: ('' if ylim[0] == 0 else 'EQ') if v == 0 else f'{v:g}°N'))
    ax.tick_params(which='both', top=True, right=True)


def geo(ax, segments, xlim, ylim, xstep, ystep, lw=0.4):
    ax.add_collection(LineCollection(segments, colors=COAST_COLOUR, linewidths=lw, zorder=3))
    ax.set(xlim=xlim, ylim=ylim)
    ax.set_aspect(1 / np.cos(np.deg2rad(np.mean(ylim))))
    ax.xaxis.set_major_locator(MultipleLocator(xstep))
    ax.yaxis.set_major_locator(MultipleLocator(ystep))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}°E'))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}°N'))


def map_figure(width_cm, extents, legend_rows=1, top=0.62, gap=1.45):
    left, tick_room, row, margin = 1.22, 0.62, 0.52, 0.12
    bar_gap, bar_width, bar_room = 0.22, 0.26, 1.18
    ratio = [(x[1] - x[0]) * np.cos(np.deg2rad(np.mean(y))) / (y[1] - y[0]) for x, y in extents]
    height = (width_cm - left - gap * (len(extents) - 1) - bar_gap - bar_width - bar_room) / sum(ratio)
    bottom = tick_room + legend_rows * row + margin
    total = top + height + bottom
    fig = plt.figure(figsize=(width_cm * CM, total * CM))
    axes, x0 = [], left
    for r in ratio:
        axes.append(fig.add_axes([x0 / width_cm, bottom / total, r * height / width_cm, height / total]))
        x0 += r * height + gap
    cax = fig.add_axes([(x0 - gap + bar_gap) / width_cm, bottom / total, bar_width / width_cm, height / total])
    return fig, axes, cax, (margin + legend_rows * row / 2) / total


def coast(resolution, west, east, south, north, thin=1):
    out = []
    with zipfile.ZipFile(GSHHG) as z:
        name = next(n for n in z.namelist() if n.endswith(f'gshhs_{resolution}.b'))
        with z.open(name) as f:
            while header := f.read(44):
                _, n, flag, w, e, s, nn, *_ = struct.unpack('>11i', header)
                raw = f.read(8 * n)
                if flag & 255 != 1 or nn < south * 1e6 or s > north * 1e6 or (w > east * 1e6 and e > east * 1e6) or e < west * 1e6:
                    continue
                xy = np.frombuffer(raw, dtype='>i4').astype(float).reshape(-1, 2) / 1e6
                keep = (xy[:, 0] >= west) & (xy[:, 0] <= east) & (xy[:, 1] >= south) & (xy[:, 1] <= north)
                ids = np.flatnonzero(keep[:-1] & keep[1:])
                if not len(ids):
                    continue
                for chunk in np.split(ids, np.flatnonzero(np.diff(ids) > 1) + 1):
                    a = xy[chunk[0]:chunk[-1] + 2]
                    out.append(np.vstack((a[::thin], a[-1])))
    return out


def coast_domain():
    out = []
    with zipfile.ZipFile(GSHHG) as z:
        name = next(n for n in z.namelist() if n.endswith('gshhs_i.b'))
        with z.open(name) as f:
            while header := f.read(44):
                _, n, flag, west, east, south, north, *_ = struct.unpack('>11i', header)
                raw = f.read(8 * n)
                if flag & 255 != 1 or north < -2e6 or south > 62e6:
                    continue
                xy = np.frombuffer(raw, dtype='>i4').astype(float).reshape(-1, 2) / 1e6
                xy[:, 0] = np.where(xy[:, 0] > 180, xy[:, 0] - 360, xy[:, 0])
                keep = (xy[:, 0] >= 78) & (xy[:, 0] <= 182) & (xy[:, 1] >= -2) & (xy[:, 1] <= 62)
                ids = np.flatnonzero(keep[:-1] & keep[1:] & (np.abs(np.diff(xy[:, 0])) < 2))
                if not len(ids):
                    continue
                for chunk in np.split(ids, np.flatnonzero(np.diff(ids) > 1) + 1):
                    a = xy[chunk[0]:chunk[-1] + 2]
                    step = max(1, len(a) // 2000)
                    out.append(np.vstack((a[::step], a[-1])))
    return out
