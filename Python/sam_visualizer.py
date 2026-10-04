#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualizador de Autómata de Sufijos (SAM)  —  solo requiere matplotlib
======================================================================

Uso
---
    python sam_visualizer.py                             # ventana con "banana"
    python sam_visualizer.py abbb                        # otra cadena
    python sam_visualizer.py banana --png sam.png        # guarda PNG
    python sam_visualizer.py banana --svg sam.svg        # guarda SVG
    python sam_visualizer.py banana --pdf sam.pdf        # guarda PDF
    python sam_visualizer.py banana --paso -1            # último paso (por defecto)
    python sam_visualizer.py banana --paso 3 --strings --links

Tres vistas simultáneas:
  1) Arriba-izquierda : grafo del SAM (transiciones) en columnas por len.
  2) Arriba-derecha   : clon del grafo mostrando sólo los suffix links.
  3) Abajo            : cajas con todas las cadenas de cada estado.

Cuadro de texto: click para enfocar · doble click / Ctrl+A = seleccionar todo ·
  arrastrar o Shift+flechas = seleccionar · Backspace/Delete borran ·
  Ctrl+C / X / V = copiar / cortar / pegar (Cmd en Mac) · ✕ = borrar todo ·
  Enter = construir.

Colores: azul = último estado · amarillo = clon · verde = terminal ·
         magenta = afectado por el paso actual (y no creado en él).
"""

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass, field

import matplotlib

if any(f in sys.argv for f in ("--png", "--svg", "--pdf")):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import (
    Circle,
    FancyArrowPatch,
    FancyBboxPatch,
    Polygon,
    Rectangle,
)
from matplotlib.transforms import blended_transform_factory
from matplotlib.widgets import Button, CheckButtons, Slider

# ---------------------------------------------------------------------------
# Tema
# ---------------------------------------------------------------------------
BG = "#0f1115"
PANEL = "#151922"
FG = "#e8eaed"
MUTED = "#9aa0a6"
NODE_FILL = "#1f242d"
NODE_EDGE = "#8b919a"
LAST_FILL = "#a8c7fa"
CLONE_FILL = "#f9ab00"
TERM = "#34d399"
TOUCH = "#e879f9"  # estados afectados por el paso actual
TRANS = "#c3c9d3"
LINK = "#7f8cff"
RED = "#ff6b5e"
DARK_TXT = "#0b1220"

MAX_LEN = 30

# Medidas en unidades de datos (la fuente se ajusta a la escala).
FU = 0.22
CW = 0.6 * FU
LH = 1.5 * FU
R = 0.36

# En la vista de cajas un estado puede contener hasta MAX_LEN cadenas.
# A partir de este umbral se compacta el contenido con "⋯".
MAX_BOX_STRINGS = 6


def vis(s):
    """Representación visible de una cadena (espacio -> ␣, vacía -> ε)."""
    if s == "":
        return "ε"
    return s.replace(" ", "␣")


# ---------------------------------------------------------------------------
# SAM con historial
# ---------------------------------------------------------------------------
@dataclass
class Snap:
    ln: list
    lk: list
    nx: list
    fp: list
    cl: list
    last: int
    prefix: str
    log: list = field(default_factory=list)
    new: set = field(default_factory=set)
    touched: set = field(default_factory=set)

    @property
    def n(self):
        return len(self.ln)


def _touched_from_log(log):
    """Estados S\d+ mencionados en el log del paso."""
    out = set()
    for line in log:
        for m in re.finditer(r"S(\d+)", line):
            out.add(int(m.group(1)))
    return out


def build_history(text):
    ln, lk, nx, fp, cl = [0], [-1], [{}], [-1], [False]
    last = 0

    def snap(prefix, log, new):
        return Snap(
            list(ln),
            list(lk),
            [dict(d) for d in nx],
            list(fp),
            list(cl),
            last,
            prefix,
            log,
            set(new),
            _touched_from_log(log),
        )

    hist = [snap("", ["Estado inicial: sólo la raíz S0 (cadena vacía)."], [])]
    for i, c in enumerate(text):
        log, new = [], set()
        cur = len(ln)
        ln.append(ln[last] + 1)
        lk.append(-1)
        nx.append({})
        fp.append(i)
        cl.append(False)
        new.add(cur)
        log.append(f"Se agrega «{vis(c)}»: nuevo estado S{cur} con len = {ln[cur]}.")
        p = last
        added = []
        while p != -1 and c not in nx[p]:
            nx[p][c] = cur
            added.append(p)
            p = lk[p]
        if added:
            log.append(
                "Transición «%s» hacia S%d creada desde: %s (subiendo por los suffix links)."
                % (vis(c), cur, ", ".join(f"S{x}" for x in added))
            )
        if p == -1:
            lk[cur] = 0
            log.append(
                f"Se llegó a la raíz sin encontrar «{vis(c)}» → link(S{cur}) = S0."
            )
        else:
            q = nx[p][c]
            if ln[p] + 1 == ln[q]:
                lk[cur] = q
                log.append(
                    f"S{p} ya tiene «{vis(c)}» hacia S{q} y len(S{p})+1 = len(S{q}) "
                    f"→ link(S{cur}) = S{q}."
                )
            else:
                cn = len(ln)
                ln.append(ln[p] + 1)
                lk.append(lk[q])
                nx.append(dict(nx[q]))
                fp.append(fp[q])
                cl.append(True)
                new.add(cn)
                redirected = []
                while p != -1 and nx[p].get(c) == q:
                    nx[p][c] = cn
                    redirected.append(p)
                    p = lk[p]
                lk[q] = cn
                lk[cur] = cn
                log.append(
                    f"S{redirected[0]} ya tiene «{vis(c)}» hacia S{q} pero len(S{redirected[0]})+1 "
                    f"≠ len(S{q}) → se CLONA S{q} como S{cn} (len = {ln[cn]})."
                )
                log.append(
                    "Se redirigen a S%d las transiciones «%s» de: %s; link(S%d) = link(S%d) = S%d."
                    % (cn, vis(c), ", ".join(f"S{x}" for x in redirected), q, cur, cn)
                )
        last = cur
        hist.append(snap(text[: i + 1], log, new))
    return hist


def state_strings(s, text, v):
    """Todas las cadenas del estado v, de la más larga a la más corta."""
    if v == 0:
        return [""]
    u = s.lk[v]
    e = s.fp[v]
    return [text[e - l + 1 : e + 1] for l in range(s.ln[v], s.ln[u], -1)]


def display_strings(strs, max_n=None):
    """Compacta una lista larga conservando los extremos con un '⋯' en medio."""
    if max_n is None or len(strs) <= max_n:
        return strs
    head = max(1, max_n // 3)
    tail = max(1, max_n - head - 1)
    return strs[:head] + ["⋯"] + strs[-tail:]


def terminals(s):
    t, p = set(), s.last
    while p != -1:
        t.add(p)
        p = s.lk[p]
    return t


# ---------------------------------------------------------------------------
# Layouts
# ---------------------------------------------------------------------------
def node_size(s, text, v, strings_mode):
    if not strings_mode:
        return (2 * R, 2 * R)
    lines = [f"S{v} · L{s.ln[v]}"] + [vis(x) for x in state_strings(s, text, v)]
    w = max(len(x) for x in lines) * CW + 0.34
    h = len(lines) * LH + 0.22
    return (w, h)


def stack_column(nodes, sizes, gap=0.32):
    total = sum(sizes[v][1] for v in nodes) + gap * (len(nodes) - 1)
    cur = total / 2
    out = {}
    for v in nodes:
        h = sizes[v][1]
        out[v] = cur - h / 2
        cur -= h + gap
    return out


def layout_graph(s, sizes, strings_mode):
    n = s.n
    cols = {}
    for v in sorted(range(n), key=lambda v: (s.fp[v], s.ln[v])):
        cols.setdefault(s.ln[v], []).append(v)
    preds = [[] for _ in range(n)]
    succs = [[] for _ in range(n)]
    for u in range(n):
        for v in set(s.nx[u].values()):
            preds[v].append(u)
            succs[u].append(v)
    y = {}
    for L in sorted(cols):
        y.update(stack_column(cols[L], sizes))

    def sweep(order, nbrs):
        for L in order:
            nodes = cols[L]
            if len(nodes) < 2:
                continue
            key = {
                v: (sum(y[p] for p in nbrs[v]) / len(nbrs[v])) if nbrs[v] else y[v]
                for v in nodes
            }
            nodes.sort(key=lambda v: (-key[v], s.fp[v]))
            y.update(stack_column(nodes, sizes))

    ls = sorted(cols)
    for _ in range(4):
        sweep(ls[1:], preds)
        sweep(ls[::-1][1:], succs)

    gap = 0.95 if strings_mode else 1.0
    xc, prev_w, cur = {}, None, 0.0
    for L in ls:
        wmax = max(sizes[v][0] for v in cols[L])
        if prev_w is not None:
            cur += prev_w / 2 + gap + wmax / 2
        xc[L] = cur
        prev_w = wmax
    return {v: (xc[s.ln[v]], y[v]) for v in range(n)}


def layout_tree(s, sizes, strings_mode):
    n = s.n
    children = [[] for _ in range(n)]
    for v in range(1, n):
        children[s.lk[v]].append(v)
    for c in children:
        c.sort(key=lambda v: (s.fp[v], s.ln[v]))
    xgap = (max(w for w, _ in sizes.values()) + 0.45) if strings_mode else 1.25
    xs, depth, counter = {}, {}, [0]

    def rec(v, d):
        depth[v] = d
        if not children[v]:
            xs[v] = counter[0] * xgap
            counter[0] += 1
        else:
            for c in children[v]:
                rec(c, d + 1)
            xs[v] = (xs[children[v][0]] + xs[children[v][-1]]) / 2

    sys.setrecursionlimit(10000)
    rec(0, 0)
    levels = max(depth.values()) + 1
    hlev = [max(sizes[v][1] for v in range(n) if depth[v] == d) for d in range(levels)]
    gapy = 0.85 if strings_mode else 1.15
    ylev, cur = [], 0.0
    for d in range(levels):
        if d:
            cur -= hlev[d - 1] / 2 + gapy + hlev[d] / 2
        ylev.append(cur)
    return {v: (xs[v], ylev[depth[v]]) for v in range(n)}


# ---------------------------------------------------------------------------
# Dibujo
# ---------------------------------------------------------------------------
def setup_axes(ax, xmin, xmax, ymin, ymax, title):
    """Ajusta límites 1:1 en la figura usando sólo la posición del axes
    (sin depender del renderer, que puede no estar listo la primera vez)."""
    ax.clear()
    ax.set_facecolor(PANEL)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#2a2f3a")

    pos = ax.get_position()
    fw, fh = ax.figure.get_size_inches()
    pw = max(pos.width * fw, 1e-3)
    ph = max(pos.height * fh, 1e-3)
    aspect = pw / ph

    xr, yr = xmax - xmin, ymax - ymin
    if xr / yr > aspect:
        new_yr = xr / aspect
        cy = (ymin + ymax) / 2
        ymin, ymax = cy - new_yr / 2, cy + new_yr / 2
    else:
        new_xr = yr * aspect
        cx = (xmin + xmax) / 2
        xmin, xmax = cx - new_xr / 2, cx + new_xr / 2
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_title(title, color=FG, loc="left", fontsize=12, pad=6)

    ppu = pw / (xmax - xmin) * 72.0  # puntos por unidad de datos
    return ppu


def fsz(units, ppu):
    return max(3.0, min(24.0, units * ppu))


def bounds(pos, sizes, margin=0.7):
    xmin = min(pos[v][0] - sizes[v][0] / 2 for v in pos) - margin
    xmax = max(pos[v][0] + sizes[v][0] / 2 for v in pos) + margin
    ymin = min(pos[v][1] - sizes[v][1] / 2 for v in pos) - margin
    ymax = max(pos[v][1] + sizes[v][1] / 2 for v in pos) + margin
    return xmin, xmax, ymin, ymax


def arrow(
    ax,
    p1,
    p2,
    pa=None,
    pb=None,
    color=FG,
    rad=0.0,
    lw=1.4,
    ls="-",
    z=2,
    ms=13,
    alpha=1.0,
):
    a = FancyArrowPatch(
        p1,
        p2,
        patchA=pa,
        patchB=pb,
        arrowstyle="-|>",
        mutation_scale=ms,
        connectionstyle=f"arc3,rad={rad}",
        color=color,
        lw=lw,
        linestyle=ls,
        zorder=z,
        shrinkA=0,
        shrinkB=1.5,
        alpha=alpha,
    )
    ax.add_patch(a)
    return a


def curve_mid(p1, p2, rad):
    (x1, y1), (x2, y2) = p1, p2
    dx, dy = x2 - x1, y2 - y1
    return ((x1 + x2) / 2 + 0.5 * rad * dy, (y1 + y2) / 2 - 0.5 * rad * dx)


def edge_label(ax, p1, p2, rad, text, ppu, color=FG, bg=PANEL, size=0.24):
    mx, my = curve_mid(p1, p2, rad)
    ax.text(
        mx,
        my,
        text,
        color=color,
        fontsize=fsz(size, ppu),
        fontweight="bold",
        family="monospace",
        ha="center",
        va="center",
        zorder=6,
        bbox=dict(boxstyle="round,pad=0.12", fc=bg, ec="none", alpha=0.9),
    )


def group_transitions(s):
    g = {}
    for u in range(s.n):
        for c, v in s.nx[u].items():
            g.setdefault((u, v), []).append(c)
    return {k: ",".join(vis(c) for c in sorted(cs)) for k, cs in g.items()}


def draw_nodes(ax, s, text, pos, sizes, strings_mode, ppu):
    term = terminals(s)
    patches = {}
    for v, (x, y) in pos.items():
        w, h = sizes[v]
        is_last = v == s.last
        is_clone = s.cl[v]
        is_new = v in s.new
        is_touch = (v in s.touched) and (not is_new) and (not is_last) and (v != 0)

        fc = LAST_FILL if is_last else (CLONE_FILL if is_clone else NODE_FILL)
        if is_new:
            ec, lw = "#ffffff", 2.4
        elif is_touch:
            ec, lw = TOUCH, 2.2
        else:
            ec, lw = NODE_EDGE, 1.5
        tc = DARK_TXT if (is_last or is_clone) else FG
        sub = "#3b4557" if (is_last or is_clone) else MUTED

        if strings_mode:
            patch = FancyBboxPatch(
                (x - w / 2, y - h / 2),
                w,
                h,
                boxstyle="round,pad=0,rounding_size=0.1",
                fc=fc,
                ec=ec,
                lw=lw,
                zorder=3,
            )
            ring = FancyBboxPatch(
                (x - w / 2 - 0.07, y - h / 2 - 0.07),
                w + 0.14,
                h + 0.14,
                boxstyle="round,pad=0,rounding_size=0.15",
                fc="none",
                ec=TERM,
                lw=1.8,
                zorder=3,
            )
        else:
            patch = Circle((x, y), R, fc=fc, ec=ec, lw=lw, zorder=3)
            ring = Circle((x, y), R + 0.075, fc="none", ec=TERM, lw=1.8, zorder=3)
        ax.add_patch(patch)
        if v in term:
            ax.add_patch(ring)
        patches[v] = patch

        if strings_mode:
            top = y + h / 2 - 0.11
            ax.text(
                x,
                top - LH / 2,
                f"S{v} · L{s.ln[v]}",
                color=sub,
                fontsize=fsz(FU, ppu),
                family="monospace",
                ha="center",
                va="center",
                zorder=4,
                fontweight="bold",
            )
            for i, st in enumerate(state_strings(s, text, v)):
                ax.text(
                    x,
                    top - LH * (i + 1.5),
                    vis(st),
                    color=tc,
                    fontsize=fsz(FU, ppu),
                    family="monospace",
                    ha="center",
                    va="center",
                    zorder=4,
                )
        else:
            ax.text(
                x,
                y + 0.09,
                f"S{v}",
                color=tc,
                fontsize=fsz(0.25, ppu),
                family="monospace",
                fontweight="bold",
                ha="center",
                va="center",
                zorder=4,
            )
            ax.text(
                x,
                y - 0.14,
                f"L:{s.ln[v]}",
                color=sub,
                fontsize=fsz(0.19, ppu),
                family="monospace",
                ha="center",
                va="center",
                zorder=4,
            )
    return patches


def route_rad(pos, sizes, u, v, prefer, margin=0.24, samples=30):
    """Curvatura para u->v que evita atravesar otros nodos (muestreo denso)."""
    p1, p2 = pos[u], pos[v]
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    others = [(pos[w], sizes[w]) for w in pos if w not in (u, v)]

    def clear(rad):
        cx, cy = (x1 + x2) / 2 + rad * dy, (y1 + y2) / 2 - rad * dx
        for i in range(1, samples + 1):
            t = i / (samples + 1.0)
            bx = (1 - t) ** 2 * x1 + 2 * t * (1 - t) * cx + t * t * x2
            by = (1 - t) ** 2 * y1 + 2 * t * (1 - t) * cy + t * t * y2
            for (px, py), (w, h) in others:
                if abs(bx - px) < w / 2 + margin and abs(by - py) < h / 2 + margin:
                    return False
        return True

    for m in (0.0, 0.1, 0.2, 0.3, 0.42, 0.56, 0.72, 0.9, 1.15):
        for sgn in (prefer, -prefer):
            r = sgn * m
            if clear(r):
                return r
    return prefer * 0.9


def draw_graph_panel(ax, s, text, sizes, strings_mode, show_links):
    pos = layout_graph(s, sizes, strings_mode)
    cy = sum(p[1] for p in pos.values()) / len(pos)
    routes = {}
    for (u, v), lab in group_transitions(s).items():
        p1, p2 = pos[u], pos[v]
        span = s.ln[v] - s.ln[u]
        prefer = -1 if (p1[1] + p2[1]) / 2 >= cy else 1
        if span == 1:
            rad = 0.0 if abs(p1[1] - p2[1]) < 1e-6 else prefer * 0.06
        else:
            rad = route_rad(pos, sizes, u, v, prefer)
        routes[(u, v)] = (rad, lab)

    xmin, xmax, ymin, ymax = bounds(pos, sizes)
    extra = [curve_mid(pos[u], pos[v], r) for (u, v), (r, _) in routes.items()]
    if show_links:
        extra += [curve_mid(pos[v], pos[s.lk[v]], 0.28) for v in range(1, s.n)]
    for mx, my in extra:
        xmin, xmax = min(xmin, mx - 0.4), max(xmax, mx + 0.4)
        ymin, ymax = min(ymin, my - 0.4), max(ymax, my + 0.4)
    ppu = setup_axes(
        ax, xmin, xmax, ymin, ymax, "1 · Autómata de sufijos (transiciones)"
    )
    patches = draw_nodes(ax, s, text, pos, sizes, strings_mode, ppu)
    for (u, v), (rad, lab) in routes.items():
        arrow(ax, pos[u], pos[v], patches[u], patches[v], TRANS, rad, lw=1.3)
        edge_label(ax, pos[u], pos[v], rad, lab, ppu)
    if show_links:
        for v in range(1, s.n):
            u = s.lk[v]
            arrow(
                ax,
                pos[v],
                pos[u],
                patches[v],
                patches[u],
                LINK,
                0.28,
                lw=1.1,
                ls="--",
                alpha=0.9,
            )
    return pos


def draw_link_panel(ax, s, text, sizes, strings_mode):
    pos = layout_tree(s, sizes, strings_mode)
    ppu = setup_axes(ax, *bounds(pos, sizes), "2 · Suffix links (clon del grafo)")
    patches = draw_nodes(ax, s, text, pos, sizes, strings_mode, ppu)
    for v in range(1, s.n):
        u = s.lk[v]
        arrow(ax, pos[v], pos[u], patches[v], patches[u], LINK, 0.0, lw=1.7)
    return pos


def rad_for(dist):
    return 0.34 if dist <= 4 else 0.34 * (4.0 / dist) ** 0.5


def draw_boxes_panel(ax, s, text, show_trans):
    n = s.n
    order = sorted(range(n), key=lambda v: (s.fp[v], s.ln[v]))
    term = terminals(s)
    PADX, PADY, GAP = 0.16, 0.11, 0.42

    info = {}
    cursor = 0.0
    hmax = 0.0
    for v in order:
        strs = display_strings(state_strings(s, text, v), MAX_BOX_STRINGS)
        k = len(strs)
        if v == 0:
            w, h = 0.55, 0.42
        else:
            w = len(vis(strs[0])) * CW + 2 * PADX
            h = k * LH + 2 * PADY
        info[v] = dict(x0=cursor, w=w, h=h, strs=strs, xc=cursor + w / 2)
        cursor += w + GAP
        hmax = max(hmax, h)
    total = cursor - GAP

    y_top = hmax + 0.4
    y_bot = -0.85
    hb = ha = 0.0
    for v in range(1, n):
        d = abs(info[v]["xc"] - info[s.lk[v]]["xc"])
        hb = max(hb, 0.5 * rad_for(d) * d)
    trans = group_transitions(s)
    if show_trans:
        for u, v in trans:
            d = abs(info[v]["xc"] - info[u]["xc"])
            ha = max(ha, 0.5 * rad_for(d) * d)
    xmin, xmax = -0.6, total + 0.6
    ymin = (y_bot - ha - 0.6) if show_trans else -0.8
    ymax = y_top + hb + 0.6
    ppu = setup_axes(
        ax,
        xmin,
        xmax,
        ymin,
        ymax,
        "3 · Cajas: cada estado guarda todas sus cadenas (azul = suffix link, rojo = transición)",
    )

    dots_top, dots_bot = {}, {}
    for v in order:
        d = info[v]
        x0, w, h = d["x0"], d["w"], d["h"]
        k = len(d["strs"])
        slant = 0.16 if v == 0 else (k - 1) * CW + PADX * 0.9
        pts = [(x0, 0), (x0 + w, 0), (x0 + w, h), (x0 + slant, h)]

        is_last, is_clone, is_new = v == s.last, s.cl[v], v in s.new
        is_touch = (v in s.touched) and (not is_new) and (not is_last) and (v != 0)

        fc = "#26344d" if is_last else ("#3a2f10" if is_clone else NODE_FILL)
        if is_last:
            ec = LAST_FILL
        elif is_clone:
            ec = CLONE_FILL
        elif is_new:
            ec = "#ffffff"
        elif is_touch:
            ec = TOUCH
        else:
            ec = "#e8eaed"
        lw = 2.4 if is_new else (2.2 if is_touch else 1.6)
        ax.add_patch(Polygon(pts, closed=True, fc=fc, ec=ec, lw=lw, zorder=3))

        if v == 0:
            ax.text(
                x0 + w * 0.62,
                h / 2,
                "ε",
                color=FG,
                fontsize=fsz(FU, ppu),
                family="monospace",
                ha="center",
                va="center",
                zorder=4,
            )
        else:
            for i, st in enumerate(d["strs"]):
                ax.text(
                    x0 + w - PADX,
                    PADY + LH * (i + 0.5),
                    vis(st),
                    color=FG,
                    fontsize=fsz(FU, ppu),
                    family="monospace",
                    ha="right",
                    va="center",
                    zorder=4,
                )
        ax.text(
            d["xc"],
            -0.3,
            f"S{v}",
            color=TERM if v in term else MUTED,
            fontsize=fsz(0.2, ppu),
            family="monospace",
            fontweight="bold",
            ha="center",
            va="center",
            zorder=4,
        )

        ax.plot(
            [d["xc"], d["xc"]], [h, y_top], color=LINK, lw=0.8, alpha=0.35, zorder=2
        )
        ax.plot(d["xc"], y_top, "o", color=LINK, ms=max(2.5, 0.09 * ppu), zorder=5)
        dots_top[v] = (d["xc"], y_top)
        if show_trans:
            ax.plot(
                [d["xc"], d["xc"]],
                [-0.45, y_bot],
                color=RED,
                lw=0.8,
                alpha=0.35,
                zorder=2,
            )
            ax.plot(d["xc"], y_bot, "o", color=RED, ms=max(2.5, 0.09 * ppu), zorder=5)
            dots_bot[v] = (d["xc"], y_bot)

    for v in range(1, n):
        u = s.lk[v]
        dist = abs(dots_top[v][0] - dots_top[u][0])
        arrow(
            ax, dots_top[v], dots_top[u], None, None, LINK, rad_for(dist), lw=1.5, ms=11
        )
    if show_trans:
        for (u, v), lab in trans.items():
            dist = abs(dots_bot[v][0] - dots_bot[u][0])
            r = rad_for(dist)
            arrow(
                ax,
                dots_bot[u],
                dots_bot[v],
                None,
                None,
                RED,
                r,
                lw=1.3,
                ms=10,
                alpha=0.95,
            )
            edge_label(
                ax, dots_bot[u], dots_bot[v], r, lab, ppu, color="#ffb1a8", size=0.22
            )
    return info


# ---------------------------------------------------------------------------
# Portapapeles (Tk / Qt / comandos del sistema / pyperclip)
# ---------------------------------------------------------------------------
def _run(cmd, text=None):
    try:
        r = subprocess.run(cmd, input=text, capture_output=True, text=True, timeout=3)
        return r if r.returncode == 0 else None
    except Exception:
        return None


def clipboard_get(fig):
    canvas = fig.canvas
    if hasattr(canvas, "get_tk_widget"):
        try:
            return canvas.get_tk_widget().clipboard_get()
        except Exception:
            return ""
    try:
        from matplotlib.backends.qt_compat import QtWidgets

        if QtWidgets.QApplication.instance() is not None:
            return QtWidgets.QApplication.clipboard().text()
    except Exception:
        pass
    for cmd in (
        ["pbpaste"],
        ["wl-paste", "-n"],
        ["xclip", "-selection", "clipboard", "-o"],
        ["xsel", "-b", "-o"],
        ["powershell", "-noprofile", "-command", "Get-Clipboard"],
    ):
        r = _run(cmd)
        if r is not None:
            return r.stdout
    try:
        import pyperclip

        return pyperclip.paste()
    except Exception:
        pass
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        try:
            return root.clipboard_get()
        finally:
            root.destroy()
    except Exception:
        return ""


def clipboard_set(fig, text):
    canvas = fig.canvas
    if hasattr(canvas, "get_tk_widget"):
        try:
            w = canvas.get_tk_widget()
            w.clipboard_clear()
            w.clipboard_append(text)
            w.update_idletasks()
            return True
        except Exception:
            return False
    try:
        from matplotlib.backends.qt_compat import QtWidgets

        if QtWidgets.QApplication.instance() is not None:
            QtWidgets.QApplication.clipboard().setText(text)
            return True
    except Exception:
        pass
    for cmd in (
        ["pbcopy"],
        ["wl-copy"],
        ["xclip", "-selection", "clipboard"],
        ["xsel", "-b", "-i"],
        ["clip"],
    ):
        if _run(cmd, text) is not None:
            return True
    try:
        import pyperclip

        pyperclip.copy(text)
        return True
    except Exception:
        return False


def export_dialog(fig, default_name):
    """Diálogo de guardar como. Devuelve None si el usuario cancela."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        canvas = fig.canvas
        if hasattr(canvas, "get_tk_widget"):
            # Reutilizamos el root de TkAgg para no crear un Tk() extra
            parent = canvas.get_tk_widget().winfo_toplevel()
            owns_root = False
        else:
            parent = tk.Tk()
            parent.withdraw()
            owns_root = True
        try:
            path = filedialog.asksaveasfilename(
                parent=parent,
                title="Guardar como",
                defaultextension=".png",
                initialfile=default_name,
                filetypes=[
                    ("PNG", "*.png"),
                    ("SVG", "*.svg"),
                    ("PDF", "*.pdf"),
                    ("Todos", "*.*"),
                ],
            )
        finally:
            if owns_root:
                parent.destroy()
        return path or None
    except Exception as e:
        print("No se pudo abrir el diálogo (se guardará con el nombre por defecto):", e)
        return default_name


# ---------------------------------------------------------------------------
# Cuadro de texto con selección, copiar/cortar/pegar, borrar todo
# ---------------------------------------------------------------------------
class InputBox:
    PAD = 0.6

    def __init__(
        self,
        fig,
        rect,
        text="",
        max_len=MAX_LEN,
        on_submit=None,
        label="Cadena",
        fontsize=12,
    ):
        self.fig, self.max_len, self.on_submit = fig, max_len, on_submit
        self.text = text[:max_len]
        self.cur, self.anchor = len(self.text), None
        self.focused = self.dragging = self.warn = False
        self.caret_on = True
        self.scroll, self.cap, self.cw = 0.0, 30.0, 8.0
        self._keymaps, self._timer = None, None

        ax = self.ax = fig.add_axes(rect)
        ax.set_facecolor("#20242e")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylim(0, 1)
        tr = blended_transform_factory(ax.transData, ax.transAxes)
        self.sel_patch = Rectangle(
            (0, 0.14), 0, 0.72, transform=tr, fc="#3d5f9c", ec="none", zorder=1
        )
        ax.add_patch(self.sel_patch)
        self.t = ax.text(
            self.PAD,
            0.5,
            "",
            transform=tr,
            color=FG,
            family="monospace",
            fontsize=fontsize,
            ha="left",
            va="center",
            zorder=3,
            clip_on=True,
        )
        self.caret = Line2D(
            [0, 0], [0.16, 0.84], transform=tr, color=FG, lw=1.5, zorder=4
        )
        ax.add_line(self.caret)
        ax.text(
            -0.04,
            0.5,
            label,
            transform=ax.transAxes,
            ha="right",
            va="center",
            color=FG,
            fontsize=11,
        )

        cv = fig.canvas
        cv.mpl_connect("key_press_event", self._on_key)
        cv.mpl_connect("button_press_event", self._on_press)
        cv.mpl_connect("motion_notify_event", self._on_motion)
        cv.mpl_connect("button_release_event", self._on_release)
        cv.mpl_connect("close_event", lambda e: self.blur())
        self.relayout()

    # ---- API --------------------------------------------------------------
    def set_text(self, text):
        self.text = text[: self.max_len]
        self.cur, self.anchor = len(self.text), None
        self._redraw()

    def focus(self):
        if self.focused:
            return
        self.focused, self.caret_on = True, True
        # Desactivamos atajos globales de matplotlib (s = guardar, q = cerrar…)
        self._keymaps = {
            k: list(v)
            for k, v in matplotlib.rcParams.items()
            if k.startswith("keymap.")
        }
        for k in self._keymaps:
            matplotlib.rcParams[k] = []
        try:
            self._timer = self.fig.canvas.new_timer(interval=530)
            self._timer.add_callback(self._blink)
            self._timer.start()
        except Exception:
            self._timer = None
        self._redraw()

    def blur(self):
        if not self.focused:
            return
        self.focused = self.dragging = self.warn = False
        if self._keymaps:
            for k, v in self._keymaps.items():
                matplotlib.rcParams[k] = v
            self._keymaps = None
        if self._timer is not None:
            try:
                self._timer.stop()
            except Exception:
                pass
            self._timer = None
        self._redraw()

    def relayout(self):
        try:
            r = self.fig.canvas.get_renderer()
            w, _, _ = r.get_text_width_height_descent(
                "M" * 20, self.t.get_fontproperties(), ismath=False
            )
            self.cw = max(1.0, w / 20.0)
        except Exception:
            self.cw = self.t.get_fontsize() * 0.6 * self.fig.dpi / 72.0
        try:
            width = self.ax.get_window_extent().width
        except Exception:
            width = 200.0
        self.cap = max(4.0, width / self.cw)
        self._redraw()

    # ---- selección / edición ---------------------------------------------
    def _sel(self):
        if self.anchor is None or self.anchor == self.cur:
            return None
        return (min(self.anchor, self.cur), max(self.anchor, self.cur))

    def _insert(self, s):
        a, b = self._sel() or (self.cur, self.cur)
        room = self.max_len - (len(self.text) - (b - a))
        if len(s) > room:
            s = s[: max(room, 0)]
            self.warn = True
        self.text = self.text[:a] + s + self.text[b:]
        self.cur, self.anchor = a + len(s), None

    def _delete(self, direction, word=False):
        sel = self._sel()
        if sel:
            a, b = sel
        elif direction < 0:
            a, b = (
                self._word_left(self.cur) if word else max(0, self.cur - 1)
            ), self.cur
        else:
            a, b = self.cur, (
                self._word_right(self.cur)
                if word
                else min(len(self.text), self.cur + 1)
            )
        self.text = self.text[:a] + self.text[b:]
        self.cur, self.anchor = a, None

    def _word_left(self, i):
        while i > 0 and self.text[i - 1] == " ":
            i -= 1
        while i > 0 and self.text[i - 1] != " ":
            i -= 1
        return i

    def _word_right(self, i):
        n = len(self.text)
        while i < n and self.text[i] == " ":
            i += 1
        while i < n and self.text[i] != " ":
            i += 1
        return i

    def _move(self, new, shift):
        if shift:
            if self.anchor is None:
                self.anchor = self.cur
        else:
            self.anchor = None
        self.cur = max(0, min(len(self.text), new))

    def _copy(self):
        sel = self._sel()
        if sel:
            clipboard_set(self.fig, self.text[sel[0] : sel[1]])
            return True
        return False

    # ---- eventos ----------------------------------------------------------
    def _on_key(self, e):
        if not self.focused or not e.key:
            return
        key = e.key
        if key == "+":
            mods, base = set(), "+"
        elif key.endswith("++"):
            mods, base = set(key[:-2].split("+")), "+"
        else:
            *m, base = key.split("+")
            mods = set(m)
        cmd = bool(mods & {"ctrl", "cmd", "super", "meta"})
        shift = "shift" in mods
        self.caret_on, self.warn = True, False

        if cmd:
            if base == "a":
                self.anchor, self.cur = 0, len(self.text)
            elif base == "c":
                self._copy()
            elif base == "x":
                if self._copy():
                    self._delete(-1)
            elif base == "v":
                clip = re.sub(r"[\r\n]+", "", clipboard_get(self.fig) or "").replace(
                    "\t", " "
                )
                if clip:
                    self._insert(clip)
            elif base == "left":
                self._move(self._word_left(self.cur), shift)
            elif base == "right":
                self._move(self._word_right(self.cur), shift)
            elif base == "backspace":
                self._delete(-1, word=True)
            elif base == "delete":
                self._delete(+1, word=True)
        elif base == "left":
            sel = self._sel()
            self._move(sel[0] if (sel and not shift) else self.cur - 1, shift)
        elif base == "right":
            sel = self._sel()
            self._move(sel[1] if (sel and not shift) else self.cur + 1, shift)
        elif base == "home":
            self._move(0, shift)
        elif base == "end":
            self._move(len(self.text), shift)
        elif base == "backspace":
            self._delete(-1)
        elif base == "delete":
            self._delete(+1)
        elif base in ("enter", "return"):
            if self.on_submit:
                self.on_submit(self.text)
        elif base == "escape":
            self.blur()
            return
        elif base == "space" or (len(base) == 1 and base.isprintable()):
            self._insert(" " if base == "space" else base)
        self._redraw()

    def _idx(self, e):
        x = self.ax.transData.inverted().transform((e.x, e.y))[0]
        return max(0, min(len(self.text), int(round(x - self.PAD))))

    def _on_press(self, e):
        if e.inaxes is self.ax and e.button == 1:
            self.focus()
            if getattr(e, "dblclick", False):
                self.anchor, self.cur = 0, len(self.text)
                self.dragging = False
            else:
                self.cur = self.anchor = self._idx(e)
                self.dragging = True
            self.caret_on = True
            self._redraw()
        elif self.focused:
            self.blur()

    def _on_motion(self, e):
        if self.dragging and e.x is not None:
            self.cur = self._idx(e)
            self._redraw()

    def _on_release(self, e):
        if self.dragging:
            self.dragging = False
            if self.anchor == self.cur:
                self.anchor = None
            self._redraw()

    def _blink(self):
        self.caret_on = not self.caret_on
        self._redraw()

    # ---- dibujo -----------------------------------------------------------
    def _redraw(self):
        pos = self.PAD + self.cur
        if pos < self.scroll:
            self.scroll = max(0.0, pos - 1)
        elif pos > self.scroll + self.cap - 1:
            self.scroll = pos - self.cap + 1
        if len(self.text) + 2 * self.PAD <= self.cap:
            self.scroll = 0.0
        self.ax.set_xlim(self.scroll, self.scroll + self.cap)
        self.t.set_text(self.text)
        sel = self._sel()
        self.sel_patch.set_visible(bool(sel))
        if sel:
            self.sel_patch.set_x(self.PAD + sel[0])
            self.sel_patch.set_width(sel[1] - sel[0])
        self.caret.set_xdata([pos, pos])
        self.caret.set_visible(self.focused and self.caret_on)
        color = "#f9ab00" if self.warn else (LAST_FILL if self.focused else "#2a2f3a")
        for sp in self.ax.spines.values():
            sp.set_color(color)
            sp.set_linewidth(1.8 if self.focused else 1.0)
        self.fig.canvas.draw_idle()


# ---------------------------------------------------------------------------
# Aplicación
# ---------------------------------------------------------------------------
class App:
    def __init__(
        self, text, interactive=True, step=None, strings=False, links=False, trans=True
    ):
        self.interactive = interactive
        self.strings_mode, self.show_links, self.show_trans = strings, links, trans
        self._busy = False
        self._timer = None

        self.fig = plt.figure(figsize=(19, 11), facecolor=BG)
        bottom = 0.11 if interactive else 0.03
        gs = self.fig.add_gridspec(
            2,
            2,
            height_ratios=[1.0, 1.0],
            width_ratios=[1.5, 1],
            left=0.012,
            right=0.988,
            top=0.815,
            bottom=bottom,
            hspace=0.13,
            wspace=0.03,
        )
        self.ax_g = self.fig.add_subplot(gs[0, 0])
        self.ax_t = self.fig.add_subplot(gs[0, 1])
        self.ax_b = self.fig.add_subplot(gs[1, :])

        self.t_title = self.fig.text(
            0.015, 0.985, "", color=FG, fontsize=17, fontweight="bold", va="top"
        )
        self.t_info = self.fig.text(
            0.015,
            0.947,
            "",
            color=LAST_FILL,
            fontsize=11.5,
            va="top",
            family="monospace",
        )
        self.t_log = self.fig.text(
            0.015, 0.922, "", color=MUTED, fontsize=9.5, va="top", linespacing=1.45
        )

        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                ls="none",
                mfc=LAST_FILL,
                mec=LAST_FILL,
                ms=9,
                label="último estado",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                ls="none",
                mfc=CLONE_FILL,
                mec=CLONE_FILL,
                ms=9,
                label="clon",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                ls="none",
                mfc="none",
                mec=TERM,
                mew=2,
                ms=10,
                label="terminal",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                ls="none",
                mfc=NODE_FILL,
                mec=TOUCH,
                mew=2,
                ms=9,
                label="afectado en el paso",
            ),
            Line2D([0], [0], color=TRANS, lw=2, label="transición"),
            Line2D([0], [0], color=LINK, lw=2, label="suffix link"),
        ]
        leg = self.fig.legend(
            handles=handles,
            loc="upper right",
            ncol=6,
            frameon=False,
            bbox_to_anchor=(0.985, 0.99),
            fontsize=10,
        )
        for t in leg.get_texts():
            t.set_color(FG)

        self.set_text(text, step)
        if interactive:
            self._build_widgets(text)
        self.fig.canvas.mpl_connect("resize_event", lambda e: self.render())
        self.render()

    # ---- datos ------------------------------------------------------------
    def set_text(self, text, step=None):
        self.text = text
        self.hist = build_history(text)
        # step None o negativo -> último paso
        if step is None or step < 0:
            self.step = len(text)
        else:
            self.step = min(step, len(text))

    # ---- widgets ----------------------------------------------------------
    def _build_widgets(self, text):
        f = self.fig

        def styled_ax(rect):
            a = f.add_axes(rect)
            a.set_facecolor("#20242e")
            return a

        self.inp = InputBox(
            f, [0.075, 0.025, 0.15, 0.045], text, on_submit=lambda _t: self._rebuild()
        )
        self.b_clear = Button(
            styled_ax([0.23, 0.025, 0.027, 0.045]),
            "✕",
            color="#20242e",
            hovercolor="#5a2a2a",
        )
        self.b_clear.label.set_color(FG)
        self.b_clear.on_clicked(lambda _e: (self.inp.set_text(""), self.inp.focus()))

        self.b_build = Button(
            styled_ax([0.262, 0.025, 0.068, 0.045]),
            "Construir",
            color="#2f4a7a",
            hovercolor="#3d5f9c",
        )
        f.text(
            0.075,
            0.008,
            "Ctrl+A / doble click: seleccionar todo · Ctrl+C / X / V: copiar / cortar / pegar · Enter: construir",
            color=MUTED,
            fontsize=8,
            va="bottom",
        )

        self.b_prev = Button(
            styled_ax([0.345, 0.025, 0.035, 0.045]),
            "◀",
            color="#20242e",
            hovercolor="#2a3040",
        )
        n = len(text)
        ax_sl = styled_ax([0.395, 0.033, 0.18, 0.03])
        try:
            self.sl = Slider(
                ax_sl,
                "",
                0,
                max(1, n),
                valinit=self.step,
                valstep=1,
                color=LAST_FILL,
                initcolor="none",
            )
        except TypeError:
            self.sl = Slider(
                ax_sl, "", 0, max(1, n), valinit=self.step, valstep=1, color=LAST_FILL
            )
        self.sl.valtext.set_color(FG)

        self.b_next = Button(
            styled_ax([0.585, 0.025, 0.035, 0.045]),
            "▶",
            color="#20242e",
            hovercolor="#2a3040",
        )
        self.b_play = Button(
            styled_ax([0.625, 0.025, 0.055, 0.045]),
            "Play",
            color="#20242e",
            hovercolor="#2a3040",
        )
        self.b_export = Button(
            styled_ax([0.69, 0.025, 0.065, 0.045]),
            "Exportar",
            color="#20242e",
            hovercolor="#2a3040",
        )
        for b in (self.b_build, self.b_prev, self.b_next, self.b_play, self.b_export):
            b.label.set_color(FG)

        self.b_build.on_clicked(lambda _e: self._rebuild())
        self.b_prev.on_clicked(lambda _e: self._goto(self.step - 1))
        self.b_next.on_clicked(lambda _e: self._goto(self.step + 1))
        self.b_play.on_clicked(lambda _e: self._toggle_play())
        self.b_export.on_clicked(lambda _e: self._export())
        self.sl.on_changed(lambda v: self._goto(int(v), from_slider=True))

        ax_ck = styled_ax([0.775, 0.008, 0.215, 0.08])
        ax_ck.set_facecolor(BG)
        labels = [
            "Strings en los nodos",
            "Suffix links sobre el grafo",
            "Transiciones en las cajas",
        ]
        states = [self.strings_mode, self.show_links, self.show_trans]
        try:
            self.ck = CheckButtons(
                ax_ck,
                labels,
                states,
                frame_props=dict(edgecolor=FG, facecolor=BG, sizes=[120] * 3),
                check_props=dict(color=TERM, linewidth=2.5),
            )
        except TypeError:
            self.ck = CheckButtons(ax_ck, labels, states)
        for lab in self.ck.labels:
            lab.set_color(FG)
            lab.set_fontsize(10)
        self.ck.on_clicked(self._on_check)

    def _on_check(self, label):
        if label.startswith("Strings"):
            self.strings_mode = not self.strings_mode
        elif label.startswith("Suffix"):
            self.show_links = not self.show_links
        else:
            self.show_trans = not self.show_trans
        self.render()

    def _rebuild(self):
        t = self.inp.text[:MAX_LEN]
        if not t.strip():
            self.t_log.set_text("(la cadena está vacía: escribe algo y pulsa Enter)")
            self.fig.canvas.draw_idle()
            return
        self._stop_play()
        self.set_text(t)
        self._busy = True
        self.sl.valmax = max(1, len(t))
        self.sl.ax.set_xlim(self.sl.valmin, self.sl.valmax)
        self.sl.set_val(self.step)
        self._busy = False
        self.render()

    def _goto(self, k, from_slider=False):
        if self._busy:
            return
        k = max(0, min(k, len(self.text)))
        self.step = k
        if not from_slider and self.interactive:
            self._busy = True
            self.sl.set_val(k)
            self._busy = False
        self.render()

    def _toggle_play(self):
        if self._timer is not None:
            self._stop_play()
            return
        if self.step >= len(self.text):
            self.step = 0
        self._timer = self.fig.canvas.new_timer(interval=900)
        self._timer.add_callback(self._tick)
        self._timer.start()
        self.b_play.label.set_text("Pausa")

    def _tick(self):
        if self.step >= len(self.text):
            self._stop_play()
            return
        self._goto(self.step + 1)

    def _stop_play(self):
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
            if self.interactive:
                self.b_play.label.set_text("Play")

    def _export(self):
        default = f"sam_{self.text or 'vacio'}_paso{self.step}.png"
        path = export_dialog(self.fig, default)
        if not path:
            return
        try:
            self.fig.savefig(path, dpi=140, facecolor=BG)
            print("Guardado:", path)
        except Exception as e:
            print("Error al guardar:", e)

    # ---- render -----------------------------------------------------------
    def render(self):
        s = self.hist[self.step]
        self.t_title.set_text(
            f"Visualizador de Autómata de Sufijos (SAM) — «{vis(self.text)}»"
        )
        self.t_info.set_text(
            f"Paso {self.step}/{len(self.text)}   prefijo: «{vis(s.prefix)}»   "
            f"ESTADO ACTUAL: S{s.last}   LONGITUD: {s.ln[s.last]}   estados: {s.n}"
        )
        self.t_log.set_text("\n".join(s.log))

        # sizes calculados una sola vez por render
        sizes = {v: node_size(s, self.text, v, self.strings_mode) for v in range(s.n)}
        draw_graph_panel(
            self.ax_g, s, self.text, sizes, self.strings_mode, self.show_links
        )
        draw_link_panel(self.ax_t, s, self.text, sizes, self.strings_mode)
        draw_boxes_panel(self.ax_b, s, self.text, self.show_trans)

        if self.interactive:
            self.inp.relayout()
        self.fig.canvas.draw_idle()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Visualizador de Autómata de Sufijos (SAM)"
    )
    ap.add_argument(
        "texto",
        nargs="?",
        default="banana",
        help="cadena a procesar (por defecto: banana)",
    )
    ap.add_argument("--png", help="guardar PNG y salir")
    ap.add_argument("--svg", help="guardar SVG y salir")
    ap.add_argument("--pdf", help="guardar PDF y salir")
    ap.add_argument(
        "--paso",
        type=int,
        default=None,
        help="paso a mostrar (por defecto: el final; usa -1 para el último)",
    )
    ap.add_argument(
        "--strings", action="store_true", help="mostrar las cadenas dentro de los nodos"
    )
    ap.add_argument(
        "--links",
        action="store_true",
        help="dibujar suffix links sobre el grafo de transiciones",
    )
    ap.add_argument(
        "--sin-trans-cajas",
        action="store_true",
        help="ocultar transiciones en la vista de cajas",
    )
    a = ap.parse_args()

    text = a.texto[:MAX_LEN]
    interactive = not any([a.png, a.svg, a.pdf])
    app = App(
        text,
        interactive=interactive,
        step=a.paso,
        strings=a.strings,
        links=a.links,
        trans=not a.sin_trans_cajas,
    )

    for path in (a.png, a.svg, a.pdf):
        if path:
            app.render()
            app.fig.savefig(path, dpi=140, facecolor=BG)
            print("Guardado:", path)
            return

    plt.show()


if __name__ == "__main__":
    main()
