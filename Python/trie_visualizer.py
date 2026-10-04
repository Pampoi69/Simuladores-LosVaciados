#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualizador de Trie / Árbol de Prefijos  —  solo requiere matplotlib
=====================================================================

Modelo de datos calcado del struct C++:

    struct Trie {
        vector<vi> trie;         // trie[u][c] = hijo
        vi word_count;           // palabras que PASAN por el nodo (count_prefix)
        vi word_end_count;       // palabras que TERMINAN en el nodo (count_word)
        int next_node;           // 0 = raíz
    };

Cada nodo muestra AMBOS contadores. Se añade un cuadro de «Consulta» que simula
en vivo las cuatro operaciones (search / count_word / count_prefix / starts_with)
y resalta el camino consultado en el árbol.

Uso
---
    python trie_visualizer.py                              # ventana, "casa,caso,carro,cara"
    python trie_visualizer.py casa caso carro cara          # palabras sueltas
    python trie_visualizer.py "casa,caso" --png trie.png    # guarda PNG
    python trie_visualizer.py "casa,caso" --svg trie.svg    # guarda SVG
    python trie_visualizer.py "casa,caso" --pdf trie.pdf    # guarda PDF
    python trie_visualizer.py "casa,caso" --paso 5 --strings
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
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
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
CUR_FILL = "#a8c7fa"        # nodo actual
TERM = "#34d399"            # word_end_count > 0 (fin de palabra)
EDGE_C = "#c3c9d3"          # aristas normales
PATH_C = "#ff9d3c"          # camino en curso
QUERY_C = "#22d3ee"         # camino de la consulta
TOUCH = "#e879f9"           # palabra en curso (panel derecho)
P_BADGE = "#94a3b8"         # badge word_count
DARK_TXT = "#0b1220"

MAX_WORD_LEN = 16
MAX_WORDS = 10

FU = 0.22
CW = 0.6 * FU
LH = 1.5 * FU
R = 0.34

LOG_MAX_LINES = 5           # el log no debe invadir los paneles


def vis(s):
    return "ε" if s == "" else s.replace(" ", "␣")


# ---------------------------------------------------------------------------
# Historial del Trie con word_count y word_end_count
# ---------------------------------------------------------------------------
@dataclass
class Snap:
    parent: list
    char_in: list
    children: list
    word_count: list
    word_end_count: list
    cur: int
    words: list
    completed: set
    label: str
    log: list = field(default_factory=list)
    new: set = field(default_factory=set)
    path: set = field(default_factory=set)
    word_idx: int = -1
    char_idx: int = -1

    @property
    def n(self):
        return len(self.parent)


def parse_words(raw, max_word=MAX_WORD_LEN, max_words=MAX_WORDS):
    raw = raw.strip()
    if not raw:
        return []
    parts = raw.split(",") if "," in raw else raw.split()
    out = []
    for p in parts:
        w = p.strip()
        if not w:
            continue
        out.append(w[:max_word])
        if len(out) >= max_words:
            break
    return out


def node_prefix(parent, char_in, v):
    chars = []
    while v > 0:
        chars.append(char_in[v])
        v = parent[v]
    return "".join(reversed(chars))


def build_history(words):
    parent, char_in, depth = [-1], [""], [0]
    children = [{}]
    word_count = [0]
    word_end_count = [0]

    def snap(cur, wds, completed, label, log, new, path, word_idx=-1, char_idx=-1):
        return Snap(list(parent), list(char_in),
                    [dict(d) for d in children],
                    list(word_count), list(word_end_count),
                    cur, list(wds), set(completed), label, log,
                    set(new), set(path), word_idx, char_idx)

    hist = [snap(0, words, set(),
                 "Trie vacío: sólo la raíz S0 (prefijo ε).",
                 ["Se crea la raíz S0 (prefijo vacío ε).",
                  "word_count[S0] = 0  (la raíz nunca se incrementa).",
                  "word_end_count[S0] = 0."],
                 set(), set())]

    completed = set()
    for wi, w in enumerate(words):
        p = 0
        path = {0}
        for ci, c in enumerate(w):
            new, log = set(), []
            if c in children[p]:
                v = children[p][c]
                log.append(f"«{vis(c)}» ya existe desde S{p} → se reutiliza S{v} "
                           f"(prefijo «{vis(w[:ci+1])}»).")
            else:
                v = len(parent)
                parent.append(p); char_in.append(c); depth.append(depth[p] + 1)
                children.append({}); word_count.append(0); word_end_count.append(0)
                children[p][c] = v
                new.add(v)
                log.append(f"No existía «{vis(c)}» desde S{p} → se crea S{v} "
                           f"(prefijo «{vis(w[:ci+1])}»).")
            p = v
            path.add(p)

            word_count[p] += 1
            log.append(f"word_count[S{p}] ← {word_count[p]}  "
                       f"(«{vis(w[:ci+1])}» es prefijo de {word_count[p]} palabra(s)).")

            if ci == len(w) - 1:
                word_end_count[p] += 1
                if word_end_count[p] == 1:
                    log.append(f"Fin de «{vis(w)}»: S{p} es TERMINAL  "
                               f"(word_end_count = 1).")
                else:
                    log.append(f"Fin de «{vis(w)}» (otra vez):  "
                               f"word_end_count[S{p}] = {word_end_count[p]}.")
                completed.add(wi)

            label = (f"Palabra {wi+1}/{len(words)} «{vis(w)}» — "
                     f"letra {ci+1}/{len(w)} «{vis(c)}»")
            hist.append(snap(p, words, completed, label, log, new, path, wi, ci))
    return hist


def query_trie(s, text):
    u = 0
    path = [0]
    for c in text:
        if c not in s.children[u]:
            return path, -1, False, None
        u = s.children[u][c]
        path.append(u)
    results = {
        'search':        s.word_end_count[u] > 0,
        'count_word':    s.word_end_count[u],
        'count_prefix':  s.word_count[u],
        'starts_with':   True,
    }
    return path, u, True, results


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
def node_size(s, v, strings_mode):
    if not strings_mode:
        return (2 * R, 2 * R)
    pre = node_prefix(s.parent, s.char_in, v)
    label = vis(pre) if v else "ε"
    info = f"S{v} · P:{s.word_count[v]} · E:{s.word_end_count[v]}"
    w = max(len(info), len(label)) * CW + 0.36
    h = 2 * LH + 0.22
    return (w, h)


def layout_tree(s, sizes):
    n = s.n
    children_order = [[] for _ in range(n)]
    for v in range(1, n):
        children_order[s.parent[v]].append(v)

    xgap = max(w for w, _ in sizes.values()) + 0.5
    ygap = max(h for _, h in sizes.values()) + 0.95
    xs, counter = {}, [0.0]

    def rec(v):
        kids = children_order[v]
        if not kids:
            xs[v] = counter[0]
            counter[0] += xgap
        else:
            for c in kids:
                rec(c)
            xs[v] = (xs[kids[0]] + xs[kids[-1]]) / 2

    sys.setrecursionlimit(10000)
    rec(0)

    depth = [0] * n
    for v in range(1, n):
        depth[v] = depth[s.parent[v]] + 1
    return {v: (xs[v], -depth[v] * ygap) for v in range(n)}


# ---------------------------------------------------------------------------
# Dibujo
# ---------------------------------------------------------------------------
def setup_axes(ax, xmin, xmax, ymin, ymax, title):
    ax.clear()
    ax.set_facecolor(PANEL)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#2a2f3a")

    pos = ax.get_position()
    fw, fh = ax.figure.get_size_inches()
    pw = max(pos.width * fw, 1e-3)
    ph = max(pos.height * fh, 1e-3)
    aspect = pw / ph

    xr, yr = max(xmax - xmin, 1e-6), max(ymax - ymin, 1e-6)
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

    ppu = pw / (xmax - xmin) * 72.0
    return ppu


def fsz(units, ppu):
    return max(3.0, min(24.0, units * ppu))


def bounds(pos, sizes, margin=0.7):
    xmin = min(pos[v][0] - sizes[v][0] / 2 for v in pos) - margin
    xmax = max(pos[v][0] + sizes[v][0] / 2 for v in pos) + margin
    ymin = min(pos[v][1] - sizes[v][1] / 2 for v in pos) - margin
    ymax = max(pos[v][1] + sizes[v][1] / 2 for v in pos) + margin
    return xmin, xmax, ymin, ymax


def arrow(ax, p1, p2, pa=None, pb=None, color=FG, lw=1.6, z=2, ms=13, alpha=1.0):
    a = FancyArrowPatch(p1, p2, patchA=pa, patchB=pb, arrowstyle="-|>",
                        mutation_scale=ms, color=color, lw=lw, zorder=z,
                        shrinkA=0, shrinkB=1.5, alpha=alpha)
    ax.add_patch(a)
    return a


def draw_badge(ax, x, y, text, bg, ppu, size=0.15):
    ax.text(x, y, text, color=DARK_TXT, fontsize=fsz(size, ppu),
            family="monospace", fontweight="bold",
            ha="center", va="center", zorder=8,
            bbox=dict(boxstyle="circle,pad=0.20", fc=bg, ec="none"))


def draw_tree_panel(ax, s, strings_mode, highlight, query_path=None, query_node=-1):
    query_path = query_path or set()
    sizes = {v: node_size(s, v, strings_mode) for v in range(s.n)}
    pos = layout_tree(s, sizes)
    ppu = setup_axes(ax, *bounds(pos, sizes), "Trie (árbol de prefijos)")

    patches = {}
    for v in range(s.n):
        x, y = pos[v]
        w, h = sizes[v]
        is_cur = v == s.cur
        is_new = v in s.new
        on_path = highlight and v in s.path
        on_query = v in query_path and v != 0

        fc = CUR_FILL if is_cur else NODE_FILL
        if is_new:
            ec, lw = "#ffffff", 2.6
        elif is_cur:
            ec, lw = CUR_FILL, 2.2
        elif on_query:
            ec, lw = QUERY_C, 2.4
        elif on_path:
            ec, lw = PATH_C, 2.0
        else:
            ec, lw = NODE_EDGE, 1.5
        tc = DARK_TXT if is_cur else FG

        if strings_mode:
            patch = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                   boxstyle="round,pad=0,rounding_size=0.09",
                                   fc=fc, ec=ec, lw=lw, zorder=3)
        else:
            patch = Circle((x, y), R, fc=fc, ec=ec, lw=lw, zorder=3)
        ax.add_patch(patch)
        patches[v] = patch

        if s.word_end_count[v] > 0:
            if strings_mode:
                ring = FancyBboxPatch((x - w / 2 - 0.06, y - h / 2 - 0.06),
                                      w + 0.12, h + 0.12,
                                      boxstyle="round,pad=0,rounding_size=0.13",
                                      fc="none", ec=TERM, lw=1.8, zorder=3)
            else:
                ring = Circle((x, y), R + 0.075, fc="none", ec=TERM, lw=1.8, zorder=3)
            ax.add_patch(ring)

        if strings_mode:
            pre = node_prefix(s.parent, s.char_in, v)
            ax.text(x, y + h / 2 - LH / 2 - 0.05,
                    f"S{v} · P:{s.word_count[v]} · E:{s.word_end_count[v]}",
                    color=MUTED if not is_cur else "#3b4557",
                    fontsize=fsz(FU * 0.9, ppu), family="monospace",
                    fontweight="bold", ha="center", va="center", zorder=4)
            ax.text(x, y - h / 2 + LH / 2 + 0.05, vis(pre) if v else "ε",
                    color=tc, fontsize=fsz(FU, ppu), family="monospace",
                    ha="center", va="center", zorder=4)
        else:
            label = "ε" if v == 0 else vis(s.char_in[v])
            ax.text(x, y, label, color=tc, fontsize=fsz(0.27, ppu),
                    family="monospace", fontweight="bold",
                    ha="center", va="center", zorder=4)

            if s.word_count[v] > 0:
                draw_badge(ax, x - R - 0.13, y + R + 0.13,
                           str(s.word_count[v]), P_BADGE, ppu)
            if s.word_end_count[v] > 0:
                draw_badge(ax, x + R + 0.13, y + R + 0.13,
                           str(s.word_end_count[v]), TERM, ppu)

    cur_edge = (s.parent[s.cur], s.cur) if s.cur != 0 else None
    for v in range(1, s.n):
        u = s.parent[v]
        is_current = (cur_edge is not None) and ((u, v) == cur_edge)
        on_path = highlight and (v in s.path) and (u in s.path)
        on_query = (u in query_path) and (v in query_path)

        if is_current:
            color, lw, z, alpha = PATH_C, 3.2, 2.5, 1.0
        elif on_query:
            color, lw, z, alpha = QUERY_C, 2.6, 2.4, 1.0
        elif on_path:
            color, lw, z, alpha = PATH_C, 2.0, 2.0, 0.85
        else:
            color, lw, z, alpha = EDGE_C, 1.4, 1.0, 1.0

        arrow(ax, pos[u], pos[v], patches[u], patches[v], color,
              lw=lw, z=z, ms=14 if (is_current or on_query) else 13, alpha=alpha)

        x1, y1 = pos[u]
        x2, y2 = pos[v]
        dx, dy = x2 - x1, y2 - y1
        L = max((dx * dx + dy * dy) ** 0.5, 1e-6)
        px, py = -dy / L, dx / L
        off = 0.20
        mx = (x1 + x2) / 2 + px * off
        my = (y1 + y2) / 2 + py * off
        ax.text(mx, my, vis(s.char_in[v]),
                color=QUERY_C if on_query else (PATH_C if (is_current or on_path) else FG),
                fontsize=fsz(0.26 if (is_current or on_query) else 0.24, ppu),
                family="monospace", fontweight="bold",
                ha="center", va="center", zorder=6,
                bbox=dict(boxstyle="round,pad=0.12", fc=PANEL, ec="none", alpha=0.9))
    return pos


def draw_words_panel(ax, s):
    ax.clear()
    ax.set_facecolor(PANEL)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#2a2f3a")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Palabras a insertar", color=FG, loc="left", fontsize=12, pad=6)

    pos = ax.get_position()
    fw = ax.figure.get_size_inches()[0]
    pw = max(pos.width * fw, 1e-3)
    ppu = pw * 72.0

    n = max(len(s.words), 1)
    row_h = min(0.12, 0.92 / n)
    top = 0.94
    for i, w in enumerate(s.words):
        y = top - i * row_h
        done = i in s.completed
        is_current = (i == s.word_idx) and not done
        if done:
            mark, color, weight, text_color = "✓", TERM, "bold", FG
        elif is_current:
            mark, color, weight, text_color = "▶", PATH_C, "bold", PATH_C
        else:
            mark, color, weight, text_color = "·", MUTED, "normal", MUTED
        ax.text(0.05, y, mark, color=color, fontsize=fsz(0.24, ppu),
                family="monospace", fontweight="bold", ha="left", va="center")
        ax.text(0.20, y, vis(w) if w else "ε", color=text_color,
                fontsize=fsz(0.20, ppu), family="monospace",
                fontweight=weight, ha="left", va="center")
    ax.text(0.05, max(top - n * row_h - 0.06, 0.03),
            f"{len(s.completed)}/{len(s.words)} completas",
            color=MUTED, fontsize=fsz(0.15, ppu),
            family="monospace", ha="left", va="bottom")


# ---------------------------------------------------------------------------
# Portapapeles
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
    for cmd in (["pbpaste"], ["wl-paste", "-n"], ["xclip", "-selection", "clipboard", "-o"],
                ["xsel", "-b", "-o"], ["powershell", "-noprofile", "-command", "Get-Clipboard"]):
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
        root = tk.Tk(); root.withdraw()
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
            w.clipboard_clear(); w.clipboard_append(text)
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
    for cmd in (["pbcopy"], ["wl-copy"], ["xclip", "-selection", "clipboard"],
                ["xsel", "-b", "-i"], ["clip"]):
        if _run(cmd, text) is not None:
            return True
    try:
        import pyperclip
        pyperclip.copy(text)
        return True
    except Exception:
        return False


def export_dialog(fig, default_name):
    try:
        import tkinter as tk
        from tkinter import filedialog
        canvas = fig.canvas
        if hasattr(canvas, "get_tk_widget"):
            parent = canvas.get_tk_widget().winfo_toplevel()
            owns = False
        else:
            parent = tk.Tk(); parent.withdraw(); owns = True
        try:
            path = filedialog.asksaveasfilename(
                parent=parent, title="Guardar como",
                defaultextension=".png", initialfile=default_name,
                filetypes=[("PNG", "*.png"), ("SVG", "*.svg"),
                           ("PDF", "*.pdf"), ("Todos", "*.*")])
        finally:
            if owns:
                parent.destroy()
        return path or None
    except Exception as e:
        print("No se pudo abrir el diálogo:", e)
        return default_name


# ---------------------------------------------------------------------------
# Cuadro de texto
# ---------------------------------------------------------------------------
class InputBox:
    PAD = 0.6

    def __init__(self, fig, rect, text="", max_len=200, on_submit=None,
                 on_change=None, label="Texto", fontsize=12):
        self.fig, self.max_len = fig, max_len
        self.on_submit, self.on_change = on_submit, on_change
        self.text = text[:max_len]
        self._last_notified = self.text
        self.cur, self.anchor = len(self.text), None
        self.focused = self.dragging = self.warn = False
        self.caret_on = True
        self.scroll, self.cap, self.cw = 0.0, 30.0, 8.0
        self._keymaps, self._timer = None, None

        ax = self.ax = fig.add_axes(rect)
        ax.set_facecolor("#20242e")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_ylim(0, 1)
        tr = blended_transform_factory(ax.transData, ax.transAxes)
        self.sel_patch = Rectangle((0, 0.14), 0, 0.72, transform=tr,
                                   fc="#3d5f9c", ec="none", zorder=1)
        ax.add_patch(self.sel_patch)
        self.t = ax.text(self.PAD, 0.5, "", transform=tr, color=FG,
                         family="monospace", fontsize=fontsize,
                         ha="left", va="center", zorder=3, clip_on=True)
        self.caret = Line2D([0, 0], [0.16, 0.84], transform=tr,
                            color=FG, lw=1.5, zorder=4)
        ax.add_line(self.caret)
        ax.text(-0.04, 0.5, label, transform=ax.transAxes, ha="right",
                va="center", color=FG, fontsize=11)

        cv = fig.canvas
        cv.mpl_connect("key_press_event", self._on_key)
        cv.mpl_connect("button_press_event", self._on_press)
        cv.mpl_connect("motion_notify_event", self._on_motion)
        cv.mpl_connect("button_release_event", self._on_release)
        cv.mpl_connect("close_event", lambda e: self.blur())
        self.relayout()

    def set_text(self, text):
        self.text = text[: self.max_len]
        self.cur, self.anchor = len(self.text), None
        self._redraw()
        self._notify()

    def focus(self):
        if self.focused:
            return
        self.focused, self.caret_on = True, True
        self._keymaps = {k: list(v) for k, v in matplotlib.rcParams.items()
                         if k.startswith("keymap.")}
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
                "M" * 20, self.t.get_fontproperties(), ismath=False)
            self.cw = max(1.0, w / 20.0)
        except Exception:
            self.cw = self.t.get_fontsize() * 0.6 * self.fig.dpi / 72.0
        try:
            width = self.ax.get_window_extent().width
        except Exception:
            width = 200.0
        self.cap = max(4.0, width / self.cw)
        self._redraw()

    def _notify(self):
        if self.on_change and self.text != self._last_notified:
            self._last_notified = self.text
            try:
                self.on_change(self.text)
            except Exception:
                pass

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
            a, b = (self._word_left(self.cur) if word else max(0, self.cur - 1)), self.cur
        else:
            a, b = self.cur, (self._word_right(self.cur) if word else min(len(self.text), self.cur + 1))
        self.text = self.text[:a] + self.text[b:]
        self.cur, self.anchor = a, None

    def _word_left(self, i):
        while i > 0 and self.text[i - 1] in " ,":
            i -= 1
        while i > 0 and self.text[i - 1] not in " ,":
            i -= 1
        return i

    def _word_right(self, i):
        n = len(self.text)
        while i < n and self.text[i] in " ,":
            i += 1
        while i < n and self.text[i] not in " ,":
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
            clipboard_set(self.fig, self.text[sel[0]:sel[1]])
            return True
        return False

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
        changed = False

        if cmd:
            if base == "a":
                self.anchor, self.cur = 0, len(self.text)
            elif base == "c":
                self._copy()
            elif base == "x":
                if self._copy():
                    self._delete(-1); changed = True
            elif base == "v":
                clip = re.sub(r"[\r\n]+", ",", clipboard_get(self.fig) or "").replace("\t", " ")
                if clip:
                    self._insert(clip); changed = True
            elif base == "left":
                self._move(self._word_left(self.cur), shift)
            elif base == "right":
                self._move(self._word_right(self.cur), shift)
            elif base == "backspace":
                self._delete(-1, word=True); changed = True
            elif base == "delete":
                self._delete(+1, word=True); changed = True
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
            self._delete(-1); changed = True
        elif base == "delete":
            self._delete(+1); changed = True
        elif base in ("enter", "return"):
            if self.on_submit:
                self.on_submit(self.text)
        elif base == "escape":
            self.blur()
            return
        elif base in (",", "space") or (len(base) == 1 and base.isprintable()):
            self._insert("," if base == "," else (" " if base == "space" else base))
            changed = True
        self._redraw()
        if changed:
            self._notify()

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
        color = "#f9ab00" if self.warn else (CUR_FILL if self.focused else "#2a2f3a")
        for sp in self.ax.spines.values():
            sp.set_color(color)
            sp.set_linewidth(1.8 if self.focused else 1.0)
        self.fig.canvas.draw_idle()


# ---------------------------------------------------------------------------
# Aplicación
# ---------------------------------------------------------------------------
class App:
    def __init__(self, raw_words, interactive=True, step=None,
                 strings=False, highlight=True):
        self.interactive = interactive
        self.strings_mode, self.highlight = strings, highlight
        self._busy = False
        self._timer = None
        self.query_text = ""

        self.fig = plt.figure(figsize=(19, 11), facecolor=BG)
        bottom = 0.185 if interactive else 0.03
        gs = self.fig.add_gridspec(1, 2, width_ratios=[3.1, 1],
                                   left=0.015, right=0.985, top=0.80,
                                   bottom=bottom, wspace=0.03)
        self.ax_t = self.fig.add_subplot(gs[0, 0])
        self.ax_w = self.fig.add_subplot(gs[0, 1])

        # --- Cabecera: cada elemento en su propia fila -------------------
        self.t_title = self.fig.text(0.015, 0.985, "", color=FG, fontsize=14,
                                     fontweight="bold", va="top")
        # Fila del legend (se crea abajo)
        self.t_info = self.fig.text(0.015, 0.918, "", color=CUR_FILL, fontsize=11,
                                    va="top", family="monospace")
        self.t_log = self.fig.text(0.015, 0.892, "", color=MUTED, fontsize=9,
                                   va="top", linespacing=1.4)

        handles = [
            Line2D([0], [0], marker="o", ls="none", mfc=CUR_FILL, mec=CUR_FILL,
                   ms=9, label="actual"),
            Line2D([0], [0], marker="o", ls="none", mfc="none", mec=TERM,
                   mew=2, ms=10, label="fin palabra (E>0)"),
            Line2D([0], [0], marker="o", ls="none", mfc=P_BADGE, mec=P_BADGE,
                   ms=9, label="P: word_count"),
            Line2D([0], [0], color=PATH_C, lw=2.4, label="ruta paso"),
            Line2D([0], [0], color=QUERY_C, lw=2.4, label="consulta"),
            Line2D([0], [0], marker="o", ls="none", mfc="none", mec="#ffffff",
                   mew=2, ms=10, label="nuevo"),
        ]
        # Legend en SU PROPIA fila, empezando por la izquierda
        leg = self.fig.legend(handles=handles, loc="upper left", ncol=6,
                              frameon=False, bbox_to_anchor=(0.015, 0.960),
                              fontsize=9, columnspacing=1.0, handletextpad=0.4,
                              borderaxespad=0.0)
        for t in leg.get_texts():
            t.set_color(FG)

        self.set_words(raw_words, step)
        if interactive:
            self._build_widgets(", ".join(raw_words))
        self.fig.canvas.mpl_connect("resize_event", lambda e: self.render())
        self.render()

    def set_words(self, words, step=None):
        self.words = [w[:MAX_WORD_LEN] for w in words[:MAX_WORDS]]
        self.hist = build_history(self.words)
        total = len(self.hist) - 1
        if step is None or step < 0:
            self.step = total
        else:
            self.step = max(0, min(step, total))

    def _build_widgets(self, initial_text):
        f = self.fig

        def styled_ax(rect):
            a = f.add_axes(rect)
            a.set_facecolor("#20242e")
            return a

        # --- Fila 1: palabras -------------------------------------------
        self.inp = InputBox(f, [0.075, 0.115, 0.18, 0.04], initial_text,
                            max_len=200, label="Palabras",
                            on_submit=lambda _t: self._rebuild())
        self.b_clear = Button(styled_ax([0.26, 0.115, 0.025, 0.04]), "✕",
                              color="#20242e", hovercolor="#5a2a2a")
        self.b_clear.label.set_color(FG)
        self.b_clear.on_clicked(lambda _e: (self.inp.set_text(""), self.inp.focus()))

        self.b_build = Button(styled_ax([0.29, 0.115, 0.07, 0.04]), "Construir",
                              color="#2f4a7a", hovercolor="#3d5f9c")
        self.b_build.label.set_color(FG)
        self.b_build.on_clicked(lambda _e: self._rebuild())

        # --- Fila 2: consulta -------------------------------------------
        self.qinp = InputBox(f, [0.075, 0.06, 0.18, 0.04], "",
                             max_len=100, label="Consulta",
                             on_submit=lambda _t: self.render())
        self.qinp.on_change = self._on_query_change
        self.q_info = f.text(0.27, 0.08, "", color=FG, fontsize=10,
                             va="center", family="monospace")

        f.text(0.075, 0.005,
               "Ctrl+A / doble click: seleccionar todo · Ctrl+C/X/V: copiar/cortar/pegar · "
               "coma o espacio separa palabras · Enter: construir · Consulta se evalúa en vivo",
               color=MUTED, fontsize=8, va="bottom")

        # --- Fila 3: controles ------------------------------------------
        self.b_prev = Button(styled_ax([0.075, 0.02, 0.03, 0.032]), "◀",
                             color="#20242e", hovercolor="#2a3040")
        total = max(1, len(self.hist) - 1)
        ax_sl = styled_ax([0.11, 0.025, 0.15, 0.022])
        try:
            self.sl = Slider(ax_sl, "", 0, total, valinit=self.step, valstep=1,
                             color=CUR_FILL, initcolor="none")
        except TypeError:
            self.sl = Slider(ax_sl, "", 0, total, valinit=self.step, valstep=1,
                             color=CUR_FILL)
        self.sl.valtext.set_color(FG)
        self.b_next = Button(styled_ax([0.265, 0.02, 0.03, 0.032]), "▶",
                             color="#20242e", hovercolor="#2a3040")
        self.b_play = Button(styled_ax([0.30, 0.02, 0.045, 0.032]), "Play",
                             color="#20242e", hovercolor="#2a3040")
        self.b_export = Button(styled_ax([0.35, 0.02, 0.065, 0.032]), "Exportar",
                               color="#20242e", hovercolor="#2a3040")
        for b in (self.b_prev, self.b_next, self.b_play, self.b_export):
            b.label.set_color(FG)
        self.b_prev.on_clicked(lambda _e: self._goto(self.step - 1))
        self.b_next.on_clicked(lambda _e: self._goto(self.step + 1))
        self.b_play.on_clicked(lambda _e: self._toggle_play())
        self.b_export.on_clicked(lambda _e: self._export())
        self.sl.on_changed(lambda v: self._goto(int(v), from_slider=True))

        # --- Checkboxes a la derecha ------------------------------------
        ax_ck = styled_ax([0.72, 0.01, 0.26, 0.15])
        ax_ck.set_facecolor(BG)
        labels = ["Prefijos en nodos", "Resaltar ruta"]
        states = [self.strings_mode, self.highlight]
        try:
            self.ck = CheckButtons(ax_ck, labels, states,
                                   frame_props=dict(edgecolor=FG, facecolor=BG,
                                                    sizes=[110, 110]),
                                   check_props=dict(color=TERM, linewidth=2.5))
        except TypeError:
            self.ck = CheckButtons(ax_ck, labels, states)
        for lab in self.ck.labels:
            lab.set_color(FG)
            lab.set_fontsize(10)
        self.ck.on_clicked(self._on_check)

    def _on_query_change(self, text):
        self.query_text = text
        self.render()

    def _on_check(self, label):
        if label.startswith("Prefijos"):
            self.strings_mode = not self.strings_mode
        else:
            self.highlight = not self.highlight
        self.render()

    def _rebuild(self):
        words = parse_words(self.inp.text)
        if not words:
            self.t_log.set_text("(no hay palabras válidas: escribe algo y pulsa Enter)")
            self.fig.canvas.draw_idle()
            return
        self._stop_play()
        self.set_words(words)
        self._busy = True
        total = max(1, len(self.hist) - 1)
        self.sl.valmax = total
        self.sl.ax.set_xlim(self.sl.valmin, total)
        self.sl.set_val(self.step)
        self._busy = False
        self.render()

    def _goto(self, k, from_slider=False):
        if self._busy:
            return
        k = max(0, min(k, len(self.hist) - 1))
        self.step = k
        if not from_slider and self.interactive:
            self._busy = True
            self.sl.set_val(k)
            self._busy = False
        self.render()

    def _toggle_play(self):
        if self._timer is not None:
            self._stop_play(); return
        if self.step >= len(self.hist) - 1:
            self.step = 0
        self._timer = self.fig.canvas.new_timer(interval=700)
        self._timer.add_callback(self._tick)
        self._timer.start()
        self.b_play.label.set_text("Pausa")

    def _tick(self):
        if self.step >= len(self.hist) - 1:
            self._stop_play(); return
        self._goto(self.step + 1)

    def _stop_play(self):
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
            if self.interactive:
                self.b_play.label.set_text("Play")

    def _export(self):
        base = "_".join(self.words[:3]) or "vacio"
        default = f"trie_{base}_paso{self.step}.png"
        path = export_dialog(self.fig, default)
        if not path:
            return
        try:
            self.fig.savefig(path, dpi=140, facecolor=BG)
            print("Guardado:", path)
        except Exception as e:
            print("Error al guardar:", e)

    def render(self):
        s = self.hist[self.step]

        # Título abreviado
        if self.words:
            shown, rest = list(self.words), 0
            txt = ", ".join(vis(w) for w in shown)
            while len(shown) > 1 and len(txt) > 40:
                rest += 1
                shown = shown[:-1]
                txt = ", ".join(vis(w) for w in shown)
            title = "«" + txt + (f", … (+{rest})" if rest else "") + "»"
        else:
            title = "(sin palabras)"
        self.t_title.set_text(f"Visualizador de Trie — {title}")

        self.t_info.set_text(
            f"Paso {self.step}/{len(self.hist)-1}   {s.label}   "
            f"NODO ACTUAL: S{s.cur}   nodos: {s.n}")

        # Log + resultado de la consulta
        log_lines = list(s.log)
        query_path, query_node, q_results = set(), -1, None
        if self.interactive and self.query_text:
            q = self.qinp.text.strip()
            if q:
                query_path, query_node, exists, q_results = query_trie(s, q)
                if not exists:
                    q_results = None
                else:
                    log_lines.append("")
                    log_lines.append(
                        f"CONSULTA «{vis(q)}» → S{query_node}: "
                        f"search = {str(q_results['search']).lower()} · "
                        f"count_word = {q_results['count_word']} · "
                        f"count_prefix = {q_results['count_prefix']} · "
                        f"starts_with = true")

        # Recorta el log para que NUNCA invada los paneles
        if len(log_lines) > LOG_MAX_LINES:
            log_lines = log_lines[-LOG_MAX_LINES:]
        self.t_log.set_text("\n".join(log_lines))

        # Cuadro resumen de la consulta
        if self.interactive:
            if self.query_text and q_results is not None:
                self.q_info.set_text(
                    f"search: {str(q_results['search']).lower()}   ·   "
                    f"count_word: {q_results['count_word']}   ·   "
                    f"count_prefix: {q_results['count_prefix']}   ·   "
                    f"starts_with: true")
                self.q_info.set_color(QUERY_C)
            elif self.query_text:
                self.q_info.set_text("starts_with: false   ·   search: false   ·   "
                                     "count_word: 0   ·   count_prefix: 0")
                self.q_info.set_color(MUTED)
            else:
                self.q_info.set_text("")

        draw_tree_panel(self.ax_t, s, self.strings_mode, self.highlight,
                        query_path=query_path, query_node=query_node)
        draw_words_panel(self.ax_w, s)

        if self.interactive:
            self.inp.relayout()
            if hasattr(self, 'qinp'):
                self.qinp.relayout()
        self.fig.canvas.draw_idle()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Visualizador de Trie / Árbol de Prefijos")
    ap.add_argument("palabras", nargs="*",
                    help="palabras a insertar (o una sola cadena separada por comas)")
    ap.add_argument("--png", help="guardar PNG y salir")
    ap.add_argument("--svg", help="guardar SVG y salir")
    ap.add_argument("--pdf", help="guardar PDF y salir")
    ap.add_argument("--paso", type=int, default=None,
                    help="paso a mostrar (por defecto: el final; -1 = último)")
    ap.add_argument("--strings", action="store_true",
                    help="modo caja: muestra S, word_count, word_end_count y prefijo")
    ap.add_argument("--sin-resaltado", action="store_true",
                    help="no resaltar el camino de la palabra en curso")
    a = ap.parse_args()

    if a.palabras:
        raw = ",".join(a.palabras) if len(a.palabras) > 1 else a.palabras[0]
    else:
        raw = "casa,caso,carro,cara"
    words = parse_words(raw)
    if not words:
        print("No hay palabras válidas.")
        return

    interactive = not any([a.png, a.svg, a.pdf])
    app = App(words, interactive=interactive, step=a.paso,
              strings=a.strings, highlight=not a.sin_resaltado)

    for path in (a.png, a.svg, a.pdf):
        if path:
            app.render()
            app.fig.savefig(path, dpi=140, facecolor=BG)
            print("Guardado:", path)
            return

    plt.show()


if __name__ == "__main__":
    main()
