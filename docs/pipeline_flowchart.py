"""
Draw the ETL pipeline flowchart (docs/pipeline_flowchart.svg and .png).

Every box is placed on a fixed grid and every text line is measured with
the real fonts, so nothing overlaps or wraps unevenly. Edit the content in
the stage sections of build() and re-run:

    python docs/pipeline_flowchart.py

Needs Pillow (text measurement) and rsvg-convert (PNG export; Homebrew
librsvg). Text measurement uses the macOS Helvetica and Menlo fonts.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from html import escape
from pathlib import Path

from PIL import ImageFont

OUT_DIR = Path(__file__).resolve().parent
SVG_PATH = OUT_DIR / "pipeline_flowchart.svg"
PNG_PATH = OUT_DIR / "pipeline_flowchart.png"
PNG_SCALE = 2

# --------------------------------------------------
# Style
# --------------------------------------------------

SANS = "Helvetica, Arial, sans-serif"
MONO = "Menlo, SFMono-Regular, Consolas, monospace"
FONT_FILES = {
    ("sans", False): ("/System/Library/Fonts/Helvetica.ttc", 0),
    ("sans", True): ("/System/Library/Fonts/Helvetica.ttc", 1),
    ("mono", False): ("/System/Library/Fonts/Menlo.ttc", 0),
    ("mono", True): ("/System/Library/Fonts/Menlo.ttc", 1),
}

TEXT = "#1F2A44"
MUTED = "#5B6678"
ARROW = "#4A5A70"
OUTPUT_FILL = "#F7F8FA"
OUTPUT_BORDER = "#BCC7D3"

# One hue per stage: (strong, box fill, box border, container border).
HUES = {
    1: ("#4F6D93", "#EDF2F8", "#A9BBD2", "#D3DDEA"),   # slate blue
    2: ("#4E8C5A", "#EDF6EF", "#A8CDB0", "#D2E6D6"),   # green
    3: ("#B7832F", "#FBF4E8", "#E2C79A", "#EFE1C8"),   # ochre
    4: ("#3C76B0", "#EAF2FA", "#A5C3E2", "#D0E0F0"),   # blue
    5: ("#7B5EA7", "#F3EFF9", "#C4B4DC", "#E0D7EE"),   # violet
    6: ("#2D8A80", "#E3F2F0", "#8CC5BE", "#C9E5E1"),   # teal (final output)
}

TITLE_SIZE = 14
NOTE_SIZE = 13
NOTE_LINE = 17
TITLE_LINE = 19
PILL_SIZE = 12
PILL_H = 24
PAD_X = 15
PAD_Y = 10
ICON = 18

# --------------------------------------------------
# Grid
# --------------------------------------------------

W = 1600
CONTAINER_X0, CONTAINER_X1 = 40, 1560
MAIN_X0, MAIN_X1 = 470, 1130            # the vertical chain
MAIN_CX = (MAIN_X0 + MAIN_X1) // 2      # 800, also the canvas centre
SIDE_X0, SIDE_X1 = 70, 430              # dashed side boxes, left of the chain
OUT_BRACKET_X = 1152                    # outputs, right of the chain
OUT_X0, OUT_X1 = 1164, 1548
LEFT_X0, LEFT_X1 = 90, 760              # parallel branches
RIGHT_X0, RIGHT_X1 = 840, 1510
STAGE_GAP = 18                          # between containers
ROW_GAP = 20                            # between rows inside a container
LABEL_H = 40                            # container label band
BOTTOM_PAD = 16                         # last row to container bottom
BELOW_GAP = 14                          # branch box to its output / side box


_fonts = {}


def font(kind, bold, size):
    key = (kind, bold, size)
    if key not in _fonts:
        path, index = FONT_FILES[(kind, bold)]
        _fonts[key] = ImageFont.truetype(path, size, index=index)
    return _fonts[key]


def text_width(text, kind="sans", bold=False, size=NOTE_SIZE):
    return font(kind, bold, size).getlength(text)


# --------------------------------------------------
# Rich text: `code` spans are monospace
# --------------------------------------------------

def parse_rich(text):
    """'a `b` c' -> [('a ', 'sans'), ('b', 'mono'), (' c', 'sans')]"""
    parts = re.split(r"(`[^`]+`)", text)
    return [(p[1:-1], "mono") if p.startswith("`") else (p, "sans") for p in parts if p]


def rich_words(text):
    """Split into words, each a list of (text, kind) runs; spaces separate words."""
    words, current = [], []
    for chunk, kind in parse_rich(text):
        pieces = chunk.split(" ")
        for i, piece in enumerate(pieces):
            if i > 0:
                if current:
                    words.append(current)
                current = []
            if piece:
                current.append((piece, kind))
    if current:
        words.append(current)
    return words


def runs_width(runs, size, bold=False):
    return sum(text_width(t, k, bold, size - 1 if k == "mono" else size) for t, k in runs)


def wrap_rich(text, max_width, size=NOTE_SIZE):
    """Lines of words that fit max_width."""
    space = text_width(" ", "sans", False, size)
    lines, line, width = [], [], 0.0
    for word in rich_words(text):
        w = runs_width(word, size)
        extra = w if not line else w + space
        if line and width + extra > max_width:
            lines.append(line)
            line, width = [word], w
        else:
            line.append(word)
            width += extra
    if line:
        lines.append(line)
    return lines


def render_runs(words, x, y, size, color, bold=False, anchor="start", sans_color=None):
    """One <text> line from a list of words (each a list of runs)."""
    spans = []
    for i, word in enumerate(words):
        for j, (t, kind) in enumerate(word):
            prefix = " " if i > 0 and j == 0 else ""
            if kind == "mono":
                if prefix:
                    spans.append(f'<tspan font-family="{SANS}">{prefix}</tspan>')
                spans.append(f'<tspan font-family="{MONO}" font-size="{size - 1}">'
                             f'{escape(t)}</tspan>')
            else:
                fill = f' fill="{sans_color}"' if sans_color else ""
                spans.append(f"<tspan{fill}>{escape(prefix + t)}</tspan>")
    weight = ' font-weight="bold"' if bold else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{SANS}" font-size="{size}" '
            f'fill="{color}" text-anchor="{anchor}"{weight}>{"".join(spans)}</text>')


# --------------------------------------------------
# Icons: simple line drawings on a 20 x 20 grid
# --------------------------------------------------

ICONS = {
    "pin": '<path d="M10 18.5C10 18.5 4 12 4 7.8a6 6 0 0 1 12 0C16 12 10 18.5 10 18.5Z"/>'
           '<circle cx="10" cy="7.8" r="2.2"/>',
    "river": '<path d="M2 6.5c2.2-2 4.4-2 6.6 0s4.4 2 6.6 0 2.8-1.4 2.8-1.4"/>'
             '<path d="M2 11c2.2-2 4.4-2 6.6 0s4.4 2 6.6 0 2.8-1.4 2.8-1.4"/>'
             '<path d="M2 15.5c2.2-2 4.4-2 6.6 0s4.4 2 6.6 0 2.8-1.4 2.8-1.4"/>',
    "target": '<circle cx="10" cy="10" r="6.5"/><circle cx="10" cy="10" r="2"/>'
              '<path d="M10 1.5v3.5M10 15v3.5M1.5 10h3.5M15 10h3.5"/>',
    "database": '<ellipse cx="10" cy="4.8" rx="6.5" ry="2.6"/>'
                '<path d="M3.5 4.8v10.4c0 1.4 2.9 2.6 6.5 2.6s6.5-1.2 6.5-2.6V4.8"/>'
                '<path d="M3.5 10c0 1.4 2.9 2.6 6.5 2.6s6.5-1.2 6.5-2.6"/>',
    "leaf": '<path d="M4 16.5C4 8.5 9 3.5 17 3.5 17 11.5 12 16.5 4 16.5Z"/>'
            '<path d="M4 16.5 12 8.5"/>',
    "cloud": '<path d="M6 12.5a3.6 3.6 0 0 1 .4-7.2 4.6 4.6 0 0 1 8.7 1.6 2.9 2.9 0 0 1-.5 5.6Z"/>'
             '<path d="M7.5 15.2 6.7 17.8M11 15.2l-.8 2.6M14.5 15.2l-.8 2.6"/>',
    "table": '<rect x="2.5" y="3.5" width="15" height="13" rx="1.5"/>'
             '<path d="M2.5 8h15M2.5 12.3h15M7.5 8v8.5"/>',
}


def icon_svg(name, x, y, color):
    scale = ICON / 20
    return (f'<g transform="translate({x:.1f},{y:.1f}) scale({scale:.3f})" fill="none" '
            f'stroke="{color}" stroke-width="1.7" stroke-linecap="round" '
            f'stroke-linejoin="round">{ICONS[name]}</g>')


# --------------------------------------------------
# Drawing primitives
# --------------------------------------------------

class Canvas:
    def __init__(self):
        self.back = []    # containers
        self.lines = []   # arrows and connectors (under boxes)
        self.front = []   # boxes and text

    def container(self, y0, y1, stage, label, sublabel=None):
        strong, _, _, border = HUES[stage]
        self.back.append(
            f'<rect x="{CONTAINER_X0}" y="{y0:.1f}" width="{CONTAINER_X1 - CONTAINER_X0}" '
            f'height="{y1 - y0:.1f}" rx="12" fill="#FFFFFF" stroke="{border}" '
            f'stroke-width="1.4"/>')
        cx, cy = CONTAINER_X0 + 26, y0 + 20
        self.back.append(f'<circle cx="{cx}" cy="{cy:.1f}" r="10.5" fill="{strong}"/>')
        self.back.append(
            f'<text x="{cx}" y="{cy + 4.5:.1f}" font-family="{SANS}" font-size="12.5" '
            f'font-weight="bold" fill="#FFFFFF" text-anchor="middle">{stage}</text>')
        sub = (f'<tspan font-weight="normal" fill="{MUTED}">  {escape(sublabel)}</tspan>'
               if sublabel else "")
        self.back.append(
            f'<text x="{cx + 19}" y="{cy + 5.5:.1f}" font-family="{SANS}" font-size="16" '
            f'font-weight="bold" fill="{strong}">{escape(label)}{sub}</text>')

    def arrow(self, points, dashed=False, head=True):
        d = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(points))
        dash = ' stroke-dasharray="6 5"' if dashed else ""
        marker = ' marker-end="url(#head)"' if head else ""
        self.lines.append(f'<path d="{d}" fill="none" stroke="{ARROW}" stroke-width="1.6"'
                          f'{dash}{marker}/>')

    def connector(self, points):
        d = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(points))
        self.lines.append(f'<path d="{d}" fill="none" stroke="{OUTPUT_BORDER}" '
                          f'stroke-width="1.3"/>')

    def svg(self, height):
        defs = (
            '<defs><marker id="head" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" '
            f'markerHeight="8" orient="auto-start-reverse"><path d="M0,1 L9,5 L0,9 z" '
            f'fill="{ARROW}"/></marker></defs>')
        body = "\n".join(self.back + self.lines + self.front)
        # xml:space keeps the spaces between tspans (renderers drop them otherwise).
        return (f'<svg xmlns="http://www.w3.org/2000/svg" xml:space="preserve" '
                f'width="{W}" height="{height}" '
                f'viewBox="0 0 {W} {height}">\n{defs}\n'
                f'<rect width="{W}" height="{height}" fill="#FFFFFF"/>\n{body}\n</svg>\n')


class Box:
    """A process box: bold title (monospace unless sans_title), then notes."""

    def __init__(self, title, notes, x0, x1, stage, dashed=False, title_note=None,
                 icon=None, sans_title=False):
        self.title, self.notes, self.x0, self.x1 = title, notes, x0, x1
        self.stage, self.dashed, self.title_note = stage, dashed, title_note
        self.icon, self.sans_title = icon, sans_title
        inner = x1 - x0 - 2 * PAD_X
        self.indent = 13 if len(notes) > 1 else 0
        self.wrapped = [wrap_rich(n, inner - self.indent) for n in notes]
        n_lines = sum(len(w) for w in self.wrapped)
        gaps = 3 * max(len(notes) - 1, 0)
        self.h = 2 * PAD_Y + TITLE_LINE + 3 + n_lines * NOTE_LINE + gaps
        self.y0 = 0

    @property
    def cx(self):
        return (self.x0 + self.x1) / 2

    @property
    def y1(self):
        return self.y0 + self.h

    @property
    def cy(self):
        return self.y0 + self.h / 2

    def draw(self, canvas):
        strong, fill, border, _ = HUES[self.stage]
        if self.dashed:
            fill, border, dash = "#FFFFFF", strong, ' stroke-dasharray="6 5"'
        else:
            dash = ""
        canvas.front.append(
            f'<rect x="{self.x0}" y="{self.y0:.1f}" width="{self.x1 - self.x0}" '
            f'height="{self.h:.1f}" rx="9" fill="{fill}" stroke="{border}" '
            f'stroke-width="1.4"{dash}/>')
        x = self.x0 + PAD_X
        y = self.y0 + PAD_Y + 14
        family = SANS if self.sans_title else MONO
        size = TITLE_SIZE + 1 if self.sans_title else TITLE_SIZE
        title = (f'<text x="{x}" y="{y:.1f}" font-family="{family}" font-size="{size}" '
                 f'font-weight="bold" fill="{TEXT}">{escape(self.title)}')
        if self.title_note:
            title += (f'<tspan font-family="{SANS}" font-weight="normal" font-size="{NOTE_SIZE}" '
                      f'fill="{MUTED}">  {escape(self.title_note)}</tspan>')
        canvas.front.append(title + "</text>")
        if self.icon:
            canvas.front.append(icon_svg(self.icon, self.x1 - PAD_X - ICON,
                                         self.y0 + PAD_Y + 1, strong))
        y += TITLE_LINE + 3
        for lines in self.wrapped:
            for i, words in enumerate(lines):
                if self.indent and i == 0:
                    canvas.front.append(
                        f'<circle cx="{x + 4}" cy="{y - 4.5:.1f}" r="2.1" fill="{strong}"/>')
                canvas.front.append(render_runs(words, x + self.indent, y, NOTE_SIZE, TEXT))
                y += NOTE_LINE
            y += 3


def same_height(*boxes):
    h = max(b.h for b in boxes)
    for b in boxes:
        b.h = h


def pill_width(label):
    words = rich_words(label)
    return runs_width([r for w in words for r in w], PILL_SIZE) \
        + text_width(" ", size=PILL_SIZE) * (len(words) - 1) + 24


def draw_pill(canvas, label, x0, y0, accent=False):
    w = pill_width(label)
    if accent:
        fill, stroke = HUES[6][2], HUES[6][0]
    else:
        fill, stroke = OUTPUT_FILL, OUTPUT_BORDER
    canvas.front.append(
        f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{w:.1f}" height="{PILL_H}" rx="{PILL_H / 2}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.2"/>')
    # Plain words in a pill (e.g. 'cleaned') are a muted annotation.
    canvas.front.append(render_runs(rich_words(label), x0 + 12, y0 + 16.5, PILL_SIZE, TEXT,
                                    sans_color=MUTED if "`" in label else None))
    return w


def pill_rows(labels, max_width):
    rows, row, width = [], [], 0.0
    for label in labels:
        w = pill_width(label)
        if row and width + 8 + w > max_width:
            rows.append(row)
            row, width = [], 0.0
        row.append(label)
        width += w if len(row) == 1 else w + 8
    if row:
        rows.append(row)
    return rows


def outputs_height(labels, max_width=OUT_X1 - OUT_X0):
    n = len(pill_rows(labels, max_width))
    return n * PILL_H + (n - 1) * 6


def outputs_right(canvas, box, labels, accent=False):
    """Pills right of a chain box, joined to it by a bracket."""
    rows = pill_rows(labels, OUT_X1 - OUT_X0)
    h = outputs_height(labels)
    top = box.cy - h / 2
    canvas.connector([(box.x1, box.cy), (OUT_BRACKET_X, box.cy)])
    if len(rows) > 1:
        canvas.connector([(OUT_BRACKET_X, top + PILL_H / 2),
                          (OUT_BRACKET_X, top + h - PILL_H / 2)])
    for i, row in enumerate(rows):
        y = top + i * (PILL_H + 6)
        canvas.connector([(OUT_BRACKET_X, y + PILL_H / 2), (OUT_X0, y + PILL_H / 2)])
        x = OUT_X0
        for label in row:
            x += draw_pill(canvas, label, x, y, accent) + 8


def outputs_below(canvas, box, labels, y0):
    """Pills centred under a branch box."""
    widths = [pill_width(label) for label in labels]
    total = sum(widths) + 8 * (len(widths) - 1)
    canvas.connector([(box.cx, box.y1), (box.cx, y0)])
    x = box.cx - total / 2
    for label, w in zip(labels, widths):
        draw_pill(canvas, label, x, y0)
        x += w + 8


def merge(canvas, left, right, target_y):
    """Join two side-by-side branch boxes into one arrow down the centre line."""
    y = left.cy
    canvas.arrow([(left.x1, y), (MAIN_CX, y)], head=False)
    canvas.arrow([(right.x0, y), (MAIN_CX, y)], head=False)
    canvas.arrow([(MAIN_CX, y), (MAIN_CX, target_y)])


def fork(canvas, from_y, fork_y, left, right):
    """Split the centre line into two side-by-side branch boxes."""
    canvas.arrow([(MAIN_CX, from_y), (MAIN_CX, fork_y), (left.cx, fork_y), (left.cx, left.y0)])
    canvas.arrow([(MAIN_CX, fork_y), (right.cx, fork_y), (right.cx, right.y0)])


# --------------------------------------------------
# The diagram
# --------------------------------------------------

def build():
    c = Canvas()
    y = 24

    # 1 Scope -------------------------------------------------------------
    s1_top = y
    dams = Box("American Rivers dam removal list",
               ["One-time manual download from Figshare (March 2026): location, river and "
                "removal year of every dam removal in the US."],
               LEFT_X0, LEFT_X1, 1, dashed=True, sans_title=True)
    wbd = Box("download_huc8 → unzip_huc8",
              ["One-time download of the national Watershed Boundary Dataset; keep only the "
               "HUC8 layer."],
              RIGHT_X0, RIGHT_X1, 1, dashed=True)
    same_height(dams, wbd)
    dams.y0 = wbd.y0 = y + LABEL_H
    pills1_y = dams.y1 + BELOW_GAP
    scope = Box("case_study.env + sample_random_pilot",
                ["Choose the case study: every dam removal (`all_removals`) or a random sample.",
                 "Place each dam in the HUC8 polygon that contains it; those HUC8s are the "
                 "study area for stations and streams.",
                 "Set the before/after window around each removal year (±5 years)."],
                MAIN_X0, MAIN_X1, 1)
    scope.y0 = pills1_y + PILL_H + ROW_GAP
    s1_bottom = scope.y1 + BOTTOM_PAD
    c.container(s1_top, s1_bottom, 1, "Scope")
    for box in (dams, wbd, scope):
        box.draw(c)
    outputs_below(c, dams, ["`ARDamRemovalList_Figshare_Mar2026.csv`"], pills1_y)
    outputs_below(c, wbd, ["`WBD_National_GPKG.gpkg`"], pills1_y)
    merge(c, dams, wbd, scope.y0)
    y = s1_bottom + STAGE_GAP

    # 2 Stations & streams --------------------------------------------------
    s2_top = y
    stations = Box("download_wqp_stations → select_candidate_stations",
                   ["All Water Quality Portal stations in the study HUC8s; keep only Stream "
                    "and River/Stream sites with valid coordinates.",
                    "No straight-line distance filter: the river network decides relevance."],
                   LEFT_X0, LEFT_X1, 2, icon="pin")
    streams = Box("download_streams → unzip_nhdplus",
                  ["NHDPlus High Resolution packages (the HU4 package where no HUC8 package "
                   "exists).",
                   "Keep only 4 layers (flowlines, flow table, catchments, reach attributes) "
                   "in a slim `.gpkg`."],
                  RIGHT_X0, RIGHT_X1, 2, icon="river")
    same_height(stations, streams)
    stations.y0 = streams.y0 = y + LABEL_H + 6
    below_y = stations.y1 + BELOW_GAP
    network = Box("build_stream_network", ["Network diagnostics only (optional)."],
                  RIGHT_X0 + 150, RIGHT_X1 - 150, 2, dashed=True)
    network.y0 = below_y
    s2_bottom = max(network.y1, below_y + PILL_H) + BOTTOM_PAD
    c.container(s2_top, s2_bottom, 2, "Stations & streams")
    for box in (stations, streams, network):
        box.draw(c)
    outputs_below(c, stations, ["`pilot_station_selection.csv`"], below_y)
    c.arrow([(streams.cx, streams.y1), (streams.cx, network.y0)], dashed=True)
    fork(c, scope.y1, s2_top + LABEL_H - 10, stations, streams)
    y = s2_bottom + STAGE_GAP

    # 3 Network placement ---------------------------------------------------
    s3_top = y
    snap = Box("snap_features",
               ["Transform coordinates to NAD83 using datum grids; stop if a grid is missing "
                "rather than approximate.",
                "Only stream/river reaches within distance limits (dams 500 m, stations "
                "1,000 m).",
                "Score candidate reaches on distance, river name and drainage area → accepted "
                "/ ambiguous / rejected; manual overrides from `snap_overrides.csv`.",
                "Only accepted points continue; diagnostics are kept."],
               MAIN_X0, MAIN_X1, 3, icon="target")
    validation = Box("snap_validation / compare_snapping",
                     ["Optional QA: test snapping rule changes on a sample of HUC8s before "
                      "applying them."],
                     SIDE_X0, SIDE_X1, 3, dashed=True)
    link = Box("link_dams",
               ["Walk the downstream flow graph from each dam: each station is upstream / "
                "downstream / same reach / unconnected.",
                "Distance measured along the river, not straight-line."],
               MAIN_X0, MAIN_X1, 3)
    snap.y0 = y + LABEL_H
    validation.y0 = snap.cy - validation.h / 2
    link.y0 = snap.y1 + ROW_GAP
    s3_bottom = link.y1 + BOTTOM_PAD
    c.container(s3_top, s3_bottom, 3, "Network placement")
    for box in (snap, validation, link):
        box.draw(c)
    c.arrow([(validation.x1, validation.cy), (snap.x0, validation.cy)], dashed=True)
    outputs_right(c, snap, ["`snapped_features_{huc8}.gpkg`"])
    outputs_right(c, link, ["`dam_station_links_{huc8}.csv`"])
    merge(c, stations, streams, snap.y0)
    c.arrow([(MAIN_CX, snap.y1), (MAIN_CX, link.y0)])
    y = s3_bottom + STAGE_GAP

    # 4 Measurements -> warehouse -------------------------------------------
    s4_top = y
    chain = [
        (Box("download_wqp_results",
             ["Only linked stations, only within each dam's before/after window."],
             MAIN_X0, MAIN_X1, 4),
         ["raw CSV batches"]),
        (Box("load_core_tables",
             ["Core tables with reach type, name, drainage area, slope, snap distance and "
              "snap status."],
             MAIN_X0, MAIN_X1, 4),
         ["`dam_removal`", "`wq_station`", "`dam_station_link`", "`huc8_watershed`",
          "`streams`"]),
        (Box("clean_wq_results",
             ["Map parameters, standardize units, drop QC blanks/replicates and implausible "
              "values, deduplicate.",
              "Non-detects = ½ detection limit, flagged; keeps `method_id`."],
             MAIN_X0, MAIN_X1, 4),
         ["`wq_measurement` cleaned", "`wq_measurement_raw` unfiltered audit copy"]),
        (Box("build_daily_measurements",
             ["One value per station · day · depth · sample type · method.",
              "Repeats combined by median; measured values preferred over non-detects."],
             MAIN_X0, MAIN_X1, 4),
         ["`wq_measurement_daily`"]),
        (Box("build_views",
             ["Eligibility: ≥ 3 samples in ≥ 2 years both before AND after removal; the "
              "removal year is excluded.",
              "Exposure: each measurement is assigned to its nearest upstream removal; "
              "competing removals are flagged, not assigned.",
              "Station QC flags: far snap, unsuitable reach, saline, non-river name. Rows are "
              "flagged, not dropped."],
             MAIN_X0, MAIN_X1, 4, title_note="views only, no data copied", icon="database"),
         ["`warehouse.duckdb`"]),
    ]
    yy = y + LABEL_H
    for box, labels in chain:
        box.h = max(box.h, outputs_height(labels) + 2 * PAD_Y)
        box.y0 = yy
        yy = box.y1 + ROW_GAP
    s4_bottom = chain[-1][0].y1 + BOTTOM_PAD
    c.container(s4_top, s4_bottom, 4, "Measurements → warehouse", "Parquet + DuckDB")
    c.arrow([(MAIN_CX, link.y1), (MAIN_CX, chain[0][0].y0)])
    for i, (box, labels) in enumerate(chain):
        box.draw(c)
        outputs_right(c, box, labels)
        if i:
            c.arrow([(MAIN_CX, chain[i - 1][0].y1), (MAIN_CX, box.y0)])
    views = chain[-1][0]
    y = s4_bottom + STAGE_GAP

    # 5 Watershed covariates ------------------------------------------------
    s5_top = y
    watershed = Box("build_station_watershed",
                    ["Upstream catchments of each station → upstream area.",
                     "Placeholder (−9999) and truncated areas are flagged, never stored as "
                     "values."],
                    MAIN_X0, MAIN_X1, 5)
    land = Box("download_nlcd → load_land_cover",
               ["Annual NLCD 30 m land-cover mix of the upstream area."],
               LEFT_X0, LEFT_X1, 5, icon="leaf")
    climate = Box("download_climate → load_climate",
                  ["PRISM monthly precipitation and temperature, area-weighted over the "
                   "upstream area."],
                  RIGHT_X0, RIGHT_X1, 5, icon="cloud")
    watershed.y0 = y + LABEL_H
    fork5_y = watershed.y1 + ROW_GAP
    same_height(land, climate)
    land.y0 = climate.y0 = fork5_y + ROW_GAP
    pills5_y = land.y1 + BELOW_GAP
    s5_bottom = pills5_y + PILL_H + BOTTOM_PAD
    c.container(s5_top, s5_bottom, 5, "Watershed covariates", "eligible stations only")
    for box in (watershed, land, climate):
        box.draw(c)
    outputs_right(c, watershed, ["`station_watershed`", "`station_catchment`"])
    outputs_below(c, land, ["`land_cover`"], pills5_y)
    outputs_below(c, climate, ["`climate`"], pills5_y)
    c.arrow([(MAIN_CX, views.y1), (MAIN_CX, watershed.y0)])
    fork(c, watershed.y1, fork5_y, land, climate)
    y = s5_bottom + STAGE_GAP

    # 6 Analysis table ------------------------------------------------------
    s6_top = y
    analysis = Box("build_analysis_dataset",
                   ["One row per pre/post measurement × eligible dam–station pair, with dam, "
                    "watershed, climate and land-cover columns.",
                    "Land cover is a fixed baseline: the last NLCD year before removal.",
                    "Climate: the sample month plus the previous month's precipitation.",
                    "Exposure status, QC flags and missing-covariate flags are columns, not "
                    "filters.",
                    "Associations only, no causal claim."],
                   MAIN_X0, MAIN_X1, 6, icon="table")
    analysis.y0 = y + LABEL_H
    s6_bottom = analysis.y1 + BOTTOM_PAD
    c.container(s6_top, s6_bottom, 6, "Analysis table")
    analysis.draw(c)
    outputs_right(c, analysis, ["`analysis_measurement`"], accent=True)
    merge(c, land, climate, analysis.y0)
    y = s6_bottom + 20

    # Footer and legend -----------------------------------------------------
    c.front.append(
        f'<text x="{CONTAINER_X0}" y="{y + 12}" font-family="{SANS}" font-size="12" '
        f'fill="{MUTED}">Every stage validates before writing and replaces outputs atomically; '
        f'if a stage fails, every later stage is cancelled.</text>')
    lx = CONTAINER_X1 - 500
    ly = y + 8
    c.arrow([(lx, ly), (lx + 36, ly)])
    c.front.append(f'<text x="{lx + 44}" y="{ly + 4}" font-family="{SANS}" font-size="12" '
                   f'fill="{MUTED}">pipeline chain</text>')
    lx += 140
    c.arrow([(lx, ly), (lx + 36, ly)], dashed=True)
    c.front.append(f'<text x="{lx + 44}" y="{ly + 4}" font-family="{SANS}" font-size="12" '
                   f'fill="{MUTED}">optional / one-time</text>')
    lx += 172
    draw_pill(c, "data output", lx, ly - PILL_H / 2)
    c.front.append(f'<text x="{lx + pill_width("data output") + 8}" y="{ly + 4}" '
                   f'font-family="{SANS}" font-size="12" fill="{MUTED}">file / table</text>')
    return c.svg(int(y + 30))


def main():
    SVG_PATH.write_text(build(), encoding="utf-8")
    print(f"Wrote {SVG_PATH}")
    if shutil.which("rsvg-convert"):
        subprocess.run(["rsvg-convert", "-z", str(PNG_SCALE), "-o", str(PNG_PATH),
                        str(SVG_PATH)], check=True)
        print(f"Wrote {PNG_PATH}")
    else:
        print("rsvg-convert not found; PNG not written")


if __name__ == "__main__":
    main()
