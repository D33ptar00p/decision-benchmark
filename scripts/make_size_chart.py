#!/usr/bin/env python3
"""Builds reports/model-size-vs-performance.html: how model size relates to accuracy and speed.

Reads the official-harness runs archived in reports/data/raw/harness/ (3 runs x 2,934 cases per
model, all on AC power) and draws three charts as inline SVG (no libraries, light and dark mode):
  A. accuracy vs model size     B. latency vs model size     C. per dataset: extra accuracy vs extra time
plus a preliminary section (D, E) that adds the two TEV models on the same 40 cases (10 from each of 4 datasets).

    uv run python scripts/make_size_chart.py

Only two sizes were benchmarked properly (Laya 421M and Strands Decider ~2B), so the charts show those
and mark the larger sizes as "not measured" instead of drawing a trend through two points.
"""
from __future__ import annotations

import collections
import gzip
import html
import json
import math
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "reports" / "data" / "raw" / "harness"
OUT = ROOT / "reports" / "model-size-vs-performance.html"
CUSTOM = ROOT / "reports" / "data" / "raw" / "custom_runner"

# model -> (folder, label, parameters in billions; nominal, from the model names)
MODELS = {
    "laya_en": ("laya-english", "Laya English", 0.421),
    "laya_td": ("laya-typed-decisions", "Laya typed-decisions", 0.421),
    "strands": ("strands-decision", "Strands Decider 2B", 2.0),
}
TYPES = {"choice": "choice", "noul": "yes/no", "score": "score"}
TYPE_VAR = {"choice": "var(--c1)", "noul": "var(--c2)", "score": "var(--c3)"}


def load(path: Path):
    return json.load(gzip.open(path, "rt"))


def collect() -> dict:
    """Per model: accuracy (%) and latency (ms per case, end to end) overall, by type and by dataset."""
    out = {}
    for key, (folder, label, params) in MODELS.items():
        runs = []
        for i in (1, 2, 3):
            base = RAW / folder / f"run{i}"
            timings = load(f"{base}.timing.json.gz")
            preds = {p["id"]: p for p in load(f"{base}.json.gz")}
            runs.append((timings, preds))
        meta = {i: (p["domain"], p["type"]) for i, p in runs[0][1].items()}

        def agg(ids):
            acc = st.mean(100 * sum(1 for i in ids if preds[i]["correct"]) / len(ids) for _, preds in runs)
            lat = st.mean(st.mean(t[i] for i in ids) for t, _ in runs)
            return {"acc": acc, "lat": lat, "n": len(ids)}

        groups = collections.defaultdict(list)
        for i, (d, t) in meta.items():
            groups[(d, t)].append(i)
        out[key] = {
            "label": label, "params": params, "overall": agg(list(meta)),
            "by_type": {t: agg([i for i, (d, ty) in meta.items() if ty == t]) for t in TYPES},
            "by_dataset": {k: agg(v) for k, v in groups.items()},
        }
    return out


# ------------------------------------------------------------------------------ svg helpers
def spread(ys: list[float], gap: float = 14.0) -> list[float]:
    """Nudge label positions apart so text does not overlap, keeping the block centred."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for a, b in zip(order, order[1:]):
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    shift = (sum(ys) - sum(out)) / len(ys)
    return [o + shift for o in out]


def esc(s: object) -> str:
    return html.escape(str(s), quote=True)


class Axes:
    """Maps data to pixels for one chart (x optionally logarithmic)."""

    def __init__(self, w, h, ml, mr, mt, mb, x0, x1, y0, y1, xlog=True):
        self.w, self.h, self.ml, self.mr, self.mt, self.mb = w, h, ml, mr, mt, mb
        self.x0, self.x1, self.y0, self.y1, self.xlog = x0, x1, y0, y1, xlog

    @property
    def pw(self):
        return self.w - self.ml - self.mr

    @property
    def ph(self):
        return self.h - self.mt - self.mb

    def x(self, v):
        import math
        f = (math.log(v / self.x0) / math.log(self.x1 / self.x0)) if self.xlog else (v - self.x0) / (self.x1 - self.x0)
        return self.ml + f * self.pw

    def y(self, v):
        return self.mt + (1 - (v - self.y0) / (self.y1 - self.y0)) * self.ph


def frame(ax: Axes, xticks, xfmt, yticks, yfmt, ylabel, xlabel) -> list[str]:
    s = []
    for v in yticks:
        y = ax.y(v)
        s.append(f'<line class="grid" x1="{ax.ml}" x2="{ax.w - ax.mr}" y1="{y:.1f}" y2="{y:.1f}"/>')
        s.append(f'<text class="tick" x="{ax.ml - 8}" y="{y + 4:.1f}" text-anchor="end">{esc(yfmt(v))}</text>')
    for v in xticks:
        x = ax.x(v)
        s.append(f'<line class="tickmark" x1="{x:.1f}" x2="{x:.1f}" y1="{ax.h - ax.mb}" y2="{ax.h - ax.mb + 5}"/>')
        s.append(f'<text class="tick" x="{x:.1f}" y="{ax.h - ax.mb + 19}" text-anchor="middle">{esc(xfmt(v))}</text>')
    s.append(f'<line class="axis" x1="{ax.ml}" x2="{ax.w - ax.mr}" y1="{ax.h - ax.mb}" y2="{ax.h - ax.mb}"/>')
    s.append(f'<text class="axislabel" x="{ax.ml + ax.pw / 2:.0f}" y="{ax.h - 8}" text-anchor="middle">{esc(xlabel)}</text>')
    s.append(f'<text class="axislabel" transform="translate(14 {ax.mt + ax.ph / 2:.0f}) rotate(-90)" text-anchor="middle">{esc(ylabel)}</text>')
    return s


def size_chart(data: dict, metric: str, ylabel: str, y0, y1, yticks, yfmt, label_fmt, title: str, desc: str) -> str:
    """Slope chart: each question type (and overall) from the 421M model to the ~2B model."""
    W, H = 640, 400
    ax = Axes(W, H, 62, 128, 20, 70, 0.3, 12.0, y0, y1)
    small, big = data["laya_td"], data["strands"]
    eng = data["laya_en"]
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-labelledby="t-{metric} d-{metric}">',
         f'<title id="t-{metric}">{esc(title)}</title><desc id="d-{metric}">{esc(desc)}</desc>']
    # region beyond the largest measured size
    xs, xe = ax.x(2.5), ax.x(12.0)
    s.append(f'<rect class="nodata" x="{xs:.1f}" y="{ax.mt}" width="{xe - xs:.1f}" height="{ax.ph}"/>')
    s.append(f'<text class="note" x="{(xs + xe) / 2:.0f}" y="{ax.mt + 16}" text-anchor="middle">not measured at scale</text>')
    s.append(f'<text class="note" x="{(xs + xe) / 2:.0f}" y="{ax.mt + 31}" text-anchor="middle">(TEV 4B: 40-case check below)</text>')
    s += frame(ax, [0.5, 1, 2, 4, 8], lambda v: f"{v:g}B", yticks, yfmt, ylabel, "model size (billions of parameters, log scale)")
    s.append(f'<text class="modelname" x="{ax.x(small["params"]):.1f}" y="{ax.h - ax.mb + 40}" text-anchor="middle">Laya (421M)</text>')
    s.append(f'<text class="modelname" x="{ax.x(big["params"]):.1f}" y="{ax.h - ax.mb + 40}" text-anchor="middle">Strands Decider (~2B)</text>')

    rows = [("overall", "overall", "var(--ink)", small["overall"], big["overall"], eng["overall"])]
    rows += [(t, TYPES[t], TYPE_VAR[t], small["by_type"][t], big["by_type"][t], eng["by_type"][t]) for t in TYPES]
    xa, xb = ax.x(small["params"]), ax.x(big["params"])
    left_y = spread([ax.y(r[3][metric]) for r in rows])
    right_y = spread([ax.y(r[4][metric]) for r in rows])
    for (key, name, color, a, b, e), ly, ry in zip(rows, left_y, right_y):
        sw = 3.2 if key == "overall" else 2.2
        s.append(f'<g class="series">')
        s.append(f'<line x1="{xa:.1f}" y1="{ax.y(a[metric]):.1f}" x2="{xb:.1f}" y2="{ax.y(b[metric]):.1f}" stroke="{color}" stroke-width="{sw}"/>')
        # Laya English: same size, different training (hollow ring)
        s.append(f'<circle class="ring" cx="{xa:.1f}" cy="{ax.y(e[metric]):.1f}" r="6" stroke="{color}"><title>{esc(name)}: Laya English {esc(label_fmt(e[metric]))}</title></circle>')
        s.append(f'<circle cx="{xa:.1f}" cy="{ax.y(a[metric]):.1f}" r="5.5" fill="{color}"><title>{esc(name)}: Laya typed-decisions {esc(label_fmt(a[metric]))}</title></circle>')
        s.append(f'<circle cx="{xb:.1f}" cy="{ax.y(b[metric]):.1f}" r="5.5" fill="{color}"><title>{esc(name)}: Strands Decider {esc(label_fmt(b[metric]))}</title></circle>')
        s.append(f'<text class="val" x="{xa - 11:.1f}" y="{ly + 4:.1f}" text-anchor="end">{esc(label_fmt(a[metric]))}</text>')
        s.append(f'<text class="val" x="{xb + 11:.1f}" y="{ry + 4:.1f}">{esc(name)} {esc(label_fmt(b[metric]))}</text>')
        s.append('</g>')
    s.append('</svg>')
    return "\n".join(s)


def scatter_chart(data: dict) -> tuple[str, list[dict]]:
    """Per JevBench dataset: extra latency (x) against extra accuracy (y), Strands vs Laya typed-decisions."""
    W, H = 760, 460
    pts = []
    for key, small in data["laya_td"]["by_dataset"].items():
        big = data["strands"]["by_dataset"][key]
        d, t = key
        pts.append({"dataset": d, "type": t, "ratio": big["lat"] / small["lat"], "gain": big["acc"] - small["acc"],
                    "acc_small": small["acc"], "acc_big": big["acc"], "lat_small": small["lat"], "lat_big": big["lat"], "n": big["n"]})
    ax = Axes(W, H, 62, 24, 20, 60, 2.0, 10.0, -45, 80)
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-labelledby="t-c d-c">',
         '<title id="t-c">Extra accuracy against extra time, per dataset</title>'
         '<desc id="d-c">Each dot is one JevBench dataset. Horizontal: how many times slower Strands Decider is than Laya typed-decisions. '
         'Vertical: how many accuracy points Strands gains (above the line) or loses (below it).</desc>']
    s += frame(ax, [2, 3, 4, 6, 8, 10], lambda v: f"{v:g}x", [-40, -20, 0, 20, 40, 60, 80], lambda v: f"{v:+d}" if v else "0",
               "accuracy gain of Strands over Laya typed (points)", "how many times slower Strands Decider is (log scale)")
    z = ax.y(0)
    s.append(f'<line class="zero" x1="{ax.ml}" x2="{W - ax.mr}" y1="{z:.1f}" y2="{z:.1f}"/>')
    s.append(f'<text class="note" x="{ax.ml + 6}" y="{ax.mt + 14}">above the line: Strands is more accurate</text>')
    s.append(f'<text class="note" x="{ax.ml + 6}" y="{ax.h - ax.mb - 8}">below the line: Laya typed-decisions is more accurate</text>')
    # label the extremes
    ranked = sorted(pts, key=lambda p: p["gain"])
    labelled = ranked[:4] + ranked[-4:]
    ys = spread([ax.y(p["gain"]) for p in labelled], gap=13)
    label_y = {id(p): y for p, y in zip(labelled, ys)}
    for p in pts:
        cx, cy = ax.x(p["ratio"]), ax.y(p["gain"])
        tip = (f'{p["dataset"]} ({TYPES[p["type"]]}, {p["n"]} cases): Strands {p["acc_big"]:.1f}% at {p["lat_big"]:.0f} ms, '
               f'Laya typed {p["acc_small"]:.1f}% at {p["lat_small"]:.0f} ms')
        s.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5.5" fill="{TYPE_VAR[p["type"]]}" class="dot"><title>{esc(tip)}</title></circle>')
        if id(p) in label_y:
            anchor, dx = ("end", -10) if cx > ax.x(6.0) else ("start", 10)
            s.append(f'<text class="val" x="{cx + dx:.1f}" y="{label_y[id(p)] + 4:.1f}" text-anchor="{anchor}">{esc(p["dataset"].replace("_", " "))} ({esc(TYPES[p["type"]])}) {p["gain"]:+.0f}</text>')
    s.append('</svg>')
    return "\n".join(s), pts


# ------------------------------------------------------------------------------ preliminary 4-size check
# model key -> (label, size in B, marker shape). Sizes are nominal: Laya 421M; TEV1 0.8B and 4B from the Ollama
# model names (ollama show reports 4.2B for tev1:latest); Strands Decider 2B from its name.
PRELIM = {
    "laya_en": ("Laya English", 0.421, "ring"), "laya_td": ("Laya typed-decisions", 0.421, "circle"),
    "tev08": ("TEV1 0.8B", 0.8, "tri"), "strands": ("Strands Decider 2B", 2.0, "square"), "tev4": ("TEV1 4B", 4.2, "tri"),
}
PRELIM_DATASETS = ["banking77", "decision-bench", "jevbench-noul", "jevbench-score"]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion, in percent."""
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return 100 * (centre - half), 100 * (centre + half)


def prelim_collect() -> dict:
    tev = {k: load(CUSTOM / "tev_smoke_40cases" / f"{f}.json.gz")[0] for k, f in (("tev08", "tev-0.8b"), ("tev4", "tev"))}
    ids = {d: [x["id"] for x in tev["tev4"][d]["results"]] for d in PRELIM_DATASETS}
    out = {}
    runs_by = {"laya_en": load(CUSTOM / "laya-english.json.gz"), "laya_td": load(CUSTOM / "laya-typed-decisions.json.gz"),
               "strands": load(CUSTOM / "strands-decision.json.gz"), "tev08": [tev["tev08"]], "tev4": [tev["tev4"]]}
    for key, runs in runs_by.items():
        per_run = []
        for run in runs:
            cases = []
            for d in PRELIM_DATASETS:
                got = {x["id"]: x for x in run[d]["results"]}
                cases += [got[i] for i in ids[d]]            # the same 40 case ids for every model
            per_run.append(cases)
        correct = [sum(1 for c in cases if c["ok"]) for cases in per_run]
        assert len(set(correct)) == 1, f"{key}: answers differ between runs"
        lat = [st.mean(c["ms"] for c in cases) for cases in per_run]
        k, n = correct[0], len(per_run[0])
        lo, hi = wilson(k, n)
        label, size, shape = PRELIM[key]
        out[key] = {"key": key, "label": label, "size": size, "shape": shape, "k": k, "n": n, "acc": 100 * k / n, "lo": lo, "hi": hi,
                    "lat": min(lat), "runs": len(runs), "errors": sum(1 for c in per_run[0] if c["err"])}
    return out


def marker(shape: str, cx: float, cy: float, tip: str) -> str:
    t = f"<title>{esc(tip)}</title>"
    if shape == "ring":
        return f'<circle class="ring" cx="{cx:.1f}" cy="{cy:.1f}" r="6" stroke="var(--ink)">{t}</circle>'
    if shape == "circle":
        return f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6" fill="var(--ink)">{t}</circle>'
    if shape == "square":
        return f'<rect x="{cx - 5.5:.1f}" y="{cy - 5.5:.1f}" width="11" height="11" fill="var(--ink)">{t}</rect>'
    return f'<polygon points="{cx:.1f},{cy - 7:.1f} {cx + 6.5:.1f},{cy + 5:.1f} {cx - 6.5:.1f},{cy + 5:.1f}" fill="var(--ink)">{t}</polygon>'


def prelim_chart(pc: dict, metric: str, ylabel: str, y0, y1, yticks, yfmt, title: str, desc: str, with_ci: bool) -> str:
    W, H = 640, 400
    ax = Axes(W, H, 62, 24, 20, 70, 0.2, 6.0, y0, y1)
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-labelledby="t-{metric}p d-{metric}p">',
         f'<title id="t-{metric}p">{esc(title)}</title><desc id="d-{metric}p">{esc(desc)}</desc>']
    s += frame(ax, [0.5, 1, 2, 4], lambda v: f"{v:g}B", yticks, yfmt, ylabel, "model size (billions of parameters, log scale)")
    items = sorted(pc.values(), key=lambda m: (m["size"], m["label"]))
    short = {"laya_en": "Laya English", "laya_td": "Laya typed", "tev08": "TEV1 0.8B", "strands": "Strands 2B", "tev4": "TEV1 4B"}
    # models of the same size (the two Laya checkpoints) are nudged apart sideways so both stay visible
    by_size = collections.defaultdict(list)
    for m in items:
        by_size[m["size"]].append(m)
    shift = {}
    for group in by_size.values():
        for k, m in enumerate(sorted(group, key=lambda m: m["label"])):
            shift[m["key"]] = 0 if len(group) == 1 else (-11 if k == 0 else 11)
    for m in items:
        cx, cy = ax.x(m["size"]) + shift[m["key"]], ax.y(m[metric])
        if with_ci:
            s.append(f'<line class="ci" x1="{cx:.1f}" x2="{cx:.1f}" y1="{ax.y(m["lo"]):.1f}" y2="{ax.y(m["hi"]):.1f}"/>')
            for yy in (m["lo"], m["hi"]):
                s.append(f'<line class="ci" x1="{cx - 5:.1f}" x2="{cx + 5:.1f}" y1="{ax.y(yy):.1f}" y2="{ax.y(yy):.1f}"/>')
        tip = f'{m["label"]}: {m["k"]}/{m["n"]} correct ({m["acc"]:.1f}%, 95% interval {m["lo"]:.0f}-{m["hi"]:.0f}%), {m["lat"]:.0f} ms per case'
        s.append(marker(m["shape"], cx, cy, tip))
        value = f'{m["acc"]:.0f}%  ({m["lo"]:.0f}-{m["hi"]:.0f})' if metric == "acc" else f'{m["lat"]:.0f} ms'
        below = m["key"] == "laya_en"
        top = ax.y(m["hi"]) if with_ci else cy
        bottom = ax.y(m["lo"]) if with_ci else cy
        if with_ci:
            y1, y2 = (bottom + 17, bottom + 31) if below else (top - 22, top - 8)
            lx, anchor = cx, "middle"
        else:  # no bars: both labels above the markers; the Laya pair is split left / right
            y1, y2 = top - 22, top - 8
            lx, anchor = {"laya_en": (cx - 9, "end"), "laya_td": (cx + 9, "start")}.get(m["key"], (cx, "middle"))
        s.append(f'<text class="lbl" x="{lx:.1f}" y="{y1:.1f}" text-anchor="{anchor}">{esc(short[m["key"]])}</text>')
        s.append(f'<text class="val" x="{lx:.1f}" y="{y2:.1f}" text-anchor="{anchor}">{esc(value)}</text>')
    s.append("</svg>")
    return "\n".join(s)


# ------------------------------------------------------------------------------ page
CSS = """
:root { color-scheme: light; --bg:#fcfcfb; --card:#fff; --ink:#0b0b0b; --muted:#52514e; --line:#e4e2dd; --grid:#ecebe7;
        --c1:#2a78d6; --c2:#eb6834; --c3:#1baf7a; --band:rgba(120,113,108,.10); }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { color-scheme: dark; --bg:#1a1a19; --card:#222220; --ink:#ffffff;
        --muted:#c3c2b7; --line:#3a3936; --grid:#2e2d2b; --c1:#3987e5; --c2:#d95926; --c3:#199e70; --band:rgba(200,196,185,.08); } }
:root[data-theme="dark"] { color-scheme: dark; --bg:#1a1a19; --card:#222220; --ink:#ffffff; --muted:#c3c2b7; --line:#3a3936;
        --grid:#2e2d2b; --c1:#3987e5; --c2:#d95926; --c3:#199e70; --band:rgba(200,196,185,.08); }
body { margin:0; padding:28px 16px 48px; background:var(--bg); color:var(--ink); font:15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width:1180px; margin:0 auto; }
h1 { font-size:24px; margin:0 0 6px; } h2 { font-size:17px; margin:0 0 4px; }
.sub { color:var(--muted); margin:0 0 20px; max-width:80ch; }
.takeaways { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:14px 20px; margin-bottom:20px; }
.takeaways ul { margin:6px 0 0; padding-left:20px; } .takeaways li { margin:4px 0; }
.grid2 { display:grid; grid-template-columns:repeat(auto-fit, minmax(420px, 1fr)); gap:16px; }
figure { margin:0; background:var(--card); border:1px solid var(--line); border-radius:12px; padding:14px 14px 10px; }
figcaption { color:var(--muted); font-size:13px; margin:2px 4px 6px; }
svg { width:100%; height:auto; display:block; }
.grid { stroke:var(--grid); stroke-width:1; } .axis { stroke:var(--muted); stroke-width:1; } .tickmark { stroke:var(--muted); }
.zero { stroke:var(--ink); stroke-width:1.5; stroke-dasharray:5 4; opacity:.55; }
.tick { fill:var(--muted); font-size:13px; } .axislabel { fill:var(--muted); font-size:13px; }
.val { fill:var(--ink); font-size:13px; } .note { fill:var(--muted); font-size:12px; } .modelname { fill:var(--ink); font-size:13px; font-weight:600; }
.lbl { fill:var(--ink); font-size:13px; font-weight:600; } .nodata { fill:var(--band); } .ci { stroke:var(--muted); stroke-width:1.6; } .ring { fill:var(--card); stroke-width:2; } .dot { stroke:var(--card); stroke-width:1.5; }
.legend { display:flex; flex-wrap:wrap; gap:6px 18px; font-size:13px; color:var(--muted); margin:10px 4px 0; }
.legend i { display:inline-block; width:11px; height:11px; border-radius:50%; margin-right:6px; vertical-align:-1px; }
.legend i.hollow { background:var(--card); border:2px solid var(--muted); width:9px; height:9px; }
details { margin-top:20px; background:var(--card); border:1px solid var(--line); border-radius:12px; padding:10px 16px; }
summary { cursor:pointer; font-weight:600; }
table { border-collapse:collapse; width:100%; margin-top:10px; font-size:14px; font-variant-numeric:tabular-nums; }
th, td { text-align:right; padding:6px 10px; border-bottom:1px solid var(--line); } th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) { text-align:left; }
.caveats { margin-top:20px; color:var(--muted); font-size:14px; } .caveats li { margin:4px 0; }
"""


def size_text(b: float) -> str:
    return f"{b * 1000:.0f}M" if b < 1 else f"~{b:g}B"


def build(data: dict) -> str:
    small, big, eng = data["laya_td"], data["strands"], data["laya_en"]
    ratio_p = big["params"] / small["params"]
    d_acc = big["overall"]["acc"] - small["overall"]["acc"]
    lat_x = big["overall"]["lat"] / small["overall"]["lat"]
    per_type = {t: big["by_type"][t]["acc"] - small["by_type"][t]["acc"] for t in TYPES}
    same_size_gap = small["overall"]["acc"] - eng["overall"]["acc"]
    chart_a = size_chart(
        data, "acc", "accuracy (% correct)", 30, 90, [30, 40, 50, 60, 70, 80, 90], lambda v: f"{v:g}%", lambda v: f"{v:.1f}%",
        "Accuracy against model size",
        f"From the 421M Laya model to the roughly 2B Strands Decider, overall accuracy goes from {small['overall']['acc']:.1f}% to {big['overall']['acc']:.1f}%, "
        f"with different directions for each question type.")
    chart_b = size_chart(
        data, "lat", "time per case (ms, end to end)", 0, 160, [0, 40, 80, 120, 160], lambda v: f"{v:g}", lambda v: f"{v:.0f} ms",
        "Time per case against model size",
        f"From 421M to about 2B parameters, time per case goes from {small['overall']['lat']:.0f} ms to {big['overall']['lat']:.0f} ms.")
    chart_c, pts = scatter_chart(data)
    pc = prelim_collect()
    chart_d = prelim_chart(pc, "acc", "accuracy on the same 40 cases (% correct)", 0, 100, [0, 20, 40, 60, 80, 100], lambda v: f"{v:g}%",
                           "Accuracy against size, four sizes, 40 cases",
                           "Preliminary: five models on the same 40 cases, with 95% intervals that are wide because there are only 40 cases.", True)
    chart_e = prelim_chart(pc, "lat", "time per case (ms)", 0, 500, [0, 100, 200, 300, 400, 500], lambda v: f"{v:g}",
                           "Time per case against size, four sizes, 40 cases",
                           "Preliminary: time per case for the same five models.", False)
    prelim_rows = "".join(
        f"<tr><td>{esc(m['label'])}</td><td>{size_text(m['size'])}</td><td>{m['k']}/{m['n']}</td><td>{m['acc']:.1f}%</td>"
        f"<td>{m['lo']:.0f}-{m['hi']:.0f}%</td><td>{m['lat']:.0f} ms</td><td>{m['errors']}</td><td>{m['runs']}</td></tr>"
        for m in sorted(pc.values(), key=lambda m: (m["size"], m["label"])))
    n_gain = sum(1 for p in pts if p["gain"] > 0)
    rows = []
    for key in ("laya_en", "laya_td", "strands"):
        m = data[key]
        for name, grp in (("overall", m["overall"]), *((TYPES[t], m["by_type"][t]) for t in TYPES)):
            rows.append(f"<tr><td>{esc(m['label'])}</td><td>{esc(name)}</td><td>{size_text(m['params'])}</td>"
                        f"<td>{grp['acc']:.1f}%</td><td>{grp['lat']:.0f} ms</td><td>{grp['n']:,}</td></tr>")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Model size vs performance</title><style>{CSS}</style></head>
<body><main>
<h1>Does a bigger decision model help?</h1>
<p class="sub">Model size against accuracy and speed, from the official JevBench harness (all 2,934 cases, 3 runs per model, AC power, local
server). Latency is end to end: HTTP round trip plus the model. Only two sizes were benchmarked properly, so this shows what the data
supports and nothing beyond it.</p>

<section class="takeaways" aria-label="Key findings"><h2>What the data says</h2><ul>
<li><b>Speed follows size.</b> {ratio_p:.1f}x more parameters (421M to ~2B) is {lat_x:.1f}x slower per case ({small['overall']['lat']:.0f} to {big['overall']['lat']:.0f} ms).
 Both Laya checkpoints, which have the same size, run at the same speed.</li>
<li><b>Accuracy does not simply follow size.</b> The larger model gains {d_acc:+.1f} points overall ({small['overall']['acc']:.1f}% to {big['overall']['acc']:.1f}%), but unevenly:
 choice {per_type['choice']:+.1f}, yes/no {per_type['noul']:+.1f}, score {per_type['score']:+.1f} points. It is more accurate on only {n_gain} of {len(pts)} datasets.</li>
<li><b>Training matters more than size here.</b> At the same size (421M) the two Laya checkpoints differ by {same_size_gap:.1f} points overall
 ({eng['overall']['acc']:.1f}% against {small['overall']['acc']:.1f}%), more than the {d_acc:.1f}-point gap that {ratio_p:.1f}x more parameters buys.</li>
</ul></section>

<div class="grid2">
<figure>{chart_a}<figcaption><b>A. Accuracy against size.</b> Each line is one question type, from the best small model to the larger model.
Hollow rings: Laya English (same 421M size, different training).</figcaption></figure>
<figure>{chart_b}<figcaption><b>B. Time per case against size.</b> Same layout. The rings sit on the dots: same size, same speed.</figcaption></figure>
</div>

<figure style="margin-top:16px">{chart_c}<figcaption><b>C. What the extra size buys, per dataset.</b> Each dot is one of JevBench's 22 datasets (Strands Decider ~2B compared
with Laya typed-decisions 421M). Right = a bigger slowdown; up = a bigger accuracy gain. Hover a dot for the numbers.</figcaption>
<div class="legend"><span><i style="background:var(--c1)"></i>choice</span><span><i style="background:var(--c2)"></i>yes/no</span>
<span><i style="background:var(--c3)"></i>score</span></div></figure>

<h2 style="margin-top:28px">Preliminary: four sizes on the same 40 cases</h2>
<p class="sub">TEV1 0.8B and 4B were only checked on 40 cases (the first 10 of banking77, decision-bench, jevbench-noul and jevbench-score), so this section is a
rough look, not a result. Intervals show how wide the uncertainty is. Nimble 9B was left out; a full 3-run test of the TEV models was started and
stopped before it finished.</p>
<div class="grid2">
<figure>{chart_d}<figcaption><b>D. Accuracy against size.</b> Bars are 95% intervals. Shapes: circle = Laya (hollow = English), square = Strands Decider, triangle = TEV.</figcaption></figure>
<figure>{chart_e}<figcaption><b>E. Time per case against size.</b> Strands and Laya: fastest of their five earlier runs (best case, because the laptop was
partly on battery). TEV: one run on AC power with a cool GPU. All include the HTTP round trip.</figcaption></figure>
</div>
<details><summary>Preliminary data table</summary>
<table><thead><tr><th>Model</th><th>Size</th><th>Correct</th><th>Accuracy</th><th>95% interval</th><th>Time per case</th><th>Errors</th><th>Runs</th></tr></thead>
<tbody>{prelim_rows}</tbody></table>
<p class="sub" style="margin-top:8px">Errors: cases the model refused. Both TEV models reject inputs over 2,050 tokens (one Decision Bench case here).</p></details>

<details open><summary>Data table</summary>
<table><thead><tr><th>Model</th><th>Question type</th><th>Size</th><th>Accuracy</th><th>Time per case</th><th>Cases</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></details>

<ul class="caveats"><li><b>Only two sizes benchmarked properly.</b> Laya (421M) and Strands Decider (~2B) have full results (charts A to C). TEV 0.8B and 4B have only the 40-case check (D and E), and Nimble 9B was left out.</li>
<li><b>Size is confounded.</b> The two models also differ in design (a small encoder against a language-model decoder), training data and how they read inputs, so the gaps are not only about parameter count.</li>
<li><b>Latency is end to end</b> through a local HTTP server and was measured on a laptop GPU that throttles with heat (runs of the larger model drifted from 316 to 344 s), so treat speeds as approximate.</li>
<li>Parameter counts are nominal, taken from the model names.</li></ul>
</main></body></html>
"""


def main() -> None:
    data = collect()
    OUT.write_text(build(data))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
