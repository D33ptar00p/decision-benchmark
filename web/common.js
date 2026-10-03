const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
let LABELS = {};
const label = m => LABELS[m] || m;
const fmt = ms => ms >= 1000 ? (ms / 1000).toFixed(2) + " s" : ms.toFixed(0) + " ms";

const KNOWN = ["strands-decision", "nimble", "tev", "tev-0-8b", "laya"];
const color = m => { const k = m.replace(/[^a-z0-9-]/g, "-"); return `var(--s-${KNOWN.includes(k) ? k : "other"})`; };

// One column per item {label, value, color}; `max` fixes the scale (e.g. 100 for percentages).
function chart(title, items, fmtValue, max) {
  const top = max ?? Math.max(...items.map(i => i.value), 1);
  const cols = items.map(i => `<div class="col" title="${esc(i.label)}: ${fmtValue(i.value)}">
      <span class="val">${fmtValue(i.value)}</span>
      <div class="mark" style="height:${Math.max(i.value / top, 0) * 100 * 0.8}%;--c:${i.color}"></div></div>`).join("");
  const names = items.map(i => `<span><i class="key" style="--c:${i.color}"></i>${esc(i.label)}</span>`).join("");
  return `<div class="chart"><h3>${title}</h3><div class="cols">${cols}</div><div class="names">${names}</div></div>`;
}

async function loadModels() {
  const ms = await fetch("/api/models").then(r => r.json());
  LABELS = Object.fromEntries(ms.map(m => [m.name, m.label]));
  return ms;
}

// Score tasks add MAE and QWK (over their ordinal cases); other tasks leave the columns out.
const hasOrdinal = rows => rows.some(r => r.mae != null || r.qwk != null);
const ordinalHead = rows => hasOrdinal(rows)
  ? `<th title="Mean absolute error in levels; lower is better (ordinal cases only)">MAE</th><th title="Quadratic weighted kappa: 1 = perfect, 0 = chance (ordinal cases only)">QWK</th>` : "";
const ordinalCells = (r, rows) => hasOrdinal(rows)
  ? `<td>${r.mae == null ? "-" : r.mae.toFixed(2)}</td><td>${r.qwk == null ? "-" : r.qwk.toFixed(2)}</td>` : "";
