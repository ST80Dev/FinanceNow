/* FinanceNow — dashboard statica (vanilla JS, nessuna dipendenza).
 * Legge solo i JSON in data/ prodotti da pipeline/export.py.
 * Viste (routing via hash): #/  panoramica · #/t/<id>  strumento · #/pattern · #/modelli · #/metodo
 */
"use strict";

const REGIMES = ["calmo", "normale", "turbolento"];
const ANOMALY_LABELS = {
  rendimento: "Rendimento",
  gap: "Gap di apertura",
  range: "Escursione high-low",
  volume: "Volume",
  rendimento_vs_bench: "Rendimento vs benchmark",
};
const MODEL_LABELS = {
  rw: "rw (baseline)",
  hist_mean: "media storica (baseline)",
  har: "HAR-RV",
  garch: "GARCH(1,1)-t",
  egarch: "EGARCH(1,1)-t",
};
const CLASS_LABELS = { index: "Indice", etf: "ETF", stock: "Azione" };

const state = { index: null, models: null, digest: null, patterns: null, tickers: {}, sort: { key: null, dir: -1 }, horizon: "h1" };
const app = document.getElementById("app");
const tooltip = document.getElementById("tooltip");

/* ------------------------------------------------------------------ utilità */

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
const isNum = (v) => typeof v === "number" && Number.isFinite(v);
function pct(v, digits = 1, signed = false) {
  if (!isNum(v)) return "–";
  const s = (v * 100).toFixed(digits).replace(".", ",") + "%";
  return signed && v > 0 ? "+" + s : s;
}
function num(v, digits = 2, signed = false) {
  if (!isNum(v)) return "–";
  const s = v.toFixed(digits).replace(".", ",");
  return signed && v > 0 ? "+" + s : s;
}
function signClass(v) { return !isNum(v) ? "" : v > 0 ? "pos" : v < 0 ? "neg" : ""; }
function fmtDate(iso) {
  if (!iso) return "–";
  const [y, m, d] = iso.slice(0, 10).split("-");
  return `${d}/${m}/${y}`;
}
function badge(regime) {
  return regime ? `<span class="badge ${esc(regime)}">${esc(regime)}</span>` : `<span class="badge none">n.d.</span>`;
}

async function getJSON(path) {
  const res = await fetch(path, { cache: "no-cache" });
  if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
  return res.json();
}

/* ------------------------------------------------------------------ grafici SVG */

const SVGNS = "http://www.w3.org/2000/svg";
function el(tag, attrs = {}, parent) {
  const n = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (parent) parent.appendChild(n);
  return n;
}

function niceTicks(min, max, count = 4) {
  if (min === max) { min -= 1; max += 1; }
  const raw = (max - min) / count;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) || raw;
  const ticks = [];
  for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-9; v += step) ticks.push(+v.toFixed(10));
  return ticks;
}

/**
 * Grafico a linea su indice di seduta (niente buchi per weekend/festivi).
 * opts: dates, values, format(v), bands (stati di regime per seduta), zero (bool), area (bool),
 *       yMin/yMax fissi, label (per tooltip e accessibilità).
 */
function lineChart(container, opts) {
  container.innerHTML = "";
  const width = Math.max(280, container.clientWidth || 600);
  const height = width < 500 ? 190 : 230;
  const m = { top: 10, right: 12, bottom: 24, left: 52 };
  const iw = width - m.left - m.right;
  const ih = height - m.top - m.bottom;
  const { dates, values } = opts;
  const n = values.length;
  const svg = el("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": opts.label || "grafico" }, container);
  if (!n) return;

  const finite = values.filter(isNum);
  let lo = opts.yMin ?? Math.min(...finite);
  let hi = opts.yMax ?? Math.max(...finite);
  if (opts.zero) { lo = Math.min(lo, 0); hi = Math.max(hi, 0); }
  if (!finite.length) { lo = 0; hi = 1; }
  const pad = (hi - lo) * 0.06 || 0.01;
  if (opts.yMin === undefined) lo -= pad;
  if (opts.yMax === undefined) hi += pad;
  const x = (i) => m.left + (n === 1 ? iw / 2 : (i / (n - 1)) * iw);
  const y = (v) => m.top + ih - ((v - lo) / (hi - lo)) * ih;

  // Bande di regime
  if (opts.bands) {
    const g = el("g", {}, svg);
    let start = 0;
    for (let i = 1; i <= n; i++) {
      if (i === n || opts.bands[i] !== opts.bands[start]) {
        const s = opts.bands[start];
        if (s !== null && s !== undefined) {
          const x0 = start === 0 ? m.left : (x(start - 1) + x(start)) / 2;
          const x1 = i === n ? m.left + iw : (x(i - 1) + x(i)) / 2;
          el("rect", { class: `band s${s}`, x: x0, y: m.top, width: Math.max(0.5, x1 - x0), height: ih }, g);
        }
        start = i;
      }
    }
  }

  // Griglia e asse Y
  const axis = el("g", { class: "axis" }, svg);
  for (const t of niceTicks(lo, hi)) {
    if (t < lo || t > hi) continue;
    el("line", { class: "gridline", x1: m.left, x2: m.left + iw, y1: y(t), y2: y(t) }, axis);
    el("text", { x: m.left - 6, y: y(t) + 4, "text-anchor": "end" }, axis).textContent = opts.format(t);
  }
  if (opts.zero && lo < 0 && hi > 0) el("line", { class: "zero", x1: m.left, x2: m.left + iw, y1: y(0), y2: y(0) }, svg);

  // Asse X: anni, o mesi su intervalli brevi
  const short = n < 400;
  let lastLabelX = -Infinity;
  for (let i = 0; i < n; i++) {
    const d = dates[i];
    const prev = dates[i - 1];
    const isTick = i > 0 && (short ? d.slice(0, 7) !== prev.slice(0, 7) && ["01", "04", "07", "10"].includes(d.slice(5, 7))
                                   : d.slice(0, 4) !== prev.slice(0, 4));
    if (!isTick) continue;
    const xi = x(i);
    if (xi - lastLabelX < 46) continue;
    lastLabelX = xi;
    el("line", { class: "gridline", x1: xi, x2: xi, y1: m.top, y2: m.top + ih }, axis);
    el("text", { x: xi, y: height - 6, "text-anchor": "middle" }, axis).textContent =
      short ? `${d.slice(5, 7)}/${d.slice(2, 4)}` : d.slice(0, 4);
  }

  // Serie (interrotta sui valori mancanti)
  let dPath = "", areaPath = "", open = false, segStart = 0;
  const base = y(opts.zero ? 0 : lo);
  for (let i = 0; i < n; i++) {
    const v = values[i];
    if (!isNum(v)) {
      if (open && opts.area) areaPath += `L${x(i - 1)},${base}L${x(segStart)},${base}Z`;
      open = false;
      continue;
    }
    if (!open) { dPath += `M${x(i)},${y(v)}`; areaPath += `M${x(i)},${y(v)}`; segStart = i; open = true; }
    else { dPath += `L${x(i)},${y(v)}`; areaPath += `L${x(i)},${y(v)}`; }
  }
  if (open && opts.area) areaPath += `L${x(n - 1)},${base}L${x(segStart)},${base}Z`;
  if (opts.area) el("path", { class: "area", d: areaPath }, svg);
  el("path", { class: "series", d: dPath }, svg);

  // Cursore e tooltip
  const cursor = el("line", { class: "cursor", y1: m.top, y2: m.top + ih, visibility: "hidden" }, svg);
  const dot = el("circle", { class: "dot", r: 3, visibility: "hidden" }, svg);
  const hit = el("rect", { x: m.left, y: m.top, width: iw, height: ih, fill: "transparent" }, svg);
  const show = (evt) => {
    const rect = svg.getBoundingClientRect();
    const px = ((evt.clientX - rect.left) / rect.width) * width;
    const i = Math.max(0, Math.min(n - 1, Math.round(((px - m.left) / iw) * (n - 1))));
    const v = values[i];
    cursor.setAttribute("x1", x(i)); cursor.setAttribute("x2", x(i)); cursor.setAttribute("visibility", "visible");
    if (isNum(v)) { dot.setAttribute("cx", x(i)); dot.setAttribute("cy", y(v)); dot.setAttribute("visibility", "visible"); }
    else dot.setAttribute("visibility", "hidden");
    const reg = opts.bands && opts.bands[i] !== null && opts.bands[i] !== undefined ? ` · ${badge(REGIMES[opts.bands[i]])}` : "";
    tooltip.innerHTML = `<strong>${fmtDate(dates[i])}</strong><br>${esc(opts.label)}: <span class="num">${opts.format(v)}</span>${reg}`;
    tooltip.hidden = false;
    const tx = Math.min(evt.clientX + 14, window.innerWidth - tooltip.offsetWidth - 8);
    tooltip.style.left = `${Math.max(8, tx)}px`;
    tooltip.style.top = `${evt.clientY + 14}px`;
  };
  const hide = () => { cursor.setAttribute("visibility", "hidden"); dot.setAttribute("visibility", "hidden"); tooltip.hidden = true; };
  hit.addEventListener("pointermove", show);
  hit.addEventListener("pointerdown", show);
  hit.addEventListener("pointerleave", hide);
}

/* ------------------------------------------------------------------ stato e intestazione */

function renderStatus() {
  const idx = state.index;
  const box = document.getElementById("status");
  if (!idx) { box.innerHTML = ""; return; }
  const parts = [`Dati al <strong>${fmtDate(idx.as_of)}</strong>`,
                 `aggiornato ${fmtDate(idx.generated)}`, `modelli v${esc(idx.model_version)}`];
  const ds = idx.data_status || {};
  if (ds.missing?.length) parts.push(`<span class="warn">Senza dati: ${esc(ds.missing.join(", "))}</span>`);
  if (ds.stale?.length) parts.push(`<span class="warn">Non aggiornati: ${esc(ds.stale.join(", "))}</span>`);
  box.innerHTML = parts.map((p) => `<span>${p}</span>`).join("");
}

function setActiveTab(tab) {
  document.querySelectorAll(".tabs a").forEach((a) => a.classList.toggle("active", a.dataset.tab === tab));
}

/* ------------------------------------------------------------------ panoramica */

const COLUMNS = [
  { key: "name", label: "Strumento", left: true, render: (r) => `<span class="name">${esc(r.name)}<small>${esc(r.id.toUpperCase())} · ${esc(CLASS_LABELS[r.asset_class] || r.asset_class)}</small></span>` },
  { key: "regime_rank", label: "Regime", left: true, render: (r) => `${badge(r.regime)} <span class="muted small num">${pct(r.regime_prob, 0)}</span>` },
  { key: "days_in_state", label: "Da (sedute)", render: (r) => `<span class="num">${r.days_in_state ?? "–"}</span>` },
  { key: "rv_21", label: "Vol. 21g", render: (r) => `<span class="num">${pct(r.rv_21)}</span>` },
  { key: "drawdown", label: "Drawdown", render: (r) => `<span class="num ${signClass(r.drawdown)}">${pct(r.drawdown)}</span>` },
  { key: "mom_3m", label: "Mom. 3m", render: (r) => `<span class="num ${signClass(r.mom_3m)}">${pct(r.mom_3m, 1, true)}</span>` },
  { key: "mom_12m", label: "Mom. 12m", render: (r) => `<span class="num ${signClass(r.mom_12m)}">${pct(r.mom_12m, 1, true)}</span>` },
  { key: "beta_252", label: "Beta 1a", render: (r) => `<span class="num">${num(r.beta_252)}</span>` },
  { key: "anomalies_12m", label: "Anomalie 12m", render: (r) => `<span class="num">${r.anomalies_12m ?? "–"}</span>` },
  { key: "last_anomaly_date", label: "Ultima anomalia", left: true, render: (r) => r.last_anomaly ? `<span class="name small">${fmtDate(r.last_anomaly.date)}<small>${esc(ANOMALY_LABELS[r.last_anomaly.kind] || r.last_anomaly.kind)}</small></span>` : "–" },
];

function sortValue(r, key) {
  if (key === "regime_rank") return r.regime ? REGIMES.indexOf(r.regime) + (r.regime_prob || 0) / 10 : -1;
  if (key === "last_anomaly_date") return r.last_anomaly?.date || "";
  return r[key];
}

function renderOverview() {
  setActiveTab("overview");
  const rows = [...state.index.instruments];
  const { key, dir } = state.sort;
  if (key) {
    rows.sort((a, b) => {
      const va = sortValue(a, key), vb = sortValue(b, key);
      if (va === vb) return 0;
      if (va === null || va === undefined) return 1;
      if (vb === null || vb === undefined) return -1;
      return (va > vb ? 1 : -1) * dir;
    });
  }
  const counts = REGIMES.map((g) => [g, rows.filter((r) => r.regime === g).length]);
  app.innerHTML = `
    ${digestBlock(state.digest)}
    <h2>Tutti gli strumenti</h2>
    <p class="muted">Stato attuale di ogni strumento. Clicca una riga per il dettaglio; clicca un'intestazione per ordinare.</p>
    <div class="counts">${counts.map(([g, c]) => `${badge(g)} <span class="small">${c}</span>`).join("&nbsp;&nbsp;")}</div>
    <div class="table-scroll"><table>
      <thead><tr>${COLUMNS.map((c) => `<th class="sortable ${c.left ? "left" : ""} ${key === c.key ? "sorted" + (dir > 0 ? " asc" : "") : ""}" data-key="${c.key}">${c.label}</th>`).join("")}</tr></thead>
      <tbody>${rows.map((r) => `<tr class="clickable" data-id="${esc(r.id)}">${COLUMNS.map((c) => `<td class="${c.left ? "left" : ""}">${c.render(r)}</td>`).join("")}</tr>`).join("")}</tbody>
    </table></div>
    <p class="muted small" style="margin-top:10px">Volatilità annualizzate. Momentum = somma dei log-rendimenti. Beta rispetto al Nasdaq 100 su 252 sedute.
      Le anomalie sono movimenti con z-score robusto ≥ 4 rispetto all'ultimo anno.</p>`;
  app.querySelectorAll("th.sortable").forEach((th) => th.addEventListener("click", () => {
    const k = th.dataset.key;
    state.sort = { key: k, dir: state.sort.key === k ? -state.sort.dir : -1 };
    renderOverview();
  }));
  app.querySelectorAll("tr.clickable").forEach((tr) => tr.addEventListener("click", () => { location.hash = `#/t/${tr.dataset.id}`; }));
}

/* ------------------------------------------------------------------ sintesi */

const SEV_LABEL = { alta: "Priorità alta", media: "Da tenere d'occhio", bassa: "Nota" };

function digestBlock(dg) {
  if (!dg) return `<h1>Panoramica</h1>`;
  const cap = (t) => t.charAt(0).toUpperCase() + t.slice(1);
  const items = dg.instruments.map((inst) => `
    <li class="dg-item sev-${esc(inst.max_severity)}">
      <a class="dg-name" href="#/t/${esc(inst.id)}">${esc(inst.name)} <span class="muted">${esc(inst.id.toUpperCase())}</span></a>
      <ul>${inst.signals.map((s) => `<li><span class="sev-dot sev-${esc(s.severity)}" title="${esc(SEV_LABEL[s.severity])}"></span>${esc(cap(s.text))}</li>`).join("")}</ul>
      ${(inst.history || []).map((h) => `<p class="dg-history"><a href="#/pattern">${esc(cap(h.event))}</a> (${fmtDate(h.date)}): ${esc(h.text)}</p>`).join("")}
    </li>`).join("");
  return `
    <section class="digest">
      <h1>Sintesi del ${fmtDate(dg.as_of)}</h1>
      <p class="dg-headline">${esc(dg.headline)}</p>
      ${dg.market?.length ? `<ul class="dg-market">${dg.market.map((m) => `<li>${esc(m)}</li>`).join("")}</ul>` : ""}
      ${dg.missing?.length ? `<p class="warn">Senza dati: ${esc(dg.missing.join(", "))}</p>` : ""}
      ${items ? `<ul class="dg-list">${items}</ul>` : ""}
      ${dg.quiet?.length ? `<p class="muted small">Nessuna novità: ${esc(dg.quiet.map((q) => q.toUpperCase()).join(", "))}.</p>` : ""}
      <details class="small muted"><summary>Come vengono scelti i segnali</summary><pre class="dg-rules">${esc(dg.rules)}</pre></details>
    </section>`;
}

/* ------------------------------------------------------------------ dettaglio strumento */

const RANGES = [["1A", 252], ["3A", 756], ["5A", 1260], ["Tutto", Infinity]];

async function renderTicker(id) {
  setActiveTab("overview");
  if (!state.index.instruments.some((r) => r.id === id)) {
    app.innerHTML = `<a class="back" href="#/">← Panoramica</a><p>Strumento non trovato: ${esc(id)}</p>`;
    return;
  }
  app.innerHTML = `<p class="muted">Caricamento ${esc(id.toUpperCase())}…</p>`;
  const t = state.tickers[id] || (state.tickers[id] = await getJSON(`data/tickers/${encodeURIComponent(id)}.json`));
  const f = t.features_latest || {};
  const reg = t.regime || {};
  const live = t.live_latest;
  const fc = live?.vol_ann?.h1 || {};
  const probs = reg.probabilities || {};

  app.innerHTML = `
    <a class="back" href="#/">← Panoramica</a>
    <h1>${esc(t.name)} <span class="muted">${esc(t.id.toUpperCase())}</span></h1>
    <p class="muted">${esc(CLASS_LABELS[t.asset_class] || t.asset_class)}${t.benchmark ? ` · benchmark ${esc(t.benchmark.toUpperCase())}` : ""}
      · dati al ${fmtDate(t.as_of)} · storico dal ${fmtDate(t.data_period?.first)} (${t.data_period?.sessions ?? "–"} sedute)</p>

    <div class="cards">
      <div class="card">
        <h3>Regime</h3>
        <div>${badge(reg.current)} <span class="muted small">da ${reg.days_in_state ?? "–"} sedute</span></div>
        <div class="probs">${REGIMES.map((g) => `<i class="${g}" style="width:${(probs[g] || 0) * 100}%" title="${g} ${pct(probs[g], 0)}"></i>`).join("")}</div>
        <div class="small muted">${REGIMES.map((g) => `${g} ${pct(probs[g], 0)}`).join(" · ")}</div>
      </div>
      <div class="card">
        <h3>Volatilità (annualizzata)</h3>
        <div class="kv"><span>Realizzata 21 sedute</span><span class="num">${pct(f.rv_21)}</span></div>
        <div class="kv"><span>Realizzata 63 sedute</span><span class="num">${pct(f.rv_63)}</span></div>
        <div class="kv"><span>Garman-Klass 21</span><span class="num">${pct(f.gk_21)}</span></div>
        <div class="kv"><span><span class="series-tag live">LIVE</span>GARCH domani</span><span class="num">${pct(fc.garch)}</span></div>
        <div class="kv small muted"><span>baseline rw</span><span class="num">${pct(fc.rw)}</span></div>
      </div>
      <div class="card">
        <h3>Rischio</h3>
        <div class="kv"><span>Drawdown dal massimo</span><span class="num ${signClass(f.drawdown)}">${pct(f.drawdown)}</span></div>
        <div class="kv"><span>Beta 1 anno</span><span class="num">${num(f.beta_252)}</span></div>
        <div class="kv"><span>Correlazione 1 anno</span><span class="num">${num(f.corr_252)}</span></div>
        <div class="kv"><span>Asimmetria / curtosi 63g</span><span class="num">${num(f.skew_63)} / ${num(f.kurt_63)}</span></div>
      </div>
      <div class="card">
        <h3>Momentum e trend</h3>
        <div class="kv"><span>1 / 3 mesi</span><span class="num"><span class="${signClass(f.mom_1m)}">${pct(f.mom_1m, 1, true)}</span> / <span class="${signClass(f.mom_3m)}">${pct(f.mom_3m, 1, true)}</span></span></div>
        <div class="kv"><span>6 / 12 mesi</span><span class="num"><span class="${signClass(f.mom_6m)}">${pct(f.mom_6m, 1, true)}</span> / <span class="${signClass(f.mom_12m)}">${pct(f.mom_12m, 1, true)}</span></span></div>
        <div class="kv"><span>Distanza da MA 200</span><span class="num ${signClass(f.dist_ma200)}">${pct(f.dist_ma200, 1, true)}</span></div>
        ${t.benchmark ? `<div class="kv"><span>Forza relativa 12m</span><span class="num ${signClass(f.rs_12m)}">${pct(f.rs_12m, 1, true)}</span></div>` : ""}
      </div>
    </div>
    ${live ? `<p class="muted small"><span class="series-tag live">LIVE</span>Previsione registrata la sera del ${fmtDate(live.origin)} (modelli v${esc(live.model_version)}): volatilità attesa per la seduta successiva.</p>` : ""}

    <div class="chart-head" style="margin-top:14px">
      <h2 style="margin:0">Storico</h2>
      <div class="ranges" id="ranges">${RANGES.map(([l, d], i) => `<button data-days="${d}" class="${i === 1 ? "active" : ""}">${l}</button>`).join("")}</div>
    </div>
    <div class="legend">${REGIMES.map((g, i) => `<span><i class="band-legend s${i}" style="background:var(--${g}-bg);border:1px solid var(--${g})"></i>${g}</span>`).join("")}<span>sfondo = regime stimato quel giorno</span></div>
    <div id="charts"></div>

    <h2>Regimi: statistiche</h2>
    ${regimeTable(reg)}

    <h2>Anomalie recenti</h2>
    ${anomalyTable(t.anomalies)}`;

  const charts = [
    { key: "rv_21", label: "Volatilità realizzata 21 sedute", fmt: (v) => pct(v, 0), bands: true },
    { key: "drawdown", label: "Drawdown dal massimo", fmt: (v) => pct(v, 0), area: true, zero: true },
    { key: "p_turbolento", label: "Probabilità regime turbolento", fmt: (v) => pct(v, 0), yMin: 0, yMax: 1, area: true },
    { key: "beta_252", label: "Beta 1 anno vs benchmark", fmt: (v) => num(v, 2), zero: false },
  ].filter((c) => (t.series[c.key] || []).some(isNum));

  const box = document.getElementById("charts");
  box.innerHTML = charts.map((c, i) => `<div class="chart-card"><div class="chart-head"><h3>${c.label}</h3></div><div class="chart" id="chart-${i}"></div></div>`).join("");
  let days = 756;
  const draw = () => {
    const n = t.dates.length;
    const from = Math.max(0, n - Math.min(days, n));
    const slice = (arr) => arr.slice(from);
    charts.forEach((c, i) => lineChart(document.getElementById(`chart-${i}`), {
      dates: slice(t.dates), values: slice(t.series[c.key]), format: c.fmt, label: c.label,
      area: c.area, zero: c.zero, yMin: c.yMin, yMax: c.yMax,
      bands: c.bands ? slice(t.series.regime_state) : null,
    }));
  };
  document.getElementById("ranges").addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    days = Number(b.dataset.days === "Infinity" ? Infinity : b.dataset.days);
    document.querySelectorAll("#ranges button").forEach((x) => x.classList.toggle("active", x === b));
    draw();
  });
  draw();
  state.redraw = draw;
}

function regimeTable(reg) {
  if (!reg?.available) return `<p class="notice">Regimi non disponibili (storico insufficiente).</p>`;
  return `<div class="table-scroll"><table>
    <thead><tr><th class="left">Regime</th><th>Quota delle sedute</th><th>Rendimento medio ann.</th><th>Volatilità ann.</th><th>Durata attesa (sedute)</th></tr></thead>
    <tbody>${reg.states.map((s) => `<tr><td class="left">${badge(s.label)}</td><td class="num">${pct(s.share_of_days, 0)}</td>
      <td class="num ${signClass(s.ret_ann)}">${pct(s.ret_ann, 1, true)}</td><td class="num">${pct(s.vol_ann)}</td>
      <td class="num">${num(s.expected_duration_days, 0)}</td></tr>`).join("")}</tbody></table></div>
    <p class="muted small" style="margin-top:8px">HMM a 3 stati su rendimento e volatilità a 21 sedute; probabilità filtrate (solo dati fino a ogni giorno),
      parametri ri-stimati ogni anno. Statistiche descrittive dello storico: non sono previsioni.</p>`;
}

function anomalyTable(an) {
  const recent = [...(an?.recent || [])].reverse();
  if (!recent.length) return `<p class="notice">Nessuna anomalia registrata.</p>`;
  const counts = Object.entries(an.count_by_kind || {}).map(([k, c]) => `${esc(ANOMALY_LABELS[k] || k)}: ${c}`).join(" · ");
  return `<p class="muted small">Totale storico per tipo — ${counts}. Ultimi 12 mesi: ${an.count_last_365_days ?? "–"}.</p>
    <div class="table-scroll"><table>
    <thead><tr><th class="left">Data</th><th class="left">Tipo</th><th>z-score</th><th>Valore</th></tr></thead>
    <tbody>${recent.slice(0, 25).map((a) => `<tr><td class="left">${fmtDate(a.date)}</td><td class="left">${esc(ANOMALY_LABELS[a.kind] || a.kind)}</td>
      <td class="num ${signClass(a.z)}">${num(a.z, 1, true)}</td><td class="num">${a.kind === "volume" ? (isNum(a.value) ? "×" + num(Math.exp(a.value), 1) + " mediana" : "–") : pct(a.value, 2, true)}</td></tr>`).join("")}</tbody>
    </table></div>
    <p class="muted small" style="margin-top:8px">${esc(an.method || "")}. Per il volume il valore è il rapporto con la mediana dell'ultimo anno.</p>`;
}

/* ------------------------------------------------------------------ modelli di volatilità */

function scoreCell(met) {
  if (!met || !isNum(met.qlike_diff_vs_rw)) return `<td class="num">–</td>`;
  const sig = isNum(met.dm_vs_rw) && Math.abs(met.dm_vs_rw) > 1.96;
  const cls = met.qlike_diff_vs_rw < 0 ? (sig ? "win" : "") : (sig ? "lose" : "");
  return `<td class="num ${cls}" title="t di Diebold-Mariano ${num(met.dm_vs_rw, 1)} · copertura 90% ${pct(met.coverage_90, 0)}">${num(met.qlike_diff_vs_rw, 3, true)}<span class="muted small"> (${num(met.dm_vs_rw, 1, true)})</span></td>`;
}

function seriesBlock(kind, data, h) {
  const models = Object.keys(MODEL_LABELS).filter((m) => m !== "rw" && Object.values(data.aggregate?.[h] || {}).length && data.aggregate[h][m]);
  const agg = data.aggregate?.[h] || {};
  const ids = Object.keys(data.instruments || {}).filter((id) => data.instruments[id][h]?.n);
  if (!ids.length) return null;
  return `
    <div class="table-scroll"><table>
      <thead><tr><th class="left">Modello</th><th>Strumenti</th><th>Battono rw</th><th>Significativi</th><th>Diff. media QLIKE vs rw</th></tr></thead>
      <tbody>${models.map((m) => `<tr><td class="left">${esc(MODEL_LABELS[m])}</td><td class="num">${agg[m].n_instruments}</td>
        <td class="num">${agg[m].wins}</td><td class="num">${agg[m].significant_wins}</td>
        <td class="num ${agg[m].mean_qlike_diff_vs_rw < 0 ? "win" : "lose"}">${num(agg[m].mean_qlike_diff_vs_rw, 3, true)}</td></tr>`).join("")}</tbody>
    </table></div>
    <h3 style="margin-top:14px">Per strumento <span class="muted small">— differenza di QLIKE vs rw (t di Diebold-Mariano)</span></h3>
    <div class="table-scroll"><table>
      <thead><tr><th class="left">Strumento</th><th>Previsioni</th><th>Periodo</th>${models.map((m) => `<th>${esc(MODEL_LABELS[m])}</th>`).join("")}</tr></thead>
      <tbody>${ids.map((id) => {
        const r = data.instruments[id][h];
        return `<tr><td class="left"><a href="#/t/${esc(id)}">${esc(id.toUpperCase())}</a></td><td class="num">${r.n}</td>
          <td class="small">${fmtDate(r.first_origin)} – ${fmtDate(r.last_origin)}</td>${models.map((m) => scoreCell(r.models?.[m])).join("")}</tr>`;
      }).join("")}</tbody>
    </table></div>`;
}

function renderModels() {
  setActiveTab("models");
  const md = state.models;
  const h = state.horizon;
  const bt = md.backtest, lv = md.live;
  const btBlock = bt.available ? seriesBlock("backtest", bt, h) : null;
  const lvBlock = lv.available ? seriesBlock("live", lv, h) : null;
  app.innerHTML = `
    <h1>Modelli di volatilità</h1>
    <p class="muted">Previsione della varianza dei rendimenti ${h === "h1" ? "della prossima seduta" : "delle prossime 5 sedute"}.
      Ogni modello è confrontato con la baseline <strong>rw</strong> (varianza delle ultime 21 sedute):
      valori negativi = meglio della baseline; in verde i miglioramenti statisticamente significativi (|t| &gt; 1,96).</p>
    <div class="toggle" id="horizon">
      <button data-h="h1" class="${h === "h1" ? "active" : ""}">Orizzonte 1 seduta</button>
      <button data-h="h5" class="${h === "h5" ? "active" : ""}">Orizzonte 5 sedute</button>
    </div>

    <section class="series-panel live">
      <h2 style="margin-top:0"><span class="series-tag live">LIVE</span>Previsioni registrate ogni sera</h2>
      <p class="muted small">Scritte da daily.yml prima di conoscere il risultato e mai ricalcolate: è l'unica prova pulita.
        ${lv.n_forecasts_logged || 0} previsioni nel log${lv.first_origin ? `, dal ${fmtDate(lv.first_origin)}` : ""}.</p>
      ${lvBlock || `<p class="notice">In accumulo: le valutazioni compaiono quando i risultati delle previsioni registrate sono noti
        (dopo 1 seduta per l'orizzonte 1, dopo 5 per l'orizzonte 5). Servono settimane prima che i numeri siano significativi.</p>`}
    </section>

    <section class="series-panel backtest">
      <h2 style="margin-top:0"><span class="series-tag backtest">BACKTEST</span>Walk-forward sullo storico</h2>
      <p class="muted small">Ricostruzione retroattiva: a ogni data i modelli usano solo i dati precedenti, ma la configurazione è stata scelta
        guardando lo storico. Periodo di test finale escluso (dal ${fmtDate(bt.holdout_start)})${bt.use_holdout ? " — <strong>ATTENZIONE: incluso in questo run</strong>" : ""}.
        Configurazioni provate finora: ${bt.n_configurations_tried ?? "–"}.</p>
      ${btBlock || `<p class="notice">Backtest non ancora disponibile.</p>`}
    </section>
    <p class="muted small">QLIKE: perdita robusta per previsioni di varianza (più bassa = meglio). Copertura 90% nel tooltip delle celle.</p>`;
  document.getElementById("horizon").addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    state.horizon = b.dataset.h;
    renderModels();
  });
}

/* ------------------------------------------------------------------ pattern */

function cmpCell(m, fmt, higherIsWorse = true) {
  if (!m) return `<td class="num">–</td>`;
  const worse = higherIsWorse ? m.excess > 0 : m.excess < 0;
  const cls = m.significant ? (worse ? "lose" : "win") : "";
  const ci = m.ci90 ? ` · IC 90% eccesso ${fmt(m.ci90[0])} / ${fmt(m.ci90[1])}` : "";
  return `<td class="num ${cls}" title="eccesso medio ${fmt(m.excess)}${ci}">${fmt(m.event)} <span class="muted small">vs ${fmt(m.baseline)}</span></td>`;
}

function renderPatterns() {
  setActiveTab("patterns");
  const P = state.patterns;
  if (!P || !P.backtest) {
    app.innerHTML = `<h1>Pattern</h1><p class="notice">Statistiche non ancora disponibili: compaiono dopo il primo run di daily.yml.</p>`;
    return;
  }
  const bt = P.backtest;
  const types = Object.entries(bt.event_types || {});
  const x = (v) => (isNum(v) ? num(v, 2) + "×" : "–");
  const p0 = (v) => pct(v, 0);
  const p1 = (v) => pct(v, 1, true);
  const current = P.current || [];
  const lv = P.live;

  const cards = types.map(([key, label]) => {
    const st = bt.stats?.[key] || {};
    const main = st[`h${bt.main_horizon}`] || {};
    const rows = (bt.horizons || []).map((h) => {
      const s = st[`h${h}`];
      if (!s || !s.n) return `<tr><td class="left">${h} sedute</td><td class="num">0</td><td colspan="3" class="muted left">nessun caso</td></tr>`;
      return `<tr><td class="left">${h} sedute</td><td class="num">${s.n} <span class="muted small">(${s.n_weeks ?? "–"} sett.)</span></td>
        ${cmpCell(s.vol_ratio, x)}${cmpCell(s.dd10, p0)}${cmpCell(s.ret, p1, false)}</tr>`;
    }).join("");
    const counts = Object.entries(bt.events_by_instrument?.[key] || {}).map(([id, c]) => `${id.toUpperCase()} ${c}`).join(" · ");
    return `<div class="chart-card">
      <div class="chart-head"><h3>${esc(label.charAt(0).toUpperCase() + label.slice(1))}</h3><span class="muted small">${main.n || 0} casi a ${bt.main_horizon} sedute</span></div>
      <div class="table-scroll" style="margin:8px 0"><table>
        <thead><tr><th class="left">Dopo</th><th>Casi</th><th>Volatilità vs tipica</th><th>Calo ≥ 10%</th><th>Rendimento medio</th></tr></thead>
        <tbody>${rows}</tbody></table></div>
      ${counts ? `<p class="muted small">Casi per strumento: ${esc(counts)}</p>` : ""}
    </div>`;
  }).join("");

  const liveRows = Object.entries(lv?.by_type || {}).map(([k, v]) => `<tr><td class="left">${esc(bt.event_types[k] || k)}</td><td class="num">${v.n}</td>
    <td class="num">${x(v.realized_vol_ratio)} <span class="muted small">attesa ${x(v.expected_vol_ratio)} · normale ${x(v.baseline_vol_ratio)}</span></td>
    <td class="num">${p0(v.realized_dd10)} <span class="muted small">atteso ${p0(v.expected_dd10)} · normale ${p0(v.baseline_dd10)}</span></td></tr>`).join("");

  app.innerHTML = `
    <h1>Pattern</h1>
    <p class="muted">Cosa è successo <strong>storicamente</strong> nelle sedute successive a certi eventi, confrontato con un giorno qualsiasi dello stesso titolo
      (<em>vs</em>). È statistica sul passato, non una previsione. Rendimento e calo sono misurati dall'<strong>apertura del giorno dopo</strong>,
      il primo prezzo a cui si può agire dopo un'analisi fatta a mercato chiuso. In verde/rosso le differenze statisticamente nette
      (intervallo al 90% che esclude lo zero e differenza rilevante; eventi della stessa settimana raggruppati).
      Rosso = più volatilità o più rischio di calo, o rendimento più basso del normale.
      Attenzione ai confronti multipli: le celle sono circa 70, quindi qualche risultato "netto" isolato può essere casuale;
      contano gli schemi coerenti tra orizzonti ed eventi.</p>

    <h2>Eventi delle ultime sedute</h2>
    ${current.length ? `<div class="table-scroll"><table><thead><tr><th class="left">Strumento</th><th class="left">Evento</th><th class="left">Data</th><th class="left">Storicamente</th></tr></thead>
      <tbody>${current.map((e) => `<tr><td class="left"><a href="#/t/${esc(e.id)}">${esc(e.id.toUpperCase())}</a></td><td class="left">${esc(bt.event_types[e.type] || e.type)}</td>
        <td class="left">${fmtDate(e.date)}</td><td class="left wrap-cell">${esc(e.history || "casi storici insufficienti")}</td></tr>`).join("")}</tbody></table></div>`
      : `<p class="notice">Nessun evento nelle ultime sedute.</p>`}

    <section class="series-panel live">
      <h2 style="margin-top:0"><span class="series-tag live">LIVE</span>Verifica sugli eventi nuovi</h2>
      <p class="muted small">Ogni evento nuovo viene registrato con l'esito che lo storico si aspetta; dopo ${bt.main_horizon} sedute si confronta con quello realizzato.
        Eventi registrati: ${lv?.n_logged ?? 0}, già valutabili: ${lv?.n_evaluated ?? 0}.</p>
      ${liveRows ? `<div class="table-scroll"><table><thead><tr><th class="left">Evento</th><th>Casi</th><th>Volatilità vs tipica</th><th>Calo ≥ 10%</th></tr></thead><tbody>${liveRows}</tbody></table></div>`
        : `<p class="notice">In accumulo: servono almeno ${bt.main_horizon} sedute dopo i primi eventi registrati.</p>`}
    </section>

    <section class="series-panel backtest">
      <h2 style="margin-top:0"><span class="series-tag backtest">BACKTEST</span>Statistiche storiche per tipo di evento</h2>
      <p class="muted small">Eventi dal 2010 su tutti gli strumenti dell'universo, solo finestre chiuse prima del periodo di test finale (${fmtDate(bt.holdout_start)}).
        Volatilità in multipli di quella tipica del titolo (1× = normale). Il rendimento è il dato meno affidabile: rumore e bias di selezione dell'universo.</p>
      ${cards}
    </section>`;
}

/* ------------------------------------------------------------------ metodo */

function renderMethod() {
  setActiveTab("method");
  app.innerHTML = `<div class="method">
    <h1>Metodo</h1>
    <p>FinanceNow <strong>descrive e misura</strong> il comportamento di indici e azioni: regimi, volatilità, anomalie, rischio.
      Non prevede i prezzi. I modelli predittivi (solo volatilità, per ora) sono sempre confrontati con baseline banali.</p>
    <h2>Dati</h2>
    <ul>
      <li>Prezzi e volumi giornalieri (OHLCV) aggiustati per split e dividendi, da yfinance, dal 2010.
        Aggiornamento ogni sera dopo la chiusura di Wall Street.</li>
      <li><strong>Solo prezzi e volumi</strong>: nessun dato di bilancio, macroeconomico, di notizie o calendario degli utili.</li>
      <li>I prezzi grezzi non sono pubblicati: la dashboard mostra solo grandezze derivate.</li>
      <li>Universo fisso: Nasdaq 100, QQQ, S&amp;P 500 e le 10 azioni più capitalizzate del Nasdaq al 26/10/2026.
        <strong>Bias di selezione</strong>: sono i vincitori di oggi, il loro passato è eccezionale per costruzione.</li>
    </ul>
    <h2>Analisi</h2>
    <ul>
      <li>Tutto su log-rendimenti; ogni valore di un giorno usa solo dati fino a quel giorno.</li>
      <li><strong>Regimi</strong>: modello di Markov nascosto a 3 stati (calmo, normale, turbolento) su rendimento e volatilità a 21 sedute.</li>
      <li><strong>Anomalie</strong>: z-score robusto (mediana e MAD dell'ultimo anno) su rendimento, gap, escursione, volume e rendimento rispetto al benchmark; soglia |z| ≥ 4.</li>
      <li><strong>Volatilità</strong>: GARCH, EGARCH e HAR-RV contro le baseline rw (ultime 21 sedute) e media storica, valutati con QLIKE.</li>
    </ul>
    <h2>Validazione</h2>
    <ul>
      <li><span class="series-tag backtest">BACKTEST</span>walk-forward sullo storico, con un periodo finale di test escluso.</li>
      <li><span class="series-tag live">LIVE</span>previsioni registrate ogni sera e valutate solo dopo: l'unica prova pulita.</li>
    </ul>
    <p class="muted">Strumento di analisi personale, non consulenza finanziaria.</p>
  </div>`;
}

/* ------------------------------------------------------------------ router */

async function route() {
  tooltip.hidden = true;
  state.redraw = null;
  const hash = location.hash || "#/";
  try {
    if (hash.startsWith("#/t/")) await renderTicker(decodeURIComponent(hash.slice(4)));
    else if (hash === "#/modelli") renderModels();
    else if (hash === "#/pattern") renderPatterns();
    else if (hash === "#/metodo") renderMethod();
    else renderOverview();
  } catch (err) {
    app.innerHTML = `<p class="notice">Errore nel caricamento dei dati: ${esc(err.message)}</p>`;
  }
  window.scrollTo(0, 0);
}

async function init() {
  try {
    [state.index, state.models] = await Promise.all([getJSON("data/index.json"), getJSON("data/models.json")]);
    state.digest = await getJSON("data/digest.json").catch(() => null);
    state.patterns = await getJSON("data/patterns.json").catch(() => null);
  } catch (err) {
    app.innerHTML = `<p class="notice">Dati non disponibili: ${esc(err.message)}.<br>
      La dashboard si popola dopo il primo run di <code>daily.yml</code>.</p>`;
    return;
  }
  renderStatus();
  window.addEventListener("hashchange", route);
  let timer;
  window.addEventListener("resize", () => {
    clearTimeout(timer);
    timer = setTimeout(() => state.redraw && state.redraw(), 150);
  });
  route();
}

init();
