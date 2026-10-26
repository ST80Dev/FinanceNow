/* FinanceNow — dashboard statica (vanilla JS, nessuna dipendenza).
 * Legge solo i JSON in data/ prodotti da pipeline/export.py.
 * Viste (routing via hash): #/  panoramica · #/t/<id>  strumento · #/pattern · #/modelli · #/guida · #/metodo
 * Linguaggio semplice: ogni grandezza ha una spiegazione nel tooltip (attributo data-tip),
 * che compare passando sopra col mouse o toccando sul telefono.
 */
"use strict";

const REGIMES = ["calmo", "normale", "turbolento"];
const PHASE = { calmo: "calma", normale: "normale", turbolento: "agitata" };
const ANOMALY_LABELS = {
  rendimento: "Movimento del prezzo",
  gap: "Salto all'apertura",
  range: "Escursione nella giornata",
  volume: "Volumi scambiati",
  rendimento_vs_bench: "Movimento diverso dal Nasdaq",
};
const MODEL_LABELS = {
  rw: "rw (baseline)",
  hist_mean: "media storica (baseline)",
  har: "HAR-RV",
  garch: "GARCH(1,1)-t",
  egarch: "EGARCH(1,1)-t",
};
const CLASS_LABELS = { index: "Indice", etf: "ETF", stock: "Azione" };
const SITUATION_ICON = { tempesta: "🔴", pressione: "🟠", ripresa: "🔵", norma: "⚪", tranquillo: "🟢" };
const RISK_LABEL = { alto: "Rischio alto", normale: "Rischio normale", basso: "Rischio basso" };

/* Spiegazioni in parole semplici: usate nei tooltip e nella pagina Guida. */
const GLOSSARY = {
  situazione: ["Situazione", "Il giudizio dell'app su com'è messo il titolo adesso, ricavato da regole fisse.\n" +
    "🔴 In tempesta: movimenti ampi e nervosi in corso.\n🟠 Sotto pressione: forte calo dal massimo o agitazione in aumento.\n" +
    "🔵 In ripresa: appena uscito da una fase agitata.\n⚪ Nella norma.\n🟢 Tranquillo: fase calma, movimenti contenuti."],
  rischio: ["Rischio nel prossimo mese", "Quanto è probabile che il prossimo mese sia movimentato, con oscillazioni ampie o cali.\n" +
    "Sale se il titolo si muove già più del solito, se in passato situazioni simili sono state seguite da mesi agitati, " +
    "o se la trimestrale è vicina. Non dice se salirà o scenderà."],
  oscillazione: ["Oscillazione tipica al mese", "Di quanto si muove di solito il prezzo in un mese, in su o in giù: " +
    "circa 2 mesi su 3 il movimento resta entro questo valore.\nÈ la volatilità (annualizzata) divisa per √12. " +
    "Più è alta, più il titolo è nervoso: un ±10% al mese è tanto, un ±3% è poco."],
  solito: ["Rispetto al solito", "Confronta quanto si muove oggi con tutta la storia del titolo.\n" +
    "50% = come al solito. 90% = si muove più che nel 90% dei giorni passati (molto agitato). " +
    "10% = insolitamente calmo.\nServe perché il ±10% al mese è normale per Tesla ma eccezionale per Apple."],
  massimo: ["Distanza dal massimo", "Quanto è sotto il prezzo più alto mai raggiunto (drawdown).\n" +
    "Le perdite non sono simmetriche: dopo un −20% serve un +25% per tornare in pari, dopo un −50% serve un +100%. " +
    "Sotto il −20% si parla comunemente di fase ribassista."],
  andamento: ["Andamento", "Quanto è cambiato il prezzo nel periodo indicato, in percentuale semplice " +
    "(dividendi e frazionamenti già considerati). Descrive il passato, non promette nulla sul futuro."],
  beta: ["Quanto segue il Nasdaq", "Di quanto amplifica i movimenti del Nasdaq 100 (beta, ultimo anno).\n" +
    "1,0 = si muove come l'indice. 1,5 = se il Nasdaq fa +1% di solito fa +1,5% (e −1,5% quando scende). " +
    "0,7 = più tranquillo del mercato."],
  fase: ["Fase (regime)", "Un modello statistico (HMM) riconosce tre \"stati d'animo\" del titolo guardando rendimenti e agitazione: " +
    "calma, normale, agitata. Le fasi tendono a durare settimane o mesi: quella attuale dice in che tipo di periodo siamo. " +
    "Si basa solo sui dati fino a oggi."],
  tendenza: ["Tendenza", "La direzione del prezzo finora: \"in salita\" se è sopra la sua media degli ultimi 200 giorni " +
    "e negli ultimi 3 mesi è salito; \"in discesa\" se è il contrario; altrimenti senza direzione chiara. È una descrizione, non una previsione."],
  direzione: ["Direzione (sperimentale)", "Probabilità stimata che tra un mese il prezzo sia più alto di oggi, " +
    "confrontata con la frequenza storica del titolo (es. sale nel 62% dei mesi).\n" +
    "\"Più favorevole del solito\" = il modello vede una probabilità più alta della frequenza storica.\n" +
    "È affidabile solo se nel test sullo storico ha battuto la frequenza storica: finché non succede va presa come curiosità."],
  intervallo: ["Intervallo plausibile", "Dove è finito il prezzo dopo un mese in 8 casi su 10 in situazioni di agitazione simile " +
    "(10° e 90° percentile). Il resto delle volte è andato oltre, in su o in giù."],
  trimestrale: ["Trimestrale", "La pubblicazione dei conti trimestrali. Il giorno dopo il prezzo si muove di solito molto più del normale, " +
    "in una direzione difficile da prevedere. Un giorno insolito in quella data è \"spiegato\" dai conti."],
  anomalia: ["Giorno insolito (anomalia)", "Un giorno in cui prezzo, salto di apertura, escursione o volumi si sono mossi " +
    "almeno 4 volte più dello scarto tipico dell'ultimo anno. Capita di rado: segnala che è successo qualcosa."],
  pattern: ["Pattern", "Cosa è successo in passato nel mese dopo un certo evento, confrontato con un giorno qualsiasi dello stesso titolo. " +
    "Statistica sul passato, non garanzia."],
  netto: ["Differenza chiara", "La differenza rispetto a un giorno qualsiasi è abbastanza grande e abbastanza costante da non poter " +
    "essere spiegata dal caso (intervallo di confidenza al 90% che esclude lo zero, più una soglia minima di rilevanza)."],
  serie: ["Backtest e live", "BACKTEST = prova sullo storico, ricostruita a ritroso. LIVE = previsioni scritte ogni sera prima di " +
    "conoscere il risultato e verificate dopo: è l'unica prova pulita, ma serve tempo per accumularla."],
};

const state = { index: null, models: null, digest: null, patterns: null, assessment: null, direction: null,
  tickers: {}, sort: { key: null, dir: -1 }, horizon: "h1" };
const app = document.getElementById("app");
const tooltip = document.getElementById("tooltip");

/* ------------------------------------------------------------------ utilità */

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
const isNum = (v) => typeof v === "number" && Number.isFinite(v);
function pct(v, digits = 1, signed = false) {
  if (!isNum(v)) return "–";
  const s = (v * 100).toFixed(digits).replace(".", ",").replace(/^-(0(,0+)?)$/, "$1") + "%";   // niente "-0%"
  return signed && v > 0 ? "+" + s : s;
}
function num(v, digits = 2, signed = false) {
  if (!isNum(v)) return "–";
  const s = v.toFixed(digits).replace(".", ",");
  return signed && v > 0 ? "+" + s : s;
}
/* Log-rendimento -> variazione percentuale semplice (quella che ci si aspetta di leggere). */
const simple = (v) => (isNum(v) ? Math.expm1(v) : null);
function signClass(v) { return !isNum(v) ? "" : v > 0 ? "pos" : v < 0 ? "neg" : ""; }
function fmtDate(iso) {
  if (!iso) return "–";
  const [y, m, d] = iso.slice(0, 10).split("-");
  return `${d}/${m}/${y}`;
}
const shortDate = (iso) => (iso ? `${iso.slice(8, 10)}/${iso.slice(5, 7)}` : "–");
const cap = (t) => (t ? t.charAt(0).toUpperCase() + t.slice(1) : "");

function badge(regime) {
  return regime ? `<span class="badge ${esc(regime)}" data-tip="${tipText("fase")}">fase ${esc(PHASE[regime] || regime)}</span>`
    : `<span class="badge none">n.d.</span>`;
}

/* Testo per data-tip (titolo in grassetto + spiegazione); \n diventa a capo. */
function tipText(key, extra = "") {
  const g = GLOSSARY[key];
  const body = g ? `**${g[0]}**\n${g[1]}` : "";
  return esc(extra ? (body ? `${body}\n\n${extra}` : extra) : body);
}
const info = (key) => `<span class="info" tabindex="0" data-tip="${tipText(key)}" aria-label="spiegazione">i</span>`;

function instName(r, linked = false) {
  const tip = esc(`**${r.name}** (${r.id.toUpperCase()})\n${r.description || ""}`);
  const label = `${esc(r.name)}<small>${esc(r.id.toUpperCase())} · ${esc(CLASS_LABELS[r.asset_class] || r.asset_class || "")}</small>`;
  return linked ? `<a class="name" href="#/t/${esc(r.id)}" data-tip="${tip}">${label}</a>` : `<span class="name" data-tip="${tip}">${label}</span>`;
}

async function getJSON(path) {
  const res = await fetch(path, { cache: "no-cache" });
  if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
  return res.json();
}

/* ------------------------------------------------------------------ tooltip */

function renderTip(text) {
  return esc(text).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>").replace(/\n/g, "<br>");
}
function placeTip(x, y) {
  const tx = Math.min(x + 14, window.innerWidth - tooltip.offsetWidth - 8);
  let ty = y + 16;
  if (ty + tooltip.offsetHeight > window.innerHeight - 8) ty = Math.max(8, y - tooltip.offsetHeight - 12);
  tooltip.style.left = `${Math.max(8, tx)}px`;
  tooltip.style.top = `${ty}px`;
}
let tipOwner = null;
function showTip(target, x, y) {
  tipOwner = target;
  tooltip.innerHTML = renderTip(target.getAttribute("data-tip"));
  tooltip.hidden = false;
  tooltip.classList.add("wide");
  placeTip(x, y);
}
function hideTip() { tipOwner = null; tooltip.hidden = true; tooltip.classList.remove("wide"); }

document.addEventListener("pointerover", (e) => {
  if (e.pointerType === "touch") return;
  const t = e.target.closest("[data-tip]");
  if (t && t.getAttribute("data-tip")) showTip(t, e.clientX, e.clientY);
});
document.addEventListener("pointermove", (e) => {
  if (tipOwner && e.pointerType !== "touch") placeTip(e.clientX, e.clientY);
});
document.addEventListener("pointerout", (e) => {
  if (tipOwner && e.pointerType !== "touch" && !tipOwner.contains(e.relatedTarget)) hideTip();
});
/* Sul telefono: un tocco mostra la spiegazione, un altro tocco altrove la chiude. */
document.addEventListener("click", (e) => {
  const t = e.target.closest("[data-tip]");
  if (t && t.getAttribute("data-tip")) {
    const isInfo = t.classList.contains("info");
    const touch = e.pointerType === "touch";
    if (isInfo || (touch && !t.closest("a"))) {
      e.preventDefault();
      e.stopPropagation();          // niente ordinamento della colonna / apertura della riga
      if (tipOwner === t) { hideTip(); return; }
      const r = t.getBoundingClientRect();
      showTip(t, r.left, r.bottom);
      return;
    }
  }
  if (tipOwner) hideTip();
}, true);
/* Da tastiera (Tab) la spiegazione compare al focus; col mouse o col dito ci pensa il click. */
let lastPointerDown = 0;
document.addEventListener("pointerdown", () => { lastPointerDown = Date.now(); }, true);
document.addEventListener("focusin", (e) => {
  const t = e.target.closest?.(".info[data-tip]");
  if (t && Date.now() - lastPointerDown > 600) { const r = t.getBoundingClientRect(); showTip(t, r.left, r.bottom); }
});
document.addEventListener("focusout", (e) => { if (e.target === tipOwner) hideTip(); });

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
  const parts = [`Dati al <strong>${fmtDate(idx.as_of)}</strong> (chiusura di Wall Street)`,
                 `aggiornato ${fmtDate(idx.generated)}`];
  const ds = idx.data_status || {};
  if (ds.missing?.length) parts.push(`<span class="warn">Senza dati: ${esc(ds.missing.join(", "))}</span>`);
  if (ds.stale?.length) parts.push(`<span class="warn">Non aggiornati: ${esc(ds.stale.join(", "))}</span>`);
  box.innerHTML = parts.map((p) => `<span>${p}</span>`).join("");
}

function setActiveTab(tab) {
  document.querySelectorAll(".tabs a").forEach((a) => a.classList.toggle("active", a.dataset.tab === tab));
}

/* ------------------------------------------------------------------ elementi della valutazione */

function situationBadge(s) {
  if (!s) return "";
  return `<span class="sit sit-${esc(s.key)}" data-tip="${tipText("situazione")}">${SITUATION_ICON[s.key] || ""} ${esc(s.label)}</span>`;
}
function riskBadge(r) {
  if (!r) return "";
  const why = r.reasons?.length ? "Perché: " + r.reasons.join("; ") + "." : "Nessun motivo particolare: rischio nella media del titolo.";
  return `<span class="risk risk-${esc(r.level)}" data-tip="${tipText("rischio", why)}">${esc(RISK_LABEL[r.level] || r.level)}</span>`;
}

function evidenceText(ev) {
  if (!ev) return "";
  const parts = [];
  if (isNum(ev.dd10_event)) parts.push(`un ulteriore calo di almeno il 10% è capitato nel ${pct(ev.dd10_event, 0)} dei casi (in un mese qualsiasi: ${pct(ev.dd10_base, 0)})`);
  if (isNum(ev.vol_event)) parts.push(`il titolo si è mosso ${num(ev.vol_event, 1)} volte il suo solito (normalmente ${num(ev.vol_base, 1)})`);
  return `In passato, nel mese dopo situazioni simili (${ev.n} casi): ${parts.join("; ")}.`;
}

function outlookBlock(ol, full = false) {
  if (!ol) return `<span class="muted">non disponibile</span>`;
  const rel = ol.reliable ? `<span class="chip good" data-tip="${tipText("direzione", "Nel test sullo storico il modello ha battuto la frequenza storica in modo netto.")}">affidabile</span>`
    : `<span class="chip warn-chip" data-tip="${tipText("direzione", "Nel test sullo storico il modello NON ha battuto la frequenza storica di salita: per ora è un indizio, non un'indicazione. Dettagli nella pagina Modelli.")}">non affidabile</span>`;
  const probs = `<span class="num">${pct(ol.p_up, 0)}</span> <span class="muted small">di salire in un mese (di solito ${pct(ol.p_base, 0)})</span>`;
  return `<div><span class="ol-${esc(ol.key)}"><strong>${esc(ol.label)}</strong></span> ${rel}</div><div class="small">${probs}</div>` +
    (full && ol.drivers?.length ? `<div class="small muted">Cosa lo muove: ${esc(ol.drivers.join("; "))}.</div>` : "");
}

function factGrid(a, row, full = false) {
  const r = a.risk || {};
  const tr = a.trend || {};
  const range = r.range_month ? `<div class="small muted" data-tip="${tipText("intervallo")}">8 volte su 10 tra <span class="num">${pct(r.range_month[0], 0, true)}</span> e <span class="num">${pct(r.range_month[1], 0, true)}</span></div>` : "";
  return `<div class="facts">
    <div class="fact"><div class="fact-label">Oscillazione tipica ${info("oscillazione")}</div>
      <div class="fact-value num">±${pct(r.month_move, 0)} <span class="small muted">al mese</span></div>
      ${isNum(r.vol_percentile) ? `<div class="small muted" data-tip="${tipText("solito")}">si muove più che nel ${Math.round(r.vol_percentile * 100)}% della sua storia</div>` : ""}${range}</div>
    <div class="fact"><div class="fact-label">Tendenza ${info("tendenza")}</div>
      <div class="fact-value trend-${esc(tr.key)}">${esc(tr.label || "–")}</div>
      <div class="small muted">3 mesi <span class="num ${signClass(tr.ret_3m)}">${pct(tr.ret_3m, 0, true)}</span> · 1 anno <span class="num ${signClass(tr.ret_12m)}">${pct(tr.ret_12m, 0, true)}</span>
      ${row && isNum(row.drawdown) ? ` · <span data-tip="${tipText("massimo")}">dal massimo <span class="num ${signClass(row.drawdown)}">${pct(row.drawdown, 0)}</span></span>` : ""}</div></div>
    <div class="fact fact-wide"><div class="fact-label">Direzione a un mese ${info("direzione")}</div>${outlookBlock(a.outlook, full)}</div>
  </div>`;
}

function assessCard(a, row, full = false) {
  const events = (a.events || []).filter((e) => full || e.soon || e.kind === "trimestrale_passata");
  const r = a.risk || {};
  return `<article class="acard sit-border-${esc(a.situation.key)}">
    <div class="acard-head">
      ${full ? "" : `<div class="acard-name">${instName(a, true)}</div>`}
      <div class="acard-badges">${situationBadge(a.situation)} ${riskBadge(r)}</div>
    </div>
    <p class="acard-why">${esc(cap(a.situation.why))}.</p>
    ${factGrid(a, row, full)}
    ${events.length ? `<ul class="events">${events.map((e) => `<li data-tip="${tipText("trimestrale")}">📅 ${esc(e.text)}</li>`).join("")}</ul>` : ""}
    ${r.evidence ? `<p class="small evidence" data-tip="${tipText("pattern")}">${esc(evidenceText(r.evidence))}</p>` : ""}
    ${a.signals?.length ? `<details class="small"><summary>Tutti i segnali (${a.signals.length})</summary><ul class="sig-list">
      ${a.signals.map((s) => `<li><span class="sev-dot sev-${esc(s.severity)}"></span>${esc(cap(s.text))}</li>`).join("")}</ul></details>` : ""}
  </article>`;
}

/* ------------------------------------------------------------------ panoramica */

const COLUMNS = [
  { key: "name", label: "Strumento", left: true, render: (r) => instName(r, true) },
  { key: "sit_rank", label: "Situazione", tip: "situazione", left: true, render: (r, a) => situationBadge(a?.situation) },
  { key: "risk_rank", label: "Rischio", tip: "rischio", left: true, render: (r, a) => riskBadge(a?.risk) },
  { key: "rv_21", label: "Oscillazione/mese", tip: "oscillazione", render: (r) => `<span class="num">±${pct(isNum(r.rv_21) ? r.rv_21 / Math.sqrt(12) : null, 0)}</span>` },
  { key: "vol_pct", label: "Rispetto al solito", tip: "solito", render: (r, a) => solitoBar(a?.risk?.vol_percentile) },
  { key: "drawdown", label: "Dal massimo", tip: "massimo", render: (r) => `<span class="num ${signClass(r.drawdown)}">${pct(r.drawdown, 0)}</span>` },
  { key: "mom_3m", label: "3 mesi", tip: "andamento", render: (r) => `<span class="num ${signClass(r.mom_3m)}">${pct(simple(r.mom_3m), 1, true)}</span>` },
  { key: "mom_12m", label: "1 anno", tip: "andamento", render: (r) => `<span class="num ${signClass(r.mom_12m)}">${pct(simple(r.mom_12m), 1, true)}</span>` },
  { key: "beta_252", label: "Segue il Nasdaq", tip: "beta", render: (r) => betaText(r.beta_252) },
  { key: "earn_next", label: "Trimestrale", tip: "trimestrale", left: true, render: (r, a) => earnCell(a) },
];

function solitoBar(p) {
  if (!isNum(p)) return "–";
  const cls = p >= 0.9 ? "hot" : p >= 0.7 ? "warm" : p <= 0.1 ? "cold" : "";
  return `<span class="meter ${cls}" data-tip="${tipText("solito", `Oggi: più che nel ${Math.round(p * 100)}% della sua storia.`)}"><i style="width:${Math.round(p * 100)}%"></i></span><span class="num small"> ${Math.round(p * 100)}%</span>`;
}
function betaText(b) {
  if (!isNum(b)) return `<span class="muted">–</span>`;
  const word = b >= 1.25 ? "amplifica" : b <= 0.8 ? "attenua" : "simile";
  return `<span class="num" data-tip="${tipText("beta", `Se il Nasdaq fa ±1%, di solito questo titolo fa ±${num(b, 1)}%.`)}">${num(b, 1)}×</span> <span class="small muted">${word}</span>`;
}
function earnCell(a) {
  const e = (a?.events || []).find((x) => x.kind === "trimestrale");
  if (!e) return `<span class="muted">–</span>`;
  const soon = e.sessions_until <= 10;
  return `<span class="${soon ? "earn-soon" : ""}" data-tip="${esc(e.text)}">${shortDate(e.date)} <span class="small muted">(${e.sessions_until <= 1 ? "domani" : "tra " + e.sessions_until + " sedute"})</span></span>`;
}

function quietEarn(a) {
  const e = (a.events || []).find((x) => x.kind === "trimestrale" && x.soon);
  return e ? `<span class="earn-soon small"> · 📅 trimestrale ${e.sessions_until <= 1 ? "domani" : "tra " + e.sessions_until + " sedute"}</span>` : "";
}

function sortValue(r, a, key) {
  if (key === "sit_rank") return a ? { tempesta: 4, pressione: 3, ripresa: 2, norma: 1, tranquillo: 0 }[a.situation.key] : -1;
  if (key === "risk_rank") return a ? { alto: 2, normale: 1, basso: 0 }[a.risk.level] : -1;
  if (key === "vol_pct") return a?.risk?.vol_percentile;
  if (key === "earn_next") { const e = (a?.events || []).find((x) => x.kind === "trimestrale"); return e ? -e.sessions_until : null; }
  return r[key];
}

function renderOverview() {
  setActiveTab("overview");
  const A = state.assessment;
  if (!A) { app.innerHTML = `<h1>Panoramica</h1><p class="notice">Valutazione non ancora disponibile: compare dopo il prossimo run di daily.yml.</p>`; return; }
  const byId = Object.fromEntries(A.instruments.map((a) => [a.id, a]));
  const rowsById = Object.fromEntries(state.index.instruments.map((r) => [r.id, r]));
  const focus = A.instruments.filter((a) => ["tempesta", "pressione", "ripresa"].includes(a.situation.key) || a.risk.level === "alto");
  const quiet = A.instruments.filter((a) => !focus.includes(a));
  const m = A.market || {};

  const rows = [...state.index.instruments];
  const { key, dir } = state.sort;
  if (key) {
    rows.sort((x, y) => {
      const va = sortValue(x, byId[x.id], key), vb = sortValue(y, byId[y.id], key);
      if (va === vb) return 0;
      if (va === null || va === undefined) return 1;
      if (vb === null || vb === undefined) return -1;
      return (va > vb ? 1 : -1) * dir;
    });
  }
  const live = A.live || {};
  app.innerHTML = `
    <section class="market">
      <h1>Situazione al ${fmtDate(A.as_of)}</h1>
      <p class="market-head">${m.situation ? situationBadge(m.situation) : ""} ${m.risk ? riskBadge({ level: m.risk, reasons: [] }) : ""}</p>
      <p class="market-text">${esc(m.headline || "")}</p>
      <p class="muted">${esc(m.detail || "")}</p>
    </section>

    <h2>Da guardare <span class="muted small">(${focus.length})</span></h2>
    ${focus.length ? `<div class="acards">${focus.map((a) => assessCard(a, rowsById[a.id])).join("")}</div>`
      : `<p class="notice">Niente di particolare oggi: tutti gli strumenti sono tranquilli o nella norma.</p>`}

    ${quiet.length ? `<h2>Senza novità <span class="muted small">(${quiet.length})</span></h2>
      <div class="quiet">${quiet.map((a) => `<a class="quiet-item" href="#/t/${esc(a.id)}" data-tip="${esc(`**${a.name}**\n${a.description || ""}\n\n${cap(a.situation.why)}.`)}">
        ${SITUATION_ICON[a.situation.key]} <strong>${esc(a.name)}</strong> <span class="muted small">${esc(a.id.toUpperCase())} · ${esc(a.situation.label.toLowerCase())} · ${esc((RISK_LABEL[a.risk.level] || "").toLowerCase())}</span>
        ${quietEarn(a)}</a>`).join("")}</div>` : ""}

    <h2>Tutti gli strumenti</h2>
    <p class="muted small">Passa sopra (o tocca) titoli, intestazioni e valori per la spiegazione. Clicca un'intestazione per ordinare, un nome per il dettaglio.</p>
    <div class="table-scroll"><table>
      <thead><tr>${COLUMNS.map((c) => `<th class="sortable ${c.left ? "left" : ""} ${key === c.key ? "sorted" + (dir > 0 ? " asc" : "") : ""}" data-key="${c.key}">${c.label}${c.tip ? " " + info(c.tip) : ""}</th>`).join("")}</tr></thead>
      <tbody>${rows.map((r) => `<tr class="clickable" data-id="${esc(r.id)}">${COLUMNS.map((c) => `<td class="${c.left ? "left" : ""}">${c.render(r, byId[r.id])}</td>`).join("")}</tr>`).join("")}</tbody>
    </table></div>

    <section class="series-panel live">
      <h2 style="margin-top:0"><span class="series-tag live">LIVE</span>Quanto ci azzecca il giudizio ${info("serie")}</h2>
      <p class="muted small">Ogni sera il giudizio viene registrato; dopo un mese si controlla cosa è successo davvero.
        Giudizi registrati: ${live.n_logged ?? 0}${live.first_date ? ` dal ${fmtDate(live.first_date)}` : ""}, già verificabili: ${live.n_evaluated ?? 0}.</p>
      ${live.by_risk ? `<div class="table-scroll"><table><thead><tr><th class="left">Rischio indicato</th><th>Casi</th><th>Calo ≥ 10% nel mese dopo</th><th>Agitazione (1 = solita)</th></tr></thead>
        <tbody>${["alto", "normale", "basso"].filter((k) => live.by_risk[k]).map((k) => `<tr><td class="left">${riskBadge({ level: k, reasons: [] })}</td>
          <td class="num">${live.by_risk[k].n}</td><td class="num">${pct(live.by_risk[k].dd10, 0)}</td><td class="num">${num(live.by_risk[k].vol_ratio, 2)}</td></tr>`).join("")}</tbody></table></div>
        ${live.by_outlook ? `<p class="small" style="margin-top:8px">Direzione: ${Object.entries(live.by_outlook).map(([k, v]) => `"${esc(k)}" → salito nel ${pct(v.up_rate, 0)} di ${v.n} casi`).join(" · ")}.</p>` : ""}`
        : `<p class="notice">In accumulo: le prime verifiche arrivano un mese dopo i primi giudizi registrati.</p>`}
    </section>

    <details class="small muted rules"><summary>Come decide l'app</summary><pre class="dg-rules">${esc(A.rules || "")}</pre></details>`;

  app.querySelectorAll("th.sortable").forEach((th) => th.addEventListener("click", () => {
    const k = th.dataset.key;
    state.sort = { key: k, dir: state.sort.key === k ? -state.sort.dir : -1 };
    renderOverview();
  }));
  app.querySelectorAll("tr.clickable").forEach((tr) => tr.addEventListener("click", (e) => {
    if (e.target.closest("a")) return;
    location.hash = `#/t/${tr.dataset.id}`;
  }));
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
  const probs = reg.probabilities || {};
  const a = state.assessment?.instruments.find((x) => x.id === id);
  const row = state.index.instruments.find((r) => r.id === id);
  const fcLive = t.live_latest?.vol_ann?.h1 || {};

  app.innerHTML = `
    <a class="back" href="#/">← Panoramica</a>
    <h1>${esc(t.name)} <span class="muted">${esc(t.id.toUpperCase())}</span></h1>
    <p class="lead">${esc(t.description || "")}</p>
    <p class="muted small">${esc(CLASS_LABELS[t.asset_class] || t.asset_class)}${t.benchmark ? ` · confrontato con ${esc(t.benchmark.toUpperCase())}` : ""}
      · dati al ${fmtDate(t.as_of)} · storico dal ${fmtDate(t.data_period?.first)}</p>

    ${a ? assessCard(a, row, true) : ""}

    ${earningsBlock(t.earnings)}

    <h2>I numeri in dettaglio</h2>
    <div class="cards">
      <div class="card">
        <h3>Fase ${info("fase")}</h3>
        <div>${badge(reg.current)} <span class="muted small">da ${reg.days_in_state ?? "–"} sedute</span></div>
        <div class="probs">${REGIMES.map((g) => `<i class="${g}" style="width:${(probs[g] || 0) * 100}%"></i>`).join("")}</div>
        <div class="small muted">${REGIMES.map((g) => `${PHASE[g]} ${pct(probs[g], 0)}`).join(" · ")}</div>
      </div>
      <div class="card">
        <h3>Quanto si muove ${info("oscillazione")}</h3>
        <div class="kv"><span>Ultimo mese</span><span class="num">±${pct(isNum(f.rv_21) ? f.rv_21 / Math.sqrt(12) : null, 1)} al mese</span></div>
        <div class="kv"><span>Ultimi 3 mesi</span><span class="num">±${pct(isNum(f.rv_63) ? f.rv_63 / Math.sqrt(12) : null, 1)} al mese</span></div>
        <div class="kv"><span data-tip="Volatilità annualizzata a 21 sedute: la misura tecnica da cui deriva l'oscillazione al mese.">Volatilità annua (tecnico)</span><span class="num">${pct(f.rv_21)}</span></div>
        <div class="kv small muted"><span><span class="series-tag live">LIVE</span>Attesa domani (GARCH)</span><span class="num">±${pct(isNum(fcLive.garch) ? fcLive.garch / Math.sqrt(252) : null, 1)}</span></div>
      </div>
      <div class="card">
        <h3>Cali e legame col mercato</h3>
        <div class="kv"><span data-tip="${tipText("massimo")}">Dal massimo storico</span><span class="num ${signClass(f.drawdown)}">${pct(f.drawdown)}</span></div>
        <div class="kv"><span data-tip="${tipText("beta")}">Quanto segue il Nasdaq</span><span class="num">${isNum(f.beta_252) ? num(f.beta_252, 2) + "×" : "–"}</span></div>
        <div class="kv"><span data-tip="Correlazione (ultimo anno): 1 = si muove sempre insieme al Nasdaq, 0 = per conto suo.">Si muove insieme al Nasdaq</span><span class="num">${num(f.corr_252)}</span></div>
        <div class="kv"><span data-tip="Asimmetria: negativa = i giorni molto brutti sono più frequenti di quelli molto belli (ultimi 3 mesi).">Asimmetria dei movimenti</span><span class="num">${num(f.skew_63)}</span></div>
      </div>
      <div class="card">
        <h3>Andamento ${info("andamento")}</h3>
        <div class="kv"><span>1 mese / 3 mesi</span><span class="num"><span class="${signClass(f.mom_1m)}">${pct(simple(f.mom_1m), 1, true)}</span> / <span class="${signClass(f.mom_3m)}">${pct(simple(f.mom_3m), 1, true)}</span></span></div>
        <div class="kv"><span>6 mesi / 1 anno</span><span class="num"><span class="${signClass(f.mom_6m)}">${pct(simple(f.mom_6m), 1, true)}</span> / <span class="${signClass(f.mom_12m)}">${pct(simple(f.mom_12m), 1, true)}</span></span></div>
        <div class="kv"><span data-tip="Quanto è sopra (+) o sotto (−) la media dei prezzi degli ultimi 200 giorni: sopra è considerato segno di tendenza positiva.">Rispetto alla media a 200 giorni</span><span class="num ${signClass(f.dist_ma200)}">${pct(simple(f.dist_ma200), 1, true)}</span></div>
        ${t.benchmark ? `<div class="kv"><span data-tip="Andamento dell'ultimo anno meno quello del Nasdaq 100: positivo = ha fatto meglio del mercato.">Meglio/peggio del Nasdaq (1 anno)</span><span class="num ${signClass(f.rs_12m)}">${pct(f.rs_12m, 1, true)}</span></div>` : ""}
      </div>
    </div>

    <div class="chart-head" style="margin-top:14px">
      <h2 style="margin:0">Storico</h2>
      <div class="ranges" id="ranges">${RANGES.map(([l, d], i) => `<button data-days="${d}" class="${i === 1 ? "active" : ""}">${l}</button>`).join("")}</div>
    </div>
    <div class="legend">${REGIMES.map((g, i) => `<span><i class="band-legend s${i}" style="background:var(--${g}-bg);border:1px solid var(--${g})"></i>fase ${PHASE[g]}</span>`).join("")}<span>sfondo = fase stimata quel giorno</span></div>
    <div id="charts"></div>

    <h2>Le fasi in passato ${info("fase")}</h2>
    ${regimeTable(reg)}

    <h2>Giorni insoliti recenti ${info("anomalia")}</h2>
    ${anomalyTable(t.anomalies)}`;

  const charts = [
    { key: "rv_21", label: "Quanto si muove (volatilità annua a 21 sedute)", fmt: (v) => pct(v, 0), bands: true },
    { key: "drawdown", label: "Distanza dal massimo", fmt: (v) => pct(v, 0), area: true, zero: true },
    { key: "p_turbolento", label: "Probabilità di fase agitata", fmt: (v) => pct(v, 0), yMin: 0, yMax: 1, area: true },
    { key: "beta_252", label: "Quanto segue il Nasdaq (beta a 1 anno)", fmt: (v) => num(v, 2), zero: false },
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

function earningsBlock(e) {
  if (!e?.available) return "";
  const nxt = e.next;
  return `<h2>Trimestrali ${info("trimestrale")}</h2>
    <div class="cards">
      <div class="card"><h3>Prossima</h3>
        ${nxt ? `<div class="big">${fmtDate(nxt.date)}</div><div class="small muted">${esc(nxt.timing)} · reazione ${nxt.sessions_until <= 1 ? "domani" : "tra circa " + nxt.sessions_until + " sedute"}</div>`
              : `<div class="muted">data non ancora nota</div>`}</div>
      <div class="card"><h3>Di solito il giorno dopo i conti</h3>
        <div class="big num">±${pct(simple(e.typical_move), 1)}</div>
        <div class="small muted">contro ±${pct(simple(e.typical_day_move), 1)} di un giorno qualsiasi (${num(e.move_ratio, 1)} volte). Saliti nel ${pct(e.share_up, 0)} di ${e.n_reactions} casi: la direzione è quasi una moneta.</div></div>
    </div>
    ${e.history?.length ? `<details class="small"><summary>Ultime reazioni</summary><div class="table-scroll"><table>
      <thead><tr><th class="left">Conti del</th><th class="left">Quando</th><th>Reazione</th><th>Rispetto al Nasdaq</th></tr></thead>
      <tbody>${[...e.history].reverse().map((h) => `<tr><td class="left">${fmtDate(h.date)}</td><td class="left">${esc(h.timing)}</td>
        <td class="num ${signClass(h.reaction)}">${pct(simple(h.reaction), 1, true)}</td><td class="num ${signClass(h.reaction_vs_bench)}">${pct(simple(h.reaction_vs_bench), 1, true)}</td></tr>`).join("")}</tbody>
    </table></div></details>` : ""}`;
}

function regimeTable(reg) {
  if (!reg?.available) return `<p class="notice">Fasi non disponibili (storico insufficiente).</p>`;
  return `<div class="table-scroll"><table>
    <thead><tr><th class="left">Fase</th><th>Quota del tempo</th><th>Rendimento medio annuo</th><th>Oscillazione al mese</th><th>Durata tipica</th></tr></thead>
    <tbody>${reg.states.map((s) => `<tr><td class="left">${badge(s.label)}</td><td class="num">${pct(s.share_of_days, 0)}</td>
      <td class="num ${signClass(s.ret_ann)}">${pct(s.ret_ann, 1, true)}</td><td class="num">±${pct(isNum(s.vol_ann) ? s.vol_ann / Math.sqrt(12) : null, 1)}</td>
      <td class="num">${isNum(s.expected_duration_days) ? num(s.expected_duration_days, 0) + " sedute" : "–"}</td></tr>`).join("")}</tbody></table></div>
    <p class="muted small" style="margin-top:8px">Com'è andato il titolo, in media, in ciascuna fase da inizio storico a oggi. Descrive il passato: non è una previsione.</p>`;
}

function anomalyTable(an) {
  const recent = [...(an?.recent || [])].reverse();
  if (!recent.length) return `<p class="notice">Nessun giorno insolito registrato.</p>`;
  return `<p class="muted small">Ultimi 12 mesi: ${an.count_last_365_days ?? "–"}${isNum(an.count_last_365_days_earnings) ? `, di cui ${an.count_last_365_days_earnings} nei giorni delle trimestrali (spiegati dai conti)` : ""}.</p>
    <div class="table-scroll"><table>
    <thead><tr><th class="left">Data</th><th class="left">Cosa</th><th>Valore</th><th data-tip="Quante volte più dello scarto tipico dell'ultimo anno (z-score robusto). Da 4 in su è insolito.">Quanto insolito</th><th class="left">Trimestrale</th></tr></thead>
    <tbody>${recent.slice(0, 25).map((x) => `<tr><td class="left">${fmtDate(x.date)}</td><td class="left">${esc(ANOMALY_LABELS[x.kind] || x.kind)}</td>
      <td class="num">${x.kind === "volume" ? (isNum(x.value) ? "×" + num(Math.exp(x.value), 1) + " il solito" : "–") : pct(simple(x.value), 1, true)}</td>
      <td class="num">${isNum(x.z) ? num(Math.abs(x.z), 0) + " volte" : "–"}</td><td class="left">${x.earnings ? "📅 sì" : ""}</td></tr>`).join("")}</tbody>
    </table></div>`;
}

/* ------------------------------------------------------------------ pattern */

const EVENT_TEXT = {
  regime_turbolento: ["Entrata in fase agitata", "Il modello delle fasi passa a \"agitata\": movimenti ampi e nervosi."],
  uscita_turbolento: ["Uscita dalla fase agitata", "Fine di un periodo agitato: il titolo torna in fase normale o calma."],
  crollo_anomalo: ["Crollo anomalo", "Un calo in un giorno così forte (almeno 4 volte lo scarto tipico) che per quel titolo capita di rado."],
  balzo_anomalo: ["Balzo anomalo", "Un rialzo in un giorno così forte (almeno 4 volte lo scarto tipico) che per quel titolo capita di rado."],
  volume_anomalo: ["Volumi anomali", "In un giorno si scambiano molte più azioni del solito: spesso segnala una notizia."],
  vol_estrema: ["Agitazione estrema", "Il titolo si muove più che nel 95% della sua storia fino a quel giorno."],
  vol_accelerazione: ["Agitazione in aumento", "Nell'ultima settimana si muove almeno 1,5 volte più che negli ultimi 3 mesi."],
  drawdown_20: ["Calo oltre il 20% dal massimo", "Il prezzo scende per la prima volta sotto il −20% dal suo massimo storico."],
  vigilia_trimestrale: ["Settimana prima della trimestrale", "5 sedute prima della reazione ai conti trimestrali."],
  trimestrale_positiva: ["Trimestrale accolta bene", "Il giorno dei conti il titolo fa meglio del Nasdaq."],
  trimestrale_negativa: ["Trimestrale accolta male", "Il giorno dei conti il titolo fa peggio del Nasdaq."],
};

function compareBar(label, ev, base, fmt, significant, higherIsWorse = true) {
  if (!isNum(ev) || !isNum(base)) return "";
  const max = Math.max(ev, base) * 1.15 || 1;
  const worse = higherIsWorse ? ev > base : ev < base;
  const verdict = significant ? (worse ? `<span class="v-worse">più del solito</span>` : `<span class="v-better">meno del solito</span>`)
    : `<span class="muted">nessuna differenza chiara</span>`;
  return `<div class="cmp"><div class="cmp-label">${label} <span class="small">${verdict}</span></div>
    <div class="cmp-row"><span class="cmp-name">dopo l'evento</span><span class="cmp-bar ${significant ? (worse ? "worse" : "better") : ""}"><i style="width:${(ev / max) * 100}%"></i></span><span class="num">${fmt(ev)}</span></div>
    <div class="cmp-row"><span class="cmp-name">giorno qualsiasi</span><span class="cmp-bar base"><i style="width:${(base / max) * 100}%"></i></span><span class="num">${fmt(base)}</span></div></div>`;
}

function patternSentence(name, s) {
  const v = s.vol_ratio || {}, d = s.dd10 || {};
  const parts = [];
  if (v.significant) parts.push(v.excess > 0 ? "il titolo è stato più agitato del solito" : "il titolo è stato più tranquillo del solito");
  if (d.significant) parts.push(d.excess > 0 ? "un calo del 10% è stato più frequente" : "un calo del 10% è stato meno frequente");
  if (!parts.length) return `Dopo "${name.toLowerCase()}", il mese successivo non è stato diverso da un mese qualsiasi.`;
  return `Dopo "${name.toLowerCase()}", nel mese successivo ${parts.join(" e ")}.`;
}

function renderPatterns() {
  setActiveTab("patterns");
  const P = state.patterns;
  if (!P || !P.backtest) {
    app.innerHTML = `<h1>Pattern</h1><p class="notice">Statistiche non ancora disponibili: compaiono dopo il primo run di daily.yml.</p>`;
    return;
  }
  const bt = P.backtest;
  const H = `h${bt.main_horizon}`;
  const names = Object.fromEntries((state.index?.instruments || []).map((r) => [r.id, r]));
  const current = P.current || [];
  const lv = P.live;
  const x = (v) => (isNum(v) ? num(v, 2) + "×" : "–");
  const p0 = (v) => pct(v, 0);
  const p1 = (v) => pct(simple(v), 1, true);

  const cards = Object.keys(bt.event_types || {}).map((key) => {
    const st = bt.stats?.[key] || {};
    const s = st[H] || {};
    const [title, what] = EVENT_TEXT[key] || [cap(bt.event_types[key]), ""];
    if (!s.n) return `<div class="chart-card"><h3>${esc(title)}</h3><p class="muted small">${esc(what)}</p><p class="notice">Nessun caso.</p></div>`;
    const detail = (bt.horizons || []).map((h) => {
      const q = st[`h${h}`];
      if (!q || !q.n) return "";
      const cell = (m, fmt) => `<td class="num ${m?.significant ? "strong" : ""}">${fmt(m?.event)} <span class="muted small">vs ${fmt(m?.baseline)}</span></td>`;
      return `<tr><td class="left">${h} sedute</td><td class="num">${q.n}</td>${cell(q.vol_ratio, x)}${cell(q.dd10, p0)}${cell(q.ret, p1)}</tr>`;
    }).join("");
    return `<div class="chart-card pattern-card">
      <div class="chart-head"><h3>${esc(title)}</h3><span class="muted small">${s.n} casi dal 2010</span></div>
      <p class="small muted">${esc(what)}</p>
      <p class="verdict-line">${esc(patternSentence(title, s))}</p>
      ${compareBar(`Agitazione nel mese dopo ${info("oscillazione")}`, s.vol_ratio?.event, s.vol_ratio?.baseline, x, s.vol_ratio?.significant)}
      ${compareBar(`Probabilità di un calo del 10% ${info("netto")}`, s.dd10?.event, s.dd10?.baseline, p0, s.dd10?.significant)}
      <p class="small muted">Rendimento medio nel mese dopo: ${p1(s.ret?.event)} (giorno qualsiasi ${p1(s.ret?.baseline)})${s.ret?.significant ? " — differenza chiara, ma è il dato meno affidabile" : " — nessuna differenza chiara"}.</p>
      <details class="small"><summary>Dettagli per orizzonte</summary><div class="table-scroll" style="margin-top:6px"><table>
        <thead><tr><th class="left">Dopo</th><th>Casi</th><th>Agitazione</th><th>Calo ≥ 10%</th><th>Rendimento</th></tr></thead><tbody>${detail}</tbody></table></div>
        <p class="muted">In grassetto le differenze chiare. Agitazione in multipli di quella tipica del titolo (1× = normale).</p></details>
    </div>`;
  }).join("");

  const liveRows = Object.entries(lv?.by_type || {}).map(([k, v]) => `<tr><td class="left">${esc((EVENT_TEXT[k] || [bt.event_types[k] || k])[0])}</td><td class="num">${v.n}</td>
    <td class="num">${x(v.realized_vol_ratio)} <span class="muted small">atteso ${x(v.expected_vol_ratio)} · normale ${x(v.baseline_vol_ratio)}</span></td>
    <td class="num">${p0(v.realized_dd10)} <span class="muted small">atteso ${p0(v.expected_dd10)} · normale ${p0(v.baseline_dd10)}</span></td></tr>`).join("");

  app.innerHTML = `
    <h1>Pattern ${info("pattern")}</h1>
    <p class="lead">Cosa è successo <strong>in passato</strong> nel mese dopo certi eventi, confrontato con un mese qualsiasi dello stesso titolo.
      L'app usa questi risultati da sola per giudicare il rischio nella Panoramica: contano solo gli eventi con una differenza chiara.</p>

    <h2>Eventi delle ultime sedute</h2>
    ${current.length ? `<ul class="events-now">${current.map((e) => {
      const n = names[e.id] || { id: e.id, name: e.id.toUpperCase() };
      return `<li>${instName(n, true)} <span class="ev-name">${esc((EVENT_TEXT[e.type] || [bt.event_types[e.type] || e.type])[0])}</span> <span class="muted small">${fmtDate(e.date)}</span></li>`;
    }).join("")}</ul>` : `<p class="notice">Nessun evento nelle ultime sedute.</p>`}

    <details class="explain"><summary><strong>Come si incastrano i segnali</strong></summary>
      <p>Molti segnali descrivono <strong>lo stesso fatto visto da lati diversi</strong>: un crollo anomalo fa salire subito l'agitazione dell'ultima settimana
        (agitazione in aumento), se continua porta il modello a riconoscere una fase agitata e spesso coincide con la discesa sotto il −20% dal massimo.
        Tre segnali nello stesso giorno sono quindi <strong>un solo evento</strong>, non tre conferme indipendenti: per questo nel giudizio del rischio
        l'app conta al massimo un punto per gli eventi storici.</p>
      <p>Cosa insegnano i dati finora: dopo questi eventi cambia soprattutto <strong>quanto</strong> si muove il titolo (agitazione e rischio di cali),
        quasi mai <strong>in che direzione</strong>. Le trimestrali fanno eccezione solo nella settimana prima dei conti, che è più movimentata perché include il giorno della reazione.</p>
    </details>

    <section class="series-panel backtest">
      <h2 style="margin-top:0"><span class="series-tag backtest">BACKTEST</span>Cosa dice lo storico ${info("serie")}</h2>
      <p class="muted small">Eventi dal 2010 su tutti gli strumenti, solo finestre chiuse prima del periodo di test finale (${fmtDate(bt.holdout_start)}).
        Con tanti confronti qualche differenza "chiara" isolata può essere casuale: contano gli schemi coerenti.</p>
      <div class="pattern-grid">${cards}</div>
    </section>

    <section class="series-panel live">
      <h2 style="margin-top:0"><span class="series-tag live">LIVE</span>Verifica sugli eventi nuovi</h2>
      <p class="muted small">Ogni evento nuovo viene registrato con l'esito che lo storico si aspetta; dopo ${bt.main_horizon} sedute si confronta con quello vero.
        Eventi registrati: ${lv?.n_logged ?? 0}, già verificabili: ${lv?.n_evaluated ?? 0}.</p>
      ${liveRows ? `<div class="table-scroll"><table><thead><tr><th class="left">Evento</th><th>Casi</th><th>Agitazione</th><th>Calo ≥ 10%</th></tr></thead><tbody>${liveRows}</tbody></table></div>`
        : `<p class="notice">In accumulo: servono almeno ${bt.main_horizon} sedute dopo i primi eventi registrati.</p>`}
    </section>`;
}

/* ------------------------------------------------------------------ modelli */

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
    <details class="small"><summary>Per strumento</summary>
    <div class="table-scroll" style="margin-top:6px"><table>
      <thead><tr><th class="left">Strumento</th><th>Previsioni</th><th>Periodo</th>${models.map((m) => `<th>${esc(MODEL_LABELS[m])}</th>`).join("")}</tr></thead>
      <tbody>${ids.map((id) => {
        const r = data.instruments[id][h];
        return `<tr><td class="left"><a href="#/t/${esc(id)}">${esc(id.toUpperCase())}</a></td><td class="num">${r.n}</td>
          <td class="small">${fmtDate(r.first_origin)} – ${fmtDate(r.last_origin)}</td>${models.map((m) => scoreCell(r.models?.[m])).join("")}</tr>`;
      }).join("")}</tbody>
    </table></div></details>`;
}

const VERDICT_TEXT = {
  "vantaggio": ["good", "Sì: batte la frequenza storica in modo netto."],
  "nessun vantaggio dimostrato": ["warn-chip", "No: non fa meglio della semplice frequenza storica di salita."],
  "peggio della baseline": ["bad-chip", "No: fa peggio della semplice frequenza storica di salita."],
  "non valutabile": ["", "Non ancora valutabile."],
};

function directionBlock(m, h) {
  if (!m || !m.n) return `<p class="notice">Non disponibile.</p>`;
  const [cls, txt] = VERDICT_TEXT[m.verdict] || ["", m.verdict];
  const calls = m.calls || {};
  const callRow = (k, label) => calls[k]?.n ? `<tr><td class="left">${label}</td><td class="num">${pct(calls[k].share, 0)}</td>
    <td class="num">${pct(calls[k].up_rate, 0)}</td><td class="num">${pct(calls[k].base_rate, 0)}</td><td class="num ${signClass(calls[k].mean_ret)}">${pct(calls[k].mean_ret, 1, true)}</td></tr>` : "";
  const hl = h === "h21" ? "un mese (21 sedute)" : "una settimana (5 sedute)";
  return `<p><strong>Indovina la direzione a ${hl} meglio della frequenza storica?</strong> <span class="chip ${cls}">${esc(txt)}</span></p>
    <p class="small muted">${m.n.toLocaleString("it-IT")} previsioni dal ${fmtDate(m.first_origin)} al ${fmtDate(m.last_origin)}; il prezzo è salito nel ${pct(m.up_rate, 0)} dei casi.
      Errore di previsione (Brier, più basso = meglio): modello ${num(m.brier.modello, 4)}, frequenza storica ${num(m.brier.freq_storica, 4)}, moneta ${num(m.brier.moneta, 4)}
      · t di Diebold-Mariano ${num(m.dm_vs_freq_storica, 2)} · intervallo all'80% centrato nel ${pct(m.coverage_80, 0)} dei casi.</p>
    <div class="table-scroll"><table><thead><tr><th class="left">Quando il modello è…</th><th>Quota dei giorni</th><th>Poi salito</th><th>Atteso (frequenza storica)</th><th>Rendimento medio</th></tr></thead>
      <tbody>${callRow("ottimista", "più ottimista del solito")}${callRow("neutro", "in linea con il solito")}${callRow("pessimista", "più pessimista del solito")}</tbody></table></div>`;
}

function renderModels() {
  setActiveTab("models");
  const md = state.models;
  const D = state.direction;
  const h = state.horizon;
  const bt = md.backtest, lv = md.live;
  const btBlock = bt.available ? seriesBlock("backtest", bt, h) : null;
  const lvBlock = lv.available ? seriesBlock("live", lv, h) : null;
  const dlv = D?.live?.horizons || {};
  app.innerHTML = `
    <h1>Modelli</h1>
    <p class="lead">Qui si controlla se le previsioni dell'app valgono qualcosa. Ogni modello è confrontato con una regola banale (baseline):
      se non la batte, la sua previsione non va presa sul serio.</p>

    <h2>Direzione del prezzo <span class="chip warn-chip">sperimentale</span> ${info("direzione")}</h2>
    ${D ? `
    <section class="series-panel live">
      <h3><span class="series-tag live">LIVE</span>Previsioni registrate ogni sera</h3>
      <p class="muted small">${D.live?.n_logged ?? 0} previsioni nel log${D.live?.first_origin ? ` dal ${fmtDate(D.live.first_origin)}` : ""}.</p>
      ${["h21", "h5"].map((k) => dlv[k]?.n ? directionBlock(dlv[k], k) : "").join("") || `<p class="notice">In accumulo: le verifiche arrivano dopo 5 sedute (settimana) e 21 sedute (mese).</p>`}
    </section>
    <section class="series-panel backtest">
      <h3><span class="series-tag backtest">BACKTEST</span>Prova sullo storico (walk-forward)</h3>
      <p class="muted small">A ogni data il modello usa solo i dati già noti. Periodo di test finale escluso (dal ${fmtDate(D.backtest?.holdout_start)}).
        Configurazioni provate finora (tutti i moduli): ${D.backtest?.n_configurations_tried ?? "–"}.</p>
      ${directionBlock(D.backtest?.horizons?.h21, "h21")}
      <details class="small"><summary>A una settimana</summary>${directionBlock(D.backtest?.horizons?.h5, "h5")}</details>
      ${coefTable(D)}
    </section>` : `<p class="notice">Non ancora disponibile.</p>`}

    <h2>Volatilità (quanto si muoverà)</h2>
    <p class="muted">Previsione dell'agitazione ${h === "h1" ? "della prossima seduta" : "delle prossime 5 sedute"}, confrontata con la baseline
      <strong>rw</strong> (agitazione delle ultime 21 sedute): valori negativi = meglio della baseline; in verde i miglioramenti netti.</p>
    <div class="toggle" id="horizon">
      <button data-h="h1" class="${h === "h1" ? "active" : ""}">Orizzonte 1 seduta</button>
      <button data-h="h5" class="${h === "h5" ? "active" : ""}">Orizzonte 5 sedute</button>
    </div>
    <section class="series-panel live">
      <h3><span class="series-tag live">LIVE</span>Previsioni registrate ogni sera</h3>
      <p class="muted small">${lv.n_forecasts_logged || 0} previsioni nel log${lv.first_origin ? `, dal ${fmtDate(lv.first_origin)}` : ""}.</p>
      ${lvBlock || `<p class="notice">In accumulo: servono settimane prima che i numeri siano significativi.</p>`}
    </section>
    <section class="series-panel backtest">
      <h3><span class="series-tag backtest">BACKTEST</span>Walk-forward sullo storico</h3>
      <p class="muted small">Periodo di test finale escluso (dal ${fmtDate(bt.holdout_start)})${bt.use_holdout ? " — <strong>ATTENZIONE: incluso in questo run</strong>" : ""}.</p>
      ${btBlock || `<p class="notice">Backtest non ancora disponibile.</p>`}
    </section>
    <p class="muted small">QLIKE: misura dell'errore per previsioni di varianza (più bassa = meglio). Brier: errore medio al quadrato di una probabilità (0 = perfetto, 0,25 = tirare a moneta).</p>`;
  document.getElementById("horizon").addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    state.horizon = b.dataset.h;
    renderModels();
  });
}

function coefTable(D) {
  const c = D?.backtest?.horizons?.h21?.coefficients;
  if (!c) return "";
  const labels = D.feature_labels || {};
  const items = Object.entries(c).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
  return `<details class="small"><summary>Cosa ha imparato il modello (a un mese)</summary>
    <p class="muted">Peso di ogni fattore nell'ultima stima prima del test finale: positivo = spinge verso la salita, negativo verso la discesa.
      Pesi piccoli e un modello che non batte la baseline dicono la stessa cosa: questi fattori spiegano poco della direzione.</p>
    <div class="table-scroll"><table><thead><tr><th class="left">Fattore</th><th>Peso</th></tr></thead>
    <tbody>${items.map(([k, v]) => `<tr><td class="left">${esc(labels[k] || k)}</td><td class="num ${signClass(v)}">${num(v, 3, true)}</td></tr>`).join("")}</tbody></table></div></details>`;
}

/* ------------------------------------------------------------------ guida */

function renderGuide() {
  setActiveTab("guide");
  const order = ["situazione", "rischio", "oscillazione", "solito", "massimo", "andamento", "tendenza", "direzione", "intervallo",
    "beta", "fase", "trimestrale", "anomalia", "pattern", "netto", "serie"];
  app.innerHTML = `<div class="method">
    <h1>Come leggere FinanceNow</h1>
    <p class="lead">L'app guarda ogni sera prezzi e volumi, li confronta con la storia di ciascun titolo e ti dice in poche parole
      com'è messo, quanto rischia di muoversi e cosa arriva nei prossimi giorni. Qui trovi il significato di ogni termine.</p>
    <h2>Tre cose da sapere prima</h2>
    <ul>
      <li><strong>Agitazione e direzione sono due cose diverse.</strong> Prevedere <em>quanto</em> si muoverà un titolo è possibile con discreta precisione
        (l'agitazione tende a durare); prevedere <em>se</em> salirà o scenderà è molto più difficile. Per questo l'app è sicura sul rischio e prudente sulla direzione.</li>
      <li><strong>Ogni titolo ha il suo "normale".</strong> Un ±10% al mese è ordinario per Tesla e straordinario per Apple: l'app confronta sempre il titolo con la sua storia.</li>
      <li><strong>Le prove contano più delle idee.</strong> Ogni previsione viene registrata prima di sapere com'è andata e verificata dopo (serie LIVE).
        Una previsione che non batte una regola banale viene segnata come "non affidabile".</li>
    </ul>
    <h2>Glossario</h2>
    <dl class="glossary">${order.map((k) => `<dt>${esc(GLOSSARY[k][0])}</dt><dd>${renderTip(GLOSSARY[k][1])}</dd>`).join("")}</dl>
    <p class="muted">Strumento di analisi personale, non consulenza finanziaria.</p>
  </div>`;
}

/* ------------------------------------------------------------------ metodo */

function renderMethod() {
  setActiveTab("method");
  app.innerHTML = `<div class="method">
    <h1>Metodo (tecnico)</h1>
    <p>FinanceNow <strong>descrive e misura</strong> il comportamento di indici e azioni: regimi, volatilità, anomalie, rischio.
      I moduli predittivi (volatilità, direzione) sono sempre confrontati con baseline banali e registrati nella serie live.</p>
    <h2>Dati</h2>
    <ul>
      <li>Prezzi e volumi giornalieri (OHLCV) aggiustati per split e dividendi, da yfinance, dal 2010.
        Aggiornamento ogni sera dopo la chiusura di Wall Street.</li>
      <li>Date delle trimestrali da yfinance (pagina calendario di Yahoo); giorno di reazione secondo l'orario dell'annuncio.</li>
      <li>Nessun dato di bilancio, macroeconomico o di notizie.</li>
      <li>I prezzi grezzi non sono pubblicati: la dashboard mostra solo grandezze derivate.</li>
      <li>Universo fisso: Nasdaq 100, QQQ, S&amp;P 500 e le 10 azioni più capitalizzate del Nasdaq al 26/10/2026.
        <strong>Bias di selezione</strong>: sono i vincitori di oggi, il loro passato è eccezionale per costruzione.</li>
    </ul>
    <h2>Analisi</h2>
    <ul>
      <li>Tutto su log-rendimenti; ogni valore di un giorno usa solo dati fino a quel giorno.</li>
      <li><strong>Regimi</strong>: modello di Markov nascosto a 3 stati su rendimento e volatilità a 21 sedute, probabilità filtrate.</li>
      <li><strong>Anomalie</strong>: z-score robusto (mediana e MAD dell'ultimo anno), soglia |z| ≥ 4; quelle nei giorni delle trimestrali sono marcate.</li>
      <li><strong>Volatilità</strong>: GARCH, EGARCH e HAR-RV contro le baseline rw e media storica, valutati con QLIKE.</li>
      <li><strong>Direzione</strong> (sperimentale): logistica L2 su tutti i titoli con offset = frequenza storica di salita del titolo;
        walk-forward con embargo, Brier e Diebold-Mariano contro la frequenza storica.</li>
      <li><strong>Valutazione</strong>: regole esplicite (vedi "Come decide l'app" nella Panoramica) che usano solo eventi con effetto storico netto.</li>
    </ul>
    <h2>Validazione</h2>
    <ul>
      <li><span class="series-tag backtest">BACKTEST</span>walk-forward sullo storico, con un periodo finale di test escluso (dal 03/11/2025).</li>
      <li><span class="series-tag live">LIVE</span>previsioni e giudizi registrati ogni sera e valutati solo dopo: l'unica prova pulita.</li>
    </ul>
    <p class="muted">Strumento di analisi personale, non consulenza finanziaria.</p>
  </div>`;
}

/* ------------------------------------------------------------------ router */

async function route() {
  hideTip();
  state.redraw = null;
  const hash = location.hash || "#/";
  try {
    if (hash.startsWith("#/t/")) await renderTicker(decodeURIComponent(hash.slice(4)));
    else if (hash === "#/modelli") renderModels();
    else if (hash === "#/pattern") renderPatterns();
    else if (hash === "#/guida") renderGuide();
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
    [state.assessment, state.patterns, state.direction] = await Promise.all([
      getJSON("data/assessment.json").catch(() => null),
      getJSON("data/patterns.json").catch(() => null),
      getJSON("data/direction.json").catch(() => null),
    ]);
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
