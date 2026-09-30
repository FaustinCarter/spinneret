"use strict";
/* ChargeCell GUI. Plain JavaScript, no build step, works offline. */
const { Plot, decodeF32, decodeU8 } = window.ChargePlot;

// ---------------------------------------------------------------- helpers
const $ = (s, el = document) => el.querySelector(s);
function h(tag, attrs = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "class") el.className = v;
    else if (k === "html") el.innerHTML = v;
    else if (k === "value") el.value = v;
    else if (v === true) el.setAttribute(k, "");
    else el.setAttribute(k, v);
  }
  for (const kid of kids.flat(Infinity)) {
    if (kid == null || kid === false) continue;
    el.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  }
  return el;
}
function toast(msg, kind = "") {
  const t = h("div", { class: `toast ${kind}`, role: kind === "error" ? "alert" : "status" }, msg);
  $("#toasts").append(t);
  setTimeout(() => t.remove(), kind === "error" ? 7000 : 3500);
}
async function api(path, opts = {}) {
  const o = { ...opts };
  const quiet = o.quiet; delete o.quiet;
  if (o.json !== undefined) {
    o.body = JSON.stringify(o.json);
    o.headers = { "Content-Type": "application/json" };
    delete o.json;
  }
  const r = await fetch(path, o);
  const ct = r.headers.get("content-type") || "";
  const body = ct.includes("json") ? await r.json() : await r.text();
  if (!r.ok) {
    const msg = (body && body.detail) || body || r.statusText;
    if (quiet) return null;
    toast(typeof msg === "string" ? msg : JSON.stringify(msg), "error");
    throw new Error(msg);
  }
  return body;
}
const mV = v => (v * 1e3).toFixed(1);
const V4 = v => Number(v).toFixed(4);
const pct = v => `${Math.round(v * 100)}%`;
const shortId = id => (id || "").replace(/^(scan|model|job)-/, "");
const when = iso => iso ? iso.slice(0, 16).replace("T", " ") : "";
const annotatorName = () => localStorage.getItem("cc-annotator") || "";

const STATUS_LABEL = {
  FOUND: "(1,1) found",
  NOT_IN_WINDOW: "(1,1) is not in this scan",
  UNINTERPRETABLE: "Can't read this scan",
};
const KIND_STATUS_LABEL = {
  PvT: { FOUND: "One-electron point found", NOT_IN_WINDOW: "Not in this scan", UNINTERPRETABLE: "Can't read this scan" },
  tiebar: { FOUND: "Tie bar found", NOT_IN_WINDOW: "Tie bar is not in this scan", UNINTERPRETABLE: "Can't read this scan" },
};
const statusLabel = (kind, st) => (KIND_STATUS_LABEL[kind] || STATUS_LABEL)[st];
/* The three scan kinds, one model each (kinds.py has the same names). */
const KINDS = {
  PvP: { short: "Plunger vs plunger", goal: "Find the (1,1) cell of a pair of dots.", noun: "(1,1) cell" },
  PvT: { short: "Plunger vs tunnel gate", goal: "Load exactly one electron with a good tunnel rate.", noun: "one-electron point" },
  tiebar: { short: "Tie bar", goal: "Zoom on the (1,1)-(2,0) line: find the tie bar and its two triple points for readout.", noun: "tie bar" },
};
const KIND_LABEL = { PvP: "Plunger vs plunger", PvT: "Plunger vs tunnel gate", tiebar: "Tie bar (zoom on (1,1)-(2,0))" };
const kindShort = k => (KINDS[k || "PvP"] || { short: k }).short;
const SCAN_SOURCE = { api: "measurement software", file: "file", virtual_device: "practice device", synthetic: "simulated" };
const SHORT_STATUS = { FOUND: "Found", NOT_IN_WINDOW: "Not in view", UNINTERPRETABLE: "Can't read" };
const REASON_LABEL = {
  none: "(1,1) in view, electrons counted from empty",
  no_transitions: "No charge lines visible",
  occupancy_too_low: "Scan stops before (1,1)",
  no_reference: "No empty region to count electrons from",
  partially_visible: "(1,1) cut off at the edge",
  low_snr: "Too noisy",
  sensor_insensitive: "Sensor not sensitive",
  dots_merged: "Dots merged into one",
  charge_instability: "Charges jump during the scan",
  resolution_too_coarse: "Too few points",
  tunnel_rate_too_low: "Tunnel gate too closed",
  reservoir_too_open: "Tunnel gate too open",
  no_tiebar: "No tie bar in view",
};
const REASONS_BY_STATUS = {
  FOUND: ["none"],
  NOT_IN_WINDOW: ["no_transitions", "occupancy_too_low", "no_reference", "partially_visible"],
  UNINTERPRETABLE: ["low_snr", "sensor_insensitive", "dots_merged", "charge_instability", "resolution_too_coarse"],
};
const KIND_REASONS = {
  PvT: { FOUND: ["none"], NOT_IN_WINDOW: ["no_transitions", "occupancy_too_low", "no_reference", "tunnel_rate_too_low", "reservoir_too_open"],
         UNINTERPRETABLE: ["low_snr", "sensor_insensitive", "charge_instability", "resolution_too_coarse"] },
  tiebar: { FOUND: ["none"], NOT_IN_WINDOW: ["no_tiebar", "partially_visible"],
            UNINTERPRETABLE: ["low_snr", "sensor_insensitive", "dots_merged", "charge_instability", "resolution_too_coarse"] },
};
const reasonsFor = (kind, st) => (KIND_REASONS[kind] || REASONS_BY_STATUS)[st] || [];
const KIND_REASON_LABEL = {
  PvT: { none: "Empty dot, two loading lines, clean tunnel-gate range", occupancy_too_low: "Second loading line not in view",
         no_reference: "Empty dot not in view", no_transitions: "No loading lines visible" },
  tiebar: { none: "Tie bar and both triple points in view", partially_visible: "Tie bar cut off at the edge" },
};
const reasonLabel = (kind, r) => (KIND_REASON_LABEL[kind] || {})[r] || REASON_LABEL[r] || r;
const FAMILY_COLOR = { a: "#2E5AAC", b: "#13808A", interdot: "#17202B", spectator: "#7A4FB5", sensor: "#8A7A1E" };
const FAMILIES = ["a", "b", "interdot", "spectator", "sensor"];

// ---------------------------------------------------------------- shared state
const S = { scans: [], cache: new Map(), status: null, running: new Set(), pendingDraft: null, models: [] };

async function loadModels() {
  S.models = await api("/api/models", { quiet: true }) || [];
  return S.models;
}
const modelName = id => ((S.models || []).find(m => m.id === id) || {}).display_name || shortId(id);
const inUse = kind => (S.models || []).find(m => m.in_use && (m.kind || "PvP") === kind);

async function loadScanList() {
  S.scans = await api("/api/scans");
  return S.scans;
}
async function getScan(id) {
  if (S.cache.has(id)) return S.cache.get(id);
  const d = await api(`/api/scans/${encodeURIComponent(id)}`);
  const scan = {
    id, meta: d.meta, nx: d.nx, ny: d.ny, x: d.x, y: d.y, signal: decodeF32(d.signal),
    xLabel: d.meta.x_gate, yLabel: d.meta.y_gate,
  };
  if (S.cache.size > 40) S.cache.clear();
  S.cache.set(id, scan);
  return scan;
}

/* Scan chooser used by Review and Label: previous / dropdown / next. */
function scanPicker(page) {
  const sel = h("select", { "aria-label": "Scan", onchange: () => go(page, sel.value) });
  const step = d => {
    const i = S.scans.findIndex(s => s.id === sel.value);
    const j = i + d;
    if (j >= 0 && j < S.scans.length) go(page, S.scans[j].id);
  };
  const el = h("div", { class: "row" },
    h("button", { onclick: () => step(-1), title: "Newer scan" }, "\u2190"),
    sel,
    h("button", { onclick: () => step(1), title: "Older scan" }, "\u2192"));
  el.update = current => {
    sel.replaceChildren(...S.scans.map(s => h("option", { value: s.id, selected: s.id === current },
      `${when(s.created)}  ${s.x_gate}/${s.y_gate}  ${s.device}${s.label ? "  \u2713" : ""}`)));
  };
  return el;
}
function go(page, id) { location.hash = `#/${page}${id ? "/" + encodeURIComponent(id) : ""}`; }

/* Plot controls shared by Review and Label. */
function plotBar(plot, extra = []) {
  const viewSeg = h("div", { class: "seg" });
  const setMode = m => {
    plot.setMode(m);
    [...viewSeg.children].forEach(b => b.classList.toggle("on", b.dataset.m === m));
  };
  viewSeg.append(
    h("button", { "data-m": "signal", class: "on", onclick: () => setMode("signal") }, "Signal"),
    h("button", { "data-m": "gradient", onclick: () => setMode("gradient"), title: "Shows where the signal changes fastest: faint lines stand out (G)" }, "Edges"));
  const bar = h("div", { class: "plot-bar" },
    viewSeg,
    h("label", {}, "Colours", h("select", { onchange: e => plot.setCmap(e.target.value) },
      h("option", { value: "gray" }, "Grey"), h("option", { value: "viridis" }, "Viridis"))),
    h("label", {}, "Contrast", h("input", { type: "range", min: 0, max: 15, step: 0.5, value: 1,
      oninput: e => plot.setClip(+e.target.value), "aria-label": "Contrast clipping" })),
    h("label", { class: "check" }, h("input", { type: "checkbox", onchange: e => plot.setInvert(e.target.checked) }), "Invert"),
    ...extra,
    h("span", { class: "spacer" }),
    h("button", { onclick: () => plot.resetView(), title: "Scroll to zoom, Shift-drag to move" }, "Reset view"));
  bar.setMode = setMode;
  return bar;
}

// ---------------------------------------------------------------- overlay builders
function codeImage(code, nx, ny, single = false) {
  // single: one dot (PvT), its count in a; the one-electron region is highlighted
  const c = document.createElement("canvas");
  c.width = nx; c.height = ny;
  const ctx = c.getContext("2d"), img = ctx.createImageData(nx, ny);
  for (let k = 0; k < nx * ny; k++) {
    const a = code[k] >> 3, b = single ? 0 : code[k] & 7;
    if (a === 7 || b === 7) continue;
    let rgba;
    if (single ? a === 1 : a === 1 && b === 1) rgba = [46, 90, 172, 105];
    else rgba = (a + b) % 2 ? [23, 32, 43, 26] : [255, 255, 255, 18];
    img.data.set(rgba, 4 * k);
  }
  ctx.putImageData(img, 0, 0);
  return c;
}
function codeLabels(code, nx, ny, extent, minFrac = 0.012) {
  const acc = new Map();
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const c = code[j * nx + i];
    const r = acc.get(c) || { n: 0, i: 0, j: 0 };
    r.n++; r.i += i; r.j += j; acc.set(c, r);
  }
  const [xa, xb, ya, yb] = extent;
  const out = [];
  for (const [c, r] of acc) {
    if (r.n < minFrac * nx * ny) continue;
    const a = c >> 3, b = c & 7;
    if (a === 7 && b === 7) continue;
    const i = r.i / r.n, j = r.j / r.n;
    out.push({ a, b, x: xa + i * (xb - xa) / Math.max(1, nx - 1), y: ya + j * (yb - ya) / Math.max(1, ny - 1) });
  }
  return out;
}
function drawCodeLabels(ctx, plot, labels) {
  ctx.textAlign = "center"; ctx.textBaseline = "middle";
  for (const l of labels) {
    const [X, Y] = plot.toScreen(l.x, l.y);
    const txt = l.single ? String(l.a) : `(${l.a === 7 ? "?" : l.a},${l.b === 7 ? "?" : l.b})`;
    const is11 = l.single ? l.a === 1 : l.a === 1 && l.b === 1;
    ctx.font = is11 ? "700 14px system-ui" : "12px system-ui";
    ctx.lineWidth = 3; ctx.strokeStyle = "rgba(255,255,255,0.85)";
    ctx.strokeText(txt, X, Y);
    ctx.fillStyle = is11 ? "#1F3F7A" : "#17202B";
    ctx.fillText(txt, X, Y);
  }
}
function linesImage(lines, S) {
  const c = document.createElement("canvas");
  c.width = S; c.height = S;
  const ctx = c.getContext("2d"), img = ctx.createImageData(S, S);
  const maps = FAMILIES.map(f => decodeU8(lines[f]));
  const rgb = FAMILIES.map(f => [1, 3, 5].map(k => parseInt(FAMILY_COLOR[f].slice(k, k + 2), 16)));
  for (let k = 0; k < S * S; k++) {
    let best = -1, bv = 127;
    for (let f = 0; f < 5; f++) if (maps[f][k] > bv) { bv = maps[f][k]; best = f; }
    if (best >= 0) img.data.set([...rgb[best], 60 + (bv - 127) * 1.5], 4 * k);
  }
  ctx.putImageData(img, 0, 0);
  return c;
}
function dashedRect(ctx, plot, x0, x1, y0, y1, color, label) {
  const [X0, Y0] = plot.toScreen(x0, y1), [X1, Y1] = plot.toScreen(x1, y0);
  ctx.setLineDash([7, 5]); ctx.lineWidth = 2; ctx.strokeStyle = color;
  ctx.strokeRect(X0, Y0, X1 - X0, Y1 - Y0);
  ctx.setLineDash([]);
  if (label) {
    ctx.font = "600 12px system-ui"; ctx.textAlign = "left"; ctx.textBaseline = "bottom";
    ctx.lineWidth = 3; ctx.strokeStyle = "rgba(255,255,255,0.9)";
    ctx.strokeText(label, X0 + 2, Y0 - 3); ctx.fillStyle = color; ctx.fillText(label, X0 + 2, Y0 - 3);
  }
}
function arrow(ctx, X0, Y0, X1, Y1, color) {
  const d = Math.hypot(X1 - X0, Y1 - Y0);
  if (d < 12) return;
  ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 2.5;
  ctx.beginPath(); ctx.moveTo(X0, Y0); ctx.lineTo(X1, Y1); ctx.stroke();
  const ang = Math.atan2(Y1 - Y0, X1 - X0), s = 11;
  ctx.beginPath(); ctx.moveTo(X1, Y1);
  ctx.lineTo(X1 - s * Math.cos(ang - 0.4), Y1 - s * Math.sin(ang - 0.4));
  ctx.lineTo(X1 - s * Math.cos(ang + 0.4), Y1 - s * Math.sin(ang + 0.4));
  ctx.closePath(); ctx.fill();
}

// ---------------------------------------------------------------- Review page
const pages = {};
pages.review = {
  build() {
    const root = $("#page-review");
    this.picker = scanPicker("review");
    this.canvas = h("canvas", { "aria-label": "Charge stability diagram" });
    this.plot = new Plot(this.canvas);
    this.plot.panOnDrag = true;
    this.show = { cells: true, lines: false, next: true };
    const toggle = (key, label) => h("label", { class: "check" },
      h("input", { type: "checkbox", checked: this.show[key], onchange: e => { this.show[key] = e.target.checked; this.plot.render(); } }), label);
    this.plot.layers = [(ctx, p) => this.drawLayers(ctx, p)];
    this.side = h("div", { class: "stack" });
    root.append(
      h("header", {}, h("h1", {}, "Review"), this.picker,
        h("button", { class: "primary", onclick: e => this.analyze(e), title: "Let the model read this scan" }, "Analyse"),
        h("span", { class: "spacer" }),
        practiceButtons()),
      this.body = h("div", { class: "grid-2" },
        h("div", {}, h("div", { class: "plot-wrap" }, this.canvas,
          plotBar(this.plot, [toggle("cells", "Cells"), toggle("lines", "Lines"), toggle("next", "Next scan")]))),
        this.side));
    this.empty = h("div", { class: "panel empty", hidden: true },
      h("h2", {}, "No scans yet"),
      h("p", {}, "Import scan files on the Scans page, or let your measurement software send them (see PROTOCOL.md). To try ChargeCell first, start a practice device: a simulated device that you can scan as often as you like."),
      h("div", { class: "row" },
        h("a", { class: "button primary", href: "#/scans/import" }, "Import scans"),
        practiceButtons()));
    root.append(this.empty);
  },

  async enter(id) {
    await Promise.all([loadScanList(), loadModels()]);
    const has = S.scans.length > 0;
    this.body.hidden = !has; this.empty.hidden = has; this.picker.hidden = !has;
    if (!has) return;
    id = id && S.scans.find(s => s.id === id) ? id : S.scans[0].id;
    this.id = id;
    this.picker.update(id);
    this.scan = await getScan(id);
    const summary = S.scans.find(s => s.id === id);
    this.analysis = summary && summary.analysis ? await api(`/api/scans/${encodeURIComponent(id)}/analysis`, { quiet: true }) : null;
    this.prepare();
    this.plot.setScan(this.scan);
    this.renderSide();
  },

  prepare() {
    const r = this.analysis;
    this.occ = this.lines = this.labels = null;
    this.plot.extra = null;
    if (!r) return;
    const o = r.overlays;
    if (o && o.occ_code) {
      const code = decodeU8(o.occ_code);
      this.occ = codeImage(code, o.size, o.size);
      this.labels = codeLabels(code, o.size, o.size, o.extent);
      this.lines = linesImage(o.lines, o.size);
    }
    const rec = r.recommendation || {};
    const w0 = rec.next_window && rec.kind !== "fix_then_rescan" ? rec.next_window : (rec.tiebar_window || rec.retune_window);
    const w = w0 ? onAxes(w0, this.scan) : null;
    if (w) {
      let ext = [Math.min(w.x[0], this.scan.x[0]), Math.max(w.x[1], this.scan.x[this.scan.nx - 1]),
                 Math.min(w.y[0], this.scan.y[0]), Math.max(w.y[1], this.scan.y[this.scan.ny - 1])];
      if (rec.target) {
        const t = rec.target, xg = this.scan.xLabel, yg = this.scan.yLabel;
        ext = [Math.min(ext[0], t[xg]), Math.max(ext[1], t[xg]), Math.min(ext[2], t[yg]), Math.max(ext[3], t[yg])];
      }
      this.plot.extra = ext;
    }
  },

  drawLayers(ctx, plot) {
    const r = this.analysis, sc = this.scan;
    const [x0, x1, y0, y1] = plot.dataExtent();
    if (plot.extra) {             // current window outline, visible when zoomed out
      const [X0, Y0] = plot.toScreen(x0, y1), [X1, Y1] = plot.toScreen(x1, y0);
      ctx.strokeStyle = "#17202B"; ctx.lineWidth = 1; ctx.strokeRect(X0, Y0, X1 - X0, Y1 - Y0);
    }
    if (!r) return;
    drawFeatures(ctx, plot, r, sc);
    if (this.show.cells && this.occ) {
      plot.drawGridImage(this.occ, r.overlays.extent);
      drawCodeLabels(ctx, plot, this.labels);
    }
    if (this.show.lines && this.lines && r.overlays) plot.drawGridImage(this.lines, r.overlays.extent);
    if (r.cell && r.cell.polygon_v && r.status === "FOUND") {
      ctx.strokeStyle = "#2E5AAC"; ctx.lineWidth = 2.5; ctx.beginPath();
      r.cell.polygon_v.forEach(([x, y], k) => { const [X, Y] = plot.toScreen(x, y); k ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); });
      ctx.closePath(); ctx.stroke();
    }
    const kp = r.keypoints || {};
    if (kp.cell_centre && r.status === "FOUND") {
      const [X, Y] = plot.toScreen(kp.cell_centre[sc.xLabel], kp.cell_centre[sc.yLabel]);
      ctx.strokeStyle = "#fff"; ctx.lineWidth = 4;
      ctx.beginPath(); ctx.moveTo(X - 9, Y); ctx.lineTo(X + 9, Y); ctx.moveTo(X, Y - 9); ctx.lineTo(X, Y + 9); ctx.stroke();
      ctx.strokeStyle = "#1D7A4C"; ctx.lineWidth = 2; ctx.stroke();
    }
    if (kp.readout_20 && r.status === "FOUND") {
      const [X, Y] = plot.toScreen(kp.readout_20[sc.xLabel], kp.readout_20[sc.yLabel]);
      ctx.fillStyle = "#1D7A4C"; ctx.beginPath();
      ctx.moveTo(X, Y - 6); ctx.lineTo(X + 6, Y); ctx.lineTo(X, Y + 6); ctx.lineTo(X - 6, Y); ctx.closePath(); ctx.fill();
    }
    const rec = r.recommendation || {};
    if (!this.show.next) return;
    const w = rec.next_window && rec.kind !== "fix_then_rescan" ? onAxes(rec.next_window, sc) : null;
    if (w) {
      const col = rec.kind === "explore" ? "#A86400" : "#2E5AAC";
      const label = { explore: "Explore here next",
                      confirm: `Rescan to confirm (${rec.averaging || 4}x averaging)` }[rec.kind] || "Next scan";
      dashedRect(ctx, plot, w.x[0], w.x[1], w.y[0], w.y[1], col, label);
      if (rec.kind !== "confirm") {   // a confirmation rescans the same window: nothing to point at
        const [Xa, Ya] = plot.toScreen((x0 + x1) / 2, (y0 + y1) / 2);
        const [Xb, Yb] = plot.toScreen((w.x[0] + w.x[1]) / 2, (w.y[0] + w.y[1]) / 2);
        arrow(ctx, Xa, Ya, Xb, Yb, col);
      }
      if (rec.target && (rec.kind === "move" || rec.kind === "confirm")) {
        const [Xt, Yt] = plot.toScreen(rec.target[sc.xLabel], rec.target[sc.yLabel]);
        ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(Xt, Yt, 8, 0, 2 * Math.PI); ctx.stroke();
      }
    }
    if (rec.tiebar_window) {
      const t = onAxes(rec.tiebar_window, sc);
      dashedRect(ctx, plot, t.x[0], t.x[1], t.y[0], t.y[1], "#1D7A4C", "Tie-bar scan");
    }
  },

  async analyze(e) {
    if (!this.id) return;
    const btn = e && e.target;
    if (btn) btn.disabled = true;
    try {
      this.analysis = await api(`/api/scans/${encodeURIComponent(this.id)}/analyze`, { method: "POST" });
      this.prepare();
      this.plot.resetView();
      this.renderSide();
      loadScanList().then(() => this.picker.update(this.id));
    } finally { if (btn) btn.disabled = false; }
  },

  renderSide() {
    const r = this.analysis, sc = this.scan, side = this.side;
    const meta = sc.meta;
    const summary = S.scans.find(s => s.id === this.id);
    const ref = (r && r.run) || (summary && summary.run);
    const info = h("div", { class: "panel small" },
      h("div", { class: "row" }, h("b", {}, `${meta.x_gate} vs ${meta.y_gate}`), h("span", { class: "muted" }, `${sc.nx} x ${sc.ny} points`)),
      h("div", { class: "muted" }, `${meta.device}${meta.cooldown ? " \u00b7 cooldown " + meta.cooldown : ""} \u00b7 ${SCAN_SOURCE[meta.source] || meta.source} \u00b7 ${when(meta.created)}`),
      meta.notes ? h("div", { class: "muted" }, meta.notes) : null,
      ref ? h("div", {}, h("a", { href: `#/runs/${encodeURIComponent(ref.run_id)}` }, "Show this scan in the History")) : null);
    const kind = meta.kind || "PvP", using = inUse(kind);
    if (!r) {
      side.replaceChildren(h("div", { class: "panel decision" },
        h("div", { class: "status-word status-none" }, "Not analysed yet"),
        h("p", {}, `Let the model read this ${kindShort(kind).toLowerCase()} scan: it says whether the ${KINDS[kind].noun} is in view, and where to scan next if not.`),
        using ? h("div", { class: "stack", style: "gap:6px" },
          h("div", {}, h("button", { class: "primary", onclick: e => this.analyze(e) }, "Analyse this scan")),
          h("div", { class: "small muted" }, "Model: ", using.display_name, " \u00b7 ", h("a", { href: "#/models" }, "Change model")))
          : h("div", { class: "stack", style: "gap:6px" },
            h("p", { class: "warn", style: "margin:0" }, `There is no model for ${kindShort(kind).toLowerCase()} scans yet.`),
            h("div", { class: "row" }, h("a", { class: "button primary", href: "#/models" }, "Add a model"),
              h("a", { class: "button", href: `#/train/${kind}` }, "Train one")))), info);
      return;
    }
    const rec = r.recommendation || {};
    const flags = h("div", { class: "row" },
      (r.demotion && r.demotion.length) ? null : h("span", { class: "muted small" }, `The model is ${pct(r.confidence)} sure`),
      r.needs_review ? h("span", { class: "flag review", title: "The model is unsure or its checks disagreed." }, "Please check by eye") : h("span", { class: "flag ok" }, "Checks passed"));
    const decision = h("div", { class: `panel decision ${r.status}` },
      h("div", { class: `status-word status-${r.status}` }, statusLabel(r.kind, r.status)),
      h("p", {}, r.reason_text), flags,
      (r.demotion && r.demotion.length) ? h("p", { class: "warn small" }, "Not called found because " + r.demotion.join("; ") + ".") : null,
      h("div", { class: "small muted", style: "margin-top:6px" }, "Analysed with ", h("b", {}, modelName(r.model_id)),
        r.model_id !== (using || {}).id ? h("span", {}, using ? `, which is no longer in use. Analyse again to use ${using.display_name}.` : ".") : null,
        " \u00b7 ", h("a", { href: "#/models" }, "Change model")));

    const todo = h("div", { class: "panel" }, h("h2", {}, "What to do next"),
      h("p", { class: "headline" }, rec.headline || ""),
      rec.steps && rec.steps.length ? h("ul", { class: "steps" }, rec.steps.map(s => h("li", {}, s))) : null);
    const win = rec.next_window || rec.tiebar_window || rec.retune_window;
    if (win) {
      const key = rec.next_window ? "next_window" : rec.tiebar_window ? "tiebar_window" : "retune_window";
      const row = (g, a) => h("tr", {}, h("td", {}, h("b", {}, g)), h("td", { class: "num" }, V4(a[0])), h("td", {}, "to"),
        h("td", { class: "num" }, V4(a[1])), h("td", {}, "V"), h("td", { class: "num muted" }, `${a[2]} points`));
      todo.append(h("div", { class: "window-box" },
        h("div", { class: "muted small" }, { tiebar_window: "Tie-bar scan window (readout setup)", retune_window: "Rescan after changing the exchange gate" }[key] || "Next scan window"),
        h("table", {}, row(win.x_gate, win.x), row(win.y_gate, win.y))),
        h("div", { class: "row", style: "margin-top:10px" },
          h("button", { onclick: () => {
            const txt = `${win.x_gate}: ${V4(win.x[0])} to ${V4(win.x[1])} V, ${win.x[2]} points\n${win.y_gate}: ${V4(win.y[0])} to ${V4(win.y[1])} V, ${win.y[2]} points`;
            navigator.clipboard.writeText(txt).then(() => toast("Scan settings copied"));
          } }, "Copy settings"),
          h("a", { class: "button", href: `/api/v1/scans/${encodeURIComponent(this.id)}/response?download=true`, title: "The result in the chargecell/1 format your measurement code can read" }, "Download result (JSON)"),
          meta.source === "virtual_device" ? h("button", { class: "primary", onclick: e => this.runNext(e, key) }, key === "tiebar_window" ? "Take the tie-bar scan on the practice device" : "Measure it on the practice device") : null));
    }
    if (rec.warnings && rec.warnings.length) todo.append(h("div", { style: "margin-top:10px" }, rec.warnings.map(w => h("p", { class: "warn" }, w))));
    if (rec.confidence) todo.append(h("p", { class: "muted small", style: "margin-top:8px" }, `How sure this advice is: ${rec.confidence}. Based on: ${rec.basis || "the scan alone"}.`));

    const specs = (r.spectators || []).length ? h("div", { class: "panel small" }, h("h3", { style: "margin-top:0" }, "Neighbouring dots (spectators)"),
      r.spectators.map(sp => h("div", {}, h("b", {}, sp.gate), ` at ${sp.voltage != null ? V4(sp.voltage) + " V" : "unknown voltage"}: `,
        sp.status === "verified" ? h("span", { class: "flag ok" }, "one electron (matches an earlier scan)") : h("span", { class: "flag" }, sp.status === "unverified" ? "not checked yet" : sp.status)))) : null;

    const teach = h("div", { class: "panel" }, h("h2", {}, "Teach the model"),
      h("p", { class: "small muted" }, "Your answers become training data for the next model. If the result is right, save it as a label; if not, correct it on the Label page."),
      h("div", { class: "row" },
        h("button", { onclick: () => this.accept() }, "It's right: save as label"),
        h("button", { onclick: () => { S.pendingDraft = this.id; go("label", this.id); } }, "Correct it")),
      meta.source === "virtual_device" ? h("div", { style: "margin-top:10px" }, h("button", { onclick: e => this.reveal(e) }, "Reveal the true answer")) : null);

    const probs = r.probabilities ? h("details", { class: "panel" }, h("summary", {}, "How sure the model is"),
      h("div", { class: "probs", style: "margin-top:10px" },
        Object.entries(r.probabilities.status).map(([k, v]) => [h("span", {}, SHORT_STATUS[k]), h("div", { class: "bar" }, h("i", { style: `width:${v * 100}%` })), h("span", {}, pct(v))])),
      h("p", { class: "small muted", style: "margin-top:8px" },
        (r.probabilities.ref || []).length === 2 ? `Empty region visible: ${meta.x_gate} ${pct(r.probabilities.ref[0])}, ${meta.y_gate} ${pct(r.probabilities.ref[1])}. ` :
          (r.probabilities.ref || []).length === 1 ? `Empty dot visible: ${pct(r.probabilities.ref[0])}. ` : "",
        `Disagreement between the model's networks: ${r.uncertainty.mutual_info} (0 means they agree). It says "found" only at ${pct(r.found_threshold)} confidence or more.`),
      r.quality && r.quality.warnings.length ? h("p", { class: "warn small" }, r.quality.warnings.join(" ")) : null) : null;
    side.replaceChildren(...[decision, todo, specs, teach, probs, info].filter(Boolean));
  },

  async runNext(e, key = "next_window") {
    e.target.disabled = true;
    try {
      const d = await api(`/api/scans/${encodeURIComponent(this.id)}/run_next?window=${key}`, { method: "POST" });
      toast("Measured and analysed the next scan");
      go("review", d.scan_id);
    } finally { e.target.disabled = false; }
  },
  async reveal(e) {
    const t = await api(`/api/scans/${encodeURIComponent(this.id)}/truth`);
    const kind = this.scan.meta.kind || "PvP", xg = this.scan.xLabel, yg = this.scan.yLabel;
    let more = "";
    if (t.target_V) more = ` The (1,1) centre is at ${xg} = ${V4(t.target_V[0])} V, ${yg} = ${V4(t.target_V[1])} V. Spectator holds ${t.spectator_occupancy} electron(s).`;
    if (t.operating_point_V) more = ` One electron at plunger ${V4(t.operating_point_V[0])} V, tunnel gate ${V4(t.operating_point_V[1])} V; electrons load cleanly for the tunnel gate between ${V4(t.T_open_V)} and ${V4(t.T_broad_V)} V.`;
    if (t.tp_low_V) more = ` Triple points at (${V4(t.tp_low_V[0])}, ${V4(t.tp_low_V[1])}) and (${V4(t.tp_high_V[0])}, ${V4(t.tp_high_V[1])}) V; coupling ratio ${t.coupling_ratio.toFixed(2)}.`;
    e.target.replaceWith(h("p", { class: "small" }, h("b", {}, "Truth: "), `${statusLabel(kind, t.status)} (${reasonLabel(kind, t.reason)}).${more}`));
  },
  async accept() {
    const r = this.analysis;
    const d = await api(`/api/scans/${encodeURIComponent(this.id)}/draft_from_model`, { method: "POST" });
    const ann = { ...d.annotation, status: r.status, reason: r.reason, annotator: annotatorName() || "operator" };
    await api(`/api/scans/${encodeURIComponent(this.id)}/annotation`, { method: "PUT", json: ann });
    toast("Saved as a label. You can look it over on the Label page later.");
    await loadScanList(); this.picker.update(this.id);
  },
};

/* A window {x_gate, y_gate, x, y} on the scan's own axes (PvT analyses may be transposed). */
function onAxes(w, sc) {
  return w.x_gate === sc.xLabel ? w : { ...w, x_gate: w.y_gate, y_gate: w.x_gate, x: w.y, y: w.x };
}

/* Keypoints of PvT and tie-bar results (points are gate -> V, polylines are on the analysis axes). */
function drawFeatures(ctx, plot, r, sc) {
  const feats = r.features || [];
  if (!feats.length) return;
  const swap = r.scan && r.scan.x_gate !== sc.xLabel;
  const pt = (x, y) => swap ? plot.toScreen(y, x) : plot.toScreen(x, y);
  const at = p => plot.toScreen(p[sc.xLabel], p[sc.yLabel]);
  for (const f of feats) {
    if (f.polyline && f.polyline.length > 1) {
      ctx.strokeStyle = f.type === "tie_bar" ? "#1D7A4C" : FAMILY_COLOR.a;
      ctx.lineWidth = f.type === "tie_bar" ? 3.5 : 2;
      ctx.beginPath();
      f.polyline.forEach(([x, y], k) => { const [X, Y] = pt(x, y); k ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); });
      ctx.stroke();
      if (f.type === "loading_line") {
        const [X, Y] = pt(...f.polyline[f.polyline.length - 1]);
        ctx.font = "12px system-ui, sans-serif"; ctx.fillStyle = "#fff";
        ctx.fillRect(X + 4, Y - 14, ctx.measureText(f.label).width + 6, 16);
        ctx.fillStyle = FAMILY_COLOR.a; ctx.fillText(f.label, X + 7, Y - 2);
      }
    }
    if (f.type === "triple_point") {
      const [X, Y] = at(f.point);
      ctx.fillStyle = "#fff"; ctx.beginPath(); ctx.arc(X, Y, 6, 0, 2 * Math.PI); ctx.fill();
      ctx.fillStyle = "#13808A"; ctx.beginPath(); ctx.arc(X, Y, 4, 0, 2 * Math.PI); ctx.fill();
    }
    if (f.type === "operating_point" || f.type === "readout_point") {
      const [X, Y] = at(f.point);
      ctx.strokeStyle = "#fff"; ctx.lineWidth = 4;
      ctx.beginPath(); ctx.moveTo(X - 9, Y); ctx.lineTo(X + 9, Y); ctx.moveTo(X, Y - 9); ctx.lineTo(X, Y + 9); ctx.stroke();
      ctx.strokeStyle = "#1D7A4C"; ctx.lineWidth = 2; ctx.stroke();
      ctx.beginPath(); ctx.arc(X, Y, 7, 0, 2 * Math.PI); ctx.stroke();
    }
  }
}

/* "New practice device" with a choice of scan kind. */
function practiceButtons() {
  const kind = h("select", { "aria-label": "Practice scan kind", title: "What to practise" },
    ...Object.entries(KIND_LABEL).map(([k, t]) => h("option", { value: k }, t)));
  return h("span", { class: "row", style: "gap:6px" }, h("button", { onclick: () => startPractice(kind.value) }, "New practice device"), kind);
}

async function startPractice(kind = "PvP") {
  const d = await api("/api/virtual", { method: "POST", json: { kind } });
  toast(`Practice device ${d.device} created. Its first ${KIND_LABEL[kind] || kind} scan is ready.`);
  await loadScanList();
  go("review", d.scan_id);
}

// ---------------------------------------------------------------- Label page
const TOOL_KEY = { a: "a_boundaries", b: "b_boundaries", s: "spectator_lines", e: "sensor_lines" };
const TOOL_COLOR = { a: FAMILY_COLOR.a, b: FAMILY_COLOR.b, s: FAMILY_COLOR.spectator, e: FAMILY_COLOR.sensor };

pages.label = {
  build() {
    const root = $("#page-label");
    this.picker = scanPicker("label");
    this.canvas = h("canvas", { "aria-label": "Scan to annotate", tabindex: 0 });
    this.plot = new Plot(this.canvas);
    this.plot.layers = [(ctx, p) => this.drawLayers(ctx, p)];
    this.tool = "a";
    this.showCells = true;
    this.undoStack = []; this.redoStack = [];
    this.tools = h("div", { class: "seg" });
    this.kind = "PvP";
    this.buildTools();
    this.bar = plotBar(this.plot, [h("label", { class: "check" }, h("input", { type: "checkbox", checked: true,
      onchange: e => { this.showCells = e.target.checked; this.plot.render(); } }), "Cells")]);
    this.side = h("div", { class: "stack" });
    root.append(
      h("header", {}, h("h1", {}, "Label"), this.picker,
        h("button", { onclick: () => this.draftFromModel(), title: "Start from the model's reading of this scan and correct it" }, "Start from the model's reading"),
        h("button", { onclick: () => this.undo(), title: "Ctrl+Z" }, "Undo"),
        h("button", { onclick: () => this.redo(), title: "Ctrl+Shift+Z" }, "Redo"),
        h("button", { class: "danger", onclick: () => this.clearAll() }, "Clear lines")),
      this.body = h("div", { class: "grid-2" },
        h("div", {}, h("div", { class: "tools" }, this.tools), h("div", { class: "plot-wrap" }, this.canvas, this.bar)),
        this.side));
    this.emptyEl = h("div", { class: "panel empty", hidden: true }, h("h2", {}, "Nothing to label"),
      h("p", {}, "A label is your answer for a scan: where the charge lines are, how many electrons each region holds, and what the scan shows. Labels teach the next model. Import scans first; the ones the model is least sure about are offered first."),
      h("a", { class: "button primary", href: "#/scans/import" }, "Import scans"));
    root.append(this.emptyEl);
    this.plot.hooks = {
      down: (p, e) => this.onDown(p, e), drag: p => this.onDrag(p), up: () => this.onUp(),
      hover: p => { this.hover = p; }, dblclick: () => this.finish(), context: p => this.onContext(p),
    };
    document.addEventListener("keydown", e => this.onKey(e));
    this.setTool("a");
  },

  // tools per scan kind: PvT scans have one dot, whose loading lines are drawn along the
  // plunger axis (dot-A boundaries if the plunger is on x, dot-B boundaries if it is on y)
  buildTools() {
    const toolBtn = (t, label, key, cls) => h("button", { class: `tool-${cls}`, "data-tool": t, onclick: () => this.setTool(t), title: `Shortcut: ${key}` }, label, " ", h("kbd", {}, key));
    const kind = this.kind;
    const dots = kind === "PvT"
      ? [toolBtn(this.loadTool || "a", "Loading line", "L", this.loadTool || "a")]
      : kind === "tiebar"
        ? [toolBtn("a", "Dot A 1\u21922", "A", "a"), toolBtn("b", "Dot B 0\u21921", "B", "b")]
        : [toolBtn("a", "Dot A boundary", "A", "a"), toolBtn("b", "Dot B boundary", "B", "b")];
    this.tools.replaceChildren(toolBtn("v", "Select / move", "V", "v"), ...dots,
      toolBtn("s", "Spectator line", "S", "s"), toolBtn("e", "Sensor line", "E", "e"));
    this.tools.title = "Spectator line: a charge line of a neighbouring dot that is not being swept. Sensor line: a line caused by the charge sensor itself, not by the dots.";
  },

  async enter(id) {
    await loadScanList();
    this.queue = await api("/api/label_queue");
    const has = S.scans.length > 0;
    this.body.hidden = !has; this.emptyEl.hidden = has; this.picker.hidden = !has;
    if (!has) return;
    if (!id || !S.scans.find(s => s.id === id)) id = (this.queue[0] || S.scans[0]).id;
    this.id = id;
    this.picker.update(id);
    this.scan = await getScan(id);
    const d = await api(`/api/scans/${encodeURIComponent(id)}/annotation`);
    this.kind = this.scan.meta.kind || "PvP";
    this.pOnX = d.plunger_on_x !== false;
    this.loadTool = this.pOnX ? "a" : "b";
    this.picking = null;
    this.buildTools();
    this.setTool(this.kind === "PvT" ? this.loadTool : "a");
    this.ann = d.annotation;
    if (!this.ann.annotator) this.ann.annotator = annotatorName();
    this.historyCount = d.history;
    this.setPreview(d);
    this.undoStack = []; this.redoStack = [];
    this.active = null; this.selected = null; this.drag = null;
    this.plot.setScan(this.scan);
    this.renderSide();
    if (S.pendingDraft === id) { S.pendingDraft = null; await this.draftFromModel(); }
  },

  setPreview(d) {
    this.suggestion = d.suggestion;
    const code = decodeU8(d.occ_code);
    const sc = this.scan, single = this.kind === "PvT";
    const ext = [sc.x[0], sc.x[sc.nx - 1], sc.y[0], sc.y[sc.ny - 1]];
    this.cells = codeImage(code, sc.nx, sc.ny, single);
    this.cellLabels = codeLabels(code, sc.nx, sc.ny, ext, 0.008)
      .filter(l => !single || l.a !== 7).map(l => single ? { ...l, single: true } : l);
    this.cellExtent = ext;
  },

  setTool(t) {
    this.finish();
    this.tool = t;
    [...this.tools.children].forEach(b => b.classList.toggle("on", b.dataset.tool === t));
    this.canvas.style.cursor = t === "v" ? "default" : "crosshair";
  },

  snapshot() { this.undoStack.push(JSON.stringify(this.ann)); this.redoStack = []; if (this.undoStack.length > 200) this.undoStack.shift(); },
  undo() { if (!this.undoStack.length) return; this.redoStack.push(JSON.stringify(this.ann)); this.ann = JSON.parse(this.undoStack.pop()); this.active = null; this.changed(); },
  redo() { if (!this.redoStack.length) return; this.undoStack.push(JSON.stringify(this.ann)); this.ann = JSON.parse(this.redoStack.pop()); this.active = null; this.changed(); },

  changed(sideToo = true) {
    this.plot.render();
    if (sideToo) this.renderSide();
    clearTimeout(this._pv);
    this._pv = setTimeout(async () => {
      const d = await api(`/api/scans/${encodeURIComponent(this.id)}/annotation/preview`, { method: "POST", json: this.ann });
      this.setPreview(d); this.plot.render(); this.updateSuggestion();
    }, 180);
  },

  hit(p) {
    let best = null;
    for (const [t, key] of Object.entries(TOOL_KEY)) {
      (this.ann[key] || []).forEach((poly, idx) => poly.forEach(([x, y], vi) => {
        const [X, Y] = this.plot.toScreen(x, y), d = Math.hypot(X - p.X, Y - p.Y);
        if (d < 8 && (!best || d < best.d)) best = { t, idx, vi, d };
      }));
    }
    if (best) return best;
    for (const [t, key] of Object.entries(TOOL_KEY)) {
      (this.ann[key] || []).forEach((poly, idx) => {
        for (let k = 0; k + 1 < poly.length; k++) {
          const [X0, Y0] = this.plot.toScreen(...poly[k]), [X1, Y1] = this.plot.toScreen(...poly[k + 1]);
          const L2 = (X1 - X0) ** 2 + (Y1 - Y0) ** 2 || 1;
          const u = Math.max(0, Math.min(1, ((p.X - X0) * (X1 - X0) + (p.Y - Y0) * (Y1 - Y0)) / L2));
          const d = Math.hypot(p.X - (X0 + u * (X1 - X0)), p.Y - (Y0 + u * (Y1 - Y0)));
          if (d < 6 && (!best || d < best.d)) best = { t, idx, vi: null, seg: k, d };
        }
      });
    }
    return best;
  },

  onDown(p, e) {
    if (this.picking) {                     // PvT: set an end of the clean tunnel-gate range
      this.snapshot();
      const T = this.pOnX ? p.y : p.x;
      const rng = [...(this.ann.clean_T || [null, null])];
      rng[this.picking === "lo" ? 0 : 1] = T;
      if (rng[0] != null && rng[1] != null && rng[0] > rng[1]) rng.reverse();
      this.ann.clean_T = rng;
      this.picking = null;
      this.canvas.style.cursor = this.tool === "v" ? "default" : "crosshair";
      this.changed();
      return true;
    }
    if (this.tool === "v") {
      const hit = this.hit(p);
      this.selected = hit ? { t: hit.t, idx: hit.idx } : null;
      if (hit && hit.vi != null) { this.snapshot(); this.drag = hit; }
      else if (hit && e.altKey) {           // Alt-click on a segment inserts a vertex
        this.snapshot();
        this.ann[TOOL_KEY[hit.t]][hit.idx].splice(hit.seg + 1, 0, [p.x, p.y]);
        this.drag = { ...hit, vi: hit.seg + 1 };
      }
      this.plot.render();
      return true;
    }
    const key = TOOL_KEY[this.tool];
    this.snapshot();
    if (!this.active) {
      this.ann[key] = this.ann[key] || [];
      this.ann[key].push([[p.x, p.y]]);
      this.active = { t: this.tool, idx: this.ann[key].length - 1 };
    } else {
      this.ann[key][this.active.idx].push([p.x, p.y]);
    }
    this.plot.render();
    return true;
  },
  onDrag(p) {
    if (!this.drag) return;
    this.ann[TOOL_KEY[this.drag.t]][this.drag.idx][this.drag.vi] = [p.x, p.y];
  },
  onUp() { if (this.drag) { this.drag = null; this.changed(); } },
  onContext(p) {
    const hit = this.hit(p);
    if (!hit || hit.vi == null) return;
    this.snapshot();
    const poly = this.ann[TOOL_KEY[hit.t]][hit.idx];
    poly.splice(hit.vi, 1);
    if (poly.length < 2) this.ann[TOOL_KEY[hit.t]].splice(hit.idx, 1);
    this.selected = null; this.changed();
  },
  finish() {
    if (!this.active) return;
    const key = TOOL_KEY[this.active.t], poly = this.ann[key][this.active.idx];
    // drop near-duplicate trailing points from a double-click
    while (poly.length > 1) {
      const [X0, Y0] = this.plot.toScreen(...poly[poly.length - 2]), [X1, Y1] = this.plot.toScreen(...poly[poly.length - 1]);
      if (Math.hypot(X1 - X0, Y1 - Y0) < 4) poly.pop(); else break;
    }
    if (poly.length < 2) this.ann[key].splice(this.active.idx, 1);
    this.active = null;
    this.changed();
  },
  clearAll() {
    this.snapshot();
    for (const key of Object.values(TOOL_KEY)) this.ann[key] = [];
    this.active = null; this.changed();
  },

  onKey(e) {
    if (location.hash.split("/")[1] !== "label" || !this.ann) return;
    const tag = (document.activeElement || {}).tagName;
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
    const k = e.key.toLowerCase();
    if ((e.ctrlKey || e.metaKey) && k === "z") { e.preventDefault(); e.shiftKey ? this.redo() : this.undo(); return; }
    if ((e.ctrlKey || e.metaKey) && k === "s") { e.preventDefault(); this.save(false); return; }
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    if (this.kind === "PvT" && ["a", "b", "l"].includes(k)) { this.setTool(this.loadTool); e.preventDefault(); }
    else if (["v", "a", "b", "s", "e"].includes(k)) { this.setTool(k); e.preventDefault(); }
    else if (k === "g") { this.bar.setMode(this.plot.mode === "signal" ? "gradient" : "signal"); }
    else if (k === "enter") { this.finish(); e.preventDefault(); }
    else if (k === "escape") { if (this.active) this.finish(); this.selected = null; this.plot.render(); }
    else if (k === "backspace" || k === "delete") {
      e.preventDefault();
      if (this.active) {
        this.snapshot();
        const poly = this.ann[TOOL_KEY[this.active.t]][this.active.idx];
        poly.pop();
        if (!poly.length) { this.ann[TOOL_KEY[this.active.t]].splice(this.active.idx, 1); this.active = null; }
        this.changed();
      } else if (this.selected) {
        this.snapshot();
        this.ann[TOOL_KEY[this.selected.t]].splice(this.selected.idx, 1);
        this.selected = null; this.changed();
      }
    }
  },

  drawLayers(ctx, plot) {
    if (!this.ann) return;
    if (this.showCells && this.cells) {
      plot.drawGridImage(this.cells, this.cellExtent);
      drawCodeLabels(ctx, plot, this.cellLabels || []);
    }
    if (this.kind === "PvT") this.drawCleanRange(ctx, plot);
    for (const [t, key] of Object.entries(TOOL_KEY)) {
      (this.ann[key] || []).forEach((poly, idx) => {
        const sel = this.selected && this.selected.t === t && this.selected.idx === idx;
        const act = this.active && this.active.t === t && this.active.idx === idx;
        ctx.strokeStyle = TOOL_COLOR[t]; ctx.lineWidth = sel ? 4 : 2.2;
        ctx.setLineDash(t === "s" || t === "e" ? [6, 4] : []);
        ctx.beginPath();
        poly.forEach(([x, y], k) => { const [X, Y] = plot.toScreen(x, y); k ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); });
        if (act && this.hover) ctx.lineTo(this.hover.X, this.hover.Y);
        ctx.stroke(); ctx.setLineDash([]);
        ctx.fillStyle = "#fff";
        poly.forEach(([x, y]) => {
          const [X, Y] = plot.toScreen(x, y);
          ctx.fillRect(X - 3.5, Y - 3.5, 7, 7); ctx.strokeStyle = TOOL_COLOR[t]; ctx.lineWidth = 1.5; ctx.strokeRect(X - 3.5, Y - 3.5, 7, 7);
        });
      });
    }
  },

  // PvT: shade the tunnel-gate values where electrons do not load cleanly
  drawCleanRange(ctx, plot) {
    const rng = this.ann.clean_T;
    if (!rng || (rng[0] == null && rng[1] == null)) return;
    const sc = this.scan, onX = this.pOnX;
    const Tmin = onX ? sc.y[0] : sc.x[0], Tmax = onX ? sc.y[sc.ny - 1] : sc.x[sc.nx - 1];
    const Pmin = onX ? sc.x[0] : sc.y[0], Pmax = onX ? sc.x[sc.nx - 1] : sc.y[sc.ny - 1];
    const pt = (P, T) => onX ? plot.toScreen(P, T) : plot.toScreen(T, P);
    const band = (t0, t1, fill, text) => {
      const [X0, Y0] = pt(Pmin, t0), [X1, Y1] = pt(Pmax, t1);
      ctx.fillStyle = fill;
      ctx.fillRect(Math.min(X0, X1), Math.min(Y0, Y1), Math.abs(X1 - X0), Math.abs(Y1 - Y0));
      const [Xm, Ym] = pt(Pmin + 0.5 * (Pmax - Pmin), 0.5 * (t0 + t1));
      ctx.fillStyle = "#7A4A00"; ctx.font = "600 12px system-ui"; ctx.textAlign = "center";
      ctx.fillText(text, Xm, Ym);
    };
    const [lo, hi] = rng;
    if (lo != null && lo > Tmin) band(Tmin, Math.min(lo, Tmax), "rgba(168,100,0,0.14)", "tunnel gate too closed");
    if (hi != null && hi < Tmax) band(Math.max(hi, Tmin), Tmax, "rgba(168,50,42,0.12)", "tunnel gate too open");
    ctx.strokeStyle = "#A86400"; ctx.lineWidth = 2; ctx.setLineDash([7, 4]);
    for (const T of [lo, hi]) {
      if (T == null) continue;
      const [X0, Y0] = pt(Pmin, T), [X1, Y1] = pt(Pmax, T);
      ctx.beginPath(); ctx.moveTo(X0, Y0); ctx.lineTo(X1, Y1); ctx.stroke();
    }
    ctx.setLineDash([]);
  },

  updateSuggestion() {
    if (!this.sugEl) return;
    const s = this.suggestion;
    this.sugEl.replaceChildren(s ? h("span", {}, "From your lines: ", h("b", {}, SHORT_STATUS[s.status]), ` (${reasonLabel(this.kind, s.reason)}) `,
      (this.ann.status !== s.status || this.ann.reason !== s.reason) ? h("button", { onclick: () => { this.ann.status = s.status; this.ann.reason = s.reason; this.renderSide(); } }, "Use this") : null) : "");
  },

  renderSide() {
    const ann = this.ann, sc = this.scan;
    const offset = (key, label, help) => {
      const seg = h("div", { class: "seg", role: "group", "aria-label": label });
      [0, 1, 2, 3, null].forEach(v => seg.append(h("button", {
        class: ann[key] === v ? "on" : "", onclick: () => { this.snapshot(); ann[key] = v; this.changed(); },
      }, v === null ? "Unknown" : String(v))));
      return h("div", {}, h("div", { class: "small", style: "margin-bottom:4px" }, label), seg, h("div", { class: "small muted", style: "margin-top:3px" }, help));
    };
    const kind = this.kind;
    const statusSeg = h("div", { class: "stack", style: "gap:4px" },
      ...["FOUND", "NOT_IN_WINDOW", "UNINTERPRETABLE"].map(st => h("label", { class: "check" },
        h("input", { type: "radio", name: "ann-status", checked: ann.status === st, onchange: () => {
          ann.status = st;
          const s = this.suggestion;
          if (s && s.status === st) ann.reason = s.reason;
          else if (!reasonsFor(kind, st).includes(ann.reason)) ann.reason = reasonsFor(kind, st)[0];
          this.renderSide();
        } }), h("span", { class: `pill ${st}` }, statusLabel(kind, st)))));
    const reasonSel = h("select", { disabled: !ann.status, onchange: e => { ann.reason = e.target.value; } },
      reasonsFor(kind, ann.status).map(r => h("option", { value: r, selected: ann.reason === r }, reasonLabel(kind, r))));
    this.sugEl = h("div", { class: "small" });
    const nA = (ann.a_boundaries || []).length, nB = (ann.b_boundaries || []).length;
    const extras = `${(ann.spectator_lines || []).length} spectator, ${(ann.sensor_lines || []).length} sensor line(s).`;
    let counts;
    if (kind === "PvT") {
      const key = this.pOnX ? "a" : "b", P = this.pOnX ? sc.xLabel : sc.yLabel, T = this.pOnX ? sc.yLabel : sc.xLabel;
      const nL = (ann[`${key}_boundaries`] || []).length;
      counts = h("div", { class: "panel stack" },
        h("h2", { style: "margin:0" }, "Loading lines"),
        h("p", { class: "small muted", style: "margin:0" },
          `Draw each loading line of the dot under ${P} (where it gains an electron) ${this.pOnX ? "from bottom to top" : "from left to right"}, following it across the ${T} range. Then say how many electrons the dot holds ${this.pOnX ? "left of" : "below"} the first line.`),
        offset(`${key}_offset`, `Electrons ${this.pOnX ? "left of" : "below"} the first loading line`, "0 = the empty dot is visible (at least an electron spacing of it)"),
        h("div", { class: "small muted" }, `${nL} loading line${nL === 1 ? "" : "s"}, ${extras}`));
      const rng = ann.clean_T || [null, null];
      const setEnd = (k, v) => { this.snapshot(); const r = [...(ann.clean_T || [null, null])]; r[k] = v === "" ? null : Number(v); ann.clean_T = r; this.changed(); };
      const endField = (k, label) => h("label", {}, label, h("div", { class: "row", style: "gap:6px;flex-wrap:nowrap" },
        h("input", { type: "number", step: "any", style: "width:9em", value: rng[k] == null ? "" : String(Number(rng[k].toFixed(6))),
          onchange: e => setEnd(k, e.target.value) }), h("span", { class: "small" }, "V"),
        h("button", { onclick: () => { this.picking = k ? "hi" : "lo"; this.canvas.style.cursor = "cell"; toast(`Click on the plot at the ${T} value where electrons ${k ? "stop" : "start"} loading cleanly.`); } }, "Pick on plot")));
      this.rangeEl = h("div", { class: "panel stack" },
        h("h2", { style: "margin:0" }, `Clean ${T} range`),
        h("p", { class: "small muted", style: "margin:0" },
          `Where electrons load cleanly: the loading lines are sharp and continuous. Below this range they fade or jump sideways (${T} too closed); above it they smear out (${T} too open). Leave a field empty if the clean range continues beyond the window.`),
        endField(0, "Clean from"), endField(1, "Clean up to"),
        h("div", {}, h("button", { onclick: () => { this.snapshot(); ann.clean_T = null; this.changed(); } }, "Clear range")));
    } else {
      const tb = kind === "tiebar";
      counts = h("div", { class: "panel stack" },
        h("h2", { style: "margin:0" }, "Electron counts"),
        h("p", { class: "small muted", style: "margin:0" }, tb
          ? `Draw dot A's boundary where it goes from 1 to 2 electrons, bottom to top: along the ${sc.xLabel} line, along the tie bar, and on. Draw dot B's 0\u21921 boundary left to right, through the same tie bar. The counts below are those of a tie-bar zoom; change them only if your window shows other cells.`
          : `Draw each dot A boundary bottom to top (it follows the ${sc.xLabel} line and its interdot jogs), and each dot B boundary left to right. Then say how many electrons sit left of / below them.`),
        offset("a_offset", `Dot A (${sc.xLabel}) left of all A boundaries`, tb ? "1 in a tie-bar zoom" : "0 = empty region visible"),
        offset("b_offset", `Dot B (${sc.yLabel}) below all B boundaries`, tb ? "0 in a tie-bar zoom" : "Unknown if no empty region is in view"),
        h("div", { class: "small muted" }, `${nA} A boundar${nA === 1 ? "y" : "ies"}, ${nB} B boundar${nB === 1 ? "y" : "ies"}, ${extras}`));
      this.rangeEl = null;
    }
    this.side.replaceChildren(
      ...[counts, this.rangeEl].filter(Boolean),
      h("div", { class: "panel stack" },
        h("h2", { style: "margin:0" }, "Outcome"), statusSeg,
        h("label", {}, "Reason", reasonSel), this.sugEl,
        h("label", {}, "Your name", h("input", { value: ann.annotator || "", oninput: e => { ann.annotator = e.target.value; localStorage.setItem("cc-annotator", e.target.value); } })),
        h("label", {}, "Notes", h("textarea", { rows: 2, oninput: e => { ann.notes = e.target.value; } }, ann.notes || "")),
        h("label", { class: "check" }, h("input", { type: "checkbox", checked: !!ann.reviewed, onchange: e => { ann.reviewed = e.target.checked; } }), "Checked by a second person"),
        h("div", { class: "row" },
          h("button", { class: "primary", onclick: () => this.save(false) }, "Save label"),
          h("button", { onclick: () => this.save(true) }, "Save and next")),
        h("div", { class: "small muted" }, `${this.historyCount || 0} saved version(s). Started from: ${(ann.origin || "manual").startsWith("model") ? "the model's reading" : (ann.origin || "manual") === "manual" ? "scratch" : ann.origin === "api" ? "your measurement software" : ann.origin}.`)),
      h("details", { class: "panel" }, h("summary", {}, "Shortcuts"),
        h("div", { class: "shortcut-list", style: "margin-top:8px" },
          h("kbd", {}, "A B S E"), h("span", {}, "draw dot A / dot B / spectator / sensor lines (L for loading lines)"),
          h("kbd", {}, "click"), h("span", {}, "add a point; double-click or Enter finishes"),
          h("kbd", {}, "V"), h("span", {}, "select and drag points; Alt-click inserts a point"),
          h("kbd", {}, "right-click"), h("span", {}, "delete a point"),
          h("kbd", {}, "Backspace"), h("span", {}, "remove last point / selected line"),
          h("kbd", {}, "Ctrl Z"), h("span", {}, "undo (Shift for redo)"),
          h("kbd", {}, "G"), h("span", {}, "switch to the edge view (makes faint lines stand out)"),
          h("kbd", {}, "scroll, Shift-drag"), h("span", {}, "zoom, pan"))),
      h("div", { class: "panel" }, h("h2", {}, "Scans to label next"),
        h("p", { class: "small muted", style: "margin:0" }, "Scans without a label, the ones the model is least sure about first. These teach it the most."),
        h("div", { class: "queue" }, (this.queue || []).map(q => h("a", { href: `#/label/${encodeURIComponent(q.id)}`, class: q.id === this.id ? "current" : "" },
          h("span", {}, `${when(q.created)} ${q.x_gate}/${q.y_gate}${(q.kind || "PvP") !== "PvP" ? " \u00b7 " + kindShort(q.kind) : ""}`),
          h("span", { class: `pill ${(q.analysis || {}).status || ""}` }, q.analysis ? SHORT_STATUS[q.analysis.status] : "not analysed"))))));
    this.updateSuggestion();
  },

  async draftFromModel() {
    const d = await api(`/api/scans/${encodeURIComponent(this.id)}/draft_from_model`, { method: "POST" });
    this.snapshot();
    const keep = { annotator: this.ann.annotator, notes: this.ann.notes };
    this.ann = { ...d.annotation, ...keep };
    this.setPreview(d);
    this.changed();
    toast("Loaded the model's reading. Correct anything that is wrong, then save.");
  },

  async save(next) {
    this.finish();
    const ann = this.ann;
    if (!ann.status) { toast("Choose an outcome before saving.", "error"); return; }
    if (!ann.annotator) { toast("Add your name, so everyone can see who made each label.", "error"); return; }
    const s = this.suggestion;
    const why = { PvT: "the empty dot, two loading lines and a clean tunnel-gate range", tiebar: "a complete tie bar inside the window" }[this.kind] ||
      "a complete (1,1) cell with an empty region for both dots";
    if (ann.status === "FOUND" && s && s.status !== "FOUND" && !confirm(`Your lines do not show ${why}. Save as FOUND anyway?`)) return;
    const d = await api(`/api/scans/${encodeURIComponent(this.id)}/annotation`, { method: "PUT", json: ann });
    this.historyCount = d.history;
    toast("Label saved");
    S.cache.delete(this.id);
    if (next) {
      const q = await api("/api/label_queue");
      const nxt = q.find(x => x.id !== this.id);
      if (nxt) go("label", nxt.id); else { toast("All scans are labelled."); this.enter(this.id); }
    } else {
      this.queue = await api("/api/label_queue");
      await loadScanList(); this.picker.update(this.id); this.renderSide();
    }
  },
};

// ---------------------------------------------------------------- Scans page
pages.scans = {
  build() {
    const root = $("#page-scans");
    this.filters = { device: "", labelled: "", source: "", kind: "" };
    this.importPanel = this.buildImport();
    this.importPanel.hidden = true;
    this.table = h("div", { class: "table-wrap" });
    this.filterBar = h("div", { class: "row", style: "margin-bottom:10px" });
    root.append(
      h("header", {}, h("h1", {}, "Scans"),
        h("button", { class: "primary", onclick: () => { this.importPanel.hidden = !this.importPanel.hidden; } }, "Import scan files"),
        h("button", { onclick: () => this.analyzeAll(), title: "Let the models read every scan that has not been analysed yet" }, "Analyse all new scans"),
        h("span", { class: "spacer" }),
        practiceButtons()),
      this.importPanel, this.filterBar, this.table);
  },
  buildImport() {
    const f = {};
    const field = (label, el, key) => { f[key] = el; return h("label", {}, label, el); };
    const files = h("input", { type: "file", multiple: true, accept: ".npz,.json,.csv,.txt,.dat,.tsv" });
    const result = h("div", { class: "small" });
    const panel = h("div", { class: "panel stack", style: "margin-bottom:14px" },
      h("h2", { style: "margin:0" }, "Import scans"),
      h("p", { class: "small muted", style: "margin:0" },
        "Files your measurement software wrote in ChargeCell's format (.json) already say which gates were swept. For other files (.npz, .csv, .txt, .dat), fill in the gates below. A CSV table: first row = voltages of the horizontal gate, first column = voltages of the vertical gate. Or three columns: x voltage, y voltage, signal."),
      files,
      h("div", { class: "fields" },
        field("X gate (horizontal)", h("input", { placeholder: "e.g. P1" }), "x_gate"),
        field("Y gate (vertical)", h("input", { placeholder: "e.g. P2" }), "y_gate"),
        field("Device", h("input", { value: "default" }), "device"),
        field("Cooldown", h("input", { placeholder: "e.g. CD7" }), "cooldown"),
        field("Voltages in file", h("select", {}, h("option", { value: "V" }, "volts"), h("option", { value: "mV" }, "millivolts")), "axis_units"),
        field("Scan kind", h("select", { title: "Plunger vs tunnel gate: a plunger on one axis and the tunnel gate to its reservoir on the other. Tie bar: a plunger vs plunger zoom on the (1,1)-(2,0) line." },
          h("option", { value: "" }, "as in the file (else plunger vs plunger)"), ...Object.entries(KIND_LABEL).map(([k, t]) => h("option", { value: k }, t))), "kind"),
        field("Notes", h("input", {}), "notes")),
      h("p", { class: "small muted", style: "margin:0" }, "Please fill in the cooldown. When a model is tested, scans from one cooldown are kept together (all for training or all for testing), so the test results stay honest."),
      h("div", { class: "row" }, h("button", { class: "primary", onclick: async e => {
        if (!files.files.length) { toast("Choose one or more files first.", "error"); return; }
        const fd = new FormData();
        for (const file of files.files) fd.append("files", file);
        for (const [k, el] of Object.entries(f)) fd.append(k, el.value);
        e.target.disabled = true;
        try {
          const r = await api("/api/scans/import", { method: "POST", body: fd });
          result.replaceChildren(h("p", {}, `Imported ${r.imported.length} scan(s).`),
            ...r.errors.map(er => h("p", { class: "warn" }, er)));
          if (r.imported.length) toast(`Imported ${r.imported.length} scan(s)`);
          files.value = "";
          this.enter();
        } finally { e.target.disabled = false; }
      } }, "Import")), result);
    return panel;
  },
  async enter(arg) {
    if (arg === "import") this.importPanel.hidden = false;
    await loadScanList();
    const devices = [...new Set(S.scans.map(s => s.device))];
    const sel = (key, opts, label) => h("label", { class: "row", style: "gap:6px" }, label, h("select", {
      onchange: e => { this.filters[key] = e.target.value; this.renderTable(); } },
      opts.map(([v, t]) => h("option", { value: v, selected: this.filters[key] === v }, t))));
    this.filterBar.replaceChildren(
      sel("device", [["", "All devices"], ...devices.map(d => [d, d])], "Device"),
      sel("labelled", [["", "All"], ["yes", "Labelled"], ["no", "Not labelled"]], "Label"),
      sel("source", [["", "Anywhere"], ["api", "Measurement software"], ["file", "Imported files"], ["virtual_device", "Practice devices"]], "From"),
      sel("kind", [["", "All kinds"], ...Object.keys(KINDS).map(k => [k, kindShort(k)])], "Kind"),
      h("span", { class: "muted small" }, `${S.scans.length} scans`));
    this.renderTable();
  },
  renderTable() {
    const f = this.filters;
    const rows = S.scans.filter(s => (!f.device || s.device === f.device) &&
      (!f.source || s.source === f.source) && (!f.kind || (s.kind || "PvP") === f.kind) &&
      (!f.labelled || (f.labelled === "yes") === !!(s.label && s.label.status)));
    if (!S.scans.length) {
      this.table.replaceChildren(h("div", { class: "empty" }, h("h2", {}, "No scans yet"),
        h("p", {}, "Import scan files (button above), let your measurement software send them, or start a practice device: a simulated device you can scan as often as you like.")));
      return;
    }
    this.table.replaceChildren(h("table", { class: "list" },
      h("thead", {}, h("tr", {}, ["Measured", "Gates", "Kind", "Device", "From", "Your label", "ChargeCell says", ""].map(t => h("th", {}, t)))),
      h("tbody", {}, rows.map(s => {
        const a = s.analysis, l = s.label;
        return h("tr", { class: "click", onclick: e => { if (e.target.tagName !== "BUTTON") go("review", s.id); } },
          h("td", {}, when(s.created), h("div", { class: "id" }, shortId(s.id))),
          h("td", {}, `${s.x_gate} / ${s.y_gate}`, h("div", { class: "id" }, `${(s.shape || [])[1]} x ${(s.shape || [])[0]}`)),
          h("td", {}, kindShort(s.kind)),
          h("td", {}, s.device, s.cooldown ? h("div", { class: "id" }, s.cooldown) : null),
          h("td", {}, SCAN_SOURCE[s.source] || s.source),
          h("td", {}, l && l.status ? h("span", { class: `pill ${l.status}` }, SHORT_STATUS[l.status]) : h("span", { class: "muted" }, "none"),
            l && l.annotator ? h("div", { class: "id" }, l.annotator) : null),
          h("td", {}, a ? [h("span", { class: `pill ${a.status}` }, SHORT_STATUS[a.status]), " ", h("span", { class: "muted small" }, pct(a.confidence || 0)),
            a.needs_review ? h("div", {}, h("span", { class: "flag review" }, "Please check")) : null] : h("span", { class: "muted" }, "not analysed")),
          h("td", {}, h("div", { class: "row", style: "flex-wrap:nowrap" },
            h("button", { onclick: () => go("label", s.id) }, "Label"),
            h("button", { class: "danger", onclick: async () => {
              if (!confirm("Delete this scan, its label and its analysis? This cannot be undone.")) return;
              await api(`/api/scans/${encodeURIComponent(s.id)}`, { method: "DELETE" });
              this.enter();
            } }, "Delete"))));
      }))));
  },
  async analyzeAll() {
    await api("/api/analyze_all?only_new=true", { method: "POST" });
    toast("Analysing new scans in the background. Progress shows at the bottom left.");
    pollStatus();
  },
};

// ---------------------------------------------------------------- Runs page (automation tree)
const GRADE = {
  pass: ["✓", "OK"], warn: ["!", "Check"], fail: ["✗", "Failed"],
  info: ["·", "Action"], open: ["…", "In progress"],
};
const SOURCE_LABEL = { backend: "measurement software", gui: "this web page", cli: "command line", practice: "practice device" };
const gradeMark = g => h("span", { class: `grade ${g}`, title: (GRADE[g] || ["", g])[1], "aria-label": (GRADE[g] || ["", g])[1] }, (GRADE[g] || ["?"])[0]);

pages.runs = {
  build() {
    const root = $("#page-runs");
    this.filters = { device: "", source: "" };
    this.filterBar = h("div", { class: "row" });
    this.list = h("div", { class: "table-wrap runs-list" });
    this.tree = h("div", { class: "panel run-tree" });
    this.stats = h("div", { class: "panel" });
    root.append(
      h("header", {}, h("h1", {}, "History"), this.filterBar),
      h("p", { class: "muted small", style: "max-width:900px" },
        "Every scan ChargeCell analyses is recorded here. A run is one tune-up session on one device (a new run starts after 4 hours without scans). Each run lists what was measured, what ChargeCell concluded and advised, whether the next scan followed that advice, and any notes. Each step is marked when it happens: ",
        gradeMark("pass"), " OK, ", gradeMark("warn"), " a person should check, ", gradeMark("fail"), " failed, ", gradeMark("open"), " in progress."),
      h("div", { class: "runs-layout" }, this.list, this.tree),
      h("div", { style: "margin-top:16px" }, this.stats));
  },

  async enter(id) {
    const q = new URLSearchParams();
    if (this.filters.device) q.set("device", this.filters.device);
    if (this.filters.source) q.set("source", this.filters.source);
    const [rows, all] = await Promise.all([api(`/api/v1/runs?${q}`), api("/api/v1/runs")]);
    this.rows = rows;
    const devices = [...new Set(all.map(r => r.device))];
    const sel = (key, opts, label) => h("label", { class: "row", style: "gap:6px" }, label, h("select", {
      onchange: e => { this.filters[key] = e.target.value; this.enter(this.id); } },
      opts.map(([v, t]) => h("option", { value: v, selected: this.filters[key] === v }, t))));
    this.filterBar.replaceChildren(
      sel("device", [["", "All devices"], ...devices.map(d => [d, d])], "Device"),
      sel("source", [["", "Anywhere"], ...Object.entries(SOURCE_LABEL)], "Recorded from"),
      h("button", { onclick: () => this.enter(this.id) }, "Refresh"));
    this.id = id && rows.find(r => r.id === id) ? id : (rows[0] || {}).id;
    this.renderList();
    await this.renderTree();
    this.renderStats(await api(`/api/v1/run_stats?${q}`));
  },

  renderList() {
    if (!this.rows.length) {
      this.list.replaceChildren(h("div", { class: "empty" }, h("h2", {}, "No runs yet"),
        h("p", {}, "A run appears as soon as a scan is analysed: sent by your measurement software, analysed on the Review page, or measured on a practice device.")));
      return;
    }
    this.list.replaceChildren(h("table", { class: "list" },
      h("thead", {}, h("tr", {}, ["", "Run", "Goals"].map(t => h("th", {}, t)))),
      h("tbody", {}, this.rows.map(r => h("tr", { class: `click${r.id === this.id ? " current" : ""}`, onclick: () => go("runs", r.id) },
        h("td", {}, gradeMark(r.grade)),
        h("td", {}, h("div", {}, r.title), h("div", { class: "id" }, `${r.device} · ${when(r.created)} · ${r.n_scans} scan${r.n_scans === 1 ? "" : "s"} · ${SOURCE_LABEL[r.source] || r.source}`)),
        h("td", {}, h("div", { class: "stage-chips" }, r.stages.map(st => h("span", { class: `chip ${st.grade}`, title: st.title }, gradeMark(st.grade), " ", kindShort(st.kind))))))))));
  },

  async renderTree() {
    if (!this.id) { this.tree.replaceChildren(h("p", { class: "muted" }, "Select a run.")); return; }
    const run = await api(`/api/v1/runs/${encodeURIComponent(this.id)}`);
    const root = run.nodes[0];
    const rows = run.nodes.slice(1).map(n => {
      const d = n.data || {};
      const extra = [];
      if (n.type === "measure") {
        if (d.followed === "no") extra.push(h("div", { class: "warn small" }, `Did not follow the last advice: ${d.deviation}.`));
        if (d.followed === "yes") extra.push(h("div", { class: "muted small" }, "Followed the last advice."));
        extra.push(h("button", { class: "small-btn", onclick: () => go("review", d.scan_id) }, "Open scan"));
      }
      if (n.type === "analysis" && d.checks && d.checks.length) extra.push(h("div", { class: "muted small" }, "Not called found because " + d.checks.join("; ") + "."));
      if (n.type === "advice" && d.window) {
        const w = d.window;
        extra.push(h("div", { class: "muted small" }, `${w.x_gate} ${V4(w.x[0])} to ${V4(w.x[1])} V, ${w.y_gate} ${V4(w.y[0])} to ${V4(w.y[1])} V (${w.x[2]} x ${w.y[2]} points)`));
      }
      const title = n.type === "analysis" ? `${SHORT_STATUS[d.status] || d.status}${d.reason && d.reason !== "none" ? ": " + reasonLabel(d.kind, d.reason) : ""}` : n.title;
      return h("div", { class: `tnode t-${n.type}`, style: `padding-left:${(n.depth - 1) * 20 + 4}px` },
        gradeMark(n.grade),
        h("div", { class: "tbody" },
          h("div", {}, h("span", { class: "ttime" }, n.time.slice(11, 19)), " ", h("b", {}, title), n.summary ? h("span", { class: "tsum" }, " — ", n.summary) : null),
          extra));
    });
    const note = h("input", { placeholder: "Add a note to this run", style: "flex:1" });
    this.tree.replaceChildren(
      h("div", { class: "row" }, gradeMark(run.grade), h("h2", { style: "margin:0" }, root.title), h("span", { class: "spacer" }),
        h("a", { class: "button", href: `/api/v1/runs/${encodeURIComponent(run.id)}?download=true` }, "Download (JSON)"),
        h("a", { class: "button", href: `/api/v1/runs/${encodeURIComponent(run.id)}?format=text&download=true` }, "Download (text)")),
      h("p", { class: "muted small" }, `${run.device}${run.cooldown ? " · cooldown " + run.cooldown : ""} · started ${when(run.created)} · ${root.summary}${run.closed && run.closed.note ? " · " + run.closed.note : ""}`),
      h("div", { class: "tree" }, rows.length ? rows : h("p", { class: "muted" }, "Nothing recorded yet.")),
      h("div", { class: "row", style: "margin-top:12px" }, note,
        h("button", { onclick: async () => {
          if (!note.value.trim()) return;
          await api(`/api/v1/runs/${encodeURIComponent(run.id)}/events`, { method: "POST", json: { text: note.value.trim(), note: true, by: annotatorName() } });
          this.renderTree();
        } }, "Add note"),
        run.closed ? null : h("button", { onclick: async () => {
          await api(`/api/v1/runs/${encodeURIComponent(run.id)}/close`, { method: "POST", json: { result: "done" } });
          toast("Run marked as finished"); this.enter(run.id);
        } }, "Mark run finished")));
  },

  renderStats(st) {
    const kinds = Object.entries(st.per_kind || {});
    const frac = (a, b) => b ? `${a} of ${b} (${pct(a / b)})` : "–";
    const list = (items, empty) => items.length ? h("ul", { class: "steps small" }, items.slice(0, 6).map(f =>
      h("li", {}, `${kindShort(f.kind)} · ${SHORT_STATUS[f.status] || f.status}: ${reasonLabel(f.kind, f.reason)} — ${f.count}`))) : h("p", { class: "muted small" }, empty);
    const adv = st.advice_followed || { yes: 0, no: 0 }, rv = st.reviews || { agree: 0, disagree: 0 };
    this.stats.replaceChildren(
      h("h2", {}, `Across ${st.runs} run${st.runs === 1 ? "" : "s"}`),
      kinds.length ? h("div", { class: "table-wrap" }, h("table", { class: "list" },
        h("thead", {}, h("tr", {}, ["Scan kind", "Goal reached", "Scans needed (typical)", "Scans needed (9 in 10 runs)", "Got stuck", "Left unfinished", "In progress", "Wrong “found” (practice)"].map(t => h("th", {}, t)))),
        h("tbody", {}, kinds.map(([k, v]) => h("tr", {},
          h("td", {}, kindShort(k)),
          h("td", {}, frac(v.reached, v.stages - v.open)),
          h("td", {}, v.median_scans ?? "–"), h("td", {}, v.p90_scans ?? "–"),
          h("td", {}, v.stalled), h("td", {}, v.left), h("td", {}, v.open),
          h("td", { class: v.wrong_found ? "warn" : "" }, v.wrong_found)))))) : h("p", { class: "muted small" }, "No goals recorded yet."),
      h("div", { class: "two-col", style: "margin-top:12px" },
        h("div", {}, h("h3", {}, "Scans that failed or needed review"), list(st.failure_modes || [], "None.")),
        h("div", {}, h("h3", {}, "Where unfinished goals stopped"), list(st.stall_points || [], "None."))),
      h("p", { class: "small", style: "margin-top:8px" },
        `Next scans that followed the advice: ${frac(adv.yes, adv.yes + adv.no)}. `,
        `Labels that agreed with the analysis: ${frac(rv.agree, rv.agree + rv.disagree)}.`));
  },
};

// ---------------------------------------------------------------- Simulated scans page
const SIZE_HINT = { PvP: 96, PvT: 64, tiebar: 64 };
pages.synthetic = {
  build() {
    const root = $("#page-synthetic");
    const f = this.f = {
      name: h("input", { value: "sim-" + new Date().toISOString().slice(0, 10) }),
      kind: h("select", { onchange: () => { f.size.value = SIZE_HINT[f.kind.value]; } }, Object.entries(KIND_LABEL).map(([k, t]) => h("option", { value: k }, t))),
      n: h("input", { type: "number", min: 50, step: 50, value: 3000 }),
      size: h("select", {}, [64, 96, 128].map(v => h("option", { value: v, selected: v === 96 }, `${v} x ${v} pixels`))),
      preset: h("select", {}, h("option", { value: "mixed" }, "Both (linear and triangular)"),
        h("option", { value: "hrl_linear" }, "Three dots in a line"), h("option", { value: "hrl_triangle" }, "Three dots in a triangle")),
      found: h("input", { type: "number", min: 0, max: 100, value: 40 }),
      notin: h("input", { type: "number", min: 0, max: 100, value: 35 }),
      bad: h("input", { type: "number", min: 0, max: 100, value: 25 }),
      workers: h("input", { type: "number", min: 1, max: 64, value: Math.max(1, (navigator.hardwareConcurrency || 2) - 1) }),
      seed: h("input", { type: "number", value: Math.floor(Math.random() * 1e6) }),
    };
    const L = (t, el, help) => h("label", {}, t, el, help ? h("span", { class: "hint" }, help) : null);
    this.list = h("div", { class: "stack" });
    this.preview = h("div", {});
    root.append(h("header", {}, h("h1", {}, "Simulated scans")),
      h("p", { class: "muted small", style: "max-width:900px" },
        "Models learn from simulated scans, where the right answer is known exactly. Make a set of simulated scans here, then train a model on it on the Train page. The simulation covers three dots with a charge sensor that can drift, noise, charges that jump, and (for plunger vs tunnel gate scans) electrons that tunnel too slowly or too fast."),
      h("div", { class: "two-col" },
        h("div", { class: "panel stack" },
          h("h2", { style: "margin:0" }, "Make a set of simulated scans"),
          h("div", { class: "fields" }, L("Name", f.name), h("label", { class: "wide" }, "Scan kind", f.kind),
            L("Number of scans", f.n, "3000 or more for a useful model"),
            L("Image size", f.size, "Must match the model: 96 for plunger vs plunger, 64 for the others"),
            h("label", { class: "wide" }, "Device layout", f.preset)),
          h("details", {}, h("summary", {}, "More settings"),
            h("p", { class: "small muted", style: "margin:8px 0" }, "What share of the scans should show each outcome (the simulation decides the final answer for each scan):"),
            h("div", { class: "fields" }, L("Goal in view (%)", f.found), L("Goal not in view (%)", f.notin), L("Can't be read (%)", f.bad)),
            h("div", { class: "fields", style: "margin-top:8px" },
              L("Processor cores to use", f.workers), L("Random seed", f.seed, "The same seed makes the same scans"))),
          h("p", { class: "small muted", style: "margin:0" }, "Roughly 15 scans per second per processor core."),
          h("div", { class: "row" }, h("button", { class: "primary", onclick: () => this.generate() }, "Make scans"))),
        h("div", { class: "stack" }, h("h2", { style: "margin:4px 0 0" }, "Sets of simulated scans"), this.list)),
      h("div", { style: "margin-top:18px" }, this.preview));
  },
  async enter() { this.refresh(); },
  async refresh() {
    const ds = await api("/api/synthetic");
    if (!ds.length) { this.list.replaceChildren(h("p", { class: "muted" }, "None yet. Make a set to train your first model.")); return; }
    this.list.replaceChildren(...ds.map(d => {
      const c = d.counts || {}, tot = Object.values(c).reduce((a, b) => a + b, 0) || 1;
      const by = st => Object.entries(c).filter(([k]) => k.startsWith(st)).reduce((a, [, v]) => a + v, 0);
      return h("div", { class: "panel small" },
        h("div", { class: "row" }, h("b", {}, d.name), h("span", { class: "muted" }, `${kindShort(d.kind)} · ${d.n} scans · ${d.size} x ${d.size} pixels`), h("span", { class: "spacer" }),
          h("button", { onclick: () => this.showPreview(d.name) }, "Look at some"),
          h("a", { class: "button", href: `#/train/${d.kind || "PvP"}` }, "Train on it"),
          h("button", { class: "danger", onclick: async () => { if (confirm(`Delete the set ${d.name}? This cannot be undone.`)) { await api(`/api/synthetic/${d.name}`, { method: "DELETE" }); this.refresh(); } } }, "Delete")),
        h("div", { class: "countbar", title: "goal in view / not in view / can't be read" },
          h("i", { style: `width:${by("FOUND") / tot * 100}%;background:var(--found)` }),
          h("i", { style: `width:${by("NOT_IN") / tot * 100}%;background:var(--move)` }),
          h("i", { style: `width:${by("UNINT") / tot * 100}%;background:var(--fail)` })),
        h("div", { class: "muted" }, `Goal in view ${by("FOUND")}, not in view ${by("NOT_IN")}, can't be read ${by("UNINT")}. Made ${when(d.created)}.`));
    }));
  },
  async generate() {
    const f = this.f, tot = +f.found.value + +f.notin.value + +f.bad.value;
    if (tot <= 0) { toast("The three shares must add up to more than 0.", "error"); return; }
    await api("/api/synthetic", { method: "POST", json: {
      name: f.name.value, kind: f.kind.value, n: +f.n.value, size: +f.size.value, preset: f.preset.value, seed: +f.seed.value,
      workers: +f.workers.value, mix: [f.found.value / tot, f.notin.value / tot, f.bad.value / tot] } });
    toast("Making scans in the background. Progress shows at the bottom left.");
    f.seed.value = Math.floor(Math.random() * 1e6);
    pollStatus();
  },
  async showPreview(name) {
    const items = await api(`/api/synthetic/${name}/preview?n=18`);
    this.preview.replaceChildren(h("h2", {}, `Some scans from ${name}`), h("div", { class: "thumbs" }, items.map(it => {
      const c = h("canvas", { width: it.size, height: it.size });
      const v = decodeF32(it.signal), ctx = c.getContext("2d"), img = ctx.createImageData(it.size, it.size);
      const s = Array.from(v).sort((a, b) => a - b), lo = s[Math.floor(s.length * 0.01)], hi = s[Math.floor(s.length * 0.99)];
      for (let j = 0; j < it.size; j++) for (let i = 0; i < it.size; i++) {
        const t = Math.min(1, Math.max(0, (v[j * it.size + i] - lo) / (hi - lo || 1))) * 255;
        img.data.set([t, t, t, 255], 4 * ((it.size - 1 - j) * it.size + i));
      }
      ctx.putImageData(img, 0, 0);
      const m = it.meta;
      return h("div", { class: "thumb" }, c, h("div", { class: "cap" }, h("span", { class: `pill ${m.status}` }, SHORT_STATUS[m.status]),
        h("div", { class: "muted" }, reasonLabel(m.kind, m.reason)), h("div", { class: "muted" }, `${m.pair ? "dots " + m.pair.join("-") : "dot " + ((m.dot ?? 0) + 1)}${m.artifact && m.artifact !== "normal" ? ", problem added: " + reasonLabel(m.kind, m.artifact).toLowerCase() : ""}`)));
    })));
  },
};

// ---------------------------------------------------------------- small shared pieces
/* A command to copy: shown in a dark box with a Copy button. */
function cmdBox(text) {
  return h("div", { class: "cmd-row" }, h("pre", { class: "cmd" }, text),
    h("button", { onclick: () => navigator.clipboard.writeText(text).then(() => toast("Copied")) }, "Copy"));
}
/* One line on how good a model is, from its test results. */
function qualityLine(m) {
  const q = m.quality || {};
  if (q.precision == null) return "No test results stored with this model.";
  return `Right ${pct(q.precision)} of the time when it said "found"; recognised ${pct(q.recall || 0)} of the scans that showed the ${KINDS[m.kind || "PvP"].noun}` +
    ` (tested on ${q.n} ${q.source === "real" ? "of your labelled" : "simulated"} scans).`;
}
function trainedLine(m) {
  const t = m.trained_on, a = m.added;
  const bits = [];
  if (t) bits.push(`Trained on ${t.computer} (${t.device})${t.minutes != null ? ` in ${t.minutes < 90 ? Math.round(t.minutes) + " min" : (t.minutes / 60).toFixed(1) + " h"}` : ""}`);
  if (a && a.source) bits.push(`added from ${a.source}`);
  bits.push(`made ${when(m.created)}`);
  const s = bits.join(", ");
  return s.charAt(0).toUpperCase() + s.slice(1) + ".";
}
async function useModel(m) {
  const r = await api(`/api/models/${encodeURIComponent(m.id)}/activate`, { method: "POST" });
  toast(`${kindShort(r.kind)} scans are now read by "${r.name}".`);
  await loadModels();
  pollStatus();
  return r;
}
async function reanalyse(kind) {
  await api(`/api/analyze_all?only_new=false&kind=${encodeURIComponent(kind)}`, { method: "POST" });
  toast(`Analysing all ${kindShort(kind).toLowerCase()} scans again in the background. Progress shows at the bottom left.`);
  pollStatus();
}

// ---------------------------------------------------------------- Home page
pages.home = {
  build() {
    this.body = h("div", { class: "stack", style: "gap:18px" });
    $("#page-home").append(h("header", {}, h("h1", {}, "ChargeCell")), this.body);
  },
  async enter() {
    const [st, , , devices] = await Promise.all([api("/api/status"), loadModels(), loadScanList(), api("/api/devices")]);
    const task = (href, title, text, onclick) => h(onclick ? "button" : "a", { class: "card", href: onclick ? null : href, onclick },
      h("b", {}, title), h("span", {}, text));
    const kinds = Object.keys(KINDS);
    const missing = kinds.filter(k => !inUse(k));
    const nByKind = k => S.scans.filter(s => (s.kind || "PvP") === k).length;
    const devicesSet = devices.some(d => Object.keys(d.safe_limits || {}).length);
    const steps = [
      [missing.length < kinds.length, "Get a model",
        missing.length === kinds.length ? "No model is in use yet. Add a model file you were given, or train one."
          : missing.length ? `There is no model yet for ${missing.map(k => kindShort(k).toLowerCase()).join(" or ")} scans.` : "Every kind of scan has a model.",
        [h("a", { class: "button", href: "#/models" }, "Add a model file"), h("a", { class: "button", href: "#/train" }, "Train one")]],
      [st.n_scans > 0, "Get some scans",
        "Import scan files, let your measurement software send them, or try a practice device (a simulated device).",
        [h("a", { class: "button", href: "#/scans/import" }, "Import scans"), practiceButtons()]],
      [devicesSet, "Enter your device's gates and safe voltage limits",
        "Optional, but suggested scans then always stay inside your limits.",
        [h("a", { class: "button", href: "#/device" }, "Device settings")]],
      [st.n_labelled > 0, "Label a few of your own scans",
        "Where ChargeCell is wrong, correct it. Your labels are used to test and train the next model.",
        [h("a", { class: "button", href: "#/label" }, "Label scans")]],
    ];
    const allDone = steps.every(s => s[0]);
    const checklist = h(allDone ? "details" : "div", { class: "panel" },
      h(allDone ? "summary" : "h2", {}, allDone ? "Getting started (all done)" : "Getting started"),
      h("ol", { class: "checklist" }, steps.map(([done, title, text, actions]) => h("li", { class: done ? "done" : "" },
        h("span", { class: `tick${done ? " done" : ""}`, "aria-label": done ? "done" : "to do" }, done ? "✓" : ""),
        h("div", {}, h("b", {}, title), h("div", { class: "small muted" }, text),
          done ? null : h("div", { class: "row", style: "margin-top:6px" }, actions))))));
    const models = h("div", { class: "panel" },
      h("div", { class: "row" }, h("h2", { style: "margin:0" }, "Models in use"), h("span", { class: "spacer" }),
        h("a", { class: "button", href: "#/models" }, "Switch models")),
      h("table", { class: "list plain", style: "margin-top:8px" }, h("tbody", {}, kinds.map(k => {
        const m = inUse(k);
        return h("tr", {},
          h("td", {}, h("b", {}, KINDS[k].short), h("div", { class: "small muted" }, KINDS[k].goal)),
          h("td", {}, m ? [h("div", {}, m.display_name), h("div", { class: "small muted" }, qualityLine(m))]
            : h("span", { class: "muted" }, "No model yet")),
          h("td", { class: "small muted" }, `${nByKind(k)} scan${nByKind(k) === 1 ? "" : "s"}`));
      }))));
    const jobs = (st.running_jobs || []);
    this.body.replaceChildren(...[
      h("p", { class: "lead" }, "ChargeCell reads charge-stability scans of your quantum-dot device, tells you what each scan shows, and suggests where to scan next. It only gives advice: it never changes a gate voltage."),
      h("div", { class: "cards" },
        task("#/review", "Look at a scan", S.scans.length ? `See what ChargeCell reads in your latest scan and where to scan next. ${S.scans.length} scan${S.scans.length === 1 ? "" : "s"} so far.` : "No scans yet: import some or try a practice device."),
        task("#/scans/import", "Import scans", "Add scan files from your measurement computer."),
        task(null, "Try a practice device", "A simulated device: scan it, read the scan, follow the advice. Nothing real is touched.", () => startPractice("PvP")),
        task("#/label", "Label scans", "Correct ChargeCell where it is wrong. Your labels train the next model."),
        task("#/models", "Switch models", "See how good each model is and choose which one reads your scans."),
        task("#/train", "Train a model", "On this computer, or on another computer with a GPU.")),
      jobs.length ? h("div", { class: "panel" }, h("h2", {}, "Running now"),
        jobs.map(j => h("div", { class: "small", style: "margin-bottom:6px" }, h("b", {}, j.title), ": ", jobText(j),
          j.kind === "train" ? [" · ", h("a", { href: "#/train" }, "Details")] : null))) : null,
      allDone ? models : h("div", { class: "two-col" }, checklist, models),
      allDone ? checklist : null,
      h("p", { class: "small muted" }, "New to the words used here? The README has a glossary. Your measurement software can talk to ChargeCell directly: see PROTOCOL.md.")].filter(Boolean));
  },
};

// ---------------------------------------------------------------- Models page
pages.models = {
  build() {
    const root = $("#page-models");
    this.addPanel = this.buildAdd();
    this.addPanel.hidden = true;
    this.body = h("div", { class: "stack", style: "gap:18px" });
    this.banners = {};
    root.append(
      h("header", {}, h("h1", {}, "Models"),
        h("button", { class: "primary", onclick: () => { this.addPanel.hidden = !this.addPanel.hidden; } }, "Add a model file"),
        h("a", { class: "button", href: "#/train" }, "Train a new model")),
      h("p", { class: "muted small", style: "max-width:900px" },
        "Each kind of scan has its own model, and one model per kind is in use: it reads every new scan of that kind. Switching takes effect at once. Scans that were already analysed keep their result until you analyse them again. To use a model on another computer, download it here and add the file there."),
      this.addPanel, this.body);
  },
  buildAdd() {
    const zip = h("input", { type: "file", accept: ".zip" });
    const folder = h("input", { type: "file", webkitdirectory: true, multiple: true });
    const name = h("input", { placeholder: "keep the model's own name" });
    const use = h("input", { type: "checkbox", checked: true });
    const add = async e => {
      const files = zip.files.length ? [...zip.files]
        : [...folder.files].filter(f => f.name === "model.json" || /^member_\d+\.pt$/.test(f.name));
      if (!files.length) { toast("Choose a model file (.zip) or a model folder first.", "error"); return; }
      const fd = new FormData();
      files.forEach(f => fd.append("files", f, f.name));
      fd.append("use", use.checked ? "true" : "false");
      fd.append("name", name.value.trim());
      e.target.disabled = true;
      try {
        const m = await api("/api/models/add", { method: "POST", body: fd });
        toast(`Added "${m.display_name}" for ${kindShort(m.kind).toLowerCase()} scans${m.in_use ? ". It is now in use" : ""}.`);
        zip.value = ""; folder.value = ""; name.value = "";
        this.addPanel.hidden = true;
        if (m.in_use) await this.afterSwitch(m.kind);
        pollStatus();
        this.enter();
      } finally { e.target.disabled = false; }
    };
    return h("div", { class: "panel stack" },
      h("h2", { style: "margin:0" }, "Add a model"),
      h("p", { class: "small muted", style: "margin:0" },
        "A model file is a .zip that ChargeCell makes when you download a model, or when a model is trained on another computer. You can also choose a model folder: the folder with model.json and member_0.pt inside."),
      h("div", { class: "fields" },
        h("label", { class: "wide" }, "Model file (.zip)", zip),
        h("label", { class: "wide" }, "or a model folder", folder),
        h("label", { class: "wide" }, "Name (optional)", name)),
      h("label", { class: "check" }, use, "Use it now for its kind of scans"),
      h("div", { class: "row" }, h("button", { class: "primary", onclick: add }, "Add model")));
  },
  async enter() {
    const ms = await loadModels();
    this.body.replaceChildren(...Object.keys(KINDS).map(k => this.kindPanel(k, ms.filter(m => (m.kind || "PvP") === k))));
  },
  async afterSwitch(kind) {
    await loadScanList();
    const cur = inUse(kind);
    const n = S.scans.filter(s => (s.kind || "PvP") === kind && s.analysis && cur && s.analysis.model_id !== cur.id).length;
    this.banners[kind] = n ? { n, name: cur.display_name } : null;
  },
  kindPanel(kind, ms) {
    const cur = ms.find(m => m.in_use);
    const others = ms.filter(m => !m.in_use).sort((a, b) => (b.created || "").localeCompare(a.created || ""));
    const ban = this.banners[kind];
    return h("div", { class: "panel stack" },
      h("div", {}, h("h2", { style: "margin:0" }, KINDS[kind].short), h("div", { class: "small muted" }, KINDS[kind].goal)),
      ban ? h("div", { class: "banner row" },
        h("span", {}, `${ban.n} ${kindShort(kind).toLowerCase()} scan${ban.n === 1 ? " was" : "s were"} analysed by another model. Analyse ${ban.n === 1 ? "it" : "them"} again with "${ban.name}"?`),
        h("button", { class: "primary", onclick: async () => { await reanalyse(kind); this.banners[kind] = null; this.enter(); } }, "Analyse again"),
        h("button", { onclick: () => { this.banners[kind] = null; this.enter(); } }, "Not now")) : null,
      cur ? this.card(cur, true) : h("div", { class: "empty-inline" },
        h("p", { style: "margin:0 0 8px" }, "No model in use for these scans yet."),
        h("div", { class: "row" }, h("button", { onclick: () => { this.addPanel.hidden = false; this.addPanel.scrollIntoView({ behavior: "smooth" }); } }, "Add a model file"),
          h("a", { class: "button", href: `#/train/${kind}` }, "Train one"))),
      others.length ? h("div", { class: "stack", style: "gap:8px" },
        h("h3", { style: "margin:6px 0 0" }, `Other versions (${others.length})`), others.map(m => this.card(m, false))) : null);
  },
  card(m, current) {
    const rename = async () => {
      const n = prompt("New name for this model", m.display_name);
      if (n == null || !n.trim()) return;
      await api(`/api/models/${encodeURIComponent(m.id)}`, { method: "PATCH", json: { name: n.trim() } });
      toast("Renamed"); await loadModels(); this.enter(); pollStatus();
    };
    const del = async () => {
      if (!confirm(`Delete the model "${m.display_name}"? This cannot be undone. (Download it first if you might want it back.)`)) return;
      await api(`/api/models/${encodeURIComponent(m.id)}`, { method: "DELETE" });
      toast("Model deleted"); this.enter();
    };
    const use = async e => {
      e.target.disabled = true;
      try { await useModel(m); await this.afterSwitch(m.kind || "PvP"); this.enter(); } finally { e.target.disabled = false; }
    };
    const notes = (m.config || {}).notes;
    return h("div", { class: `model-card${current ? " current" : ""}` },
      h("div", { class: "row" },
        h("b", { class: "model-name" }, m.display_name),
        current ? h("span", { class: "flag ok" }, "In use") : null,
        h("span", { class: "spacer" }),
        current ? null : h("button", { class: "primary", onclick: use }, "Use this one"),
        h("button", { onclick: rename }, "Rename"),
        h("a", { class: "button", href: `/api/models/${encodeURIComponent(m.id)}/download`, title: "A .zip file you can add to ChargeCell on another computer" }, "Download"),
        current ? null : h("button", { class: "danger", onclick: del }, "Delete")),
      current ? h("ul", { class: "steps small", style: "margin-top:8px" }, (m.quality.lines || []).map(l => h("li", {}, l)))
        : h("div", { class: "small", style: "margin-top:4px" }, qualityLine(m)),
      h("div", { class: "small muted", style: "margin-top:4px" }, trainedLine(m),
        m.networks ? ` ${m.networks} network${m.networks === 1 ? "" : "s"}, images of ${m.image_size} x ${m.image_size} pixels.` : ""),
      notes ? h("div", { class: "small", style: "margin-top:4px" }, "Notes: ", notes) : null,
      m.split_note ? h("p", { class: "warn small" }, m.split_note) : null,
      this.results(m));
  },
  results(m) {
    const metr = m.metrics || {};
    const block = (name, x) => x ? h("div", {}, h("h3", {}, name === "real" ? `Your labelled scans kept out of training (${x.n})` : `Simulated scans kept out of training (${x.n})`),
      h("div", { class: "metric-grid" }, [
        ["found_precision", "of its “found” answers were right"],
        ["found_recall", "of scans with the goal in view were recognised"],
        ["status_accuracy", "of answers (found / not in view / can't read) were right"],
        ["occupancy_pixel_accuracy", "of pixels had the right electron count"],
      ].filter(([k]) => x[k] != null).map(([k, t]) => h("div", { class: "metric" }, h("b", {}, pct(x[k])), h("span", {}, t)))),
      x.confusion && x.confusion_labels ? h("table", { class: "list", style: "margin-top:8px;width:auto" },
        h("tr", {}, h("th", {}, "True answer ↓ / model said →"), x.confusion_labels.map(l => h("th", {}, SHORT_STATUS[l]))),
        x.confusion.map((row, i) => h("tr", {}, h("td", {}, SHORT_STATUS[x.confusion_labels[i]]), row.map(v => h("td", {}, v))))) : null) : null;
    if (!metr.real && !metr.synthetic) return null;
    return h("details", { style: "margin-top:6px" }, h("summary", {}, "Test results"),
      block("real", metr.real), block("synthetic", metr.synthetic),
      h("p", { class: "small muted" }, `It says "found" only when it is at least ${pct(m.found_threshold || 0)} sure.`));
  },
};

// ---------------------------------------------------------------- Train page
const TRAIN_DEFAULTS = { PvP: { epochs: 14, ensemble: 3 }, PvT: { epochs: 12, ensemble: 2 }, tiebar: { epochs: 14, ensemble: 2 } };
function jobText(j) {
  const w = j.worker || {};
  switch (j.state) {
    case "queued": return "waiting for another job on this computer to finish";
    case "preparing": return "packing the training data for the other computer";
    case "waiting": return "waiting for a training computer to pick it up";
    case "running": return j.where === "remote" ? `training on ${w.computer || "another computer"}: ${j.message}` : j.message;
    case "done": return j.result && j.result.name ? `finished: "${j.result.name}"` : "finished";
    case "failed": return `failed: ${j.error || j.message}`;
    case "cancelled": return "cancelled";
    case "interrupted": return "stopped: ChargeCell was restarted while this ran";
    default: return j.message || j.state;
  }
}
const minutesAgo = iso => iso ? Math.max(0, Math.round((Date.now() - new Date(iso).getTime()) / 60000)) : null;

pages.train = {
  build() {
    const root = $("#page-train");
    this.kind = "PvP";
    this.where = "here";
    const f = this.f = {
      name: h("input", {}),
      use_when_done: h("input", { type: "checkbox" }),
      use_real: h("input", { type: "checkbox", checked: true }),
      only_reviewed: h("input", { type: "checkbox" }),
      real_weight: h("input", { type: "number", min: 1, max: 50, value: 5 }),
      epochs: h("input", { type: "number", min: 1, max: 200, value: 14 }),
      ensemble: h("input", { type: "number", min: 1, max: 8, value: 3 }),
      base: h("select", {}, h("option", { value: 12 }, "Small (faster)"), h("option", { value: 16, selected: true }, "Standard"), h("option", { value: 24 }, "Large (slower)")),
      batch_size: h("input", { type: "number", min: 4, max: 256, value: 32 }),
      lr: h("input", { type: "number", step: 0.0005, value: 0.002 }),
      target_precision: h("input", { type: "number", min: 80, max: 99.9, step: 0.5, value: 97 }),
      notes: h("input", { placeholder: "for example: what is different about this one" }),
    };
    const L = (t, el, help) => h("label", {}, t, el, help ? h("span", { class: "hint" }, help) : null);
    const step = (n, title, ...kids) => h("div", { class: "panel stack" }, h("h2", { style: "margin:0" }, h("span", { class: "step-num" }, n), title), ...kids);
    this.kindBox = h("div", { class: "stack", style: "gap:6px" });
    this.dataBox = h("div", { class: "stack", style: "gap:6px" });
    this.whereBox = h("div", { class: "stack", style: "gap:6px" });
    this.remoteHelp = h("div", { class: "stack", style: "gap:8px" });
    this.startBtn = h("button", { class: "primary", onclick: e => this.start(e) }, "Start training");
    this.progress = h("div", { class: "stack" });
    this.recent = h("div", {});
    root.append(h("header", {}, h("h1", {}, "Train a model")),
      h("div", { class: "grid-train" },
        h("div", { class: "stack" },
          step(1, "What kind of scans is it for?", this.kindBox),
          step(2, "What should it learn from?", this.dataBox),
          step(3, "Where should it train?", this.whereBox, this.remoteHelp),
          step(4, "Name it and start",
            h("div", { class: "fields" }, h("label", { class: "wide" }, "Name", f.name)),
            h("label", { class: "check" }, f.use_when_done, "Use the new model as soon as it is ready"),
            h("div", { class: "hint" }, "Otherwise switch to it on the Models page after looking at its test results. If no model is in use for this kind of scan yet, the new one is used anyway."),
            h("details", {}, h("summary", {}, "More settings"),
              h("div", { class: "fields", style: "margin-top:8px" },
                L("Passes over the data", f.epochs, "More passes learn more, up to a point, and take longer"),
                L("Networks to average", f.ensemble, "More networks give steadier answers and take longer"),
                L("Network size", f.base),
                L("Weight of a labelled scan", f.real_weight, "How many simulated scans one of your labelled scans counts as"),
                L("“Found” must be right at least (%)", f.target_precision, "It says found only where it was right this often in testing"),
                L("Scans per training step", f.batch_size, "Lower it if the GPU runs out of memory"),
                L("Learning rate", f.lr, "Leave as it is unless training is unstable")),
              h("label", { class: "check", style: "margin-top:8px" }, f.only_reviewed, "Only use labels checked by a second person"),
              h("div", { class: "fields", style: "margin-top:8px" }, h("label", { class: "wide" }, "Notes", f.notes))),
            h("div", { class: "row" }, this.startBtn))),
        h("div", { class: "stack" }, this.progress, this.recent)));
  },

  async enter(arg) {
    if (arg && KINDS[arg]) this.setKind(arg, false);
    const [ds] = await Promise.all([api("/api/synthetic"), loadScanList(), loadModels()]);
    this.datasets = ds;
    this.renderKinds(); this.renderData(); this.renderWhere();
    this.watch();
  },
  setKind(k, render = true) {
    this.kind = k;
    const d = TRAIN_DEFAULTS[k];
    this.f.epochs.value = d.epochs; this.f.ensemble.value = d.ensemble;
    if (render) { this.renderKinds(); this.renderData(); }
  },
  renderKinds() {
    this.kindBox.replaceChildren(...Object.entries(KINDS).map(([k, info]) => {
      const m = inUse(k);
      return h("label", { class: `choice${this.kind === k ? " on" : ""}` },
        h("input", { type: "radio", name: "train-kind", checked: this.kind === k, onchange: () => this.setKind(k) }),
        h("div", {}, h("b", {}, info.short), h("div", { class: "small muted" }, info.goal,
          m ? ` In use now: "${m.display_name}".` : " No model in use yet.")));
    }));
    this.f.name.placeholder = `${KINDS[this.kind].short} model, ${new Date().toISOString().slice(0, 10)}`;
  },
  renderData() {
    const sets = (this.datasets || []).filter(d => (d.kind || "PvP") === this.kind);
    const nLab = S.scans.filter(s => (s.kind || "PvP") === this.kind && s.label && s.label.status).length;
    const size0 = sets.length ? sets[0].size : null;
    this.setChecks = sets.map(d => h("input", { type: "checkbox", value: d.name, "data-size": d.size, checked: d.size === size0, onchange: () => this.checkSizes() }));
    this.sizeWarn = h("p", { class: "warn small", hidden: true, style: "margin:0" });
    this.f.use_real.checked = nLab > 0;
    this.f.use_real.disabled = nLab === 0;
    this.dataBox.replaceChildren(
      sets.length ? h("div", { class: "small muted" }, "Simulated scans (the right answers are known exactly):") :
        h("div", { class: "banner" }, `There are no simulated ${kindShort(this.kind).toLowerCase()} scans yet. `,
          h("a", { href: "#/synthetic" }, "Make some"), " first (3000 or more), then come back."),
      ...sets.map((d, i) => h("label", { class: "check" }, this.setChecks[i], `${d.name}: ${d.n} scans, ${d.size} x ${d.size} pixels`)),
      this.sizeWarn,
      h("label", { class: "check", style: "margin-top:4px" }, this.f.use_real,
        nLab ? `Also learn from your labelled ${kindShort(this.kind).toLowerCase()} scans (${nLab})` : `Your labelled scans (none of this kind yet)`),
      h("div", { class: "hint" }, "Some of the scans are kept out of training to test the model. The test results are shown on the Models page."));
    this.checkSizes();
  },
  checkSizes() {
    const sizes = [...new Set((this.setChecks || []).filter(c => c.checked).map(c => +c.dataset.size))];
    const bad = sizes.length > 1;
    this.sizeWarn.hidden = !bad;
    this.sizeWarn.textContent = bad ? "These sets have different image sizes. Choose sets of one size." : "";
    this.startBtn.disabled = bad;
    return sizes[0] || { PvP: 96, PvT: 64, tiebar: 64 }[this.kind];
  },
  renderWhere() {
    const opt = (w, title, text) => h("label", { class: `choice${this.where === w ? " on" : ""}` },
      h("input", { type: "radio", name: "train-where", checked: this.where === w, onchange: () => { this.where = w; this.renderWhere(); } }),
      h("div", {}, h("b", {}, title), h("div", { class: "small muted" }, text)));
    this.whereBox.replaceChildren(
      opt("here", "This computer", "Uses this computer's GPU if it has one, else its processor. On a processor, a model takes from about 20 minutes to 2 hours, and the computer is slow meanwhile."),
      opt("elsewhere", "Another computer, for example one with a GPU", "ChargeCell packs the training data into one file. A training computer picks it up, trains, and sends the model back here. You can follow its progress on this page."));
    this.startBtn.textContent = this.where === "here" ? "Start training" : "Send to a training computer";
    if (this.where === "here") { this.remoteHelp.replaceChildren(); return; }
    this.renderRemoteHelp();
  },
  async renderRemoteHelp() {
    const w = this.setup || (this.setup = await api("/api/worker/setup"));
    const cmd = url => `chargecell worker --server ${url} --token ${w.token}`;
    const parts = [
      h("div", { class: "small" }, h("b", {}, "On the training computer: "),
        `install ChargeCell ${w.chargecell} (the same version as here) and a PyTorch with GPU support (see pytorch.org). Then run the command below. It keeps running and trains every job you send, one at a time.`)];
    if (w.reachable) {
      parts.push(cmdBox(cmd(w.urls[0])));
      if (w.urls.length > 1) parts.push(h("div", { class: "hint" }, `If the training computer cannot find "${w.urls[0].split("//")[1].split(":")[0]}", use ${w.urls.slice(1).join(" or ")} instead.`));
    } else {
      parts.push(
        h("div", { class: "banner small" }, "This ChargeCell only accepts connections from this computer. Choose one of two ways:"),
        h("div", { class: "small" }, h("b", {}, "A. Let the lab network reach it. "), "Stop ChargeCell and start it again with:"),
        cmdBox(`chargecell -w <workspace folder> serve --host 0.0.0.0 --port ${w.port}`),
        h("div", { class: "hint" }, "Only on a network you trust: the web page has no password. This page will then show the worker command."),
        h("div", { class: "small" }, h("b", {}, "B. Keep it private and connect through SSH. "), `On the training computer, run the first command (it keeps a secure tunnel open), then the second one in another terminal:`),
        cmdBox(`ssh -N -L ${w.port}:localhost:${w.port} <your user name>@${w.computer}`),
        cmdBox(cmd(`http://localhost:${w.port}`)));
    }
    parts.push(h("div", { class: "hint" }, "No network between the two computers? Start the training anyway, then download the training-job file from the panel on the right and carry it over. On the training computer, ",
      h("code", {}, "chargecell train-job <file>"), " writes a model file; add it on the Models page."));
    this.remoteHelp.replaceChildren(h("div", { class: "remote-help stack", style: "gap:8px" }, ...parts));
  },

  async start(e) {
    const f = this.f;
    const synthetic = (this.setChecks || []).filter(c => c.checked).map(c => c.value);
    const body = {
      kind: this.kind, synthetic, size: this.checkSizes(), use_real: f.use_real.checked && !f.use_real.disabled,
      only_reviewed: f.only_reviewed.checked, real_weight: +f.real_weight.value, epochs: +f.epochs.value,
      ensemble: +f.ensemble.value, base: +f.base.value, batch_size: +f.batch_size.value, lr: +f.lr.value,
      target_precision: Math.min(0.999, +f.target_precision.value / 100), notes: f.notes.value,
      name: f.name.value.trim(), use_when_done: f.use_when_done.checked, where: this.where,
    };
    e.target.disabled = true;
    try {
      const job = await api("/api/train", { method: "POST", json: body });
      toast(this.where === "here" ? "Training started" : "Preparing the training data for the other computer");
      this.jobId = job.id;
      f.name.value = "";
      pollStatus();
      this.watch();
    } finally { e.target.disabled = false; this.checkSizes(); }
  },

  async watch() {
    clearTimeout(this._t);
    if (location.hash.split("/")[1] !== "train") return;
    const jobs = (await api("/api/jobs", { quiet: true }) || []).filter(j => j.kind === "train");
    const live = ["queued", "preparing", "waiting", "running"];
    const job = jobs.find(j => j.id === this.jobId) || jobs.find(j => live.includes(j.state)) || jobs[0];
    if (job) {
      const full = await api(`/api/jobs/${job.id}`, { quiet: true });
      if (full && full.state === "done" && full.result && !S.models.find(m => m.id === full.result.model_id)) await loadModels();
      if (full) this.renderProgress(full);
    } else this.progress.replaceChildren(h("div", { class: "panel muted" }, "No training yet. Your training will show here."));
    this.renderRecent(jobs, job);
    if (jobs.some(j => live.includes(j.state))) this._t = setTimeout(() => this.watch(), 2000);
  },
  renderProgress(j) {
    const chart = h("canvas", { class: "chart" });
    const w = j.worker || {}, ago = minutesAgo(j.last_seen), remote = j.where === "remote";
    const lines = [];
    if (j.state === "waiting") lines.push(
      h("p", { class: "small", style: "margin:0" }, "Waiting for a training computer. Start the worker command there (step 3 on the left), or carry the training-job file over by hand."),
      h("div", { class: "row" }, h("a", { class: "button", href: `/api/jobs/${j.id}/job-file` }, "Download the training-job file")));
    if (j.state === "running" && remote) lines.push(h("p", { class: "small", style: "margin:0" },
      `Training on ${w.computer || "another computer"} (${w.device || "unknown device"}). Last news ${ago === 0 ? "just now" : ago + " min ago"}.`),
      ago != null && ago >= 5 ? h("p", { class: "warn small", style: "margin:0" }, `No news for ${ago} minutes. Is the worker still running on ${w.computer || "the training computer"}?`) : null);
    if (j.state === "done" && j.result) {
      const m = (S.models || []).find(x => x.id === j.result.model_id);
      lines.push(h("p", { class: "small", style: "margin:0" }, `The new model is called "${j.result.name}". `,
        h("a", { href: "#/models" }, "See its test results on the Models page"), "."),
        m && !m.in_use ? h("div", {}, h("button", { class: "primary", onclick: async () => { await useModel(m); this.watch(); } }, "Use it now")) :
          m ? h("div", {}, h("span", { class: "flag ok" }, "In use")) : null);
    }
    if (j.state === "failed") lines.push(h("p", { class: "warn", style: "margin:0" }, j.error || j.message),
      remote && j.job_file ? h("div", { class: "row" }, h("button", { onclick: async () => { await api(`/api/jobs/${j.id}/retry`, { method: "POST" }); toast("Offered to training computers again"); this.watch(); } }, "Try again"),
        h("span", { class: "hint" }, "The same training data is offered to training computers again.")) : null);
    if (j.state === "interrupted") lines.push(h("p", { class: "warn small", style: "margin:0" }, "ChargeCell was restarted while this was training here. Start the training again."));
    const active = ["queued", "preparing", "waiting", "running"].includes(j.state);
    this.progress.replaceChildren(h("div", { class: "panel stack" },
      h("div", { class: "row" }, h("h2", { style: "margin:0" }, active ? "Training now" : "Latest training"), h("span", { class: "spacer" }),
        active ? h("button", { onclick: async () => { if (confirm("Stop this training?")) { await api(`/api/jobs/${j.id}/cancel`, { method: "POST" }); this.watch(); } } }, "Cancel") : null),
      h("div", {}, h("b", {}, j.title), h("div", { class: "small muted" }, `${remote ? "On another computer" : "On this computer"} · started ${when(j.created)}`)),
      h("div", { class: "progress" }, h("i", { style: `width:${j.progress * 100}%` })),
      h("div", { class: "small" }, jobText(j)),
      ...lines,
      (j.history || []).length || j.state === "running" ? [chart, h("div", { class: "hint" }, "Lines: training error of each network (lower is better; left scale). Dots: share of test scans with the right answer (right scale).")] : null));
    if ((j.history || []).length || j.state === "running") requestAnimationFrame(() => drawTrainChart(chart, j.history || []));
  },
  renderRecent(jobs, shown) {
    const rest = jobs.filter(j => !shown || j.id !== shown.id).slice(0, 6);
    this.recent.replaceChildren(rest.length ? h("div", { class: "panel" }, h("h2", {}, "Earlier training"),
      h("div", { class: "stack", style: "gap:6px" }, rest.map(j => h("a", { class: "job-row", href: "#/train", onclick: e => { e.preventDefault(); this.jobId = j.id; this.watch(); } },
        h("b", {}, j.title), h("span", { class: "small muted" }, `${when(j.created)} · ${jobText(j)}`))))) : "");
  },
};

function drawTrainChart(c, hist) {
  const dpr = window.devicePixelRatio || 1, w = c.clientWidth, hh = c.clientHeight;
  c.width = w * dpr; c.height = hh * dpr;
  const ctx = c.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  if (!hist.length) { ctx.fillStyle = "#5B6878"; ctx.font = "13px system-ui"; ctx.fillText("Waiting for the first pass over the data", 10, 20); return; }
  const L = 40, R = 40, T = 10, B = 22, W = w - L - R, H = hh - T - B;
  const members = [...new Set(hist.map(r => r.member))];
  const maxEp = Math.max(...hist.map(r => r.epoch)), maxLoss = Math.max(...hist.map(r => r.train_loss));
  const X = e => L + (maxEp > 1 ? (e - 1) / (maxEp - 1) : 0.5) * W;
  ctx.strokeStyle = "#CBD3DC"; ctx.strokeRect(L, T, W, H);
  ctx.fillStyle = "#5B6878"; ctx.font = "11px system-ui";
  ctx.fillText(maxLoss.toFixed(2), 4, T + 8); ctx.fillText("0", 28, T + H); ctx.fillText("100%", w - R + 4, T + 8);
  ctx.fillText("pass", L + W / 2 - 12, hh - 4);
  const colors = ["#2E5AAC", "#13808A", "#7A4FB5", "#8A7A1E", "#A86400", "#17202B"];
  members.forEach((m, k) => {
    const rs = hist.filter(r => r.member === m);
    ctx.strokeStyle = colors[k % colors.length]; ctx.lineWidth = 2; ctx.beginPath();
    rs.forEach((r, i) => { const x = X(r.epoch), y = T + H - r.train_loss / maxLoss * H; i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    ctx.stroke();
    ctx.fillStyle = colors[k % colors.length];
    rs.forEach(r => { if (r.val_status_acc != null) { ctx.beginPath(); ctx.arc(X(r.epoch), T + H - r.val_status_acc * H, 3, 0, 7); ctx.fill(); } });
  });
}

// ---------------------------------------------------------------- Device page
pages.device = {
  build() {
    this.root = $("#page-device");
    this.form = h("div", { class: "stack" });
    this.root.append(h("header", {}, h("h1", {}, "Device settings"), this.select = h("select", { onchange: () => this.show(this.select.value) }),
      h("button", { onclick: () => { const n = prompt("Name of the new device"); if (n) this.show(n, true); } }, "New device")), this.form);
  },
  async enter() {
    this.devices = await api("/api/devices");
    this.select.replaceChildren(...this.devices.map(d => h("option", { value: d.name }, d.name)));
    this.show(this.current && this.devices.find(d => d.name === this.current) ? this.current : this.devices[0].name);
  },
  show(name, isNew = false) {
    this.current = name;
    const base = this.devices.find(d => d.name === name) || { ...this.devices[0], name, description: "" };
    if (isNew) this.select.append(h("option", { value: name, selected: true }, name));
    this.select.value = name;
    const d = JSON.parse(JSON.stringify(base));
    const gates = [...new Set([...d.plungers, ...Object.values(d.barriers || {}), ...Object.values(d.tunnel_gates || {}), ...Object.keys(d.safe_limits)])];
    const inp = (v, attrs = {}) => h("input", { value: v ?? "", ...attrs });
    const f = {
      description: inp(d.description), carrier: h("select", {}, h("option", { value: "electron", selected: d.carrier === "electron" }, "Electrons"),
        h("option", { value: "hole", selected: d.carrier === "hole" }, "Holes")),
      plungers: inp(d.plungers.join(", ")), sensor_gate: inp(d.sensor_gate),
      barriers: inp(Object.entries(d.barriers).map(([k, v]) => `${k}=${v}`).join(", ")),
      tunnel_gates: inp(Object.entries(d.tunnel_gates || {}).map(([k, v]) => `${k}=${v}`).join(", ")),
      max_step: inp(d.max_step != null ? d.max_step * 1e3 : "", { type: "number", step: "any", placeholder: "auto: one window" }),
      barrier_step: inp(d.barrier_step != null ? d.barrier_step * 1e3 : "", { type: "number", step: "any", placeholder: "auto" }),
      electron_temperature: inp(d.electron_temperature != null ? d.electron_temperature * 1e3 : "", { type: "number", step: "any", placeholder: "optional" }),
      tiebar_target: inp(d.tiebar_coupling_target ? d.tiebar_coupling_target.join(", ") : "", { placeholder: "optional, e.g. 0.05, 0.3" }),
      points_per_addition: inp(d.points_per_addition, { type: "number" }), min_points: inp(d.min_points, { type: "number" }), max_points: inp(d.max_points, { type: "number" }),
      virtual: h("textarea", { rows: 5, placeholder: '{"virtual": ["vP1","vP2"], "physical": ["P1","P2","M1"], "matrix": [[1,0.2],[0.15,1],[-0.3,-0.2]]}' },
        d.virtual_gates ? JSON.stringify(d.virtual_gates) : ""),
      notes: h("textarea", { rows: 2 }, d.notes || ""),
    };
    const rows = gates.map(g => ({ g, lo: inp(d.safe_limits[g]?.[0] ?? "", { type: "number", step: "any" }), hi: inp(d.safe_limits[g]?.[1] ?? "", { type: "number", step: "any" }),
      add: inp(d.addition_voltage[g] != null ? d.addition_voltage[g] * 1e3 : "", { type: "number", step: "any" }),
      lever: inp((d.lever_arm || {})[g] ?? "", { type: "number", step: "any" }) }));
    const pairs = txt => { const o = {}; txt.split(",").map(t => t.trim()).filter(Boolean).forEach(t => { const [k, v] = t.split("="); if (k && v) o[k.trim()] = v.trim(); }); return o; };
    const num = el => el.value === "" ? null : +el.value;
    const L = (t, el, help) => h("label", { title: help || null }, t, el);
    this.form.replaceChildren(
      h("div", { class: "panel stack" }, h("h2", { style: "margin:0" }, name),
        h("p", { class: "small muted", style: "margin:0" }, "All settings are optional; ChargeCell assumes nothing about your voltages. Settings left blank are measured from your scans, or taken as a fraction of the current scan. Suggested scans stay inside the safe limits you set, and larger moves are split into steps."),
        h("div", { class: "fields" }, L("Description", f.description), L("Carriers", f.carrier, "Electrons: more plunger voltage adds electrons. Holes: less voltage adds holes."),
          L("Plunger gates", f.plungers, "Comma separated, e.g. P1, P2, P3"), L("Sensor gate", f.sensor_gate), L("Exchange gates", f.barriers, "Which gate sits between each pair, e.g. P1-P2=X1, P2-P3=X2"),
          L("Tunnel gates", f.tunnel_gates, "Reservoir tunnel gate next to each edge plunger, e.g. P1=T1, P3=T2"),
          L("Largest single step (mV)", f.max_step, "The largest gate change ChargeCell will suggest in one go; bigger moves are split into steps. Blank: at most one scan width"),
          L("Exchange-gate step (mV)", f.barrier_step, "How far to change an exchange gate when the dots are merged or the tie bar needs adjusting. Blank: a quarter of the spacing between electrons"),
          L("Electron temperature (mK)", f.electron_temperature, "Optional. With lever arms, tie-bar results are also given in µeV"),
          L("Tie-bar coupling target", f.tiebar_target, "Optional band for the tie-bar coupling ratio (interdot width / tie-bar length), e.g. 0.05, 0.3"),
          L("Points per electron", f.points_per_addition, "How many scan points suggested scans put between two electrons"), L("Fewest points per axis", f.min_points), L("Most points per axis", f.max_points)),
        h("p", { class: "small muted", style: "margin:0" }, "Gate names follow the HRL convention: P plungers, X exchange gates, T reservoir tunnel gates, M sensor. Hover over a field for help.")),
      h("div", { class: "panel" }, h("h2", {}, "Gates"),
        h("table", { class: "list" }, h("tr", {}, h("th", {}, "Gate"), h("th", {}, "Safe minimum (V)"), h("th", {}, "Safe maximum (V)"), h("th", {}, "Typical spacing between electrons (mV)"), h("th", {}, "Lever arm (eV/V)")),
          rows.map(r => h("tr", {}, h("td", {}, h("b", {}, r.g)), h("td", {}, r.lo), h("td", {}, r.hi), h("td", {}, r.add), h("td", {}, r.lever)))),
        h("p", { class: "small muted" }, "All optional. The spacing is a starting guess; once scans are analysed, measured spacings on this device take over. Lever arms are only used to express tie-bar results in energy units.")),
      h("details", { class: "panel" }, h("summary", {}, "Virtual gates (advanced)"),
        h("p", { class: "small muted" }, "If you scan in virtual plunger coordinates, give the matrix that turns virtual changes into physical gate changes. Guidance will then list the physical moves too."),
        f.virtual),
      h("div", { class: "panel" }, L("Notes", f.notes)),
      h("div", { class: "row" }, h("button", { class: "primary", onclick: async () => {
        let vg = null;
        if (f.virtual.value.trim()) { try { vg = JSON.parse(f.virtual.value); } catch (e) { toast("The virtual gate matrix is not valid JSON.", "error"); return; } }
        const plungers = f.plungers.value.split(",").map(s => s.trim()).filter(Boolean);
        const safe = {}, add = {}, lever = {};
        rows.forEach(r => { if (r.lo.value !== "" && r.hi.value !== "") safe[r.g] = [+r.lo.value, +r.hi.value]; if (r.add.value !== "") add[r.g] = +r.add.value / 1e3; if (r.lever.value !== "") lever[r.g] = +r.lever.value; });
        const ms = num(f.max_step), bs = num(f.barrier_step), te = num(f.electron_temperature);
        const tt = f.tiebar_target.value.split(",").map(t => t.trim()).filter(Boolean).map(Number);
        await api(`/api/devices/${encodeURIComponent(name)}`, { method: "PUT", json: {
          ...d, description: f.description.value, carrier: f.carrier.value, plungers, sensor_gate: f.sensor_gate.value,
          barriers: pairs(f.barriers.value), tunnel_gates: pairs(f.tunnel_gates.value),
          max_step: ms ? ms / 1e3 : null, barrier_step: bs ? bs / 1e3 : null, electron_temperature: te ? te / 1e3 : null,
          tiebar_coupling_target: tt.length === 2 ? tt : null, lever_arm: lever, points_per_addition: +f.points_per_addition.value,
          min_points: +f.min_points.value, max_points: +f.max_points.value, safe_limits: safe, addition_voltage: add, virtual_gates: vg, notes: f.notes.value } });
        toast("Device settings saved");
        this.enter();
      } }, "Save device settings")));
  },
};

// ---------------------------------------------------------------- router + polling
async function pollStatus() {
  const st = await api("/api/status", { quiet: true });
  if (!st) return;
  S.status = st;
  const running = st.running_jobs || [];
  const ind = $("#job-indicator");
  if (running.length) {
    const j = running.find(x => x.state === "running") || running[0];
    ind.hidden = false;
    ind.replaceChildren(h("a", { href: j.kind === "train" ? "#/train" : j.kind === "synthetic" ? "#/synthetic" : "#/scans" },
      h("div", {}, j.title), h("div", { class: "small muted" }, jobText(j)), h("div", { class: "bar" }, h("i", { style: `width:${j.progress * 100}%` })),
      running.length > 1 ? h("div", { class: "small muted" }, `and ${running.length - 1} more`) : null));
  } else ind.hidden = true;
  for (const id of S.running) {
    if (!running.find(j => j.id === id)) {
      const j = await api(`/api/jobs/${id}`, { quiet: true });
      if (j) toast(j.state === "done" ? `Finished: ${j.title}` : `${j.title}: ${jobText(j)}`, j.state === "failed" ? "error" : "");
      if (j && j.kind === "train") await loadModels();
      const page = location.hash.split("/")[1];
      if (page === "synthetic") pages.synthetic.refresh();
      if (page === "train") pages.train.watch();
      if (page === "scans") pages.scans.enter();
      if (page === "home" || page === "models") pages[page].enter();
    }
  }
  S.running = new Set(running.map(j => j.id));
  const act = st.active_models || {};
  $("#model-indicator").replaceChildren(h("a", { href: "#/models", title: "Models in use. Click to switch." },
    h("div", { class: "rail-label" }, "Models in use"),
    Object.keys(KINDS).map(k => h("div", { class: act[k] ? "" : "none" }, act[k] ? "\u2713 " : "\u2013 ", KINDS[k].short))));
}

async function route() {
  const parts = location.hash.replace(/^#\/?/, "").split("/");
  const name = pages[parts[0]] ? parts[0] : "home";
  const arg = parts[1] ? decodeURIComponent(parts[1]) : null;
  document.querySelectorAll(".page").forEach(p => { p.hidden = p.id !== `page-${name}`; });
  document.querySelectorAll(".rail a").forEach(a => a.classList.toggle("active", a.dataset.page === name));
  const page = pages[name];
  if (!page.built) { page.build(); page.built = true; }
  try { await page.enter(arg); } catch (e) { console.error(e); }
}
window.addEventListener("hashchange", route);
pollStatus().then(route);
setInterval(pollStatus, 1500);
