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
  NOT_IN_WINDOW: "(1,1) not in this window",
  UNINTERPRETABLE: "Can't interpret this scan",
};
const KIND_STATUS_LABEL = {
  PvT: { FOUND: "One-electron point found", NOT_IN_WINDOW: "Not in this window", UNINTERPRETABLE: "Can't interpret this scan" },
  tiebar: { FOUND: "Tie bar found", NOT_IN_WINDOW: "Tie bar not in this window", UNINTERPRETABLE: "Can't interpret this scan" },
};
const statusLabel = (kind, st) => (KIND_STATUS_LABEL[kind] || STATUS_LABEL)[st];
const KIND_LABEL = { PvP: "Plunger vs plunger", PvT: "Plunger vs tunnel gate", tiebar: "Tie bar (zoom on (1,1)-(2,0))" };
const SHORT_STATUS = { FOUND: "Found", NOT_IN_WINDOW: "Not in window", UNINTERPRETABLE: "Uninterpretable" };
const REASON_LABEL = {
  none: "(1,1) visible and anchored",
  no_transitions: "No transitions visible",
  occupancy_too_low: "Window stops before (1,1)",
  no_reference: "No empty region to count from",
  partially_visible: "(1,1) cut off by the edge",
  low_snr: "Too noisy",
  sensor_insensitive: "Sensor lost sensitivity",
  dots_merged: "Dots merged (one-dot pattern)",
  charge_instability: "Charge jumps between sweeps",
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
const FAMILY_COLOR = { a: "#2E5AAC", b: "#13808A", interdot: "#17202B", spectator: "#7A4FB5", sensor: "#8A7A1E" };
const FAMILIES = ["a", "b", "interdot", "spectator", "sensor"];

// ---------------------------------------------------------------- shared state
const S = { scans: [], cache: new Map(), status: null, running: new Set(), pendingDraft: null };

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
    h("button", { "data-m": "gradient", onclick: () => setMode("gradient"), title: "Gradient magnitude: makes faint lines visible (G)" }, "Gradient"));
  const bar = h("div", { class: "plot-bar" },
    viewSeg,
    h("label", {}, "Colours", h("select", { onchange: e => plot.setCmap(e.target.value) },
      h("option", { value: "gray" }, "Grey"), h("option", { value: "viridis" }, "Viridis"))),
    h("label", {}, "Contrast", h("input", { type: "range", min: 0, max: 15, step: 0.5, value: 1,
      oninput: e => plot.setClip(+e.target.value), "aria-label": "Contrast clipping" })),
    h("label", { class: "check" }, h("input", { type: "checkbox", onchange: e => plot.setInvert(e.target.checked) }), "Invert"),
    ...extra,
    h("span", { class: "spacer" }),
    h("button", { onclick: () => plot.resetView(), title: "Scroll to zoom, shift-drag to pan" }, "Reset view"));
  bar.setMode = setMode;
  return bar;
}

// ---------------------------------------------------------------- overlay builders
function codeImage(code, nx, ny, style = "cells") {
  const c = document.createElement("canvas");
  c.width = nx; c.height = ny;
  const ctx = c.getContext("2d"), img = ctx.createImageData(nx, ny);
  for (let k = 0; k < nx * ny; k++) {
    const a = code[k] >> 3, b = code[k] & 7;
    if (a === 7 || b === 7) continue;
    let rgba;
    if (a === 1 && b === 1) rgba = [46, 90, 172, 105];
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
    const txt = `(${l.a === 7 ? "?" : l.a},${l.b === 7 ? "?" : l.b})`;
    const is11 = l.a === 1 && l.b === 1;
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
        h("button", { class: "primary", onclick: e => this.analyze(e) }, "Analyse"),
        h("span", { class: "spacer" }),
        h("button", { onclick: () => startPractice() }, "New practice device")),
      this.body = h("div", { class: "grid-2" },
        h("div", {}, h("div", { class: "plot-wrap" }, this.canvas,
          plotBar(this.plot, [toggle("cells", "Cells"), toggle("lines", "Lines"), toggle("next", "Next scan")]))),
        this.side));
    this.empty = h("div", { class: "panel empty", hidden: true },
      h("h2", {}, "No scans yet"),
      h("p", {}, "Import scan files on the Scans page, send scans from your measurement setup through the ChargeCell API, or start a practice device to try the whole loop on a simulated triple dot."),
      h("div", { class: "row" },
        h("a", { class: "button primary", href: "#/scans" }, "Import scans"),
        h("button", { onclick: () => startPractice() }, "New practice device")));
    root.append(this.empty);
  },

  async enter(id) {
    await loadScanList();
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
      dashedRect(ctx, plot, w.x[0], w.x[1], w.y[0], w.y[1], col, rec.kind === "explore" ? "Explore here next" : "Next scan");
      const [Xa, Ya] = plot.toScreen((x0 + x1) / 2, (y0 + y1) / 2);
      const [Xb, Yb] = plot.toScreen((w.x[0] + w.x[1]) / 2, (w.y[0] + w.y[1]) / 2);
      arrow(ctx, Xa, Ya, Xb, Yb, col);
      if (rec.target && rec.kind === "move") {
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
      h("div", { class: "muted" }, `${meta.device}${meta.cooldown ? " \u00b7 cooldown " + meta.cooldown : ""} \u00b7 ${meta.source} \u00b7 ${when(meta.created)}`),
      meta.notes ? h("div", { class: "muted" }, meta.notes) : null,
      ref ? h("div", {}, h("a", { href: `#/runs/${encodeURIComponent(ref.run_id)}` }, "Show this scan's run (audit log)")) : null);
    if (!r) {
      side.replaceChildren(h("div", { class: "panel decision" },
        h("div", { class: "status-word status-none" }, "Not analysed yet"),
        h("p", {}, "Run the model to find the (1,1) cell or get directions for the next scan."),
        h("button", { class: "primary", onclick: e => this.analyze(e) }, "Analyse this scan"),
        S.status && !S.status.active_model ? h("p", { class: "warn" }, "No trained model yet. Train one on the Train page first.") : null), info);
      return;
    }
    const rec = r.recommendation || {};
    const flags = h("div", { class: "row" },
      h("span", { class: "muted small" }, `Confidence ${pct(r.confidence)}`),
      r.needs_review ? h("span", { class: "flag review", title: "The model is unsure or its checks disagreed. A person should look." }, "Needs review") : h("span", { class: "flag ok" }, "Checks passed"));
    const decision = h("div", { class: `panel decision ${r.status}` },
      h("div", { class: `status-word status-${r.status}` }, statusLabel(r.kind, r.status)),
      h("p", {}, r.reason_text), flags,
      (r.demotion && r.demotion.length) ? h("p", { class: "warn small" }, "Held back from FOUND because " + r.demotion.join("; ") + ".") : null);

    const todo = h("div", { class: "panel" }, h("h2", {}, "What to do next"),
      h("p", { class: "headline" }, rec.headline || ""),
      rec.steps && rec.steps.length ? h("ul", { class: "steps" }, rec.steps.map(s => h("li", {}, s))) : null);
    const win = rec.next_window || rec.tiebar_window || rec.retune_window;
    if (win) {
      const key = rec.next_window ? "next_window" : rec.tiebar_window ? "tiebar_window" : "retune_window";
      const row = (g, a) => h("tr", {}, h("td", {}, h("b", {}, g)), h("td", { class: "num" }, V4(a[0])), h("td", {}, "to"),
        h("td", { class: "num" }, V4(a[1])), h("td", {}, "V"), h("td", { class: "num muted" }, `${a[2]} pts`));
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
    if (rec.confidence) todo.append(h("p", { class: "muted small", style: "margin-top:8px" }, `Guidance confidence: ${rec.confidence}. Basis: ${rec.basis || "n/a"}.`));

    const specs = (r.spectators || []).length ? h("div", { class: "panel small" }, h("h3", { style: "margin-top:0" }, "Spectator dots"),
      r.spectators.map(sp => h("div", {}, h("b", {}, sp.gate), ` at ${sp.voltage != null ? V4(sp.voltage) + " V" : "unknown voltage"}: `,
        sp.status === "verified" ? h("span", { class: "flag ok" }, "one electron (consistent with an earlier scan)") : h("span", { class: "flag" }, sp.status)))) : null;

    const teach = (r.kind || "PvP") !== "PvP" ? h("div", { class: "panel small" }, h("h2", {}, "Teach the model"),
      h("p", { class: "small muted" }, `Labelling ${KIND_LABEL[r.kind] || r.kind} scans in the GUI is not available yet. Labelled scans can be sent with the ChargeCell API (docs/PROTOCOL.md).`),
      meta.source === "virtual_device" ? h("button", { onclick: e => this.reveal(e) }, "Reveal the true answer") : null) :
      h("div", { class: "panel" }, h("h2", {}, "Teach the model"),
      h("p", { class: "small muted" }, "Your corrections become training data. Accept the result if it is right, or fix it in the labeller."),
      h("div", { class: "row" },
        h("button", { onclick: () => this.accept() }, "Accept as label"),
        h("button", { onclick: () => { S.pendingDraft = this.id; go("label", this.id); } }, "Correct in labeller")),
      meta.source === "virtual_device" ? h("div", { style: "margin-top:10px" }, h("button", { onclick: e => this.reveal(e) }, "Reveal the true answer")) : null);

    const probs = r.probabilities ? h("details", { class: "panel" }, h("summary", {}, "Model details"),
      h("div", { class: "probs", style: "margin-top:10px" },
        Object.entries(r.probabilities.status).map(([k, v]) => [h("span", {}, SHORT_STATUS[k]), h("div", { class: "bar" }, h("i", { style: `width:${v * 100}%` })), h("span", {}, pct(v))])),
      h("p", { class: "small muted", style: "margin-top:8px" },
        (r.probabilities.ref || []).length === 2 ? `Empty region visible: ${meta.x_gate} ${pct(r.probabilities.ref[0])}, ${meta.y_gate} ${pct(r.probabilities.ref[1])}. ` :
          (r.probabilities.ref || []).length === 1 ? `Empty dot visible: ${pct(r.probabilities.ref[0])}. ` : "",
        `Ensemble disagreement ${r.uncertainty.mutual_info}. FOUND threshold ${r.found_threshold}. Model ${shortId(r.model_id)}.`),
      r.quality && r.quality.warnings.length ? h("p", { class: "warn small" }, r.quality.warnings.join(" ")) : null) : null;
    side.replaceChildren(decision, todo, specs, teach, probs, info);
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
    e.target.replaceWith(h("p", { class: "small" }, h("b", {}, "Truth: "), `${statusLabel(kind, t.status)} (${REASON_LABEL[t.reason] || t.reason}).${more}`));
  },
  async accept() {
    const r = this.analysis;
    const d = await api(`/api/scans/${encodeURIComponent(this.id)}/draft_from_model`, { method: "POST" });
    const ann = { ...d.annotation, status: r.status, reason: r.reason, annotator: annotatorName() || "operator" };
    await api(`/api/scans/${encodeURIComponent(this.id)}/annotation`, { method: "PUT", json: ann });
    toast("Saved as a label. Check it in the labeller when you have a moment.");
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
    const toolBtn = (t, label, key, cls) => h("button", { class: `tool-${cls}`, "data-tool": t, onclick: () => this.setTool(t), title: `Shortcut: ${key}` }, label, " ", h("kbd", {}, key));
    this.tools = h("div", { class: "seg" },
      toolBtn("v", "Select / move", "V", "v"), toolBtn("a", "Dot A boundary", "A", "a"),
      toolBtn("b", "Dot B boundary", "B", "b"), toolBtn("s", "Spectator line", "S", "s"),
      toolBtn("e", "Sensor artefact", "E", "e"));
    this.bar = plotBar(this.plot, [h("label", { class: "check" }, h("input", { type: "checkbox", checked: true,
      onchange: e => { this.showCells = e.target.checked; this.plot.render(); } }), "Cells")]);
    this.side = h("div", { class: "stack" });
    root.append(
      h("header", {}, h("h1", {}, "Label"), this.picker,
        h("button", { onclick: () => this.draftFromModel(), title: "Start from the model's prediction and correct it" }, "Draft from model"),
        h("button", { onclick: () => this.undo(), title: "Ctrl+Z" }, "Undo"),
        h("button", { onclick: () => this.redo(), title: "Ctrl+Shift+Z" }, "Redo"),
        h("button", { class: "danger", onclick: () => this.clearAll() }, "Clear lines")),
      this.body = h("div", { class: "grid-2" },
        h("div", {}, h("div", { class: "tools" }, this.tools), h("div", { class: "plot-wrap" }, this.canvas, this.bar)),
        this.side));
    this.emptyEl = h("div", { class: "panel empty", hidden: true }, h("h2", {}, "Nothing to label"),
      h("p", {}, "Import scans first. Scans the model is least sure about are listed here first."),
      h("a", { class: "button primary", href: "#/scans" }, "Import scans"));
    root.append(this.emptyEl);
    this.plot.hooks = {
      down: (p, e) => this.onDown(p, e), drag: p => this.onDrag(p), up: () => this.onUp(),
      hover: p => { this.hover = p; }, dblclick: () => this.finish(), context: p => this.onContext(p),
    };
    document.addEventListener("keydown", e => this.onKey(e));
    this.setTool("a");
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
    if ((this.scan.meta.kind || "PvP") !== "PvP") {
      this.ann = null; this.cells = null; this.cellLabels = null;
      this.plot.setScan(this.scan);
      this.side.replaceChildren(h("div", { class: "panel" }, h("h2", {}, "Not available yet"),
        h("p", {}, `This is a ${KIND_LABEL[this.scan.meta.kind] || this.scan.meta.kind} scan. The labeller draws plunger-vs-plunger labels only; labels for this kind can be sent with the ChargeCell API (docs/PROTOCOL.md).`)));
      return;
    }
    const d = await api(`/api/scans/${encodeURIComponent(id)}/annotation`);
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
    const sc = this.scan;
    const ext = [sc.x[0], sc.x[sc.nx - 1], sc.y[0], sc.y[sc.ny - 1]];
    this.cells = codeImage(code, sc.nx, sc.ny);
    this.cellLabels = codeLabels(code, sc.nx, sc.ny, ext, 0.008);
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
    if (["v", "a", "b", "s", "e"].includes(k)) { this.setTool(k); e.preventDefault(); }
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

  updateSuggestion() {
    if (!this.sugEl) return;
    const s = this.suggestion;
    this.sugEl.replaceChildren(s ? h("span", {}, "From your lines: ", h("b", {}, SHORT_STATUS[s.status]), ` (${REASON_LABEL[s.reason]}) `,
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
    const statusSeg = h("div", { class: "stack", style: "gap:4px" },
      ...["FOUND", "NOT_IN_WINDOW", "UNINTERPRETABLE"].map(st => h("label", { class: "check" },
        h("input", { type: "radio", name: "ann-status", checked: ann.status === st, onchange: () => {
          ann.status = st;
          const s = this.suggestion;
          if (s && s.status === st) ann.reason = s.reason;
          else if (!REASONS_BY_STATUS[st].includes(ann.reason)) ann.reason = REASONS_BY_STATUS[st][0];
          this.renderSide();
        } }), h("span", { class: `pill ${st}` }, STATUS_LABEL[st]))));
    const reasonSel = h("select", { disabled: !ann.status, onchange: e => { ann.reason = e.target.value; } },
      (REASONS_BY_STATUS[ann.status] || []).map(r => h("option", { value: r, selected: ann.reason === r }, REASON_LABEL[r])));
    this.sugEl = h("div", { class: "small" });
    const nA = (ann.a_boundaries || []).length, nB = (ann.b_boundaries || []).length;
    this.side.replaceChildren(
      h("div", { class: "panel stack" },
        h("h2", { style: "margin:0" }, "Electron counts"),
        h("p", { class: "small muted", style: "margin:0" },
          `Draw each dot A boundary bottom to top (it follows the ${sc.xLabel} line and its interdot jogs), and each dot B boundary left to right. Then say how many electrons sit left of / below them.`),
        offset("a_offset", `Dot A (${sc.xLabel}) left of all A boundaries`, "0 = empty region visible"),
        offset("b_offset", `Dot B (${sc.yLabel}) below all B boundaries`, "Unknown if no empty region is in view"),
        h("div", { class: "small muted" }, `${nA} A boundar${nA === 1 ? "y" : "ies"}, ${nB} B boundar${nB === 1 ? "y" : "ies"}, ${(ann.spectator_lines || []).length} spectator, ${(ann.sensor_lines || []).length} sensor line(s).`)),
      h("div", { class: "panel stack" },
        h("h2", { style: "margin:0" }, "Outcome"), statusSeg,
        h("label", {}, "Reason", reasonSel), this.sugEl,
        h("label", {}, "Your name", h("input", { value: ann.annotator || "", oninput: e => { ann.annotator = e.target.value; localStorage.setItem("cc-annotator", e.target.value); } })),
        h("label", {}, "Notes", h("textarea", { rows: 2, oninput: e => { ann.notes = e.target.value; } }, ann.notes || "")),
        h("label", { class: "check" }, h("input", { type: "checkbox", checked: !!ann.reviewed, onchange: e => { ann.reviewed = e.target.checked; } }), "Second check done"),
        h("div", { class: "row" },
          h("button", { class: "primary", onclick: () => this.save(false) }, "Save label"),
          h("button", { onclick: () => this.save(true) }, "Save and next")),
        h("div", { class: "small muted" }, `${this.historyCount || 0} saved version(s). Origin: ${ann.origin || "manual"}.`)),
      h("details", { class: "panel" }, h("summary", {}, "Shortcuts"),
        h("div", { class: "shortcut-list", style: "margin-top:8px" },
          h("kbd", {}, "A B S E"), h("span", {}, "draw dot A / dot B / spectator / sensor lines"),
          h("kbd", {}, "click"), h("span", {}, "add a point; double-click or Enter finishes"),
          h("kbd", {}, "V"), h("span", {}, "select and drag points; Alt-click inserts a point"),
          h("kbd", {}, "right-click"), h("span", {}, "delete a point"),
          h("kbd", {}, "Backspace"), h("span", {}, "remove last point / selected line"),
          h("kbd", {}, "Ctrl Z"), h("span", {}, "undo (Shift for redo)"),
          h("kbd", {}, "G"), h("span", {}, "toggle gradient view"),
          h("kbd", {}, "scroll, Shift-drag"), h("span", {}, "zoom, pan"))),
      h("div", { class: "panel" }, h("h2", {}, "Up next"),
        h("p", { class: "small muted", style: "margin:0" }, "Unlabelled scans, most uncertain first. These teach the model the most."),
        h("div", { class: "queue" }, (this.queue || []).map(q => h("a", { href: `#/label/${encodeURIComponent(q.id)}`, class: q.id === this.id ? "current" : "" },
          h("span", {}, `${when(q.created)} ${q.x_gate}/${q.y_gate}`),
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
    toast("Draft loaded from the model. Correct anything that is wrong, then save.");
  },

  async save(next) {
    this.finish();
    const ann = this.ann;
    if (!ann.status) { toast("Choose an outcome before saving.", "error"); return; }
    if (!ann.annotator) { toast("Add your name so labels can be traced.", "error"); return; }
    const s = this.suggestion;
    if (ann.status === "FOUND" && s && s.status !== "FOUND" &&
        !confirm("Your lines do not show a complete (1,1) cell with an empty region for both dots. Save as FOUND anyway?")) return;
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
        h("button", { class: "primary", onclick: () => { this.importPanel.hidden = !this.importPanel.hidden; } }, "Import files"),
        practiceButtons(),
        h("button", { onclick: () => this.analyzeAll() }, "Analyse all new scans")),
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
        "ChargeCell request files (.json in the chargecell/1 format) carry their own gate names and voltages. For other files, say which gates were swept. Matrix CSV: first row = x voltages, first column = y voltages. Or three columns x, y, signal."),
      files,
      h("div", { class: "fields" },
        field("X gate (horizontal)", h("input", { placeholder: "e.g. P1" }), "x_gate"),
        field("Y gate (vertical)", h("input", { placeholder: "e.g. P2" }), "y_gate"),
        field("Device", h("input", { value: "default" }), "device"),
        field("Cooldown", h("input", { placeholder: "e.g. CD7" }), "cooldown"),
        field("Voltages in file", h("select", {}, h("option", { value: "V" }, "volts"), h("option", { value: "mV" }, "millivolts")), "axis_units"),
        field("Scan kind", h("select", { title: "PvT: the plunger on one axis and its reservoir tunnel gate on the other. Tie bar: a plunger-plunger zoom on the (1,1)-(2,0) transition." },
          h("option", { value: "" }, "from file (else plunger vs plunger)"), ...Object.entries(KIND_LABEL).map(([k, t]) => h("option", { value: k }, t))), "kind"),
        field("Notes", h("input", {}), "notes")),
      h("p", { class: "small muted", style: "margin:0" }, "Always fill in the cooldown: training keeps each device and cooldown on one side of the train/test split so the scores stay honest."),
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
  async enter() {
    await loadScanList();
    const devices = [...new Set(S.scans.map(s => s.device))];
    const sel = (key, opts, label) => h("label", { class: "row", style: "gap:6px" }, label, h("select", {
      onchange: e => { this.filters[key] = e.target.value; this.renderTable(); } },
      opts.map(([v, t]) => h("option", { value: v, selected: this.filters[key] === v }, t))));
    this.filterBar.replaceChildren(
      sel("device", [["", "All devices"], ...devices.map(d => [d, d])], "Device"),
      sel("labelled", [["", "All"], ["yes", "Labelled"], ["no", "Not labelled"]], "Label"),
      sel("source", [["", "All sources"], ["api", "Sent by API"], ["file", "Other files"], ["virtual_device", "Practice devices"]], "Source"),
      sel("kind", [["", "All kinds"], ...Object.keys(KIND_LABEL).map(k => [k, k])], "Kind"),
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
        h("p", {}, "Import files above, or start a practice device to try the workflow on a simulated triple dot.")));
      return;
    }
    this.table.replaceChildren(h("table", { class: "list" },
      h("thead", {}, h("tr", {}, ["Measured", "Gates", "Kind", "Device", "Source", "Label", "Model says", ""].map(t => h("th", {}, t)))),
      h("tbody", {}, rows.map(s => {
        const a = s.analysis, l = s.label;
        return h("tr", { class: "click", onclick: e => { if (e.target.tagName !== "BUTTON") go("review", s.id); } },
          h("td", {}, when(s.created), h("div", { class: "id" }, shortId(s.id))),
          h("td", {}, `${s.x_gate} / ${s.y_gate}`, h("div", { class: "id" }, `${(s.shape || [])[1]} x ${(s.shape || [])[0]}`)),
          h("td", {}, s.kind || "PvP"),
          h("td", {}, s.device, s.cooldown ? h("div", { class: "id" }, s.cooldown) : null),
          h("td", {}, { api: "API", file: "file", virtual_device: "practice", synthetic: "synthetic" }[s.source] || s.source),
          h("td", {}, l && l.status ? h("span", { class: `pill ${l.status}` }, SHORT_STATUS[l.status]) : h("span", { class: "muted" }, "none"),
            l && l.annotator ? h("div", { class: "id" }, l.annotator) : null),
          h("td", {}, a ? [h("span", { class: `pill ${a.status}` }, SHORT_STATUS[a.status]), " ", h("span", { class: "muted small" }, pct(a.confidence || 0)),
            a.needs_review ? h("div", {}, h("span", { class: "flag review" }, "Needs review")) : null] : h("span", { class: "muted" }, "not analysed")),
          h("td", {}, h("div", { class: "row", style: "flex-wrap:nowrap" },
            (s.kind || "PvP") === "PvP" ? h("button", { onclick: () => go("label", s.id) }, "Label") : null,
            h("button", { class: "danger", onclick: async () => {
              if (!confirm("Delete this scan, its label and analysis?")) return;
              await api(`/api/scans/${encodeURIComponent(s.id)}`, { method: "DELETE" });
              this.enter();
            } }, "Delete"))));
      }))));
  },
  async analyzeAll() {
    await api("/api/analyze_all?only_new=true", { method: "POST" });
    toast("Analysing new scans in the background");
    pollStatus();
  },
};

// ---------------------------------------------------------------- Runs page (automation tree)
const GRADE = {
  pass: ["✓", "OK"], warn: ["!", "Check"], fail: ["✗", "Failed"],
  info: ["·", "Action"], open: ["…", "In progress"],
};
const SOURCE_LABEL = { backend: "measurement code", gui: "GUI", cli: "command line", practice: "practice device" };
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
      h("header", {}, h("h1", {}, "Runs"), this.filterBar),
      h("p", { class: "muted small", style: "max-width:900px" },
        "Every scan ChargeCell analyses is recorded in a run: what was measured, what ChargeCell concluded, what it advised, whether the next scan followed the advice, and what the operator reported doing. Each step is graded when it happens: ",
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
      sel("source", [["", "All"], ...Object.entries(SOURCE_LABEL)], "Recorded from"),
      h("button", { onclick: () => this.enter(this.id) }, "Refresh"));
    this.id = id && rows.find(r => r.id === id) ? id : (rows[0] || {}).id;
    this.renderList();
    await this.renderTree();
    this.renderStats(await api(`/api/v1/run_stats?${q}`));
  },

  renderList() {
    if (!this.rows.length) {
      this.list.replaceChildren(h("div", { class: "empty" }, h("h2", {}, "No runs yet"),
        h("p", {}, "Runs appear as soon as scans are analysed: from your measurement code, from the Review page, or from a practice device.")));
      return;
    }
    this.list.replaceChildren(h("table", { class: "list" },
      h("thead", {}, h("tr", {}, ["", "Run", "Stages"].map(t => h("th", {}, t)))),
      h("tbody", {}, this.rows.map(r => h("tr", { class: `click${r.id === this.id ? " current" : ""}`, onclick: () => go("runs", r.id) },
        h("td", {}, gradeMark(r.grade)),
        h("td", {}, h("div", {}, r.title), h("div", { class: "id" }, `${r.device} · ${when(r.created)} · ${r.n_scans} scan${r.n_scans === 1 ? "" : "s"} · ${SOURCE_LABEL[r.source] || r.source}`)),
        h("td", {}, h("div", { class: "stage-chips" }, r.stages.map(st => h("span", { class: `chip ${st.grade}`, title: st.title }, gradeMark(st.grade), " ", st.kind)))))))));
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
      if (n.type === "analysis" && d.checks && d.checks.length) extra.push(h("div", { class: "muted small" }, "Held back from FOUND: " + d.checks.join("; ") + "."));
      if (n.type === "advice" && d.window) {
        const w = d.window;
        extra.push(h("div", { class: "muted small" }, `${w.x_gate} ${V4(w.x[0])} to ${V4(w.x[1])} V, ${w.y_gate} ${V4(w.y[0])} to ${V4(w.y[1])} V (${w.x[2]} x ${w.y[2]} points)`));
      }
      const title = n.type === "analysis" ? `${SHORT_STATUS[d.status] || d.status}${d.reason && d.reason !== "none" ? ": " + (REASON_LABEL[d.reason] || d.reason) : ""}` : n.title;
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
      h("li", {}, `${f.kind} · ${SHORT_STATUS[f.status] || f.status}: ${REASON_LABEL[f.reason] || f.reason} — ${f.count}`))) : h("p", { class: "muted small" }, empty);
    const adv = st.advice_followed || { yes: 0, no: 0 }, rv = st.reviews || { agree: 0, disagree: 0 };
    this.stats.replaceChildren(
      h("h2", {}, `Across ${st.runs} run${st.runs === 1 ? "" : "s"}`),
      kinds.length ? h("div", { class: "table-wrap" }, h("table", { class: "list" },
        h("thead", {}, h("tr", {}, ["Scan kind", "Goals reached", "Median scans to goal", "90% within", "Stalled", "Left unfinished", "In progress", "Wrong “found” (practice)"].map(t => h("th", {}, t)))),
        h("tbody", {}, kinds.map(([k, v]) => h("tr", {},
          h("td", {}, KIND_LABEL[k] || k),
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

// ---------------------------------------------------------------- Synthetic page
pages.synthetic = {
  build() {
    const root = $("#page-synthetic");
    const f = this.f = {
      name: h("input", { value: "sim-" + new Date().toISOString().slice(0, 10) }),
      kind: h("select", {}, Object.entries(KIND_LABEL).map(([k, t]) => h("option", { value: k }, t))),
      n: h("input", { type: "number", min: 50, step: 50, value: 3000 }),
      size: h("select", {}, [64, 96, 128].map(v => h("option", { value: v, selected: v === 96 }, `${v} x ${v}`))),
      preset: h("select", {}, h("option", { value: "mixed" }, "Mixed linear and triangular"),
        h("option", { value: "hrl_linear" }, "Linear triple dot"), h("option", { value: "hrl_triangle" }, "Triangular triple dot")),
      found: h("input", { type: "number", min: 0, max: 100, value: 40 }),
      notin: h("input", { type: "number", min: 0, max: 100, value: 35 }),
      bad: h("input", { type: "number", min: 0, max: 100, value: 25 }),
      workers: h("input", { type: "number", min: 1, max: 64, value: Math.max(1, (navigator.hardwareConcurrency || 2) - 1) }),
      seed: h("input", { type: "number", value: Math.floor(Math.random() * 1e6) }),
    };
    const L = (t, el) => h("label", {}, t, el);
    this.list = h("div", { class: "stack" });
    this.preview = h("div", {});
    root.append(h("header", {}, h("h1", {}, "Synthetic data")),
      h("div", { class: "two-col" },
        h("div", { class: "panel stack" },
          h("h2", { style: "margin:0" }, "Generate a dataset"),
          h("p", { class: "small muted", style: "margin:0" },
            "Simulated triple-dot scans with exact labels: constant-interaction model with tunnel coupling, a charge sensor that can drift off its flank, 1/f and telegraph noise, charge jumps and latching; for plunger-vs-tunnel-gate scans also the reservoir tunnel rate (slow: lines latch and vanish; fast: lines smear). The outcome mix controls how often each case is aimed for; final labels come from the physics."),
          h("div", { class: "fields" }, L("Name", f.name), h("label", { class: "wide" }, "Scan kind", f.kind), L("Number of scans", f.n), L("Image size", f.size), h("label", { class: "wide" }, "Device style", f.preset)),
          h("div", { class: "fields" }, L("Aim: found (%)", f.found), L("Aim: not in window (%)", f.notin), L("Aim: uninterpretable (%)", f.bad)),
          h("div", { class: "fields" }, L("Worker processes", f.workers), L("Random seed", f.seed)),
          h("p", { class: "small muted", style: "margin:0" }, "Image size must match the model you train. Roughly 15 scans per second per worker."),
          h("div", { class: "row" }, h("button", { class: "primary", onclick: () => this.generate() }, "Generate"))),
        h("div", { class: "stack" }, h("h2", { style: "margin:4px 0 0" }, "Datasets"), this.list)),
      h("div", { style: "margin-top:18px" }, this.preview));
  },
  async enter() { this.refresh(); },
  async refresh() {
    const ds = await api("/api/synthetic");
    if (!ds.length) { this.list.replaceChildren(h("p", { class: "muted" }, "No datasets yet. Generate one to train your first model.")); return; }
    this.list.replaceChildren(...ds.map(d => {
      const c = d.counts || {}, tot = Object.values(c).reduce((a, b) => a + b, 0) || 1;
      const by = st => Object.entries(c).filter(([k]) => k.startsWith(st)).reduce((a, [, v]) => a + v, 0);
      return h("div", { class: "panel small" },
        h("div", { class: "row" }, h("b", {}, d.name), h("span", { class: "muted" }, `${d.kind || "PvP"}, ${d.n} scans, ${d.size}px, ${d.preset}`), h("span", { class: "spacer" }),
          h("button", { onclick: () => this.showPreview(d.name) }, "Preview"),
          h("button", { class: "danger", onclick: async () => { if (confirm(`Delete dataset ${d.name}?`)) { await api(`/api/synthetic/${d.name}`, { method: "DELETE" }); this.refresh(); } } }, "Delete")),
        h("div", { class: "countbar", title: "found / not in window / uninterpretable" },
          h("i", { style: `width:${by("FOUND") / tot * 100}%;background:var(--found)` }),
          h("i", { style: `width:${by("NOT_IN") / tot * 100}%;background:var(--move)` }),
          h("i", { style: `width:${by("UNINT") / tot * 100}%;background:var(--fail)` })),
        h("div", { class: "muted" }, `Found ${by("FOUND")}, not in window ${by("NOT_IN")}, uninterpretable ${by("UNINT")}. Created ${when(d.created)}.`));
    }));
  },
  async generate() {
    const f = this.f, tot = +f.found.value + +f.notin.value + +f.bad.value;
    if (tot <= 0) { toast("The outcome mix must add up to more than 0.", "error"); return; }
    await api("/api/synthetic", { method: "POST", json: {
      name: f.name.value, kind: f.kind.value, n: +f.n.value, size: +f.size.value, preset: f.preset.value, seed: +f.seed.value,
      workers: +f.workers.value, mix: [f.found.value / tot, f.notin.value / tot, f.bad.value / tot] } });
    toast("Generating in the background. Progress shows in the sidebar.");
    f.seed.value = Math.floor(Math.random() * 1e6);
    pollStatus();
  },
  async showPreview(name) {
    const items = await api(`/api/synthetic/${name}/preview?n=18`);
    this.preview.replaceChildren(h("h2", {}, `Preview: ${name}`), h("div", { class: "thumbs" }, items.map(it => {
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
        h("div", { class: "muted" }, REASON_LABEL[m.reason] || m.reason), h("div", { class: "muted" }, `${m.pair ? "dots " + m.pair.join("-") : "dot " + ((m.dot ?? 0) + 1)}, ${m.artifact}`)));
    })));
  },
};

// ---------------------------------------------------------------- Train page
pages.train = {
  build() {
    const root = $("#page-train");
    const f = this.f = {
      kind: h("select", { onchange: () => this.renderDatasets() }, Object.entries(KIND_LABEL).map(([k, t]) => h("option", { value: k }, t))),
      size: h("select", { onchange: () => this.renderDatasets() }, [64, 96, 128].map(v => h("option", { value: v, selected: v === 96 }, `${v} x ${v}`))),
      use_real: h("input", { type: "checkbox", checked: true }),
      only_reviewed: h("input", { type: "checkbox" }),
      real_weight: h("input", { type: "number", min: 1, max: 50, value: 5 }),
      epochs: h("input", { type: "number", min: 1, max: 200, value: 12 }),
      ensemble: h("input", { type: "number", min: 1, max: 8, value: 3 }),
      base: h("select", {}, h("option", { value: 12 }, "Small (fast)"), h("option", { value: 16, selected: true }, "Standard"), h("option", { value: 24 }, "Large")),
      batch_size: h("input", { type: "number", min: 4, max: 256, value: 32 }),
      lr: h("input", { type: "number", step: 0.0005, value: 0.002 }),
      target_precision: h("input", { type: "number", min: 0.8, max: 0.999, step: 0.005, value: 0.97 }),
      notes: h("input", { placeholder: "What changed in this version?" }),
    };
    const L = (t, el) => h("label", {}, t, el);
    this.dsBox = h("div", { class: "stack", style: "gap:4px" });
    this.progress = h("div", {});
    this.models = h("div", { class: "stack" });
    root.append(h("header", {}, h("h1", {}, "Train")),
      h("div", { class: "two-col" },
        h("div", { class: "panel stack" },
          h("h2", { style: "margin:0" }, "Train a new model version"),
          h("p", { class: "small muted", style: "margin:0" }, "Each scan kind has its own model: plunger vs plunger finds (1,1), plunger vs tunnel gate loads one electron, the tie bar sets up readout."),
          L("Scan kind", f.kind), L("Image size", f.size),
          h("div", {}, h("div", { class: "small muted" }, "Synthetic datasets"), this.dsBox),
          h("label", { class: "check" }, f.use_real, h("span", { id: "real-count" }, "Include labelled real scans")),
          h("label", { class: "check" }, f.only_reviewed, "Only labels with a second check"),
          h("div", { class: "fields" }, L("Weight of each real scan", f.real_weight), L("Epochs", f.epochs), L("Ensemble members", f.ensemble), L("Network size", f.base)),
          h("details", {}, h("summary", {}, "Advanced"), h("div", { class: "fields", style: "margin-top:8px" },
            L("Batch size", f.batch_size), L("Learning rate", f.lr), L("Target precision for FOUND", f.target_precision))),
          L("Notes", f.notes),
          h("p", { class: "small muted", style: "margin:0" }, "The FOUND threshold is calibrated on held-out data so that at least the target fraction of FOUND calls are right; otherwise the model defers. Real scans are split by device and cooldown."),
          h("div", { class: "row" }, h("button", { class: "primary", onclick: () => this.start() }, "Start training"))),
        h("div", { class: "stack" }, this.progress)),
      h("h2", { style: "margin-top:22px" }, "Model versions"), this.models);
  },
  async enter() {
    this.datasets = await api("/api/synthetic");
    this.renderDatasets();
    const st = await api("/api/status");
    $("#real-count").textContent = `Include labelled real scans (${st.n_labelled} labelled)`;
    this.refresh();
    this.watch();
  },
  renderDatasets() {
    const size = +this.f.size.value, kind = this.f.kind.value;
    const ok = (this.datasets || []).filter(d => d.size === size && (d.kind || "PvP") === kind);
    this.dsBox.replaceChildren(...(ok.length ? ok.map(d => h("label", { class: "check" },
      h("input", { type: "checkbox", value: d.name, checked: true }), `${d.name} (${d.n} scans)`)) :
      [h("p", { class: "small muted", style: "margin:0" }, `No ${size}px ${kind} datasets. Generate one on the Synthetic data page, or pick another size.`)]));
  },
  async start() {
    const f = this.f;
    const synthetic = [...this.dsBox.querySelectorAll("input:checked")].map(i => i.value);
    const body = { synthetic, kind: f.kind.value, size: +f.size.value, use_real: f.use_real.checked, only_reviewed: f.only_reviewed.checked,
      real_weight: +f.real_weight.value, epochs: +f.epochs.value, ensemble: +f.ensemble.value, base: +f.base.value,
      batch_size: +f.batch_size.value, lr: +f.lr.value, target_precision: +f.target_precision.value, notes: f.notes.value };
    const job = await api("/api/train", { method: "POST", json: body });
    toast("Training started");
    this.jobId = job.id;
    this.watch();
  },
  async watch() {
    clearTimeout(this._t);
    if (location.hash.split("/")[1] !== "train") return;
    const jobs = await api("/api/jobs", { quiet: true }) || [];
    const job = jobs.find(j => j.kind === "train" && (j.id === this.jobId || ["running", "queued"].includes(j.state))) || jobs.find(j => j.kind === "train");
    if (job) {
      const full = await api(`/api/jobs/${job.id}`, { quiet: true });
      if (full) this.renderProgress(full);
      if (["running", "queued"].includes(job.state)) this._t = setTimeout(() => this.watch(), 2000);
    } else this.progress.replaceChildren(h("div", { class: "panel muted" }, "No training runs yet."));
  },
  renderProgress(j) {
    const chart = h("canvas", { class: "chart" });
    this.progress.replaceChildren(h("div", { class: "panel stack" },
      h("div", { class: "row" }, h("h2", { style: "margin:0" }, "Latest training run"), h("span", { class: "spacer" }),
        ["running", "queued"].includes(j.state) ? h("button", { onclick: () => api(`/api/jobs/${j.id}/cancel`, { method: "POST" }) }, "Cancel") : null),
      h("div", { class: "progress" }, h("i", { style: `width:${j.progress * 100}%` })),
      h("div", { class: "small" }, `${j.state}: ${j.message}`),
      j.error ? h("p", { class: "warn" }, j.error) : null,
      chart, h("div", { class: "small muted" }, "Lines: training loss per ensemble member (left axis). Dots: validation status accuracy (right axis).")));
    requestAnimationFrame(() => drawTrainChart(chart, j.history || []));
    if (j.state === "done") this.refresh();
  },
  async refresh() {
    const ms = await api("/api/models");
    if (!ms.length) { this.models.replaceChildren(h("p", { class: "muted" }, "No model versions yet.")); return; }
    this.models.replaceChildren(...ms.map(m => {
      const metr = m.metrics || {};
      const block = (name, x) => x ? h("div", {}, h("h3", {}, name === "real" ? `Held-out real scans (${x.n})` : `Held-out synthetic scans (${x.n})`),
        h("div", { class: "metric-grid" },
          h("div", { class: "metric" }, h("b", {}, pct(x.found_precision)), h("span", {}, "FOUND calls that are right")),
          h("div", { class: "metric" }, h("b", {}, pct(x.found_recall)), h("span", {}, "true FOUND windows found")),
          h("div", { class: "metric" }, h("b", {}, pct(x.status_accuracy)), h("span", {}, "outcome correct")),
          h("div", { class: "metric" }, h("b", {}, pct(x.occupancy_pixel_accuracy)), h("span", {}, "electron count per pixel"))),
        h("details", {}, h("summary", {}, "Confusion matrix and line scores"),
          h("table", { class: "list", style: "margin-top:6px;width:auto" },
            h("tr", {}, h("th", {}, "true \\ predicted"), x.confusion_labels.map(l => h("th", {}, SHORT_STATUS[l]))),
            x.confusion.map((row, i) => h("tr", {}, h("td", {}, SHORT_STATUS[x.confusion_labels[i]]), row.map(v => h("td", {}, v))))),
          h("p", { class: "small" }, "Line F1: " + Object.entries(x.line_f1).map(([k, v]) => `${k} ${v}`).join(", ")))) : null;
      return h("div", { class: "panel" },
        h("div", { class: "row" }, h("b", {}, shortId(m.id)), h("span", { class: "pill" }, m.kind || "PvP"), h("span", { class: "muted small" }, when(m.created)),
          m.active ? h("span", { class: "flag ok" }, "Active") : h("button", { onclick: async () => { await api(`/api/models/${m.id}/activate`, { method: "POST" }); toast("Model activated"); this.refresh(); pollStatus(); } }, "Make active"),
          h("span", { class: "spacer" }),
          h("span", { class: "small muted" }, `${m.config.ensemble} members, ${m.config.size}px, ${m.data.n_train} training scans (${m.data.n_real_train} real), FOUND threshold ${m.found_threshold}`)),
        m.config.notes ? h("p", { class: "small" }, m.config.notes) : null,
        m.split_note ? h("p", { class: "warn small" }, m.split_note) : null,
        block("real", metr.real), block("synthetic", metr.synthetic),
        !metr.real ? h("p", { class: "small muted" }, "No held-out real scans yet: label scans from at least two cooldowns to measure real-data performance.") : null);
    }));
  },
};

function drawTrainChart(c, hist) {
  const dpr = window.devicePixelRatio || 1, w = c.clientWidth, hh = c.clientHeight;
  c.width = w * dpr; c.height = hh * dpr;
  const ctx = c.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  if (!hist.length) { ctx.fillStyle = "#5B6878"; ctx.font = "13px system-ui"; ctx.fillText("Waiting for the first epoch", 10, 20); return; }
  const L = 40, R = 40, T = 10, B = 22, W = w - L - R, H = hh - T - B;
  const members = [...new Set(hist.map(r => r.member))];
  const maxEp = Math.max(...hist.map(r => r.epoch)), maxLoss = Math.max(...hist.map(r => r.train_loss));
  const X = e => L + (maxEp > 1 ? (e - 1) / (maxEp - 1) : 0.5) * W;
  ctx.strokeStyle = "#CBD3DC"; ctx.strokeRect(L, T, W, H);
  ctx.fillStyle = "#5B6878"; ctx.font = "11px system-ui";
  ctx.fillText(maxLoss.toFixed(2), 4, T + 8); ctx.fillText("0", 28, T + H); ctx.fillText("100%", w - R + 4, T + 8);
  ctx.fillText("epoch", L + W / 2 - 14, hh - 4);
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
    this.root.append(h("header", {}, h("h1", {}, "Device"), this.select = h("select", { onchange: () => this.show(this.select.value) }),
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
        h("p", { class: "small muted", style: "margin:0" }, "All settings are optional; ChargeCell assumes no voltage scale. Unset values are measured from your scans or taken as a fraction of the current window. Recommended scans stay inside the safe limits you set, and moves larger than the step limit are split."),
        h("div", { class: "fields" }, L("Description", f.description), L("Carriers", f.carrier, "Electrons: more plunger voltage adds electrons. Holes: less voltage adds holes."),
          L("Plunger gates", f.plungers, "Comma separated, e.g. P1, P2, P3"), L("Sensor gate", f.sensor_gate), L("Exchange gates", f.barriers, "Which gate sits between each pair, e.g. P1-P2=X1, P2-P3=X2"),
          L("Tunnel gates", f.tunnel_gates, "Reservoir tunnel gate next to each edge plunger, e.g. P1=T1, P3=T2"),
          L("Max move (mV)", f.max_step, "Largest single DC step ChargeCell will suggest; bigger moves are split. Blank: at most one window width"),
          L("Exchange-gate step (mV)", f.barrier_step, "How far to change an exchange gate for merged dots or tie-bar coupling. Blank: a quarter of the electron spacing"),
          L("Electron temperature (mK)", f.electron_temperature, "Optional. With lever arms, tie-bar results are also given in µeV"),
          L("Tie-bar coupling target", f.tiebar_target, "Optional band for the tie-bar coupling ratio (interdot width / tie-bar length), e.g. 0.05, 0.3"),
          L("Points per electron", f.points_per_addition, "Resolution target for suggested scans"), L("Min points per axis", f.min_points), L("Max points per axis", f.max_points)),
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
    const j = running[0];
    ind.hidden = false;
    ind.replaceChildren(h("div", {}, j.title), h("div", { class: "small muted" }, j.message), h("div", { class: "bar" }, h("i", { style: `width:${j.progress * 100}%` })));
  } else ind.hidden = true;
  for (const id of S.running) {
    if (!running.find(j => j.id === id)) {
      const j = await api(`/api/jobs/${id}`, { quiet: true });
      if (j) toast(j.state === "done" ? `Finished: ${j.title}` : `${j.title}: ${j.message}`, j.state === "failed" ? "error" : "");
      const page = location.hash.split("/")[1];
      if (page === "synthetic") pages.synthetic.refresh();
      if (page === "train") { pages.train.enter(); }
      if (page === "scans") pages.scans.enter();
    }
  }
  S.running = new Set(running.map(j => j.id));
  const act = Object.keys(st.active_models || {});
  $("#model-indicator").textContent = act.length ? `Models: ${act.join(", ")}` : "No model yet";
}

async function route() {
  const parts = location.hash.replace(/^#\/?/, "").split("/");
  const name = pages[parts[0]] ? parts[0] : "review";
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
