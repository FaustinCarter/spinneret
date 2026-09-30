(() => {
"use strict";
/* Canvas plot for charge-stability diagrams. Data coordinates are volts; axes show mV. */

const VIRIDIS = [[68,1,84],[72,40,120],[62,74,137],[49,104,142],[38,130,142],[31,158,137],
                 [53,183,121],[110,206,88],[181,222,43],[253,231,37]];
const CMAPS = {
  gray: t => [t * 255, t * 255, t * 255],
  viridis: t => {
    const x = Math.min(0.9999, Math.max(0, t)) * (VIRIDIS.length - 1);
    const i = Math.floor(x), f = x - i, a = VIRIDIS[i], b = VIRIDIS[i + 1];
    return [a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2])];
  },
};

function b64ToBytes(b64) {
  const bin = atob(b64);
  const u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return u8;
}
function decodeF32(b64) { return new Float32Array(b64ToBytes(b64).buffer); }
function decodeU8(b64) { return b64ToBytes(b64); }

function niceTicks(lo, hi, n = 6) {
  const span = hi - lo;
  if (!(span > 0)) return [lo];
  const raw = span / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => s >= raw) || 10 * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-12; v += step) out.push(v);
  return out;
}

class Plot {
  constructor(canvas) {
    this.c = canvas;
    this.ctx = canvas.getContext("2d");
    this.m = { l: 64, r: 14, t: 14, b: 46 };
    this.scan = null;
    this.mode = "signal";
    this.cmap = "gray";
    this.clip = 1;
    this.invert = false;
    this.layers = [];
    this.extra = null;
    this.view = null;
    this.hooks = {};
    this.panOnDrag = false;
    this.cursor = null;
    new ResizeObserver(() => this.render()).observe(canvas);
    canvas.addEventListener("wheel", e => this._wheel(e), { passive: false });
    canvas.addEventListener("pointerdown", e => this._down(e));
    canvas.addEventListener("pointermove", e => this._move(e));
    canvas.addEventListener("pointerup", e => this._up(e));
    canvas.addEventListener("pointerleave", () => { this.cursor = null; this.render(); });
    canvas.addEventListener("dblclick", e => this.hooks.dblclick && this.hooks.dblclick(this._pt(e), e));
    canvas.addEventListener("contextmenu", e => {
      if (this.hooks.context) { e.preventDefault(); this.hooks.context(this._pt(e), e); }
    });
  }

  setScan(scan) {
    this.scan = scan;
    this._buildImage();
    this.resetView();
  }

  setMode(mode) { this.mode = mode; this._buildImage(); this.render(); }
  setCmap(c) { this.cmap = c; this._buildImage(); this.render(); }
  setClip(p) { this.clip = p; this._buildImage(); this.render(); }
  setInvert(v) { this.invert = v; this._buildImage(); this.render(); }

  dataExtent() {
    const s = this.scan, nx = s.nx, ny = s.ny;
    const dx = nx > 1 ? (s.x[nx - 1] - s.x[0]) / (nx - 1) : 1e-3;
    const dy = ny > 1 ? (s.y[ny - 1] - s.y[0]) / (ny - 1) : 1e-3;
    return [s.x[0] - dx / 2, s.x[nx - 1] + dx / 2, s.y[0] - dy / 2, s.y[ny - 1] + dy / 2];
  }

  resetView() {
    if (!this.scan) return;
    let [x0, x1, y0, y1] = this.dataExtent();
    if (this.extra) {
      x0 = Math.min(x0, this.extra[0]); x1 = Math.max(x1, this.extra[1]);
      y0 = Math.min(y0, this.extra[2]); y1 = Math.max(y1, this.extra[3]);
    }
    const px = (x1 - x0) * 0.03, py = (y1 - y0) * 0.03;
    this.view = [x0 - px, x1 + px, y0 - py, y1 + py];
    this.render();
  }

  _buildImage() {
    const s = this.scan;
    if (!s) return;
    const { nx, ny } = s;
    let v = s.signal;
    if (this.mode === "gradient") {
      const g = new Float32Array(nx * ny);
      for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
        const i0 = Math.max(0, i - 1), i1 = Math.min(nx - 1, i + 1);
        const j0 = Math.max(0, j - 1), j1 = Math.min(ny - 1, j + 1);
        const gx = (v[j * nx + i1] - v[j * nx + i0]) / Math.max(1, i1 - i0);
        const gy = (v[j1 * nx + i] - v[j0 * nx + i]) / Math.max(1, j1 - j0);
        g[j * nx + i] = Math.hypot(gx, gy);
      }
      v = g;
    }
    const sorted = Array.from(v).filter(Number.isFinite).sort((a, b) => a - b);
    const q = p => sorted[Math.min(sorted.length - 1, Math.max(0, Math.round(p / 100 * (sorted.length - 1))))];
    const lo = this.mode === "gradient" ? q(1) : q(this.clip);
    const hi = q(100 - this.clip) || lo + 1;
    const off = document.createElement("canvas");
    off.width = nx; off.height = ny;
    const octx = off.getContext("2d");
    const img = octx.createImageData(nx, ny);
    const cm = CMAPS[this.cmap] || CMAPS.gray;
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      let t = (v[j * nx + i] - lo) / (hi - lo || 1);
      t = Math.min(1, Math.max(0, t));
      if (this.invert) t = 1 - t;
      const [r, g, b] = cm(t), k = 4 * (j * nx + i);
      img.data[k] = r; img.data[k + 1] = g; img.data[k + 2] = b; img.data[k + 3] = 255;
    }
    octx.putImageData(img, 0, 0);
    this.image = off;
  }

  area() {
    const w = this.c.clientWidth, h = this.c.clientHeight;
    return { L: this.m.l, T: this.m.t, W: Math.max(10, w - this.m.l - this.m.r), H: Math.max(10, h - this.m.t - this.m.b) };
  }
  toScreen(vx, vy) {
    const a = this.area(), [x0, x1, y0, y1] = this.view;
    return [a.L + (vx - x0) / (x1 - x0) * a.W, a.T + (y1 - vy) / (y1 - y0) * a.H];
  }
  toData(X, Y) {
    const a = this.area(), [x0, x1, y0, y1] = this.view;
    return [x0 + (X - a.L) / a.W * (x1 - x0), y1 - (Y - a.T) / a.H * (y1 - y0)];
  }
  pxPerVolt() {
    const a = this.area(), [x0, x1, y0, y1] = this.view;
    return [a.W / (x1 - x0), a.H / (y1 - y0)];
  }

  /* Draw an image whose pixel (u,v) centre sits at (xa + u*dx, ya + v*dy); handles reversed grids. */
  drawGridImage(img, extent) {
    const [xa, xb, ya, yb] = extent, nx = img.width, ny = img.height;
    const dx = nx > 1 ? (xb - xa) / (nx - 1) : 0, dy = ny > 1 ? (yb - ya) / (ny - 1) : 0;
    const [kx, ky] = this.pxPerVolt();
    const [X0, Y0] = this.toScreen(xa - dx / 2, ya - dy / 2);
    const ctx = this.ctx;
    ctx.save();
    ctx.translate(X0, Y0);
    ctx.scale(dx * kx, -dy * ky);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(img, 0, 0);
    ctx.restore();
  }

  render() {
    const c = this.c, ctx = this.ctx, dpr = window.devicePixelRatio || 1;
    const w = c.clientWidth, h = c.clientHeight;
    if (!w || !h) return;
    if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) {
      c.width = Math.round(w * dpr); c.height = Math.round(h * dpr);
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    if (!this.scan || !this.view) {
      ctx.fillStyle = "#5B6878"; ctx.font = "14px system-ui";
      ctx.fillText("No scan selected", 20, 30);
      return;
    }
    const a = this.area();
    ctx.fillStyle = "#F4F6F8";
    ctx.fillRect(a.L, a.T, a.W, a.H);
    ctx.save();
    ctx.beginPath(); ctx.rect(a.L, a.T, a.W, a.H); ctx.clip();
    const s = this.scan;
    this.drawGridImage(this.image, [s.x[0], s.x[s.nx - 1], s.y[0], s.y[s.ny - 1]]);
    for (const layer of this.layers) { ctx.save(); layer(ctx, this); ctx.restore(); }
    ctx.restore();
    this._axes(a);
    if (this.cursor) {
      const [vx, vy] = this.cursor;
      const txt = `${s.xLabel} ${(vx * 1e3).toFixed(1)} mV   ${s.yLabel} ${(vy * 1e3).toFixed(1)} mV`;
      ctx.font = "12px system-ui";
      const tw = ctx.measureText(txt).width;
      ctx.fillStyle = "rgba(255,255,255,0.88)";
      ctx.fillRect(a.L + a.W - tw - 14, a.T + 4, tw + 10, 18);
      ctx.fillStyle = "#17202B";
      ctx.fillText(txt, a.L + a.W - tw - 9, a.T + 17);
    }
  }

  _axes(a) {
    const ctx = this.ctx, [x0, x1, y0, y1] = this.view;
    ctx.strokeStyle = "#9AA7B6"; ctx.lineWidth = 1;
    ctx.strokeRect(a.L + 0.5, a.T + 0.5, a.W - 1, a.H - 1);
    ctx.fillStyle = "#17202B"; ctx.font = "12px system-ui";
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    for (const t of niceTicks(x0 * 1e3, x1 * 1e3, Math.max(3, Math.floor(a.W / 90)))) {
      const [X] = this.toScreen(t / 1e3, y0);
      ctx.beginPath(); ctx.moveTo(X, a.T + a.H); ctx.lineTo(X, a.T + a.H + 5); ctx.stroke();
      ctx.fillText(+t.toFixed(3), X, a.T + a.H + 7);
    }
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    for (const t of niceTicks(y0 * 1e3, y1 * 1e3, Math.max(3, Math.floor(a.H / 70)))) {
      const [, Y] = this.toScreen(x0, t / 1e3);
      ctx.beginPath(); ctx.moveTo(a.L - 5, Y); ctx.lineTo(a.L, Y); ctx.stroke();
      ctx.fillText(+t.toFixed(3), a.L - 7, Y);
    }
    ctx.textAlign = "center"; ctx.textBaseline = "alphabetic"; ctx.font = "600 13px system-ui";
    ctx.fillText(`${this.scan.xLabel} (mV)`, a.L + a.W / 2, a.T + a.H + 40);
    ctx.save(); ctx.translate(15, a.T + a.H / 2); ctx.rotate(-Math.PI / 2);
    ctx.fillText(`${this.scan.yLabel} (mV)`, 0, 0); ctx.restore();
  }

  _pt(e) {
    const r = this.c.getBoundingClientRect();
    const X = e.clientX - r.left, Y = e.clientY - r.top;
    const [x, y] = this.toData(X, Y);
    return { x, y, X, Y };
  }
  _wheel(e) {
    if (!this.view) return;
    e.preventDefault();
    const p = this._pt(e), f = Math.exp(e.deltaY * 0.0015);
    const [x0, x1, y0, y1] = this.view;
    this.view = [p.x + (x0 - p.x) * f, p.x + (x1 - p.x) * f, p.y + (y0 - p.y) * f, p.y + (y1 - p.y) * f];
    this.render();
  }
  _down(e) {
    if (!this.view) return;
    const p = this._pt(e);
    const pan = e.button === 1 || e.shiftKey || (this.panOnDrag && e.button === 0);
    if (!pan && e.button === 0 && this.hooks.down && this.hooks.down(p, e)) {
      this._drag = "hook"; this.c.setPointerCapture(e.pointerId); return;
    }
    if (pan) {
      this._drag = { X: p.X, Y: p.Y, view: this.view.slice() };
      this.c.setPointerCapture(e.pointerId);
    }
  }
  _move(e) {
    if (!this.view) return;
    const p = this._pt(e);
    this.cursor = [p.x, p.y];
    if (this._drag === "hook") { this.hooks.drag && this.hooks.drag(p, e); }
    else if (this._drag) {
      const a = this.area(), [x0, x1, y0, y1] = this._drag.view;
      const ddx = (p.X - this._drag.X) / a.W * (x1 - x0), ddy = (p.Y - this._drag.Y) / a.H * (y1 - y0);
      this.view = [x0 - ddx, x1 - ddx, y0 + ddy, y1 + ddy];
    } else if (this.hooks.hover) { this.hooks.hover(p, e); }
    this.render();
  }
  _up(e) {
    if (this._drag === "hook" && this.hooks.up) this.hooks.up(this._pt(e), e);
    this._drag = null;
  }
}

window.ChargePlot = { Plot, decodeF32, decodeU8, niceTicks, CMAPS };
})();
