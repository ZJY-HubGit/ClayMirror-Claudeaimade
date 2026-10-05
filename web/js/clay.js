// Clay deformation stack.
//  * Every deformation is anchored to a body part (face mesh vertex, hand bone,
//    pose bone …) and stores its parameters in that part's local frame, so it
//    keeps following the body after the user lets go.
//  * Deformations are composed sequentially (later ones act on the already
//    deformed image), which is what makes repeated pulling feel like clay.
//  * The math below is mirrored 1:1 in the WebGL fragment shader.
import { resolveAnchor } from './parts.js';

export const PRIM = { PULL: 0, INFLATE: 1, STRETCH: 2 };
export const PF = 16; // floats per primitive (4 RGBA32F texels)
const HOLD_MS = 700, FADE_IN = 6, FADE_OUT = 4;

// ------------------------------------------------------------ primitive math
// layout: [type, group, cx, cy,  r, p0, p1, p2,  p3, p4, p5, p6,  p7, p8, p9, p10]
// pull   : c = destination point, p0,p1 = displacement
// inflate: c = centre, p0 = amount (>0 grow, <0 shrink)
// stretch: c = destination centre, p0,p1 = source centre, p2,p3 = axis (cos,sin),
//          p4 = su, p5 = sv, p6,p7 = twist (cos,sin), p8 = plateau
function smooth01(e0, e1, x) { const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0))); return t * t * (3 - 2 * t); }

export function invPrim(P, o, x, y) {
  const type = P[o], cx = P[o + 2], cy = P[o + 3], r = P[o + 4];
  const dx = x - cx, dy = y - cy;
  if (type === 0) {
    const q = (dx * dx + dy * dy) / (r * r);
    if (q >= 1) return [x, y];
    const w = (1 - q) * (1 - q);
    return [x - P[o + 5] * w, y - P[o + 6] * w];
  }
  if (type === 1) {
    const q = (dx * dx + dy * dy) / (r * r);
    if (q >= 1) return [x, y];
    const k = 1 - q, f = 1 - P[o + 5] * k * k;
    return [cx + dx * f, cy + dy * f];
  }
  // stretch
  const ca = P[o + 7], sa = P[o + 8], su = P[o + 9], sv = P[o + 10], ct = P[o + 11], st = P[o + 12];
  const du = dx * ca + dy * sa, dv = -dx * sa + dy * ca;
  const ru = r * Math.max(su, 1), rv = r * Math.max(sv, 1);
  const rho = Math.sqrt((du * du) / (ru * ru) + (dv * dv) / (rv * rv));
  if (rho >= 1) return [x, y];
  const w = 1 - smooth01(P[o + 13], 1, rho);
  // e = Rot(-twist)·d
  const ex = dx * ct + dy * st, ey = -dx * st + dy * ct;
  const eu = ex * ca + ey * sa, ev = -ex * sa + ey * ca;
  const su_ = eu / su, sv_ = ev / sv;
  const sx = P[o + 5] + su_ * ca - sv_ * sa, sy = P[o + 6] + su_ * sa + sv_ * ca;
  return [x + (sx - x) * w, y + (sy - y) * w];
}

// conservative "can this primitive move point (x,y)?" test for forward mapping
function touches(P, o, x, y) {
  const type = P[o], cx = P[o + 2], cy = P[o + 3], r = P[o + 4];
  if (type === 0) {
    const R = r + Math.hypot(P[o + 5], P[o + 6]);
    return (x - cx) ** 2 + (y - cy) ** 2 < R * R;
  }
  if (type === 1) return (x - cx) ** 2 + (y - cy) ** 2 < r * r * 1.2;
  const R = r * Math.max(P[o + 9], P[o + 10], 1) + Math.hypot(P[o + 5] - cx, P[o + 6] - cy) + r * 0.2;
  return (x - cx) ** 2 + (y - cy) ** 2 < R * R || (x - P[o + 5]) ** 2 + (y - P[o + 6]) ** 2 < R * R;
}

// forward map of one primitive: solve invPrim(X) = (x,y) with Newton iterations
export function fwdPrim(P, o, x, y) {
  if (!touches(P, o, x, y)) return [x, y];
  let X = x, Y = y;
  if (P[o] === 0) { // good initial guess for pulls
    const q = ((x - P[o + 2]) ** 2 + (y - P[o + 3]) ** 2) / (P[o + 4] ** 2);
    if (q < 1) { const w = (1 - q) * (1 - q); X += P[o + 5] * w; Y += P[o + 6] * w; }
  }
  const h = 0.5;
  for (let it = 0; it < 8; it++) {
    const [gx, gy] = invPrim(P, o, X, Y);
    const ex = gx - x, ey = gy - y;
    if (ex * ex + ey * ey < 1e-4) break;
    const [ax, ay] = invPrim(P, o, X + h, Y);
    const [bx, by] = invPrim(P, o, X, Y + h);
    const j00 = (ax - gx) / h, j10 = (ay - gy) / h, j01 = (bx - gx) / h, j11 = (by - gy) / h;
    const det = j00 * j11 - j01 * j10;
    if (Math.abs(det) < 1e-6) { X -= ex * 0.5; Y -= ey * 0.5; continue; }
    let sx = (j11 * ex - j01 * ey) / det, sy = (-j10 * ex + j00 * ey) / det;
    const m = Math.hypot(sx, sy), lim = P[o + 4] * 0.5;
    if (m > lim) { sx *= lim / m; sy *= lim / m; }
    X -= sx; Y -= sy;
  }
  return [X, Y];
}

export function invChain(P, n, x, y, group = 0) {
  for (let i = n - 1; i >= 0; i--) {
    const o = i * PF;
    if (group && P[o + 1] !== group) continue;
    [x, y] = invPrim(P, o, x, y);
  }
  return [x, y];
}
export function fwdChain(P, n, x, y) {
  for (let i = 0; i < n; i++) [x, y] = fwdPrim(P, i * PF, x, y);
  return [x, y];
}

// ------------------------------------------------------------------- stack
let nextId = 1;
const rotv = (x, y, a) => { const c = Math.cos(a), s = Math.sin(a); return [x * c - y * s, x * s + y * c]; };

export class Clay {
  constructor() {
    this.items = [];
    this.P = new Float32Array(PF * 256);
    this.n = 0;
    this.maxItems = 48;
    this.resetting = 0;
  }
  get count() { return this.items.length; }

  add(item) {
    item.id = nextId++;
    item.vis = 0;
    item.born = performance.now();
    item.frame = null;
    item.last = null;
    this.items.push(item);
    while (this.items.length > this.maxItems) this.items.shift();
    return item;
  }
  remove(item) { const i = this.items.indexOf(item); if (i >= 0) this.items.splice(i, 1); }
  undo() {
    for (let i = this.items.length - 1; i >= 0; i--) {
      if (!this.items[i].live) { const [it] = this.items.splice(i, 1); return it; }
    }
    return null;
  }
  reset() { this.resetting = 1; }

  _ensure(nPrims) {
    if (this.P.length < nPrims * PF) {
      const Q = new Float32Array(Math.max(nPrims, this.P.length / PF * 2) * PF);
      Q.set(this.P); this.P = Q;
    }
  }
  _push(type, group, cx, cy, r, ...p) {
    this._ensure(this.n + 1);
    const o = this.n * PF, P = this.P;
    P.fill(0, o, o + PF);
    P[o] = type; P[o + 1] = group; P[o + 2] = cx; P[o + 3] = cy; P[o + 4] = r;
    for (let i = 0; i < p.length; i++) P[o + 5 + i] = p[i];
    this.n++;
  }

  // Resolve every item for this frame and rebuild the primitive buffer.
  build(F, now, dt = 1 / 30) {
    this.n = 0;
    if (this.resetting) {
      this.resetting = Math.max(0, this.resetting - dt * 3.2);
      if (this.resetting === 0) this.items = this.items.filter((it) => it.live);
    }
    const resetScale = this.resetting ? this.resetting : 1;
    for (const it of this.items) {
      let a = resolveAnchor(it.anchor, F);
      if (a) { it.last = { ...a, t: now }; }
      else if (it.last && now - it.last.t < HOLD_MS) a = it.last;
      const target = a ? 1 : 0;
      it.vis += (target - it.vis) * Math.min(1, dt * (target > it.vis ? FADE_IN : FADE_OUT));
      if (!a || it.vis < 0.01) { it.frame = null; continue; }
      // where is the anchor in the already-deformed picture?
      const [Dx, Dy] = fwdChain(this.P, this.n, a.x, a.y);
      it.frame = { x: Dx, y: Dy, rot: a.rot, s: a.s, rawX: a.x, rawY: a.y };
      const k = it.vis * (it.live ? 1 : resetScale);
      this._emit(it, it.frame, k);
    }
    return { P: this.P, n: this.n };
  }

  _emit(it, f, k) {
    const r = it.r * f.s;
    if (it.kind === 'pull') {
      // drag path in display space, resampled into equal steps (≤ 0.42·r each,
      // so every step is fold-free); consecutive steps compose into "taffy"
      const pts = it.path.map(([lx, ly]) => { const [ox, oy] = rotv(lx, ly, f.rot); return [f.x + ox * f.s * k, f.y + oy * f.s * k]; });
      const cum = [0];
      for (let i = 1; i < pts.length; i++) cum.push(cum[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]));
      const Ltot = cum[cum.length - 1];
      if (Ltot < 0.5) return;
      const nSteps = Math.min(32, Math.max(1, Math.ceil(Ltot / (r * 0.42))));
      let seg = 1, [px, py] = pts[0];
      for (let st = 1; st <= nSteps; st++) {
        const at = (Ltot * st) / nSteps;
        while (seg < pts.length - 1 && cum[seg] < at) seg++;
        const L = cum[seg] - cum[seg - 1] || 1;
        const u = Math.min(1, Math.max(0, (at - cum[seg - 1]) / L));
        const nx = pts[seg - 1][0] + (pts[seg][0] - pts[seg - 1][0]) * u, ny = pts[seg - 1][1] + (pts[seg][1] - pts[seg - 1][1]) * u;
        this._push(PRIM.PULL, it.group, nx, ny, r, nx - px, ny - py);
        px = nx; py = ny;
      }
    } else if (it.kind === 'inflate') {
      if (Math.abs(it.amount) > 1e-3) this._push(PRIM.INFLATE, it.group, f.x, f.y, r, it.amount * k);
    } else if (it.kind === 'stretch') {
      const ax = f.rot + it.axis, tw = it.twist * k;
      const su = 1 + (it.su - 1) * k, sv = 1 + (it.sv - 1) * k;
      const [ox, oy] = rotv(it.shift[0], it.shift[1], f.rot);
      const c1x = f.x + ox * f.s * k, c1y = f.y + oy * f.s * k;
      this._push(PRIM.STRETCH, it.group, c1x, c1y, r, f.x, f.y, Math.cos(ax), Math.sin(ax), su, sv, Math.cos(tw), Math.sin(tw), 0.5);
    }
  }

  // helpers for interaction
  sample(x, y) { return invChain(this.P, this.n, x, y, 0); }
  forward(x, y) { return fwdChain(this.P, this.n, x, y); }
  // primitives emitted before `item` (to place a new grab in the deformed picture)
  localMagnification(x, y) {
    const e = 2;
    const [ax, ay] = this.sample(x - e, y), [bx, by] = this.sample(x + e, y);
    const [cx, cy] = this.sample(x, y - e), [dx, dy] = this.sample(x, y + e);
    const sx = Math.hypot(bx - ax, by - ay) / (2 * e), sy = Math.hypot(dx - cx, dy - cy) / (2 * e);
    const m = 1 / Math.max(0.2, Math.sqrt(sx * sy));
    return Math.min(3, Math.max(0.5, m));
  }
}

// ------------------------------------------------------- local conversions
export function toLocal(f, x, y) {
  const [lx, ly] = rotv(x - f.x, y - f.y, -f.rot);
  return [lx / f.s, ly / f.s];
}
export function angleToLocal(f, a) { return a - f.rot; }
