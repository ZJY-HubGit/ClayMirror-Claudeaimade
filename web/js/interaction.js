// Gestures → clay deformations.
//   pinch + drag            → pull ("拉扯")
//   pinch + hold still      → inflate / shrink ("充气放大")
//   two pinches on one part → stretch / squeeze / twist ("双手拉伸")
//   both palms open (1.6 s) → reset
// Mouse / touch work the same way (mouse button or finger = pinch).
import { classify, PARTS, partLabel, featureAnchor, headAnchor, targetOutline, resolveAnchor } from './parts.js';
import { toLocal } from './clay.js';

const PINCH_ON = 0.27, PINCH_OFF = 0.42, HOVER = 0.8;
const HOLD_TIME = 0.45;
const INFLATE_RATE = 0.5;
const FACE_PARTS = new Set(['eye', 'brow', 'nose', 'mouth', 'chin', 'cheek', 'forehead', 'face', 'head', 'ear']);
const hyp = Math.hypot;
const normA = (a) => Math.atan2(Math.sin(a), Math.cos(a));

function isOpenPalm(lm, size) {
  const d = (a, b) => hyp(lm[a * 3] - lm[b * 3], lm[a * 3 + 1] - lm[b * 3 + 1]);
  for (const [tip, pip] of [[8, 6], [12, 10], [16, 14], [20, 18]]) if (d(tip, 0) < d(pip, 0) * 1.12) return false;
  if (d(4, 5) < 0.42 * size || d(4, 8) < 0.55 * size) return false;
  return true;
}

export class Interaction {
  constructor(clay, sfx) {
    this.clay = clay;
    this.sfx = sfx;
    this.cursors = new Map();
    this.pointers = new Map();
    this.mode = 'grow';
    this.resetT = 0;
    this.resetInfo = null;
    this.cooldown = 0;
    this.frameSize = [1280, 720];
    this.onEvent = null;
    this.wheelItem = null;
  }

  // ------------------------------------------------------------ pointers
  pointerDown(id, x, y, type) { this.pointers.set(id, { x, y, down: true, type }); }
  pointerMove(id, x, y, type) {
    const p = this.pointers.get(id);
    if (p) { p.x = x; p.y = y; }
    else if (type === 'mouse') this.pointers.set(id, { x, y, down: false, type });
  }
  pointerUp(id) { const p = this.pointers.get(id); if (p) { p.down = false; if (p.type !== 'mouse') p.remove = true; } }
  pointerLeave(id) { const p = this.pointers.get(id); if (p && !p.down) this.pointers.delete(id); }

  wheel(x, y, delta, F) {
    const [ax, ay] = this.clay.sample(x, y);
    const t = classify(ax, ay, F, null, this.frameSize);
    const cfg = PARTS[t.part];
    const now = performance.now();
    let it = this.wheelItem;
    if (!it || now - it.wheelT > 1500 || it.part !== t.part || it.side !== t.side || !this.clay.items.includes(it)) {
      const anchor = t.part === 'head' ? headAnchor(F) : featureAnchor(t, F) || t.anchor;
      it = this.clay.add({ kind: 'inflate', anchor, group: t.group, part: t.part, side: t.side, label: partLabel(t),
        r: cfg.inflate * this.clay.localMagnification(x, y), amount: 0 });
      this.wheelItem = it;
    }
    it.wheelT = now;
    it.amount = Math.max(-0.6, Math.min(cfg.inflateMax, it.amount - delta * 0.0011));
    this.sfx && this.sfx.tick(it.amount);
  }

  // --------------------------------------------------------------- main
  update(F, det, now, dt) {
    if (this.cooldown > 0) this.cooldown -= dt;
    const seen = new Set();
    // hands → cursors
    for (const h of det.hands) {
      if (!h.side) continue;
      const id = 'hand-' + h.side;
      seen.add(id);
      const c = this._cursor(id, 'hand');
      const lm = h.lm;
      const size = hyp(lm[0] - lm[27], lm[1] - lm[28]) || 1;
      const d = hyp(lm[12] - lm[24], lm[13] - lm[25], (lm[14] - lm[26]) * 0.5) / size;
      c.side = h.side; c.size = size; c.lm = lm; c.pinchDist = d;
      const nx = (lm[12] + lm[24]) / 2, ny = (lm[13] + lm[25]) / 2;
      const sp = c.lastSeen ? hyp(nx - c.x, ny - c.y) / Math.max(dt, 1e-3) / size : 0;
      c.speed = c.speed == null ? sp : c.speed * 0.7 + sp * 0.3;
      c.x = nx; c.y = ny;
      if (!c.pinch) { c.on = d < PINCH_ON ? c.on + 1 : 0; if (c.on >= 2) { c.pinch = true; c.off = 0; } }
      else { c.off = d > PINCH_OFF ? c.off + 1 : 0; if (c.off >= 2) { c.pinch = false; c.on = 0; } }
      // fingers opening: freeze the grab so the hand moving away does not overshoot
      c.releasing = c.pinch && d > PINCH_OFF;
      c.hover = d < HOVER;
      c.closeness = Math.max(0, Math.min(1, (HOVER - d) / (HOVER - PINCH_ON)));
      c.open = isOpenPalm(lm, size);
      c.lastSeen = now;
    }
    // pointers → cursors
    for (const [pid, p] of this.pointers) {
      const id = 'ptr-' + pid;
      seen.add(id);
      const c = this._cursor(id, p.type === 'mouse' ? 'mouse' : 'touch');
      c.x = p.x; c.y = p.y; c.pinch = p.down; c.hover = true; c.closeness = p.down ? 1 : 0.6; c.side = null; c.size = 60;
      c.lastSeen = now;
      if (p.remove && !p.down) this.pointers.delete(pid);
    }
    // vanished cursors
    for (const [id, c] of this.cursors) {
      if (seen.has(id)) continue;
      if (now - (c.lastSeen || 0) < 250 && c.source === 'hand') { c.stale = true; continue; } // brief tracking loss
      if (c.grab) this._end(c, now);
      this.cursors.delete(id);
    }

    // transitions
    for (const c of this.cursors.values()) {
      if (c.stale && !seen.has(c.id)) continue;
      c.stale = false;
      if (!c.pinch) c.blocked = false;
      if (c.pinch && !c.grab && !c.blocked && !c.wasPinch) this._start(c, F, now);
      else if (!c.pinch && c.grab) this._end(c, now);
      c.wasPinch = c.pinch;
    }
    const driven = new Set();
    for (const c of this.cursors.values()) {
      if (c.grab && !driven.has(c.grab)) { driven.add(c.grab); this._drive(c.grab, c, F, now, dt); }
      // hover target (for highlight + label)
      if (!c.grab && c.hover && (c.source !== 'hand' || c.closeness > 0.15)) {
        const [ax, ay] = this.clay.sample(c.x, c.y);
        c.target = classify(ax, ay, F, this._excluded(c), this.frameSize);
      } else if (!c.grab) c.target = null;
    }
    this._resetGesture(det, now, dt);
  }

  // hands that are pinching (tools) are never the target of a grab
  _excluded(c) {
    const ex = [c.side];
    for (const o of this.cursors.values()) if (o !== c && o.source === 'hand' && !o.stale && (o.pinch || o.closeness > 0.55)) ex.push(o.side);
    return ex;
  }

  _cursor(id, source) {
    let c = this.cursors.get(id);
    if (!c) { c = { id, source, x: 0, y: 0, pinch: false, wasPinch: false, on: 0, off: 0, grab: null, target: null, closeness: 0 }; this.cursors.set(id, c); }
    return c;
  }

  _frameFor(anchor, F) {
    const a = resolveAnchor(anchor, F);
    if (!a) return null;
    const [x, y] = this.clay.forward(a.x, a.y);
    return { x, y, rot: a.rot, s: a.s };
  }

  _start(c, F, now) {
    // second hand / finger joining an existing fresh grab → two-handed stretch
    for (const o of this.cursors.values()) {
      if (o === c || !o.grab || o.grab.kind === 'stretch') continue;
      if ((o.source === 'hand') !== (c.source === 'hand')) continue;
      const g = o.grab;
      const fresh = g.kind === 'pending' || (now - g.t0 < 900 && g.moved < 40);
      if (!fresh) continue;
      const [bx, by] = this.clay.sample(c.x, c.y);
      const tb = classify(bx, by, F, this._excluded(c), this.frameSize);
      if (this._sameRegion(g.target, tb, o, c)) { this._startStretch(o, c, F, now); return; }
    }
    const [ax, ay] = this.clay.sample(c.x, c.y);
    const target = classify(ax, ay, F, this._excluded(c), this.frameSize);
    c.grab = { kind: 'pending', target, P: [c.x, c.y], t0: now, moved: 0, item: null, cursor: c };
    c.target = target;
    this.sfx && this.sfx.grab();
  }

  _sameRegion(a, b, ca, cb) {
    if (!a || !b) return false;
    const dist = hyp(ca.x - cb.x, ca.y - cb.y);
    if (dist < 8) return false;
    if (FACE_PARTS.has(a.part) && FACE_PARTS.has(b.part)) return true;
    if ((a.part === 'finger' || a.part === 'palm') && (b.part === 'finger' || b.part === 'palm')) return a.side === b.side;
    if (a.part === b.part && a.part !== 'screen') return true;
    if (a.part === 'screen' && b.part === 'screen') return true;
    const unit = Math.max(a.s || 60, b.s || 60);
    return dist < unit * 2.2 && a.part !== 'screen' && b.part !== 'screen' && !FACE_PARTS.has(a.part) && !FACE_PARTS.has(b.part);
  }

  _startStretch(ca, cb, F, now) {
    const old = ca.grab;
    if (old.item) this.clay.remove(old.item);
    const PA = [ca.x, ca.y], PB = [cb.x, cb.y];
    const C0 = [(PA[0] + PB[0]) / 2, (PA[1] + PB[1]) / 2];
    const [rx, ry] = this.clay.sample(C0[0], C0[1]);
    // anchor at the midpoint between the two pinches (never on a grabbing hand)
    const t = classify(rx, ry, F, [ca.side, cb.side], this.frameSize);
    const f = this._frameFor(t.anchor, F) || { x: C0[0], y: C0[1], rot: 0, s: t.s || 60 };
    const L0 = Math.max(hyp(PB[0] - PA[0], PB[1] - PA[1]), 10);
    const ang0 = Math.atan2(PB[1] - PA[1], PB[0] - PA[0]);
    const cfg = PARTS[t.part] || PARTS.face;
    const item = this.clay.add({ kind: 'stretch', anchor: t.anchor, group: t.group, part: t.part, side: t.side, label: partLabel(t),
      r: Math.max((1.05 * L0) / f.s, cfg.pull), axis: ang0 - f.rot, su: 1, sv: 1, twist: 0, shift: [0, 0], live: true });
    item.frame = f;
    const g = { kind: 'stretch', cursors: [ca, cb], L0, ang0, item, t0: now, target: t, moved: 0 };
    ca.grab = g; cb.grab = g;
    ca.target = t; cb.target = t;
    this.sfx && this.sfx.stretch();
    this.onEvent && this.onEvent('stretch', item);
  }

  _makePull(g, F) {
    const t = g.target, cfg = PARTS[t.part];
    const mag = this.clay.localMagnification(g.P[0], g.P[1]);
    const item = this.clay.add({ kind: 'pull', anchor: t.anchor, group: t.group, part: t.part, side: t.side, label: partLabel(t),
      r: cfg.pull * mag, path: [[0, 0], [0, 0]], live: true });
    item.frame = this._frameFor(t.anchor, F);
    g.kind = 'pull';
    g.item = item;
    this.onEvent && this.onEvent('pull', item);
  }

  _makeInflate(g, F) {
    const t = g.target, cfg = PARTS[t.part];
    let anchor = t.anchor;
    if (t.part === 'head') anchor = headAnchor(F) || anchor;
    else if (t.featureIdx) anchor = featureAnchor(t, F) || anchor;
    const mag = this.clay.localMagnification(g.P[0], g.P[1]);
    const item = this.clay.add({ kind: 'inflate', anchor, group: t.group, part: t.part, side: t.side, label: partLabel(t),
      r: cfg.inflate * mag, amount: 0, live: true });
    item.frame = this._frameFor(anchor, F);
    g.kind = 'inflate';
    g.item = item;
    g.inflating = true;
    g.inflateT0 = performance.now();
    g.sign = this.mode === 'shrink' ? -1 : 1;
    g.max = g.sign > 0 ? cfg.inflateMax : 0.55;
    this.sfx && this.sfx.inflateStart(g.sign);
    this.onEvent && this.onEvent('inflate', item);
  }

  _drive(g, c, F, now, dt) {
    if (g.kind === 'stretch') {
      if (g.cursors.some((o) => o.releasing)) return;
      return this._driveStretch(g, F, now, dt);
    }
    if (c.releasing) return;
    const Q = [c.x, c.y];
    const dist = hyp(Q[0] - g.P[0], Q[1] - g.P[1]);
    g.moved = Math.max(g.moved, dist);
    const unit = (g.target && g.target.s) || 60;
    const thr = Math.max(9, 0.09 * unit * (PARTS[g.target.part] || PARTS.face).pull * 2);
    if (g.kind === 'pending') {
      if (dist > thr) this._makePull(g, F);
      else if (now - g.t0 > HOLD_TIME * 1000) this._makeInflate(g, F);
      g.hold = Math.min(1, (now - g.t0) / (HOLD_TIME * 1000));
    }
    // started inflating but then clearly moved away: the user meant to pull
    if (g.kind === 'inflate' && g.inflating && dist > thr * 2.2 && Math.abs(g.item.amount) < 0.15 && performance.now() - g.inflateT0 < 900) {
      this.clay.remove(g.item);
      this.sfx && this.sfx.inflateStop();
      g.item = null;
      g.kind = 'pending';
      this._makePull(g, F);
    }
    if (g.kind === 'pull') {
      const it = g.item, f = it.frame;
      if (!f) return;
      const [lx, ly] = toLocal(f, Q[0], Q[1]);
      const path = it.path;
      path[path.length - 1] = [lx, ly];
      const prev = path[path.length - 2];
      if (hyp(lx - prev[0], ly - prev[1]) > it.r * 0.3 && path.length < 64) path.push([lx, ly]);
      this.sfx && this.sfx.drag(Math.min(1, hyp(lx, ly) / (it.r * 2)));
    } else if (g.kind === 'inflate') {
      const it = g.item;
      if (g.inflating) {
        it.amount += g.sign * INFLATE_RATE * dt * (1 - Math.abs(it.amount) / (g.max + 0.15));
        it.amount = g.sign > 0 ? Math.min(g.max, it.amount) : Math.max(-g.max, it.amount);
        this.sfx && this.sfx.inflate(Math.abs(it.amount) / g.max);
        if (dist > thr * 3.5) { g.inflating = false; this.sfx && this.sfx.inflateStop(); }
      }
    }
  }

  _driveStretch(g, F, now, dt) {
    const [a, b] = g.cursors, it = g.item, f = it.frame;
    if (!f) return;
    const L = Math.max(hyp(b.x - a.x, b.y - a.y), 1);
    const su = Math.max(0.7, Math.min(3.5, L / g.L0));
    let tw = normA(Math.atan2(b.y - a.y, b.x - a.x) - g.ang0);
    const dead = 0.14;
    tw = Math.abs(tw) < dead ? 0 : tw - Math.sign(tw) * dead;
    tw = Math.max(-0.75, Math.min(0.75, tw * 0.85));
    it.su = su;
    it.sv = Math.max(0.82, Math.min(1.1, Math.pow(su, -0.18)));
    it.twist = tw;
    it.shift = toLocal(f, (a.x + b.x) / 2, (a.y + b.y) / 2);
    g.moved = Math.max(g.moved, Math.abs(L - g.L0));
    this.sfx && this.sfx.drag(Math.min(1, Math.abs(su - 1)));
  }

  _end(c, now) {
    const g = c.grab;
    if (!g) return;
    if (g.kind === 'stretch') {
      for (const o of g.cursors) { if (o !== c) o.blocked = true; o.grab = null; }
      g.item.live = false;
      this.sfx && this.sfx.release();
      this.onEvent && this.onEvent('commit', g.item);
      return;
    }
    c.grab = null;
    if (g.kind === 'pending') { this.sfx && this.sfx.cancel(); return; }
    if (g.item) {
      g.item.live = false;
      if (g.kind === 'pull' && g.item.path.length >= 2) {
        const last = g.item.path[g.item.path.length - 1];
        if (hyp(last[0], last[1]) < 0.02) this.clay.remove(g.item);
      }
      if (g.kind === 'inflate' && Math.abs(g.item.amount) < 0.02) this.clay.remove(g.item);
    }
    if (g.kind === 'inflate') this.sfx && this.sfx.inflateStop();
    this.sfx && this.sfx.release();
    this.onEvent && this.onEvent('commit', g.item);
  }

  cancelAll() {
    for (const c of this.cursors.values()) {
      if (c.grab && c.grab.item) c.grab.item.live = false;
      if (c.grab) c.blocked = true;
      c.grab = null;
    }
  }

  _resetGesture(det, now, dt) {
    const open = [...this.cursors.values()].filter((c) => c.source === 'hand' && !c.stale && c.open && !c.pinch && !c.grab && (c.speed || 0) < 1.5);
    if (open.length === 2 && this.clay.count > 0 && this.cooldown <= 0) {
      this.resetT += dt;
      const [a, b] = open;
      this.resetInfo = { progress: Math.min(1, this.resetT / 1.6), x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, hands: open.map((c) => [c.x, c.y]) };
      if (this.resetT >= 1.6) {
        this.clay.reset();
        this.resetT = 0;
        this.cooldown = 2.5;
        this.resetInfo = null;
        this.sfx && this.sfx.reset();
        this.onEvent && this.onEvent('reset');
      }
    } else {
      this.resetT = Math.max(0, this.resetT - dt * 2);
      this.resetInfo = this.resetT > 0 && this.resetInfo ? { ...this.resetInfo, progress: this.resetT / 1.6 } : null;
    }
  }

  // for the overlay
  snapshot() {
    return {
      cursors: [...this.cursors.values()].filter((c) => !c.stale),
      reset: this.resetInfo,
    };
  }
  outlineOf(target, F) { return targetOutline(target, F); }
}
