// Body-part knowledge: which part sits under a point, and how to anchor a
// deformation to it so that it follows the body afterwards.

// FaceMesh index sets (MediaPipe face_mesh_connections). "left/right" = the
// person's own left/right.
export const FACE_IDX = {
  oval: [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109],
  lipsOuter: [0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61, 185, 40, 39, 37],
  lipsInner: [13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78, 191, 80, 81, 82],
  leftEye: [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466],
  rightEye: [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
  leftBrow: [276, 283, 282, 295, 285, 336, 296, 334, 293, 300],
  rightBrow: [46, 53, 52, 65, 55, 107, 66, 105, 63, 70],
  noseBridge: [168, 6, 197, 195, 5, 4, 1],
  nose: [168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 98, 97, 326, 327, 294, 278, 344, 440, 275, 45, 220, 115, 48, 64],
  noseBall: [4, 1, 5, 19, 98, 327, 195, 45, 275],
  chin: [152, 175, 199, 200, 148, 377, 176, 400],
  leftCheek: [425, 411, 280, 352, 376, 427],
  rightCheek: [205, 187, 50, 123, 147, 207],
  forehead: [151, 9, 108, 337, 69, 299, 10],
};

export const HAND_BONES = [
  // [a, b, finger] ; finger: 0 thumb .. 4 pinky, 5 palm
  [1, 2, 0], [2, 3, 0], [3, 4, 0],
  [5, 6, 1], [6, 7, 1], [7, 8, 1],
  [9, 10, 2], [10, 11, 2], [11, 12, 2],
  [13, 14, 3], [14, 15, 3], [15, 16, 3],
  [17, 18, 4], [18, 19, 4], [19, 20, 4],
  [0, 1, 5], [0, 5, 5], [0, 17, 5], [5, 9, 5], [9, 13, 5], [13, 17, 5],
];
export const FINGER_NAMES = ['拇指', '食指', '中指', '无名指', '小指', '手掌'];

// pose bones: endpoint = landmark index or [i, j] (midpoint)
export const POSE_BONES = [
  { a: 11, b: 13, part: 'upperArm', side: 'L', w: 0.24 },
  { a: 12, b: 14, part: 'upperArm', side: 'R', w: 0.24 },
  { a: 13, b: 15, part: 'forearm', side: 'L', w: 0.2 },
  { a: 14, b: 16, part: 'forearm', side: 'R', w: 0.2 },
  { a: 15, b: 19, part: 'forearm', side: 'L', w: 0.18 },
  { a: 16, b: 20, part: 'forearm', side: 'R', w: 0.18 },
  { a: 23, b: 25, part: 'leg', side: 'L', w: 0.3 },
  { a: 24, b: 26, part: 'leg', side: 'R', w: 0.3 },
  { a: 25, b: 27, part: 'leg', side: 'L', w: 0.22 },
  { a: 26, b: 28, part: 'leg', side: 'R', w: 0.22 },
  { a: [11, 12], b: 0, part: 'neck', w: 0.22 },
  { a: 11, b: 12, part: 'shoulder', w: 0.22 },
  { a: 11, b: 23, part: 'torso', side: 'L', w: 0.3 },
  { a: 12, b: 24, part: 'torso', side: 'R', w: 0.3 },
  { a: [11, 12], b: [23, 24], part: 'torso', w: 0.5 },
];

// radius multipliers are in the anchor's unit: face = eye distance (IOD),
// hand = palm length, pose = shoulder width.
export const PARTS = {
  eye:      { label: '眼睛', icon: '👁️', pull: 0.42, inflate: 0.42, inflateMax: 0.7 },
  brow:     { label: '眉毛', icon: '〰️', pull: 0.45, inflate: 0.38, inflateMax: 0.5 },
  nose:     { label: '鼻子', icon: '👃', pull: 0.5, inflate: 0.48, inflateMax: 0.7 },
  mouth:    { label: '嘴巴', icon: '👄', pull: 0.55, inflate: 0.6, inflateMax: 0.68 },
  chin:     { label: '下巴', icon: '🫠', pull: 0.85, inflate: 0.62, inflateMax: 0.55 },
  cheek:    { label: '脸颊', icon: '😊', pull: 0.85, inflate: 0.62, inflateMax: 0.6 },
  forehead: { label: '额头', icon: '🧠', pull: 0.95, inflate: 0.8, inflateMax: 0.55 },
  face:     { label: '脸', icon: '🙂', pull: 0.85, inflate: 0.75, inflateMax: 0.55 },
  head:     { label: '头', icon: '🗿', pull: 1.15, inflate: 1.9, inflateMax: 0.5 },
  ear:      { label: '耳朵', icon: '👂', pull: 0.55, inflate: 0.5, inflateMax: 0.65 },
  finger:   { label: '手指', icon: '☝️', pull: 0.55, inflate: 0.32, inflateMax: 0.6 },
  palm:     { label: '手掌', icon: '✋', pull: 0.65, inflate: 0.62, inflateMax: 0.6 },
  upperArm: { label: '上臂', icon: '💪', pull: 0.42, inflate: 0.36, inflateMax: 0.65 },
  forearm:  { label: '前臂', icon: '🦾', pull: 0.36, inflate: 0.3, inflateMax: 0.6 },
  shoulder: { label: '肩膀', icon: '🤷', pull: 0.45, inflate: 0.4, inflateMax: 0.55 },
  torso:    { label: '身体', icon: '👕', pull: 0.55, inflate: 0.6, inflateMax: 0.5 },
  neck:     { label: '脖子', icon: '🦒', pull: 0.4, inflate: 0.3, inflateMax: 0.5 },
  leg:      { label: '腿', icon: '🦵', pull: 0.45, inflate: 0.4, inflateMax: 0.55 },
  screen:   { label: '画面', icon: '✨', pull: 0.9, inflate: 0.8, inflateMax: 0.6 },
};
const SIDE_LABEL = { L: '左', R: '右' };
export function partLabel(t) {
  if (!t) return '';
  if (t.part === 'finger') return (t.side ? SIDE_LABEL[t.side] + '手' : '') + FINGER_NAMES[t.finger];
  if (t.part === 'palm') return (t.side ? SIDE_LABEL[t.side] : '') + '手掌';
  const short = { eye: '眼', brow: '眉', ear: '耳' };
  if (t.side && short[t.part]) return SIDE_LABEL[t.side] + short[t.part];
  const base = PARTS[t.part].label;
  return (t.side && t.part !== 'torso' ? SIDE_LABEL[t.side] : '') + base;
}

// ---------------------------------------------------------------- geometry
const hyp = Math.hypot;
function centroid(lm, idx, stride = 3) {
  let x = 0, y = 0;
  for (const i of idx) { x += lm[i * stride]; y += lm[i * stride + 1]; }
  return [x / idx.length, y / idx.length];
}
function segDist(px, py, ax, ay, bx, by) {
  const vx = bx - ax, vy = by - ay, L2 = vx * vx + vy * vy || 1e-9;
  let t = ((px - ax) * vx + (py - ay) * vy) / L2;
  const tc = Math.max(0, Math.min(1, t));
  return { d: hyp(px - ax - tc * vx, py - ay - tc * vy), t, n: ((px - ax) * -vy + (py - ay) * vx) / Math.sqrt(L2) };
}
export function pointInPoly(x, y, pts) {
  let inside = false;
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
    const [xi, yi] = pts[i], [xj, yj] = pts[j];
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}
function polyDist(x, y, pts) {
  let best = Infinity;
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
    best = Math.min(best, segDist(x, y, pts[j][0], pts[j][1], pts[i][0], pts[i][1]).d);
  }
  return best;
}
export function polyPts(lm, idx, stride = 3) { return idx.map((i) => [lm[i * stride], lm[i * stride + 1]]); }
export function convexHull(points) {
  const p = points.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower = [], upper = [];
  for (const q of p) { while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], q) <= 0) lower.pop(); lower.push(q); }
  for (let i = p.length - 1; i >= 0; i--) { const q = p[i]; while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], q) <= 0) upper.pop(); upper.push(q); }
  upper.pop(); lower.pop();
  return lower.concat(upper);
}

// ------------------------------------------------------------- body frames
// Per-frame summary of everything anchors may refer to.
export function buildFrames(det) {
  const F = { face: null, hands: {}, pose: null };
  if (det.face) {
    const lm = det.face.lm;
    const le = centroid(lm, FACE_IDX.leftEye), re = centroid(lm, FACE_IDX.rightEye);
    const iod = hyp(le[0] - re[0], le[1] - re[1]) || 1;
    const roll = Math.atan2(le[1] - re[1], le[0] - re[0]);
    F.face = { lm, le, re, iod, roll, mid: [(le[0] + re[0]) / 2, (le[1] + re[1]) / 2] };
  }
  for (const h of det.hands || []) {
    if (!h.side || F.hands[h.side]) continue;
    const lm = h.lm;
    const size = hyp(lm[0] - lm[27], lm[1] - lm[28], (lm[2] - lm[29]) * 0.6) || 1;
    F.hands[h.side] = { lm, size, raw: h.raw };
  }
  if (det.pose) {
    const lm = det.pose.lm;
    const P = (i) => [lm[i * 4], lm[i * 4 + 1]];
    const sw = hyp(P(11)[0] - P(12)[0], P(11)[1] - P(12)[1]);
    const ms = [(P(11)[0] + P(12)[0]) / 2, (P(11)[1] + P(12)[1]) / 2], mh = [(P(23)[0] + P(24)[0]) / 2, (P(23)[1] + P(24)[1]) / 2];
    const torso = hyp(ms[0] - mh[0], ms[1] - mh[1]);
    F.pose = { lm, unit: Math.max(sw, torso * 0.55, 1), vis: (i) => lm[i * 4 + 3] };
  }
  return F;
}

function posePt(lm, e) {
  if (Array.isArray(e)) return [(lm[e[0] * 4] + lm[e[1] * 4]) / 2, (lm[e[0] * 4 + 1] + lm[e[1] * 4 + 1]) / 2];
  return [lm[e * 4], lm[e * 4 + 1]];
}
function poseVis(lm, e) {
  if (Array.isArray(e)) return Math.min(lm[e[0] * 4 + 3], lm[e[1] * 4 + 3]);
  return lm[e * 4 + 3];
}

// Resolve an anchor into {x, y, rot, s} in raw camera pixels (or null).
export function resolveAnchor(a, F) {
  if (a.type === 'face') {
    const f = F.face; if (!f) return null;
    const [bx, by] = Array.isArray(a.idx) ? centroid(f.lm, a.idx) : [f.lm[a.idx * 3], f.lm[a.idx * 3 + 1]];
    const c = Math.cos(f.roll), s = Math.sin(f.roll);
    return { x: bx + (c * a.ox - s * a.oy) * f.iod, y: by + (s * a.ox + c * a.oy) * f.iod, rot: f.roll, s: f.iod };
  }
  if (a.type === 'hand') {
    const h = F.hands[a.side]; if (!h) return null;
    const lm = h.lm;
    const ax = lm[a.a * 3], ay = lm[a.a * 3 + 1], vx = lm[a.b * 3] - ax, vy = lm[a.b * 3 + 1] - ay;
    const L = hyp(vx, vy) || 1;
    return { x: ax + a.t * vx - (vy / L) * a.n * h.size, y: ay + a.t * vy + (vx / L) * a.n * h.size, rot: Math.atan2(vy, vx), s: h.size };
  }
  if (a.type === 'pose') {
    const p = F.pose; if (!p) return null;
    if (poseVis(p.lm, a.a) < 0.3 || poseVis(p.lm, a.b) < 0.3) return null;
    const A = posePt(p.lm, a.a), B = posePt(p.lm, a.b);
    const vx = B[0] - A[0], vy = B[1] - A[1], L = hyp(vx, vy) || 1;
    return { x: A[0] + a.t * vx - (vy / L) * a.n * p.unit, y: A[1] + a.t * vy + (vx / L) * a.n * p.unit, rot: Math.atan2(vy, vx), s: p.unit };
  }
  if (a.type === 'screen') return { x: a.x, y: a.y, rot: 0, s: a.s };
  return null;
}

// --------------------------------------------------------------- targeting
// Find what body part lies under raw point (x, y). `exclude` = hand side doing
// the grabbing (its own fingers are never the target).
export function classify(x, y, F, exclude = null, frameSize = null) {
  // 1) another hand
  let best = null;
  const ex = Array.isArray(exclude) ? exclude : [exclude];
  for (const side of ['L', 'R']) {
    if (ex.includes(side)) continue;
    const h = F.hands[side]; if (!h) continue;
    const lm = h.lm;
    const palm = [0, 1, 5, 9, 13, 17].map((i) => [lm[i * 3], lm[i * 3 + 1]]);
    for (const [a, b, finger] of HAND_BONES) {
      const r = segDist(x, y, lm[a * 3], lm[a * 3 + 1], lm[b * 3], lm[b * 3 + 1]);
      const tipExtra = (b % 4 === 0 && b > 0 && r.t > 1) ? 0.12 * h.size : 0;
      const lim = (finger === 5 ? 0.16 : 0.15) * h.size + tipExtra;
      if (r.d < lim && (!best || r.d / lim < best.score)) best = { score: r.d / lim, side, a, b, finger, t: r.t, n: r.n / h.size, s: h.size };
    }
    if (pointInPoly(x, y, palm) && (!best || best.score > 0.5)) {
      const r = segDist(x, y, lm[0], lm[1], lm[27], lm[28]);
      best = { score: 0.5, side, a: 0, b: 9, finger: 5, t: r.t, n: r.n / h.size, s: h.size };
    }
  }
  if (best) {
    const part = best.finger === 5 ? 'palm' : 'finger';
    return { part, side: best.side, finger: best.finger, group: best.side === 'L' ? 1 : 2,
      anchor: { type: 'hand', side: best.side, a: best.a, b: best.b, t: best.t, n: best.n }, s: best.s };
  }
  // 2) face / head
  const f = F.face;
  if (f) {
    const r = classifyFace(x, y, f);
    if (r) return r;
  }
  // 3) body
  const p = F.pose;
  if (p) {
    let bb = null;
    for (const bone of POSE_BONES) {
      if (poseVis(p.lm, bone.a) < 0.4 || poseVis(p.lm, bone.b) < 0.4) continue;
      const A = posePt(p.lm, bone.a), B = posePt(p.lm, bone.b);
      const r = segDist(x, y, A[0], A[1], B[0], B[1]);
      const lim = bone.w * p.unit;
      if (r.t > -0.15 && r.t < 1.15 && r.d < lim && (!bb || r.d / lim < bb.score)) bb = { score: r.d / lim, bone, r, L: hyp(B[0] - A[0], B[1] - A[1]) };
    }
    if (bb) {
      return { part: bb.bone.part, side: bb.bone.side, group: 0, s: p.unit,
        anchor: { type: 'pose', a: bb.bone.a, b: bb.bone.b, t: bb.r.t, n: bb.r.n / p.unit } };
    }
  }
  // 4) nothing recognised: pin to the screen
  const s = frameSize ? Math.min(frameSize[0], frameSize[1]) * 0.12 : 80;
  return { part: 'screen', group: 0, s, anchor: { type: 'screen', x, y, s } };
}

function faceLocal(f, x, y) {
  const dx = x - f.mid[0], dy = y - f.mid[1], c = Math.cos(-f.roll), s = Math.sin(-f.roll);
  return [(dx * c - dy * s) / f.iod, (dx * s + dy * c) / f.iod];
}
function nearestFaceIdx(f, x, y) {
  let best = 0, bd = Infinity;
  const lm = f.lm;
  for (let i = 0; i < 468; i++) {
    const d = (lm[i * 3] - x) ** 2 + (lm[i * 3 + 1] - y) ** 2;
    if (d < bd) { bd = d; best = i; }
  }
  return best;
}
function faceAnchorAt(f, x, y, idx = null) {
  const lm = f.lm;
  const base = idx == null ? nearestFaceIdx(f, x, y) : idx;
  const [bx, by] = Array.isArray(base) ? centroid(lm, base) : [lm[base * 3], lm[base * 3 + 1]];
  const c = Math.cos(-f.roll), s = Math.sin(-f.roll);
  const dx = (x - bx) / f.iod, dy = (y - by) / f.iod;
  return { type: 'face', idx: base, ox: dx * c - dy * s, oy: dx * s + dy * c };
}

function classifyFace(x, y, f) {
  const lm = f.lm, iod = f.iod;
  const P = (i) => [lm[i * 3], lm[i * 3 + 1]];
  const mk = (part, extra = {}) => {
    const featureIdx = extra.featureIdx || null;
    return { part, group: 0, s: iod, anchor: faceAnchorAt(f, x, y), featureIdx, ...extra };
  };
  // eyes
  for (const [side, idx] of [['L', FACE_IDX.leftEye], ['R', FACE_IDX.rightEye]]) {
    const c = centroid(lm, idx);
    const poly = polyPts(lm, idx);
    if (hyp(x - c[0], y - c[1]) < 0.3 * iod || (pointInPoly(x, y, poly) || polyDist(x, y, poly) < 0.1 * iod))
      return mk('eye', { side, featureIdx: idx });
  }
  // mouth
  const lips = polyPts(lm, FACE_IDX.lipsOuter);
  if (pointInPoly(x, y, lips) || polyDist(x, y, lips) < 0.13 * iod) return mk('mouth', { featureIdx: FACE_IDX.lipsOuter });
  // nose
  const bridge = polyPts(lm, FACE_IDX.noseBridge);
  let nd = Infinity;
  for (let i = 1; i < bridge.length; i++) nd = Math.min(nd, segDist(x, y, bridge[i - 1][0], bridge[i - 1][1], bridge[i][0], bridge[i][1]).d);
  const noseHull = convexHull(polyPts(lm, FACE_IDX.nose));
  if (nd < 0.16 * iod || pointInPoly(x, y, noseHull)) return mk('nose', { featureIdx: FACE_IDX.noseBall });
  // brows
  for (const [side, idx] of [['L', FACE_IDX.leftBrow], ['R', FACE_IDX.rightBrow]]) {
    const poly = polyPts(lm, idx);
    if (pointInPoly(x, y, poly) || polyDist(x, y, poly) < 0.09 * iod) return mk('brow', { side, featureIdx: idx });
  }
  const oval = polyPts(lm, FACE_IDX.oval);
  const [lx, ly] = faceLocal(f, x, y);
  const lyOf = (i) => faceLocal(f, ...P(i))[1];
  const mouthY = (lyOf(13) + lyOf(14)) / 2, chinY = lyOf(152), browY = (lyOf(105) + lyOf(334)) / 2;
  const inOval = pointInPoly(x, y, oval);
  const edge = inOval ? 0 : polyDist(x, y, oval) / iod;
  if (inOval || edge < 0.35) {
    if (ly > (mouthY + chinY) / 2 - 0.05 && Math.abs(lx) < 0.75) return mk('chin', { featureIdx: FACE_IDX.chin });
    if (ly < browY - 0.08) return mk(edge > 0.05 ? 'head' : 'forehead', { featureIdx: edge > 0.05 ? null : FACE_IDX.forehead });
    if (!inOval && ly > -0.25 && ly < 0.75 && Math.abs(lx) > 0.9) return mk('ear', { side: lx > 0 ? 'L' : 'R' });
    if (Math.abs(lx) > 0.22) return mk('cheek', { side: lx > 0 ? 'L' : 'R', featureIdx: lx > 0 ? FACE_IDX.leftCheek : FACE_IDX.rightCheek });
    return mk('face');
  }
  // around the head (hair, top of head): ellipse centred above the eyes
  const hx = lx / 1.75, hy = (ly + 0.35) / 2.05;
  if (hx * hx + hy * hy < 1 && ly < mouthY) return mk('head');
  return null;
}

// Feature centre used when inflating a feature (snaps to the eye / nose …)
export function featureAnchor(target, F) {
  if (!target.featureIdx || !F.face) return null;
  return { type: 'face', idx: target.featureIdx, ox: 0, oy: 0 };
}
export function headAnchor(F) {
  if (!F.face) return null;
  // between the eyes, lifted a little: inflating it gives a "big head"
  return { type: 'face', idx: [168, 6, 9, 151], ox: 0, oy: -0.1 };
}

// Outline (raw pixel polygon) of a target, used for highlighting.
export function targetOutline(t, F) {
  if (!t) return null;
  const f = F.face;
  switch (t.part) {
    case 'eye': return f && polyPts(f.lm, t.side === 'L' ? FACE_IDX.leftEye : FACE_IDX.rightEye);
    case 'brow': return f && polyPts(f.lm, t.side === 'L' ? FACE_IDX.leftBrow : FACE_IDX.rightBrow);
    case 'mouth': return f && polyPts(f.lm, FACE_IDX.lipsOuter);
    case 'nose': return f && convexHull(polyPts(f.lm, FACE_IDX.nose));
    case 'chin': case 'cheek': case 'forehead': case 'face': case 'head': case 'ear':
      return f && polyPts(f.lm, FACE_IDX.oval);
    default: return null;
  }
}

// ------------------------------------------------------------ face keeper
// When a hand covers the face the face mesh can drop out for a moment. The
// body model still sees the head, so we move the last good face mesh along with
// the body's head keypoints – deformations stay glued to the face.
const HEAD_PTS = [0, 2, 5, 7, 8, 9, 10];
export class FaceKeeper {
  constructor() { this.last = null; this.virt = null; }
  reset() { this.last = null; }
  update(det, t) {
    const p = det.pose && det.pose.lm;
    if (det.face) {
      if (p) {
        this.last = { lm: Float32Array.from(det.face.lm), pts: HEAD_PTS.map((i) => [p[i * 4], p[i * 4 + 1], p[i * 4 + 3]]), t };
      }
      return det.face;
    }
    if (!p || !this.last || t - this.last.t > 3000) return null;
    const src = [], dst = [];
    HEAD_PTS.forEach((i, k) => {
      const s = this.last.pts[k];
      if (s[2] > 0.5 && p[i * 4 + 3] > 0.5) { src.push(s); dst.push([p[i * 4], p[i * 4 + 1]]); }
    });
    if (src.length < 3) return null;
    let msx = 0, msy = 0, mdx = 0, mdy = 0;
    for (let k = 0; k < src.length; k++) { msx += src[k][0]; msy += src[k][1]; mdx += dst[k][0]; mdy += dst[k][1]; }
    msx /= src.length; msy /= src.length; mdx /= src.length; mdy /= src.length;
    let a = 0, b = 0, ss = 0;
    for (let k = 0; k < src.length; k++) {
      const xs = src[k][0] - msx, ys = src[k][1] - msy, xd = dst[k][0] - mdx, yd = dst[k][1] - mdy;
      a += xs * xd + ys * yd; b += xs * yd - ys * xd; ss += xs * xs + ys * ys;
    }
    if (ss < 1e-3) return null;
    const th = Math.atan2(b, a), sc = Math.hypot(a, b) / ss, c = Math.cos(th) * sc, s = Math.sin(th) * sc;
    const lm = (this.virt && this.virt.length === this.last.lm.length) ? this.virt : new Float32Array(this.last.lm.length);
    const L = this.last.lm;
    for (let i = 0; i < L.length; i += 3) {
      const x = L[i] - msx, y = L[i + 1] - msy;
      lm[i] = mdx + c * x - s * y; lm[i + 1] = mdy + s * x + c * y; lm[i + 2] = L[i + 2] * sc;
    }
    this.virt = lm;
    return { lm, virtual: true };
  }
}
