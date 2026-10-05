// Vision pipeline: MediaPipe-compatible hand / face / pose tracking that runs on
// any tensor engine (OpenVINO NPU server, WebNN, WebGPU, WASM).
// Detectors run only when something is not tracked yet; tracked objects get a
// new region-of-interest from their own landmarks every frame (like MediaPipe).
import { PointSetFilter, OneEuro } from './filters.js';

export const MODELS = {
  palm:     { size: 192, norm: '01',  label: '手掌检测' },
  hand:     { size: 224, norm: '01',  label: '手部关键点' },
  face_det: { size: 128, norm: '-11', label: '人脸检测' },
  face:     { size: 192, norm: '01',  label: '脸部网格' },
  pose_det: { size: 224, norm: '-11', label: '人体检测' },
  pose:     { size: 256, norm: '01',  label: '身体姿态' },
};

const DET = {
  palm:     { strides: [8, 16, 16, 16],     numKp: 7, coords: 18, minScore: 0.5, nms: 0.3, box: 'Identity',   score: 'Identity_1' },
  face_det: { strides: [8, 16, 16, 16],     numKp: 6, coords: 16, minScore: 0.5, nms: 0.3, box: 'regressors', score: 'classificators' },
  pose_det: { strides: [8, 16, 32, 32, 32], numKp: 4, coords: 12, minScore: 0.5, nms: 0.3, box: 'Identity',   score: 'Identity_1' },
};

export const OUT = {
  hand: { lm: 'Identity', presence: 'Identity_1', handed: 'Identity_2' },
  face: { lm: 'conv2d_21', flag: 'conv2d_31' },
  pose: { lm: 'Identity', flag: 'Identity_1' },
};

// --------------------------------------------------------------- geometry
const sigmoid = (x) => 1 / (1 + Math.exp(-Math.max(-80, Math.min(80, x))));
export function normRad(a) { return a - 2 * Math.PI * Math.floor((a + Math.PI) / (2 * Math.PI)); }
function rotationFrom(x0, y0, x1, y1, targetDeg) {
  return normRad((targetDeg * Math.PI) / 180 - Math.atan2(-(y1 - y0), x1 - x0));
}
// MediaPipe RectTransformationCalculator (pixel units)
function transformRect(r, { scale = 1, shiftX = 0, shiftY = 0, squareLong = true }) {
  let { cx, cy, w, h } = r;
  const c = Math.cos(r.rot), s = Math.sin(r.rot);
  cx += w * shiftX * c - h * shiftY * s;
  cy += w * shiftX * s + h * shiftY * c;
  if (squareLong) { const L = Math.max(w, h); w = L; h = L; }
  return { cx, cy, w: w * scale, h: h * scale, rot: r.rot };
}
// crop-space (pixels of a size×size tensor) -> image pixels
export function project(roi, size, x, y) {
  const u = x / size - 0.5, v = y / size - 0.5;
  const c = Math.cos(roi.rot), s = Math.sin(roi.rot);
  return [roi.cx + c * u * roi.w - s * v * roi.h, roi.cy + s * u * roi.w + c * v * roi.h];
}
function roiCorners(r) {
  const c = Math.cos(r.rot), s = Math.sin(r.rot), pts = [];
  for (const [u, v] of [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])
    pts.push([r.cx + c * u * r.w - s * v * r.h, r.cy + s * u * r.w + c * v * r.h]);
  return pts;
}
function aabb(pts) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const [x, y] of pts) { if (x < x0) x0 = x; if (y < y0) y0 = y; if (x > x1) x1 = x; if (y > y1) y1 = y; }
  return { x0, y0, x1, y1 };
}
function iou(a, b) {
  const ix = Math.max(0, Math.min(a.x1, b.x1) - Math.max(a.x0, b.x0));
  const iy = Math.max(0, Math.min(a.y1, b.y1) - Math.max(a.y0, b.y0));
  const inter = ix * iy;
  const ua = (a.x1 - a.x0) * (a.y1 - a.y0) + (b.x1 - b.x0) * (b.y1 - b.y0) - inter;
  return ua > 0 ? inter / ua : 0;
}

// --------------------------------------------------------------- SSD anchors
export function ssdAnchors(inputSize, strides) {
  const anchors = [];
  let layer = 0;
  while (layer < strides.length) {
    let last = layer, repeats = 0;
    while (last < strides.length && strides[last] === strides[layer]) { repeats += 2; last++; }
    const fm = Math.ceil(inputSize / strides[layer]);
    for (let y = 0; y < fm; y++)
      for (let x = 0; x < fm; x++)
        for (let r = 0; r < repeats; r++) anchors.push([(x + 0.5) / fm, (y + 0.5) / fm]);
    layer = last;
  }
  return anchors;
}
const ANCHORS = {};
function anchorsFor(model) {
  if (!ANCHORS[model]) ANCHORS[model] = ssdAnchors(MODELS[model].size, DET[model].strides);
  return ANCHORS[model];
}

function weightedNMS(dets, thresh) {
  dets.sort((a, b) => b.score - a.score);
  const out = [];
  let remaining = dets;
  while (remaining.length) {
    const top = remaining[0];
    const cluster = [], rest = [];
    for (const d of remaining) (iou(top.box, d.box) > thresh ? cluster : rest).push(d);
    remaining = rest;
    if (cluster.length === 1) { out.push(top); continue; }
    let ws = 0;
    const box = { x0: 0, y0: 0, x1: 0, y1: 0 };
    const kps = top.kps.map(() => [0, 0]);
    for (const d of cluster) {
      const w = d.score; ws += w;
      box.x0 += d.box.x0 * w; box.y0 += d.box.y0 * w; box.x1 += d.box.x1 * w; box.y1 += d.box.y1 * w;
      d.kps.forEach((k, i) => { kps[i][0] += k[0] * w; kps[i][1] += k[1] * w; });
    }
    box.x0 /= ws; box.y0 /= ws; box.x1 /= ws; box.y1 /= ws;
    kps.forEach((k) => { k[0] /= ws; k[1] /= ws; });
    out.push({ score: top.score, box, kps });
  }
  return out;
}

// Decode sparse SSD output {idx, scores(logits), boxes} into image-space detections
function decodeDetections(model, res, lb) {
  const det = DET[model], S = MODELS[model].size, anchors = anchorsFor(model);
  const dets = [];
  for (let n = 0; n < res.idx.length; n++) {
    const score = sigmoid(res.scores[n]);
    if (score < det.minScore) continue;
    const a = anchors[res.idx[n]];
    const b = res.boxes.subarray(n * det.coords, (n + 1) * det.coords);
    const xc = b[0] / S + a[0], yc = b[1] / S + a[1], w = b[2] / S, h = b[3] / S;
    const toImg = (x, y) => [lb.cx + (x - 0.5) * lb.w, lb.cy + (y - 0.5) * lb.h];
    const p0 = toImg(xc - w / 2, yc - h / 2), p1 = toImg(xc + w / 2, yc + h / 2);
    const kps = [];
    for (let k = 0; k < det.numKp; k++) kps.push(toImg(b[4 + 2 * k] / S + a[0], b[5 + 2 * k] / S + a[1]));
    dets.push({ score, box: { x0: p0[0], y0: p0[1], x1: p1[0], y1: p1[1] }, kps });
  }
  return weightedNMS(dets, det.nms);
}

// --------------------------------------------------------------- ROI rules
function palmToRoi(d) {
  const [k0, k2] = [d.kps[0], d.kps[2]];
  const r = { cx: (d.box.x0 + d.box.x1) / 2, cy: (d.box.y0 + d.box.y1) / 2, w: d.box.x1 - d.box.x0, h: d.box.y1 - d.box.y0,
    rot: rotationFrom(k0[0], k0[1], k2[0], k2[1], 90) };
  return transformRect(r, { scale: 2.6, shiftY: -0.5 });
}
function faceDetToRoi(d) {
  const [k0, k1] = d.kps;
  const r = { cx: (d.box.x0 + d.box.x1) / 2, cy: (d.box.y0 + d.box.y1) / 2, w: d.box.x1 - d.box.x0, h: d.box.y1 - d.box.y0,
    rot: rotationFrom(k0[0], k0[1], k1[0], k1[1], 0) };
  return transformRect(r, { scale: 1.5 });
}
function alignmentRoi(c, e, scale) {
  const size = 2 * Math.hypot(e[0] - c[0], e[1] - c[1]);
  return transformRect({ cx: c[0], cy: c[1], w: size, h: size, rot: rotationFrom(c[0], c[1], e[0], e[1], 90) }, { scale });
}
const HAND_ROI_IDX = [0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18];
function handRoiFromLandmarks(lm) {
  const P = (i) => [lm[i * 3], lm[i * 3 + 1]];
  const [x0, y0] = P(0);
  let x1 = (P(5)[0] + P(13)[0]) / 2, y1 = (P(5)[1] + P(13)[1]) / 2;
  x1 = (x1 + P(9)[0]) / 2; y1 = (y1 + P(9)[1]) / 2;
  const rot = rotationFrom(x0, y0, x1, y1, 90);
  const pts = HAND_ROI_IDX.map(P);
  const bb = aabb(pts);
  const acx = (bb.x0 + bb.x1) / 2, acy = (bb.y0 + bb.y1) / 2;
  const rev = -rot, c = Math.cos(rev), s = Math.sin(rev);
  let mx0 = Infinity, my0 = Infinity, mx1 = -Infinity, my1 = -Infinity;
  for (const [x, y] of pts) {
    const ox = x - acx, oy = y - acy;
    const px = ox * c - oy * s, py = ox * s + oy * c;
    mx0 = Math.min(mx0, px); mx1 = Math.max(mx1, px); my0 = Math.min(my0, py); my1 = Math.max(my1, py);
  }
  const pcx = (mx0 + mx1) / 2, pcy = (my0 + my1) / 2;
  const cr = Math.cos(rot), sr = Math.sin(rot);
  const r = { cx: pcx * cr - pcy * sr + acx, cy: pcx * sr + pcy * cr + acy, w: mx1 - mx0, h: my1 - my0, rot };
  return transformRect(r, { scale: 2.0, shiftY: -0.1 });
}
function faceRoiFromLandmarks(lm) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (let i = 0; i < 468; i++) {
    const x = lm[i * 3], y = lm[i * 3 + 1];
    if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
  }
  const rot = rotationFrom(lm[33 * 3], lm[33 * 3 + 1], lm[263 * 3], lm[263 * 3 + 1], 0);
  return transformRect({ cx: (x0 + x1) / 2, cy: (y0 + y1) / 2, w: x1 - x0, h: y1 - y0, rot }, { scale: 1.5 });
}

// --------------------------------------------------------------- crops
class CropPool {
  constructor() { this.pool = []; }
  get(i, size) {
    let c = this.pool[i];
    if (!c || c.size !== size) {
      const canvas = document.createElement('canvas');
      canvas.width = canvas.height = size;
      const ctx = canvas.getContext('2d', { willReadFrequently: true, alpha: false });
      c = this.pool[i] = { canvas, ctx, size, rgb: new Uint8Array(size * size * 3) };
    }
    return c;
  }
}
// Render the rotated ROI of `src` into a size×size canvas.
// Detectors use a zero border; landmark models use MediaPipe's default
// BORDER_REPLICATE (matters a lot for the body model whose ROI usually extends
// far below a webcam frame).
export const CROP = { quality: 'auto' };
function cropMatrix(roi, size) {
  const co = Math.cos(roi.rot), si = Math.sin(roi.rot);
  // image = A·out + b ; A = R·diag(w/s, h/s)
  const A = [co * roi.w / size, -si * roi.h / size, si * roi.w / size, co * roi.h / size]; // [a00 a01 a10 a11]
  const b = [roi.cx - (co * roi.w / 2 - si * roi.h / 2), roi.cy - (si * roi.w / 2 + co * roi.h / 2)];
  return { A, b };
}
function renderCrop(c, src, roi, edges) {
  const { ctx, size } = c;
  const co = Math.cos(roi.rot), si = Math.sin(roi.rot);
  const ia = (co * size) / roi.w, ib = (-si * size) / roi.h, ic = (si * size) / roi.w, id = (co * size) / roi.h;
  const bx = roi.cx - (co * roi.w / 2 - si * roi.h / 2), by = roi.cy - (si * roi.w / 2 + co * roi.h / 2);
  const e = -(ia * bx + ic * by), f = -(ib * bx + id * by);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, size, size);
  ctx.imageSmoothingEnabled = true;
  // bilinear like MediaPipe's warpAffine (verified: closest detections to the reference)
  ctx.imageSmoothingQuality = CROP.quality !== 'auto' ? CROP.quality : 'low';
  ctx.setTransform(ia, ib, ic, id, e, f);
  ctx.drawImage(src, 0, 0);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  c.ready = false;
  c.edges = null;
  if (edges) {
    // does the ROI leave the image?
    const { W, H } = edges;
    const { A, b } = cropMatrix(roi, size);
    let out = false;
    for (const [u, v] of [[0, 0], [size, 0], [0, size], [size, size]]) {
      const x = A[0] * u + A[1] * v + b[0], y = A[2] * u + A[3] * v + b[1];
      if (x < 0 || y < 0 || x > W || y > H) { out = true; break; }
    }
    if (out) { c.edges = edges; c.mat = { A, b }; c.needsFix = true; }
  }
}
function fixBorder(c, d) {
  // replace pixels whose source lies outside the frame by the clamped edge pixel
  const { A, b } = c.mat, { W, H, top, bottom, left, right } = c.edges, size = c.size;
  for (let oy = 0; oy < size; oy++) {
    const v = oy + 0.5;
    for (let ox = 0; ox < size; ox++) {
      const u = ox + 0.5;
      const x = A[0] * u + A[1] * v + b[0], y = A[2] * u + A[3] * v + b[1];
      if (x >= 0.5 && y >= 0.5 && x <= W - 0.5 && y <= H - 0.5) continue;
      const cx = Math.min(W - 1, Math.max(0, x | 0)), cy = Math.min(H - 1, Math.max(0, y | 0));
      let s, k;
      if (y < 0.5) { s = top; k = cx * 4; }
      else if (y > H - 0.5) { s = bottom; k = cx * 4; }
      else if (x < 0.5) { s = left; k = cy * 4; }
      else { s = right; k = cy * 4; }
      const o = (oy * size + ox) * 4;
      d[o] = s[k]; d[o + 1] = s[k + 1]; d[o + 2] = s[k + 2];
    }
  }
}
function cropRGB(c) {
  if (c.ready) return c.rgb;
  const img = c.ctx.getImageData(0, 0, c.size, c.size);
  const d = img.data, rgb = c.rgb;
  if (c.needsFix) {
    fixBorder(c, d);
    c.needsFix = false;
    c.ctx.putImageData(img, 0, 0); // keep canvas in sync (JPEG path)
  }
  for (let i = 0, j = 0; i < d.length; i += 4, j += 3) { rgb[j] = d[i]; rgb[j + 1] = d[i + 1]; rgb[j + 2] = d[i + 2]; }
  c.ready = true;
  return rgb;
}
function frameEdges(src, W, H) {
  const ctx = src.getContext('2d', { willReadFrequently: true });
  return {
    W, H,
    top: ctx.getImageData(0, 0, W, 1).data,
    bottom: ctx.getImageData(0, H - 1, W, 1).data,
    left: ctx.getImageData(0, 0, 1, H).data,
    right: ctx.getImageData(W - 1, 0, 1, H).data,
  };
}

// --------------------------------------------------------------- pipeline
let nextTrackId = 1;

export class VisionPipeline {
  constructor(engine) {
    this.engine = engine;
    this.crops = new CropPool();
    this.frame = 0;
    this.hands = [];
    this.face = null;
    this.pose = null;
    this.maxHands = 2;
    this.enable = { hands: true, face: true, pose: true };
    this.handFilterParams = [1.6, 6.0];
    this.faceFilterParams = [1.0, 4.0];
    this.poseFilterParams = [1.0, 3.0];
    this.stats = { models: {}, engineMs: 0, roundtripMs: 0 };
    this.lastT = 0;
  }
  setEngine(engine) { this.engine = engine; }
  reset() { this.hands = []; this.face = null; this.pose = null; }

  async process(src, W, H, t) {
    this.frame++;
    const dt = this.lastT ? Math.min(0.2, (t - this.lastT) / 1000) : 1 / 30;
    this.lastT = t;
    const S = Math.max(W, H);
    const lb = { cx: W / 2, cy: H / 2, w: S, h: S, rot: 0 };
    const jobs = [];
    if (this.enable.hands) for (const h of this.hands) jobs.push({ kind: 'hand', model: 'hand', roi: h.roi, track: h });
    else this.hands = [];
    if (this.enable.face && this.face) jobs.push({ kind: 'face', model: 'face', roi: this.face.roi });
    if (!this.enable.face) this.face = null;
    // slow engines (e.g. WASM on a weak CPU): refresh the body only every 3rd frame
    if (this.enable.pose && this.pose && (!this.slow || this.frame % 3 === 0)) jobs.push({ kind: 'pose', model: 'pose', roi: this.pose.roi });
    if (!this.enable.pose) this.pose = null;
    const nh = this.hands.length;
    if (this.enable.hands && nh < this.maxHands && (nh === 0 || this.frame % (this.slow ? 4 : 2) === 0))
      jobs.push({ kind: 'palm', model: 'palm', roi: lb, sparse: { s: DET.palm.score, b: DET.palm.box, min: -0.3 } });
    if (this.enable.face && !this.face && this.frame % 2 === 1)
      jobs.push({ kind: 'face_det', model: 'face_det', roi: lb, sparse: { s: DET.face_det.score, b: DET.face_det.box, min: -0.3 } });
    if (this.enable.pose && !this.pose && this.frame % 3 === 1)
      jobs.push({ kind: 'pose_det', model: 'pose_det', roi: lb, sparse: { s: DET.pose_det.score, b: DET.pose_det.box, min: -0.3 } });

    let results = [];
    if (jobs.length) {
      let edges = null;
      jobs.forEach((j, i) => {
        const c = this.crops.get(i, MODELS[j.model].size);
        const landmark = !j.sparse;
        if (landmark && !edges && src.getContext) edges = frameEdges(src, W, H);
        renderCrop(c, src, j.roi, landmark ? edges : null);
        j.input = { canvas: c.canvas, size: c.size, rgb: () => cropRGB(c), prepare: () => cropRGB(c) };
      });
      const t0 = performance.now();
      results = await this.engine.run(jobs.map((j) => ({ model: j.model, input: j.input, sparse: j.sparse || null })));
      this.stats.roundtripMs = performance.now() - t0;
      this.rtEma = this.rtEma == null ? this.stats.roundtripMs : this.rtEma * 0.9 + this.stats.roundtripMs * 0.1;
      this.slow = this.rtEma > 70;
    }

    let engineMs = 0;
    const newHands = [];
    let faceDets = null, poseDets = null, palmDets = null;
    jobs.forEach((j, i) => {
      const r = results[i];
      if (!r || r.error) return;
      const st = (this.stats.models[j.model] ||= { ms: 0, device: '', n: 0 });
      st.ms = st.n ? st.ms * 0.85 + r.ms * 0.15 : r.ms; st.n++; st.device = r.device;
      engineMs = Math.max(engineMs, r.ms);
      if (j.kind === 'hand') { const h = this._hand(j.track, r.out, dt); if (h) newHands.push(h); }
      else if (j.kind === 'face') this._face(r.out, dt);
      else if (j.kind === 'pose') this._pose(r.out, dt);
      else if (j.kind === 'palm') palmDets = decodeDetections('palm', r, lb);
      else if (j.kind === 'face_det') faceDets = decodeDetections('face_det', r, lb);
      else if (j.kind === 'pose_det') poseDets = decodeDetections('pose_det', r, lb);
    });
    // tracks whose landmark job failed this frame (engine hiccup) are kept for a moment
    for (const h of this.hands) if (!newHands.includes(h) && !jobs.some((j) => j.track === h && results[jobs.indexOf(j)] && !results[jobs.indexOf(j)].error)) {
      if (++h.miss < 3) newHands.push(h);
    }
    this.hands = this._dedupeHands(newHands);
    if (palmDets) this._addPalms(palmDets, W, H);
    if (faceDets && faceDets.length && !this.face) this._startFace(faceDetToRoi(faceDets[0]));
    if (poseDets && poseDets.length && !this.pose) this._startPose(alignmentRoi(poseDets[0].kps[0], poseDets[0].kps[1], 1.25));
    this._assignSides();
    this.stats.engineMs = engineMs;

    return {
      W, H, t,
      hands: this.hands.filter((h) => h.lm).map((h) => ({ id: h.id, side: h.side, score: h.score, lm: h.slm, raw: h.lm, roi: h.roiUsed })),
      face: this.face && this.face.lm ? { lm: this.face.slm, raw: this.face.lm, roi: this.face.roiUsed, score: this.face.score } : null,
      pose: this.pose && this.pose.lm ? { lm: this.pose.slm, raw: this.pose.lm, roi: this.pose.roiUsed, score: this.pose.score } : null,
      rois: { hands: this.hands.map((h) => h.roi), face: this.face && this.face.roi, pose: this.pose && this.pose.roi },
    };
  }

  // ---- hands
  _hand(track, out, dt) {
    const presence = out[OUT.hand.presence][0];
    if (!(presence >= 0.5)) return null;
    const raw = out[OUT.hand.lm], size = MODELS.hand.size, roi = track.roi;
    const lm = track.lm || new Float32Array(63);
    for (let i = 0; i < 21; i++) {
      const [x, y] = project(roi, size, raw[i * 3], raw[i * 3 + 1]);
      lm[i * 3] = x; lm[i * 3 + 1] = y; lm[i * 3 + 2] = (raw[i * 3 + 2] / size) * roi.w;
    }
    track.lm = lm;
    track.score = presence;
    track.miss = 0;
    const p = out[OUT.hand.handed][0];
    track.handed = track.handed == null ? p : track.handed * 0.85 + p * 0.15;
    track.roiUsed = roi;
    track.roi = handRoiFromLandmarks(lm);
    const scale = Math.hypot(lm[0] - lm[27], lm[1] - lm[28]) + 1;
    track.filter ||= new PointSetFilter(21, 3, ...this.handFilterParams);
    track.slm = track.filter.apply(lm, dt, scale);
    return track;
  }
  _dedupeHands(hands) {
    const keep = [];
    const box = (h) => { const bb = aabb(roiCorners(h.roi)); return bb; };
    for (const h of hands.sort((a, b) => (b.score || 0) - (a.score || 0))) {
      if (keep.some((k) => iou(box(k), box(h)) > 0.45)) continue;
      keep.push(h);
    }
    return keep.slice(0, this.maxHands);
  }
  _addPalms(dets, W, H) {
    for (const d of dets) {
      if (this.hands.length >= this.maxHands) break;
      const roi = palmToRoi(d);
      const bb = aabb(roiCorners(roi));
      const cx = (d.box.x0 + d.box.x1) / 2, cy = (d.box.y0 + d.box.y1) / 2;
      const dup = this.hands.some((h) => {
        if (iou(aabb(roiCorners(h.roi)), bb) > 0.5) return true;
        if (!h.lm) return false;
        // palm centre inside the tight landmark box of an already tracked hand
        const pts = []; for (let i = 0; i < 21; i++) pts.push([h.lm[i * 3], h.lm[i * 3 + 1]]);
        const t = aabb(pts);
        return cx > t.x0 && cx < t.x1 && cy > t.y0 && cy < t.y1;
      });
      if (dup) continue;
      this.hands.push({ id: nextTrackId++, roi, lm: null, slm: null, score: d.score, handed: null, miss: 0, filter: null });
    }
  }
  _assignSides() {
    // Handedness output p: p > 0.5 = the person's RIGHT hand when the crop comes from the
    // raw (non-mirrored) camera frame (verified against BlazePose wrists).
    const hs = this.hands.filter((h) => h.handed != null);
    for (const h of hs) h.side = h.handed > 0.5 ? 'R' : 'L';
    if (hs.length === 2 && hs[0].side === hs[1].side) {
      const [a, b] = Math.abs(hs[0].handed - 0.5) >= Math.abs(hs[1].handed - 0.5) ? hs : [hs[1], hs[0]];
      b.side = a.side === 'L' ? 'R' : 'L';
    }
  }

  // ---- face
  _startFace(roi) { this.face = { roi, lm: null, slm: null, filter: null, score: 0 }; }
  _face(out, dt) {
    const flag = sigmoid(out[OUT.face.flag][0]);
    if (!(flag >= 0.5)) { this.face = null; return; }
    const raw = out[OUT.face.lm], size = MODELS.face.size, roi = this.face.roi;
    const lm = this.face.lm || new Float32Array(468 * 3);
    for (let i = 0; i < 468; i++) {
      const [x, y] = project(roi, size, raw[i * 3], raw[i * 3 + 1]);
      lm[i * 3] = x; lm[i * 3 + 1] = y; lm[i * 3 + 2] = (raw[i * 3 + 2] / size) * roi.w;
    }
    this.face.lm = lm;
    this.face.score = flag;
    this.face.roiUsed = roi;
    this.face.roi = faceRoiFromLandmarks(lm);
    this.face.filter ||= new PointSetFilter(468, 3, ...this.faceFilterParams);
    this.face.slm = this.face.filter.apply(lm, dt, this.face.roi.w / 1.5);
  }

  // ---- pose
  _startPose(roi) { this.pose = { roi, lm: null, slm: null, filter: null, score: 0 }; }
  _pose(out, dt) {
    const flag = out[OUT.pose.flag][0];
    if (!(flag >= 0.5)) { this.pose = null; return; }
    const raw = out[OUT.pose.lm], size = MODELS.pose.size, roi = this.pose.roi;
    const lm = this.pose.lm || new Float32Array(33 * 4);
    for (let i = 0; i < 33; i++) {
      const [x, y] = project(roi, size, raw[i * 5], raw[i * 5 + 1]);
      lm[i * 4] = x; lm[i * 4 + 1] = y; lm[i * 4 + 2] = (raw[i * 5 + 2] / size) * roi.w; lm[i * 4 + 3] = sigmoid(raw[i * 5 + 3]);
    }
    const a0 = project(roi, size, raw[33 * 5], raw[33 * 5 + 1]);
    const a1 = project(roi, size, raw[34 * 5], raw[34 * 5 + 1]);
    // MediaPipe smooths the auxiliary (ROI) landmarks heavily: one-euro(0.01, 10, 1)
    const aux = (this.pose.aux ||= [0, 1, 2, 3].map(() => new OneEuro(0.01, 10.0, 1.0)));
    const auxScale = Math.max(1, (Math.abs(a1[0] - a0[0]) + Math.abs(a1[1] - a0[1])) / 2);
    const s0 = [aux[0].filter(a0[0], dt, auxScale), aux[1].filter(a0[1], dt, auxScale)];
    const s1 = [aux[2].filter(a1[0], dt, auxScale), aux[3].filter(a1[1], dt, auxScale)];
    this.pose.lm = lm;
    this.pose.score = flag;
    this.pose.roiUsed = roi;
    this.pose.roi = alignmentRoi(s0, s1, 1.25);
    this.pose.filter ||= new PointSetFilter(33, 4, ...this.poseFilterParams);
    this.pose.slm = this.pose.filter.apply(lm, dt, this.pose.roi.w / 2.5, 3);
  }
}
