// 2D overlay: pinch cursors, part highlights, labels, skeleton debug view.
import { PARTS, partLabel, FACE_IDX, polyPts } from './parts.js';

const COL = { pull: '#53d8ff', inflate: '#ff6fb5', shrink: '#9b8cff', stretch: '#ffd34f', pending: '#ffffff', hover: '#ffffff' };
const HAND_BONES = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17]];
const POSE_BONES = [[11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [11, 23], [12, 24], [23, 24], [23, 25], [25, 27], [24, 26], [26, 28], [15, 19], [16, 20], [27, 31], [28, 32]];
const FACE_LINES = [FACE_IDX.oval, FACE_IDX.leftEye, FACE_IDX.rightEye, FACE_IDX.lipsOuter, FACE_IDX.lipsInner, FACE_IDX.leftBrow, FACE_IDX.rightBrow];

export class Overlay {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.dpr = 1;
    this.debug = false;
    this.t = 0;
  }
  resize(w, h, dpr) {
    this.dpr = dpr;
    const W = Math.round(w * dpr), H = Math.round(h * dpr);
    if (this.canvas.width !== W || this.canvas.height !== H) { this.canvas.width = W; this.canvas.height = H; }
  }

  draw(view, mirror, det, F, inter, clay, now) {
    const ctx = this.ctx, dpr = this.dpr;
    this.t = now / 1000;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const S = (x, y) => [view.ox + view.scale * (mirror ? view.W - x : x), view.oy + view.scale * y];
    const fwd = (x, y) => clay.forward(x, y);
    if (this.debug && det) this._skeleton(ctx, S, det);
    const snap = inter.snapshot();
    // stretch links
    const drawn = new Set();
    for (const c of snap.cursors) {
      const g = c.grab;
      if (g && g.kind === 'stretch' && !drawn.has(g)) {
        drawn.add(g);
        const [a, b] = g.cursors.map((o) => S(o.x, o.y));
        ctx.save();
        ctx.strokeStyle = COL.stretch; ctx.lineWidth = 3; ctx.setLineDash([10, 8]); ctx.lineDashOffset = -this.t * 40;
        ctx.shadowColor = COL.stretch; ctx.shadowBlur = 12;
        ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
        ctx.restore();
        const it = g.item;
        this._chip(ctx, (a[0] + b[0]) / 2, (a[1] + b[1]) / 2 - 26, `${PARTS[it.part] ? PARTS[it.part].icon : '↔'} ${it.label}  ×${it.su.toFixed(2)}`, COL.stretch, true);
      }
    }
    for (const c of snap.cursors) this._cursor(ctx, S, fwd, c, F, clay);
    if (snap.reset) this._reset(ctx, S, snap.reset);
  }

  _cursor(ctx, S, fwd, c, F, clay) {
    const [x, y] = S(c.x, c.y);
    const g = c.grab;
    const isHand = c.source === 'hand';
    if (isHand && !g && c.closeness <= 0.02) {
      ctx.save(); ctx.globalAlpha = 0.5; ctx.fillStyle = '#fff';
      ctx.beginPath(); ctx.arc(x, y, 3, 0, Math.PI * 2); ctx.fill(); ctx.restore();
      return;
    }
    // highlight target
    const tgt = g ? g.target : c.target;
    if (tgt && (!g || g.kind === 'pending') && (g || tgt.part !== 'screen')) this._highlight(ctx, S, fwd, tgt, F, g ? COL.pending : COL.hover, c);
    ctx.save();
    if (!g) {
      const R = isHand ? 30 - 17 * c.closeness : 13;
      ctx.strokeStyle = 'rgba(255,255,255,0.95)'; ctx.lineWidth = 2.5;
      ctx.shadowColor = 'rgba(0,0,0,0.5)'; ctx.shadowBlur = 6;
      ctx.beginPath(); ctx.arc(x, y, R, 0, Math.PI * 2); ctx.stroke();
      if (tgt && c.closeness > 0.35) this._chip(ctx, x + 24, y - 30, `${PARTS[tgt.part].icon} ${partLabel(tgt)}`, '#fff');
    } else {
      const kind = g.kind === 'inflate' && g.sign < 0 ? 'shrink' : g.kind;
      const col = COL[kind] || '#fff';
      ctx.shadowColor = col; ctx.shadowBlur = 16;
      ctx.fillStyle = col;
      ctx.beginPath(); ctx.arc(x, y, 7, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = col; ctx.lineWidth = 3;
      ctx.beginPath(); ctx.arc(x, y, 17, 0, Math.PI * 2); ctx.stroke();
      if (g.kind === 'pending') {
        ctx.strokeStyle = COL.inflate; ctx.lineWidth = 4;
        ctx.beginPath(); ctx.arc(x, y, 24, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * (g.hold || 0)); ctx.stroke();
        this._chip(ctx, x + 26, y - 34, `${PARTS[g.target.part].icon} ${partLabel(g.target)} · 拖动拉伸 / 停住放大`, '#fff');
      } else if (g.kind === 'inflate') {
        const p = Math.abs(g.item.amount) / (g.max || 0.7);
        ctx.lineWidth = 5;
        ctx.beginPath(); ctx.arc(x, y, 25, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * p); ctx.stroke();
        const pulse = 1 + 0.08 * Math.sin(this.t * 12);
        ctx.globalAlpha = 0.35; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(x, y, 32 * pulse, 0, Math.PI * 2); ctx.stroke();
        ctx.globalAlpha = 1;
        this._chip(ctx, x + 28, y - 36, `${PARTS[g.item.part].icon} ${g.item.label} ${g.sign > 0 ? '放大' : '缩小'} ${Math.round(p * 100)}%`, col, true);
      } else if (g.kind === 'pull') {
        const it = g.item;
        if (it.frame) {
          const p0 = S(it.frame.x, it.frame.y);
          ctx.globalAlpha = 0.6; ctx.setLineDash([4, 6]); ctx.lineWidth = 2;
          ctx.beginPath(); ctx.moveTo(p0[0], p0[1]); ctx.lineTo(x, y); ctx.stroke();
          ctx.setLineDash([]); ctx.globalAlpha = 1;
        }
        this._chip(ctx, x + 26, y - 34, `${PARTS[it.part].icon} ${it.label} 拉扯中`, col, true);
      }
    }
    ctx.restore();
  }

  _highlight(ctx, S, fwd, t, F, col, c) {
    let poly = null;
    const f = F.face;
    if (f && (t.part === 'eye' || t.part === 'brow' || t.part === 'mouth' || t.part === 'nose')) {
      const idx = t.part === 'eye' ? (t.side === 'L' ? FACE_IDX.leftEye : FACE_IDX.rightEye)
        : t.part === 'brow' ? (t.side === 'L' ? FACE_IDX.leftBrow : FACE_IDX.rightBrow)
        : t.part === 'mouth' ? FACE_IDX.lipsOuter : null;
      if (idx) poly = polyPts(f.lm, idx);
      else {
        const nose = polyPts(f.lm, FACE_IDX.nose);
        // convex hull for a clean nose outline
        const p = nose.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
        const cr = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
        const lo = [], up = [];
        for (const q of p) { while (lo.length >= 2 && cr(lo[lo.length - 2], lo[lo.length - 1], q) <= 0) lo.pop(); lo.push(q); }
        for (let i = p.length - 1; i >= 0; i--) { const q = p[i]; while (up.length >= 2 && cr(up[up.length - 2], up[up.length - 1], q) <= 0) up.pop(); up.push(q); }
        up.pop(); lo.pop(); poly = lo.concat(up);
      }
    }
    ctx.save();
    ctx.strokeStyle = col; ctx.lineWidth = 2.5; ctx.shadowColor = col; ctx.shadowBlur = 14;
    ctx.globalAlpha = 0.85;
    if (poly) {
      ctx.beginPath();
      poly.forEach(([px, py], i) => { const [qx, qy] = fwd(px, py); const [sx, sy] = S(qx, qy); i ? ctx.lineTo(sx, sy) : ctx.moveTo(sx, sy); });
      ctx.closePath(); ctx.stroke();
    } else {
      // soft circle showing the area that will move
      const cfg = PARTS[t.part];
      const [ax, ay] = clayRaw(c);
      const rr = (cfg ? cfg.pull : 0.8) * (t.s || 60);
      const [sx, sy] = S(ax, ay);
      const rs = rr * Math.abs(S(1, 0)[0] - S(0, 0)[0]);
      ctx.setLineDash([6, 8]); ctx.lineDashOffset = -this.t * 30;
      ctx.beginPath(); ctx.arc(sx, sy, rs, 0, Math.PI * 2); ctx.stroke();
    }
    ctx.restore();
  }

  _chip(ctx, x, y, text, col, strong = false) {
    ctx.save();
    ctx.shadowBlur = 0;
    ctx.font = '600 14px "PingFang SC","Microsoft YaHei","Noto Sans SC",system-ui,sans-serif';
    const w = ctx.measureText(text).width + 22, h = 28;
    const cw = this.canvas.width / this.dpr;
    if (x + w > cw - 8) x = cw - 8 - w;
    if (x < 8) x = 8;
    if (y < 8) y = 8;
    ctx.fillStyle = strong ? 'rgba(20,16,32,0.82)' : 'rgba(20,16,32,0.66)';
    const r = 14;
    ctx.beginPath();
    ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r); ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r);
    ctx.fill();
    ctx.strokeStyle = col; ctx.globalAlpha = 0.9; ctx.lineWidth = 1.5; ctx.stroke();
    ctx.globalAlpha = 1;
    ctx.fillStyle = '#fff';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, x + 11, y + h / 2 + 1);
    ctx.restore();
  }

  _reset(ctx, S, r) {
    const [x, y] = S(r.x, r.y);
    ctx.save();
    ctx.lineWidth = 8; ctx.lineCap = 'round';
    ctx.strokeStyle = 'rgba(255,255,255,0.25)';
    ctx.beginPath(); ctx.arc(x, y, 46, 0, Math.PI * 2); ctx.stroke();
    ctx.strokeStyle = '#7dffb2'; ctx.shadowColor = '#7dffb2'; ctx.shadowBlur = 18;
    ctx.beginPath(); ctx.arc(x, y, 46, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * r.progress); ctx.stroke();
    ctx.restore();
    this._chip(ctx, x - 92, y + 58, '🖐️🖐️ 保持张开双手 → 全部复原', '#7dffb2', true);
  }

  _skeleton(ctx, S, det) {
    ctx.save();
    ctx.lineCap = 'round';
    if (det.pose) {
      const lm = det.pose.lm;
      ctx.strokeStyle = 'rgba(120,220,255,0.85)'; ctx.lineWidth = 3;
      for (const [a, b] of POSE_BONES) {
        if (lm[a * 4 + 3] < 0.5 || lm[b * 4 + 3] < 0.5) continue;
        const p = S(lm[a * 4], lm[a * 4 + 1]), q = S(lm[b * 4], lm[b * 4 + 1]);
        ctx.beginPath(); ctx.moveTo(p[0], p[1]); ctx.lineTo(q[0], q[1]); ctx.stroke();
      }
      ctx.fillStyle = '#fff';
      for (let i = 0; i < 33; i++) {
        if (lm[i * 4 + 3] < 0.5) continue;
        const p = S(lm[i * 4], lm[i * 4 + 1]);
        ctx.beginPath(); ctx.arc(p[0], p[1], 3.5, 0, Math.PI * 2); ctx.fill();
      }
    }
    for (const h of det.hands) {
      const lm = h.lm;
      ctx.strokeStyle = h.side === 'L' ? 'rgba(255,140,200,0.95)' : 'rgba(140,255,170,0.95)'; ctx.lineWidth = 2.5;
      for (const [a, b] of HAND_BONES) {
        const p = S(lm[a * 3], lm[a * 3 + 1]), q = S(lm[b * 3], lm[b * 3 + 1]);
        ctx.beginPath(); ctx.moveTo(p[0], p[1]); ctx.lineTo(q[0], q[1]); ctx.stroke();
      }
      ctx.fillStyle = '#fff';
      for (let i = 0; i < 21; i++) { const p = S(lm[i * 3], lm[i * 3 + 1]); ctx.beginPath(); ctx.arc(p[0], p[1], 2.5, 0, Math.PI * 2); ctx.fill(); }
    }
    if (det.face) {
      const lm = det.face.lm;
      ctx.strokeStyle = 'rgba(255,255,255,0.7)'; ctx.lineWidth = 1.5;
      for (const line of FACE_LINES) {
        ctx.beginPath();
        line.forEach((i, k) => { const p = S(lm[i * 3], lm[i * 3 + 1]); k ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]); });
        ctx.closePath(); ctx.stroke();
      }
      ctx.fillStyle = 'rgba(255,255,255,0.45)';
      for (let i = 0; i < 468; i += 3) { const p = S(lm[i * 3], lm[i * 3 + 1]); ctx.fillRect(p[0] - 1, p[1] - 1, 2, 2); }
    }
    ctx.restore();
  }
}

function clayRaw(c) { return [c.x, c.y]; }
