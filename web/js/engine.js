// Tensor engines. Every engine runs the same ONNX networks:
//   ServerEngine – local OpenVINO service, Intel NPU first (GPU/CPU fallback)
//   OrtEngine    – in-browser ONNX Runtime Web: WebNN(NPU) → WebGPU → WASM(CPU)
import { MODELS } from './vision.js';

const FILES = {
  palm: 'palm_detection_full.onnx', palm_lite: 'palm_detection_lite.onnx',
  hand: 'hand_landmark_full.onnx', hand_lite: 'hand_landmark_lite.onnx',
  face_det: 'face_detection_short_range.onnx', face: 'face_landmark.onnx',
  pose_det: 'pose_detection.onnx', pose: 'pose_landmark_full.onnx',
};

function sparseFilter(out, sp) {
  const scores = out[sp.s], boxes = out[sp.b];
  const n = scores.length, coords = boxes.length / n;
  let idx = [];
  for (let i = 0; i < n; i++) if (scores[i] > sp.min) idx.push(i);
  if (idx.length > 64) idx = idx.sort((a, b) => scores[b] - scores[a]).slice(0, 64);
  const I = new Int32Array(idx), S = new Float32Array(idx.length), B = new Float32Array(idx.length * coords);
  idx.forEach((i, k) => { S[k] = scores[i]; B.set(boxes.subarray(i * coords, (i + 1) * coords), k * coords); });
  return { idx: I, scores: S, boxes: B };
}

// ------------------------------------------------------------------ server
export class ServerEngine {
  constructor({ remote = false } = {}) {
    this.kind = 'server';
    this.remote = remote;
    this.seq = 0;
    this.pending = new Map();
    this.info = null;
    this.onInfo = null;
    this.onClose = null;
    this.ws = null;
  }
  get label() { return this.remote ? '电脑 NPU 服务（局域网）' : '本机 NPU 服务 (OpenVINO)'; }
  connect(timeoutMs = 4000) {
    return new Promise((resolve, reject) => {
      const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
      const ws = new WebSocket(`${proto}//${location.host}/ws`);
      ws.binaryType = 'arraybuffer';
      const timer = setTimeout(() => { reject(new Error('timeout')); try { ws.close(); } catch {} }, timeoutMs);
      ws.onmessage = (ev) => {
        if (typeof ev.data === 'string') {
          const m = JSON.parse(ev.data);
          if (m.t === 'info') {
            this.info = m.info;
            this.onInfo && this.onInfo(m.info);
            clearTimeout(timer);
            resolve(this);
          }
          return;
        }
        this._onBinary(ev.data);
      };
      ws.onerror = () => { clearTimeout(timer); reject(new Error('websocket error')); };
      ws.onclose = () => {
        for (const [, p] of this.pending) p.reject(new Error('closed'));
        this.pending.clear();
        this.onClose && this.onClose();
      };
      this.ws = ws;
    });
  }
  setPolicy(v) { this.ws && this.ws.readyState === 1 && this.ws.send(JSON.stringify({ t: 'policy', v })); }
  requestInfo() { this.ws && this.ws.readyState === 1 && this.ws.send(JSON.stringify({ t: 'info' })); }
  get ready() { return this.ws && this.ws.readyState === 1 && this.info && (this.info.status === 'ready' || Object.values(this.info.models || {}).some((m) => m.device)); }
  devices() {
    const d = {};
    for (const [k, m] of Object.entries((this.info && this.info.models) || {})) d[k] = m.device;
    return d;
  }

  async run(jobs) {
    if (!this.ws || this.ws.readyState !== 1) throw new Error('NPU service not connected');
    const parts = [], hjobs = [];
    let off = 0;
    for (const j of jobs) {
      let bytes;
      if (this.remote) {
        if (j.input.prepare) j.input.prepare(); // applies the replicate-border fix to the canvas
        const blob = await new Promise((res) => j.input.canvas.toBlob(res, 'image/jpeg', 0.9));
        bytes = new Uint8Array(await blob.arrayBuffer());
      } else bytes = j.input.rgb();
      hjobs.push({ m: j.model, off, len: bytes.length, fmt: this.remote ? 'jpeg' : 'rgb', ...(j.sparse ? { sparse: j.sparse } : {}) });
      parts.push(bytes);
      off += bytes.length;
    }
    const id = ++this.seq;
    let hj = new TextEncoder().encode(JSON.stringify({ t: 'infer', id, jobs: hjobs }));
    const pad = (4 - ((4 + hj.length) % 4)) % 4;
    const msg = new Uint8Array(4 + hj.length + pad + off);
    new DataView(msg.buffer).setUint32(0, hj.length + pad, true);
    msg.set(hj, 4);
    msg.fill(32, 4 + hj.length, 4 + hj.length + pad);
    let p = 4 + hj.length + pad;
    for (const b of parts) { msg.set(b, p); p += b.length; }
    const done = new Promise((resolve, reject) => {
      const timer = setTimeout(() => { this.pending.delete(id); reject(new Error('inference timeout')); }, 5000);
      this.pending.set(id, { resolve: (v) => { clearTimeout(timer); resolve(v); }, reject: (e) => { clearTimeout(timer); reject(e); } });
    });
    this.ws.send(msg);
    return done;
  }

  _onBinary(buf) {
    const dv = new DataView(buf);
    const hl = dv.getUint32(0, true);
    const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, hl)));
    const base = 4 + hl;
    const results = (header.jobs || []).map((j) => {
      if (j.err) return { error: j.err };
      const out = {};
      for (const o of j.outs) {
        out[o.n] = o.t === 'i32' ? new Int32Array(buf, base + o.off, o.len / 4) : new Float32Array(buf, base + o.off, o.len / 4);
      }
      if ('idx' in out) return { idx: out.idx, scores: out.scores, boxes: out.boxes, ms: j.ms, device: j.dev };
      return { out, ms: j.ms, device: j.dev };
    });
    const p = this.pending.get(header.id);
    if (p) { this.pending.delete(header.id); header.err && !results.length ? p.reject(new Error(header.err)) : p.resolve(results); }
    else if (header.id == null && header.err) { for (const [, q] of this.pending) q.reject(new Error(header.err)); this.pending.clear(); }
  }
  close() { try { this.ws && this.ws.close(); } catch {} }
}

// --------------------------------------------------------------------- ORT
const EP_LABEL = { 'webnn-npu': 'NPU', 'webnn-gpu': 'GPU', webgpu: 'GPU', wasm: 'CPU' };

export class OrtEngine {
  constructor() {
    this.kind = 'browser';
    this.sessions = {};
    this.bufs = {};
    this.ort = null;
    this.eps = {};
    this.lite = false;
  }
  get label() { return '浏览器内置引擎 (ONNX Runtime Web)'; }
  get ready() { return Object.keys(this.sessions).length === 6; }
  devices() {
    const d = {};
    for (const [k, s] of Object.entries(this.sessions)) d[k] = s.device;
    return d;
  }

  async init(onProgress = () => {}, { forceLite = null } = {}) {
    onProgress('加载 ONNX Runtime Web …');
    const ort = await import('../vendor/ort/ort.min.mjs');
    this.ort = ort;
    ort.env.wasm.wasmPaths = new URL('../vendor/ort/', import.meta.url).href;
    ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;
    ort.env.logLevel = 'error';
    const candidates = [];
    if ('ml' in navigator) candidates.push('webnn-npu');
    if ('gpu' in navigator) {
      try {
        const ad = await Promise.race([navigator.gpu.requestAdapter(), new Promise((r) => setTimeout(() => r(null), 1500))]);
        const fallback = ad && (ad.isFallbackAdapter || (ad.info && ad.info.isFallbackAdapter));
        if (ad && !fallback) candidates.push('webgpu');
      } catch {}
    }
    candidates.push('wasm');
    const accel = candidates.some((c) => c !== 'wasm');
    this.lite = forceLite ?? !accel;
    const keys = ['palm', 'hand', 'face_det', 'face', 'pose_det', 'pose'];
    for (const key of keys) {
      const file = this.lite && (key === 'palm' || key === 'hand') ? FILES[key + '_lite'] : FILES[key];
      const url = new URL(`../models/${file}`, import.meta.url).href;
      const bytes = new Uint8Array(await (await fetch(url)).arrayBuffer());
      let ok = false;
      for (const ep of candidates) {
        onProgress(`${MODELS[key].label}: 尝试 ${ep} …`);
        try {
          const opt = { graphOptimizationLevel: 'all' };
          if (ep === 'webnn-npu') opt.executionProviders = [{ name: 'webnn', deviceType: 'npu', powerPreference: 'low-power' }];
          else opt.executionProviders = [ep];
          const withTimeout = (pr, ms) => Promise.race([pr, new Promise((_, rej) => setTimeout(() => rej(new Error('timeout')), ms))]);
          const session = await withTimeout(ort.InferenceSession.create(bytes, opt), 20000);
          // quick sanity run (some EPs fail lazily on first run); a very slow
          // accelerator (e.g. software WebGPU) is treated as unavailable
          const S = MODELS[key].size;
          const feeds = { [session.inputNames[0]]: new ort.Tensor('float32', new Float32Array(S * S * 3), [1, S, S, 3]) };
          await withTimeout(session.run(feeds), ep === 'wasm' ? 30000 : 8000);
          if (ep !== 'wasm') {
            const t1 = performance.now();
            await withTimeout(session.run(feeds), 4000);
            if (performance.now() - t1 > 400) throw new Error('too slow');
          }
          this.sessions[key] = { session, input: session.inputNames[0], device: EP_LABEL[ep], ep, file };
          ok = true;
          break;
        } catch (e) {
          console.warn(`[ort] ${key} on ${ep} failed:`, e && e.message);
        }
      }
      if (!ok) throw new Error(`无法加载模型 ${file}`);
    }
    onProgress('浏览器引擎就绪');
    return this;
  }

  async run(jobs) {
    const ort = this.ort, results = [];
    for (const j of jobs) {
      const s = this.sessions[j.model];
      const S = MODELS[j.model].size, rgb = j.input.rgb();
      const f = (this.bufs[j.model] ||= new Float32Array(S * S * 3));
      if (MODELS[j.model].norm === '01') for (let i = 0; i < f.length; i++) f[i] = rgb[i] * (1 / 255);
      else for (let i = 0; i < f.length; i++) f[i] = rgb[i] * (1 / 127.5) - 1;
      const t0 = performance.now();
      try {
        const om = await s.session.run({ [s.input]: new ort.Tensor('float32', f, [1, S, S, 3]) });
        const out = {};
        for (const k of Object.keys(om)) out[k] = om[k].data;
        const ms = performance.now() - t0;
        if (j.sparse) results.push({ ...sparseFilter(out, j.sparse), ms, device: s.device });
        else results.push({ out, ms, device: s.device });
      } catch (e) {
        results.push({ error: String(e && e.message) });
      }
    }
    return results;
  }
  close() {}
}
