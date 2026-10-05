// Tiny synthesized sound effects (no audio files needed).
export class Sfx {
  constructor() {
    this.ctx = null;
    this.enabled = true;
    this.volume = 0.5;
    this.balloon = null;
    this.lastDrag = 0;
    this.lastDragLevel = 0;
  }
  unlock() {
    try {
      if (!this.ctx) {
        this.ctx = new (window.AudioContext || window.webkitAudioContext)();
        this.master = this.ctx.createGain();
        this.master.gain.value = this.volume;
        this.master.connect(this.ctx.destination);
        const len = this.ctx.sampleRate * 0.5;
        this.noise = this.ctx.createBuffer(1, len, this.ctx.sampleRate);
        const d = this.noise.getChannelData(0);
        for (let i = 0; i < len; i++) d[i] = Math.random() * 2 - 1;
      }
      if (this.ctx.state === 'suspended') this.ctx.resume();
    } catch { this.ctx = null; }
  }
  setEnabled(v) { this.enabled = v; if (!v) this.inflateStop(); }
  get ok() { return this.enabled && this.ctx && this.ctx.state === 'running'; }

  _env(node, t, a, peak, d) {
    const g = this.ctx.createGain();
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(peak, t + a);
    g.gain.exponentialRampToValueAtTime(0.0001, t + a + d);
    node.connect(g); g.connect(this.master);
    return g;
  }
  _tone(type, f0, f1, dur, peak = 0.25, attack = 0.005) {
    const t = this.ctx.currentTime;
    const o = this.ctx.createOscillator();
    o.type = type;
    o.frequency.setValueAtTime(f0, t);
    o.frequency.exponentialRampToValueAtTime(Math.max(20, f1), t + dur);
    this._env(o, t, attack, peak, dur);
    o.start(t); o.stop(t + attack + dur + 0.05);
    return o;
  }
  _noise(f0, f1, q, dur, peak = 0.2) {
    const t = this.ctx.currentTime;
    const src = this.ctx.createBufferSource();
    src.buffer = this.noise;
    const bp = this.ctx.createBiquadFilter();
    bp.type = 'bandpass'; bp.Q.value = q;
    bp.frequency.setValueAtTime(f0, t);
    bp.frequency.exponentialRampToValueAtTime(f1, t + dur);
    src.connect(bp);
    this._env(bp, t, 0.004, peak, dur);
    src.start(t); src.stop(t + dur + 0.05);
  }

  grab() { if (!this.ok) return; this._noise(1400, 350, 1.2, 0.09, 0.35); this._tone('sine', 190, 120, 0.08, 0.25); }
  cancel() { if (!this.ok) return; this._tone('sine', 600, 500, 0.04, 0.08); }
  release() {
    if (!this.ok) return;
    const t = this.ctx.currentTime;
    const o = this.ctx.createOscillator();
    o.type = 'sine';
    o.frequency.setValueAtTime(560, t);
    o.frequency.exponentialRampToValueAtTime(230, t + 0.28);
    const lfo = this.ctx.createOscillator(), lg = this.ctx.createGain();
    lfo.frequency.value = 22; lg.gain.value = 35;
    lfo.connect(lg); lg.connect(o.frequency);
    this._env(o, t, 0.006, 0.22, 0.3);
    o.start(t); lfo.start(t); o.stop(t + 0.4); lfo.stop(t + 0.4);
  }
  stretch() { if (!this.ok) return; this._tone('triangle', 180, 420, 0.22, 0.18); this._noise(500, 1600, 2, 0.2, 0.12); }
  drag(level) {
    if (!this.ok) return;
    const now = performance.now();
    if (now - this.lastDrag < 110 || Math.abs(level - this.lastDragLevel) < 0.06) return;
    this.lastDrag = now; this.lastDragLevel = level;
    this._tone('sawtooth', 90 + level * 160, 70 + level * 120, 0.06, 0.035);
  }
  inflateStart(sign = 1) {
    if (!this.ok) return;
    this.inflateStop();
    const t = this.ctx.currentTime;
    const o = this.ctx.createOscillator();
    o.type = 'triangle';
    o.frequency.value = sign > 0 ? 200 : 520;
    const g = this.ctx.createGain();
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(0.09, t + 0.08);
    const lfo = this.ctx.createOscillator(), lg = this.ctx.createGain();
    lfo.frequency.value = 9; lg.gain.value = 6;
    lfo.connect(lg); lg.connect(o.frequency);
    o.connect(g); g.connect(this.master);
    o.start(t); lfo.start(t);
    this.balloon = { o, g, lfo, sign };
  }
  inflate(level) {
    if (!this.balloon || !this.ctx) return;
    const f = this.balloon.sign > 0 ? 200 + level * 520 : 520 - level * 330;
    this.balloon.o.frequency.setTargetAtTime(f, this.ctx.currentTime, 0.05);
  }
  inflateStop() {
    if (!this.balloon || !this.ctx) { this.balloon = null; return; }
    const { o, g, lfo } = this.balloon, t = this.ctx.currentTime;
    g.gain.cancelScheduledValues(t);
    g.gain.setValueAtTime(g.gain.value, t);
    g.gain.exponentialRampToValueAtTime(0.0001, t + 0.12);
    o.stop(t + 0.15); lfo.stop(t + 0.15);
    this.balloon = null;
  }
  tick(amount) { if (!this.ok) return; this._tone('sine', 300 + Math.abs(amount) * 600, 280 + Math.abs(amount) * 600, 0.04, 0.08); }
  reset() { if (!this.ok) return; this._noise(300, 3000, 0.8, 0.35, 0.25); [523, 659, 784].forEach((f, i) => setTimeout(() => this.ok && this._tone('sine', f, f, 0.18, 0.12), 120 + i * 80)); }
  shutter() { if (!this.ok) return; this._noise(4000, 1500, 0.7, 0.05, 0.4); setTimeout(() => this.ok && this._noise(3000, 1200, 0.7, 0.06, 0.3), 70); }
}
