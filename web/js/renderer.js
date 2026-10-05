// WebGL2 renderer: samples the camera frame through the clay deformation chain.
//  pass 1: evaluate the deformation chain on a half-resolution grid of the
//          camera frame → offset textures (all deformations / left-hand-only /
//          right-hand-only). Cheap even with hundreds of primitives.
//  pass 2: per screen pixel, look up the offset and sample the camera frame.
// Hands are composited "in front": inside a hand only its own deformations
// apply, so the fingers doing the pinching never get smeared.
import { PF } from './clay.js';

const ROW = 256; // primitives per texture row (4 texels each)

const VS = `#version 300 es
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

const CHAIN = `
uniform highp sampler2D uPrims;
uniform int uCount;
vec4 T(int i, int k) { return texelFetch(uPrims, ivec2((i % ${ROW}) * 4 + k, i / ${ROW}), 0); }
vec2 invPrim(int i, vec2 x) {
  vec4 a = T(i, 0);
  vec4 b = T(i, 1);
  vec2 c = a.zw;
  float r = b.x;
  vec2 d = x - c;
  if (a.x < 0.5) {
    float q = dot(d, d) / (r * r);
    if (q >= 1.0) return x;
    float w = 1.0 - q; w *= w;
    return x - b.yz * w;
  } else if (a.x < 1.5) {
    float q = dot(d, d) / (r * r);
    if (q >= 1.0) return x;
    float k = 1.0 - q;
    return c + d * (1.0 - b.y * k * k);
  }
  vec4 e = T(i, 2);
  vec4 f = T(i, 3);
  float ca = b.w, sa = e.x, su = e.y, sv = e.z, ct = e.w, st = f.x;
  float du = d.x * ca + d.y * sa, dv = -d.x * sa + d.y * ca;
  float ru = r * max(su, 1.0), rv = r * max(sv, 1.0);
  float rho = sqrt(du * du / (ru * ru) + dv * dv / (rv * rv));
  if (rho >= 1.0) return x;
  float w = 1.0 - smoothstep(f.y, 1.0, rho);
  vec2 ee = vec2(d.x * ct + d.y * st, -d.x * st + d.y * ct);
  float eu = ee.x * ca + ee.y * sa, ev = -ee.x * sa + ee.y * ca;
  float a1 = eu / su, a2 = ev / sv;
  vec2 src = b.yz + vec2(a1 * ca - a2 * sa, a1 * sa + a2 * ca);
  return mix(x, src, w);
}
// chains for: all primitives, group 1 only (left hand), group 2 only (right hand)
void chain3(vec2 x, out vec2 sAll, out vec2 sL, out vec2 sR) {
  sAll = x; sL = x; sR = x;
  for (int i = uCount - 1; i >= 0; --i) {
    float g = T(i, 0).y;
    sAll = invPrim(i, sAll);
    if (g > 0.5 && g < 1.5) sL = invPrim(i, sL);
    else if (g > 1.5) sR = invPrim(i, sR);
  }
}
`;

const FS_DISP = `#version 300 es
precision highp float;
precision highp int;
${CHAIN}
uniform vec2 uFrameSize;
uniform vec2 uGrid;
layout(location = 0) out vec4 o0;
layout(location = 1) out vec4 o1;
void main() {
  vec2 raw = gl_FragCoord.xy / uGrid * uFrameSize;
  vec2 a, l, r;
  chain3(raw, a, l, r);
  o0 = vec4(a - raw, l - raw);
  o1 = vec4(r - raw, 0.0, 0.0);
}`;

const FS_SHOW = `#version 300 es
precision highp float;
precision highp int;
uniform sampler2D uFrame;
uniform sampler2D uMask;
uniform sampler2D uDisp0;
uniform sampler2D uDisp1;
uniform vec2 uFrameSize;
uniform vec2 uCanvas;
uniform vec3 uView;
uniform float uMirror;
uniform float uHasMask;
uniform float uHasDisp;
uniform float uVignette;
out vec4 outColor;
void main() {
  vec2 scr = vec2(gl_FragCoord.x, uCanvas.y - gl_FragCoord.y);
  vec2 raw = (scr - uView.yz) / uView.x;
  if (uMirror > 0.5) raw.x = uFrameSize.x - raw.x;
  vec2 uv0 = raw / uFrameSize;
  vec2 s = raw;
  if (uHasDisp > 0.5) {
    vec2 g = uv0;   // grid row j <-> raw y = (j+0.5)/rows*H (same orientation as the frame)
    vec4 d0 = texture(uDisp0, g);
    s = raw + d0.xy;
    if (uHasMask > 0.5) {
      vec2 m = texture(uMask, uv0).rg;
      if (m.r > 0.02) s = mix(s, raw + d0.zw, m.r);
      if (m.g > 0.02) s = mix(s, raw + texture(uDisp1, g).xy, m.g);
    }
  }
  vec2 uv = s / uFrameSize;
  vec3 col;
  if (uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0) col = texture(uFrame, clamp(uv, 0.0, 1.0)).rgb * 0.35;
  else col = texture(uFrame, uv).rgb;
  vec2 q = scr / uCanvas - 0.5;
  col *= 1.0 - uVignette * dot(q, q) * 1.1;
  outColor = vec4(col, 1.0);
}`;

// single-pass fallback (no float render targets): full chain per screen pixel
const FS_DIRECT = `#version 300 es
precision highp float;
precision highp int;
${CHAIN}
uniform sampler2D uFrame;
uniform sampler2D uMask;
uniform vec2 uFrameSize;
uniform vec2 uCanvas;
uniform vec3 uView;
uniform float uMirror;
uniform float uHasMask;
uniform float uVignette;
out vec4 outColor;
void main() {
  vec2 scr = vec2(gl_FragCoord.x, uCanvas.y - gl_FragCoord.y);
  vec2 raw = (scr - uView.yz) / uView.x;
  if (uMirror > 0.5) raw.x = uFrameSize.x - raw.x;
  vec2 a, l, r;
  chain3(raw, a, l, r);
  vec2 s = a;
  if (uHasMask > 0.5) {
    vec2 m = texture(uMask, raw / uFrameSize).rg;
    s = mix(s, l, m.r);
    s = mix(s, r, m.g);
  }
  vec2 uv = s / uFrameSize;
  vec3 col;
  if (uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0) col = texture(uFrame, clamp(uv, 0.0, 1.0)).rgb * 0.35;
  else col = texture(uFrame, uv).rgb;
  vec2 q = scr / uCanvas - 0.5;
  col *= 1.0 - uVignette * dot(q, q) * 1.1;
  outColor = vec4(col, 1.0);
}`;

function compile(gl, type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
  return s;
}
function program(gl, fs, names) {
  const p = gl.createProgram();
  gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, VS));
  gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  const u = {};
  for (const n of names) u[n] = gl.getUniformLocation(p, n);
  return { p, u };
}

export class Renderer {
  constructor(canvas) {
    this.canvas = canvas;
    const gl = canvas.getContext('webgl2', { antialias: false, alpha: false, premultipliedAlpha: false, preserveDrawingBuffer: true, powerPreference: 'high-performance' });
    if (!gl) throw new Error('WebGL2 不可用');
    this.gl = gl;
    const common = ['uFrame', 'uMask', 'uPrims', 'uCount', 'uFrameSize', 'uCanvas', 'uView', 'uMirror', 'uHasMask', 'uVignette'];
    this.direct = program(gl, FS_DIRECT, common);
    this.floatRT = !!gl.getExtension('EXT_color_buffer_float') || !!gl.getExtension('EXT_color_buffer_half_float');
    if (this.floatRT) {
      try {
        this.disp = program(gl, FS_DISP, ['uPrims', 'uCount', 'uFrameSize', 'uGrid']);
        this.show = program(gl, FS_SHOW, ['uFrame', 'uMask', 'uDisp0', 'uDisp1', 'uFrameSize', 'uCanvas', 'uView', 'uMirror', 'uHasMask', 'uHasDisp', 'uVignette']);
      } catch (e) { console.warn('two-pass renderer unavailable', e); this.floatRT = false; }
    }
    this.vao = gl.createVertexArray();
    const mk = (filter) => {
      const t = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      return t;
    };
    this.mk = mk;
    this.frameTex = mk(gl.LINEAR);
    this.maskTex = mk(gl.LINEAR);
    this.primTex = mk(gl.NEAREST);
    this.dispTex = [mk(gl.LINEAR), mk(gl.LINEAR)];
    this.fbo = gl.createFramebuffer();
    this.grid = [0, 0];
    this.primRows = 0;
    this.maxPrims = ROW * 16;
    this.count = 0;
    this.hasMask = false;
    this.frameSize = [1, 1];
    this.vignette = 0.35;
    this.gridScale = 0.5;
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  }

  resize(w, h) {
    if (this.canvas.width !== w || this.canvas.height !== h) { this.canvas.width = w; this.canvas.height = h; }
  }
  setFrame(src, w, h) {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.frameTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, src);
    this.frameSize = [w, h];
  }
  setMask(maskCanvas, has) {
    this.hasMask = has;
    if (!has) return;
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.maskTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, maskCanvas);
  }
  setPrims(P, n) {
    const gl = this.gl;
    n = Math.min(n, this.maxPrims);
    this.count = n;
    if (!n) return;
    gl.bindTexture(gl.TEXTURE_2D, this.primTex);
    const rows = Math.ceil(n / ROW);
    const cap = Math.max(1, 1 << Math.ceil(Math.log2(rows)));
    if (cap !== this.primRows) {
      this.primRows = cap;
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, ROW * 4, cap, 0, gl.RGBA, gl.FLOAT, null);
    }
    const full = Math.floor(n / ROW), rest = n % ROW;
    if (full) gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, ROW * 4, full, gl.RGBA, gl.FLOAT, P.subarray(0, full * ROW * PF));
    if (rest) gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, full, rest * 4, 1, gl.RGBA, gl.FLOAT, P.subarray(full * ROW * PF, n * PF));
  }

  _ensureGrid() {
    const gl = this.gl;
    const gw = Math.max(8, Math.round(this.frameSize[0] * this.gridScale)), gh = Math.max(8, Math.round(this.frameSize[1] * this.gridScale));
    if (gw === this.grid[0] && gh === this.grid[1]) return true;
    this.grid = [gw, gh];
    for (const t of this.dispTex) {
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, gw, gh, 0, gl.RGBA, gl.HALF_FLOAT, null);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.dispTex[0], 0);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, this.dispTex[1], 0);
    const ok = gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    if (!ok) { console.warn('float framebuffer incomplete → single pass'); this.floatRT = false; }
    return ok;
  }

  render(view, mirror) {
    const gl = this.gl;
    gl.bindVertexArray(this.vao);
    if (this.floatRT && this._ensureGrid()) {
      const hasDisp = this.count > 0;
      if (hasDisp) {
        const { p, u } = this.disp;
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
        gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);
        gl.viewport(0, 0, this.grid[0], this.grid[1]);
        gl.useProgram(p);
        gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, this.primTex); gl.uniform1i(u.uPrims, 2);
        gl.uniform1i(u.uCount, this.count);
        gl.uniform2f(u.uFrameSize, this.frameSize[0], this.frameSize[1]);
        gl.uniform2f(u.uGrid, this.grid[0], this.grid[1]);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      }
      const { p, u } = this.show;
      gl.viewport(0, 0, this.canvas.width, this.canvas.height);
      gl.useProgram(p);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.frameTex); gl.uniform1i(u.uFrame, 0);
      gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.maskTex); gl.uniform1i(u.uMask, 1);
      gl.activeTexture(gl.TEXTURE3); gl.bindTexture(gl.TEXTURE_2D, this.dispTex[0]); gl.uniform1i(u.uDisp0, 3);
      gl.activeTexture(gl.TEXTURE4); gl.bindTexture(gl.TEXTURE_2D, this.dispTex[1]); gl.uniform1i(u.uDisp1, 4);
      gl.uniform2f(u.uFrameSize, this.frameSize[0], this.frameSize[1]);
      gl.uniform2f(u.uCanvas, this.canvas.width, this.canvas.height);
      gl.uniform3f(u.uView, view.scale, view.ox, view.oy);
      gl.uniform1f(u.uMirror, mirror ? 1 : 0);
      gl.uniform1f(u.uHasMask, this.hasMask ? 1 : 0);
      gl.uniform1f(u.uHasDisp, hasDisp ? 1 : 0);
      gl.uniform1f(u.uVignette, this.vignette);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      return;
    }
    const { p, u } = this.direct;
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.useProgram(p);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.frameTex); gl.uniform1i(u.uFrame, 0);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.maskTex); gl.uniform1i(u.uMask, 1);
    gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, this.primTex); gl.uniform1i(u.uPrims, 2);
    gl.uniform1i(u.uCount, this.count);
    gl.uniform2f(u.uFrameSize, this.frameSize[0], this.frameSize[1]);
    gl.uniform2f(u.uCanvas, this.canvas.width, this.canvas.height);
    gl.uniform3f(u.uView, view.scale, view.ox, view.oy);
    gl.uniform1f(u.uMirror, mirror ? 1 : 0);
    gl.uniform1f(u.uHasMask, this.hasMask ? 1 : 0);
    gl.uniform1f(u.uVignette, this.vignette);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }
}

// Draw hands into a small RGBA canvas: R = left hand, G = right hand.
const BONES = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17]];
export function drawHandMask(canvas, hands, frameW, frameH) {
  const W = 256, H = Math.max(16, Math.round((256 * frameH) / frameW));
  if (canvas.width !== W || canvas.height !== H) { canvas.width = W; canvas.height = H; }
  const ctx = canvas.getContext('2d');
  ctx.globalCompositeOperation = 'source-over';
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, W, H);
  if (!hands.length) return false;
  const k = W / frameW;
  ctx.globalCompositeOperation = 'lighter';
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  for (const h of hands) {
    const lm = h.raw || h.lm;
    const size = Math.hypot(lm[0] - lm[27], lm[1] - lm[28]);
    const col = h.side === 'L' ? 'rgb(255,0,0)' : 'rgb(0,255,0)';
    ctx.strokeStyle = col;
    ctx.fillStyle = col;
    ctx.lineWidth = Math.max(2, 0.3 * size * k);
    ctx.beginPath();
    for (const [a, b] of BONES) { ctx.moveTo(lm[a * 3] * k, lm[a * 3 + 1] * k); ctx.lineTo(lm[b * 3] * k, lm[b * 3 + 1] * k); }
    ctx.stroke();
    ctx.beginPath();
    for (const i of [0, 1, 5, 9, 13, 17]) ctx.lineTo(lm[i * 3] * k, lm[i * 3 + 1] * k);
    ctx.closePath();
    ctx.fill();
  }
  return true;
}
