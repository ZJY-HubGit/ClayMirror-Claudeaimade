// 橡皮泥魔镜 ClayMirror – app bootstrap, main loop and UI.
import { VisionPipeline } from './vision.js';
import { ServerEngine, OrtEngine } from './engine.js';
import { Camera } from './camera.js';
import { Clay } from './clay.js';
import { Interaction } from './interaction.js';
import { Renderer, drawHandMask } from './renderer.js';
import { Overlay } from './overlay.js';
import { Sfx } from './audio.js';
import { buildFrames, FaceKeeper } from './parts.js';

const $ = (s) => document.querySelector(s);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const MODEL_LABEL = { palm: '手掌检测', hand: '手部关键点', face_det: '人脸检测', face: '脸部网格', pose_det: '人体检测', pose: '身体姿态' };
// on another device (phone) frames are sent as JPEG over the LAN; ?remote=1 forces that mode for testing
const IS_LOCAL = ['localhost', '127.0.0.1', '[::1]', '::1'].includes(location.hostname) && !new URLSearchParams(location.search).has('remote');

const state = {
  engine: null, kind: null, info: null,
  running: false, busy: false, paused: false,
  det: { hands: [], face: null, pose: null }, F: null,
  view: null, viewDev: null, mirror: true, handMask: true,
  fps: 0, frameCount: 0, fpsT: 0, lastT: 0,
  hint: '', hintT: 0, lost: 0, reconnecting: false,
};

const video = $('#video');
const camera = new Camera(video);
const sfx = new Sfx();
const clay = new Clay();
const inter = new Interaction(clay, sfx);
let renderer = null, overlay = null, pipeline = null;
const frameCanvas = document.createElement('canvas');
const frameCtx = frameCanvas.getContext('2d', { willReadFrequently: true });
const maskCanvas = document.createElement('canvas');
const faceKeeper = new FaceKeeper();

// ------------------------------------------------------------------ helpers
function toast(msg, ms = 1900) {
  const t = $('#toast');
  t.textContent = msg;
  t.classList.add('show');
  clearTimeout(toast._t);
  toast._t = setTimeout(() => t.classList.remove('show'), ms);
}
function setEngineStatus(text, cls = '') {
  const el = $('#engine-status');
  el.className = 'engine-status ' + cls;
  $('#engine-text').textContent = text;
}
function setEngineLog(lines) { $('#engine-log').textContent = Array.isArray(lines) ? lines.slice(-4).join('\n') : lines; }
async function fetchInfo(timeout = 2500) {
  try {
    const ac = new AbortController();
    const t = setTimeout(() => ac.abort(), timeout);
    const r = await fetch('api/info', { cache: 'no-store', signal: ac.signal });
    clearTimeout(t);
    if (!r.ok) return null;
    const j = await r.json();
    return j && j.server ? j : null;
  } catch { return null; }
}
function deviceName(dev) {
  const n = state.info && state.info.device_names && state.info.device_names[dev];
  return n ? n.replace(/\(R\)|\(TM\)/g, '').replace(/\s+/g, ' ').trim() : dev;
}
function currentDevices() { return state.engine ? state.engine.devices() : {}; }

function summary() {
  const d = currentDevices();
  const lm = ['hand', 'face', 'pose'].map((k) => d[k]).filter(Boolean);
  const all = Object.values(d).filter(Boolean);
  const npuCount = all.filter((x) => x === 'NPU').length;
  const main = lm.includes('NPU') ? 'NPU' : lm.includes('GPU') ? 'GPU' : 'CPU';
  const mixed = new Set(all).size > 1;
  return { main, mixed, npuCount, total: all.length, devices: [...new Set(all)] };
}

// ------------------------------------------------------------------- engine
async function boot() {
  try {
    renderer = new Renderer($('#gl'));
  } catch (e) {
    setEngineStatus('这个浏览器不支持 WebGL2，请用最新版 Chrome / Edge / Safari 打开', 'warn');
    return;
  }
  overlay = new Overlay($('#overlay'));
  if (!Camera.supported()) {
    $('#cam-error').innerHTML = location.protocol === 'http:' && !IS_LOCAL
      ? '手机浏览器只允许在 <b>https</b> 页面使用摄像头，请用电脑上显示的 https 地址打开。'
      : '这个浏览器不支持摄像头访问。';
  }
  setEngineStatus('正在寻找本机 Intel NPU 加速服务 …');
  const info = await fetchInfo();
  if (info) await useServer(info);
  else await useBrowser('没有找到本机 NPU 服务（用 start.bat 启动即可启用 NPU）');
}

async function useServer(info) {
  try {
    let misses = 0;
    while (info && (info.status === 'loading' || info.status === 'init')) {
      setEngineStatus(info.npu ? '正在把模型编译到 Intel NPU …（首次约 10–60 秒，之后会缓存）' : '正在加载模型 …', '');
      setEngineLog((info.progress || []).map((s) => s.replace(/^\[engine\]\s*/, '')));
      await sleep(700);
      const next = await fetchInfo();
      if (next) { info = next; misses = 0; } else if (++misses > 8) throw new Error('服务没有响应');
    }
    if (!info || info.status === 'error') throw new Error((info && info.error) || '服务不可用');
    const eng = new ServerEngine({ remote: !IS_LOCAL });
    eng.onInfo = (i) => { state.info = i; updateBadge(); renderPanel(); };
    eng.onClose = () => onEngineLost();
    await eng.connect();
    state.info = eng.info;
    setEngine(eng, 'server');
    const s = summary();
    const where = IS_LOCAL ? '' : '（由电脑计算）';
    if (s.main === 'NPU') setEngineStatus(`✅ 已启用 Intel NPU · ${deviceName('NPU')}${where} · ${s.npuCount}/${s.total} 个模型在 NPU 上`, 'ok');
    else setEngineStatus(`⚠️ 没有检测到 NPU，使用 ${s.main}${s.main !== 'CPU' ? ' · ' + deviceName(s.main) : ''}${where}`, 'warn');
    setEngineLog(state.info.notes && state.info.notes.length ? state.info.notes : '');
  } catch (e) {
    console.warn(e);
    await useBrowser('NPU 服务出错：' + e.message);
  }
}

async function useBrowser(reason = '') {
  setEngineStatus('正在加载浏览器内置引擎（WebNN → WebGPU → WASM）…', '');
  try {
    const eng = new OrtEngine();
    await eng.init((m) => setEngineLog(m));
    setEngine(eng, 'browser');
    const s = summary();
    const ep = { NPU: 'WebNN · NPU', GPU: 'WebGPU · GPU', CPU: 'WASM · CPU' }[s.main];
    setEngineStatus(`${s.main === 'NPU' ? '✅' : '⚠️'} 浏览器引擎：${ep}`, s.main === 'NPU' ? 'ok' : 'warn');
    setEngineLog(reason);
  } catch (e) {
    setEngineStatus('引擎加载失败：' + e.message, 'warn');
    return;
  }
}

function setEngine(eng, kind) {
  if (state.engine && state.engine !== eng) state.engine.close();
  state.engine = eng;
  state.kind = kind;
  if (pipeline) pipeline.setEngine(eng);
  else pipeline = new VisionPipeline(eng);
  // slow engines: lighten the per-frame load
  pipeline.reset();
  $('#btn-start').disabled = false;
  updateBadge();
  renderPanel();
}

async function onEngineLost() {
  if (state.kind !== 'server' || state.reconnecting) return;
  state.reconnecting = true;
  toast('NPU 服务连接断开，正在重连 …', 3000);
  for (let i = 0; i < 4; i++) {
    await sleep(1500);
    const info = await fetchInfo();
    if (info && info.status === 'ready') {
      try {
        const eng = new ServerEngine({ remote: !IS_LOCAL });
        eng.onInfo = (inf) => { state.info = inf; updateBadge(); renderPanel(); };
        eng.onClose = () => onEngineLost();
        await eng.connect();
        state.info = eng.info;
        setEngine(eng, 'server');
        toast('已重新连接 NPU 服务');
        state.reconnecting = false;
        return;
      } catch {}
    }
  }
  state.reconnecting = false;
  toast('改用浏览器内置引擎', 2500);
  await useBrowser('NPU 服务断开');
}

// --------------------------------------------------------------------- start
$('#btn-start').addEventListener('click', async () => {
  sfx.unlock();
  $('#cam-error').textContent = '';
  try {
    await camera.start();
  } catch (e) {
    const name = e && e.name;
    $('#cam-error').innerHTML = name === 'NotAllowedError' ? '摄像头权限被拒绝了。请点地址栏的 🔒/📷 图标允许摄像头，然后刷新页面。'
      : name === 'NotFoundError' ? '没有找到摄像头。' : name === 'NotReadableError' ? '摄像头被其他程序占用（关掉会议软件等再试）。'
      : '无法打开摄像头：' + (e && e.message);
    return;
  }
  state.mirror = camera.mirrored;
  $('#start').classList.remove('show');
  document.querySelectorAll('.hud').forEach((h) => h.classList.remove('hidden'));
  state.running = true;
  toast('🤏 用拇指和食指捏住脸上的部位试试！', 3200);
  schedule();
});

function schedule() {
  if (!state.running) return;
  if ('requestVideoFrameCallback' in HTMLVideoElement.prototype) video.requestVideoFrameCallback(() => onFrame());
  else requestAnimationFrame(() => onFrame());
}
async function onFrame() {
  if (state.running && !state.busy && camera.ready && !document.hidden && state.engine) {
    state.busy = true;
    try { await processFrame(); }
    catch (e) {
      console.warn('frame error', e);
      if (state.kind === 'server' && state.engine.ws && state.engine.ws.readyState !== 1) onEngineLost();
    }
    state.busy = false;
  }
  schedule();
}

function layout(W, H) {
  const cw = window.innerWidth, ch = window.innerHeight;
  const dpr = Math.min(window.devicePixelRatio || 1, 2, 2560 / Math.max(cw, ch));
  const scale = Math.max(cw / W, ch / H);
  state.view = { scale, ox: (cw - W * scale) / 2, oy: (ch - H * scale) / 2, W, H };
  state.viewDev = { scale: scale * dpr, ox: state.view.ox * dpr, oy: state.view.oy * dpr };
  renderer.resize(Math.round(cw * dpr), Math.round(ch * dpr));
  overlay.resize(cw, ch, dpr);
}

async function processFrame() {
  const t = performance.now();
  const vw = camera.width, vh = camera.height;
  const k = Math.min(1, 1280 / Math.max(vw, vh));
  const W = Math.round(vw * k), H = Math.round(vh * k);
  if (frameCanvas.width !== W || frameCanvas.height !== H) {
    frameCanvas.width = W; frameCanvas.height = H;
    inter.frameSize = [W, H];
    pipeline.reset();
    faceKeeper.reset();
  }
  frameCtx.drawImage(video, 0, 0, W, H);
  let det = await pipeline.process(frameCanvas, W, H, t);
  if (window.__clayInject) det = window.__clayInject(det, t, W, H) || det;
  const dt = state.lastT ? Math.min(0.1, (t - state.lastT) / 1000) : 1 / 30;
  state.lastT = t;
  // anchors use the real face mesh, or a body-tracked copy of it while the face is covered
  const F = buildFrames({ ...det, face: faceKeeper.update(det, t) });
  state.det = det; state.F = F;
  const tj = performance.now();
  clay.build(F, t, dt);
  inter.update(F, det, t, dt);
  const { P, n } = clay.build(F, t, 0);
  state.jsMs = performance.now() - tj;
  layout(W, H);
  renderer.setFrame(frameCanvas, W, H);
  const hasMask = state.handMask && drawHandMask(maskCanvas, det.hands, W, H);
  renderer.setMask(maskCanvas, hasMask);
  renderer.setPrims(P, n);
  renderer.render(state.viewDev, state.mirror);
  overlay.draw(state.view, state.mirror, det, F, inter, clay, t);
  hud(det, t);
}

// ----------------------------------------------------------------------- HUD
function hud(det, t) {
  state.frameCount++;
  if (t - state.fpsT > 500) {
    state.fps = (state.frameCount * 1000) / (t - state.fpsT || 1);
    state.frameCount = 0; state.fpsT = t;
    const st = pipeline.stats;
    const ms = Object.values(st.models).reduce((a, m) => Math.max(a, m.ms), 0);
    const s = summary();
    $('#perf').textContent = `${state.fps.toFixed(0)} fps · 推理 ${ms.toFixed(1)} ms · 往返 ${st.roundtripMs.toFixed(0)} ms · ${s.devices.join('+')}`;
    const person = !!(det.face || det.pose);
    const chips = [
      ['👤 人物', person], ['🙂 脸', !!det.face],
      [`✋ 手 ×${det.hands.length}`, det.hands.length > 0], ['🕺 身体', !!det.pose],
      [`🧱 变形 ${clay.count}`, clay.count > 0],
    ];
    $('#chips').innerHTML = chips.map(([l, on]) => `<span class="chip ${on ? 'on' : ''}">${l}</span>`).join('');
    renderPanelLive();
  }
  // contextual hint
  const cur = [...inter.cursors.values()];
  const g = cur.find((c) => c.grab);
  let h;
  if (g) {
    h = g.grab.kind === 'pending' ? '拖动 = 拉扯 · 停住不动 = 吹大'
      : g.grab.kind === 'pull' ? '松手后会保留形状，可以继续捏别的地方叠加'
      : g.grab.kind === 'inflate' ? (g.grab.sign > 0 ? '捏住越久越大 🎈 松手保留' : '捏住越久越小 🫧 松手保留')
      : '拉开 = 拉伸 · 合拢 = 挤扁 · 转动 = 扭一扭';
  } else if (!det.face && !det.pose) h = '请站到镜头前 👀';
  else if (!det.hands.length) h = '举起手 ✋ 用拇指和食指去“捏”画面里的自己';
  else if (inter.resetInfo) h = '保持双手张开就会全部复原';
  else if (clay.count === 0) h = '把拇指和食指捏在一起 🤏 抓住眼睛、鼻子、脸颊或下巴';
  else h = clay.count > 3 ? '双手张开停 2 秒 = 复原 · 📸 拍照留念' : '再捏一个地方，效果会叠加 ✨';
  if (h !== state.hint) { state.hint = h; $('#hint').textContent = h; }
}

function updateBadge() {
  const b = $('#badge');
  if (!state.engine) { b.textContent = '…'; return; }
  const s = summary();
  const pre = state.kind === 'server' && !IS_LOCAL ? '电脑 ' : state.kind === 'browser' ? '浏览器 ' : '';
  const name = s.main === 'NPU' ? (state.kind === 'server' ? 'Intel NPU' : 'NPU (WebNN)') : s.main === 'GPU' ? (state.kind === 'server' ? 'GPU · ' + deviceName('GPU') : 'WebGPU') : 'CPU';
  b.className = 'badge ' + s.main.toLowerCase();
  b.innerHTML = `<span class="dot"></span>${s.main === 'NPU' ? '⚡ ' : ''}${pre}${name}${s.mixed ? ' +' : ''}`;
}

function renderPanel() {
  if (!state.engine) return;
  const pe = $('#panel-engine');
  if (state.kind === 'server') {
    const i = state.info || {};
    const devs = (i.devices || []).map((d) => `<span class="dev ${d}">${d}</span> ${deviceName(d)}`).join('<br>');
    pe.innerHTML = `引擎：<b>${state.engine.label}</b> · OpenVINO ${(i.openvino || '').split('-')[0]}<br>${devs || ''}${!IS_LOCAL ? '<br>📱 手机画面会发送到电脑，由电脑的 NPU 计算' : ''}`;
    $('#policy-row').style.display = '';
    $('#policy').value = i.policy || 'auto';
    $('#btn-engine-switch').textContent = '改用浏览器内置引擎';
    const notes = (i.notes || []).map((n) => '• ' + n).join('<br>');
    $('#panel-notes').innerHTML = (notes ? notes + '<br>' : '') +
      '“自动”策略：所有模型先尝试在 NPU 上运行；某个模型 NPU 不支持、或 NPU 每帧负载超出预算时，才把最重的模型移到 GPU / CPU。';
  } else {
    pe.innerHTML = `引擎：<b>${state.engine.label}</b><br>依次尝试：WebNN (NPU) → WebGPU (GPU) → WASM (CPU)` +
      (state.engine.lite ? '<br>CPU 模式下使用轻量手部模型' : '');
    $('#policy-row').style.display = 'none';
    $('#btn-engine-switch').textContent = '改用本机 NPU 服务';
    $('#panel-notes').innerHTML = '想用 Intel NPU：在电脑上运行 <b>start.bat</b>，再打开它显示的地址（http://localhost:8848）。';
  }
  renderPanelLive();
}
function renderPanelLive() {
  if (!$('#panel').classList.contains('show') || !state.engine) return;
  const d = currentDevices();
  const st = (pipeline && pipeline.stats.models) || {};
  const sm = (state.info && state.info.models) || {};
  $('#model-rows').innerHTML = Object.keys(MODEL_LABEL).map((k) => {
    const dev = (st[k] && st[k].device) || d[k] || '—';
    const ms = st[k] && st[k].n ? st[k].ms.toFixed(1) + ' ms' : sm[k] && sm[k].bench_ms != null ? `${sm[k].bench_ms} ms（测速）` : '—';
    return `<tr><td>${MODEL_LABEL[k]}</td><td><span class="dev ${dev}">${dev}</span></td><td>${ms}</td></tr>`;
  }).join('');
}

// ------------------------------------------------------------------ actions
function actUndo() { const it = clay.undo(); if (it) { toast('↶ 撤销：' + (it.label || '')); sfx.cancel(); } else toast('没有可以撤销的了'); }
function actReset() { inter.cancelAll(); clay.reset(); sfx.reset(); toast('⟲ 全部复原'); }
function actMode() {
  inter.mode = inter.mode === 'grow' ? 'shrink' : 'grow';
  $('#mode-icon').textContent = inter.mode === 'grow' ? '🎈' : '🫧';
  $('#mode-label').textContent = inter.mode === 'grow' ? '放大' : '缩小';
  toast(inter.mode === 'grow' ? '🎈 捏住不动 = 放大' : '🫧 捏住不动 = 缩小');
}
function actSnap() {
  const gl = $('#gl');
  const c = document.createElement('canvas');
  c.width = gl.width; c.height = gl.height;
  const x = c.getContext('2d');
  x.drawImage(gl, 0, 0);
  const fs = Math.max(16, Math.round(c.height / 32));
  x.font = `700 ${fs}px "PingFang SC","Microsoft YaHei",sans-serif`;
  x.fillStyle = 'rgba(255,255,255,0.85)';
  x.shadowColor = 'rgba(0,0,0,0.6)'; x.shadowBlur = 8;
  x.fillText('🧱 橡皮泥魔镜', fs, c.height - fs);
  c.toBlob((b) => {
    if (!b) return;
    const a = document.createElement('a');
    const d = new Date(), p = (n) => String(n).padStart(2, '0');
    a.download = `ClayMirror-${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}.png`;
    a.href = URL.createObjectURL(b);
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 5000);
  }, 'image/png');
  const f = $('#flash');
  f.classList.add('on');
  requestAnimationFrame(() => requestAnimationFrame(() => f.classList.remove('on')));
  sfx.shutter();
  toast('📸 照片已保存（下载文件夹）');
}
function actDebug() {
  overlay.debug = !overlay.debug;
  document.querySelector('[data-act="debug"]').classList.toggle('on', overlay.debug);
  toast(overlay.debug ? '🦴 显示识别结果：人脸网格 · 手部 · 身体骨架' : '已隐藏识别结果');
}
async function actCamera() {
  try {
    if (camera.devices.length <= 1 && !/Android|iPhone|iPad|Mobile/i.test(navigator.userAgent)) { toast('只找到一个摄像头'); return; }
    await camera.next();
    state.mirror = camera.mirrored;
    pipeline.reset();
    toast('📷 ' + (camera.label || (camera.facing === 'user' ? '前置摄像头' : '后置摄像头')));
  } catch (e) { toast('切换摄像头失败'); }
}
function actSound() {
  sfx.setEnabled(!sfx.enabled);
  $('#sound-icon').textContent = sfx.enabled ? '🔊' : '🔇';
  if (sfx.enabled) sfx.unlock();
}
function openModal(id) { document.querySelectorAll('.modal').forEach((m) => { if (m.id !== 'start') m.classList.remove('show'); }); $('#' + id).classList.add('show'); }
function actPhone() {
  const body = $('#phone-body');
  const lan = (state.info && state.info.lan) || [];
  let url = null;
  if (state.kind === 'server' && lan.length) url = lan[0];
  else if (location.protocol === 'https:') url = location.href.split('#')[0];
  if (!url) {
    body.innerHTML = `<p class="steps">现在是浏览器内置引擎模式，没有找到电脑上的 NPU 服务。<br>在电脑上运行 <b>start.bat</b> 后，这里会出现手机扫码地址。</p>`;
  } else {
    let qrHtml = '';
    try {
      const q = window.qrcode(0, 'M'); q.addData(url); q.make();
      qrHtml = `<div class="qr">${q.createImgTag(6, 0)}</div>`;
    } catch {}
    const others = lan.slice(1).map((u) => `<span class="url">${u}</span>`).join('');
    body.innerHTML = `${qrHtml}<div><span class="url">${url}</span>${others}</div>
      <ol class="steps">
        <li>手机和这台电脑连同一个 Wi-Fi</li>
        <li>用手机相机扫上面的二维码（或手动输入地址）</li>
        <li>出现“连接不是私密连接/不安全”时，点 <b>高级 → 继续前往</b>（这是本机自动生成的证书，只在局域网用）</li>
        <li>允许使用摄像头，点“打开摄像头开始”</li>
      </ol>
      <p class="small">手机拍到的画面会通过局域网交给电脑的 Intel NPU 计算；如果 Windows 防火墙弹窗，请允许 Python 访问专用网络。</p>`;
  }
  openModal('phone');
}

const ACTIONS = { undo: actUndo, reset: actReset, mode: actMode, snap: actSnap, debug: actDebug, camera: actCamera, phone: actPhone, sound: actSound, help: () => openModal('help') };
document.querySelectorAll('[data-act]').forEach((b) => b.addEventListener('click', () => { sfx.unlock(); ACTIONS[b.dataset.act](); }));
document.querySelectorAll('[data-close]').forEach((b) => b.addEventListener('click', () => b.closest('.modal').classList.remove('show')));
document.querySelectorAll('.modal').forEach((m) => m.addEventListener('click', (e) => { if (e.target === m && m.id !== 'start') m.classList.remove('show'); }));
$('#badge').addEventListener('click', () => { renderPanel(); openModal('panel'); renderPanelLive(); });
$('#policy').addEventListener('change', (e) => {
  if (state.kind !== 'server') return;
  state.engine.setPolicy(e.target.value);
  toast('正在按新策略重新分配模型 …', 2500);
});
$('#btn-engine-switch').addEventListener('click', async () => {
  $('#panel').classList.remove('show');
  if (state.kind === 'server') { toast('切换到浏览器内置引擎 …'); await useBrowser('手动切换'); }
  else {
    const info = await fetchInfo();
    if (info) { toast('连接本机 NPU 服务 …'); await useServer(info); } else toast('没有找到 NPU 服务，请先运行 start.bat', 3000);
  }
  updateBadge();
});

window.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'SELECT' || e.ctrlKey && e.key.toLowerCase() !== 'z' || e.metaKey) return;
  const k = e.key.toLowerCase();
  if (k === 'escape') document.querySelectorAll('.modal').forEach((m) => { if (m.id !== 'start') m.classList.remove('show'); });
  if (!state.running) return;
  const map = { z: actUndo, r: actReset, x: actMode, s: actSnap, d: actDebug, c: actCamera, m: actSound, h: () => openModal('help'), '?': () => openModal('help') };
  if (map[k]) { e.preventDefault(); map[k](); }
});

// ---------------------------------------------------------------- pointers
const ov = $('#overlay');
function screenToRaw(sx, sy) {
  const v = state.view;
  if (!v) return [sx, sy];
  let x = (sx - v.ox) / v.scale;
  const y = (sy - v.oy) / v.scale;
  if (state.mirror) x = v.W - x;
  return [x, y];
}
ov.addEventListener('pointerdown', (e) => {
  sfx.unlock();
  if (e.pointerType === 'mouse' && e.button !== 0) return;
  try { ov.setPointerCapture(e.pointerId); } catch {}
  const [x, y] = screenToRaw(e.clientX, e.clientY);
  inter.pointerDown(e.pointerId, x, y, e.pointerType);
});
ov.addEventListener('pointermove', (e) => { const [x, y] = screenToRaw(e.clientX, e.clientY); inter.pointerMove(e.pointerId, x, y, e.pointerType); });
ov.addEventListener('pointerup', (e) => inter.pointerUp(e.pointerId));
ov.addEventListener('pointercancel', (e) => inter.pointerUp(e.pointerId));
ov.addEventListener('pointerleave', (e) => inter.pointerLeave(e.pointerId));
ov.addEventListener('contextmenu', (e) => e.preventDefault());
ov.addEventListener('wheel', (e) => {
  e.preventDefault();
  if (!state.F) return;
  const [x, y] = screenToRaw(e.clientX, e.clientY);
  inter.wheel(x, y, e.deltaY, state.F);
}, { passive: false });

inter.onEvent = (type, item) => {
  if (type === 'reset') toast('⟲ 全部复原');
};

// test / debugging hook
window.__clay = { state, clay, inter, camera, get pipeline() { return pipeline; }, get engine() { return state.engine; }, actions: ACTIONS };

boot();
