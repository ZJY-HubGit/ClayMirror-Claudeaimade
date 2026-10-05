// Camera access (computer webcam or phone front/back camera).
export class Camera {
  constructor(video) {
    this.video = video;
    this.stream = null;
    this.facing = 'user';
    this.deviceId = null;
    this.devices = [];
  }
  get ready() { return this.video.readyState >= 2 && this.video.videoWidth > 0; }
  get width() { return this.video.videoWidth; }
  get height() { return this.video.videoHeight; }
  // front cameras are shown mirrored like a real mirror
  get mirrored() {
    if (this.facing === 'environment') return false;
    const track = this.stream && this.stream.getVideoTracks()[0];
    const fm = track && track.getSettings && track.getSettings().facingMode;
    return fm ? fm !== 'environment' : true;
  }

  static supported() { return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia); }

  async start({ deviceId = null, facing = null } = {}) {
    this.stop();
    if (facing) this.facing = facing;
    const video = { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30, max: 60 } };
    if (deviceId) video.deviceId = { exact: deviceId };
    else video.facingMode = this.facing;
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({ video, audio: false });
    } catch (e) {
      if (deviceId || e.name === 'OverconstrainedError') this.stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      else throw e;
    }
    this.video.srcObject = this.stream;
    this.video.muted = true;
    this.video.playsInline = true;
    await this.video.play().catch(() => {});
    await new Promise((res) => {
      if (this.ready) return res();
      const on = () => { if (this.ready) { this.video.removeEventListener('loadeddata', on); res(); } };
      this.video.addEventListener('loadeddata', on);
      setTimeout(res, 3000);
    });
    const track = this.stream.getVideoTracks()[0];
    this.deviceId = track && track.getSettings ? track.getSettings().deviceId : null;
    try { this.devices = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === 'videoinput'); } catch { this.devices = []; }
    return this;
  }

  async next() {
    if (this.devices.length > 1) {
      const i = this.devices.findIndex((d) => d.deviceId === this.deviceId);
      const d = this.devices[(i + 1) % this.devices.length];
      const label = (d.label || '').toLowerCase();
      this.facing = /back|rear|environment|后|背/.test(label) ? 'environment' : 'user';
      return this.start({ deviceId: d.deviceId });
    }
    return this.start({ facing: this.facing === 'user' ? 'environment' : 'user' });
  }

  stop() {
    if (this.stream) for (const t of this.stream.getTracks()) t.stop();
    this.stream = null;
  }

  get label() {
    const track = this.stream && this.stream.getVideoTracks()[0];
    return track ? track.label : '';
  }
}
