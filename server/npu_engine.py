"""
NPU-first inference engine built on OpenVINO.

Policy ("auto"):
  1. Every model is compiled on the Intel NPU first (Intel Core Ultra "AI Boost").
  2. If a model cannot be compiled on the NPU, it falls back to GPU, then CPU.
  3. After compiling, every model is benchmarked. If the NPU cannot keep the
     per-frame work inside the frame budget, the heaviest models are moved to
     GPU / CPU until it fits ("NPU 算力不够才用 GPU/CPU").
  4. While running, per-model latency is monitored; if the NPU stays over
     budget for a while, the policy is re-evaluated automatically.

Inputs arrive as uint8 RGB (NHWC) crops; normalisation runs inside the compiled
model (OpenVINO PrePostProcessor), i.e. on the NPU itself.
"""
from __future__ import annotations

import os
import queue
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

try:
    import openvino as ov
except ImportError:  # pragma: no cover - reported by server.py
    ov = None

# ---------------------------------------------------------------------------
# Model catalogue (files live in web/models so the browser fallback can use
# exactly the same networks).
# ---------------------------------------------------------------------------
MODEL_SPECS = {
    "palm":     dict(file="palm_detection_full.onnx",        size=192, norm="01",  role="detector", label="手掌检测",   per_frame=0.25, pool=2),
    "hand":     dict(file="hand_landmark_full.onnx",         size=224, norm="01",  role="landmark", label="手部关键点", per_frame=2.0,  pool=4),
    "face_det": dict(file="face_detection_short_range.onnx", size=128, norm="-11", role="detector", label="人脸检测",   per_frame=0.1,  pool=2),
    "face":     dict(file="face_landmark.onnx",              size=192, norm="01",  role="landmark", label="脸部网格",   per_frame=1.0,  pool=2),
    "pose_det": dict(file="pose_detection.onnx",             size=224, norm="-11", role="detector", label="人体检测",   per_frame=0.1,  pool=2),
    "pose":     dict(file="pose_landmark_full.onnx",         size=256, norm="01",  role="landmark", label="身体姿态",   per_frame=1.0,  pool=2),
}

DEVICE_ORDER = ["NPU", "GPU", "CPU"]
FRAME_BUDGET_MS = 24.0          # keep ~40 fps headroom for the per-frame model work


@dataclass
class Slot:
    key: str
    device: str
    compiled: object
    requests: "queue.Queue"
    bench_ms: float = 0.0
    ema_ms: float = 0.0
    calls: int = 0
    npu_ok: bool | None = None          # did this model compile on the NPU?
    tried: dict = field(default_factory=dict)  # device -> "ok"/error string


class NPUEngine:
    def __init__(self, models_dir: str, cache_dir: str, policy: str = "auto", log=print):
        if ov is None:
            raise RuntimeError("openvino is not installed")
        self.models_dir = models_dir
        self.cache_dir = cache_dir
        self.policy = policy.upper() if policy.lower() != "auto" else "auto"
        self.log = log
        self.core = ov.Core()
        try:
            os.makedirs(cache_dir, exist_ok=True)
            self.core.set_property({"CACHE_DIR": cache_dir})
        except Exception as e:  # caching is an optimisation only
            log(f"[engine] model cache disabled: {e}")
        self.available = [d for d in self.core.available_devices]
        self.device_names = {}
        for d in self.available:
            try:
                self.device_names[d] = str(self.core.get_property(d, "FULL_DEVICE_NAME"))
            except Exception:
                self.device_names[d] = d
        self.base_devices = sorted({d.split(".")[0] for d in self.available}, key=lambda d: DEVICE_ORDER.index(d) if d in DEVICE_ORDER else 9)
        self.slots: dict[str, Slot] = {}
        self._models = {}            # key -> ov.Model (pre-processed, uncompiled)
        self._compiled_cache = {}    # (key, device) -> compiled model
        self._lock = threading.RLock()
        self.status = "init"
        self.progress = []
        self.pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="infer")
        self.last_rebalance = time.time()
        self._rebalancing = False
        self.notes = []

    # ------------------------------------------------------------------ utils
    def _p(self, msg):
        self.progress.append(msg)
        self.progress = self.progress[-40:]
        self.log(msg)

    def has(self, dev):
        return dev in self.base_devices

    def _load_model(self, key):
        if key in self._models:
            return self._models[key]
        spec = MODEL_SPECS[key]
        model = self.core.read_model(os.path.join(self.models_dir, spec["file"]))
        ppp = ov.preprocess.PrePostProcessor(model)
        inp = ppp.input()
        inp.tensor().set_element_type(ov.Type.u8).set_layout(ov.Layout("NHWC"))
        inp.model().set_layout(ov.Layout("NHWC"))
        steps = inp.preprocess().convert_element_type(ov.Type.f32)
        if spec["norm"] == "01":
            steps.scale(255.0)
        else:
            steps.mean(127.5).scale(127.5)
        model = ppp.build()
        self._models[key] = model
        return model

    def _compile(self, key, device):
        ck = (key, device)
        if ck in self._compiled_cache:
            return self._compiled_cache[ck]
        model = self._load_model(key)
        cfg = {"PERFORMANCE_HINT": "LATENCY"}
        t0 = time.time()
        compiled = self.core.compile_model(model, device, cfg)
        self._compiled_cache[ck] = compiled
        self._p(f"  · {MODEL_SPECS[key]['label']:<6} 已在 {device} 上编译 ({(time.time()-t0)*1000:.0f} ms)")
        return compiled

    def _bench(self, compiled, key, n=12):
        size = MODEL_SPECS[key]["size"]
        x = (np.random.RandomState(0).rand(1, size, size, 3) * 255).astype(np.uint8)
        req = compiled.create_infer_request()
        for _ in range(3):
            req.infer({0: x})
        ts = []
        for _ in range(n):
            t = time.perf_counter()
            req.infer({0: x})
            ts.append((time.perf_counter() - t) * 1000)
        return statistics.median(ts)

    def _make_slot(self, key, device, compiled, bench, tried, npu_ok):
        q = queue.Queue()
        for _ in range(MODEL_SPECS[key]["pool"]):
            q.put(compiled.create_infer_request())
        return Slot(key=key, device=device, compiled=compiled, requests=q, bench_ms=bench, ema_ms=bench, npu_ok=npu_ok, tried=tried)

    # --------------------------------------------------------------- loading
    def load(self, policy: str | None = None):
        """(Re)build all model slots according to the policy. Blocking."""
        with self._lock:
            if policy:
                self.policy = policy if policy == "auto" else policy.upper()
            self.status = "loading"
            self.notes = []
            self._p(f"[engine] OpenVINO {ov.__version__} | 可用设备: " + ", ".join(f"{d} ({self.device_names.get(d, d)})" for d in self.available))
            if self.has("NPU"):
                self._p("[engine] ✅ 检测到 Intel NPU，优先在 NPU 上运行所有模型")
            else:
                self._p("[engine] ⚠️ 未检测到 NPU（需要 Intel Core Ultra + NPU 驱动），改用 GPU/CPU")

            if self.policy == "auto":
                order = [d for d in DEVICE_ORDER if self.has(d)]
            else:
                order = [self.policy] + [d for d in DEVICE_ORDER if d != self.policy and self.has(d)]
                if not self.has(self.policy):
                    self.notes.append(f"{self.policy} 不可用，已自动回退")
                    order = [d for d in DEVICE_ORDER if self.has(d)]

            new_slots = {}
            total = len(MODEL_SPECS)
            for i, key in enumerate(MODEL_SPECS):
                self._p(f"[engine] ({i+1}/{total}) 加载 {MODEL_SPECS[key]['label']} …")
                tried = {}
                chosen = None
                for dev in order:
                    try:
                        compiled = self._compile(key, dev)
                        bench = self._bench(compiled, key)
                        tried[dev] = f"ok {bench:.1f}ms"
                        chosen = (dev, compiled, bench)
                        break
                    except Exception as e:  # compile failed -> next device
                        msg = str(e).strip().splitlines()[0][:160] if str(e).strip() else type(e).__name__
                        tried[dev] = "fail: " + msg
                        self._p(f"  · {dev} 无法运行 {key}: {msg} → 尝试下一个设备")
                if chosen is None:
                    raise RuntimeError(f"model {key} could not be compiled on any device: {tried}")
                dev, compiled, bench = chosen
                npu_ok = tried.get("NPU", "").startswith("ok") if self.has("NPU") else None
                new_slots[key] = self._make_slot(key, dev, compiled, bench, tried, npu_ok)
                self._p(f"  → {MODEL_SPECS[key]['label']} 使用 {dev}，单次推理 {bench:.1f} ms")
            self.slots = new_slots
            if self.policy == "auto":
                self._balance(initial=True)
            self.status = "ready"
            self._p("[engine] 就绪: " + ", ".join(f"{k}@{s.device}({s.bench_ms:.1f}ms)" for k, s in self.slots.items()))

    def _frame_cost(self, slots=None, device=None):
        slots = slots or self.slots
        cost = {}
        for k, s in slots.items():
            ms = s.ema_ms or s.bench_ms
            cost[s.device] = cost.get(s.device, 0.0) + ms * MODEL_SPECS[k]["per_frame"]
        return cost if device is None else cost.get(device, 0.0)

    def _balance(self, initial=False):
        """Move the heaviest models off the NPU only if the NPU cannot keep up."""
        if not self.has("NPU"):
            return
        alternatives = [d for d in ("GPU", "CPU") if self.has(d)]
        if not alternatives:
            return
        moved = False
        for _ in range(len(self.slots)):
            npu_cost = self._frame_cost(device="NPU")
            if npu_cost <= FRAME_BUDGET_MS:
                break
            npu_models = sorted((s for s in self.slots.values() if s.device == "NPU"),
                                key=lambda s: (s.ema_ms or s.bench_ms) * MODEL_SPECS[s.key]["per_frame"], reverse=True)
            if not npu_models:
                break
            victim = npu_models[0]
            best = None
            for dev in alternatives:
                try:
                    compiled = self._compile(victim.key, dev)
                    bench = self._bench(compiled, victim.key)
                except Exception as e:
                    victim.tried[dev] = f"fail: {e}"
                    continue
                victim.tried[dev] = f"ok {bench:.1f}ms"
                if best is None or bench < best[2]:
                    best = (dev, compiled, bench)
            if best is None:
                break
            dev, compiled, bench = best
            # only move if the destination device would not become the new bottleneck
            dest_cost = self._frame_cost(device=dev) + bench * MODEL_SPECS[victim.key]["per_frame"]
            if dest_cost >= npu_cost:
                break
            self.slots[victim.key] = self._make_slot(victim.key, dev, compiled, bench, victim.tried, victim.npu_ok)
            note = f"NPU 每帧负载 {npu_cost:.1f}ms 超出预算 {FRAME_BUDGET_MS:.0f}ms → {MODEL_SPECS[victim.key]['label']} 改用 {dev}"
            self.notes.append(note)
            self._p("[engine] " + note)
            moved = True
        return moved

    def needs_rebalance(self):
        """Cheap check, called after requests: is the NPU persistently over budget?"""
        if self.policy != "auto" or self.status != "ready" or not self.has("NPU") or self._rebalancing:
            return False
        if time.time() - self.last_rebalance < 5:
            return False
        self.last_rebalance = time.time()
        return self._frame_cost(device="NPU") > FRAME_BUDGET_MS * 1.5

    def rebalance(self):
        """Heavy part (may compile on GPU/CPU) – run it in a worker thread."""
        self._rebalancing = True
        try:
            with self._lock:
                return bool(self._balance())
        finally:
            self._rebalancing = False

    # ------------------------------------------------------------- inference
    def infer(self, key: str, x: np.ndarray):
        slot = self.slots[key]
        req = slot.requests.get()
        try:
            t = time.perf_counter()
            req.infer({0: x})
            ms = (time.perf_counter() - t) * 1000
            outs = {}
            for i, port in enumerate(slot.compiled.outputs):
                name = port.get_any_name()
                outs[name] = np.array(req.get_output_tensor(i).data, dtype=np.float32, copy=True)
        finally:
            slot.requests.put(req)
        slot.calls += 1
        slot.ema_ms = ms if slot.calls < 3 else slot.ema_ms * 0.9 + ms * 0.1
        return outs, ms, slot.device

    # ------------------------------------------------------------------ info
    def info(self):
        models = {}
        for k, spec in MODEL_SPECS.items():
            s = self.slots.get(k)
            models[k] = {
                "label": spec["label"], "file": spec["file"], "size": spec["size"],
                "device": s.device if s else None,
                "bench_ms": round(s.bench_ms, 2) if s else None,
                "ema_ms": round(s.ema_ms, 2) if s else None,
                "npu_ok": s.npu_ok if s else None,
                "tried": s.tried if s else {},
            }
        return {
            "status": self.status,
            "policy": self.policy,
            "openvino": ov.__version__,
            "devices": self.base_devices,
            "device_names": {d.split(".")[0]: n for d, n in self.device_names.items()},
            "npu": self.has("NPU"),
            "models": models,
            "frame_cost": {d: round(v, 2) for d, v in self._frame_cost().items()} if self.slots else {},
            "budget_ms": FRAME_BUDGET_MS,
            "notes": self.notes,
            "progress": self.progress[-12:],
        }
