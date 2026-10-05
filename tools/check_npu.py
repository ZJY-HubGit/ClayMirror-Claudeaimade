"""
NPU self-check: lists OpenVINO devices and compiles + benchmarks every
ClayMirror model on each device (NPU, GPU, CPU).

    .venv\\Scripts\\python tools\\check_npu.py          (Windows)
    .venv/bin/python tools/check_npu.py              (Linux)
"""
import os
import statistics
import sys
import time

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import openvino as ov

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "server"))
from npu_engine import MODEL_SPECS  # noqa: E402

core = ov.Core()
print(f"OpenVINO {ov.__version__}")
devices = core.available_devices
for d in devices:
    try:
        name = core.get_property(d, "FULL_DEVICE_NAME")
    except Exception:
        name = "?"
    print(f"  {d:6s} {name}")
if not any(d.startswith("NPU") for d in devices):
    print("\n⚠️  没有找到 NPU。需要 Intel Core Ultra 处理器 + Windows 11 的 Intel NPU 驱动")
    print("   （设备管理器 → 神经处理器 → Intel(R) AI Boost）。")

targets = [d for d in ("NPU", "GPU", "CPU") if any(x.startswith(d) for x in devices)]
print()
print(f"{'model':32s}" + "".join(f"{d + ' 推理/编译':>16s}" for d in targets))
for key, spec in MODEL_SPECS.items():
    path = os.path.join(ROOT, "web", "models", spec["file"])
    row = f"{spec['file']:32s}"
    for dev in targets:
        try:
            t0 = time.time()
            cm = core.compile_model(core.read_model(path), dev, {"PERFORMANCE_HINT": "LATENCY"})
            ct = time.time() - t0
            req = cm.create_infer_request()
            x = np.random.rand(1, spec["size"], spec["size"], 3).astype(np.float32)
            for _ in range(3):
                req.infer({0: x})
            ts = []
            for _ in range(20):
                t = time.perf_counter()
                req.infer({0: x})
                ts.append((time.perf_counter() - t) * 1000)
            row += f"{statistics.median(ts):7.2f}ms/{ct:4.1f}s".rjust(16)
        except Exception as e:
            row += f"{'FAIL':>16s}"
            print(f"   {dev} {key}: {str(e).splitlines()[0][:150]}")
    print(row)
print("\n（NPU 第一次编译较慢；ClayMirror 运行时会把编译结果缓存到 .ov_cache，之后启动很快）")
