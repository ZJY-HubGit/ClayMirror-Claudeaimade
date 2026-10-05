"""
ClayMirror local server
  * serves the web app (http://localhost:8848, and https://<LAN-IP>:8849 for phones)
  * runs every vision model on the Intel NPU through OpenVINO (GPU/CPU fallback)

Usage:  python server/server.py [--port 8848] [--device auto|NPU|GPU|CPU] [--no-browser]
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import mimetypes
import os
import socket
import ssl
import struct
import sys
import threading
import time
import webbrowser

# Windows consoles (cp950 / cp936) cannot print every character we log.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB = os.path.join(ROOT, "web")
MODELS = os.path.join(WEB, "models")
CACHE = os.path.join(ROOT, ".ov_cache")
CERTS = os.path.join(ROOT, "server", "certs")

try:
    import numpy as np
    from aiohttp import web, WSMsgType
except ImportError as e:  # pragma: no cover
    print("缺少依赖 / missing dependency:", e)
    print("请先运行 start.bat（或 pip install -r requirements.txt）")
    sys.exit(1)

from npu_engine import NPUEngine, MODEL_SPECS  # noqa: E402

for ext, mt in {".js": "text/javascript", ".mjs": "text/javascript", ".wasm": "application/wasm",
                ".onnx": "application/octet-stream", ".css": "text/css", ".html": "text/html",
                ".svg": "image/svg+xml", ".json": "application/json", ".png": "image/png"}.items():
    mimetypes.add_type(mt, ext)

ENGINE: NPUEngine | None = None
ENGINE_ERROR: str | None = None
CLIENTS: set = set()
STATS = {"requests": 0, "jobs": 0, "started": time.time()}
ARGS = None


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- networking
def lan_ips():
    ips = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ips.append(s.getsockname()[0])
        s.close()
    except Exception:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if ip not in ips:
                ips.append(ip)
    except Exception:
        pass
    return [ip for ip in ips if not ip.startswith(("127.", "169.254."))]


def ensure_cert(ips):
    """Self-signed certificate so phone browsers allow camera access (needs HTTPS)."""
    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID
        import datetime
        import ipaddress
    except ImportError:
        return None
    os.makedirs(CERTS, exist_ok=True)
    crt, key = os.path.join(CERTS, "cert.pem"), os.path.join(CERTS, "key.pem")
    stamp = os.path.join(CERTS, "ips.txt")
    want = ",".join(sorted(ips))
    if os.path.exists(crt) and os.path.exists(key) and os.path.exists(stamp) and open(stamp).read() == want:
        return crt, key
    k = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "ClayMirror Local")])
    alt = [x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
    for ip in ips:
        try:
            alt.append(x509.IPAddress(ipaddress.ip_address(ip)))
        except ValueError:
            pass
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(k.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(now - datetime.timedelta(days=1))
            .not_valid_after(now + datetime.timedelta(days=825))
            .add_extension(x509.SubjectAlternativeName(alt), critical=False)
            .sign(k, hashes.SHA256()))
    with open(key, "wb") as f:
        f.write(k.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL, serialization.NoEncryption()))
    with open(crt, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    with open(stamp, "w") as f:
        f.write(want)
    return crt, key


# ------------------------------------------------------------------ protocol
def pack(header: dict, buffers: list[bytes]) -> bytes:
    hj = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (4 - (4 + len(hj)) % 4) % 4
    hj += b" " * pad
    return struct.pack("<I", len(hj)) + hj + b"".join(buffers)


def unpack(data: bytes):
    (hl,) = struct.unpack_from("<I", data, 0)
    header = json.loads(data[4:4 + hl].decode("utf-8"))
    return header, memoryview(data)[4 + hl:]


def decode_input(job, payload):
    size = MODEL_SPECS[job["m"]]["size"]
    raw = payload[job["off"]: job["off"] + job["len"]]
    if job.get("fmt") == "jpeg":
        from PIL import Image
        img = Image.open(io.BytesIO(bytes(raw))).convert("RGB")
        if img.size != (size, size):
            img = img.resize((size, size))
        arr = np.asarray(img, dtype=np.uint8)
    else:
        arr = np.frombuffer(raw, dtype=np.uint8)
        if arr.size == size * size * 4:
            arr = arr.reshape(size, size, 4)[:, :, :3]
        arr = arr.reshape(size, size, 3)
    return np.ascontiguousarray(arr[None])


def run_job(job, payload):
    x = decode_input(job, payload)
    outs, ms, dev = ENGINE.infer(job["m"], x)
    sp = job.get("sparse")
    if sp:
        scores = outs[sp["s"]].reshape(-1)
        boxes = outs[sp["b"]].reshape(scores.shape[0], -1)
        idx = np.nonzero(scores > sp.get("min", 0.0))[0].astype(np.int32)
        if idx.size > 64:  # keep the 64 best candidates
            idx = idx[np.argsort(-scores[idx])[:64]].astype(np.int32)
        outs = {"idx": idx, "scores": scores[idx].astype(np.float32), "boxes": boxes[idx].astype(np.float32)}
    return outs, ms, dev


async def handle_infer(data: bytes) -> bytes:
    t0 = time.perf_counter()
    header, payload = unpack(data)
    loop = asyncio.get_running_loop()
    futs = [loop.run_in_executor(ENGINE.pool, run_job, job, payload) for job in header["jobs"]]
    results = await asyncio.gather(*futs, return_exceptions=True)
    jobs_h, bufs, off = [], [], 0
    for res in results:
        if isinstance(res, Exception):
            jobs_h.append({"err": str(res)[:200]})
            continue
        outs, ms, dev = res
        oh = []
        for name, arr in outs.items():
            b = arr.tobytes()
            oh.append({"n": name, "t": "i32" if arr.dtype == np.int32 else "f32", "s": list(arr.shape), "off": off, "len": len(b)})
            bufs.append(b)
            off += len(b)
        jobs_h.append({"dev": dev, "ms": round(ms, 2), "outs": oh})
    STATS["requests"] += 1
    STATS["jobs"] += len(results)
    return pack({"t": "result", "id": header.get("id"), "jobs": jobs_h, "srv": round((time.perf_counter() - t0) * 1000, 2)}, bufs)


def public_info():
    info = ENGINE.info() if ENGINE else {"status": "error" if ENGINE_ERROR else "init", "models": {}, "devices": [], "npu": False}
    if ENGINE_ERROR:
        info["status"] = "error"
        info["error"] = ENGINE_ERROR
    ips = lan_ips()
    info["lan"] = [f"https://{ip}:{ARGS.port + 1}/" for ip in ips] if ARGS.https_ok else []
    info["server"] = "ClayMirror/1.0"
    info["stats"] = STATS
    return info


async def broadcast_info():
    msg = json.dumps({"t": "info", "info": public_info()})
    for ws in list(CLIENTS):
        try:
            await ws.send_str(msg)
        except Exception:
            CLIENTS.discard(ws)


async def rebalance_async():
    moved = await asyncio.get_running_loop().run_in_executor(None, ENGINE.rebalance)
    if moved:
        await broadcast_info()


# ------------------------------------------------------------------- routes
async def ws_handler(request):
    ws = web.WebSocketResponse(max_msg_size=64 * 1024 * 1024, heartbeat=20)
    await ws.prepare(request)
    CLIENTS.add(ws)
    await ws.send_str(json.dumps({"t": "info", "info": public_info()}))
    try:
        async for msg in ws:
            if msg.type == WSMsgType.BINARY:
                if not ENGINE or ENGINE.status != "ready" and not ENGINE.slots:
                    await ws.send_bytes(pack({"t": "result", "id": unpack(msg.data)[0].get("id"), "jobs": [], "err": "engine not ready"}, []))
                    continue
                try:
                    out = await handle_infer(msg.data)
                except Exception as e:  # never kill the socket because of one bad frame
                    out = pack({"t": "result", "id": None, "jobs": [], "err": str(e)[:300]}, [])
                await ws.send_bytes(out)
                if ENGINE.needs_rebalance():
                    asyncio.get_running_loop().create_task(rebalance_async())
            elif msg.type == WSMsgType.TEXT:
                cmd = json.loads(msg.data)
                if cmd.get("t") == "policy":
                    pol = cmd.get("v", "auto")
                    log(f"[server] 切换计算设备策略 → {pol}")
                    await asyncio.get_running_loop().run_in_executor(None, ENGINE.load, pol)
                    await broadcast_info()
                elif cmd.get("t") == "info":
                    await ws.send_str(json.dumps({"t": "info", "info": public_info()}))
            elif msg.type == WSMsgType.ERROR:
                break
    finally:
        CLIENTS.discard(ws)
    return ws


async def info_handler(request):
    return web.json_response(public_info(), headers={"Cache-Control": "no-store"})


def static_headers(path):
    h = {"Cross-Origin-Opener-Policy": "same-origin", "Cross-Origin-Embedder-Policy": "require-corp",
         "Cross-Origin-Resource-Policy": "same-origin"}
    if path.endswith((".onnx", ".wasm")):
        h["Cache-Control"] = "public, max-age=86400"
    else:
        h["Cache-Control"] = "no-cache"
    return h


async def static_handler(request):
    rel = request.match_info.get("path", "") or "index.html"
    full = os.path.normpath(os.path.join(WEB, rel))
    # big files may ship pre-compressed (foo.wasm.gz); aiohttp serves them with Content-Encoding
    if not full.startswith(WEB + os.sep) or not (os.path.isfile(full) or os.path.isfile(full + ".gz")):
        raise web.HTTPNotFound()
    ctype = mimetypes.guess_type(full)[0] or "application/octet-stream"
    resp = web.FileResponse(full, headers=static_headers(full))
    resp.content_type = ctype
    return resp


def make_app():
    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.router.add_get("/ws", ws_handler)
    app.router.add_get("/api/info", info_handler)
    app.router.add_get("/", static_handler)
    app.router.add_get("/{path:.*}", static_handler)
    return app


def load_engine():
    global ENGINE, ENGINE_ERROR
    try:
        ENGINE = NPUEngine(MODELS, CACHE, policy=ARGS.device, log=log)
        ENGINE.load()
        log("")
        log("=" * 64)
        for k, m in ENGINE.info()["models"].items():
            log(f"  {m['label']:<6} → {m['device']:<4} {m['bench_ms']:>6.1f} ms   ({m['file']})")
        log("=" * 64)
    except Exception as e:
        ENGINE_ERROR = f"{type(e).__name__}: {e}"
        log("[engine] ❌ 推理引擎启动失败:", ENGINE_ERROR)
        log("        浏览器端会自动改用内置的 WebNN/WebGPU/WASM 引擎。")


async def main_async():
    app = make_app()
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    host = "0.0.0.0" if not ARGS.local_only else "127.0.0.1"
    await web.TCPSite(runner, host, ARGS.port).start()
    ARGS.https_ok = False
    if not ARGS.local_only:
        try:
            certs = ensure_cert(lan_ips())
            if certs:
                ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
                ctx.load_cert_chain(*certs)
                await web.TCPSite(runner, host, ARGS.port + 1, ssl_context=ctx).start()
                ARGS.https_ok = True
        except Exception as e:
            log("[server] HTTPS 未启用:", e)
    url = f"http://localhost:{ARGS.port}/"
    log("")
    log(f"  🧱 橡皮泥魔镜 ClayMirror 已启动 → {url}")
    if ARGS.https_ok:
        for ip in lan_ips():
            log(f"  📱 手机（同一 Wi-Fi）打开 → https://{ip}:{ARGS.port + 1}/  （首次需点“高级 → 继续访问”）")
    log("  按 Ctrl+C 退出")
    log("")
    threading.Thread(target=load_engine, daemon=True).start()
    if not ARGS.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    last_status = None
    while True:
        await asyncio.sleep(0.5)
        st = (ENGINE.status if ENGINE else None, ENGINE_ERROR)
        if st != last_status:
            last_status = st
            await broadcast_info()


def main():
    global ARGS
    p = argparse.ArgumentParser(description="ClayMirror NPU server")
    p.add_argument("--port", type=int, default=int(os.environ.get("CLAY_PORT", 8848)))
    p.add_argument("--device", default=os.environ.get("CLAY_DEVICE", "auto"), help="auto | NPU | GPU | CPU")
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--local-only", action="store_true", help="only listen on 127.0.0.1 (no phone access)")
    ARGS = p.parse_args()
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        pass
    except OSError as e:
        log(f"[server] 端口 {ARGS.port} 无法使用: {e}")
        log("        可能已经有一个 ClayMirror 在运行，或换一个端口: python server/server.py --port 8850")


if __name__ == "__main__":
    main()
