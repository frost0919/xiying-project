"""
串口 → TDOA Chan 解算 → HTTP JSON，供 web/index.html 轮询。

README 每行 anchor_id:timestamp；多基站属于「同一帧」时往往在几毫秒内连续到达。
本脚本用 batch_gap_ms 内收到的、配置里列出的全部基站时间戳聚成一帧，
再换算成「相对第一个基站」的到达时差（秒）调用 solver.tdoa_chan。

用法（在项目根目录）:
  pip install -r requirements.txt
  copy config\\anchors.example.json config\\anchors.json
  编辑 config\\anchors.json 里的基站 id 与坐标
  python scripts\\pipeline.py --port COM3 --config config\\anchors.json

浏览器打开 web/index.html，轮询 URL 填 http://127.0.0.1:8765/positions
（若用 python -m http.server 托管页面，端口不同不影响，只要填对 pipeline 的端口）。
"""

from __future__ import annotations

import json
import queue
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_scripts = Path(__file__).resolve().parent
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))

from solver import tdoa_chan
from uwb_serial import iter_serial_lines, parse_line

# ---------- 配置 ----------

TIMESTAMP_UNIT_SCALE = {
    "s": 1.0,
    "ms": 1e-3,
    "us": 1e-6,
    "ns": 1e-9,
}


def load_config(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        cfg = json.load(f)
    need = ["anchors", "tag_label", "batch_gap_ms", "timestamp_unit"]
    for k in need:
        if k not in cfg:
            raise KeyError(f"配置缺少字段: {k}")
    anchors = cfg["anchors"]
    if not isinstance(anchors, list) or len(anchors) < 3:
        raise ValueError("anchors 至少需要 3 个基站")
    for a in anchors:
        if "id" not in a or "x" not in a or "y" not in a:
            raise ValueError("每个 anchor 需要 id, x, y")
    if cfg["timestamp_unit"] not in TIMESTAMP_UNIT_SCALE:
        raise ValueError("timestamp_unit 必须是 s / ms / us / ns 之一")
    cfg.setdefault("http_host", "127.0.0.1")
    cfg.setdefault("http_port", 8765)
    return cfg


def timestamps_to_seconds(cfg: Dict[str, Any], batch: Dict[str, float]) -> Dict[str, float]:
    scale = TIMESTAMP_UNIT_SCALE[cfg["timestamp_unit"]]
    return {k: v * scale for k, v in batch.items()}


def try_solve(cfg: Dict[str, Any], batch: Dict[str, float]) -> Optional[Tuple[float, float]]:
    """batch: raw timestamp values as sent on wire (before unit conversion)."""
    ids = [str(a["id"]) for a in cfg["anchors"]]
    ts_sec = timestamps_to_seconds(cfg, batch)
    t_list: List[float] = []
    for i in ids:
        if i not in ts_sec:
            return None
        t_list.append(ts_sec[i])
    t_ref = t_list[0]
    diffs = np.array([t_list[j] - t_ref for j in range(1, len(t_list))], dtype=float)
    arr = np.array([[float(a["x"]), float(a["y"])] for a in cfg["anchors"]], dtype=float)
    pos = tdoa_chan(arr, diffs)
    return float(pos[0]), float(pos[1])


# ---------- HTTP ---------

_state_lock = threading.Lock()
_state: Dict[str, Any] = {
    "vehicles": [],
    "last_error": None,
    "updated": None,
    "raw_batch_count": 0,
}


def _cors_headers(handler: BaseHTTPRequestHandler) -> None:
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
    handler.send_header("Access-Control-Allow-Headers", "*")


def make_handler() -> type:
    class H(BaseHTTPRequestHandler):
        def log_message(self, *args: object) -> None:
            pass

        def do_OPTIONS(self) -> None:
            self.send_response(204)
            _cors_headers(self)
            self.end_headers()

        def do_GET(self) -> None:
            path = self.path.split("?")[0].rstrip("/") or "/"
            if path == "/positions":
                with _state_lock:
                    body = {
                        "vehicles": _state["vehicles"],
                        "last_error": _state["last_error"],
                        "updated": _state["updated"],
                        "raw_batch_count": _state["raw_batch_count"],
                    }
                data = json.dumps(body, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                _cors_headers(self)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_error(404)

    return H


def run_http(host: str, port: int) -> ThreadingHTTPServer:
    srv = ThreadingHTTPServer((host, port), make_handler())
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    return srv


# ---------- 串口消费（按静默间隔聚批）----------


def consumer(
    cfg: Dict[str, Any],
    line_q: "queue.Queue[Optional[str]]",
    gap_s: float,
) -> None:
    pending: Dict[str, float] = {}
    last_activity: Optional[float] = None

    def flush() -> None:
        nonlocal pending, last_activity
        if not pending:
            last_activity = None
            return
        ids_needed = {str(a["id"]) for a in cfg["anchors"]}
        if not ids_needed.issubset(set(pending.keys())):
            with _state_lock:
                _state["last_error"] = "批次不完整，已丢弃（需全部基站各一行）"
            pending = {}
            last_activity = None
            return
        try:
            xy = try_solve(cfg, pending)
        except Exception as e:
            with _state_lock:
                _state["last_error"] = str(e)
            pending = {}
            last_activity = None
            return
        if xy is None:
            pending = {}
            last_activity = None
            return
        x, y = xy
        with _state_lock:
            _state["vehicles"] = [{"id": cfg["tag_label"], "x": x, "y": y}]
            _state["last_error"] = None
            _state["updated"] = time.time()
            _state["raw_batch_count"] = int(_state["raw_batch_count"]) + 1
        pending = {}
        last_activity = None

    while True:
        try:
            raw = line_q.get(timeout=0.05)
        except queue.Empty:
            now = time.monotonic()
            if pending and last_activity is not None and (now - last_activity) >= gap_s:
                flush()
            continue

        if raw is None:
            if pending:
                flush()
            break

        parsed = parse_line(raw)
        if parsed is None:
            continue
        aid, ts = str(parsed[0]), parsed[1]
        now = time.monotonic()
        if last_activity is not None and (now - last_activity) >= gap_s:
            flush()
        pending[aid] = ts
        last_activity = now


def serial_producer(port: str, baud: int, line_q: "queue.Queue[Optional[str]]") -> None:
    try:
        for raw in iter_serial_lines(port, baud):
            line_q.put(raw)
    except OSError as e:
        with _state_lock:
            _state["last_error"] = f"串口错误: {e}"
    finally:
        line_q.put(None)


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="串口 TDOA → HTTP /positions")
    ap.add_argument("--port", "-p", required=True, help="串口，如 COM3")
    ap.add_argument("--baud", "-b", type=int, default=115200)
    ap.add_argument(
        "--config",
        "-c",
        default="config/anchors.json",
        help="基站与参数 JSON，默认 config/anchors.json",
    )
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        print(f"找不到配置文件: {cfg_path}", file=sys.stderr)
        print("可复制 config/anchors.example.json 为 config/anchors.json 后修改。", file=sys.stderr)
        return 1

    cfg = load_config(cfg_path)
    gap_s = float(cfg["batch_gap_ms"]) / 1000.0
    host = str(cfg["http_host"])
    port_http = int(cfg["http_port"])

    line_q: queue.Queue[Optional[str]] = queue.Queue(maxsize=5000)
    t_serial = threading.Thread(
        target=serial_producer, args=(args.port, args.baud, line_q), daemon=True
    )
    t_cons = threading.Thread(target=consumer, args=(cfg, line_q, gap_s), daemon=True)
    t_serial.start()
    t_cons.start()

    run_http(host, port_http)
    print(
        f"HTTP http://{host}:{port_http}/positions  （网页轮询此地址）\n"
        f"串口 {args.port} @ {args.baud}，batch_gap_ms={cfg['batch_gap_ms']}，"
        f"timestamp_unit={cfg['timestamp_unit']}\n"
        "按 Ctrl+C 退出。",
        file=sys.stderr,
    )
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("\n退出。", file=sys.stderr)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
