"""
串口调试：只打印解析后的 anchor / timestamp，不做 TDOA。

完整流程（串口 → 解算 → 网页）请使用：
  python scripts/pipeline.py --port COM3 --config config/anchors.json
"""

from __future__ import annotations

import argparse
import sys

from uwb_serial import iter_serial_lines, list_ports, parse_line


def main() -> int:
    ap = argparse.ArgumentParser(description="读取 UWB 串口数据行 (anchor_id:timestamp)")
    ap.add_argument("--port", "-p", help="串口名，如 COM3 或 /dev/ttyUSB0")
    ap.add_argument("--baud", "-b", type=int, default=115200, help="波特率，默认 115200")
    ap.add_argument("--list", "-l", action="store_true", help="列出可用串口后退出")
    args = ap.parse_args()

    if args.list:
        list_ports()
        return 0

    if not args.port:
        print("请指定 --port，或用 --list 查看串口。", file=sys.stderr)
        return 1

    print(f"打开 {args.port} @ {args.baud}，按 Ctrl+C 结束。", file=sys.stderr)
    try:
        for raw in iter_serial_lines(args.port, args.baud):
            parsed = parse_line(raw)
            if parsed is None:
                print(f"[skip] {raw!r}", flush=True)
                continue
            aid, ts = parsed
            print(f"anchor={aid!r}\ttimestamp={ts}", flush=True)
    except KeyboardInterrupt:
        print("\n已停止。", file=sys.stderr)
        return 0
    except OSError as e:
        print(f"串口错误: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
