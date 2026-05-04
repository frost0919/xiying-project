"""
UWB 串口字节流：按行解析 anchor_id:timestamp（与 README 一致）。
"""

from __future__ import annotations

import sys
from typing import Iterator, Optional, Tuple


def parse_line(line: str) -> Optional[Tuple[str, float]]:
    s = line.strip()
    if not s or ":" not in s:
        return None
    aid, rest = s.split(":", 1)
    aid = aid.strip()
    if not aid:
        return None
    try:
        ts = float(rest.strip())
    except ValueError:
        return None
    return (aid, ts)


def iter_serial_lines(port: str, baud: int = 115200) -> Iterator[str]:
    import serial

    with serial.Serial(port, baudrate=baud, timeout=0.5) as ser:
        buf = b""
        while True:
            chunk = ser.read(256)
            if chunk:
                buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                try:
                    yield line.decode("utf-8", errors="replace")
                except Exception:
                    continue
            if not chunk and not buf:
                continue


def list_ports() -> None:
    from serial.tools import list_ports

    found = list(list_ports.comports())
    if not found:
        print("未发现串口设备。", file=sys.stderr)
        return
    for p in found:
        print(f"{p.device}\t{p.description}")
