from __future__ import annotations

import json
import socket
import struct
from dataclasses import dataclass
from typing import Any


FRAME_PREFIX = struct.Struct("!I")
HEADER_PREFIX = struct.Struct("!I")
MAX_FRAME_BYTES = 1024 * 1024 * 1024


@dataclass
class Message:
    msg_type: str
    metadata: dict[str, Any]
    payload: bytes = b""
    raw_bytes: int = 0


def _recv_exact(sock: socket.socket, nbytes: int) -> bytes:
    chunks: list[bytes] = []
    remaining = nbytes
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise EOFError("socket closed while receiving frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_message(
    sock: socket.socket,
    msg_type: str,
    metadata: dict[str, Any] | None = None,
    payload: bytes = b"",
) -> int:
    header = {
        "type": msg_type,
        "metadata": metadata or {},
        "payload_size": len(payload),
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    body = HEADER_PREFIX.pack(len(header_bytes)) + header_bytes + payload
    frame = FRAME_PREFIX.pack(len(body)) + body
    sock.sendall(frame)
    return len(frame)


def recv_message(sock: socket.socket) -> Message:
    frame_size = FRAME_PREFIX.unpack(_recv_exact(sock, FRAME_PREFIX.size))[0]
    if frame_size > MAX_FRAME_BYTES:
        raise ValueError(f"frame too large: {frame_size} bytes")
    body = _recv_exact(sock, frame_size)
    if len(body) < HEADER_PREFIX.size:
        raise ValueError("frame missing header prefix")
    header_size = HEADER_PREFIX.unpack(body[: HEADER_PREFIX.size])[0]
    header_start = HEADER_PREFIX.size
    header_end = header_start + header_size
    header = json.loads(body[header_start:header_end].decode("utf-8"))
    payload = body[header_end:]
    expected = int(header.get("payload_size", 0))
    if expected != len(payload):
        raise ValueError(f"payload size mismatch: expected {expected}, got {len(payload)}")
    return Message(
        msg_type=header["type"],
        metadata=header.get("metadata", {}),
        payload=payload,
        raw_bytes=FRAME_PREFIX.size + frame_size,
    )
