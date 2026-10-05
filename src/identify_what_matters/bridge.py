"""Localhost WebSocket bridge between the trend pipeline and a logged-in instagram.com tab.

Instagram's CSP blocks fetch() to localhost but allows `ws://localhost:*`, so the collector pasted into the
DevTools console (browser_collector.js) connects here, receives the probe plan and streams back the capture.
Standard library only (RFC 6455 subset: unfragmented text frames, close, ping).
"""

from __future__ import annotations

import base64
import hashlib
import json
import socket
import struct
from pathlib import Path

GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def _recv_exact(connection: socket.socket, size: int) -> bytes:
    data = b""
    while len(data) < size:
        chunk = connection.recv(size - len(data))
        if not chunk:
            raise ConnectionError("socket closed")
        data += chunk
    return data


def _read_message(connection: socket.socket) -> str | None:
    parts = []
    while True:
        first, second = _recv_exact(connection, 2)
        opcode, length = first & 0x0F, second & 0x7F
        if length == 126:
            length = struct.unpack(">H", _recv_exact(connection, 2))[0]
        elif length == 127:
            length = struct.unpack(">Q", _recv_exact(connection, 8))[0]
        mask = _recv_exact(connection, 4) if second & 0x80 else b"\0\0\0\0"
        payload = bytes(b ^ mask[i % 4] for i, b in enumerate(_recv_exact(connection, length)))
        if opcode == 0x8:
            return None
        if opcode == 0x9:
            _send(connection, payload, opcode=0xA)
            continue
        parts.append(payload)
        if first & 0x80:
            return b"".join(parts).decode("utf-8")


def _send(connection: socket.socket, payload: bytes | str, opcode: int = 0x1) -> None:
    data = payload.encode("utf-8") if isinstance(payload, str) else payload
    header = bytes([0x80 | opcode])
    if len(data) < 126:
        header += bytes([len(data)])
    elif len(data) < 65536:
        header += bytes([126]) + struct.pack(">H", len(data))
    else:
        header += bytes([127]) + struct.pack(">Q", len(data))
    connection.sendall(header + data)


def _handshake(connection: socket.socket) -> None:
    request = b""
    while b"\r\n\r\n" not in request:
        request += connection.recv(4096)
    headers = dict(line.split(": ", 1) for line in request.decode().split("\r\n")[1:] if ": " in line)
    origin = headers.get("Origin", "")
    if origin and not origin.endswith("instagram.com"):
        raise PermissionError(f"refusing origin {origin}")
    accept = base64.b64encode(hashlib.sha1((headers["Sec-WebSocket-Key"] + GUID).encode()).digest()).decode()
    connection.sendall(("HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                        f"Sec-WebSocket-Accept: {accept}\r\n\r\n").encode())


class BrowserBridge:
    """RPC to browser_proxy.js: Python issues each request, the logged-in tab performs the fetch."""

    def __init__(self, port: int = 8765, wait_seconds: float = 1800):
        self.server = socket.create_server(("127.0.0.1", port))
        self.server.settimeout(wait_seconds)
        self.port, self.sequence, self.connection = port, 0, None

    def connect(self) -> None:
        print(f"[bridge] waiting on ws://localhost:{self.port}: paste src/identify_what_matters/browser_proxy.js into the "
              "DevTools console of a logged-in instagram.com tab", flush=True)
        while self.connection is None:
            connection, _ = self.server.accept()
            try:
                _handshake(connection)
                connection.settimeout(120)
                self.connection = connection
            except (PermissionError, KeyError, ConnectionError) as error:
                print(f"[bridge] rejected connection: {error}", flush=True)
                connection.close()
        print("[bridge] browser connected", flush=True)

    def call(self, kind: str, arg: str) -> tuple[int, dict | None]:
        if self.connection is None:
            self.connect()
        self.sequence += 1
        _send(self.connection, json.dumps({"id": self.sequence, "kind": kind, "arg": arg}))
        while True:
            message = _read_message(self.connection)
            if message is None:
                raise ConnectionError("browser closed the bridge")
            reply = json.loads(message)
            if reply.get("id") == self.sequence:
                return reply.get("status", 0), reply.get("body")

    def close(self, summary: str = "") -> None:
        if self.connection is not None:
            try:
                _send(self.connection, json.dumps({"kind": "done", "arg": summary}))
            finally:
                self.connection.close()
        self.server.close()
