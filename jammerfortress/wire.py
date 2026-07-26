"""
SUPPORT - WIRE: stdlib WebSocket (RFC 6455). Handshake, frame codec, and a
small server-side connection wrapper. No external packages.
"""
import base64
import hashlib
import os
import socket
import struct

GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

OP_TEXT = 0x1
OP_BIN = 0x2
OP_CLOSE = 0x8
OP_PING = 0x9
OP_PONG = 0xA


def accept_key(client_key):
    return base64.b64encode(hashlib.sha1((client_key + GUID).encode()).digest()).decode()


def handshake_response(client_key):
    return ("HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Accept: {accept_key(client_key)}\r\n\r\n").encode()


def encode_frame(payload, opcode=OP_TEXT, fin=True, mask=None):
    """Encode a frame. Servers send unmasked (mask=None); clients must mask."""
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    b0 = (0x80 if fin else 0) | (opcode & 0x0F)
    length = len(payload)
    mask_bit = 0x80 if mask is not None else 0
    if length < 126:
        header = struct.pack("!BB", b0, mask_bit | length)
    elif length < 65536:
        header = struct.pack("!BBH", b0, mask_bit | 126, length)
    else:
        header = struct.pack("!BBQ", b0, mask_bit | 127, length)
    if mask is not None:
        if len(mask) != 4:
            raise ValueError("mask must be 4 bytes")
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        return header + mask + masked
    return header + payload


def encode_client_frame(payload, opcode=OP_TEXT):
    return encode_frame(payload, opcode=opcode, mask=os.urandom(4))


def decode_frame(buf):
    """Decode one frame from buf. Returns (opcode, payload, consumed).
    (None, None, 0) when the buffer does not yet contain a complete frame."""
    if len(buf) < 2:
        return None, None, 0
    b0, b1 = buf[0], buf[1]
    opcode = b0 & 0x0F
    masked = bool(b1 & 0x80)
    length = b1 & 0x7F
    idx = 2
    if length == 126:
        if len(buf) < 4:
            return None, None, 0
        length = struct.unpack("!H", buf[2:4])[0]
        idx = 4
    elif length == 127:
        if len(buf) < 10:
            return None, None, 0
        length = struct.unpack("!Q", buf[2:10])[0]
        idx = 10
    mask = None
    if masked:
        if len(buf) < idx + 4:
            return None, None, 0
        mask = buf[idx:idx + 4]
        idx += 4
    if len(buf) < idx + length:
        return None, None, 0
    payload = bytearray(buf[idx:idx + length])
    if masked:
        for i in range(length):
            payload[i] ^= mask[i % 4]
    return opcode, bytes(payload), idx + length


def encode_close(code=1000, reason=b""):
    return encode_frame(struct.pack("!H", code) + reason, opcode=OP_CLOSE)


class WebSocket:
    """Server-side connection over an accepted socket (after handshake)."""

    def __init__(self, sock):
        self.sock = sock
        self.buf = b""
        self.open = True

    def send_text(self, text):
        if not self.open:
            raise RuntimeError("websocket closed")
        self.sock.sendall(encode_frame(text, opcode=OP_TEXT))

    def recv(self, timeout=None):
        """Return (opcode, payload); ("timeout", None) if no complete frame
        arrived within timeout; (None, None) on close."""
        self.sock.settimeout(timeout)
        while True:
            op, payload, used = decode_frame(self.buf)
            if used:
                self.buf = self.buf[used:]
                if op == OP_PING:
                    self.sock.sendall(encode_frame(payload, opcode=OP_PONG))
                    continue
                if op == OP_CLOSE:
                    self.close()
                    return None, None
                return op, payload
            try:
                chunk = self.sock.recv(65536)
            except socket.timeout:
                return "timeout", None
            except OSError:
                self.close()
                return None, None
            if not chunk:
                self.close()
                return None, None
            self.buf += chunk

    def close(self, code=1000):
        if self.open:
            self.open = False
            try:
                self.sock.sendall(encode_close(code))
            except OSError:
                pass
            try:
                self.sock.close()
            except OSError:
                pass
