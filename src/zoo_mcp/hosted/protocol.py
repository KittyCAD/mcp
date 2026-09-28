"""Length-delimited JSON on anonymous parent/worker pipes; never pickle."""

import asyncio
import json
import struct
from typing import BinaryIO

MAX_MESSAGE_BYTES = 32 * 1024 * 1024


def encode(value: dict) -> bytes:
    body = json.dumps(value, separators=(",", ":")).encode()
    if len(body) > MAX_MESSAGE_BYTES:
        raise ValueError("Worker message is too large")
    return struct.pack("!I", len(body)) + body


def message_size(header: bytes) -> int:
    (size,) = struct.unpack("!I", header)
    if size > MAX_MESSAGE_BYTES:
        raise ValueError("Worker message is too large")
    return size


async def receive(reader: asyncio.StreamReader) -> dict:
    size = message_size(await reader.readexactly(4))
    return json.loads(await reader.readexactly(size))


def read_message(reader: BinaryIO) -> dict | None:
    header = reader.read(4)
    if not header:
        return None
    size = message_size(header)
    body = reader.read(size)
    if len(body) != size:
        raise EOFError("Incomplete worker message")
    return json.loads(body)
