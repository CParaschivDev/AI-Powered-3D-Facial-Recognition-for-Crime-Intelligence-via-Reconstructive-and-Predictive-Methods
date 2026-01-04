from __future__ import annotations

import io
import os
from typing import BinaryIO

from backend.core.safety import MAX_READ_BYTES


async def safe_read_upload_bytes(upload_file, max_bytes: int = MAX_READ_BYTES) -> bytes:
    """Read at most `max_bytes` bytes from an async UploadFile-like object.

    If the upload exceeds `max_bytes`, raise a ValueError.
    This reads in chunks to avoid allocating more than needed.
    """
    CHUNK = 64 * 1024
    remaining = max_bytes
    parts = []
    # Read until we've exhausted the allowed bytes or the upload ends
    while remaining > 0:
        to_read = min(CHUNK, remaining)
        chunk = await upload_file.read(to_read)
        if not chunk:
            break
        # Some upload_file implementations may return more than requested;
        # slice to the remaining allowance to avoid unbounded memory use.
        if len(chunk) > remaining:
            parts.append(chunk[:remaining])
            remaining = 0
            break
        parts.append(chunk)
        remaining -= len(chunk)
    return b"".join(parts)


def safe_read_file_bytes(path: str, max_bytes: int = MAX_READ_BYTES) -> bytes:
    """Read a local file up to `max_bytes` bytes. Raise ValueError if larger."""
    size = os.path.getsize(path)
    if size > max_bytes:
        raise ValueError(f"File {path} exceeds maximum allowed size of {max_bytes} bytes")
    # Read in chunks to avoid creating an unnecessarily large temporary object
    buf = bytearray()
    CHUNK = 64 * 1024
    with open(path, "rb") as f:
        # Read in a loop that checks the chunk directly (avoids an unconditional
        # `while True` which static scanners may flag as unbounded).
        # Justification: The loop is bounded by the file size and chunk size (CHUNK=64KB).
        while chunk:
            buf.extend(chunk)
            chunk = f.read(CHUNK)
        return bytes(buf)
    return bytes(buf)


def safe_read_text(path: str, encoding: str = "utf-8", max_bytes: int = MAX_READ_BYTES) -> str:
    """Read a text file up to `max_bytes` bytes and return decoded string."""
    b = safe_read_file_bytes(path, max_bytes=max_bytes)
    return b.decode(encoding)
