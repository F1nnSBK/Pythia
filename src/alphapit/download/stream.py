"""
Streaming decompression and memory-efficient iterators for molecular structure data.
"""

from __future__ import annotations

import io
import zlib
from typing import AsyncIterator, Iterator


class StreamingGzipDecompressor:
    """
    Progressive decompressor for gzipped HTTP streams without loading
    the entire archive into memory.
    """

    def __init__(self, buffer_size: int = 64 * 1024) -> None:
        self.buffer_size = buffer_size
        # 16 + zlib.MAX_WBITS tells zlib to expect a gzip header
        self._decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)

    def decompress_chunk(self, chunk: bytes) -> bytes:
        """Decompress a single chunk of bytes."""
        if not chunk:
            return b""
        return self._decompressor.decompress(chunk)

    def flush(self) -> bytes:
        """Flush any unconsumed bytes remaining in the decompressor."""
        return self._decompressor.flush()


async def iter_lines_from_byte_stream(
    byte_stream: AsyncIterator[bytes],
    is_gzipped: bool = False,
    encoding: str = "utf-8",
) -> AsyncIterator[str]:
    """
    Asynchronously stream bytes, decompress on-the-fly if gzipped,
    and yield decoded text lines without memory spikes.
    """
    decompressor = StreamingGzipDecompressor() if is_gzipped else None
    leftover = ""

    async for chunk in byte_stream:
        if not chunk:
            continue

        raw_bytes = decompressor.decompress_chunk(chunk) if decompressor else chunk
        if not raw_bytes:
            continue

        text_chunk = raw_bytes.decode(encoding, errors="replace")
        combined = leftover + text_chunk
        lines = combined.splitlines(keepends=True)

        if lines:
            if lines[-1].endswith(("\n", "\r")):
                leftover = ""
                for line in lines:
                    yield line.rstrip("\r\n")
            else:
                leftover = lines[-1]
                for line in lines[:-1]:
                    yield line.rstrip("\r\n")

    if decompressor:
        final_bytes = decompressor.flush()
        if final_bytes:
            final_text = final_bytes.decode(encoding, errors="replace")
            leftover += final_text

    if leftover:
        yield leftover.rstrip("\r\n")


def iter_lines_from_bytes(
    raw_data: bytes,
    is_gzipped: bool = False,
    encoding: str = "utf-8",
) -> Iterator[str]:
    """
    Synchronously iterate lines from a bytes payload, handling gzip if present.
    """
    if is_gzipped:
        decompressor = StreamingGzipDecompressor()
        decompressed = decompressor.decompress_chunk(raw_data) + decompressor.flush()
    else:
        decompressed = raw_data

    stream = io.StringIO(decompressed.decode(encoding, errors="replace"))
    for line in stream:
        yield line.rstrip("\r\n")
