"""Minimal streaming ``multipart/form-data`` reader (no ``python-multipart`` dependency).

Starlette's ``request.form()`` / FastAPI ``UploadFile`` need the optional ``python-multipart``
package, which is not installed in the target runtimes. Uploads can be gigabytes, so this
reader never buffers a part body: :meth:`MultipartReader.iter_data` yields chunks as they
arrive and only keeps ``len(delimiter)`` bytes of look-behind. Part headers are capped
(:data:`MAX_HEADER_BYTES`) and the number of parts is capped (:data:`MAX_PARTS`).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from email.message import Message

from ..errors import ValidationFailed

__all__ = ["MAX_HEADER_BYTES", "MAX_PARTS", "MultipartReader", "PartInfo", "parse_boundary"]

MAX_HEADER_BYTES = 16 * 1024
MAX_PARTS = 16
_MAX_BOUNDARY = 200


def parse_boundary(content_type: str | None) -> bytes:
    """Extract the boundary from a ``multipart/form-data`` Content-Type header."""
    if not content_type:
        raise ValidationFailed("expected a multipart/form-data request with a 'file' field")
    msg = Message()
    msg["content-type"] = content_type
    if msg.get_content_type() != "multipart/form-data":
        raise ValidationFailed("expected a multipart/form-data request with a 'file' field")
    boundary = msg.get_param("boundary")
    if not isinstance(boundary, str) or not (1 <= len(boundary) <= _MAX_BOUNDARY):
        raise ValidationFailed("multipart boundary missing or invalid")
    try:
        return boundary.encode("ascii")
    except UnicodeEncodeError:
        raise ValidationFailed("multipart boundary missing or invalid") from None


@dataclass(frozen=True)
class PartInfo:
    name: str | None
    filename: str | None
    content_type: str | None


class MultipartReader:
    """Sequential reader: ``while (p := await r.next_part()): async for chunk in r.iter_data(): ...``."""

    def __init__(self, stream: AsyncIterator[bytes], boundary: bytes) -> None:
        self._stream = stream
        self._delim = b"--" + boundary
        self._sep = b"\r\n" + self._delim  # separates a part body from the next delimiter
        self._buf = b""
        self._state = "preamble"  # preamble | headers | data | after | end
        self._eof = False
        self._parts = 0

    async def _fill(self) -> bool:
        """Read one more chunk; False at end of stream."""
        if self._eof:
            return False
        try:
            chunk = await self._stream.__anext__()
        except StopAsyncIteration:
            self._eof = True
            return False
        if chunk:
            self._buf += chunk
        return True

    def _malformed(self, why: str) -> ValidationFailed:
        return ValidationFailed(f"malformed multipart body: {why}")

    async def _need(self, n: int) -> None:
        while len(self._buf) < n:
            if not await self._fill():
                raise self._malformed("unexpected end of body")

    async def _skip_preamble(self) -> None:
        # The first delimiter may appear at offset 0 (no leading CRLF).
        while True:
            idx = self._buf.find(self._delim)
            if idx >= 0:
                self._buf = self._buf[idx + len(self._delim):]
                self._state = "after"
                return
            if len(self._buf) > MAX_HEADER_BYTES:
                raise self._malformed("boundary not found")
            if not await self._fill():
                raise self._malformed("boundary not found")

    async def _after_delimiter(self) -> None:
        await self._need(2)
        head = self._buf[:2]
        if head == b"--":
            self._state = "end"
            self._buf = b""
            return
        # Tolerate transport padding (spaces/tabs) before the CRLF (RFC 2046).
        while True:
            stripped = self._buf.lstrip(b" \t")
            if stripped:
                break
            self._buf = b""
            await self._need(1)
        self._buf = stripped
        await self._need(2)
        if self._buf[:2] != b"\r\n":
            raise self._malformed("bad delimiter line")
        self._buf = self._buf[2:]
        self._state = "headers"

    async def _read_headers(self) -> PartInfo:
        while True:
            idx = self._buf.find(b"\r\n\r\n")
            if idx >= 0:
                break
            if len(self._buf) > MAX_HEADER_BYTES:
                raise self._malformed("part headers too large")
            if not await self._fill():
                raise self._malformed("unexpected end of part headers")
        raw, self._buf = self._buf[:idx], self._buf[idx + 4:]
        if len(raw) > MAX_HEADER_BYTES:
            raise self._malformed("part headers too large")
        msg = Message()
        for line in raw.decode("utf-8", "replace").split("\r\n"):
            if not line.strip():
                continue
            key, sep, value = line.partition(":")
            if not sep:
                raise self._malformed("bad part header")
            msg[key.strip()] = value.strip()
        disp = msg.get("content-disposition")
        if disp is None or msg.get_content_disposition() != "form-data":
            raise self._malformed("part without form-data Content-Disposition")
        name = msg.get_param("name", header="content-disposition")
        filename = msg.get_filename()
        ctype = msg.get("content-type")
        self._state = "data"
        return PartInfo(
            name=name if isinstance(name, str) else None,
            filename=filename if isinstance(filename, str) else None,
            content_type=ctype,
        )

    async def next_part(self) -> PartInfo | None:
        """Advance to the next part (draining an unread body); None after the closing delimiter."""
        if self._state == "data":
            async for _ in self.iter_data():
                pass
        if self._state == "preamble":
            await self._skip_preamble()
        if self._state == "after":
            await self._after_delimiter()
        if self._state == "end":
            return None
        self._parts += 1
        if self._parts > MAX_PARTS:
            raise self._malformed("too many parts")
        return await self._read_headers()

    async def iter_data(self) -> AsyncIterator[bytes]:
        """Yield the current part's body in chunks (never buffering more than one chunk + delimiter)."""
        if self._state != "data":
            return
        keep = len(self._sep) - 1
        while True:
            idx = self._buf.find(self._sep)
            if idx >= 0:
                data, self._buf = self._buf[:idx], self._buf[idx + len(self._sep):]
                self._state = "after"
                if data:
                    yield data
                return
            if len(self._buf) > keep:
                data, self._buf = self._buf[:-keep], self._buf[-keep:]
                yield data
            if not await self._fill():
                raise self._malformed("unexpected end of part body")
