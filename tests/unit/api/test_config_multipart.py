"""Settings loading and the streaming multipart reader."""

from __future__ import annotations

import pytest

from autotune_studio.api.multipart import MultipartReader, parse_boundary
from autotune_studio.core.config import StudioSettings, load_settings
from autotune_studio.errors import ValidationFailed


# --------------------------------------------------------------------------- settings
def test_settings_defaults() -> None:
    s = load_settings({})
    assert s == StudioSettings()
    assert s.port == 7860 and s.upload_max_bytes == 2 * 1024**3 and s.cors_enabled is False
    assert s.ws_ping_interval == 20.0 and s.host == "127.0.0.1"


def test_settings_from_env() -> None:
    s = load_settings(
        {
            "AUTOTUNE_PORT": "9000",
            "AUTOTUNE_HOST": "0.0.0.0",
            "AUTOTUNE_UPLOAD_MAX_BYTES": "1024",
            "AUTOTUNE_LOG_LEVEL": "DEBUG",
            "AUTOTUNE_CORS_ENABLED": "yes",
            "AUTOTUNE_CORS_ORIGINS": "https://a.example/, https://b.example",
            "AUTOTUNE_WS_PING_INTERVAL": "5.5",
            "AUTOTUNE_REPLAY_SIZE": "10",
            "UNRELATED": "x",
        }
    )
    assert (s.port, s.host, s.upload_max_bytes, s.log_level) == (9000, "0.0.0.0", 1024, "debug")
    assert s.cors_enabled and s.cors_origins == ["https://a.example", "https://b.example"]
    assert s.ws_ping_interval == 5.5 and s.replay_size == 10
    assert load_settings({}, port=1).port == 1


@pytest.mark.parametrize(
    ("var", "value"),
    [("AUTOTUNE_PORT", "abc"), ("AUTOTUNE_PORT", "70000"), ("AUTOTUNE_CORS_ENABLED", "maybe"),
     ("AUTOTUNE_LOG_LEVEL", "loud"), ("AUTOTUNE_WS_PING_INTERVAL", "0")],
)
def test_settings_invalid(var: str, value: str) -> None:
    with pytest.raises(ValueError, match=var):
        load_settings({var: value})


# --------------------------------------------------------------------------- multipart
async def _chunks(data: bytes, size: int):
    for i in range(0, len(data), size):
        yield data[i:i + size]


def _body(boundary: str, parts: list[tuple[str, str | None, bytes]]) -> bytes:
    out = b""
    for name, filename, content in parts:
        disp = f'form-data; name="{name}"' + (f'; filename="{filename}"' if filename else "")
        out += f"--{boundary}\r\nContent-Disposition: {disp}\r\n\r\n".encode() + content + b"\r\n"
    return out + f"--{boundary}--\r\n".encode()


async def _read_all(data: bytes, boundary: str, size: int):
    r = MultipartReader(_chunks(data, size).__aiter__(), boundary.encode())
    out = []
    while (p := await r.next_part()) is not None:
        buf = b""
        async for c in r.iter_data():
            buf += c
        out.append((p.name, p.filename, buf))
    return out


@pytest.mark.parametrize("size", [1, 2, 3, 7, 16, 64, 100000])
async def test_multipart_any_chunking(size: int) -> None:
    b = "----WebKitFormBoundaryABC123"
    tricky = b"line1\r\n------WebKitFormBoundaryABC12 not quite\r\n--\r\nend\r\n"
    parts = [("note", None, b"hello"), ("file", "my data.csv", tricky), ("empty", None, b"")]
    got = await _read_all(_body(b, parts), b, size)
    assert got == parts


async def test_multipart_skips_unread_part() -> None:
    b = "xyz"
    data = _body(b, [("a", None, b"x" * 1000), ("file", "f.csv", b"1,2\n")])
    r = MultipartReader(_chunks(data, 10).__aiter__(), b.encode())
    assert (await r.next_part()).name == "a"
    p = await r.next_part()  # body of "a" drained implicitly
    assert p.name == "file" and p.filename == "f.csv"
    assert b"".join([c async for c in r.iter_data()]) == b"1,2\n"
    assert await r.next_part() is None


async def test_multipart_truncated_body() -> None:
    b = "xyz"
    data = _body(b, [("file", "f.csv", b"abcdef")])[:-12]
    with pytest.raises(ValidationFailed):
        await _read_all(data, b, 4)


def test_parse_boundary() -> None:
    assert parse_boundary('multipart/form-data; boundary="a b"') == b"a b"
    assert parse_boundary("multipart/form-data; boundary=abc") == b"abc"
    for bad in (None, "application/json", "multipart/form-data", "multipart/form-data; boundary=" + "x" * 300):
        with pytest.raises(ValidationFailed):
            parse_boundary(bad)
