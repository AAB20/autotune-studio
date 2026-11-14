"""cloudflared quick tunnel: binary resolution, verified download and process management.

Owned by: runtime-compatibility. ADR-0002.

Binary resolution order (:func:`resolve_cloudflared`):

1. ``AUTOTUNE_CLOUDFLARED`` env var - explicit path to a binary (trusted, not hashed).
2. ``cloudflared`` on ``PATH`` - user/system installed (trusted, not hashed).
3. Cached download for the pinned version (re-hashed against the pinned SHA256 on every use).
4. Download of the pinned release asset from
   ``https://github.com/cloudflare/cloudflared/releases/download/<ver>/<asset>``.

Integrity design
----------------
:data:`PINNED_SHA256` holds the SHA256 of every supported asset of
:data:`PINNED_VERSION`. The values were taken on 2026-11-14 from the GitHub
release API: the per-asset ``digest`` field computed by GitHub over the uploaded
bytes, which for every linux/windows asset is identical to the "SHA256 Checksums"
section Cloudflare publishes in the release notes. (For the darwin ``.tgz``
assets the release-notes value differs from GitHub's digest of the archive -
the notes appear to hash something other than the served archive - so the pinned
darwin values are GitHub's archive digests; darwin is UNVERIFIED end-to-end.)
A download whose hash differs is deleted and refused.

If a caller asks for a version that is not pinned, the expected checksum is
fetched at runtime from ``api.github.com/repos/cloudflare/cloudflared/releases/tags/<ver>``
(asset ``digest`` field, falling back to the release-notes checksum line). If no
checksum can be obtained, the binary is refused.
"""

from __future__ import annotations

import atexit
import collections
import hashlib
import logging
import os
import platform
import queue
import re
import shutil
import subprocess
import tarfile
import threading
import time
import weakref
from collections.abc import Callable, Sequence
from pathlib import Path

__all__ = [
    "PINNED_SHA256",
    "PINNED_VERSION",
    "CloudflaredError",
    "CloudflaredTunnel",
    "asset_name",
    "default_cache_dir",
    "download_cloudflared",
    "fetch_release_checksum",
    "parse_tunnel_url",
    "resolve_cloudflared",
    "sha256_file",
    "start_tunnel",
]

log = logging.getLogger(__name__)

PINNED_VERSION = "2026.11.0"
PINNED_SHA256: dict[str, str] = {
    "cloudflared-linux-amd64": "51f7c68b57ed7269164075d7e789efce5b59355982f06392deeb037d982a51fd",
    "cloudflared-linux-arm64": "abe54a006acf9faf2d19ee9d953cb0d0804e92f59c5382a162e57b84d086f56f",
    "cloudflared-windows-amd64.exe": "d5c1d82031da6253dcdcff64c944cd0fe063016edeb2e4fd088075fc9bd52d0a",
    "cloudflared-darwin-amd64.tgz": "8a59d51120cebbe902f0a8d6e1b7a7fd4f9a5bb4dcb3e0f05872483e75e7db6b",
    "cloudflared-darwin-arm64.tgz": "8c2f90277ef06a6080bb40b964d33f09f1188102f8798f2e344174b6cf41accf",
}
RELEASE_URL = "https://github.com/cloudflare/cloudflared/releases/download/{version}/{asset}"
RELEASE_API = "https://api.github.com/repos/cloudflare/cloudflared/releases/tags/{version}"
MAX_DOWNLOAD_BYTES = 200 * 1024 * 1024

# "api.trycloudflare.com" appears in cloudflared's *error* messages; never treat it as the tunnel URL.
_URL_RE = re.compile(r"https://([a-z0-9][a-z0-9-]*)\.trycloudflare\.com(?![a-z0-9-]|\.[a-z0-9])", re.IGNORECASE)
_RESERVED_HOSTS = {"api", "www"}


class CloudflaredError(RuntimeError):
    """Any failure to obtain, verify or start cloudflared. Message is user-actionable."""


# --------------------------------------------------------------------------- binary resolution
def asset_name(system: str | None = None, machine: str | None = None) -> str | None:
    """Release asset name for this platform, or None if unsupported."""
    system = (system or platform.system()).lower()
    machine = (machine or platform.machine()).lower()
    arch = {"x86_64": "amd64", "amd64": "amd64", "aarch64": "arm64", "arm64": "arm64"}.get(machine)
    if arch is None:
        return None
    if system == "linux":
        return f"cloudflared-linux-{arch}"
    if system == "windows":
        return "cloudflared-windows-amd64.exe" if arch == "amd64" else None
    if system == "darwin":
        return f"cloudflared-darwin-{arch}.tgz"
    return None


def default_cache_dir() -> Path:
    base = os.environ.get("AUTOTUNE_CACHE_DIR")
    if base:
        return Path(base).expanduser() / "cloudflared"
    if os.name == "nt":
        root = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
        return root / "autotune_studio" / "cloudflared"
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "autotune_studio" / "cloudflared"


def sha256_file(path: str | os.PathLike[str]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _binary_filename() -> str:
    return "cloudflared.exe" if os.name == "nt" else "cloudflared"


def fetch_release_checksum(version: str, asset: str, timeout: float = 15.0) -> str | None:
    """Fetch the expected SHA256 of ``asset`` from the GitHub release API. None if unavailable."""
    try:
        import httpx

        r = httpx.get(RELEASE_API.format(version=version), timeout=timeout,
                      headers={"Accept": "application/vnd.github+json"}, follow_redirects=True)
        if r.status_code != 200:
            return None
        data = r.json()
    except Exception:
        return None
    for a in data.get("assets", []) or []:
        if a.get("name") == asset:
            digest = str(a.get("digest") or "")
            if digest.startswith("sha256:") and re.fullmatch(r"[0-9a-f]{64}", digest[7:]):
                return digest[7:]
    m = re.search(rf"^{re.escape(asset)}:\s*([0-9a-f]{{64}})\s*$", str(data.get("body") or ""), re.MULTILINE)
    return m.group(1) if m else None


def _expected_sha(version: str, asset: str, checksum_fetcher: Callable[[str, str], str | None]) -> str:
    if version == PINNED_VERSION and asset in PINNED_SHA256:
        return PINNED_SHA256[asset]
    sha = checksum_fetcher(version, asset)
    if not sha:
        raise CloudflaredError(
            f"No trusted SHA256 available for {asset} {version}; refusing to run an unverified binary. "
            "Install cloudflared yourself and put it on PATH or set AUTOTUNE_CLOUDFLARED."
        )
    return sha.lower()


def _http_download(url: str, dest: Path, timeout: float) -> None:
    import httpx

    total = 0
    with httpx.stream("GET", url, timeout=timeout, follow_redirects=True) as r:
        r.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in r.iter_bytes(1 << 20):
                total += len(chunk)
                if total > MAX_DOWNLOAD_BYTES:
                    raise CloudflaredError("cloudflared download exceeded the size limit")
                fh.write(chunk)


def _extract_tgz_binary(archive: Path, dest: Path) -> None:
    """Extract only the regular file named 'cloudflared' (no path traversal possible)."""
    with tarfile.open(archive, "r:gz") as tf:
        for member in tf.getmembers():
            if member.isfile() and Path(member.name).name == "cloudflared":
                src = tf.extractfile(member)
                if src is None:
                    break
                with src, open(dest, "wb") as out:
                    shutil.copyfileobj(src, out)
                return
    raise CloudflaredError("cloudflared binary not found inside the downloaded archive")


def download_cloudflared(
    cache_dir: str | os.PathLike[str] | None = None,
    version: str = PINNED_VERSION,
    *,
    asset: str | None = None,
    downloader: Callable[[str, Path], None] | None = None,
    checksum_fetcher: Callable[[str, str], str | None] | None = None,
    timeout: float = 120.0,
) -> Path:
    """Return a verified cloudflared binary in the cache, downloading it if needed.

    Raises :class:`CloudflaredError` on unsupported platform, download failure or checksum mismatch.
    """
    asset = asset or asset_name()
    if asset is None:
        raise CloudflaredError(
            f"No cloudflared release for {platform.system()}/{platform.machine()}; install it manually."
        )
    expected = _expected_sha(version, asset, checksum_fetcher or fetch_release_checksum)
    vdir = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    vdir = vdir / version
    vdir.mkdir(parents=True, exist_ok=True)
    final = vdir / _binary_filename()
    marker = vdir / (asset + ".sha256")

    is_archive = asset.endswith(".tgz")
    # Marker = "<verified asset sha> <sha of installed binary>". A cached binary is reused only if the
    # asset hash equals the expected one and the binary on disk still hashes to the recorded value.
    if final.is_file() and marker.is_file():
        parts = marker.read_text().split()
        if len(parts) == 2 and parts[0] == expected and sha256_file(final) == parts[1]:
            return final
        log.warning("cached cloudflared failed re-verification; re-downloading")

    tmp = vdir / (asset + ".part")
    try:
        (downloader or (lambda u, d: _http_download(u, d, timeout)))(
            RELEASE_URL.format(version=version, asset=asset), tmp
        )
        actual = sha256_file(tmp)
        if actual != expected:
            raise CloudflaredError(
                f"Checksum mismatch for {asset} {version}: expected {expected[:12]}..., got {actual[:12]}... "
                "The download was discarded."
            )
        if is_archive:
            _extract_tgz_binary(tmp, final)
        else:
            os.replace(tmp, final)
        if os.name != "nt":
            final.chmod(0o755)
        marker.write_text(f"{expected} {sha256_file(final)}")
        return final
    except CloudflaredError:
        raise
    except Exception as exc:
        raise CloudflaredError(f"Downloading cloudflared failed ({type(exc).__name__}: {exc})") from exc
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def resolve_cloudflared(
    *,
    allow_download: bool = True,
    cache_dir: str | os.PathLike[str] | None = None,
    downloader: Callable[[str, Path], None] | None = None,
    checksum_fetcher: Callable[[str, str], str | None] | None = None,
) -> str:
    """Return a path to a usable cloudflared binary (see module docstring for order)."""
    env_path = os.environ.get("AUTOTUNE_CLOUDFLARED", "").strip()
    if env_path:
        p = Path(env_path).expanduser()
        if p.is_file():
            return str(p)
        raise CloudflaredError(f"AUTOTUNE_CLOUDFLARED points to '{p}', which is not a file.")
    on_path = shutil.which("cloudflared")
    if on_path:
        return on_path
    if not allow_download:
        raise CloudflaredError("cloudflared not found on PATH and downloading is disabled.")
    return str(download_cloudflared(cache_dir, downloader=downloader, checksum_fetcher=checksum_fetcher))


# --------------------------------------------------------------------------- tunnel process
def parse_tunnel_url(line: str) -> str | None:
    for m in _URL_RE.finditer(line):
        if m.group(1).lower() not in _RESERVED_HOSTS:
            return f"https://{m.group(1).lower()}.trycloudflare.com"
    return None


_LIVE: weakref.WeakSet[CloudflaredTunnel] = weakref.WeakSet()


@atexit.register
def _stop_all() -> None:  # pragma: no cover - interpreter shutdown
    for t in list(_LIVE):
        try:
            t.stop()
        except Exception:
            pass


class CloudflaredTunnel:
    """A running ``cloudflared tunnel --url http://HOST:PORT --no-autoupdate`` quick tunnel.

    ``binary`` may be a path or a command prefix list (tests use ``[sys.executable, fake.py]``).
    Output is drained continuously by a reader thread (so the pipe never blocks) and the
    last lines are kept in :attr:`recent_output` for diagnostics.
    """

    def __init__(self, binary: str | Sequence[str], port: int, host: str = "127.0.0.1") -> None:
        self.cmd_prefix = [binary] if isinstance(binary, (str, os.PathLike)) else list(binary)
        self.port = int(port)
        self.host = host
        self.url: str | None = None
        self.proc: subprocess.Popen[str] | None = None
        self.recent_output: collections.deque[str] = collections.deque(maxlen=50)
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._reader: threading.Thread | None = None

    @property
    def command(self) -> list[str]:
        return [*map(str, self.cmd_prefix), "tunnel", "--url", f"http://{self.host}:{self.port}", "--no-autoupdate"]

    def start(self, timeout: float = 30.0) -> str:
        """Start the process and wait up to ``timeout`` s for the trycloudflare URL. Raises CloudflaredError."""
        kwargs: dict[str, object] = {}
        if os.name == "nt":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            self.proc = subprocess.Popen(
                self.command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1, **kwargs,  # type: ignore[arg-type]
            )
        except OSError as exc:
            raise CloudflaredError(f"Could not start cloudflared ({type(exc).__name__}: {exc})") from exc
        _LIVE.add(self)
        self._reader = threading.Thread(target=self._read, name="autotune-cloudflared-reader", daemon=True)
        self._reader.start()

        deadline = time.monotonic() + timeout
        eof = False
        while time.monotonic() < deadline:
            try:
                line = self._lines.get(timeout=min(0.25, max(0.01, deadline - time.monotonic())))
            except queue.Empty:
                continue
            if line is None:  # EOF: process exited (its exit status may lag slightly behind)
                eof = True
                break
            url = parse_tunnel_url(line)
            if url:
                self.url = url
                return url
        try:
            code = self.proc.wait(timeout=2) if eof else self.proc.poll()
        except subprocess.TimeoutExpired:
            code = None
        tail = " | ".join(list(self.recent_output)[-5:])
        self.stop()
        if code is not None:
            raise CloudflaredError(f"cloudflared exited with code {code} before publishing a URL. Last output: {tail}")
        raise CloudflaredError(f"cloudflared did not publish a trycloudflare.com URL within {timeout:.0f}s. "
                               f"Last output: {tail}")

    def _read(self) -> None:
        proc = self.proc
        assert proc is not None and proc.stdout is not None
        try:
            for raw in proc.stdout:
                line = raw.rstrip()
                self.recent_output.append(line[:500])
                self._lines.put(line)
        except (OSError, ValueError):
            pass
        finally:
            self._lines.put(None)

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self, timeout: float = 5.0) -> None:
        proc = self.proc
        if proc is None:
            return
        if proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout)
                except subprocess.TimeoutExpired:
                    log.warning("cloudflared (pid %s) did not exit", proc.pid)
            except OSError:
                pass
        if proc.stdout is not None:
            try:
                proc.stdout.close()
            except OSError:
                pass
        _LIVE.discard(self)


def start_tunnel(
    port: int,
    *,
    timeout: float = 30.0,
    binary: str | Sequence[str] | None = None,
    host: str = "127.0.0.1",
    allow_download: bool = True,
) -> CloudflaredTunnel:
    """Resolve the binary (unless given), start a quick tunnel and return it once a URL is known."""
    tunnel = CloudflaredTunnel(binary or resolve_cloudflared(allow_download=allow_download), port, host)
    tunnel.start(timeout)
    return tunnel
