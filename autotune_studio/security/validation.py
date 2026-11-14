"""Path allowlist, upload validation and model-id validation. Owned by: security.

* :class:`PathPolicy` confines user-supplied paths to ``EnvironmentInfo.allowed_roots``.
  ``Path.resolve()`` follows symlinks *before* the containment check, so symlink and
  ``..`` escapes are caught. Containment is checked before existence so callers cannot
  probe for files outside the roots (both yield ``ForbiddenPath``).
* Uploads are streamed in chunks to ``<dest_dir>/<random hex>.<ext>`` with a hard size
  limit; partial files are deleted on any failure.
* :func:`validate_model_name` accepts a Hugging Face repo id or (with a policy) a local
  directory under an allowed root. URLs are never accepted.

Residual risk (documented in docs/security-review.md): check-then-open TOCTOU — a
symlink swapped in after :meth:`PathPolicy.resolve` returns is not detected.
"""

from __future__ import annotations

import os
import re
import secrets
import sys
from collections.abc import AsyncIterable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Literal

from ..errors import ForbiddenPath, NotFound, ValidationFailed

__all__ = [
    "ALLOWED_EXTENSIONS",
    "ALL_ALLOWED_EXTENSIONS",
    "DIRECTORY_KINDS",
    "PACKAGED_BUILDERS",
    "PathPolicy",
    "UploadLimits",
    "asave_upload_stream",
    "sanitize_upload_filename",
    "save_upload_stream",
    "validate_hf_dataset_name",
    "validate_model_name",
]

ALLOWED_EXTENSIONS: dict[str, set[str]] = {
    "csv": {".csv", ".tsv"},
    "json": {".json"},
    "jsonl": {".jsonl", ".ndjson"},
    "parquet": {".parquet", ".pq"},
}
ALL_ALLOWED_EXTENSIONS: frozenset[str] = frozenset().union(*ALLOWED_EXTENSIONS.values())
#: Kinds for which a directory (e.g. a partitioned parquet dataset) is acceptable.
DIRECTORY_KINDS: frozenset[str] = frozenset({"parquet"})

_IS_WINDOWS = sys.platform == "win32"
_MAX_PATH_LEN = 4096


def _norm(p: Path) -> Path:
    return Path(os.path.normcase(str(p)))


def _is_within(child: Path, root: Path) -> bool:
    """Case-insensitive on Windows (normcase), exact elsewhere."""
    return _norm(child).is_relative_to(_norm(root))


# --------------------------------------------------------------------------- path policy
class PathPolicy:
    """Allowlist of directory roots for user-supplied paths."""

    def __init__(self, allowed_roots: Sequence[str | os.PathLike[str]]) -> None:
        roots: list[Path] = []
        for r in allowed_roots:
            s = os.fspath(r)
            if not s or "\x00" in s:
                raise ValueError(f"invalid allowed root: {s!r}")
            roots.append(Path(s).expanduser().resolve())
        if not roots:
            raise ValueError("PathPolicy requires at least one allowed root")
        self.allowed_roots: tuple[Path, ...] = tuple(dict.fromkeys(roots))

    def __repr__(self) -> str:
        return f"PathPolicy(allowed_roots={[str(r) for r in self.allowed_roots]!r})"

    def is_allowed(self, path: Path) -> bool:
        return any(_is_within(path, root) for root in self.allowed_roots)

    def resolve(
        self,
        path: str | os.PathLike[str],
        must_exist: bool = True,
        kinds: set[str] | None = None,
        *,
        expect: Literal["any", "file", "dir"] = "any",
    ) -> Path:
        """Validate and canonicalise ``path``.

        :param must_exist: raise :class:`NotFound` if the (allowed) path does not exist.
        :param kinds: dataset kinds from :data:`ALLOWED_EXTENSIONS`; when given, a file must
            have one of their extensions and a directory is only accepted for
            :data:`DIRECTORY_KINDS`. ``None`` = no extension restriction.
        :param expect: additionally require a file or a directory.
        Relative paths are resolved against the first allowed root (not the process cwd).
        """
        try:
            raw = os.fspath(path)
        except TypeError:
            raise ValidationFailed("path must be a string") from None
        if not isinstance(raw, str):
            raise ValidationFailed("path must be a string")
        if "\x00" in raw:
            raise ForbiddenPath("path contains a NUL byte")
        if not raw.strip():
            raise ValidationFailed("path is empty")
        if len(raw) > _MAX_PATH_LEN:
            raise ValidationFailed("path is too long")
        if kinds is not None:
            unknown = set(kinds) - ALLOWED_EXTENSIONS.keys()
            if unknown:
                raise ValueError(f"unknown kinds: {sorted(unknown)}")

        p = Path(raw.strip()).expanduser()
        if _IS_WINDOWS:
            # Reject NTFS alternate data streams ("file.csv:evil") and other stray colons.
            tail = p.parts[1:] if p.drive else p.parts
            if any(":" in part for part in tail):
                raise ForbiddenPath("path contains an invalid ':'")
        if not p.is_absolute():
            p = self.allowed_roots[0] / p
        try:
            resolved = p.resolve()  # strict=False: follows every existing symlink component
        except (OSError, RuntimeError) as exc:  # symlink loops, permission errors
            raise ForbiddenPath("path could not be resolved") from exc

        if not self.is_allowed(resolved):
            raise ForbiddenPath(
                "path is outside the allowed directories",
                {"allowed_roots": [str(r) for r in self.allowed_roots]},
            )

        exists = resolved.exists()
        if must_exist and not exists:
            raise NotFound("path does not exist", {"path": str(resolved)})

        is_dir = exists and resolved.is_dir()
        if expect == "file" and is_dir:
            raise ValidationFailed("expected a file, got a directory")
        if expect == "dir" and exists and not is_dir:
            raise ValidationFailed("expected a directory, got a file")

        if kinds is not None:
            if is_dir:
                if not (set(kinds) & DIRECTORY_KINDS):
                    raise ValidationFailed(
                        "directories are only accepted for: " + ", ".join(sorted(DIRECTORY_KINDS))
                    )
            else:
                allowed = set().union(*(ALLOWED_EXTENSIONS[k] for k in kinds))
                ext = resolved.suffix.lower()
                if ext not in allowed:
                    raise ValidationFailed(
                        f"file extension {ext or '(none)'!r} not allowed", {"allowed": sorted(allowed)}
                    )
        return resolved


# --------------------------------------------------------------------------- uploads
_SAFE_STEM = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_upload_filename(name: str, allowed_extensions: Iterable[str] | None = None) -> tuple[str, str]:
    """Return ``(safe_stem, lowercase_ext)`` from a client-supplied filename.

    Directory components (``/`` and ``\\``), control characters and anything outside
    ``[A-Za-z0-9._-]`` are dropped; the extension must be in the allowlist (default: all
    dataset extensions). The stem is for display/metadata only — files are stored under a
    random name by :func:`save_upload_stream`.
    """
    if not isinstance(name, str):
        raise ValidationFailed("filename must be a string")
    allowed = (
        {e.lower() for e in allowed_extensions} if allowed_extensions is not None else ALL_ALLOWED_EXTENSIONS
    )
    base = re.split(r"[\\/]", name)[-1]
    base = "".join(ch for ch in base if ch.isprintable() and ch not in "\x00:")
    base = base.strip().rstrip(". ")  # Windows silently drops trailing dots/spaces
    stem, dot, ext = base.rpartition(".")
    if not dot:
        stem, ext = base, ""
    ext = ("." + ext.lower()) if ext else ""
    if ext not in allowed:
        raise ValidationFailed(
            f"file extension {ext or '(none)'!r} not allowed", {"allowed": sorted(allowed)}
        )
    stem = _SAFE_STEM.sub("_", stem).strip("._")[:100] or "upload"
    return stem, ext


@dataclass(frozen=True)
class UploadLimits:
    max_bytes: int = 2 * 1024**3  # 2 GiB
    min_bytes: int = 1  # reject empty uploads

    def __post_init__(self) -> None:
        if self.max_bytes < 1 or self.min_bytes < 0 or self.min_bytes > self.max_bytes:
            raise ValueError("invalid UploadLimits")


def _prepare_dest(dest_dir: Path, ext: str) -> tuple[Path, BinaryIO]:
    ext = ext.lower()
    if not ext.startswith("."):
        ext = "." + ext
    if ext not in ALL_ALLOWED_EXTENSIONS:
        raise ValidationFailed(
            f"file extension {ext!r} not allowed", {"allowed": sorted(ALL_ALLOWED_EXTENSIONS)}
        )
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{secrets.token_hex(16)}{ext}"
    fh = open(dest, "xb")  # noqa: SIM115 - closed by _Writer; exclusive create: never overwrite / follow a pre-planted file
    return dest, fh


class _Writer:
    def __init__(self, dest: Path, fh: BinaryIO, limits: UploadLimits) -> None:
        self.dest, self.fh, self.limits, self.size = dest, fh, limits, 0

    def write(self, chunk: bytes) -> None:
        if not isinstance(chunk, (bytes, bytearray, memoryview)):
            raise ValidationFailed("upload chunks must be bytes")
        self.size += len(chunk)
        if self.size > self.limits.max_bytes:
            raise ValidationFailed("upload exceeds maximum size", {"max_bytes": self.limits.max_bytes})
        self.fh.write(chunk)

    def finish(self) -> Path:
        self.fh.close()
        if self.size < self.limits.min_bytes:
            raise ValidationFailed("upload is empty" if self.size == 0 else "upload too small")
        return self.dest

    def abort(self) -> None:
        try:
            self.fh.close()
        finally:
            try:
                self.dest.unlink(missing_ok=True)
            except OSError:  # pragma: no cover
                pass


def save_upload_stream(
    chunks: Iterable[bytes], dest_dir: Path, ext: str, limits: UploadLimits = UploadLimits()
) -> Path:
    """Stream ``chunks`` to ``dest_dir/<random>.<ext>``; enforce ``limits``; delete partial file on failure.

    Raises :class:`ValidationFailed` for size violations or a disallowed extension. For
    async sources (Starlette ``UploadFile``) use :func:`asave_upload_stream`.
    """
    if isinstance(chunks, AsyncIterable):
        raise TypeError("async chunk source: use asave_upload_stream()")
    dest, fh = _prepare_dest(dest_dir, ext)
    w = _Writer(dest, fh, limits)
    try:
        for chunk in chunks:
            w.write(chunk)
        return w.finish()
    except BaseException:
        w.abort()
        raise


async def asave_upload_stream(
    chunks: AsyncIterable[bytes] | Iterable[bytes],
    dest_dir: Path,
    ext: str,
    limits: UploadLimits = UploadLimits(),
) -> Path:
    """Async variant of :func:`save_upload_stream` (accepts async or sync iterables).

    Disk writes are synchronous per chunk; with ~1 MiB chunks this blocks the event loop
    only briefly. Wrap in a threadpool if that matters.
    """
    dest, fh = _prepare_dest(dest_dir, ext)
    w = _Writer(dest, fh, limits)
    try:
        if isinstance(chunks, AsyncIterable):
            async for chunk in chunks:
                w.write(chunk)
        else:
            for chunk in chunks:
                w.write(chunk)
        return w.finish()
    except BaseException:
        w.abort()
        raise


# --------------------------------------------------------------------------- model ids
# owner/name, or a legacy bare id such as "gpt2" (ADR: Architect relaxed this during integration).
HF_REPO_ID_RE = re.compile(r"^(?:[A-Za-z0-9][\w.\-]*/)?[A-Za-z0-9][\w.\-]*$")
_MAX_MODEL_NAME_LEN = 256


def _looks_like_path(name: str) -> bool:
    return (
        name.startswith((".", "~", "/", "\\"))
        or "\\" in name
        or (len(name) > 1 and name[1] == ":")  # Windows drive
        or Path(name).is_absolute()
    )


def validate_model_name(name: str, policy: PathPolicy | None = None) -> str:
    """Validate a model reference; return the repo id or the resolved local directory path.

    Accepted: HF repo id ``owner/name`` (regex :data:`HF_REPO_ID_RE`, no ``..``), or —
    only when ``policy`` is given — an existing directory under an allowed root written as
    a path (absolute, ``./``, ``~``). URLs and ``scheme://`` strings are always rejected.
    """
    if not isinstance(name, str):
        raise ValidationFailed("model_name must be a string")
    if "\x00" in name:
        raise ValidationFailed("model_name contains a NUL byte")
    name = name.strip()
    if not name:
        raise ValidationFailed("model_name is empty")
    if len(name) > _MAX_MODEL_NAME_LEN:
        raise ValidationFailed("model_name is too long")
    low = name.lower()
    if "://" in name or low.startswith(("http:", "https:", "ftp:", "file:", "hf:", "s3:", "gs:", "git@")):
        raise ValidationFailed("model_name must be a Hugging Face repo id or local path, not a URL")
    if _looks_like_path(name):
        if policy is None:
            raise ValidationFailed("local model paths are not allowed here")
        return str(policy.resolve(name, must_exist=True, expect="dir"))
    if HF_REPO_ID_RE.match(name) and ".." not in name and not name.endswith((".", "-")):
        # SEC-05: transformers / unsloth check ``os.path.isdir(name)`` relative to the process cwd
        # *before* contacting the Hub, so a repo-id-shaped name can silently select a local
        # directory. Pin such names to the absolute directory, or refuse it if outside the roots.
        if os.path.exists(name):
            if policy is None:
                raise ValidationFailed("model_name matches a local path; local model paths are not allowed here")
            return str(policy.resolve(os.path.abspath(name), must_exist=True, expect="dir"))
        return name
    raise ValidationFailed("model_name must look like 'owner/model' or 'model' (Hugging Face repo id)")


# --------------------------------------------------------------------------- HF dataset ids
#: ``datasets`` packaged builders. ``load_dataset("<builder>")`` without ``data_files`` reads
#: every matching file under the process cwd (recursively), bypassing :class:`PathPolicy`.
#: Static fallback; the live list from ``datasets.packaged_modules`` is merged in when importable.
PACKAGED_BUILDERS: frozenset[str] = frozenset({
    "arrow", "audiofolder", "conll", "csv", "eval", "fasta", "fastq", "genbank", "harbor", "hdf5",
    "iceberg", "imagefolder", "json", "lance", "meshfolder", "mmcif", "niftifolder", "pandas", "parquet",
    "pdb", "pdffolder", "spark", "sql", "text", "tsfile", "videofolder", "vortex", "webdataset", "xml",
    "generator", "cache",
})


def _packaged_builders() -> frozenset[str]:
    try:  # cheap: the module only holds a dict; datasets may already be imported anyway
        from datasets.packaged_modules import _PACKAGED_DATASETS_MODULES  # type: ignore[import-not-found]

        return PACKAGED_BUILDERS | frozenset(k.lower() for k in _PACKAGED_DATASETS_MODULES)
    except Exception:  # noqa: BLE001 - datasets not installed / layout changed
        return PACKAGED_BUILDERS


def validate_hf_dataset_name(name: str) -> str:
    """Validate a Hugging Face *Hub* dataset id for ``datasets.load_dataset`` (SEC-01).

    ``datasets.load_dataset(name)`` treats ``name`` as a local directory/file if it exists
    (absolute, or relative to the process cwd) and treats packaged builder names (``csv``,
    ``json``, ``text``, ...) as "read files under the cwd". Either way local files outside the
    allowed roots would be streamed to the UI. Accepted: ``owner/name`` or a bare legacy id that
    is not a packaged builder and does not exist on the local filesystem. Local files must use
    the csv/json/jsonl/parquet sources, which go through :class:`PathPolicy`.
    """
    if not isinstance(name, str):
        raise ValidationFailed("dataset name must be a string")
    if "\x00" in name:
        raise ValidationFailed("dataset name contains a NUL byte")
    name = name.strip()
    if not name:
        raise ValidationFailed("dataset name is empty")
    if len(name) > _MAX_MODEL_NAME_LEN:
        raise ValidationFailed("dataset name is too long")
    low = name.lower()
    if "://" in name or low.startswith(("http:", "https:", "ftp:", "file:", "hf:", "s3:", "gs:", "git@")):
        raise ValidationFailed("dataset name must be a Hugging Face dataset id, not a URL")
    if _looks_like_path(name):
        raise ForbiddenPath("local paths are not accepted as Hugging Face dataset ids; use a file source")
    if not (HF_REPO_ID_RE.match(name) and ".." not in name and not name.endswith((".", "-"))):
        raise ValidationFailed("dataset name must look like 'owner/dataset' (Hugging Face dataset id)")
    if low in _packaged_builders():
        raise ValidationFailed(
            f"'{name}' is a local file builder, not a Hub dataset; use the csv/json/jsonl/parquet source"
        )
    if os.path.lexists(name):
        raise ForbiddenPath("dataset name matches a local path; use a file source")
    return name
