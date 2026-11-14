"""Path allowlist, uploads and model-name validation."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

from autotune_studio.errors import ForbiddenPath, NotFound, ValidationFailed
from autotune_studio.security.validation import (
    ALLOWED_EXTENSIONS,
    PathPolicy,
    UploadLimits,
    asave_upload_stream,
    sanitize_upload_filename,
    save_upload_stream,
    validate_model_name,
)


@pytest.fixture
def layout(tmp_path: Path):
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    (root / "data").mkdir(parents=True)
    outside.mkdir()
    (root / "data" / "train.csv").write_text("a,b\n1,2\n")
    (root / "data" / "train.jsonl").write_text('{"a":1}\n')
    (root / "data" / "notes.txt").write_text("x")
    (root / "pq").mkdir()
    (outside / "secret.csv").write_text("s\n")
    return root, outside, PathPolicy([str(root)])


def _symlink_or_skip(link: Path, target: Path, is_dir: bool = False) -> None:
    try:
        os.symlink(target, link, target_is_directory=is_dir)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(
            f"cannot create symlinks on this OS/account ({exc!r}); Windows needs Developer Mode or admin"
        )


# ----------------------------------------------------------------- path policy
def test_resolve_inside_root(layout):
    root, _, pol = layout
    p = pol.resolve(str(root / "data" / "train.csv"), kinds={"csv"})
    assert p == (root / "data" / "train.csv").resolve()


def test_relative_path_resolves_against_first_root(layout):
    root, _, pol = layout
    assert pol.resolve("data/train.csv") == (root / "data" / "train.csv").resolve()


@pytest.mark.parametrize(
    "rel",
    [
        "../outside/secret.csv",
        "data/../../outside/secret.csv",
        "data/../../../../../../../../etc/passwd",
    ],
)
def test_dotdot_traversal_rejected(layout, rel):
    root, _, pol = layout
    with pytest.raises(ForbiddenPath):
        pol.resolve(str(root) + "/" + rel)
    with pytest.raises(ForbiddenPath):
        pol.resolve(rel)


def test_absolute_outside_root_rejected(layout):
    _, outside, pol = layout
    with pytest.raises(ForbiddenPath):
        pol.resolve(str(outside / "secret.csv"))
    # Non-existent outside path is ForbiddenPath (not NotFound) - no existence oracle.
    with pytest.raises(ForbiddenPath):
        pol.resolve(str(outside / "missing.csv"))


def test_sibling_prefix_is_not_inside(tmp_path):
    (tmp_path / "root").mkdir()
    (tmp_path / "root2").mkdir()
    (tmp_path / "root2" / "x.csv").write_text("a\n")
    pol = PathPolicy([str(tmp_path / "root")])
    with pytest.raises(ForbiddenPath):
        pol.resolve(str(tmp_path / "root2" / "x.csv"))


def test_nul_byte_rejected(layout):
    root, _, pol = layout
    with pytest.raises(ForbiddenPath):
        pol.resolve(str(root / "data" / "train.csv") + "\x00.txt")


def test_empty_and_non_string(layout):
    _, _, pol = layout
    with pytest.raises(ValidationFailed):
        pol.resolve("   ")
    with pytest.raises(ValidationFailed):
        pol.resolve(123)  # type: ignore[arg-type]


def test_missing_inside_root(layout):
    root, _, pol = layout
    with pytest.raises(NotFound):
        pol.resolve(str(root / "data" / "nope.csv"))
    assert pol.resolve(str(root / "data" / "new.csv"), must_exist=False).name == "new.csv"


def test_symlink_file_escape_rejected(layout):
    root, outside, pol = layout
    link = root / "data" / "link.csv"
    _symlink_or_skip(link, outside / "secret.csv")
    with pytest.raises(ForbiddenPath):
        pol.resolve(str(link))


def test_symlink_dir_escape_rejected(layout):
    root, outside, pol = layout
    link = root / "escape"
    _symlink_or_skip(link, outside, is_dir=True)
    with pytest.raises(ForbiddenPath):
        pol.resolve(str(link / "secret.csv"))


@pytest.mark.skipif(sys.platform != "win32", reason="NTFS junctions are Windows-only")
def test_windows_junction_escape_rejected(layout):
    """Junctions need no privilege, so the reparse-point escape is exercised even without symlink rights."""
    import subprocess

    root, outside, pol = layout
    link = root / "junction"
    r = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(outside)], check=False, capture_output=True, text=True
    )
    if r.returncode != 0:
        pytest.skip(f"mklink /J failed: {r.stdout}{r.stderr}")
    assert (link / "secret.csv").exists()
    with pytest.raises(ForbiddenPath):
        pol.resolve(str(link / "secret.csv"))
    with pytest.raises(ForbiddenPath):
        pol.resolve(str(link))


def test_symlink_inside_root_allowed(layout):
    root, _, pol = layout
    link = root / "alias.csv"
    _symlink_or_skip(link, root / "data" / "train.csv")
    assert pol.resolve(str(link)) == (root / "data" / "train.csv").resolve()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific path semantics")
def test_windows_case_insensitive_and_ads(layout):
    root, _, pol = layout
    assert pol.resolve(str(root).upper() + "\\DATA\\TRAIN.CSV", kinds={"csv"})
    with pytest.raises(ForbiddenPath):
        pol.resolve(str(root / "data" / "train.csv") + ":evil")


def test_extension_allowlist(layout):
    root, _, pol = layout
    with pytest.raises(ValidationFailed):
        pol.resolve(str(root / "data" / "notes.txt"), kinds={"csv"})
    with pytest.raises(ValidationFailed):
        pol.resolve(str(root / "data" / "train.csv"), kinds={"json", "jsonl"})
    assert pol.resolve(str(root / "data" / "train.jsonl"), kinds={"jsonl"})
    assert pol.resolve(str(root / "data" / "notes.txt")) is not None  # kinds=None -> no ext check


def test_directories_only_for_parquet(layout):
    root, _, pol = layout
    assert pol.resolve(str(root / "pq"), kinds={"parquet"}).is_dir()
    with pytest.raises(ValidationFailed):
        pol.resolve(str(root / "pq"), kinds={"csv"})


def test_unknown_kind_is_programming_error(layout):
    root, _, pol = layout
    with pytest.raises(ValueError):
        pol.resolve(str(root / "data" / "train.csv"), kinds={"exe"})


def test_allowed_extensions_table():
    assert ALLOWED_EXTENSIONS == {
        "csv": {".csv", ".tsv"},
        "json": {".json"},
        "jsonl": {".jsonl", ".ndjson"},
        "parquet": {".parquet", ".pq"},
    }


def test_policy_requires_roots():
    with pytest.raises(ValueError):
        PathPolicy([])


# ----------------------------------------------------------------- uploads
@pytest.mark.parametrize(
    "name,expected",
    [
        ("train.CSV", ("train", ".csv")),
        ("../../etc/passwd.jsonl", ("passwd", ".jsonl")),
        ("C:\\Users\\x\\evil name!.parquet", ("evil_name", ".parquet")),
        ("....json", ("upload", ".json")),
        ("my data (1).tsv", ("my_data_1", ".tsv")),
    ],
)
def test_sanitize_upload_filename(name, expected):
    assert sanitize_upload_filename(name) == expected


@pytest.mark.parametrize("name", ["evil.exe", "noext", "x.csv.exe", "a.pkl", "x.csv\x00.exe", ""])
def test_sanitize_rejects_bad_extensions(name):
    with pytest.raises(ValidationFailed):
        sanitize_upload_filename(name)


def test_save_upload_stream_writes_random_name(tmp_path):
    dest = save_upload_stream([b"a,b\n", b"1,2\n"], tmp_path / "uploads", ".csv", UploadLimits(max_bytes=100))
    assert dest.parent == tmp_path / "uploads"
    assert dest.suffix == ".csv" and len(dest.stem) == 32
    assert dest.read_bytes() == b"a,b\n1,2\n"
    dest2 = save_upload_stream([b"x"], tmp_path / "uploads", "csv")
    assert dest2 != dest


def test_upload_size_limit_deletes_partial(tmp_path):
    up = tmp_path / "uploads"
    with pytest.raises(ValidationFailed):
        save_upload_stream((b"x" * 40 for _ in range(10)), up, ".csv", UploadLimits(max_bytes=100))
    assert list(up.iterdir()) == []


def test_upload_exactly_at_limit_ok(tmp_path):
    p = save_upload_stream([b"x" * 50, b"y" * 50], tmp_path, ".csv", UploadLimits(max_bytes=100))
    assert p.stat().st_size == 100


def test_upload_empty_rejected_and_cleaned(tmp_path):
    with pytest.raises(ValidationFailed):
        save_upload_stream([], tmp_path, ".json")
    assert list(tmp_path.iterdir()) == []


def test_upload_source_error_cleans_up(tmp_path):
    def gen():
        yield b"abc"
        raise RuntimeError("client disconnected")

    with pytest.raises(RuntimeError):
        save_upload_stream(gen(), tmp_path, ".csv")
    assert list(tmp_path.iterdir()) == []


def test_upload_bad_ext_and_bad_chunk(tmp_path):
    with pytest.raises(ValidationFailed):
        save_upload_stream([b"x"], tmp_path, ".exe")
    with pytest.raises(ValidationFailed):
        save_upload_stream(["str-not-bytes"], tmp_path, ".csv")  # type: ignore[list-item]
    assert list(tmp_path.iterdir()) == []


def test_default_limit_is_2gib():
    assert UploadLimits().max_bytes == 2 * 1024**3


def test_async_upload_and_limit(tmp_path):
    async def agen(n):
        for _ in range(n):
            yield b"z" * 10

    p = asyncio.run(asave_upload_stream(agen(3), tmp_path / "a", ".jsonl", UploadLimits(max_bytes=30)))
    assert p.read_bytes() == b"z" * 30
    with pytest.raises(ValidationFailed):
        asyncio.run(asave_upload_stream(agen(4), tmp_path / "b", ".jsonl", UploadLimits(max_bytes=30)))
    assert list((tmp_path / "b").iterdir()) == []
    with pytest.raises(TypeError):
        save_upload_stream(agen(1), tmp_path / "c", ".csv")  # type: ignore[arg-type]


# ----------------------------------------------------------------- model names
@pytest.mark.parametrize(
    "name",
    [
        "unsloth/Llama-3.2-1B-Instruct",
        "sshleifer/tiny-gpt2",
        "Qwen/Qwen2.5-0.5B",
        "a/b",
        "  org/model_v1.0  ",
    ],
)
def test_valid_repo_ids(name):
    assert validate_model_name(name) == name.strip()


@pytest.mark.parametrize(
    "name",
    [
        "",
        "   ",
        "https://huggingface.co/gpt2",
        "http://evil/x",
        "hf://org/model",
        "file:///etc/passwd",
        "s3://bucket/model",
        "org/../etc",
        "../org/model",
        "org/model/extra",
        "-org/model",
        "org/mo del",
        "org/model\x00",
        "org/" + "x" * 300,
        "org/model;rm -rf",
        "a/b/c",  # too many path segments
        "/abs/path",  # local path without policy
        "C:\\models\\x",
    ],
)
def test_invalid_model_names(name):
    with pytest.raises(ValidationFailed):
        validate_model_name(name)


def test_local_model_dir_with_policy(layout):
    root, outside, pol = layout
    (root / "models" / "m").mkdir(parents=True)
    assert validate_model_name(str(root / "models" / "m"), pol) == str((root / "models" / "m").resolve())
    with pytest.raises(ForbiddenPath):
        validate_model_name(str(outside), pol)
    with pytest.raises(ValidationFailed):  # a file, not a model directory
        validate_model_name(str(root / "data" / "train.csv"), pol)
    with pytest.raises(NotFound):
        validate_model_name(str(root / "models" / "missing"), pol)
