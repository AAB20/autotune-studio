"""Notebook install cells are generated from scripts/notebook_install_cell.py and must never embed tokens."""

import ast
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _gen():
    spec = importlib.util.spec_from_file_location("notebook_install_cell", ROOT / "scripts" / "notebook_install_cell.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["notebook_install_cell"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_notebooks_in_sync_with_generator():
    assert _gen().main(["--check"]) == 0


def test_install_cell_keeps_token_out_of_url_and_parses():
    gen = _gen()
    for runtime in ("colab", "kaggle"):
        src = gen.install_cell(runtime)
        ast.parse(src)
        assert "x-access-token:{token}@" not in src  # token must not be embedded in the git URL
        assert "GIT_CONFIG_VALUE_0" in src and 'GIT_TERMINAL_PROMPT="0"' in src
        nb = json.loads((ROOT / "notebooks" / f"{runtime}.ipynb").read_text(encoding="utf-8"))
        md = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")
        assert "leave `AUTOTUNE_SOURCE` empty" not in md
