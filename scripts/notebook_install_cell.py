"""Single source of truth for the install cell of ``notebooks/colab.ipynb`` and ``notebooks/kaggle.ipynb``.

The two notebooks share the install logic; only the wheel location differs.

Usage::

    python scripts/notebook_install_cell.py          # rewrite the install cells (and option text) in both notebooks
    python scripts/notebook_install_cell.py --check  # exit 1 if a notebook is out of sync (used by tests)

Install sources: the public GitHub repository (default) or an uploaded wheel. No credentials are
needed; git prompts are disabled so a bad URL fails fast instead of waiting for a login.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_URL = "https://github.com/AAB20/autotune-studio.git"

_IMPORTS = {"colab": "import os, subprocess, sys", "kaggle": "import glob, os, subprocess, sys"}

_WHEEL_BRANCH = {
    "colab": '''else:
    from google.colab import files
    uploaded = files.upload()  # choose autotune_studio-*.whl
    wheels = sorted(name for name in uploaded if name.endswith(".whl"))
    if not wheels:
        raise SystemExit("No .whl uploaded. Upload autotune_studio-<version>-py3-none-any.whl or set AUTOTUNE_SOURCE.")
    pip_install(f"/content/{wheels[-1]}[hf,train]")
''',
    "kaggle": '''else:
    wheels = sorted(glob.glob("/kaggle/input/**/autotune_studio-*.whl", recursive=True))
    if not wheels:
        raise SystemExit("No autotune_studio-*.whl under /kaggle/input. Add the wheel dataset as an input "
                         "or set AUTOTUNE_SOURCE.")
    print("installing", wheels[-1])
    pip_install(f"{wheels[-1]}[hf,train]")
''',
}

_OPTION_A = (
    "* **(a)** install from the public git repository (default: `{repo}`). No token is needed. "
    "The repository includes the built browser UI.\n"
)
_OPTION_B_PREFIX = "* **(b)** "
_OPTION_B_OLD = "leave `AUTOTUNE_SOURCE` empty and "
_OPTION_B_NEW = 'set `AUTOTUNE_SOURCE = ""` and '


def install_cell(runtime: str) -> str:
    """Return the install cell source for ``runtime`` ("colab" or "kaggle")."""
    return f'''AUTOTUNE_SOURCE = "{REPO_URL}"  # set to "" to install a wheel instead
INSTALL_GPU_EXTRAS = True   # unsloth, bitsandbytes, nvidia-ml-py

{_IMPORTS[runtime]}

def pip_install(*args):
    """Run pip quietly; on failure print its output and raise."""
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")  # never wait for an interactive git login
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args],
                       capture_output=True, text=True, env=env)
    if r.returncode != 0:
        print((r.stdout + r.stderr)[-4000:])
        raise subprocess.CalledProcessError(r.returncode, "pip install")

if AUTOTUNE_SOURCE:
    pip_install(f"autotune-studio[hf,train] @ git+{{AUTOTUNE_SOURCE}}")
{_WHEEL_BRANCH[runtime]}
if INSTALL_GPU_EXTRAS:
    try:
        pip_install("unsloth", "bitsandbytes", "nvidia-ml-py")
    except subprocess.CalledProcessError:
        print("GPU extras failed to install. See https://github.com/unslothai/unsloth for the current "
              "install command; the studio still runs with the Hugging Face backend.")'''


def _render(runtime: str, nb: dict) -> dict:
    found = False
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        if cell["cell_type"] == "code" and src.startswith("AUTOTUNE_SOURCE ="):
            src = install_cell(runtime)
            found = True
        elif cell["cell_type"] == "markdown" and "AUTOTUNE_SOURCE" in src:
            lines = src.splitlines(keepends=True)
            for i, line in enumerate(lines):
                if line.startswith("* **(a)**"):
                    lines[i] = _OPTION_A.format(repo=REPO_URL)
                elif line.startswith(_OPTION_B_PREFIX):
                    lines[i] = line.replace(_OPTION_B_OLD, _OPTION_B_NEW)
            src = "".join(lines)
        cell["source"] = src.splitlines(keepends=True)
    if not found:
        raise SystemExit(f"{runtime}: install cell (starting with 'AUTOTUNE_SOURCE =') not found")
    return nb


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    stale = []
    for runtime in ("colab", "kaggle"):
        path = ROOT / "notebooks" / f"{runtime}.ipynb"
        current = path.read_text(encoding="utf-8")
        rendered = json.dumps(_render(runtime, json.loads(current)), indent=1, ensure_ascii=False) + "\n"
        if rendered != current:
            stale.append(path.name)
            if not args.check:
                path.write_text(rendered, encoding="utf-8")
    if args.check:
        for name in stale:
            print(f"out of sync: notebooks/{name} (run python scripts/notebook_install_cell.py)")
        return 1 if stale else 0
    print("updated:", ", ".join(stale) or "nothing")
    return 0


if __name__ == "__main__":
    sys.exit(main())
