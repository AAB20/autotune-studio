"""Single source of truth for the install cell of ``notebooks/colab.ipynb`` and ``notebooks/kaggle.ipynb``.

The two notebooks share the install logic; only the secret lookup and the wheel location differ.

Usage::

    python scripts/notebook_install_cell.py          # rewrite the install cells (and option text) in both notebooks
    python scripts/notebook_install_cell.py --check  # exit 1 if a notebook is out of sync (used by tests)

Security: the GitHub token is never put in the git URL. It reaches git only through
environment-only config (``GIT_CONFIG_COUNT`` / ``http.<url>.extraHeader``) of the pip subprocess,
so it is not written to installed-package metadata (``direct_url.json``), not shown by
``pip freeze``, not stored in ``~/.gitconfig`` and not visible in the process arguments.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_URL = "https://github.com/AAB20/autotune-studio.git"

_SECRET = {
    "colab": (
        "        from google.colab import userdata\n"
        '        return userdata.get("GITHUB_TOKEN")\n'
    ),
    "kaggle": (
        "        from kaggle_secrets import UserSecretsClient\n"
        '        return UserSecretsClient().get_secret("GITHUB_TOKEN")\n'
    ),
}

_SECRET_WHERE = {
    "colab": "the Colab *Secrets* panel (key icon) and enable *Notebook access*",
    "kaggle": "Kaggle *Add-ons > Secrets* and attach it to this notebook",
}

_IMPORTS = {"colab": "import base64, os, subprocess, sys", "kaggle": "import base64, glob, os, subprocess, sys"}

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
    "* **(a)** install from the git repository (default: `{repo}`). The repository includes the built browser UI. "
    "**The repository is private:** create a read-only GitHub token (fine-grained, *Contents: Read* on this "
    "repository) and store it as `GITHUB_TOKEN` in {where}. The token is passed to git only through the "
    "environment of the install command: it is not put in the URL, not printed, and not saved in the installed "
    "package's metadata.\n"
)
_OPTION_B_PREFIX = "* **(b)** "
_OPTION_B_OLD = "leave `AUTOTUNE_SOURCE` empty and "
_OPTION_B_NEW = 'set `AUTOTUNE_SOURCE = ""` and '


def install_cell(runtime: str) -> str:
    """Return the install cell source for ``runtime`` ("colab" or "kaggle")."""
    return f'''AUTOTUNE_SOURCE = "{REPO_URL}"  # set to "" to install a wheel instead
INSTALL_GPU_EXTRAS = True   # unsloth, bitsandbytes, nvidia-ml-py

{_IMPORTS[runtime]}

def _github_token():
    try:
{_SECRET[runtime]}    except Exception as exc:  # secret missing or notebook access not granted
        print(f"No GITHUB_TOKEN secret available ({{type(exc).__name__}}).")
        return None

def pip_install(*args, token=None):
    """Run pip quietly; on failure print its output (secrets masked) and raise."""
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")  # never wait for an interactive git login
    masks = []
    if token:
        # Environment-only git config: the token never appears in the URL, argv, ~/.gitconfig or
        # the installed package's direct_url.json (which `pip freeze` prints).
        basic = base64.b64encode(f"x-access-token:{{token}}".encode()).decode()
        env.update(GIT_CONFIG_COUNT="1",
                   GIT_CONFIG_KEY_0="http.https://github.com/.extraheader",
                   GIT_CONFIG_VALUE_0=f"Authorization: Basic {{basic}}")
        masks = [token, basic]
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args],
                       capture_output=True, text=True, env=env)
    if r.returncode != 0:
        out = r.stdout + r.stderr
        for secret in masks:
            out = out.replace(secret, "***")
        print(out[-4000:])
        raise subprocess.CalledProcessError(r.returncode, "pip install")

if AUTOTUNE_SOURCE:
    token = _github_token()
    try:
        pip_install(f"autotune-studio[hf,train] @ git+{{AUTOTUNE_SOURCE}}", token=token)
    except subprocess.CalledProcessError:
        if not token:
            print("Installing from the private repository needs a GitHub token: store a read-only token as "
                  "GITHUB_TOKEN in {_SECRET_WHERE[runtime]}, then re-run this cell. "
                  'Or set AUTOTUNE_SOURCE = "" to install a wheel instead.')
        raise
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
                    lines[i] = _OPTION_A.format(repo=REPO_URL, where=_SECRET_WHERE[runtime])
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
