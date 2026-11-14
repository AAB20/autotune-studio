"""Importing the package must not pull heavy optional deps (owned by chief-architect)."""

import subprocess
import sys

HEAVY = ["torch", "transformers", "datasets", "peft", "trl", "unsloth"]


def test_light_import_does_not_load_heavy_deps():
    code = (
        "import sys, autotune_studio, autotune_studio.contracts, autotune_studio.interfaces, "
        "autotune_studio.core.events\n"
        "try:\n    import autotune_studio.studio\nexcept ModuleNotFoundError:\n    pass\n"
        f"print([m for m in {HEAVY!r} if m in sys.modules])"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]", out.stdout + out.stderr
