"""The built UI is committed (so `pip install git+...` ships it); its asset references must resolve."""

import re
from pathlib import Path

WEB = Path(__file__).resolve().parents[2] / "autotune_studio" / "web"


def test_committed_ui_bundle_is_consistent():
    index = (WEB / "index.html").read_text(encoding="utf-8")
    refs = re.findall(r'(?:src|href)="\./(assets/[^"]+)"', index)
    assert refs, "index.html references no assets"
    for ref in refs:
        assert (WEB / ref).is_file(), f"missing {ref}; run `cd frontend && npm run build`"
