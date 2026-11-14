"""Every file format x record layout through HTTP: load (no row count) -> preview -> full scan -> exact count;
files untouched; JSON-lines content with a ``.json`` extension auto-detected."""

from __future__ import annotations

import pytest

from ..conftest import sha256_file

FORMATS = ["csv", "jsonl", "json", "parquet"]
KINDS = ["instruction", "chat", "text"]
EXPECTED_FORMAT = {"instruction": "instruction", "chat": "chat", "text": "text"}


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("kind", KINDS)
def test_format_matrix_load_preview_scan(auth_client, make_dataset, fmt, kind) -> None:
    if fmt == "csv" and kind == "chat":
        pytest.skip("nested chat messages have no CSV representation")
    n = 257
    p = make_dataset(fmt, kind, n=n)
    h = sha256_file(p)
    r = auth_client.post("/api/datasets", json={"source": fmt, "path": str(p)})
    assert r.status_code == 200, r.text
    info = r.json()
    assert info["detected_format"] == EXPECTED_FORMAT[kind], info
    assert info["text_columns"]
    if fmt == "parquet":
        assert info["cardinality"] == {"rows": n, "exact": True, "method": "metadata"}
    else:
        assert info["cardinality"]["rows"] is None
    did = info["dataset_id"]
    pv = auth_client.get(f"/api/datasets/{did}/preview", params={"n": 3}).json()
    assert len(pv["rows"]) == 3

    st = auth_client.app.state.studio.scan_dataset(did)
    assert st.state == "exhausted" and st.rows_seen == n and st.malformed_rows == 0
    assert auth_client.get(f"/api/datasets/{did}").json()["cardinality"]["rows"] == n
    assert sha256_file(p) == h


def test_jsonl_content_with_json_extension_is_detected(auth_client, make_dataset, tmp_path) -> None:
    src = make_dataset("jsonl", "text", n=20)
    p = tmp_path / "data" / "actually-lines.json"
    p.write_bytes(src.read_bytes())
    r = auth_client.post("/api/datasets", json={"source": "json", "path": str(p)})
    assert r.status_code == 200, r.text
    st = auth_client.app.state.studio.scan_dataset(r.json()["dataset_id"])
    assert st.rows_seen == 20


def test_background_scan_via_http_reaches_exact_cardinality(auth_client, make_dataset) -> None:
    from .conftest import wait_until

    p = make_dataset("jsonl", "instruction", n=5000)
    did = auth_client.post("/api/datasets", json={"source": "jsonl", "path": str(p)}).json()["dataset_id"]
    r = auth_client.post(f"/api/datasets/{did}/scan")
    assert r.status_code == 200
    wait_until(lambda: auth_client.get(f"/api/datasets/{did}/stats").json()["state"] == "exhausted", timeout=60)
    assert auth_client.get(f"/api/datasets/{did}").json()["cardinality"] == {
        "rows": 5000, "exact": True, "method": "scan"}
