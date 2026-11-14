"""Dataset routes: load, upload, list, preview, stats, scan, cancel, path policy."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from .conftest import wait_for


def test_load_csv_and_inspect(client: TestClient, csv_file: Path) -> None:
    r = client.post("/api/datasets", json={"source": "csv", "path": str(csv_file)})
    assert r.status_code == 200, r.text
    info = r.json()
    did = info["dataset_id"]
    assert [c["name"] for c in info["columns"]] == ["instruction", "output"]

    assert [d["dataset_id"] for d in client.get("/api/datasets").json()] == [did]
    assert client.get(f"/api/datasets/{did}").json()["dataset_id"] == did

    prev = client.get(f"/api/datasets/{did}/preview", params={"n": 5}).json()
    assert prev["columns"] == ["instruction", "output"]
    assert len(prev["rows"]) == 5
    assert prev["rows"][0] == {"instruction": "question 0", "output": "answer 0"}

    stats = client.get(f"/api/datasets/{did}/stats")
    assert stats.status_code == 200
    assert "rows_seen" in stats.json()


def test_scan_then_stats_and_cancel(client: TestClient, dataset_id: str) -> None:
    r = client.post(f"/api/datasets/{dataset_id}/scan")
    assert r.status_code in (200, 409), r.text
    assert wait_for(lambda: client.get(f"/api/datasets/{dataset_id}/stats").json()["state"] == "exhausted")
    assert client.get(f"/api/datasets/{dataset_id}/stats").json()["rows_seen"] == 50
    r = client.post(f"/api/datasets/{dataset_id}/cancel")
    assert r.status_code == 200
    assert "state" in r.json()


def test_jsonl_load(client: TestClient, jsonl_file: Path) -> None:
    r = client.post("/api/datasets", json={"source": "jsonl", "path": str(jsonl_file)})
    assert r.status_code == 200, r.text
    assert "text" in r.json()["text_columns"] or r.json()["detected_format"] == "text"


def test_path_outside_allowed_roots_forbidden(client: TestClient, tmp_path: Path) -> None:
    outside = Path(tmp_path.anchor) / "Windows" / "win.ini" if tmp_path.anchor else Path("/etc/passwd")
    r = client.post("/api/datasets", json={"source": "csv", "path": str(outside.with_suffix(".csv"))})
    assert r.status_code == 403
    assert r.json()["code"] == "forbidden_path"
    r = client.post("/api/datasets", json={"source": "csv", "path": str(tmp_path / ".." / ".." / "x.csv")})
    assert r.status_code == 403


def test_spec_validation_422(client: TestClient) -> None:
    r = client.post("/api/datasets", json={"source": "csv"})  # path missing
    assert r.status_code == 422
    body = r.json()
    assert body["code"] == "validation_error"
    assert isinstance(body["detail"]["errors"], list) and body["detail"]["errors"]
    assert {"loc", "msg", "type"} <= set(body["detail"]["errors"][0])


def test_unknown_dataset_404(client: TestClient) -> None:
    r = client.get("/api/datasets/doesnotexist/stats")
    assert r.status_code == 404
    assert r.json()["code"] == "not_found"


def test_preview_n_out_of_range_422(client: TestClient, dataset_id: str) -> None:
    for n in (-1, 1001):
        r = client.get(f"/api/datasets/{dataset_id}/preview", params={"n": n})
        assert r.status_code == 422, n
        assert r.json()["detail"]["errors"][0]["loc"] == ["query", "n"]
    assert client.get(f"/api/datasets/{dataset_id}/preview", params={"n": 0}).status_code == 200


def test_upload_csv(client: TestClient, studio) -> None:
    content = b"instruction,output\n" + b"".join(b"q%d,a%d\n" % (i, i) for i in range(20))
    r = client.post("/api/datasets/upload", files={"file": ("../../evil name.csv", content, "text/csv")})
    assert r.status_code == 200, r.text
    info = r.json()
    stored = Path(info["spec"]["path"])
    assert stored.parent == (studio.workspace / "uploads").resolve()
    assert stored.suffix == ".csv" and "evil" not in stored.name
    assert stored.read_bytes() == content
    assert info["spec"]["source"] == "csv"
    prev = client.get(f"/api/datasets/{info['dataset_id']}/preview", params={"n": 2}).json()
    assert prev["rows"][0] == {"instruction": "q0", "output": "a0"}


def test_upload_jsonl_with_extra_fields(client: TestClient) -> None:
    content = b"".join(b'{"text": "row %d"}\n' % i for i in range(10))
    r = client.post(
        "/api/datasets/upload",
        data={"note": "ignored"},
        files={"file": ("data.ndjson", content, "application/octet-stream")},
    )
    assert r.status_code == 200, r.text
    assert r.json()["spec"]["source"] == "jsonl"


def test_upload_bad_extension_422(client: TestClient, studio) -> None:
    r = client.post("/api/datasets/upload", files={"file": ("payload.exe", b"MZ....", "application/octet-stream")})
    assert r.status_code == 422
    assert r.json()["code"] == "validation_error"
    uploads = studio.workspace / "uploads"
    assert not uploads.exists() or not any(uploads.iterdir())


def test_upload_missing_file_field_and_not_multipart(client: TestClient) -> None:
    r = client.post("/api/datasets/upload", data={"other": "x"}, files={"x": ("a.csv", b"a\n1\n")})
    assert r.status_code == 422
    r = client.post("/api/datasets/upload", json={"file": "a.csv"})
    assert r.status_code == 422


def test_upload_too_large_rejected_and_cleaned(client: TestClient, studio) -> None:
    big = b"a,b\n" + b"1,2\n" * (300 * 1024)  # ~1.2 MiB > settings.upload_max_bytes (1 MiB)
    r = client.post("/api/datasets/upload", files={"file": ("big.csv", big, "text/csv")})
    assert r.status_code == 422
    assert "max" in r.json()["message"].lower()
    uploads = studio.workspace / "uploads"
    assert not uploads.exists() or not any(uploads.iterdir())


def test_upload_unparseable_file_is_deleted(client: TestClient, studio) -> None:
    r = client.post("/api/datasets/upload", files={"file": ("broken.parquet", b"not a parquet file", "x")})
    assert r.status_code == 422
    assert r.json()["code"] == "dataset_malformed"
    assert "Traceback" not in r.text
    uploads = studio.workspace / "uploads"
    assert not uploads.exists() or not any(uploads.iterdir())
