"""Review-phase regression tests: real attacks against a running app (TestClient) + findings.

Each finding in docs/security-review.md section 4 references a test here. Tests for findings that
are still open in code owned by other agents are ``xfail(strict=True)``: they start passing (and
then fail as XPASS, forcing removal of the marker) once the proposed fix lands.
"""

from __future__ import annotations

import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import autotune_studio.training as training_pkg
from autotune_studio.api.app import create_app
from autotune_studio.contracts import CheckpointInfo, ExportRequest, HardwareInfo, ModelConfig
from autotune_studio.core.config import StudioSettings
from autotune_studio.errors import ForbiddenPath, ValidationFailed
from autotune_studio.security import (
    COOKIE_NAME,
    FailedAuthLimiter,
    PathPolicy,
    is_same_origin_request,
    redact_data,
    redaction,
    validate_hf_dataset_name,
    validate_model_name,
)
from autotune_studio.studio import Studio

FAKE_TOKEN = "hf_" + "A1b2C3d4E5" * 3  # matches the hf_ pattern; not a real token

FAKE_HW = HardwareInfo(
    cpu_count=2, ram_total_bytes=8 * 1024**3, ram_available_bytes=4 * 1024**3, disk_path=".",
    disk_total_bytes=100 * 1024**3, disk_free_bytes=50 * 1024**3, cuda_available=False,
)


# --------------------------------------------------------------------------- fixtures
@pytest.fixture(autouse=True)
def _restore_redaction() -> Iterator[None]:
    before = set(redaction._secrets)
    yield
    for s in set(redaction._secrets) - before:
        redaction.unregister_secret(s)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    r = tmp_path / "root"
    r.mkdir()
    return r


@pytest.fixture
def outside(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A directory outside every allowed root, holding a 'secret' CSV."""
    d = tmp_path_factory.mktemp("outside")
    (d / "creds.csv").write_text("user,password\nadmin,hunter2\n", encoding="utf-8")
    return d


@pytest.fixture
def studio(root: Path) -> Iterator[Studio]:
    s = Studio(workspace_dir=root / "ws", allowed_roots=[str(root)], resource_interval=0.05)
    s._hardware = FAKE_HW
    yield s
    try:
        if s.lifecycle.is_active:
            s.lifecycle.stop(wait=True, timeout=10)
    finally:
        s.sampler.stop()


@pytest.fixture
def web_dir(tmp_path: Path) -> Path:
    d = tmp_path / "web"
    (d / "assets").mkdir(parents=True)
    (d / "index.html").write_text("<!doctype html><title>SPA</title>", encoding="utf-8")
    return d


@pytest.fixture
def app(studio: Studio, web_dir: Path):
    return create_app(studio, StudioSettings(upload_max_bytes=64 * 1024, ws_ping_interval=30.0),
                      web_dir=web_dir, auth_limiter=FailedAuthLimiter(max_per_minute=1000))


@pytest.fixture
def anon(app) -> Iterator[TestClient]:
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture
def client(app, studio: Studio) -> Iterator[TestClient]:
    with TestClient(app, raise_server_exceptions=False) as c:
        c.headers["Authorization"] = f"Bearer {studio.auth.token}"
        yield c


@pytest.fixture
def cookie_client(app, studio: Studio) -> Iterator[TestClient]:
    """Browser-like client: session cookie obtained through POST /api/auth/session."""
    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.post("/api/auth/session", json={"token": studio.auth.token})
        assert r.status_code == 200
        assert c.cookies.get(COOKIE_NAME)
        yield c


@pytest.fixture
def csv_in_root(root: Path) -> Path:
    p = root / "data" / "train.csv"
    p.parent.mkdir(parents=True)
    p.write_text("instruction,output\n" + "".join(f"q{i},a{i}\n" for i in range(20)), encoding="utf-8")
    return p


# =========================================================================== SEC-01 HF name = local path
def test_sec01_validate_hf_dataset_name_unit(outside: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert validate_hf_dataset_name("tatsu-lab/alpaca") == "tatsu-lab/alpaca"
    assert validate_hf_dataset_name("  imdb ") == "imdb"
    for bad in [str(outside), str(outside / "creds.csv"), "./data", "../x", "~/x", "C:\\x", "C:x",
                "https://evil/x", "hf://datasets/x", "a/../b", "a\x00b", "", "x" * 300]:
        with pytest.raises((ValidationFailed, ForbiddenPath)):
            validate_hf_dataset_name(bad)
    for builder in ["csv", "JSON", "text", "parquet", "arrow", "pandas", "imagefolder", "webdataset"]:
        with pytest.raises(ValidationFailed):
            validate_hf_dataset_name(builder)
    monkeypatch.chdir(outside.parent)
    with pytest.raises(ForbiddenPath):  # repo-id-shaped but exists relative to the cwd
        validate_hf_dataset_name(outside.name)


def test_sec01_hf_source_cannot_read_local_dir_outside_roots(client: TestClient, outside: Path) -> None:
    r = client.post("/api/datasets", json={"source": "huggingface", "name": str(outside)})
    assert r.status_code in (403, 422), r.text[:300]


def test_sec01_hf_packaged_builder_cannot_read_cwd(
    client: TestClient, outside: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(outside)  # cwd outside the roots, containing creds.csv
    r = client.post("/api/datasets", json={"source": "huggingface", "name": "csv"})
    assert r.status_code in (403, 422), r.text[:300]


# =========================================================================== SEC-02 path / upload attacks
def test_file_source_paths_confined(client: TestClient, outside: Path, csv_in_root: Path, root: Path) -> None:
    attempts = [
        str(outside / "creds.csv"),
        str(root / ".." / outside.name / "creds.csv"),
        str(csv_in_root.parent / ".." / ".." / ".." / outside.name / "creds.csv"),
        str(csv_in_root) + "\x00.csv",
        str(csv_in_root) + ":ads.csv",
    ]
    for p in attempts:
        r = client.post("/api/datasets", json={"source": "csv", "path": p})
        assert r.status_code in (403, 422), (p, r.status_code, r.text[:200])
        assert "hunter2" not in r.text
    assert client.post("/api/datasets", json={"source": "csv", "path": str(csv_in_root)}).status_code == 200


@pytest.mark.skipif(sys.platform != "win32", reason="NTFS junctions are Windows-only")
def test_junction_escape_via_api(client: TestClient, outside: Path, root: Path) -> None:
    link = root / "jx"
    r = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(outside)], capture_output=True, text=True,
                       check=False)
    if r.returncode != 0:
        pytest.skip(f"mklink /J failed: {r.stdout}{r.stderr}")
    resp = client.post("/api/datasets", json={"source": "csv", "path": str(link / "creds.csv")})
    assert resp.status_code == 403
    assert "hunter2" not in resp.text


def test_symlink_escape_via_api(client: TestClient, outside: Path, root: Path) -> None:
    link = root / "ln.csv"
    try:
        link.symlink_to(outside / "creds.csv")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not permitted for this user")
    resp = client.post("/api/datasets", json={"source": "csv", "path": str(link)})
    assert resp.status_code == 403


def _multipart(filename_header: str, body: bytes, boundary: str = "XyZ") -> tuple[bytes, dict[str, str]]:
    data = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; {filename_header}\r\n"
        f"Content-Type: text/csv\r\n\r\n"
    ).encode() + body + f"\r\n--{boundary}--\r\n".encode()
    return data, {"content-type": f"multipart/form-data; boundary={boundary}"}


@pytest.mark.parametrize(
    "fname",
    [
        'filename="../../../evil.csv"',
        'filename="..\\\\..\\\\evil.csv"',
        'filename="C:\\\\Windows\\\\evil.csv"',
        "filename*=UTF-8''..%2F..%2Fevil.csv",
        'filename="CON.csv"',
        'filename="evil.csv:stream"',
        'filename="/etc/cron.d/evil.csv"',
    ],
)
def test_upload_filename_traversal_never_escapes(client: TestClient, studio: Studio, root: Path, fname: str) -> None:
    body, headers = _multipart(fname, b"a,b\n1,2\n3,4\n")
    r = client.post("/api/datasets/upload", content=body, headers=headers)
    if r.status_code == 200:
        stored = Path(r.json()["spec"]["path"])
        assert stored.parent == (studio.workspace / "uploads").resolve()
        assert "evil" not in stored.name and len(stored.stem) == 32
    else:
        assert r.status_code == 422
    assert not list(root.parent.rglob("evil*"))


@pytest.mark.parametrize("fname", ['filename="x.py"', 'filename="x.pkl"', 'filename="x.csv.exe"', 'filename="noext"'])
def test_upload_extension_allowlist(client: TestClient, studio: Studio, fname: str) -> None:
    body, headers = _multipart(fname, b"print(1)\n")
    r = client.post("/api/datasets/upload", content=body, headers=headers)
    assert r.status_code == 422
    uploads = studio.workspace / "uploads"
    assert not uploads.exists() or not any(uploads.iterdir())


def test_upload_oversized_rejected_and_cleaned(client: TestClient, studio: Studio) -> None:
    body, headers = _multipart('filename="big.csv"', b"a,b\n" + b"1,2\n" * 40_000)  # > 64 KiB limit
    r = client.post("/api/datasets/upload", content=body, headers=headers)
    assert r.status_code == 422
    # Without Content-Length (chunked) the streaming writer enforces the limit.
    def gen():
        yield body
    r = client.post("/api/datasets/upload", content=gen(), headers=headers)
    assert r.status_code == 422
    uploads = studio.workspace / "uploads"
    assert not uploads.exists() or not any(uploads.iterdir())


@pytest.mark.parametrize(
    "body,ctype",
    [
        (b"--b\r\n" + b"X-H: " + b"a" * 20_000 + b"\r\n\r\nx\r\n--b--\r\n", "multipart/form-data; boundary=b"),
        (b"".join(b"--b\r\nContent-Disposition: form-data; name=\"f%d\"\r\n\r\nx\r\n" % i for i in range(40))
         + b"--b--\r\n", "multipart/form-data; boundary=b"),
        (b"--b\r\nno-colon-header\r\n\r\nx\r\n--b--\r\n", "multipart/form-data; boundary=b"),
        (b"garbage", "multipart/form-data; boundary=" + "b" * 300),
        (b"garbage", "application/json"),
        (b"--b\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.csv\"\r\n\r\nunterminated",
         "multipart/form-data; boundary=b"),
    ],
)
def test_malicious_multipart_is_client_error(client: TestClient, body: bytes, ctype: str) -> None:
    r = client.post("/api/datasets/upload", content=body, headers={"content-type": ctype})
    assert r.status_code == 422, r.text[:200]


def test_preview_n_is_bounded(client: TestClient, csv_in_root: Path) -> None:
    did = client.post("/api/datasets", json={"source": "csv", "path": str(csv_in_root)}).json()["dataset_id"]
    for n in ("1000000000", "-1", "1e9", "abc"):
        assert client.get(f"/api/datasets/{did}/preview?n={n}").status_code == 422
    assert client.get(f"/api/datasets/{did}/preview?n=1000").status_code == 200
    assert client.get("/api/datasets/..%2f..%2fetc/preview").status_code in (404, 422)


def test_xss_payload_in_dataset_is_inert_json(client: TestClient, root: Path) -> None:
    """Dataset values are attacker-controlled; the API must return them as JSON data, never HTML.

    The SPA renders cells as React text children (no dangerouslySetInnerHTML anywhere in
    frontend/src — verified by review/grep), and the HTML shell carries ``script-src 'self'``.
    """
    payload = '<img src=x onerror=alert(1)><script>alert(2)</script>'
    p = root / "xss.csv"
    p.write_text(f'"{payload}",col2\n"{payload}","</td><td>x"\n', encoding="utf-8")
    did = client.post("/api/datasets", json={"source": "csv", "path": str(p)}).json()["dataset_id"]
    r = client.get(f"/api/datasets/{did}/preview?n=5")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/json")
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["cache-control"] == "no-store"
    assert r.json()["rows"][0][r.json()["columns"][0]] == payload  # data preserved, not interpreted


def test_frontend_has_no_raw_html_sinks() -> None:
    src = Path(__file__).resolve().parents[3] / "frontend" / "src"
    if not src.is_dir():
        pytest.skip("frontend sources not present")
    hits = [
        f"{f}: {needle}"
        for f in src.rglob("*.ts*")
        if "test" not in f.parts
        for needle in ("dangerouslySetInnerHTML", ".innerHTML", "eval(", "new Function(", "document.write")
        if needle in f.read_text(encoding="utf-8")
    ]
    assert hits == []


# =========================================================================== auth surface
_PUBLIC = {("GET", "/api/health"), ("HEAD", "/api/health"), ("POST", "/api/auth/session")}


def _api_routes():
    """(method, '/api' + path) for every route of every API router module (incl. auth)."""
    import importlib

    for mod in ("auth", "system", "datasets", "models", "training", "checkpoints"):
        router = importlib.import_module(f"autotune_studio.api.routes.{mod}").router
        for route in router.routes:
            for method in sorted(getattr(route, "methods", None) or {"GET"}):
                yield method, "/api" + route.path


def test_every_api_route_requires_auth(app, anon: TestClient) -> None:
    """Enumerate the router tables: every /api route other than health + auth/session is 401.

    (FastAPI 0.16x mounts included routers lazily, so ``app.routes`` does not list them.)
    """
    checked = 0
    for method, path in _api_routes():
        if (method, path) in _PUBLIC:
            continue
        url = path.replace("{dataset_id}", "abc").replace("{checkpoint_id}", "abc")
        r = anon.request(method, url)
        assert r.status_code == 401, (method, path, r.status_code)
        assert r.headers.get("www-authenticate") == "Bearer"
        checked += 1
    assert checked >= 25


def test_no_interactive_docs_or_openapi(anon: TestClient) -> None:
    for p in ("/docs", "/redoc", "/openapi.json", "/api/openapi.json", "/api/docs"):
        r = anon.get(p)
        assert "openapi" not in r.text.lower() and "swagger" not in r.text.lower()


def test_token_never_in_query_or_responses(client: TestClient, studio: Studio) -> None:
    tok = studio.auth.token
    for p in ("/api/env", "/api/launch", "/api/training/status", "/api/training/config", "/api/datasets"):
        assert tok not in client.get(p).text
    with TestClient(client.app, raise_server_exceptions=False) as c:  # query-string token is not a credential
        assert c.get(f"/api/env?token={tok}").status_code == 401
        assert c.get(f"/api/env?access_token={tok}").status_code == 401


def test_static_traversal_serves_spa_only(anon: TestClient, tmp_path: Path) -> None:
    (tmp_path / "secret.txt").write_text("TOPSECRET", encoding="utf-8")
    for p in ("/../secret.txt", "/%2e%2e/secret.txt", "/..%2fsecret.txt", "/..%5csecret.txt",
              "/%2e%2e%5csecret.txt", "/C:/Windows/win.ini", "/C:%5cWindows%5cwin.ini",
              "/assets/..%2f..%2fsecret.txt", "/api/..%2f..%2fsecret.txt"):
        r = anon.get(p)
        assert "TOPSECRET" not in r.text and "[fonts]" not in r.text.lower(), p


def _ws(app, headers: dict[str, str], cookie: str | None = None) -> str:
    with TestClient(app, raise_server_exceptions=False) as c:
        if cookie:
            c.cookies.set(COOKIE_NAME, cookie)
        try:
            with c.websocket_connect("/ws", headers=headers) as w:
                return "open:" + w.receive_json()["type"]
        except WebSocketDisconnect as e:
            return f"closed:{e.code}"


def test_websocket_auth_and_origin_matrix(app, studio: Studio) -> None:
    cookie = studio.auth.cookie_value()
    assert _ws(app, {}) == "closed:1008"
    assert _ws(app, {"origin": "http://testserver"}) == "closed:1008"
    assert _ws(app, {}, cookie=studio.auth.token) == "closed:1008"  # raw token is not a valid cookie
    assert _ws(app, {"origin": "https://evil.example"}, cookie) == "closed:1008"
    assert _ws(app, {"origin": "null"}, cookie) == "closed:1008"
    assert _ws(app, {"origin": "http://testserver.evil.example"}, cookie) == "closed:1008"
    assert _ws(app, {"origin": "http://testserver"}, cookie) == "open:hello"
    assert _ws(app, {"authorization": f"Bearer {studio.auth.token}"}) == "open:hello"


def test_session_cookie_flags(anon: TestClient, studio: Studio) -> None:
    r = anon.post("/api/auth/session", json={"token": studio.auth.token}, headers={"x-forwarded-proto": "https"})
    sc = r.headers["set-cookie"].lower()
    assert "httponly" in sc and "samesite=strict" in sc and "secure" in sc and "path=/" in sc
    assert studio.auth.token.lower() not in sc
    assert "max-age" not in sc and "expires" not in sc


# =========================================================================== SEC-03 CSRF origin check
@pytest.mark.parametrize(
    "method,path,kwargs",
    [
        ("POST", "/api/training/stop", {}),
        ("POST", "/api/checkpoints", {}),
        ("PUT", "/api/model", {"json": {"model_name": "gpt2"}}),
        ("POST", "/api/datasets/upload", {"content": b"--b--\r\n",
                                          "headers": {"content-type": "multipart/form-data; boundary=b"}}),
    ],
)
def test_sec03_cookie_csrf_cross_origin_rejected(cookie_client: TestClient, method: str, path: str, kwargs) -> None:
    headers = {**kwargs.pop("headers", {}), "origin": "https://evil.trycloudflare.com"}
    r = cookie_client.request(method, path, headers=headers, **kwargs)
    assert r.status_code == 401, r.text[:200]
    r = cookie_client.request(method, path, headers={**headers, "origin": "null"}, **kwargs)
    assert r.status_code == 401


def test_sec03_sec_fetch_site_without_origin(cookie_client: TestClient) -> None:
    assert cookie_client.post("/api/training/stop", headers={"sec-fetch-site": "cross-site"}).status_code == 401
    assert cookie_client.post("/api/training/stop", headers={"sec-fetch-site": "same-site"}).status_code == 401
    assert cookie_client.post("/api/training/stop", headers={"sec-fetch-site": "same-origin"}).status_code == 200


def test_sec03_legit_requests_still_work(cookie_client: TestClient, client: TestClient, studio: Studio) -> None:
    assert cookie_client.post("/api/training/stop", headers={"origin": "http://testserver"}).status_code == 200
    assert cookie_client.post("/api/training/stop").status_code == 200  # non-browser: no Origin
    assert cookie_client.get("/api/env", headers={"origin": "https://evil.example"}).status_code == 200  # safe
    # Bearer is not ambient: cross-origin script clients (CORS opt-in) keep working.
    assert client.post("/api/training/stop", headers={"origin": "https://other.example"}).status_code == 200
    studio._extra_origins.add("https://abc.trycloudflare.com")  # tunnel origin known after launch
    r = cookie_client.post("/api/training/stop", headers={"origin": "https://abc.trycloudflare.com"})
    assert r.status_code == 200


def test_sec03_is_same_origin_request_unit() -> None:
    h = {"host": "x.trycloudflare.com"}
    assert is_same_origin_request("GET", {**h, "origin": "https://evil"})
    assert is_same_origin_request("POST", {**h, "origin": "https://x.trycloudflare.com"})
    assert not is_same_origin_request("POST", {**h, "origin": "https://y.trycloudflare.com"})
    assert not is_same_origin_request("DELETE", {**h, "origin": "null"})
    assert is_same_origin_request("POST", h)
    assert not is_same_origin_request("PUT", {**h, "sec-fetch-site": "cross-site"})
    assert is_same_origin_request("POST", {"host": "127.0.0.1:1", "origin": "https://p.example"}, ["https://p.example"])


# =========================================================================== SEC-04 secrets in events
def test_sec04_redact_data_unit() -> None:
    data = {"message": f"boom {FAKE_TOKEN}", "nested": [{"m": f"Bearer {FAKE_TOKEN}"}, 3, None],
            "t": ("token=abcdef123",), "n": 1.5}
    out = redact_data(data)
    assert FAKE_TOKEN not in repr(out)
    assert out["n"] == 1.5 and out["nested"][1] == 3 and "token=***" in out["t"][0]


def _start_failing_job(studio: Studio, client: TestClient, csv: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(training_pkg, "plan_steps", lambda cfg, info: 2, raising=False)
    monkeypatch.setattr(training_pkg, "validate_training_request", lambda *a: [], raising=False)

    class Boom:
        backend = "hf"

        def run(self, *a, **k):
            raise RuntimeError(f"401 Client Error for url https://huggingface.co/x?token={FAKE_TOKEN} {FAKE_TOKEN}")

    studio.lifecycle.engine_factory = lambda m, t: Boom()
    assert client.post("/api/datasets", json={"source": "csv", "path": str(csv)}).status_code == 200
    assert client.put("/api/model", json={"model_name": "sshleifer/tiny-gpt2"}).status_code == 200
    assert client.post("/api/training/start").status_code == 200, client.get("/api/training/status").text
    deadline = time.monotonic() + 10
    while studio.lifecycle.is_active and time.monotonic() < deadline:
        time.sleep(0.02)


def test_sec04_exception_text_with_token_not_exposed(
    studio: Studio, client: TestClient, csv_in_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _start_failing_job(studio, client, csv_in_root, monkeypatch)
    events, _ = studio.events.replay(0)
    assert any(e.type == "error" for e in events)
    assert FAKE_TOKEN not in "".join(e.model_dump_json() for e in events)
    assert FAKE_TOKEN not in client.get("/api/training/status").text


# =========================================================================== SEC-05 model name vs cwd
def test_sec05_repo_id_shaped_name_cannot_select_local_dir_outside_roots(
    outside: Path, root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (outside / "mymodel").mkdir()
    monkeypatch.chdir(outside)
    policy = PathPolicy([root])
    with pytest.raises(ForbiddenPath):
        validate_model_name("mymodel", policy)
    with pytest.raises(ValidationFailed):
        validate_model_name("mymodel")
    assert validate_model_name("owner/not-a-local-dir", policy) == "owner/not-a-local-dir"
    (root / "inmodel").mkdir()
    monkeypatch.chdir(root)  # inside the roots: pinned to the absolute directory
    assert Path(validate_model_name("inmodel", policy)) == (root / "inmodel").resolve()


def test_trust_remote_code_cannot_be_enabled(client: TestClient) -> None:
    r = client.put("/api/model", json={"model_name": "gpt2", "trust_remote_code": True})
    assert r.status_code == 422
    assert client.put("/api/model", json={"model_name": "gpt2"}).json()["trust_remote_code"] is False


def test_output_dirs_confined(client: TestClient, outside: Path, studio: Studio, monkeypatch) -> None:
    for bad in (str(outside / "out"), "../../x", "\\\\?\\C:\\x"):
        with pytest.raises(ForbiddenPath):
            studio.path_policy.resolve(bad, must_exist=False)


# =========================================================================== SEC-06 export .partial rmtree
def test_sec06_export_does_not_delete_preexisting_partial_dir(root: Path) -> None:
    from autotune_studio.export.model_export import export_model

    ck = root / "runs" / "job" / "checkpoint-1"
    ck.mkdir(parents=True)
    (ck / "adapter_model.safetensors").write_bytes(b"x")
    victim = root / "MyProject.partial"
    victim.mkdir()
    (victim / "thesis.docx").write_text("irreplaceable", encoding="utf-8")
    info = CheckpointInfo(checkpoint_id="job-step1", path=str(ck), step=1, created_at=0.0, size_bytes=1)
    export_model(ExportRequest(output_dir=str(root / "MyProject")), info, ModelConfig(model_name="gpt2"), root)
    assert (victim / "thesis.docx").is_file()
