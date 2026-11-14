"""Log redaction and HF token discovery."""

from __future__ import annotations

import logging
import sys
import types

import pytest

from autotune_studio.security import redaction
from autotune_studio.security import secrets as sec
from autotune_studio.security.authentication import SessionAuth
from autotune_studio.security.redaction import RedactingFilter, install_redaction, redact, register_secret

HF = "hf_" + "AbCdEfGhIjKlMnOpQrStUvWxYz012345"

pytestmark = pytest.mark.usefixtures("clean_redaction")


# ----------------------------------------------------------------- redact()
@pytest.mark.parametrize(
    "text,leak",
    [
        (f"using token {HF} for download", HF),
        ("Authorization: Bearer abc.DEF-123_xyz~", "abc.DEF-123_xyz~"),
        ("GET /api/x?token=s3cr3tvalue&x=1 HTTP/1.1", "s3cr3tvalue"),
        ("open http://host/#token=fragmentsecret", "fragmentsecret"),
        ("hf_token=zzzzzzzzzz", "zzzzzzzzzz"),
        ("access_token%3Dencodedsecret", "encodedsecret"),
        ("Cookie: autotune_session=deadbeefcafebabe; other=1", "deadbeefcafebabe"),
    ],
)
def test_redact_patterns(text, leak):
    out = redact(text)
    assert leak not in out
    assert "***" in out


def test_redact_keeps_normal_text_and_short_hf_prefix():
    assert redact("loss=0.123 step=5") == "loss=0.123 step=5"
    assert redact("hf_short") == "hf_short"
    assert redact("") == ""


def test_registered_and_extra_secrets():
    register_secret("my-very-secret-value-123")
    register_secret("short")  # ignored (< MIN_SECRET_LENGTH)
    register_secret(None)
    assert redact("x my-very-secret-value-123 y") == "x *** y"
    assert redact("a short b") == "a short b"
    assert redact("p extra-secret-xyz q", extra_secrets=["extra-secret-xyz"]) == "p *** q"


def test_session_auth_registers_token_and_cookie():
    auth = SessionAuth()
    out = redact(f"token is {auth.token} cookie {auth.cookie_value()}")
    assert auth.token not in out and auth.cookie_value() not in out


# ----------------------------------------------------------------- logging integration
def test_filter_redacts_msg_args_and_exception(caplog):
    auth = SessionAuth()
    logger = logging.getLogger("autotune_studio.test.redaction")
    flt = RedactingFilter()
    caplog.handler.addFilter(flt)
    try:
        with caplog.at_level(logging.INFO, logger=logger.name):
            logger.info("session %s started with %s", auth.token, HF)
            logger.info(f"inline {auth.token}")
            logger.info("dict %(t)s", {"t": HF})
            try:
                raise ValueError(f"boom {HF}")
            except ValueError:
                logger.exception("failed")
    finally:
        caplog.handler.removeFilter(flt)
    assert caplog.records
    for rec in caplog.records:
        assert auth.token not in rec.getMessage()
        assert HF not in rec.getMessage()
    assert auth.token not in caplog.text and HF not in caplog.text
    assert "***" in caplog.text


def test_install_redaction_record_factory_covers_late_handlers(caplog):
    install_redaction(["late-registered-secret-abc"])
    logger = logging.getLogger("autotune_studio.test.late")
    with caplog.at_level(logging.INFO, logger=logger.name):
        logger.info("value=%s", "late-registered-secret-abc")
        logger.warning("hf %s", HF)
    assert "late-registered-secret-abc" not in caplog.text
    assert HF not in caplog.text


def test_install_redaction_attaches_to_uvicorn_and_root_handlers():
    handler = logging.StreamHandler()
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        flt = install_redaction(record_factory=False)
        assert flt in handler.filters
        for name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
            assert flt in logging.getLogger(name).filters
        assert install_redaction(record_factory=False) is flt  # idempotent
        assert handler.filters.count(flt) == 1
        redaction.uninstall_redaction()
        assert flt not in handler.filters
        assert flt not in logging.getLogger("uvicorn.access").filters
    finally:
        root.removeHandler(handler)


def test_uvicorn_access_style_args_keep_tuple_shape():
    rec = logging.LogRecord(
        "uvicorn.access",
        logging.INFO,
        __file__,
        1,
        '%s - "%s %s HTTP/%s" %d',
        ("1.2.3.4:5", "GET", f"/api/x?token={HF}", "1.1", 200),
        None,
    )
    RedactingFilter().filter(rec)
    assert isinstance(rec.args, tuple) and len(rec.args) == 5
    _client, _method, path, _ver, status = rec.args  # uvicorn AccessFormatter unpacks like this
    assert status == 200 and HF not in path
    assert HF not in rec.getMessage()


def test_factory_uninstall_restores_original():
    original = logging.getLogRecordFactory()
    install_redaction()
    assert logging.getLogRecordFactory() is not original
    redaction.uninstall_redaction()
    assert logging.getLogRecordFactory() is original


# ----------------------------------------------------------------- get_hf_token
@pytest.fixture
def no_platform_secrets(monkeypatch):
    for var in sec.ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setitem(sys.modules, "google.colab", None)  # import -> ModuleNotFoundError
    monkeypatch.setitem(sys.modules, "kaggle_secrets", None)


def test_hf_token_from_env_and_registered(monkeypatch, no_platform_secrets, caplog):
    monkeypatch.setenv("HF_TOKEN", f"  {HF}  ")
    with caplog.at_level(logging.DEBUG, logger="autotune_studio.security.secrets"):
        assert sec.get_hf_token() == HF
    assert HF not in caplog.text
    assert sec.hf_token_source() == "env:HF_TOKEN"
    register_secret("x")  # no-op, keeps registry sane
    assert redact(f"t={HF}") == "t=***"


def test_hf_token_env_precedence(monkeypatch, no_platform_secrets):
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hf_second_aaaaaaaaaaaaaaaaaaaaaa")
    assert sec.get_hf_token() == "hf_second_aaaaaaaaaaaaaaaaaaaaaa"
    monkeypatch.setenv("HF_TOKEN", "hf_first_bbbbbbbbbbbbbbbbbbbbbbbb")
    assert sec.get_hf_token() == "hf_first_bbbbbbbbbbbbbbbbbbbbbbbb"


def test_hf_token_none_when_nothing(no_platform_secrets, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "   ")
    assert sec.get_hf_token() is None
    assert sec.hf_token_source() is None


def test_hf_token_colab_and_kaggle_fakes(monkeypatch, no_platform_secrets):
    google = types.ModuleType("google")
    colab = types.ModuleType("google.colab")
    colab.userdata = types.SimpleNamespace(get=lambda name: "hf_colab_ccccccccccccccccccccccc")  # type: ignore[attr-defined]
    google.colab = colab  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    assert sec.get_hf_token() == "hf_colab_ccccccccccccccccccccccc"
    assert sec.hf_token_source() == "colab"

    def boom(name):
        raise RuntimeError("SecretNotFoundError")

    colab.userdata = types.SimpleNamespace(get=boom)  # type: ignore[attr-defined]

    class FakeClient:
        def get_secret(self, name):
            assert name == "HF_TOKEN"
            return "hf_kaggle_ddddddddddddddddddddddd"

    kag = types.ModuleType("kaggle_secrets")
    kag.UserSecretsClient = FakeClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "kaggle_secrets", kag)
    assert sec.get_hf_token() == "hf_kaggle_ddddddddddddddddddddddd"
    assert sec.hf_token_source() == "kaggle"


def test_hf_token_kaggle_error_swallowed(monkeypatch, no_platform_secrets):
    class Broken:
        def get_secret(self, name):
            raise ConnectionError("no secret attached")

    kag = types.ModuleType("kaggle_secrets")
    kag.UserSecretsClient = Broken  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "kaggle_secrets", kag)
    assert sec.get_hf_token() is None
