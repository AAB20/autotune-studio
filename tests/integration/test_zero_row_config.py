"""Zero row configuration (ADR-0003): no public API or route asks for a row count; unknown cardinality
requires ``max_steps`` with a clear 422; after a scan the exact length allows epoch-based planning."""

from __future__ import annotations

import inspect
import re

from fastapi.routing import APIRoute

from autotune_studio.api.schemas import ResumeCheckpointRequest, StartTrainingRequest
from autotune_studio.contracts import DatasetSpec, ExportRequest, ModelConfig, TrainingConfig
from autotune_studio.studio import Studio

#: Parameter / field names that would mean "the user supplies a row count / row cap".
ROW_PARAM = re.compile(
    r"^(max_?rows?|n_?rows?|num_?rows?|rows?|row_?count|row_?limit|limit|max_?samples?|num_?samples?|"
    r"n_?samples?|sample_?size|take|head|max_?examples?|num_?examples?|dataset_?size|length|size)$",
    re.IGNORECASE,
)

# Methods of the facade that load / inspect / stream / train on datasets.
DATASET_METHODS = ["load_dataset", "load_dataset_spec", "scan_dataset", "dataset_stats", "get_dataset",
                   "start_training", "validate_training", "resume_from_checkpoint", "cancel_dataset"]


def _bad(names) -> list[str]:
    return [n for n in names if ROW_PARAM.match(n)]


def test_facade_signatures_have_no_row_count_parameters() -> None:
    for meth in DATASET_METHODS:
        params = inspect.signature(getattr(Studio, meth)).parameters
        assert not _bad(params), f"Studio.{meth} takes a row-count parameter: {_bad(params)}"
    # load_dataset needs only a source + location.
    sig = inspect.signature(Studio.load_dataset)
    required = [p for p, v in sig.parameters.items() if v.default is inspect.Parameter.empty and p != "self"]
    assert required == ["source"]


def test_contract_models_have_no_row_count_fields() -> None:
    for model in (DatasetSpec, TrainingConfig, ModelConfig, ExportRequest, StartTrainingRequest,
                  ResumeCheckpointRequest):
        assert not _bad(model.model_fields), f"{model.__name__} has row-count field(s): {_bad(model.model_fields)}"
    required = [n for n, f in DatasetSpec.model_fields.items() if f.is_required()]
    assert required == ["source"]


def _api_routes() -> list[APIRoute]:
    """All APIRoutes mounted under /api (read from the routers: FastAPI >=0.16x wraps included routers)."""
    from autotune_studio.api.routes import auth, checkpoints, datasets, models, system, training

    return [r for m in (auth, system, datasets, models, training, checkpoints) for r in m.router.routes
            if isinstance(r, APIRoute)]


def test_no_route_requires_a_row_count() -> None:
    checked = 0
    for route in _api_routes():
        dep = route.dependant
        names = [p.name for p in dep.query_params + dep.path_params + dep.header_params]
        for b in dep.body_params:
            ann = getattr(b.field_info, "annotation", None)
            names.append(b.name)
            fields = getattr(ann, "model_fields", None)
            if fields:
                names.extend(fields)
        # Preview size (n) is a UI display size, not a dataset row cap; it is optional with a default.
        if route.path.endswith("/preview"):
            q = {p.name: p for p in dep.query_params}
            assert not q["n"].field_info.is_required()
            names = [n for n in names if n != "n"]
        assert not _bad(names), f"{route.methods} {route.path} asks for row counts: {_bad(names)}"
        required_q = [p.name for p in dep.query_params if p.field_info.is_required()]
        assert not required_q, f"{route.path} has required query params {required_q}"
        checked += 1
    assert checked >= 25


def test_unknown_cardinality_without_max_steps_is_422_steps_required(auth_client, make_dataset) -> None:
    c = auth_client
    src = make_dataset("jsonl", "text", n=30)
    info = c.post("/api/datasets", json={"source": "jsonl", "path": str(src)}).json()
    assert info["cardinality"]["rows"] is None
    did = info["dataset_id"]
    assert c.put("/api/model", json={"model_name": "hf-internal-testing/tiny-random-LlamaForCausalLM",
                                     "load_in_4bit": False}).status_code == 200
    cfg = {"method": "lora", "max_steps": None, "num_epochs": 1.0}
    assert c.put("/api/training/config", json=cfg).status_code == 200

    for r in (c.get("/api/training/validate", params={"dataset_id": did}),
              c.post("/api/training/start", json={"dataset_id": did})):
        assert r.status_code == 422, r.text
        body = r.json()
        assert body["code"] == "max_steps_required"
        assert "max_steps" in body["message"]
    assert c.get("/api/training/status").json()["state"] == "idle"

    # After an explicit full scan the length is exact -> epochs convert to steps, no max_steps needed.
    stats = c.app.state.studio.scan_dataset(did)
    assert stats.rows_seen == 30
    r = c.get("/api/training/validate", params={"dataset_id": did})
    assert r.status_code == 200, r.text
    assert r.json()["planned_steps"] > 0


def test_parquet_cardinality_known_from_metadata(auth_client, make_dataset) -> None:
    p = make_dataset("parquet", "instruction", n=123)
    info = auth_client.post("/api/datasets", json={"source": "parquet", "path": str(p)}).json()
    assert info["cardinality"] == {"rows": 123, "exact": True, "method": "metadata"}
