"""Shared API contracts for AutoTune Studio.

This module is the single source of truth for data exchanged between modules,
the HTTP/WebSocket API and the frontend (``frontend/src/services/types.ts`` is
generated from it by ``scripts/gen_ts_types.py``).

Owned by: Chief Architect. Other agents must not change field names or types
without an ADR in ``docs/decisions/``.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


# --------------------------------------------------------------------------- runtime
class RuntimeKind(str, Enum):
    COLAB = "colab"
    KAGGLE = "kaggle"
    JUPYTER = "jupyter"
    SCRIPT = "script"


class EnvironmentInfo(_Model):
    kind: RuntimeKind
    python_version: str
    platform: str
    is_notebook: bool
    workspace_dir: str = Field(description="Default writable directory for outputs.")
    allowed_roots: list[str] = Field(description="Directories the studio may read datasets from.")
    internet_access: bool | None = Field(None, description="None = not probed yet.")
    notes: list[str] = Field(default_factory=list)


class GPUInfo(_Model):
    index: int
    name: str
    total_vram_bytes: int
    free_vram_bytes: int | None = None
    utilization_pct: float | None = None
    compute_capability: str | None = None


class HardwareInfo(_Model):
    cpu_count: int
    ram_total_bytes: int
    ram_available_bytes: int
    disk_path: str
    disk_total_bytes: int
    disk_free_bytes: int
    cuda_available: bool
    cuda_version: str | None = None
    torch_version: str | None = None
    bf16_supported: bool = False
    gpus: list[GPUInfo] = Field(default_factory=list)
    nvml_available: bool = False


LaunchMethod = Literal["colab_proxy", "cloudflared", "local"]


class LaunchInfo(_Model):
    local_url: str
    public_url: str | None = None
    method: LaunchMethod
    publicly_verified: bool = Field(
        False, description="True only if an HTTP health probe through public_url succeeded."
    )
    open_url: str | None = Field(None, description="URL to open in a new tab, includes one-time auth fragment.")
    diagnostics: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- datasets
class DatasetSource(str, Enum):
    HUGGINGFACE = "huggingface"
    CSV = "csv"
    JSON = "json"
    JSONL = "jsonl"
    PARQUET = "parquet"


class DatasetSpec(_Model):
    source: DatasetSource
    name: str | None = Field(None, description="Hugging Face dataset id (source=huggingface).")
    path: str | None = Field(None, description="Local file or directory path (file sources).")
    split: str = "train"
    config_name: str | None = None
    revision: str | None = None
    text_column: str | None = Field(None, description="Override auto-detected text column.")

    @model_validator(mode="after")
    def _check_location(self) -> "DatasetSpec":
        if self.source == DatasetSource.HUGGINGFACE:
            if not self.name:
                raise ValueError("source='huggingface' requires 'name'")
        elif not self.path:
            raise ValueError(f"source='{self.source.value}' requires 'path'")
        return self


class ColumnInfo(_Model):
    name: str
    dtype: str
    nullable: bool = True
    sampled_null_count: int = 0
    examples: list[str] = Field(default_factory=list, description="Up to 3 truncated example values.")


class Cardinality(_Model):
    rows: int | None = Field(None, description="None means unknown cardinality.")
    exact: bool = False
    method: Literal["metadata", "scan", "unknown"] = "unknown"


DatasetFormat = Literal["chat", "instruction", "text", "unknown"]


class DatasetInfo(_Model):
    dataset_id: str
    spec: DatasetSpec
    columns: list[ColumnInfo]
    cardinality: Cardinality
    detected_format: DatasetFormat = "unknown"
    text_columns: list[str] = Field(default_factory=list, description="Columns used to build training text.")
    sampled_rows: int = 0
    sampled_bytes: int = 0
    warnings: list[str] = Field(default_factory=list)


StreamState = Literal["idle", "streaming", "exhausted", "failed", "cancelled"]


class StreamStats(_Model):
    state: StreamState = "idle"
    rows_seen: int = 0
    bytes_seen: int = 0
    batches_seen: int = 0
    malformed_rows: int = 0
    malformed_examples: list[str] = Field(default_factory=list)
    checksum: str = Field("", description="Rolling blake2b digest over canonical row bytes.")
    progress: float | None = Field(None, ge=0.0, le=1.0, description="None when total size unknown.")
    error: str | None = None


class DatasetPreview(_Model):
    columns: list[str]
    rows: list[dict[str, Any]]
    truncated_values: bool = False


class DatasetInspection(_Model):
    """Combined view returned by ``Studio.inspect_dataset``."""

    info: DatasetInfo
    preview: DatasetPreview
    stats: StreamStats


# --------------------------------------------------------------------------- model / training
class ModelConfig(_Model):
    model_name: str = Field(..., min_length=1)
    revision: str | None = None
    max_seq_length: int = Field(2048, ge=16, le=131072)
    load_in_4bit: bool = True
    dtype: Literal["auto", "float16", "bfloat16", "float32"] = "auto"
    trust_remote_code: Literal[False] = Field(
        False, description="Remote code execution is never allowed (security policy)."
    )

    @field_validator("model_name")
    @classmethod
    def _model_name(cls, v: str) -> str:
        v = v.strip()
        if ".." in v or v.startswith(("http://", "https://")):
            raise ValueError("model_name must be a Hugging Face repo id or local path, not a URL")
        return v


TrainingMethod = Literal["lora", "qlora"]
Optimizer = Literal["adamw_torch", "adamw_8bit", "paged_adamw_8bit", "sgd", "adafactor"]
Scheduler = Literal["linear", "cosine", "constant", "constant_with_warmup"]


class TrainingConfig(_Model):
    method: TrainingMethod = "qlora"
    learning_rate: float = Field(2e-4, gt=0, le=1.0)
    num_epochs: float | None = Field(1.0, gt=0, description="Used when dataset length is known.")
    max_steps: int | None = Field(None, gt=0, description="Required when dataset length is unknown.")
    per_device_batch_size: int = Field(2, ge=1, le=1024)
    gradient_accumulation_steps: int = Field(4, ge=1, le=4096)
    lora_r: int = Field(16, ge=1, le=1024)
    lora_alpha: int = Field(16, ge=1, le=4096)
    lora_dropout: float = Field(0.0, ge=0.0, lt=1.0)
    target_modules: list[str] | None = None
    optimizer: Optimizer = "adamw_torch"
    lr_scheduler: Scheduler = "linear"
    warmup_steps: int = Field(5, ge=0)
    weight_decay: float = Field(0.0, ge=0.0)
    logging_steps: int = Field(1, ge=1)
    save_steps: int = Field(50, ge=1)
    seed: int = 3407
    val_fraction: float = Field(0.0, ge=0.0, le=0.5, description="Deterministic hash-based split.")
    output_dir: str | None = None
    adaptive_batch_size: bool = Field(True, description="On CUDA OOM halve micro-batch, keep effective batch.")


class TrainingState(str, Enum):
    IDLE = "idle"
    PREPARING = "preparing"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    COMPLETED = "completed"
    STOPPED = "stopped"
    FAILED = "failed"


TrainingBackend = Literal["unsloth", "hf"]


class TrainingStatus(_Model):
    state: TrainingState = TrainingState.IDLE
    job_id: str | None = None
    backend: TrainingBackend | None = None
    step: int = 0
    max_steps: int | None = None
    epoch: float | None = None
    loss: float | None = None
    learning_rate: float | None = None
    started_at: float | None = None
    eta_seconds: float | None = Field(None, description="None when not calculable.")
    supports_pause: bool = True
    message: str = ""
    error: str | None = None
    effective_batch_size: int | None = None
    micro_batch_size: int | None = None


class MetricPoint(_Model):
    step: int
    loss: float | None = None
    learning_rate: float | None = None
    grad_norm: float | None = None
    epoch: float | None = None
    eval_loss: float | None = None
    timestamp: float = Field(default_factory=time.time)


class ResourceSnapshot(_Model):
    timestamp: float = Field(default_factory=time.time)
    cpu_pct: float
    ram_used_bytes: int
    ram_total_bytes: int
    vram_used_bytes: int | None = None
    vram_reserved_bytes: int | None = None
    vram_total_bytes: int | None = None
    gpu_util_pct: float | None = None


class CheckpointInfo(_Model):
    checkpoint_id: str
    path: str
    step: int
    created_at: float
    size_bytes: int
    job_id: str | None = None
    is_final: bool = False
    data_cursor: dict[str, Any] | None = None


ExportFormat = Literal["lora_adapter", "merged_16bit"]


class ExportRequest(_Model):
    checkpoint_id: str | None = Field(None, description="None = latest checkpoint.")
    format: ExportFormat = "lora_adapter"
    output_dir: str | None = None


class ExportResult(_Model):
    path: str
    format: ExportFormat
    files: list[str]
    size_bytes: int


class SessionSummary(_Model):
    job_id: str
    model_name: str
    dataset_id: str | None
    state: TrainingState
    started_at: float | None
    finished_at: float | None = None
    final_step: int = 0
    final_loss: float | None = None


# --------------------------------------------------------------------------- events / errors
EventType = Literal[
    "hello", "status", "metric", "resource", "log", "dataset", "checkpoint", "export", "error"
]


class Event(_Model):
    seq: int
    type: EventType
    ts: float = Field(default_factory=time.time)
    data: dict[str, Any] = Field(default_factory=dict)


class LogEventData(_Model):
    """Payload convention for ``type="log"`` events (extra keys allowed for context)."""

    model_config = ConfigDict(extra="allow")
    level: Literal["debug", "info", "warning", "error"] = "info"
    source: str = "studio"
    message: str


class ApiError(_Model):
    code: str
    message: str
    detail: dict[str, Any] | None = None


class ErrorCode:
    """Stable machine-readable error codes."""

    VALIDATION = "validation_error"
    NOT_FOUND = "not_found"
    CONFLICT = "conflict"
    FORBIDDEN_PATH = "forbidden_path"
    UNAUTHORIZED = "unauthorized"
    DEPENDENCY_MISSING = "dependency_missing"
    DATASET_EMPTY = "dataset_empty"
    DATASET_MALFORMED = "dataset_malformed"
    STEPS_REQUIRED = "max_steps_required"
    OOM = "out_of_memory"
    INTERNAL = "internal_error"
