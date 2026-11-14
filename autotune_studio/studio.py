"""Public Python API: the :class:`Studio` facade. Owned by: Chief Architect.

Typical notebook use::

    from autotune_studio import Studio

    studio = Studio()
    studio.launch()                     # opens the UI in a new browser tab

or fully from Python::

    studio.load_dataset(source="csv", path="/content/data.csv")   # no row count needed
    studio.configure_model(model_name="unsloth/Llama-3.2-1B-Instruct")
    studio.configure_training(method="qlora", max_steps=200)
    studio.start_training()
    studio.wait()
    studio.export_model()
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from .contracts import (
    CheckpointInfo,
    DatasetInfo,
    DatasetInspection,
    DatasetSource,
    DatasetSpec,
    EnvironmentInfo,
    ExportFormat,
    ExportRequest,
    ExportResult,
    HardwareInfo,
    LaunchInfo,
    MetricPoint,
    ModelConfig,
    ResourceSnapshot,
    SessionSummary,
    StreamStats,
    TrainingConfig,
    TrainingStatus,
)
from .core.events import EventBus
from .errors import Conflict, NotFound, ValidationFailed

log = logging.getLogger(__name__)

SUGGESTED_MODELS: list[str] = [
    "unsloth/Llama-3.2-1B-Instruct",
    "unsloth/Llama-3.2-3B-Instruct",
    "unsloth/Qwen2.5-0.5B-Instruct",
    "unsloth/Qwen2.5-1.5B-Instruct",
    "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    "unsloth/gemma-2-2b-it",
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/Phi-3.5-mini-instruct",
]

_FILE_KINDS = {
    DatasetSource.CSV: "csv",
    DatasetSource.JSON: "json",
    DatasetSource.JSONL: "jsonl",
    DatasetSource.PARQUET: "parquet",
}


class Studio:
    """Facade over runtime detection, the zero-row data engine, training, monitoring and the web UI.

    All methods are thread-safe enough to be called from a notebook while the web UI is in use.
    """

    def __init__(
        self,
        workspace_dir: str | Path | None = None,
        allowed_roots: list[str] | None = None,
        *,
        resource_interval: float = 2.0,
        replay_size: int | None = None,
    ) -> None:
        from .core.environment import detect_environment, ensure_workspace
        from .core.lifecycle import TrainingLifecycle
        from .monitoring import ResourceMonitor
        from .monitoring.metrics import MetricsStore, ResourceSampler
        from .security import PathPolicy, SessionAuth

        if replay_size is None:
            from .core.config import load_settings

            replay_size = load_settings().replay_size
        self.events = EventBus(replay_size=replay_size)
        env = detect_environment()
        updates: dict[str, Any] = {}
        if workspace_dir is not None:
            updates["workspace_dir"] = str(Path(workspace_dir).expanduser().resolve())
        roots = list(env.allowed_roots) + list(allowed_roots or [])
        ws = updates.get("workspace_dir", env.workspace_dir)
        if ws not in roots:
            roots.append(ws)
        updates["allowed_roots"] = roots
        self._env: EnvironmentInfo = env.model_copy(update=updates)
        self.workspace = ensure_workspace(self._env)
        self.path_policy = PathPolicy(self._env.allowed_roots)
        self.auth = SessionAuth()

        self._lock = threading.RLock()
        self._hardware: HardwareInfo | None = None
        self._datasets: dict[str, Any] = {}  # dataset_id -> StreamingDataset
        self._active_dataset_id: str | None = None
        self._scan_threads: dict[str, threading.Thread] = {}
        self.model_config: ModelConfig | None = None
        from .monitoring.gpu import cuda_available

        # QLoRA needs CUDA; default to plain LoRA elsewhere so the defaults are always startable.
        self.training_config: TrainingConfig = TrainingConfig(method="qlora" if cuda_available() else "lora")

        self.monitor = ResourceMonitor()
        self.sampler = ResourceSampler(self.monitor, self.events, interval=resource_interval)
        self.metrics = MetricsStore()
        self.lifecycle = TrainingLifecycle(
            self.events,
            engine_factory=self._create_engine,
            metrics=self.metrics,
            history_path=self.workspace / "sessions.json",
        )
        self._server = None
        self._exposure = None
        self._launch_info: LaunchInfo | None = None

    # ------------------------------------------------------------------ runtime
    @property
    def runs_dir(self) -> Path:
        return self.workspace / "runs"

    def detect_environment(self, refresh: bool = False) -> EnvironmentInfo:
        """Return the detected runtime (Colab / Kaggle / Jupyter / script), workspace and allowed roots."""
        if refresh:
            from .core.environment import detect_environment

            fresh = detect_environment(probe_network=True)
            self._env = fresh.model_copy(
                update={"workspace_dir": self._env.workspace_dir, "allowed_roots": self._env.allowed_roots}
            )
        return self._env

    def detect_hardware(self, refresh: bool = False) -> HardwareInfo:
        """Return CPU/RAM/disk/GPU/CUDA information. Never raises; problems are listed in ``notes``."""
        with self._lock:
            if self._hardware is None or refresh:
                from .core.hardware import detect_hardware

                self._hardware = detect_hardware(str(self.workspace))
            return self._hardware

    def resources(self) -> ResourceSnapshot:
        """Current CPU/RAM/VRAM usage snapshot."""
        return self.monitor.snapshot()

    # ------------------------------------------------------------------ datasets
    def load_dataset(
        self,
        source: str | DatasetSource,
        name: str | None = None,
        path: str | None = None,
        *,
        split: str = "train",
        config_name: str | None = None,
        revision: str | None = None,
        text_column: str | None = None,
    ) -> DatasetInfo:
        """Open a dataset for streaming. No row count is ever required.

        :param source: ``huggingface``, ``csv``, ``json``, ``jsonl`` or ``parquet``.
        :param name: Hugging Face dataset id (``source="huggingface"``).
        :param path: local file/directory (file sources); must be inside an allowed root.
        :returns: :class:`DatasetInfo` with schema, detected format and cardinality
            (``cardinality.rows is None`` means the size is unknown and will be discovered while streaming).
        :raises ForbiddenPath, NotFound, DatasetEmpty, DatasetMalformed, ValidationFailed:
        """
        spec = DatasetSpec(
            source=DatasetSource(source),
            name=name,
            path=path,
            split=split,
            config_name=config_name,
            revision=revision,
            text_column=text_column,
        )
        return self.load_dataset_spec(spec)

    def load_dataset_spec(self, spec: DatasetSpec) -> DatasetInfo:
        """Like :meth:`load_dataset` but takes a :class:`DatasetSpec`."""
        from .data_engine import open_dataset

        if spec.source == DatasetSource.HUGGINGFACE:
            # SEC-01: a "name" that is a local path or a builder name (csv/json/text/...) would make
            # datasets.load_dataset read local files outside the allowed roots.
            from .security.validation import validate_hf_dataset_name

            spec = spec.model_copy(update={"name": validate_hf_dataset_name(spec.name or "")})
        else:
            assert spec.path is not None
            resolved = self.path_policy.resolve(spec.path, must_exist=True, kinds={_FILE_KINDS[spec.source]})
            spec = spec.model_copy(update={"path": str(resolved)})
        ds = open_dataset(spec, events=self.events)
        with self._lock:
            self._datasets[ds.info.dataset_id] = ds
            self._active_dataset_id = ds.info.dataset_id
        self.events.publish("dataset", {"dataset_id": ds.info.dataset_id, **ds.stats.model_dump()})
        return ds.info

    def list_datasets(self) -> list[DatasetInfo]:
        with self._lock:
            return [d.info for d in self._datasets.values()]

    def get_dataset(self, dataset_id: str | None = None):
        """Return the StreamingDataset handle (active dataset when ``dataset_id`` is None)."""
        with self._lock:
            did = dataset_id or self._active_dataset_id
            if did is None:
                raise NotFound("no dataset loaded; call load_dataset() first")
            ds = self._datasets.get(did)
        if ds is None:
            raise NotFound(f"dataset '{did}' not found")
        return ds

    def inspect_dataset(self, dataset_id: str | None = None, n: int = 20) -> DatasetInspection:
        """Schema, a small sample preview (``n`` rows) and current streaming statistics."""
        ds = self.get_dataset(dataset_id)
        return DatasetInspection(info=ds.info, preview=ds.preview(n), stats=ds.stats)

    def dataset_stats(self, dataset_id: str | None = None) -> StreamStats:
        return self.get_dataset(dataset_id).stats

    def scan_dataset(self, dataset_id: str | None = None, background: bool = False) -> StreamStats:
        """Stream the whole dataset once to count and validate rows (rows are not kept in memory).

        With ``background=True`` the scan runs in a thread and current stats are returned immediately.
        """
        ds = self.get_dataset(dataset_id)
        did = ds.info.dataset_id
        if not background:
            return ds.scan()
        with self._lock:
            t = self._scan_threads.get(did)
            if t is not None and t.is_alive():
                raise Conflict("a scan of this dataset is already running")

            def _scan() -> None:
                try:
                    ds.scan()
                except Exception as exc:  # noqa: BLE001 - surfaced via stats + error event
                    self.events.publish("error", {"source": "dataset", "dataset_id": did, "message": str(exc)})

            t = threading.Thread(target=_scan, name=f"autotune-scan-{did}", daemon=True)
            self._scan_threads[did] = t
            t.start()
        return ds.stats

    def cancel_dataset(self, dataset_id: str | None = None) -> StreamStats:
        ds = self.get_dataset(dataset_id)
        ds.cancel()
        return ds.stats

    # ------------------------------------------------------------------ model / training config
    def configure_model(self, model_name: str, **kwargs: Any) -> ModelConfig:
        """Set the base model (Hugging Face repo id or local directory under an allowed root).

        Extra keyword arguments are :class:`ModelConfig` fields (``max_seq_length``, ``load_in_4bit``, ``dtype``,
        ``revision``). ``trust_remote_code`` is always ``False``.
        """
        from .security import validate_model_name

        name = validate_model_name(model_name, self.path_policy)
        cfg = ModelConfig(model_name=name, **kwargs)
        with self._lock:
            self.model_config = cfg
        return cfg

    def suggested_models(self) -> list[str]:
        return list(SUGGESTED_MODELS)

    def configure_training(self, method: str = "qlora", **kwargs: Any) -> TrainingConfig:
        """Set hyper-parameters (any :class:`TrainingConfig` field). Unset fields keep their current values."""
        with self._lock:
            data = self.training_config.model_dump()
            data.update(kwargs)
            data["method"] = method
            cfg = TrainingConfig.model_validate(data)
            self.training_config = cfg
        return cfg

    def set_training_config(self, cfg: TrainingConfig) -> TrainingConfig:
        with self._lock:
            self.training_config = cfg
        return cfg

    def validate_training(self, dataset_id: str | None = None) -> tuple[int, list[str]]:
        """Check that training can start; returns ``(planned_steps, warnings)``."""
        from .training import plan_steps, validate_training_request

        if self.model_config is None:
            raise ValidationFailed("no model configured; call configure_model() first")
        ds = self.get_dataset(dataset_id)
        warnings = validate_training_request(self.model_config, self.training_config, ds.info, self.detect_hardware())
        steps = plan_steps(self.training_config, ds.info)
        return steps, warnings

    # ------------------------------------------------------------------ training control
    def _create_engine(self, model_cfg: ModelConfig, train_cfg: TrainingConfig):
        from .training import create_engine

        return create_engine(model_cfg, train_cfg)

    def start_training(
        self,
        dataset_id: str | None = None,
        resume_from: str | CheckpointInfo | None = None,
    ) -> TrainingStatus:
        """Start training in the background on the active (or given) dataset.

        :param resume_from: checkpoint id or :class:`CheckpointInfo` to continue from.
        :raises ValidationFailed, StepsRequired, Conflict:
        """
        steps, warnings = self.validate_training(dataset_id)
        ds = self.get_dataset(dataset_id)
        ckpt = self._resolve_checkpoint(resume_from) if resume_from is not None else None
        import time as _t
        import uuid as _uuid

        job_id = f"job-{_t.strftime('%Y%m%d-%H%M%S')}-{_uuid.uuid4().hex[:6]}"
        cfg = self.training_config
        if not cfg.output_dir:
            cfg = cfg.model_copy(update={"output_dir": str(self.runs_dir / job_id)})
        else:
            cfg = cfg.model_copy(
                update={"output_dir": str(self.path_policy.resolve(cfg.output_dir, must_exist=False))}
            )
        assert self.model_config is not None
        self.sampler.start()
        status = self.lifecycle.start(self.model_config, cfg, ds, max_steps=steps, resume_from=ckpt, job_id=job_id)
        for w in warnings:
            self.events.publish("log", {"level": "warning", "source": "studio", "message": w})
        return status

    def training_status(self) -> TrainingStatus:
        return self.lifecycle.status()

    def training_metrics(self, since_step: int | None = None) -> list[MetricPoint]:
        return self.metrics.list(since_step)

    def stop_training(self, wait: bool = True, timeout: float = 120.0) -> TrainingStatus:
        """Stop after the current step; a checkpoint is saved before the job ends."""
        return self.lifecycle.stop(wait=wait, timeout=timeout)

    def pause_training(self) -> TrainingStatus:
        return self.lifecycle.pause()

    def resume_training(self) -> TrainingStatus:
        return self.lifecycle.resume()

    def wait(self, timeout: float | None = None) -> TrainingStatus:
        """Block until the current training job finishes (or ``timeout`` seconds pass)."""
        return self.lifecycle.wait(timeout)

    def sessions(self) -> list[SessionSummary]:
        return self.lifecycle.sessions()

    # ------------------------------------------------------------------ checkpoints / export
    def save_checkpoint(self, timeout: float = 300.0) -> CheckpointInfo:
        """Save a checkpoint of the running job now (at the next step boundary)."""
        return self.lifecycle.request_checkpoint(timeout)

    def list_checkpoints(self) -> list[CheckpointInfo]:
        from .training.checkpoints import list_checkpoints

        if not self.runs_dir.exists():
            return []
        return list_checkpoints(self.runs_dir)

    def _resolve_checkpoint(self, ref: str | CheckpointInfo | None) -> CheckpointInfo:
        if isinstance(ref, CheckpointInfo):
            return ref
        ckpts = self.list_checkpoints()
        if not ckpts:
            raise NotFound("no checkpoints found")
        if ref is None:
            return ckpts[-1]
        for c in ckpts:
            if c.checkpoint_id == ref:
                return c
        raise NotFound(f"checkpoint '{ref}' not found")

    def resume_from_checkpoint(self, checkpoint_id: str, dataset_id: str | None = None) -> TrainingStatus:
        """Start a new job continuing from a checkpoint (model/training config taken from the checkpoint state)."""
        ckpt = self._resolve_checkpoint(checkpoint_id)
        from .training.checkpoints import load_checkpoint_state

        state = load_checkpoint_state(ckpt.path)
        if state.get("model_config"):
            self.model_config = ModelConfig.model_validate(state["model_config"])
        if state.get("training_config"):
            tc = dict(state["training_config"])
            tc["output_dir"] = None
            self.training_config = TrainingConfig.model_validate(tc)
        return self.start_training(dataset_id, resume_from=ckpt)

    def export_model(
        self,
        checkpoint_id: str | None = None,
        format: ExportFormat = "lora_adapter",
        output_dir: str | None = None,
    ) -> ExportResult:
        """Export the LoRA adapter (default) or a merged 16-bit model from a checkpoint (latest by default)."""
        from .export.model_export import export_model

        if self.lifecycle.is_active and checkpoint_id is None:
            raise Conflict("training is running; stop it or pass an explicit checkpoint_id")
        ckpt = self._resolve_checkpoint(checkpoint_id)
        req = ExportRequest(checkpoint_id=ckpt.checkpoint_id, format=format, output_dir=output_dir)
        if output_dir:
            req.output_dir = str(self.path_policy.resolve(output_dir, must_exist=False))
        model_cfg = self.model_config
        if model_cfg is None:
            from .training.checkpoints import load_checkpoint_state

            mc = load_checkpoint_state(ckpt.path).get("model_config")
            if not mc:
                raise ValidationFailed("model configuration unknown for this checkpoint")
            model_cfg = ModelConfig.model_validate(mc)
        result = export_model(req, ckpt, model_cfg, self.workspace)
        self.events.publish("export", result.model_dump())
        return result

    # ------------------------------------------------------------------ web UI
    def create_app(self):
        """Build the FastAPI application bound to this studio (used by :meth:`launch` and tests)."""
        from .api.app import create_app

        return create_app(self)

    def launch(
        self,
        *,
        port: int | None = None,
        host: str = "127.0.0.1",
        open_browser: bool = True,
        prefer: str | None = None,
        tunnel_timeout: float = 30.0,
    ) -> LaunchInfo:
        """Start the backend in a background thread and expose the UI in a separate browser tab.

        :param prefer: force an adapter: ``"colab_proxy"``, ``"cloudflared"`` or ``"local"``.
        :returns: :class:`LaunchInfo`. ``publicly_verified`` is True only after a successful health probe
            through the public URL. The session token is only contained in ``open_url``'s fragment.
        """
        from .launch import ServerThread, expose, open_in_new_tab
        from .security import install_redaction

        if self._server is not None and self._server.is_running and self._launch_info is not None:
            if open_browser and self._launch_info.open_url:
                open_in_new_tab(self._launch_info.open_url, self._env)
            return self._launch_info
        install_redaction([self.auth.token])
        server = ServerThread(self.create_app(), host=host, port=port)
        server.start()
        if not server.wait_ready(30.0):
            err = server.error
            server.stop()
            raise RuntimeError(f"backend failed to start: {err}")
        self._server = server
        self.sampler.start()
        info, handle = expose(server.port, self._env, self.auth.token, prefer=prefer, host=host,
                              tunnel_timeout=tunnel_timeout)
        self._exposure = handle
        self._launch_info = info
        if info.public_url:
            self._extra_origins.add(info.public_url)
        for d in info.diagnostics:
            log.info("launch: %s", d)
        if open_browser and info.open_url:
            open_in_new_tab(info.open_url, self._env)
        return info

    @property
    def _extra_origins(self) -> set[str]:
        if not hasattr(self, "_origins"):
            self._origins: set[str] = set()
        return self._origins

    @property
    def extra_origins(self) -> list[str]:
        """Public origins (tunnel / proxy) accepted for WebSocket Origin checks."""
        return sorted(self._extra_origins)

    @property
    def launch_info(self) -> LaunchInfo | None:
        return self._launch_info

    def shutdown(self, stop_training: bool = True) -> None:
        """Stop training (saving a checkpoint), the tunnel, the web server and the resource sampler."""
        if stop_training and self.lifecycle.is_active:
            self.lifecycle.stop(wait=True, timeout=120)
        if self._exposure is not None:
            self._exposure.stop()
            self._exposure = None
        if self._server is not None:
            self._server.stop()
            self._server = None
        self._launch_info = None
        self.sampler.stop()

    def __repr__(self) -> str:
        return f"Studio(runtime={self._env.kind.value}, workspace={str(self.workspace)!r})"
