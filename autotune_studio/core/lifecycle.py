"""Single-job training lifecycle manager. Owned by: Chief Architect (reassigned from backend, see AGENTS.md).

Runs a :class:`~autotune_studio.interfaces.TrainingEngineProtocol` on a worker thread,
tracks :class:`~autotune_studio.contracts.TrainingStatus` from the engine's events,
feeds the metric history / ETA estimator and persists a session history.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..contracts import (
    CheckpointInfo,
    MetricPoint,
    ModelConfig,
    SessionSummary,
    TrainingConfig,
    TrainingState,
    TrainingStatus,
)
from ..errors import Conflict, StudioError
from ..interfaces import (
    EventSink,
    StreamingDatasetProtocol,
    TrainingControl,
    TrainingEngineProtocol,
    TrainingResult,
)
from ..monitoring.metrics import EtaEstimator, MetricsStore

log = logging.getLogger(__name__)

EngineFactory = Callable[[ModelConfig, TrainingConfig], TrainingEngineProtocol]

_ACTIVE = {TrainingState.PREPARING, TrainingState.RUNNING, TrainingState.PAUSED, TrainingState.STOPPING}
_STATUS_FIELDS = set(TrainingStatus.model_fields)


class _InterceptingSink:
    """Forwards engine events to the bus while updating lifecycle state."""

    def __init__(self, lc: TrainingLifecycle, job_id: str) -> None:
        self._lc = lc
        self._job_id = job_id

    def publish(self, type: str, data: dict[str, Any]) -> None:
        lc = self._lc
        if type == "status":
            lc._apply_status(data, publish=False)
            data = lc.status().model_dump(mode="json")
        elif type == "metric":
            try:
                point = MetricPoint.model_validate(data)
                lc.metrics.add(point)
                lc._eta.update(point.step, point.timestamp)
                lc._apply_status(
                    {
                        "step": point.step,
                        "loss": point.loss if point.loss is not None else lc._status.loss,
                        "learning_rate": point.learning_rate,
                        "epoch": point.epoch,
                    },
                    publish=False,
                )
            except Exception:  # noqa: BLE001 - malformed metric must not kill training
                log.exception("bad metric event")
        elif type == "checkpoint":
            try:
                lc._checkpoints.append(CheckpointInfo.model_validate(data))
            except Exception:  # noqa: BLE001
                log.exception("bad checkpoint event")
            lc._checkpoint_saved.set()
        lc.events.publish(type, data)


class TrainingLifecycle:
    """Owns at most one active training job."""

    def __init__(
        self,
        events: EventSink,
        engine_factory: EngineFactory,
        metrics: MetricsStore | None = None,
        history_path: str | Path | None = None,
    ) -> None:
        self.events = events
        self.engine_factory = engine_factory
        self.metrics = metrics or MetricsStore()
        self.history_path = Path(history_path) if history_path else None
        self._lock = threading.RLock()
        self._status = TrainingStatus()
        self._thread: threading.Thread | None = None
        self._control = TrainingControl()
        self._eta = EtaEstimator()
        self._checkpoints: list[CheckpointInfo] = []
        self._checkpoint_saved = threading.Event()
        self._result: TrainingResult | None = None
        self._current: SessionSummary | None = None
        self._sessions: list[SessionSummary] = self._load_history()

    # ------------------------------------------------------------------ queries
    def status(self) -> TrainingStatus:
        with self._lock:
            st = self._status.model_copy()
        if st.state in _ACTIVE:
            st.eta_seconds = self._eta.eta_seconds()
        return st

    @property
    def is_active(self) -> bool:
        with self._lock:
            return self._status.state in _ACTIVE

    @property
    def last_result(self) -> TrainingResult | None:
        return self._result

    def sessions(self) -> list[SessionSummary]:
        with self._lock:
            out = list(self._sessions)
            if self._current is not None:
                cur = self._current.model_copy(
                    update={"state": self._status.state, "final_step": self._status.step,
                            "final_loss": self._status.loss}
                )
                out.append(cur)
        return sorted(out, key=lambda s: s.started_at or 0, reverse=True)

    # ------------------------------------------------------------------ control
    def start(
        self,
        model_cfg: ModelConfig,
        train_cfg: TrainingConfig,
        dataset: StreamingDatasetProtocol,
        *,
        max_steps: int | None = None,
        resume_from: CheckpointInfo | None = None,
        job_id: str | None = None,
    ) -> TrainingStatus:
        with self._lock:
            if self._status.state in _ACTIVE:
                raise Conflict("a training job is already active", {"job_id": self._status.job_id})
            job_id = job_id or f"job-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
            self._control = TrainingControl()
            self._eta = EtaEstimator(max_steps=max_steps)
            self._checkpoints = []
            self._checkpoint_saved.clear()
            self._result = None
            self.metrics.clear()
            self._status = TrainingStatus(
                state=TrainingState.PREPARING,
                job_id=job_id,
                max_steps=max_steps,
                started_at=time.time(),
                message="preparing (loading model and data)",
            )
            self._current = SessionSummary(
                job_id=job_id,
                model_name=model_cfg.model_name,
                dataset_id=dataset.info.dataset_id,
                state=TrainingState.PREPARING,
                started_at=self._status.started_at,
            )
            self._thread = threading.Thread(
                target=self._run,
                args=(job_id, model_cfg, train_cfg, dataset, resume_from),
                name=f"autotune-train-{job_id}",
                daemon=True,
            )
        self._publish_status()
        self._thread.start()
        return self.status()

    def stop(self, wait: bool = False, timeout: float | None = 60.0) -> TrainingStatus:
        with self._lock:
            if self._status.state not in _ACTIVE:
                return self.status()
            self._control.stop.set()
            self._control.pause.clear()
            self._status.state = TrainingState.STOPPING
            self._status.message = "stopping after the current step (a checkpoint will be saved)"
        self._publish_status()
        if wait:
            return self.wait(timeout)
        return self.status()

    def pause(self) -> TrainingStatus:
        with self._lock:
            if self._status.state != TrainingState.RUNNING:
                raise Conflict(f"cannot pause in state '{self._status.state.value}'")
            if not self._status.supports_pause:
                raise Conflict("this backend does not support pausing")
            self._control.pause.set()
            self._status.state = TrainingState.PAUSED
            self._status.message = "paused"
        self._publish_status()
        return self.status()

    def resume(self) -> TrainingStatus:
        with self._lock:
            if self._status.state != TrainingState.PAUSED:
                raise Conflict(f"cannot resume in state '{self._status.state.value}'")
            self._control.pause.clear()
            self._status.state = TrainingState.RUNNING
            self._status.message = "running"
        self._publish_status()
        return self.status()

    def request_checkpoint(self, timeout: float = 300.0) -> CheckpointInfo:
        """Ask the engine to save at the next step boundary and wait for it."""
        with self._lock:
            if self._status.state not in {TrainingState.RUNNING, TrainingState.PAUSED}:
                raise Conflict("no running training job; checkpoints are saved automatically during training")
            before = len(self._checkpoints)
            self._checkpoint_saved.clear()
            self._control.save_now.set()
            paused = self._status.state == TrainingState.PAUSED
        if paused:
            # A paused engine still honours save_now between steps.
            pass
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self._checkpoint_saved.wait(0.25)
            with self._lock:
                if len(self._checkpoints) > before:
                    return self._checkpoints[-1]
                if self._status.state not in _ACTIVE:
                    break
            self._checkpoint_saved.clear()
        raise Conflict("checkpoint was not saved in time (training may be between long steps)")

    def wait(self, timeout: float | None = None) -> TrainingStatus:
        t = self._thread
        if t is not None:
            t.join(timeout)
        return self.status()

    # ------------------------------------------------------------------ internals
    def _apply_status(self, data: dict[str, Any], publish: bool = True) -> None:
        upd = {k: v for k, v in data.items() if k in _STATUS_FIELDS and k not in {"job_id", "started_at"}}
        with self._lock:
            # Control-driven states win over engine-reported "running".
            if upd.get("state") in ("running", TrainingState.RUNNING) and self._status.state in {
                TrainingState.PAUSED,
                TrainingState.STOPPING,
            }:
                upd.pop("state")
            if "state" in upd:
                upd["state"] = TrainingState(upd["state"])
            merged = self._status.model_dump()
            merged.update(upd)
            self._status = TrainingStatus.model_validate(merged)
            if self._status.max_steps:
                self._eta.set_max_steps(self._status.max_steps)
        if publish:
            self._publish_status()

    def _publish_status(self) -> None:
        self.events.publish("status", self.status().model_dump(mode="json"))

    def _run(
        self,
        job_id: str,
        model_cfg: ModelConfig,
        train_cfg: TrainingConfig,
        dataset: StreamingDatasetProtocol,
        resume_from: CheckpointInfo | None,
    ) -> None:
        sink = _InterceptingSink(self, job_id)
        final_state = TrainingState.FAILED
        error: str | None = None
        try:
            engine = self.engine_factory(model_cfg, train_cfg)
            self._apply_status({"backend": getattr(engine, "backend", None)})
            result = engine.run(job_id, model_cfg, train_cfg, dataset, sink, self._control, resume_from)
            self._result = result
            final_state = TrainingState.STOPPED if result.stopped_early else TrainingState.COMPLETED
            self._apply_status(
                {
                    "state": final_state,
                    "step": result.final_step,
                    "loss": result.final_loss if result.final_loss is not None else self._status.loss,
                    "message": "stopped by user" if result.stopped_early else "training completed",
                    "eta_seconds": None,
                },
                publish=False,
            )
        except StudioError as exc:
            error = f"{exc.code}: {exc.message}"
        except Exception as exc:  # noqa: BLE001 - report every failure to the UI
            log.exception("training job %s failed", job_id)
            error = f"{type(exc).__name__}: {exc}"
        if error is not None:
            final_state = TrainingState.FAILED
            self._apply_status({"state": final_state, "error": error, "message": "training failed"}, publish=False)
            self.events.publish("error", {"source": "training", "job_id": job_id, "message": error})
        self._publish_status()
        with self._lock:
            if self._current is not None:
                done = self._current.model_copy(
                    update={
                        "state": final_state,
                        "finished_at": time.time(),
                        "final_step": self._status.step,
                        "final_loss": self._status.loss,
                    }
                )
                self._sessions.append(done)
                self._current = None
            self._save_history()

    def _load_history(self) -> list[SessionSummary]:
        if not self.history_path or not self.history_path.is_file():
            return []
        try:
            raw = json.loads(self.history_path.read_text(encoding="utf-8"))
            return [SessionSummary.model_validate(r) for r in raw]
        except Exception:  # noqa: BLE001 - corrupt history must not block startup
            log.warning("ignoring unreadable session history %s", self.history_path)
            return []

    def _save_history(self) -> None:
        if not self.history_path:
            return
        try:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.history_path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps([s.model_dump(mode="json") for s in self._sessions[-200:]], indent=1),
                encoding="utf-8",
            )
            tmp.replace(self.history_path)
        except OSError:
            log.warning("could not write session history %s", self.history_path)
