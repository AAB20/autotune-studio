"""HF (Transformers + PEFT) training engine. Heavy module: imported lazily by :func:`create_engine`.

The engine runs synchronously on the caller's thread. It drives a ``transformers.Trainer``
(``trl.SFTTrainer`` for the Unsloth backend) over a ``datasets.IterableDataset`` built from
``StreamingDataset.iter_rows``; callbacks translate Trainer events into studio events and
honour :class:`~autotune_studio.interfaces.TrainingControl`.

Owned by: llm-training.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from transformers import TrainerCallback, TrainingArguments

from ..contracts import CheckpointInfo, MetricPoint, ModelConfig, TrainingConfig, TrainingState
from ..errors import OutOfMemory, StudioError, ValidationFailed
from ..interfaces import EventSink, StreamingDatasetProtocol, TrainingControl, TrainingResult
from ..monitoring import OOMPolicy, is_oom_error, release_cuda_memory
from . import checkpoints as ck
from .data import (
    SEQ_KEY,
    DataPosition,
    PositionRegistry,
    TrainStream,
    iter_eval_batches,
    make_collator,
    make_encoder,
)
from .planning import plan_steps, validate_training_request

log = logging.getLogger(__name__)

LLAMA_STYLE_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
_KNOWN_TARGETS = frozenset(LLAMA_STYLE_TARGETS) | {"c_attn", "c_proj", "c_fc", "query_key_value", "dense",
                                                   "dense_h_to_4h", "dense_4h_to_h", "Wqkv", "out_proj", "fc1",
                                                   "fc2", "w1", "w2", "w3", "wq", "wk", "wv", "wo"}
_OPTIM = {"adamw_torch": "adamw_torch", "adamw_8bit": "adamw_8bit", "paged_adamw_8bit": "paged_adamw_8bit",
          "sgd": "sgd", "adafactor": "adafactor"}
PAUSE_POLL_SECONDS = 0.2


def auto_target_modules(model: torch.nn.Module) -> list[str] | str:
    """LoRA targets: known attention/MLP projection names present in ``model``, else ``"all-linear"``."""
    names = {n.rsplit(".", 1)[-1] for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)}
    names.discard("lm_head")
    found = sorted(names & _KNOWN_TARGETS)
    return found or "all-linear"


def resolve_dtype(model_cfg: ModelConfig, cuda: bool) -> torch.dtype:
    """float32 on CPU; on CUDA ``auto`` -> bf16 if supported else fp16."""
    if not cuda:
        return torch.float32
    if model_cfg.dtype == "float32":
        return torch.float32
    if model_cfg.dtype == "float16":
        return torch.float16
    if model_cfg.dtype == "bfloat16":
        return torch.bfloat16
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


@dataclass
class _RunState:
    """Mutable state of one ``run`` call, shared across OOM retries."""

    job_id: str
    max_steps: int
    micro: int
    accum: int
    checkpoints: list[CheckpointInfo] = field(default_factory=list)
    last_loss: float | None = None
    last_step: int = 0
    stopped: bool = False
    saved_step: int = -1


class AutoTuneCallback(TrainerCallback):
    """Trainer callback: events (status/metric/checkpoint/log) + cooperative control (stop/pause/save_now)."""

    def __init__(self, engine: HFTrainingEngine, rs: _RunState, events: EventSink, control: TrainingControl,
                 registry: PositionRegistry, model_cfg: ModelConfig, train_cfg: TrainingConfig,
                 dataset: StreamingDatasetProtocol, encode: Any, collate: Any) -> None:
        self.engine = engine
        self.rs = rs
        self.events = events
        self.ctl = control
        self.registry = registry
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.dataset = dataset
        self.encode = encode
        self.collate = collate
        self.trainer: Any = None  # set after the trainer is built
        rows = dataset.info.cardinality.rows
        self._total_rows = rows if rows else None

    # ------------------------------------------------------------------ helpers
    def _epoch(self) -> float | None:
        pos = self.registry.last_consumed
        if self._total_rows is None:
            return None
        frac = (pos.cursor.rows_consumed / self._total_rows) if pos.cursor is not None else 0.0
        return round(pos.pass_index + min(frac, 1.0), 4)

    def _status(self, **fields: Any) -> None:
        self.engine._publish(self.events, "status", fields)

    def _wait_while_paused(self, args: Any, state: Any, control: Any) -> None:
        if not self.ctl.pause.is_set() or self.ctl.stop.is_set():
            return
        self._status(state=TrainingState.PAUSED.value, step=state.global_step, message="paused")
        self.engine._log(self.events, "info", f"training paused at step {state.global_step}")
        while self.ctl.pause.is_set() and not self.ctl.stop.is_set():
            if self.ctl.save_now.is_set():
                self.ctl.save_now.clear()
                self._direct_save(args, state, control)
            self.ctl.stop.wait(PAUSE_POLL_SECONDS)
        if not self.ctl.stop.is_set():
            self._status(state=TrainingState.RUNNING.value, step=state.global_step, message="running")
            self.engine._log(self.events, "info", f"training resumed at step {state.global_step}")

    def _direct_save(self, args: Any, state: Any, control: Any) -> None:
        """Save while blocked between steps (save_now during pause)."""
        if self.rs.saved_step == state.global_step:
            return
        self.trainer._save_checkpoint(self.trainer.model, trial=None)
        self.on_save(args, state, control)
        control.should_save = False  # already saved at this step

    # ------------------------------------------------------------------ trainer hooks
    def on_train_begin(self, args: Any, state: Any, control: Any, **kw: Any) -> None:
        self.rs.last_step = state.global_step
        self._status(state=TrainingState.RUNNING.value, step=state.global_step, max_steps=state.max_steps,
                     backend=self.engine.backend, micro_batch_size=self.rs.micro,
                     effective_batch_size=self.rs.micro * self.rs.accum, supports_pause=True,
                     epoch=self._epoch(), message="training")

    def on_step_end(self, args: Any, state: Any, control: Any, **kw: Any) -> None:
        self.rs.last_step = state.global_step
        self._wait_while_paused(args, state, control)
        if self.ctl.stop.is_set():
            self.rs.stopped = True
            control.should_training_stop = True
            if self.rs.saved_step != state.global_step:
                control.should_save = True
            self._status(step=state.global_step, message="stopping: saving checkpoint")
        if self.ctl.save_now.is_set():
            self.ctl.save_now.clear()
            if self.rs.saved_step != state.global_step:
                control.should_save = True
        self._status(step=state.global_step, epoch=self._epoch())

    def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **kw: Any) -> None:
        logs = logs or {}
        if "loss" not in logs:
            return
        loss = float(logs["loss"])
        self.rs.last_loss = loss
        point = MetricPoint(step=state.global_step, loss=loss, learning_rate=_f(logs.get("learning_rate")),
                            grad_norm=_f(logs.get("grad_norm")), epoch=self._epoch())
        self.engine._publish(self.events, "metric", point.model_dump())
        self._status(step=state.global_step, loss=loss, learning_rate=point.learning_rate, epoch=point.epoch)

    def on_save(self, args: Any, state: Any, control: Any, **kw: Any) -> None:
        step = state.global_step
        ckpt_dir = Path(args.output_dir) / f"{ck.CHECKPOINT_PREFIX}{step}"
        is_final = step >= state.max_steps
        info = ck.write_checkpoint_state(
            ckpt_dir, job_id=self.rs.job_id, step=step, data_cursor=self.registry.last_consumed.to_dict(),
            model_cfg=self.model_cfg, train_cfg=self.train_cfg, is_final=is_final, backend=self.engine.backend,
            extra={"micro_batch_size": self.rs.micro, "gradient_accumulation_steps": self.rs.accum,
                   "loss": self.rs.last_loss},
        )
        self.rs.saved_step = step
        self.rs.checkpoints = [c for c in self.rs.checkpoints if c.step != step] + [info]
        self.engine._publish(self.events, "checkpoint", info.model_dump())
        self.engine._log(self.events, "info", f"checkpoint saved at step {step}: {info.path}")
        if self.train_cfg.val_fraction > 0:
            self._evaluate(step)

    # ------------------------------------------------------------------ evaluation
    def _evaluate(self, step: int) -> None:
        model = self.trainer.model
        was_training = model.training
        total, count = 0.0, 0
        try:
            model.eval()
            device = next(p for p in model.parameters()).device
            with torch.no_grad():
                for batch in iter_eval_batches(self.dataset, self.encode, self.collate, self.train_cfg.val_fraction,
                                               self.train_cfg.seed, self.rs.micro):
                    batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                    n = int((batch["labels"][..., 1:] != -100).sum())
                    if n == 0:
                        continue
                    out = model(**batch)
                    total += float(out.loss) * n
                    count += n
        except Exception as e:
            if is_oom_error(e):
                raise
            self.engine._log(self.events, "warning", f"evaluation failed at step {step}: {e}")
            return
        finally:
            if was_training:
                model.train()
        if count:
            point = MetricPoint(step=step, eval_loss=total / count, epoch=self._epoch())
            self.engine._publish(self.events, "metric", point.model_dump())


def _f(v: Any) -> float | None:
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


class HFTrainingEngine:
    """Transformers + PEFT LoRA engine (CPU or CUDA). Implements ``TrainingEngineProtocol``.

    ``extra_callbacks`` are appended to the Trainer callbacks (used by tests, e.g. to simulate OOM).
    """

    backend = "hf"

    def __init__(self, extra_callbacks: list[Any] | None = None) -> None:
        self.extra_callbacks = list(extra_callbacks or [])
        self.attempts = 0

    # ------------------------------------------------------------------ events
    @staticmethod
    def _publish(events: EventSink, type: str, data: dict[str, Any]) -> None:
        try:
            events.publish(type, data)
        except Exception:
            log.exception("failed to publish %s event", type)

    def _log(self, events: EventSink, level: str, message: str) -> None:
        getattr(log, level, log.info)(message)
        self._publish(events, "log", {"level": level, "source": "training", "message": message})

    # ------------------------------------------------------------------ model
    def _cuda(self) -> bool:
        return bool(torch.cuda.is_available())

    def load_model_and_tokenizer(self, model_cfg: ModelConfig, train_cfg: TrainingConfig) -> tuple[Any, Any]:
        """Base model (+4-bit for QLoRA on CUDA) wrapped with a fresh PEFT LoRA adapter, and its tokenizer."""
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        cuda = self._cuda()
        dtype = resolve_dtype(model_cfg, cuda)
        kwargs: dict[str, Any] = {"revision": model_cfg.revision, "dtype": dtype, "trust_remote_code": False}
        qlora = train_cfg.method == "qlora"
        if qlora:
            if not cuda:
                raise ValidationFailed('QLoRA requires a CUDA GPU; use method="lora" on CPU.')
            from transformers import BitsAndBytesConfig  # UNVERIFIED path (needs CUDA + bitsandbytes)

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True)
            kwargs["device_map"] = {"": torch.cuda.current_device()}
        tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_name, revision=model_cfg.revision,
                                                  trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(model_cfg.model_name, **kwargs)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        model.config.use_cache = False
        if qlora:
            from peft import prepare_model_for_kbit_training

            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        elif cuda:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.enable_input_require_grads()
        targets = train_cfg.target_modules or auto_target_modules(model)
        lora = LoraConfig(r=train_cfg.lora_r, lora_alpha=train_cfg.lora_alpha, lora_dropout=train_cfg.lora_dropout,
                          target_modules=targets, bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(model, lora)
        return model, tokenizer

    # ------------------------------------------------------------------ trainer
    def training_arguments(self, train_cfg: TrainingConfig, output_dir: Path, max_steps: int, micro: int,
                           accum: int, model_cfg: ModelConfig) -> TrainingArguments:
        cuda = self._cuda()
        dtype = resolve_dtype(model_cfg, cuda)
        return TrainingArguments(**self._args_kwargs(train_cfg, output_dir, max_steps, micro, accum, cuda, dtype))

    @staticmethod
    def _args_kwargs(train_cfg: TrainingConfig, output_dir: Path, max_steps: int, micro: int, accum: int,
                     cuda: bool, dtype: torch.dtype) -> dict[str, Any]:
        return {
            "output_dir": str(output_dir),
            "max_steps": int(max_steps),
            "per_device_train_batch_size": int(micro),
            "gradient_accumulation_steps": int(accum),
            "learning_rate": train_cfg.learning_rate,
            "lr_scheduler_type": train_cfg.lr_scheduler,
            "warmup_steps": train_cfg.warmup_steps,
            "weight_decay": train_cfg.weight_decay,
            "optim": _OPTIM[train_cfg.optimizer],
            "logging_strategy": "steps",
            "logging_steps": train_cfg.logging_steps,
            "save_strategy": "steps",
            "save_steps": train_cfg.save_steps,
            "save_total_limit": None,
            "seed": train_cfg.seed,
            "report_to": "none",
            "use_cpu": not cuda,
            "bf16": bool(cuda and dtype == torch.bfloat16),
            "fp16": bool(cuda and dtype == torch.float16),
            "remove_unused_columns": False,
            "disable_tqdm": True,
            "dataloader_num_workers": 0,
            "dataloader_pin_memory": cuda,
            "ignore_data_skip": True,  # data position comes from the stream cursor, never by re-reading batches
        }

    def trainer_class(self) -> type:
        from transformers import Trainer

        return _cursor_tracking(Trainer)

    def build_trainer(self, model: Any, tokenizer: Any, args: TrainingArguments, train_ds: Any, collate: Any,
                      callbacks: list[Any]) -> Any:
        cls = self.trainer_class()
        return cls(model=model, args=args, train_dataset=train_ds, data_collator=collate,
                   processing_class=tokenizer, callbacks=callbacks)

    # ------------------------------------------------------------------ run
    def run(
        self,
        job_id: str,
        model_cfg: ModelConfig,
        train_cfg: TrainingConfig,
        dataset: StreamingDatasetProtocol,
        events: EventSink,
        control: TrainingControl,
        resume_from: CheckpointInfo | None = None,
    ) -> TrainingResult:
        """Train synchronously; returns after the final (or stop) checkpoint is on disk."""
        if not train_cfg.output_dir:
            raise ValidationFailed("training_config.output_dir must be set by the caller")
        output_dir = Path(train_cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for w in validate_training_request(model_cfg, train_cfg, dataset.info, None):
            self._log(events, "warning", w)
        max_steps = plan_steps(train_cfg, dataset.info)
        if resume_from is not None:
            self._check_resumable(resume_from)
        self._publish(events, "status", {"state": TrainingState.PREPARING.value, "backend": self.backend,
                                         "max_steps": max_steps, "supports_pause": True,
                                         "micro_batch_size": train_cfg.per_device_batch_size,
                                         "effective_batch_size": train_cfg.per_device_batch_size
                                         * train_cfg.gradient_accumulation_steps,
                                         "message": "loading model"})
        policy = OOMPolicy(train_cfg.per_device_batch_size, train_cfg.gradient_accumulation_steps, events=events)
        rs = _RunState(job_id=job_id, max_steps=max_steps, micro=policy.micro_batch, accum=policy.grad_accum)
        resume = resume_from
        while True:
            rs.micro, rs.accum = policy.micro_batch, policy.grad_accum
            try:
                self._attempt(rs, model_cfg, train_cfg, dataset, events, control, output_dir, resume)
                break
            except StudioError:
                raise
            except BaseException as e:
                if not is_oom_error(e):
                    raise
                self._release()
                if not train_cfg.adaptive_batch_size:
                    raise OutOfMemory("Out of memory (adaptive_batch_size is disabled): reduce "
                                      "per_device_batch_size or max_seq_length.") from e
                policy.on_oom()  # raises OutOfMemory at the minimum micro-batch
                latest = ck.latest_checkpoint(output_dir, job_id)
                resume = latest or resume_from
                where = (f"checkpoint step {resume.step}" if resume is not None
                         else "the beginning (no checkpoint yet; no data skipped)")
                self._log(events, "warning", f"out of memory: retrying with micro-batch {policy.micro_batch} x "
                                             f"accumulation {policy.grad_accum}, resuming from {where}")
                self._publish(events, "status", {"micro_batch_size": policy.micro_batch,
                                                 "effective_batch_size": policy.effective_batch_size,
                                                 "message": "recovering from out of memory"})
        self._release()
        return TrainingResult(job_id=job_id, final_step=rs.last_step, final_loss=rs.last_loss,
                              stopped_early=rs.stopped and rs.last_step < max_steps,
                              checkpoints=sorted(rs.checkpoints, key=lambda c: c.step), output_dir=str(output_dir))

    @staticmethod
    def _check_resumable(ckpt: CheckpointInfo) -> None:
        p = Path(ckpt.path)
        if not (p / "trainer_state.json").is_file():
            raise ValidationFailed(f"checkpoint {ckpt.checkpoint_id} has no trainer state; only checkpoints "
                                   "written by AutoTune Studio can be resumed")
        if not any((p / n).is_file() for n in ("adapter_model.safetensors", "adapter_model.bin")):
            raise ValidationFailed(f"checkpoint {ckpt.checkpoint_id} has no adapter weights")

    def _release(self) -> None:
        self._trainer = None
        release_cuda_memory()

    def _attempt(self, rs: _RunState, model_cfg: ModelConfig, train_cfg: TrainingConfig,
                 dataset: StreamingDatasetProtocol, events: EventSink, control: TrainingControl,
                 output_dir: Path, resume: CheckpointInfo | None) -> None:
        self.attempts += 1
        model, tokenizer = self.load_model_and_tokenizer(model_cfg, train_cfg)
        import datasets

        start = DataPosition.from_dict(resume.data_cursor) if resume is not None else DataPosition()
        registry = PositionRegistry(start)
        encode = make_encoder(dataset, tokenizer, model_cfg.max_seq_length)
        collate = make_collator(tokenizer.pad_token_id)
        stream_stop = threading.Event()
        gen = TrainStream(dataset, encode, registry, start, train_cfg.val_fraction, train_cfg.seed, stream_stop)
        train_ds = datasets.IterableDataset.from_generator(gen)
        args = self.training_arguments(train_cfg, output_dir, rs.max_steps, rs.micro, rs.accum, model_cfg)
        cb = AutoTuneCallback(self, rs, events, control, registry, model_cfg, train_cfg, dataset, encode, collate)
        trainer = self.build_trainer(model, tokenizer, args, train_ds, collate, [cb, *self.extra_callbacks])
        trainer.autotune_registry = registry
        from transformers.trainer_callback import PrinterCallback, ProgressCallback

        trainer.remove_callback(PrinterCallback)
        trainer.remove_callback(ProgressCallback)
        cb.trainer = trainer
        self._trainer = trainer
        t0 = time.monotonic()
        try:
            trainer.train(resume_from_checkpoint=resume.path if resume is not None else None)
        finally:
            stream_stop.set()
        # Guarantee a checkpoint at the last step (normally saved by the Trainer at max_steps / stop).
        if rs.saved_step != trainer.state.global_step and trainer.state.global_step > 0:
            trainer._save_checkpoint(trainer.model, trial=None)
            cb.on_save(args, trainer.state, trainer.control)
        rs.last_step = trainer.state.global_step
        self._log(events, "info", f"training finished at step {rs.last_step} in {time.monotonic() - t0:.1f}s")
        del trainer, model


def _cursor_tracking(base: type) -> type:
    """Subclass ``base`` (Trainer / SFTTrainer) so the trained micro-batches advance the data position."""

    class CursorTrackingTrainer(base):  # type: ignore[valid-type, misc]
        autotune_registry: PositionRegistry | None = None

        def training_step(self, model: Any, inputs: dict[str, Any], num_items_in_batch: Any = None) -> Any:
            seq = inputs.pop(SEQ_KEY, None)
            rows = inputs.pop("_atk_rows", 0)
            loss = super().training_step(model, inputs, num_items_in_batch)
            if seq is not None and self.autotune_registry is not None:
                self.autotune_registry.consume(int(seq), int(rows))
            return loss

        def prediction_step(self, model: Any, inputs: dict[str, Any], *a: Any, **kw: Any) -> Any:
            inputs.pop(SEQ_KEY, None)
            inputs.pop("_atk_rows", None)
            return super().prediction_step(model, inputs, *a, **kw)

    CursorTrackingTrainer.__name__ = f"CursorTracking{base.__name__}"
    return CursorTrackingTrainer


__all__ = ["AutoTuneCallback", "HFTrainingEngine", "auto_target_modules", "resolve_dtype"]
