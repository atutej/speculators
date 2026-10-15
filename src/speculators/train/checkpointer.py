import functools
import json
import logging
import shutil
from abc import abstractmethod
from pathlib import Path

import torch
import torch.distributed as dist
import torch.utils._pytree as pytree
from safetensors import safe_open
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.nn.parallel import DistributedDataParallel
from transformers.modeling_utils import PreTrainedModel

from speculators.train.distributed import get_rank, is_distributed
from speculators.utils.util import get_current_device

logger = logging.getLogger("speculators")

# Optimizers/schedulers may be a single object (legacy) or a list (e.g. Muon + AdamW).
OptimizerOrList = torch.optim.Optimizer | list[torch.optim.Optimizer]
SchedulerOrList = (
    torch.optim.lr_scheduler.LRScheduler | list[torch.optim.lr_scheduler.LRScheduler]
)


def _as_list(value):
    """Normalize a single object or a list/tuple of objects into a list."""
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _rank0_only(fn):
    """Execute *fn* only on rank 0, then barrier (no-op when not distributed)."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        result = None
        if get_rank() == 0:
            result = fn(*args, **kwargs)
        if is_distributed():
            dist.barrier()
        return result

    return wrapper


class BaseCheckpointer:
    """Helper class to save and load checkpoints.

    Checkpoint file structure:
    ../path/
        0/ # epoch number
            model.safetensors
            optimizer_state_dict.pt
            scheduler_state_dict.pt (optional)
        1/
            model.safetensors
            optimizer_state_dict.pt
            scheduler_state_dict.pt (optional)
        ...

    With ``layout="step"`` the periodic (mid-epoch) and interrupt checkpoints are
    instead kept as ``checkpoint-<global_step>/`` directories, like the Hugging Face
    ``Trainer``: each is written to ``checkpoint-<step>.tmp/`` and renamed only once
    every file is on disk, so a kill never leaves a half-written checkpoint; only the
    newest ``save_total_limit`` are kept (plus optional milestones). End-of-epoch
    checkpoints stay in the ``<epoch>/`` directories. Resume takes whichever complete
    checkpoint has the highest ``global_step`` in its ``training_state.json``.
    """

    STEP_PREFIX = "checkpoint-"
    TMP_SUFFIX = ".tmp"
    TRAINING_STATE_FILENAME = "training_state.json"

    def __init__(self, path: Path | str, layout: str = "epoch"):
        if layout not in ("epoch", "step"):
            raise ValueError(f"layout must be 'epoch' or 'step', got {layout!r}")
        self.path = Path(path)
        self.layout = layout
        self.previous_epoch = self._get_previous_epoch()
        # Name of the directory (inside ``path``) that the load_* methods read.
        self._resume_name: str | None = None

        if layout == "step":
            self._resolve_step_resume()

        if self.previous_epoch != -1:
            self.prev_path: Path | None = self.path / self.load_name
        else:
            self.prev_path = None

    @property
    def load_name(self) -> str:
        """Directory name that resume reads from (an epoch number or checkpoint-N)."""
        if self._resume_name is not None:
            return self._resume_name
        return str(self.previous_epoch)

    # ------------------------------------------------------------------ step layout
    def read_training_state(self, name: str) -> dict:
        """Return ``training_state.json`` of a checkpoint directory ({} if unreadable)."""
        p = self.path / name / self.TRAINING_STATE_FILENAME
        try:
            return json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            return {}

    def _is_complete(self, name: str) -> bool:
        d = self.path / name
        return (
            d.is_dir()
            and not d.is_symlink()
            and (d / "model.safetensors").exists()
            and (d / "optimizer_state_dict.pt").exists()
            and (d / self.TRAINING_STATE_FILENAME).exists()
        )

    def list_step_checkpoints(self) -> list[tuple[int, Path]]:
        """Complete ``checkpoint-<step>`` directories, oldest first."""
        found: list[tuple[int, Path]] = []
        if not self.path.exists():
            return found
        for d in self.path.iterdir():
            name = d.name
            if not name.startswith(self.STEP_PREFIX) or name.endswith(self.TMP_SUFFIX):
                continue
            try:
                step = int(name[len(self.STEP_PREFIX) :])
            except ValueError:
                continue
            if self._is_complete(name):
                found.append((step, d))
        return sorted(found)

    def _resolve_step_resume(self) -> None:
        """Pick the complete checkpoint with the highest global_step to resume from."""
        best_step = -2  # an epoch dir without training state still counts, as before
        best_name: str | None = None
        best_epoch = self.previous_epoch
        for step, d in self.list_step_checkpoints():
            state = self.read_training_state(d.name)
            if step > best_step:
                best_step, best_name = step, d.name
                best_epoch = int(state.get("epoch", 0))
        if self.previous_epoch != -1:
            state = self.read_training_state(str(self.previous_epoch))
            epoch_step = int(state.get("global_step", -1))
            if best_name is None or epoch_step >= best_step:
                return  # the newest epoch-end checkpoint wins: keep the legacy path
        if best_name is not None:
            self._resume_name = best_name
            self.previous_epoch = best_epoch

    @_rank0_only
    def prepare_step_tmp(self, tmp_name: str) -> None:
        """Remove a leftover temporary directory from a save that was killed."""
        stale = self.path / tmp_name
        if stale.exists():
            shutil.rmtree(stale)

    @_rank0_only
    def finalize_step_checkpoint(
        self,
        tmp_name: str,
        final_name: str,
        save_total_limit: int,
        keep_every_steps: int = 0,
    ) -> None:
        """Atomically publish ``tmp_name`` as ``final_name``, then prune old ones."""
        tmp, final = self.path / tmp_name, self.path / final_name
        if not tmp.is_dir():
            raise FileNotFoundError(f"temporary checkpoint {tmp} does not exist")
        old = None
        if final.exists():  # the same step saved again: swap, then drop the old copy
            old = self.path / (final_name + ".old")
            if old.exists():
                shutil.rmtree(old)
            final.rename(old)
        tmp.rename(final)
        if old is not None:
            shutil.rmtree(old, ignore_errors=True)
        self._prune_step_checkpoints(save_total_limit, keep_every_steps)

    def _prune_step_checkpoints(self, save_total_limit: int, keep_every_steps: int):
        if save_total_limit <= 0:
            return  # keep everything
        steps = self.list_step_checkpoints()
        keep = {d for _, d in steps[-save_total_limit:]}
        for step, d in steps:
            if d in keep or (keep_every_steps > 0 and step % keep_every_steps == 0):
                continue
            shutil.rmtree(d, ignore_errors=True)
        for stale in self.path.glob(f"{self.STEP_PREFIX}*{self.TMP_SUFFIX}"):
            shutil.rmtree(stale, ignore_errors=True)

    @abstractmethod
    def load_model_state_dict(
        self, model: PreTrainedModel, float_dtype: torch.dtype | None = None
    ):
        raise NotImplementedError

    @abstractmethod
    def load_optimizer_state_dict(
        self,
        model: PreTrainedModel,
        optimizer: OptimizerOrList,
        float_dtype: torch.dtype | None = None,
    ):
        raise NotImplementedError

    def load_scheduler_state_dict(self, scheduler: SchedulerOrList):
        scheduler_path = self.scheduler_path(self.load_name)
        if not scheduler_path.exists():
            return
        loaded = torch.load(scheduler_path, weights_only=True)
        schedulers = _as_list(scheduler)
        loaded_list = loaded if isinstance(loaded, list) else [loaded]
        for sched, state_dict in zip(schedulers, loaded_list, strict=True):
            sched.load_state_dict(state_dict)
            for param_group, lr in zip(
                sched.optimizer.param_groups, sched.get_last_lr(), strict=True
            ):
                param_group["lr"] = lr

    @_rank0_only
    def save_scheduler_state_dict(self, scheduler: SchedulerOrList, epoch: int | str):
        schedulers = _as_list(scheduler)
        state_dicts = [sched.state_dict() for sched in schedulers]
        # Preserve the legacy single-scheduler format when there is only one.
        payload = state_dicts[0] if len(state_dicts) == 1 else state_dicts
        torch.save(payload, self.scheduler_path(epoch))

    @abstractmethod
    def save_checkpoint(
        self,
        model: PreTrainedModel,
        optimizer: OptimizerOrList,
        epoch: int | str,
        float_dtype: torch.dtype = torch.bfloat16,
    ):
        raise NotImplementedError

    def _get_previous_epoch(self) -> int:
        if not self.path.exists():
            return -1
        last_checkpoint_num = -1
        for d in self.path.iterdir():
            if d.is_symlink():
                continue  # skip descriptive symlinks like epoch0_step16626
            if d.is_dir():
                if d.name == "interrupted":
                    logger.warning(
                        f"Found interrupted checkpoint at {d}. "
                        "To resume from it, rename it to an epoch number "
                        "(e.g., 'mv interrupted 5' to resume as epoch 5)."
                    )
                    continue
                try:
                    last_checkpoint_num = max(last_checkpoint_num, int(d.name))
                except ValueError:
                    continue
        return last_checkpoint_num

    def model_path(self, epoch: int | str):
        model_fname = "model.safetensors"
        return self.path / str(epoch) / model_fname

    def optimizer_path(self, epoch: int | str):
        optimizer_fname = "optimizer_state_dict.pt"
        return self.path / str(epoch) / optimizer_fname

    def scheduler_path(self, epoch: int | str):
        scheduler_fname = "scheduler_state_dict.pt"
        return self.path / str(epoch) / scheduler_fname

    def best_path(self) -> Path:
        return self.path / "checkpoint_best"

    def val_metrics_path(self, epoch: int) -> Path:
        return self.path / str(epoch) / "val_metrics.json"

    TRAIN_COMMAND_FILENAME = "train_command.txt"

    def _copy_train_command(self, epoch: int | str) -> None:
        src = self.path / self.TRAIN_COMMAND_FILENAME
        if src.exists():
            shutil.copy2(src, self.path / str(epoch) / self.TRAIN_COMMAND_FILENAME)

    @_rank0_only
    def save_val_metrics(self, epoch: int, val_metrics: dict[str, float]):
        path = self.val_metrics_path(epoch)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(val_metrics))

    def load_best_val_loss(self) -> float | None:
        best_epoch = self.read_best_epoch()
        if best_epoch is None:
            return None
        p = self.val_metrics_path(best_epoch)
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text())
            return float(data["loss_epoch"])
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            return None

    def read_best_epoch(self) -> int | None:
        """Return the epoch that `checkpoint_best` points to."""
        best_path = self.best_path()
        if not best_path.exists() or not best_path.is_symlink():
            return None
        try:
            target = best_path.readlink()
        except OSError:
            return None
        try:
            return int(Path(target).name)
        except ValueError:
            return None

    def load_model_state_dict_for_epoch(
        self, model: PreTrainedModel, epoch: int, float_dtype: torch.dtype | None = None
    ):
        """Temporarily load weights for a specific epoch."""
        old_epoch, old_name = self.previous_epoch, self._resume_name
        try:
            self.previous_epoch = epoch
            self._resume_name = None  # read <epoch>/, not the step checkpoint
            self.load_model_state_dict(model, float_dtype=float_dtype)
        finally:
            self.previous_epoch, self._resume_name = old_epoch, old_name

    @_rank0_only
    def update_best_symlink(self, epoch: int):
        best_path = self.best_path()
        target = Path(str(epoch))  # relative symlink inside checkpoint root

        if best_path.is_symlink() or best_path.exists():
            if best_path.is_dir() and not best_path.is_symlink():
                shutil.rmtree(best_path)
            else:
                best_path.unlink()

        best_path.symlink_to(target, target_is_directory=True)

    @_rank0_only
    def cleanup_keep_only_best(self, best_epoch: int) -> None:
        """
        Delete all epoch dir. except best_epoch, and keep best_checkpoint symlink.
        """
        keep_dir = self.path / str(best_epoch)
        best_link = self.best_path()

        # Safety checks
        if not keep_dir.exists() or not keep_dir.is_dir():
            raise FileNotFoundError(f"Best epoch dir does not exist: {keep_dir}")

        train_cmd_file = self.path / self.TRAIN_COMMAND_FILENAME

        for child in self.path.iterdir():
            # Keep the symlink itself
            if child == best_link:
                continue

            # Keep the best epoch directory
            if child == keep_dir:
                continue

            if child == train_cmd_file:
                continue

            # Delete numbered epoch directories and any other stray dirs/files
            try:
                if child.is_symlink() or child.is_file():
                    child.unlink()
                elif child.is_dir():
                    shutil.rmtree(child)
            except (FileNotFoundError, PermissionError, OSError) as exc:
                raise RuntimeError(f"Failed to delete {child}") from exc


def convert_float_dtype(sd: pytree.PyTree, dtype: torch.dtype) -> pytree.PyTree:
    def convert_fn(x):
        if isinstance(x, torch.Tensor) and x.is_floating_point():
            return x.to(dtype)
        return x

    return pytree.tree_map(convert_fn, sd)


def load_safetensors_state_dict(path: Path, device: str) -> dict[str, torch.Tensor]:
    full_state_dict = {}
    with safe_open(path, framework="pt", device=device) as f:
        for key in f.keys():  # noqa: SIM118
            full_state_dict[key] = f.get_tensor(key)
    return full_state_dict


def patch_config_dtype(config_path: Path, float_dtype: torch.dtype) -> None:
    """Patch config.json to match the actual on-disk tensor dtype.

    When models are kept in FP32 but saved as BF16, save_pretrained writes
    the in-memory dtype to config.json. This patches it to match the saved dtype.
    """
    if not config_path.exists():
        return

    config = json.loads(config_path.read_text())
    # Convert torch.bfloat16 -> "bfloat16"
    dtype_str = str(float_dtype).split(".")[-1]
    # Support both dtype (transformers 5.x) and torch_dtype (older versions)
    if "dtype" in config:
        config["dtype"] = dtype_str
    if "torch_dtype" in config:
        config["torch_dtype"] = dtype_str
    config_path.write_text(json.dumps(config, indent=2) + "\n")


class SingleGPUCheckpointer(BaseCheckpointer):
    def load_model_state_dict(
        self, model: PreTrainedModel, float_dtype: torch.dtype | None = None
    ):
        device = get_current_device()
        full_state_dict = load_safetensors_state_dict(
            self.model_path(self.load_name),
            device,
        )
        full_state_dict = convert_float_dtype(
            full_state_dict, float_dtype or model.dtype
        )
        # Note: `strict=False` because we don't load the verifier weights
        model.load_state_dict(full_state_dict, strict=False)

    def load_optimizer_state_dict(
        self,
        model: PreTrainedModel,
        optimizer: OptimizerOrList,
        float_dtype: torch.dtype | None = None,
    ):
        device = get_current_device()
        loaded = torch.load(
            self.optimizer_path(self.load_name),
            weights_only=True,
            map_location=device,
        )
        optimizers = _as_list(optimizer)
        loaded_list = loaded if isinstance(loaded, list) else [loaded]
        raw_model = (
            model.module if isinstance(model, DistributedDataParallel) else model
        )
        dtype = float_dtype or raw_model.dtype
        for opt, state_dict in zip(optimizers, loaded_list, strict=True):
            opt.load_state_dict(convert_float_dtype(state_dict, dtype))

    @_rank0_only
    def save_checkpoint(
        self,
        model: PreTrainedModel,
        optimizer: OptimizerOrList,
        epoch: int | str,
        float_dtype: torch.dtype = torch.bfloat16,
    ):
        raw_model: PreTrainedModel = (
            model.module if isinstance(model, DistributedDataParallel) else model
        )  # type: ignore[assignment]
        model_state_dict = convert_float_dtype(raw_model.state_dict(), float_dtype)
        raw_model.save_pretrained(self.path / str(epoch), state_dict=model_state_dict)
        patch_config_dtype(self.path / str(epoch) / "config.json", float_dtype)

        optimizers = _as_list(optimizer)
        state_dicts = [
            convert_float_dtype(opt.state_dict(), float_dtype) for opt in optimizers
        ]
        # Preserve the legacy single-optimizer format when there is only one.
        payload = state_dicts[0] if len(state_dicts) == 1 else state_dicts
        torch.save(payload, self.optimizer_path(epoch))
        self._copy_train_command(epoch)


class DistributedCheckpointer(BaseCheckpointer):
    def load_model_state_dict(
        self, model: PreTrainedModel, float_dtype: torch.dtype | None = None
    ):
        full_state_dict = load_safetensors_state_dict(
            self.model_path(self.load_name), "cpu"
        )
        full_state_dict = convert_float_dtype(
            full_state_dict, float_dtype or model.dtype
        )

        # Note: `strict=False` because we don't load the verifier weights
        set_model_state_dict(
            model,
            full_state_dict,  # type: ignore[arg-type]
            options=StateDictOptions(
                full_state_dict=True, broadcast_from_rank0=True, strict=False
            ),
        )
        dist.barrier()

    def load_optimizer_state_dict(
        self,
        model,
        optimizer: OptimizerOrList,
        float_dtype: torch.dtype | None = None,
    ):
        optimizers = _as_list(optimizer)
        full_state_dict = torch.load(
            self.optimizer_path(self.load_name),
            mmap=True,
            weights_only=True,
            map_location="cpu",
        )
        full_state_dict = convert_float_dtype(
            full_state_dict, float_dtype or model.dtype
        )

        set_optimizer_state_dict(
            model,
            optimizers,
            full_state_dict,
            options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True),
        )

        # Cast step counters back to float32
        for opt in optimizers:
            for state in opt.state.values():
                if "step" in state and isinstance(state["step"], torch.Tensor):
                    state["step"] = state["step"].float()

        dist.barrier()

    def save_checkpoint(
        self,
        model: PreTrainedModel,
        optimizer: OptimizerOrList,
        epoch: int | str,
        float_dtype: torch.dtype = torch.bfloat16,
    ):
        model_state_dict = get_model_state_dict(
            model, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
        )
        model_state_dict = convert_float_dtype(model_state_dict, float_dtype)

        optimizer_state_dict = get_optimizer_state_dict(
            model,
            _as_list(optimizer),
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )
        optimizer_state_dict = convert_float_dtype(optimizer_state_dict, float_dtype)

        if get_rank() == 0:
            # Only rank 0 saves the checkpoint
            model.save_pretrained(self.path / str(epoch), state_dict=model_state_dict)
            patch_config_dtype(self.path / str(epoch) / "config.json", float_dtype)
            torch.save(optimizer_state_dict, self.optimizer_path(epoch))
            self._copy_train_command(epoch)

        dist.barrier()
