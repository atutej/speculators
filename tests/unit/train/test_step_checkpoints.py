"""Unit tests for the HF-style ``checkpoint_layout="step"`` checkpoints."""

import json
import tempfile
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from speculators.train.checkpointer import (
    BaseCheckpointer,
    DistributedCheckpointer,
    SingleGPUCheckpointer,
)
from speculators.train.trainer import Trainer, TrainerConfig


class _Ckpt(BaseCheckpointer):
    """Concrete checkpointer for directory-handling tests (no tensor I/O)."""

    def load_model_state_dict(self, *a, **k):
        pass

    def load_optimizer_state_dict(self, *a, **k):
        pass

    def save_checkpoint(self, *a, **k):
        pass


def _write_ckpt(root: Path, name: str, epoch=0, local_step=0, global_step=0) -> None:
    d = root / name
    d.mkdir(parents=True)
    (d / "model.safetensors").write_bytes(b"m")
    (d / "optimizer_state_dict.pt").write_bytes(b"o")
    (d / "training_state.json").write_text(
        json.dumps(
            {"epoch": epoch, "local_step": local_step, "global_step": global_step}
        )
    )


# ------------------------------------------------------------ checkpointer level


def test_discovery_ignores_tmp_and_incomplete(tmp_path: Path) -> None:
    _write_ckpt(tmp_path, "checkpoint-10", local_step=10, global_step=10)
    _write_ckpt(tmp_path, "checkpoint-20.tmp", local_step=20, global_step=20)
    (tmp_path / "checkpoint-30").mkdir()  # killed before any file was written
    (tmp_path / "checkpoint-40").mkdir()
    (tmp_path / "checkpoint-40" / "model.safetensors").write_bytes(b"m")  # no state
    (tmp_path / "checkpoint-junk").mkdir()
    c = _Ckpt(tmp_path, layout="step")
    assert [s for s, _ in c.list_step_checkpoints()] == [10]
    assert c.load_name == "checkpoint-10"
    assert c.previous_epoch == 0
    assert c.prev_path == tmp_path / "checkpoint-10"


def test_resume_picks_highest_step_over_epoch_dir(tmp_path: Path) -> None:
    _write_ckpt(tmp_path, "0", epoch=0, local_step=5, global_step=5)
    _write_ckpt(tmp_path, "checkpoint-9", epoch=0, local_step=9, global_step=9)
    assert _Ckpt(tmp_path, layout="step").load_name == "checkpoint-9"


def test_resume_prefers_newer_epoch_end_dir(tmp_path: Path) -> None:
    _write_ckpt(tmp_path, "checkpoint-9", epoch=0, local_step=9, global_step=9)
    _write_ckpt(tmp_path, "0", epoch=0, local_step=0, global_step=12)
    assert _Ckpt(tmp_path, layout="step").load_name == "0"


def test_epoch_layout_ignores_step_dirs(tmp_path: Path) -> None:
    _write_ckpt(tmp_path, "checkpoint-9", local_step=9, global_step=9)
    c = _Ckpt(tmp_path)
    assert c.previous_epoch == -1
    assert c.prev_path is None


def test_bad_layout_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="layout"):
        _Ckpt(tmp_path, layout="hf")


def test_finalize_is_atomic_and_prunes(tmp_path: Path) -> None:
    c = _Ckpt(tmp_path, layout="step")
    for step in (10, 20, 30, 40):
        _write_ckpt(tmp_path, f"checkpoint-{step}.tmp", global_step=step)
        c.finalize_step_checkpoint(
            f"checkpoint-{step}.tmp", f"checkpoint-{step}", save_total_limit=2
        )
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "checkpoint-30",
        "checkpoint-40",
    ]


def test_prune_keeps_milestones(tmp_path: Path) -> None:
    c = _Ckpt(tmp_path, layout="step")
    for step in (100, 150, 200, 250, 300):
        _write_ckpt(tmp_path, f"checkpoint-{step}.tmp", global_step=step)
        c.finalize_step_checkpoint(
            f"checkpoint-{step}.tmp",
            f"checkpoint-{step}",
            save_total_limit=1,
            keep_every_steps=100,
        )
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "checkpoint-100",
        "checkpoint-200",
        "checkpoint-300",
    ]


def test_finalize_same_step_replaces(tmp_path: Path) -> None:
    c = _Ckpt(tmp_path, layout="step")
    _write_ckpt(tmp_path, "checkpoint-7", global_step=7)
    _write_ckpt(tmp_path, "checkpoint-7.tmp", global_step=7)
    (tmp_path / "checkpoint-7.tmp" / "marker").write_text("new")
    c.finalize_step_checkpoint("checkpoint-7.tmp", "checkpoint-7", 3)
    assert (tmp_path / "checkpoint-7" / "marker").read_text() == "new"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["checkpoint-7"]


def test_prepare_step_tmp_removes_stale(tmp_path: Path) -> None:
    c = _Ckpt(tmp_path, layout="step")
    _write_ckpt(tmp_path, "checkpoint-5.tmp", global_step=5)
    c.prepare_step_tmp("checkpoint-5.tmp")
    assert not (tmp_path / "checkpoint-5.tmp").exists()


def test_finalize_missing_tmp_raises(tmp_path: Path) -> None:
    c = _Ckpt(tmp_path, layout="step")
    with pytest.raises(FileNotFoundError):
        c.finalize_step_checkpoint("checkpoint-1.tmp", "checkpoint-1", 3)


# ------------------------------------------------------------------ trainer level


@pytest.fixture(autouse=True)
def _patch_checkpointer(monkeypatch: pytest.MonkeyPatch) -> None:
    def _save_checkpoint(self, model, optimizers, epoch, *a, **k):
        d = self.path / str(epoch)
        d.mkdir(parents=True, exist_ok=True)
        (d / "model.safetensors").write_bytes(b"m")
        (d / "optimizer_state_dict.pt").write_bytes(b"o")

    def _noop(*_a, **_k):
        return None

    for cls in (SingleGPUCheckpointer, DistributedCheckpointer):
        monkeypatch.setattr(cls, "save_checkpoint", _save_checkpoint)
        monkeypatch.setattr(cls, "save_scheduler_state_dict", _noop)
        monkeypatch.setattr(cls, "load_model_state_dict", _noop)
        monkeypatch.setattr(cls, "load_optimizer_state_dict", _noop)
        monkeypatch.setattr(cls, "load_scheduler_state_dict", _noop)


class _DS(Dataset):
    def __len__(self) -> int:
        return 100

    def __getitem__(self, i: int) -> dict:
        return {"input_ids": torch.tensor([i]), "loss_mask": torch.tensor([1.0])}


class _T(Trainer):
    def setup_model(self) -> None:
        pass

    def setup_optimizer(self) -> None:
        opt = torch.optim.AdamW([nn.Parameter(torch.zeros(1))], lr=1e-4)
        self.opt, self.optimizers = opt, [opt]
        self.scheduler, self.schedulers = None, []


def _trainer(path: str, resume: bool = False, **over) -> _T:
    cfg = TrainerConfig(
        save_path=path,
        num_epochs=2,
        lr=1e-4,
        resume_from_checkpoint=resume,
        checkpoint_freq=0.3,
        checkpoint_layout="step",
        save_total_limit=2,
        scheduler_type="none",
        **over,
    )
    return _T(nn.Identity(), cfg, DataLoader(_DS(), batch_size=10, shuffle=False))


def test_mid_epoch_save_creates_step_dir_and_resumes() -> None:
    with tempfile.TemporaryDirectory() as d:
        t = _trainer(d)
        t.global_step = 3
        t.maybe_save_checkpoint(0, local_step=3)
        assert sorted(p.name for p in Path(d).iterdir()) == ["checkpoint-3"]
        assert not (Path(d) / "0").exists()  # nothing written in the epoch layout

        t2 = _trainer(d, resume=True)
        assert t2.current_epoch == 0
        assert t2._resume_local_step == 3
        assert t2.global_step == 3


def test_step_saves_prune_to_limit() -> None:
    with tempfile.TemporaryDirectory() as d:
        t = _trainer(d)
        for local in (3, 6, 9):
            t.global_step = local
            t.maybe_save_checkpoint(0, local_step=local)
        assert sorted(p.name for p in Path(d).iterdir()) == [
            "checkpoint-6",
            "checkpoint-9",
        ]


def test_end_of_epoch_uses_epoch_dir_and_wins_resume() -> None:
    with tempfile.TemporaryDirectory() as d:
        t = _trainer(d)
        t.global_step = 3
        t.maybe_save_checkpoint(0, local_step=3)
        t.global_step = 10
        t.maybe_save_checkpoint(0, local_step=0)
        assert (Path(d) / "0" / "model.safetensors").exists()

        t2 = _trainer(d, resume=True)
        assert t2.current_epoch == 1
        assert t2._resume_local_step == 0
        assert t2.global_step == 10


def test_interrupt_save_is_a_resumable_step_checkpoint() -> None:
    with tempfile.TemporaryDirectory() as d:
        t = _trainer(d)
        t.global_step = 14
        t._cur_epoch, t._cur_local_step = 1, 4
        t.maybe_save_checkpoint("interrupted")
        assert sorted(p.name for p in Path(d).iterdir()) == ["checkpoint-14"]

        t2 = _trainer(d, resume=True)
        assert t2.current_epoch == 1
        assert t2._resume_local_step == 4
        assert t2.global_step == 14


def test_interrupt_before_first_step_falls_back_to_epoch_layout() -> None:
    with tempfile.TemporaryDirectory() as d:
        t = _trainer(d)
        t.maybe_save_checkpoint("interrupted")
        assert (Path(d) / "interrupted").is_dir()
        assert not list(Path(d).glob("checkpoint-*"))
