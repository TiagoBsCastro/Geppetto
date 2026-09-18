import importlib.util
import os
from pathlib import Path

import pytest


def load_utility():
    path = Path(__file__).parents[1] / "scripts/leonardo/restore_job_affinity.py"
    spec = importlib.util.spec_from_file_location("restore_job_affinity", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_affinity_repair_requires_slurm(monkeypatch):
    module = load_utility()
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="inside the target Slurm allocation"):
        module.restore(os.getpid())


def test_affinity_guard_matches_complete_job_id():
    module = load_utility()
    assert module.contains_job("0::/slurm/job_123/step_0", "123")
    assert module.contains_job("0::/slurm/job_123.scope/step_0", "123")
    assert not module.contains_job("0::/slurm/job_1234/step_0", "123")


def test_affinity_repair_rejects_single_cpu_step(monkeypatch):
    module = load_utility()
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setattr(module.os, "sched_getaffinity", lambda _: {0}, raising=False)
    with pytest.raises(RuntimeError, match="full CPU allocation"):
        module.restore(os.getpid())


@pytest.mark.skipif(not Path("/proc/self/cgroup").exists(), reason="Linux cgroup guard")
def test_affinity_repair_rejects_different_job(monkeypatch):
    module = load_utility()
    monkeypatch.setenv("SLURM_JOB_ID", "geppetto-test-nonexistent-job")
    monkeypatch.setattr(module.os, "sched_getaffinity", lambda _: {0, 1}, raising=False)
    with pytest.raises(RuntimeError, match="same Slurm job"):
        module.restore(os.getpid())


def test_theory_submission_does_not_pin_spawn_parent():
    script = (Path(__file__).parents[1] / "submit_theory.sh").read_text()
    assert "export OMP_PROC_BIND=FALSE" in script
    assert "unset OMP_PLACES" in script
    assert "export OMP_PROC_BIND=spread" not in script
