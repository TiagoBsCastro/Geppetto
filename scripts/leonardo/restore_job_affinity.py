#!/usr/bin/env python3
"""Restore an owned process tree to this Slurm step's allowed CPU set.

Run in an overlapping full-allocation step with --cpu-bind=none. This repairs
workers spawned after an OpenMP library pinned the parent to one CPU. It
never selects CPUs outside the invoking step's affinity or changes another
user's processes. Does not terminate or restart any work.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def contains_job(cgroup: str, job_id: str) -> bool:
    """Match a Slurm job path component, not a numeric job-ID prefix."""

    names = {f"job_{job_id}", f"job_{job_id}.scope"}
    return any(names.intersection(line.split(":", 2)[-1].split("/")) for line in cgroup.splitlines())


def restore(root_pid: int) -> tuple[int, int]:
    """Restore every extant thread in an owned root and its descendants."""

    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("run this utility inside the target Slurm allocation")
    allowed = os.sched_getaffinity(0)
    if len(allowed) < 2:
        raise RuntimeError("repair step must have the full CPU allocation, with --cpu-bind=none")
    root = Path(f"/proc/{root_pid}")
    if root.stat().st_uid != os.getuid():
        raise PermissionError("root process must belong to the invoking user")
    own_cgroup = Path("/proc/self/cgroup").read_text()
    root_cgroup = (root / "cgroup").read_text()
    job_id = os.environ["SLURM_JOB_ID"]
    if not contains_job(own_cgroup, job_id) or not contains_job(root_cgroup, job_id):
        raise RuntimeError("repair and target processes must belong to the same Slurm job")
    parents = {}
    for process in Path("/proc").glob("[0-9]*"):
        try:
            if process.stat().st_uid != os.getuid():
                continue
            status = dict(
                line.split(":", 1) for line in (process / "status").read_text().splitlines()
            )
            parents[int(process.name)] = int(status["PPid"])
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    descendants = {root_pid}
    while True:
        found = {pid for pid, parent in parents.items() if parent in descendants}
        if found <= descendants:
            break
        descendants.update(found)
    count = 0
    # Set the parent first so subsequently spawned workers inherit this mask.
    for pid in [root_pid, *sorted(descendants - {root_pid})]:
        for thread in Path(f"/proc/{pid}/task").glob("[0-9]*"):
            try:
                os.sched_setaffinity(int(thread.name), allowed)
                count += 1
            except ProcessLookupError:
                continue
    return len(descendants), count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-pid", type=int, required=True)
    args = parser.parse_args()
    processes, threads = restore(args.root_pid)
    print(
        f"Restored {processes} processes / {threads} threads to {len(os.sched_getaffinity(0))} CPUs"
    )
