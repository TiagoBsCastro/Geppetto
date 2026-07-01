"""Run and validate a PINOCCHIO reader format/splitting matrix.

This script is intentionally outside the differentiable GEPPETTO core. It
creates four PINOCCHIO runs in a scratch directory:

- ASCII, one output file
- ASCII, four output files
- binary, one output file
- binary, four output files

It then reads the generated snapshot catalogues and PLC catalogues through
``geppetto.io`` and verifies that all four configurations contain equivalent
physical data.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from geppetto.io import read_pinocchio_lightcone_catalog, read_pinocchio_snapshot_catalog

DEFAULT_MATRIX_DIR = Path("/tmp/geppetto-pinocchio-reader-validation")
DEFAULT_PINOCCHIO_DIR = Path("/home/tcastro/CodexSessions/Pinocchio-local")
DEFAULT_MPIRUN = Path("/home/tcastro/miniforge3/envs/geppetto-dev/bin/mpirun")
N_MPI_RANKS = 4


@dataclass(frozen=True)
class MatrixCase:
    name: str
    ascii_output: bool
    num_files: int

    @property
    def run_flag(self) -> str:
        return f"reader_{self.name}"


CASES = (
    MatrixCase("ascii_1", ascii_output=True, num_files=1),
    MatrixCase("ascii_4", ascii_output=True, num_files=4),
    MatrixCase("binary_1", ascii_output=False, num_files=1),
    MatrixCase("binary_4", ascii_output=False, num_files=4),
)


def prepare_matrix(
    matrix_dir: Path = DEFAULT_MATRIX_DIR,
    pinocchio_dir: Path = DEFAULT_PINOCCHIO_DIR,
    *,
    overwrite: bool = False,
    starting_z_for_plc: float = 0.1,
) -> None:
    """Prepare four isolated PINOCCHIO example directories under ``matrix_dir``."""

    example_dir = pinocchio_dir / "example"
    if not example_dir.exists():
        raise FileNotFoundError(f"PINOCCHIO example directory not found: {example_dir}")

    matrix_dir.mkdir(parents=True, exist_ok=True)
    for case in CASES:
        run_dir = matrix_dir / case.name
        if run_dir.exists():
            if not overwrite:
                continue
            shutil.rmtree(run_dir)
        shutil.copytree(example_dir, run_dir, ignore=shutil.ignore_patterns("*.png", "run.log"))
        _patch_parameter_file(run_dir / "parameter_file", case, starting_z_for_plc)


def run_matrix(
    matrix_dir: Path = DEFAULT_MATRIX_DIR,
    pinocchio_dir: Path = DEFAULT_PINOCCHIO_DIR,
    mpirun: Path = DEFAULT_MPIRUN,
    *,
    force: bool = False,
) -> None:
    """Run missing matrix cases with four MPI ranks."""

    executable = pinocchio_dir / "src" / "pinocchio.x"
    if not executable.exists():
        raise FileNotFoundError(f"PINOCCHIO executable not found: {executable}")
    if not mpirun.exists():
        raise FileNotFoundError(f"MPI launcher not found: {mpirun}")

    env = os.environ.copy()
    env["PATH"] = f"{mpirun.parent}:{env.get('PATH', '')}"
    for case in CASES:
        run_dir = matrix_dir / case.name
        if not run_dir.exists():
            raise FileNotFoundError(f"Matrix case directory is missing: {run_dir}")
        if not force and _case_outputs_exist(run_dir, case):
            continue

        log_path = run_dir / f"run_{case.name}.log"
        command = [
            str(mpirun),
            "-np",
            str(N_MPI_RANKS),
            str(executable),
            "parameter_file",
        ]
        with log_path.open("w", encoding="utf-8") as log_file:
            subprocess.run(
                command,
                cwd=run_dir,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=True,
            )


def validate_matrix(matrix_dir: Path = DEFAULT_MATRIX_DIR) -> dict[str, int]:
    """Validate reader equivalence across all four matrix cases."""

    snapshots = {case.name: _read_snapshot_case(matrix_dir / case.name, case) for case in CASES}
    plcs = {case.name: _read_plc_case(matrix_dir / case.name, case) for case in CASES}

    _validate_nonempty(snapshots, "snapshot")
    _validate_nonempty(plcs, "PLC")

    _compare_snapshots(snapshots["ascii_1"], snapshots["ascii_4"], strict=False)
    _compare_snapshots(snapshots["binary_1"], snapshots["binary_4"], strict=True)
    _compare_snapshots(snapshots["ascii_1"], snapshots["binary_1"], strict=False)
    _compare_snapshots(snapshots["ascii_4"], snapshots["binary_4"], strict=False)

    _compare_plcs(plcs["ascii_1"], plcs["ascii_4"], strict=False)
    _compare_plcs(plcs["binary_1"], plcs["binary_4"], strict=True)
    _compare_plcs(plcs["ascii_1"], plcs["binary_1"], strict=False)
    _compare_plcs(plcs["ascii_4"], plcs["binary_4"], strict=False)

    for case in CASES:
        snapshots[case.name].to_halo_catalog(position="final")
        plcs[case.name].to_lightcone_catalog(redshift="true")

    return {
        "snapshot_rows": len(snapshots["ascii_1"]),
        "plc_rows": len(plcs["ascii_1"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, default=DEFAULT_MATRIX_DIR)
    parser.add_argument("--pinocchio-dir", type=Path, default=DEFAULT_PINOCCHIO_DIR)
    parser.add_argument("--mpirun", type=Path, default=DEFAULT_MPIRUN)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--all", action="store_true", help="prepare, run, and validate")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--force-run", action="store_true")
    parser.add_argument("--starting-z-for-plc", type=float, default=0.1)
    args = parser.parse_args()

    if args.all or args.prepare:
        prepare_matrix(
            args.matrix_dir,
            args.pinocchio_dir,
            overwrite=args.overwrite,
            starting_z_for_plc=args.starting_z_for_plc,
        )
    if args.all or args.run:
        run_matrix(args.matrix_dir, args.pinocchio_dir, args.mpirun, force=args.force_run)
    if args.all or args.validate:
        summary = validate_matrix(args.matrix_dir)
        print(
            "PINOCCHIO reader matrix validated: "
            f"{summary['snapshot_rows']} snapshot rows, {summary['plc_rows']} PLC rows"
        )


def _patch_parameter_file(path: Path, case: MatrixCase, starting_z_for_plc: float) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    patched: list[str] = []
    seen_catalog_in_ascii = False
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("RunFlag"):
            patched.append(f"RunFlag                {case.run_flag}")
        elif stripped.startswith("CatalogInAscii") or stripped.startswith("% CatalogInAscii"):
            seen_catalog_in_ascii = True
            if case.ascii_output:
                patched.append("CatalogInAscii")
            else:
                patched.append("% CatalogInAscii")
        elif stripped.startswith("NumFiles"):
            patched.append(f"NumFiles               {case.num_files}")
        elif stripped.startswith("MinHaloMass"):
            patched.append("MinHaloMass            1")
        elif stripped.startswith("StartingzForPLC"):
            patched.append(f"StartingzForPLC        {starting_z_for_plc:.6g}")
        elif stripped.startswith("HubbleTableFile"):
            patched.append(f"% {line}")
        else:
            patched.append(line)

    if case.ascii_output and not seen_catalog_in_ascii:
        patched.append("CatalogInAscii")
    path.write_text("\n".join(patched) + "\n", encoding="utf-8")


def _case_outputs_exist(run_dir: Path, case: MatrixCase) -> bool:
    snapshot = run_dir / f"pinocchio.0.0000.{case.run_flag}.catalog.out"
    plc = run_dir / f"pinocchio.{case.run_flag}.plc.out"
    return _base_or_split_exists(snapshot) and _base_or_split_exists(plc)


def _base_or_split_exists(path: Path) -> bool:
    return path.exists() or Path(f"{path}.0").exists()


def _read_snapshot_case(run_dir: Path, case: MatrixCase):
    path = run_dir / f"pinocchio.0.0000.{case.run_flag}.catalog.out"
    return read_pinocchio_snapshot_catalog(path)


def _read_plc_case(run_dir: Path, case: MatrixCase):
    path = run_dir / f"pinocchio.{case.run_flag}.plc.out"
    return read_pinocchio_lightcone_catalog(path)


def _validate_nonempty(catalogs: dict[str, object], label: str) -> None:
    for name, catalog in catalogs.items():
        if len(catalog) == 0:
            raise AssertionError(f"{label} catalog is empty for case {name}")


def _compare_snapshots(left, right, *, strict: bool) -> None:
    li = _sort_indices(left.group_ids)
    ri = _sort_indices(right.group_ids)
    np.testing.assert_array_equal(left.group_ids[li], right.group_ids[ri])
    np.testing.assert_array_equal(left.n_particles[li], right.n_particles[ri])
    np.testing.assert_allclose(left.redshift, right.redshift, rtol=0.0, atol=1.0e-12)
    _assert_float_close(left.masses_msun_h[li], right.masses_msun_h[ri], strict=strict)
    _assert_float_close(
        left.initial_positions_mpc_h[li],
        right.initial_positions_mpc_h[ri],
        strict=strict,
        loose_atol=6.0e-3,
    )
    _assert_float_close(
        left.final_positions_mpc_h[li],
        right.final_positions_mpc_h[ri],
        strict=strict,
        loose_atol=6.0e-3,
    )
    _assert_float_close(
        left.velocities_km_s[li],
        right.velocities_km_s[ri],
        strict=strict,
        loose_atol=6.0e-3,
    )


def _compare_plcs(left, right, *, strict: bool) -> None:
    li = _sort_indices(left.group_ids)
    ri = _sort_indices(right.group_ids)
    np.testing.assert_array_equal(left.group_ids[li], right.group_ids[ri])
    for field in (
        "true_redshift",
        "positions_mpc_h",
        "velocities_km_s",
        "masses_msun_h",
        "theta_deg",
        "phi_deg",
        "los_velocity_km_s",
        "observed_redshift",
    ):
        _assert_float_close(getattr(left, field)[li], getattr(right, field)[ri], strict=strict)
    _assert_float_close(left.chi_mpc_h[li], right.chi_mpc_h[ri], strict=strict)
    _assert_float_close(left.unit_vectors[li], right.unit_vectors[ri], strict=strict)


def _sort_indices(values: np.ndarray) -> np.ndarray:
    return np.argsort(values, kind="stable")


def _assert_float_close(
    left: np.ndarray, right: np.ndarray, *, strict: bool, loose_atol: float = 1.0e-4
) -> None:
    if strict:
        np.testing.assert_allclose(left, right, rtol=1.0e-7, atol=1.0e-7)
    else:
        np.testing.assert_allclose(left, right, rtol=5.0e-5, atol=loose_atol)


if __name__ == "__main__":
    main()
