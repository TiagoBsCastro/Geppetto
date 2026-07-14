"""Run a PINOCCHIO-to-NFW map calibration pipeline on mass-map segments.

This diagnostic script reads a PINOCCHIO parameter file, mass-sheet table,
on-the-fly HEALPix mass-map FITS files, and a PLC halo catalogue. It can run on
one selected segment or on all existing segments discovered from a glob. For
each segment it writes a compressed NPZ containing the projected NFW one-halo
mass divided by the PINOCCHIO particle mass. Array rows follow the compact
pixel domain of the corresponding ``*.massmap.segXXX.fits`` file.

The NFW map is intended for calibrating a concentration--mass relation against
a theoretical prediction while preserving PINOCCHIO's segment bounds and
compact pixel ordering. In derivative modes the script also saves map-level
derivatives with respect to concentration amplitude, mass slope, and redshift
slope. HEALPix stencil construction is fixed geometry and is not differentiated.
In all-segments mode the output is one lean NPZ per input segment plus a small
CSV provenance manifest. Pixel indices and HEALPix metadata remain in the
original PINOCCHIO FITS files and are not duplicated. No global light-cone map
is merged in this script.

Precision policy
----------------
This production calibration script defaults to JAX x64 because PINOCCHIO
readers preserve float64 inputs and small-angle HEALPix geometry is
precision-sensitive. Memory-constrained runs can opt into float32 with
``--jax-precision float32``.

Coordinate-basis warning
------------------------
The script assumes that the PLC halo catalogue directions and the PINOCCHIO
mass-map pixels are expressed in the same internal PLC basis. This should be
true when both files come from the same PINOCCHIO PLC run. If the halo catalogue
has been converted to an external sky basis, rotate it back into the PINOCCHIO
PLC basis before binning.

Example
-------
python examples/paint_halo_particles_for_pinocchio_segment.py \\
  --params path/to/parameter_file.params \\
  --sheets path/to/pinocchio.RUN.sheets.out \\
  --mass-map path/to/pinocchio.RUN.massmap.seg000.fits \\
  --plc-catalog path/to/pinocchio.RUN.plc.out \\
  --sheet-index 0 \\
  --output path/to/halo_particles.seg000.npz \\
  --nfw-overdensity 200 \\
  --nfw-reference-density critical

The default mode paints the NFW map. Use ``--mode derivatives`` to also save
map-level derivatives, ``--mode profile`` to print timings, or
``--mode derivatives-profile`` to do both.

Paint only:

  python examples/paint_halo_particles_for_pinocchio_segment.py ... --mode paint

Paint with map-level concentration derivatives:

  python examples/paint_halo_particles_for_pinocchio_segment.py ... --mode derivatives

Paint all discovered mass-map segments:

  python examples/paint_halo_particles_for_pinocchio_segment.py \\
    --params path/to/parameter_file.params \\
    --sheets path/to/pinocchio.RUN.sheets.out \\
    --plc-catalog path/to/pinocchio.RUN.plc.out \\
    --mass-map-glob "path/to/pinocchio.RUN.massmap.seg*.fits" \\
    --output-dir path/to/painted_segments \\
    --nfw-overdensity 200 \\
    --nfw-reference-density critical \\
    --mode derivatives
"""

from __future__ import annotations

# ruff: noqa: E402, I001

import argparse
import csv
import glob
import os
import re
import subprocess
import sys
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Literal


def _preparse_jax_precision(argv: list[str] | None = None) -> str:
    """Return the requested JAX precision before importing JAX-heavy modules."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--jax-precision",
        choices=("float64", "float32"),
        default="float64",
    )
    args, _ = parser.parse_known_args(argv)
    return str(args.jax_precision)


_CONFIGURED_JAX_PRECISION = _preparse_jax_precision()

from jax import config as jax_config

jax_config.update("jax_enable_x64", _CONFIGURED_JAX_PRECISION == "float64")

import jax
import jax.numpy as jnp
import numpy as np

from geppetto._sparse_jit import (
    paint_nfw_particle_count_map_and_concentration_jvps_jit,
    paint_nfw_particle_count_map_sparse_jit,
)
from geppetto import (
    ConcentrationParams,
    NFWProfileParams,
    paint_lightcone_particle_count_map,
)
from geppetto.catalog import LightconeHaloCatalog, LightconeSparseStencil
from geppetto.io import (
    PinocchioMassMap,
    PinocchioRunMetadata,
    build_lightcone_sparse_stencil_bruteforce,
    healpix_pixel_area_sr,
    healpix_pixel_unit_vectors,
    read_pinocchio_hubble_table,
    read_pinocchio_lightcone_catalog,
    read_pinocchio_lightcone_light_catalog,
    read_pinocchio_mass_map_fits,
    read_pinocchio_mass_sheets,
    read_pinocchio_parameter_file,
)
from geppetto.profiles import nfw_scale_radius_and_density

# Kept as a module attribute for regression tests proving the default sparse
# calibration path never calls the dense validation builder.
_BRUTE_FORCE_STENCIL_BUILDER_REGRESSION_SENTINEL = build_lightcone_sparse_stencil_bruteforce
_SEGMENT_RE = re.compile(r"seg(\d+)")
_PIXEL_INDEX_DENSE_MAX_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True)
class MpiContext:
    """Minimal MPI runtime state for optional rank-per-PLC-part painting."""

    enabled: bool = False
    comm: Any | None = None
    rank: int = 0
    size: int = 1
    sum_op: Any | None = None

    @property
    def is_root(self) -> bool:
        return self.rank == 0


@dataclass(frozen=True)
class ResolvedHaloMassDefinition:
    """Spherical-overdensity definition resolved for one map segment."""

    mode: Literal["constant", "per_segment", "bryan_norman"]
    profile_mode: Literal["constant", "bryan_norman"]
    reference_density: Literal["critical", "mean"]
    profile_overdensity: float
    reported_overdensity: float | None
    source: Path | None = None

    @property
    def label(self) -> str:
        """Return a compact manifest label for this mass interpretation."""

        suffix = "c" if self.reference_density == "critical" else "m"
        if self.mode == "bryan_norman":
            return f"virial_bn98{suffix}"
        assert self.reported_overdensity is not None
        return f"{self.reported_overdensity:g}{suffix}"


@dataclass(frozen=True)
class HaloMassDefinition:
    """Host-side mass-definition configuration for the segment workflow."""

    mode: Literal["constant", "per_segment", "bryan_norman"]
    reference_density: Literal["critical", "mean"]
    constant_overdensity: float | None = None
    per_segment_overdensity: np.ndarray | None = None
    source: Path | None = None

    def resolve(self, segment_index: int) -> ResolvedHaloMassDefinition:
        """Resolve this configuration for one sheet-row index."""

        if self.mode == "bryan_norman":
            return ResolvedHaloMassDefinition(
                mode=self.mode,
                profile_mode="bryan_norman",
                reference_density=self.reference_density,
                profile_overdensity=200.0,
                reported_overdensity=None,
            )
        if self.mode == "constant":
            assert self.constant_overdensity is not None
            value = float(self.constant_overdensity)
        else:
            assert self.per_segment_overdensity is not None
            if segment_index < 0 or segment_index >= self.per_segment_overdensity.size:
                raise ValueError(
                    f"segment index {segment_index} has no per-segment overdensity value"
                )
            value = float(self.per_segment_overdensity[segment_index])
        return ResolvedHaloMassDefinition(
            mode=self.mode,
            profile_mode="constant",
            reference_density=self.reference_density,
            profile_overdensity=value,
            reported_overdensity=value,
            source=self.source,
        )


@dataclass(frozen=True)
class ExecutionProvenance:
    """Execution metadata recorded once in every manifest row."""

    mpi_rank_count: int
    segment_worker_count: int
    git_commit: str


def default_halo_mass_definition() -> HaloMassDefinition:
    """Return the legacy 200c interpretation for direct Python callers."""

    return HaloMassDefinition(
        mode="constant",
        reference_density="critical",
        constant_overdensity=200.0,
    )


@dataclass
class CalibrationSegmentResult:
    """Computed segment payload, before any output files are written."""

    segment_index: int
    mass_map_path: Path
    output_npz: Path
    bounds: dict[str, float]
    inclusive_upper: bool
    nfw_diagnostics: dict[str, bool | float | int | str | np.ndarray]
    profile_stencil_diagnostics: StencilBuildDiagnostics | None = None


@dataclass(frozen=True)
class ComputedCalibrationSegment:
    """One completed rank-local segment and its worker wall time."""

    result: CalibrationSegmentResult
    compute_seconds: float


@dataclass(frozen=True)
class SegmentExecutionTiming:
    """Rank-local timings for one segment in the bounded MPI pipeline."""

    compute_seconds: float
    result_wait_seconds: float
    reduction_seconds: float
    stencil_diagnostics: StencilBuildDiagnostics | None = None

    def as_array(self) -> np.ndarray:
        stencil = self.stencil_diagnostics or StencilBuildDiagnostics()
        return np.asarray(
            (
                self.compute_seconds,
                self.result_wait_seconds,
                self.reduction_seconds,
                stencil.elapsed_seconds,
                stencil.query_disc_seconds,
                stencil.compact_lookup_seconds,
                stencil.pix2vec_filter_seconds,
                stencil.concatenate_seconds,
                stencil.jax_transfer_seconds,
                stencil.n_halos,
                stencil.n_query_pixels_total,
                stencil.n_inside_domain_total,
                stencil.n_kept_pairs_total,
                stencil.n_subpixel_radius_halos,
            ),
            dtype=np.float64,
        )


@dataclass
class StencilBuildDiagnostics:
    """Host-side counters for HEALPix sparse-stencil construction."""

    n_halos: int = 0
    n_halos_with_query_pixels: int = 0
    n_halos_with_inside_pixels: int = 0
    n_halos_with_kept_pairs: int = 0
    n_query_pixels_total: int = 0
    n_inside_domain_total: int = 0
    n_kept_pairs_total: int = 0
    n_subpixel_radius_halos: int = 0
    elapsed_seconds: float = 0.0
    query_disc_seconds: float = 0.0
    compact_lookup_seconds: float = 0.0
    pix2vec_filter_seconds: float = 0.0
    concatenate_seconds: float = 0.0
    jax_transfer_seconds: float = 0.0
    halo_has_kept_pair: np.ndarray | None = None

    @property
    def residual_seconds(self) -> float:
        """Return stencil time not assigned to an explicitly timed phase."""

        measured = (
            self.query_disc_seconds
            + self.compact_lookup_seconds
            + self.pix2vec_filter_seconds
            + self.concatenate_seconds
            + self.jax_transfer_seconds
        )
        return max(0.0, self.elapsed_seconds - measured)

    @property
    def inside_over_query(self) -> float:
        if self.n_query_pixels_total == 0:
            return 0.0
        return self.n_inside_domain_total / self.n_query_pixels_total

    @property
    def kept_over_query(self) -> float:
        if self.n_query_pixels_total == 0:
            return 0.0
        return self.n_kept_pairs_total / self.n_query_pixels_total

    @property
    def kept_over_inside(self) -> float:
        if self.n_inside_domain_total == 0:
            return 0.0
        return self.n_kept_pairs_total / self.n_inside_domain_total


@dataclass
class StencilProfileRecorder:
    """Mutable side channel for profile-only stencil diagnostics."""

    diagnostics: StencilBuildDiagnostics | None = None


@contextmanager
def timed_stage(name: str, enabled: bool = True):
    """Print elapsed wall-clock time for a named stage when enabled."""

    if not enabled:
        yield
        return

    t0 = perf_counter()
    try:
        yield
    finally:
        dt = perf_counter() - t0
        print(f"[profile] {name:<40s} {dt:9.4f} s")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Paint a PINOCCHIO PLC halo catalogue into an NFW one-halo "
            "particle-count map on one compact mass-map segment."
        )
    )
    parser.add_argument("--params", type=Path, required=True, help="PINOCCHIO parameter file")
    parser.add_argument("--sheets", type=Path, required=True, help="PINOCCHIO *.sheets.out file")
    parser.add_argument(
        "--mass-map",
        type=Path,
        help="PINOCCHIO *.massmap.segXXX.fits file",
    )
    parser.add_argument("--plc-catalog", type=Path, required=True, help="PINOCCHIO PLC catalogue")
    parser.add_argument("--sheet-index", type=int, help="Mass-sheet row index")
    parser.add_argument("--output", type=Path, help="Output .npz file")
    parser.add_argument(
        "--mass-map-glob",
        help="Glob matching PINOCCHIO *.massmap.segXXX.fits files for all-segments mode",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for all-segments painted NFW NPZ outputs and manifest",
    )
    parser.add_argument(
        "--catalog-format",
        choices=("auto", "ascii", "binary"),
        default="auto",
        help="PLC catalogue format",
    )
    parser.add_argument(
        "--redshift-mode",
        choices=("true", "observed"),
        default="true",
        help="Halo redshift column to use",
    )
    parser.add_argument(
        "--bounds",
        choices=("z", "chi"),
        default="z",
        help="Segment-bound coordinate used to select halos",
    )
    parser.add_argument(
        "--hubble-table",
        type=Path,
        help="PINOCCHIO Hubble table required for --light-plc",
    )
    parser.add_argument(
        "--light-plc",
        action="store_true",
        help="Read a light PLC catalogue without Cartesian positions",
    )
    parser.add_argument(
        "--last-segment-inclusive",
        action="store_true",
        help=(
            "In single-segment mode, use an inclusive upper segment bound. "
            "All-segments mode determines inclusivity from the physical final sheet."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("paint", "derivatives", "profile", "derivatives-profile"),
        default="paint",
        help=(
            "Pipeline mode. 'paint' saves the NFW painted map. "
            "'derivatives' also saves map-level derivatives with respect to "
            "concentration--mass parameters. 'profile' prints timings. "
            "'derivatives-profile' computes derivatives and prints timings."
        ),
    )
    parser.add_argument(
        "--concentration-amplitude",
        type=float,
        default=5.71,
        help="Amplitude of the concentration--mass relation.",
    )
    parser.add_argument(
        "--concentration-mass-slope",
        type=float,
        default=-0.084,
        help="Mass slope of the concentration--mass relation.",
    )
    parser.add_argument(
        "--concentration-redshift-slope",
        type=float,
        default=-0.47,
        help="Redshift slope of the concentration--mass relation.",
    )
    parser.add_argument(
        "--concentration-mass-pivot",
        type=float,
        default=2.0e12,
        help="Mass pivot in Msun/h for the concentration--mass relation.",
    )
    mass_definition_group = parser.add_mutually_exclusive_group(required=True)
    mass_definition_group.add_argument(
        "--nfw-overdensity",
        type=float,
        help="Constant spherical overdensity Delta for the NFW halo mass.",
    )
    mass_definition_group.add_argument(
        "--nfw-virial-overdensity",
        action="store_true",
        help="Use the redshift-dependent Bryan--Norman virial overdensity.",
    )
    mass_definition_group.add_argument(
        "--nfw-overdensity-by-segment",
        type=Path,
        help="One-dimensional .npy array containing one Delta value per sheet row.",
    )
    parser.add_argument(
        "--nfw-reference-density",
        choices=("critical", "mean"),
        required=True,
        help="Reference density for the NFW spherical-overdensity mass.",
    )
    parser.add_argument(
        "--truncation-width-fraction",
        type=float,
        default=0.05,
        help="Smooth truncation-width fraction for the NFW profile.",
    )
    parser.add_argument(
        "--nfw-chunk-size",
        type=int,
        default=1024,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--nfw-taper-radius-factor",
        type=float,
        default=10.0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--nfw-dense-demo",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--profile-jax-repeat",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--jax-precision",
        choices=("float64", "float32"),
        default="float64",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--mpi-plc-parts",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--segment-workers",
        type=int,
        default=1,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def load_halo_mass_definition(
    args: argparse.Namespace,
    sheets: Any,
) -> HaloMassDefinition:
    """Load and validate the user-selected NFW spherical-overdensity definition."""

    reference_density = str(args.nfw_reference_density)
    if reference_density not in ("critical", "mean"):
        raise ValueError("--nfw-reference-density must be 'critical' or 'mean'")

    constant = args.nfw_overdensity
    use_virial = bool(args.nfw_virial_overdensity)
    source = args.nfw_overdensity_by_segment
    selected_modes = int(constant is not None) + int(use_virial) + int(source is not None)
    if selected_modes != 1:
        raise ValueError(
            "Select exactly one of --nfw-overdensity, --nfw-virial-overdensity, "
            "or --nfw-overdensity-by-segment"
        )

    if constant is not None:
        value = float(constant)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("--nfw-overdensity must be finite and positive")
        return HaloMassDefinition(
            mode="constant",
            reference_density=reference_density,
            constant_overdensity=value,
        )

    if use_virial:
        return HaloMassDefinition(
            mode="bryan_norman",
            reference_density=reference_density,
        )

    source = Path(source)
    if source.suffix.lower() != ".npy":
        raise ValueError("--nfw-overdensity-by-segment must point to a .npy file")
    try:
        values = np.load(source, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read per-segment overdensity array: {source}") from exc
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("per-segment overdensity array must be one-dimensional")
    if values.shape[0] != len(sheets):
        raise ValueError(
            "per-segment overdensity array length must match the sheet table: "
            f"got {values.shape[0]}, expected {len(sheets)}"
        )
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("per-segment overdensity values must be finite and positive")
    values.setflags(write=False)
    return HaloMassDefinition(
        mode="per_segment",
        reference_density=reference_density,
        per_segment_overdensity=values,
        source=source,
    )


def validate_segment_workflow_args(args: argparse.Namespace) -> str:
    """Return ``single`` or ``all`` after validating segment workflow arguments."""

    single_segment_args = (
        args.mass_map is not None,
        args.sheet_index is not None,
        args.output is not None,
    )
    all_segment_args = (
        args.mass_map_glob is not None,
        args.output_dir is not None,
    )

    if any(single_segment_args) and any(all_segment_args):
        raise ValueError(
            "Use either single-segment inputs "
            "(--mass-map, --sheet-index, --output) or all-segments inputs "
            "(--mass-map-glob, --output-dir), not both."
        )
    if all(single_segment_args) and not any(all_segment_args):
        return "single"
    if all(all_segment_args) and not any(single_segment_args):
        if bool(args.last_segment_inclusive):
            raise ValueError(
                "--last-segment-inclusive is only valid in single-segment mode"
            )
        return "all"
    raise ValueError(
        "Provide either --mass-map, --sheet-index, --output "
        "or --mass-map-glob, --output-dir."
    )


def _mpi_environment_size_hint() -> int:
    """Return a best-effort MPI world-size hint without importing mpi4py."""

    for name in (
        "OMPI_COMM_WORLD_SIZE",
        "PMI_SIZE",
        "PMIX_SIZE",
        "MPI_LOCALNRANKS",
        "MV2_COMM_WORLD_SIZE",
    ):
        value = os.environ.get(name)
        if value is None:
            continue
        try:
            size = int(value)
        except ValueError:
            continue
        if size > 1:
            return size
    return 1


def initialize_mpi_context(requested: bool) -> MpiContext:
    """Initialize optional MPI state only when requested or launched under MPI."""

    if not requested and _mpi_environment_size_hint() <= 1:
        return MpiContext()

    try:
        from mpi4py import MPI
    except ImportError as exc:
        raise RuntimeError("MPI execution requires mpi4py; install geppetto[mpi]") from exc

    comm = MPI.COMM_WORLD
    size = int(comm.Get_size())
    rank = int(comm.Get_rank())
    return MpiContext(
        enabled=bool(requested),
        comm=comm,
        rank=rank,
        size=size,
        sum_op=MPI.SUM,
    )


def discover_plc_catalog_parts(path: Path) -> list[Path]:
    """Return contiguous split PLC part files named ``path.0``, ``path.1``, ..."""

    base = Path(path)
    pattern = re.compile(rf"{re.escape(base.name)}\.(\d+)$")
    parts: list[tuple[int, Path]] = []
    for candidate in base.parent.glob(f"{base.name}.*"):
        match = pattern.fullmatch(candidate.name)
        if match is not None:
            parts.append((int(match.group(1)), candidate))

    if not parts:
        raise ValueError(f"No split PLC part files found for MPI mode: {base}.0, {base}.1, ...")

    parts = sorted(parts, key=lambda item: item[0])
    indices = [index for index, _ in parts]
    expected = list(range(len(parts)))
    if indices != expected:
        raise ValueError(
            "Split PLC part files must be contiguous from 0: "
            f"found {indices}, expected {expected}"
        )
    return [path for _, path in parts]


def validate_mpi_plc_part_count(parts: list[Path], mpi_size: int) -> None:
    """Require exactly one MPI rank per split PLC part."""

    n_parts = len(parts)
    if mpi_size != n_parts:
        raise ValueError(
            "MPI PLC part mode requires one rank per PLC part: "
            f"Nmpi={mpi_size}, Nparts={n_parts}"
        )


def plc_catalog_path_for_rank(plc_catalog: Path, mpi_context: MpiContext) -> Path:
    """Return the catalogue path this rank should read."""

    if not mpi_context.enabled:
        return Path(plc_catalog)
    parts = discover_plc_catalog_parts(Path(plc_catalog))
    validate_mpi_plc_part_count(parts, mpi_context.size)
    return parts[mpi_context.rank]


def validate_mpi_workflow_args(
    args: argparse.Namespace,
    *,
    workflow: str,
    mpi_context: MpiContext,
) -> None:
    """Validate MPI-specific workflow constraints."""

    segment_workers = int(getattr(args, "segment_workers", 1))
    if segment_workers < 1:
        raise ValueError("--segment-workers must be at least 1")
    if mpi_context.size > 1 and not bool(getattr(args, "mpi_plc_parts", False)):
        raise ValueError("MPI world size > 1 requires --mpi-plc-parts")
    if mpi_context.enabled:
        parts = discover_plc_catalog_parts(Path(args.plc_catalog))
        validate_mpi_plc_part_count(parts, mpi_context.size)


def warn_if_float32_precision(args: argparse.Namespace, mpi_context: MpiContext) -> None:
    """Warn root users when they trade calibration accuracy for memory."""

    if args.jax_precision == "float32" and mpi_context.is_root:
        print(
            "Warning: --jax-precision float32 saves memory but can reduce "
            "small-angle NFW/stencil geometry accuracy."
        )


def resolve_git_commit() -> str:
    """Return execution commit provenance without failing an installed run."""

    configured = os.environ.get("GEPPETTO_GIT_COMMIT")
    if configured:
        return configured
    repository = Path(__file__).resolve().parents[1]
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10.0,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    commit = result.stdout.strip()
    return commit or "unknown"


def execution_provenance(
    args: argparse.Namespace,
    mpi_context: MpiContext,
) -> ExecutionProvenance:
    """Build root-written execution provenance for manifest rows."""

    return ExecutionProvenance(
        mpi_rank_count=mpi_context.size if mpi_context.enabled else 1,
        segment_worker_count=int(getattr(args, "segment_workers", 1)),
        git_commit=resolve_git_commit() if mpi_context.is_root else "",
    )


def abort_mpi_job(mpi_context: MpiContext, exc: BaseException) -> None:
    """Report a fatal rank-local exception and abort the distributed job."""

    print(
        f"Fatal MPI workflow error on rank {mpi_context.rank}/{mpi_context.size}: {exc}",
        file=sys.stderr,
        flush=True,
    )
    traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
    sys.stderr.flush()
    if mpi_context.comm is None:
        raise RuntimeError("MPI context is enabled but has no communicator") from exc
    mpi_context.comm.Abort(1)
    raise RuntimeError("MPI communicator returned after Abort") from exc


def parse_segment_index_from_mass_map_path(path: Path) -> int:
    """Return segment index parsed from a PINOCCHIO mass-map segment filename."""

    match = _SEGMENT_RE.search(path.name)
    if match is None:
        raise ValueError(f"Cannot parse segment index from mass-map filename: {path}")
    return int(match.group(1))


def discover_mass_map_segments(pattern: str) -> list[tuple[int, Path]]:
    """Return sorted ``(segment_index, path)`` pairs matching a mass-map glob."""

    paths = [Path(path) for path in glob.glob(pattern)]
    if not paths:
        raise ValueError(f"No mass-map segments match glob: {pattern}")

    segments: list[tuple[int, Path]] = []
    seen: dict[int, Path] = {}
    for path in paths:
        segment_index = parse_segment_index_from_mass_map_path(path)
        if segment_index in seen:
            raise ValueError(
                "Duplicate mass-map segment index "
                f"{segment_index}: {seen[segment_index]} and {path}"
            )
        seen[segment_index] = path
        segments.append((segment_index, path))
    return sorted(segments, key=lambda item: item[0])


def validate_mass_map_segment_batch(
    segments: list[tuple[int, Path]],
    sheets: Any,
) -> None:
    """Validate that a discovered batch is consecutive and indexes the sheets."""

    indices = [segment_index for segment_index, _ in segments]
    expected = list(range(indices[0], indices[-1] + 1))
    if indices != expected:
        raise ValueError(
            "Mass-map segment indices must form one contiguous batch: "
            f"found {indices}, expected {expected}"
        )
    n_sheet = len(sheets)
    invalid = [index for index in indices if index < 0 or index >= n_sheet]
    if invalid:
        raise ValueError(
            "Mass-map segment indices are outside the sheet table: "
            f"invalid {invalid}, valid range [0, {n_sheet})"
        )


def segment_output_path(output_dir: Path, segment_index: int) -> Path:
    """Return the compressed NPZ output path for one segment index."""

    tag = f"seg{segment_index:03d}"
    return output_dir / f"painted_nfw.{tag}.npz"


def load_lightcone_catalog(
    args: argparse.Namespace,
    plc_catalog_path: Path | None = None,
) -> LightconeHaloCatalog:
    """Load a full or light PINOCCHIO PLC catalogue as a GEPPETTO catalogue."""

    source = Path(args.plc_catalog) if plc_catalog_path is None else Path(plc_catalog_path)
    if args.light_plc:
        if args.hubble_table is None:
            raise ValueError("--hubble-table is required when --light-plc is used")
        raw = read_pinocchio_lightcone_light_catalog(
            source,
            format=args.catalog_format,
        )
        distance_interpolator = read_pinocchio_hubble_table(args.hubble_table)
        return raw.to_lightcone_catalog(
            distance_interpolator,
            redshift=args.redshift_mode,
        )

    raw = read_pinocchio_lightcone_catalog(source, format=args.catalog_format)
    return raw.to_lightcone_catalog(redshift=args.redshift_mode)


def load_rank_local_lightcone_catalog(
    args: argparse.Namespace,
    mpi_context: MpiContext,
) -> LightconeHaloCatalog:
    """Load all PLC parts in serial mode or this rank's one PLC part in MPI mode."""

    return load_lightcone_catalog(
        args,
        plc_catalog_path_for_rank(Path(args.plc_catalog), mpi_context),
    )


def segment_bounds(sheets: Any, sheet_index: int) -> dict[str, float]:
    """Return sorted redshift, scale-factor, and distance bounds for one sheet."""

    n_sheet = len(sheets)
    if sheet_index < 0 or sheet_index >= n_sheet:
        raise ValueError(f"sheet_index {sheet_index} is outside [0, {n_sheet})")

    z_lo = min(float(sheets.z_lo[sheet_index]), float(sheets.z_hi[sheet_index]))
    z_hi = max(float(sheets.z_lo[sheet_index]), float(sheets.z_hi[sheet_index]))
    chi_lo = min(
        float(sheets.chi_lo_mpc_h[sheet_index]),
        float(sheets.chi_hi_mpc_h[sheet_index]),
    )
    chi_hi = max(
        float(sheets.chi_lo_mpc_h[sheet_index]),
        float(sheets.chi_hi_mpc_h[sheet_index]),
    )
    return {
        "sheet_index": int(sheet_index),
        "z_lo": z_lo,
        "z_hi": z_hi,
        "a_lo": 1.0 / (1.0 + z_hi),
        "a_hi": 1.0 / (1.0 + z_lo),
        "chi_lo_mpc_h": chi_lo,
        "chi_hi_mpc_h": chi_hi,
    }


def validate_mass_map(mass_map: PinocchioMassMap) -> None:
    """Check the mass-map fields needed for compact RING pixel binning."""

    if mass_map.ordering.upper() != "RING":
        raise ValueError(f"Only RING mass maps are supported, got ORDERING={mass_map.ordering!r}")

    pixels = np.asarray(mass_map.pixel)
    temperature = np.asarray(mass_map.temperature)
    if pixels.ndim != 1:
        raise ValueError("mass_map.pixel must be one-dimensional")
    if temperature.ndim != 1:
        raise ValueError("mass_map.temperature must be one-dimensional")
    if pixels.shape != temperature.shape:
        raise ValueError("mass_map.pixel and mass_map.temperature must have the same length")


@dataclass(frozen=True)
class MassMapPixelIndex:
    """Order-preserving compact-row lookup for one PINOCCHIO mass-map segment."""

    backend: str
    dense_rows: np.ndarray | None = None
    sorted_pixels: np.ndarray | None = None
    sorted_rows: np.ndarray | None = None

    @classmethod
    def from_mass_map(
        cls,
        mass_map: PinocchioMassMap,
        *,
        max_dense_bytes: int = _PIXEL_INDEX_DENSE_MAX_BYTES,
    ) -> MassMapPixelIndex:
        """Build a reusable global-HEALPix-pixel to compact-row lookup."""

        validate_mass_map(mass_map)
        pixels = np.asarray(mass_map.pixel, dtype=np.int64)
        return cls.from_pixels(pixels, max_dense_bytes=max_dense_bytes)

    @classmethod
    def from_pixels(
        cls,
        pixels: np.ndarray,
        *,
        max_dense_bytes: int = _PIXEL_INDEX_DENSE_MAX_BYTES,
    ) -> MassMapPixelIndex:
        """Build a compact-row lookup from global HEALPix pixel numbers."""

        pixels = np.asarray(pixels, dtype=np.int64)
        if pixels.ndim != 1:
            raise ValueError("pixels must be one-dimensional")
        if max_dense_bytes < 0:
            raise ValueError("max_dense_bytes must be non-negative")
        if pixels.size == 0:
            return cls(backend="empty")
        if np.any(pixels < 0):
            raise ValueError("pixels must be non-negative")

        rows = np.arange(pixels.shape[0], dtype=np.int64)
        max_pixel = int(np.max(pixels))
        dense_bytes = (max_pixel + 1) * np.dtype(np.int64).itemsize
        if dense_bytes <= max_dense_bytes:
            dense_rows = np.full(max_pixel + 1, -1, dtype=np.int64)
            dense_rows[pixels] = rows
            return cls(backend="dense", dense_rows=dense_rows)

        order = np.argsort(pixels)
        return cls(
            backend="sorted",
            sorted_pixels=pixels[order],
            sorted_rows=rows[order],
        )

    def lookup(self, pixels: np.ndarray) -> np.ndarray:
        """Return compact rows for global HEALPix pixels, or ``-1`` if absent."""

        query = np.asarray(pixels, dtype=np.int64)
        flat_query = query.ravel()
        flat_rows = np.full(flat_query.shape, -1, dtype=np.int64)
        if flat_query.size == 0 or self.backend == "empty":
            return flat_rows.reshape(query.shape)

        nonnegative = flat_query >= 0
        if self.backend == "dense":
            if self.dense_rows is None:
                raise RuntimeError("dense pixel index is missing dense rows")
            in_range = nonnegative & (flat_query < self.dense_rows.shape[0])
            flat_rows[in_range] = self.dense_rows[flat_query[in_range]]
        elif self.backend == "sorted":
            if self.sorted_pixels is None or self.sorted_rows is None:
                raise RuntimeError("sorted pixel index is missing sorted arrays")
            valid_positions = np.nonzero(nonnegative)[0]
            valid_pixels = flat_query[valid_positions]
            insert = np.searchsorted(self.sorted_pixels, valid_pixels)
            inside = insert < self.sorted_pixels.shape[0]
            matched_positions = valid_positions[inside]
            matched_insert = insert[inside]
            matched_pixels = valid_pixels[inside]
            found = self.sorted_pixels[matched_insert] == matched_pixels
            flat_rows[matched_positions[found]] = self.sorted_rows[matched_insert[found]]
        else:
            raise RuntimeError(f"unknown pixel-index backend {self.backend!r}")
        return flat_rows.reshape(query.shape)


def validate_catalog_for_binning(catalog: LightconeHaloCatalog) -> None:
    """Check catalogue array shapes before segment selection and binning."""

    mass = np.asarray(catalog.mass)
    redshift = np.asarray(catalog.redshift)
    chi = np.asarray(catalog.chi)
    unit_vector = np.asarray(catalog.unit_vector)

    if mass.ndim != 1:
        raise ValueError("catalog.mass must have shape (n_halo,)")
    n_halo = int(mass.shape[0])
    if redshift.ndim != 1 or redshift.shape[0] != n_halo:
        raise ValueError("catalog.redshift must have shape matching catalog.mass")
    if chi.ndim != 1 or chi.shape[0] != n_halo:
        raise ValueError("catalog.chi must have shape matching catalog.mass")
    if unit_vector.ndim != 2 or unit_vector.shape != (n_halo, 3):
        raise ValueError("catalog.unit_vector must have shape (n_halo, 3)")
    if not np.all(np.isfinite(mass)):
        raise ValueError("catalog.mass values must be finite")
    if not np.all(np.isfinite(redshift)):
        raise ValueError("catalog.redshift values must be finite")
    if not np.all(np.isfinite(chi)):
        raise ValueError("catalog.chi values must be finite")
    if not np.all(np.isfinite(unit_vector)):
        raise ValueError("catalog.unit_vector values must be finite")


def select_segment_mask(
    catalog: LightconeHaloCatalog,
    bounds: dict[str, float],
    mode: str,
    inclusive_upper: bool,
) -> np.ndarray:
    """Select halos inside the requested segment bounds."""

    if mode == "z":
        values = np.asarray(catalog.redshift)
        lo = bounds["z_lo"]
        hi = bounds["z_hi"]
    elif mode == "chi":
        values = np.asarray(catalog.chi)
        lo = bounds["chi_lo_mpc_h"]
        hi = bounds["chi_hi_mpc_h"]
    else:
        raise ValueError("mode must be 'z' or 'chi'")

    if inclusive_upper:
        return (values >= lo) & (values <= hi)
    return (values >= lo) & (values < hi)


def selected_lightcone_catalog(catalog: LightconeHaloCatalog, mask: np.ndarray) -> LightconeHaloCatalog:
    """Return the segment-selected catalogue as JAX arrays for differentiable painters."""

    mask = np.asarray(mask, dtype=bool)
    return LightconeHaloCatalog(
        unit_vector=jnp.asarray(np.asarray(catalog.unit_vector)[mask]),
        chi=jnp.asarray(np.asarray(catalog.chi)[mask]),
        mass=jnp.asarray(np.asarray(catalog.mass)[mask]),
        redshift=jnp.asarray(np.asarray(catalog.redshift)[mask]),
    )


def nfw_stencil_rmax_mpc_h(
    catalog: LightconeHaloCatalog,
    metadata: PinocchioRunMetadata,
    concentration_params: ConcentrationParams,
    profile_params: NFWProfileParams,
    taper_radius_factor: float,
) -> np.ndarray:
    """Return fixed sparse-stencil NFW support radii in comoving ``Mpc/h``."""

    if taper_radius_factor < 0.0:
        raise ValueError("taper_radius_factor must be non-negative")

    r_delta, _, _, _ = nfw_scale_radius_and_density(
        catalog.mass,
        catalog.redshift,
        metadata.cosmology,
        concentration_params,
        profile_params,
    )
    r_delta_np = np.asarray(r_delta, dtype=np.float64)
    if not profile_params.smooth_truncation:
        return r_delta_np

    width = float(profile_params.truncation_width_fraction) * r_delta_np
    return r_delta_np + float(taper_radius_factor) * width


def _compression_factor(dense_pair_count: int, sparse_pair_count: int) -> float:
    if sparse_pair_count > 0:
        return float(dense_pair_count) / float(sparse_pair_count)
    if dense_pair_count > 0:
        return float("inf")
    return 1.0


def sparse_pair_bucket_size(n_pair: int) -> int:
    """Return the next power-of-two sparse pair count, preserving zero."""

    if n_pair < 0:
        raise ValueError("n_pair must be non-negative")
    if n_pair == 0:
        return 0
    return 1 << (n_pair - 1).bit_length()


def bucket_sparse_stencil_for_rank_catalog(
    stencil: LightconeSparseStencil,
    selected_mask: np.ndarray,
    rank_catalog_size: int,
    *,
    profile_diagnostics: StencilBuildDiagnostics | None = None,
) -> LightconeSparseStencil:
    """Remap selected-catalogue IDs and zero-pad one sparse JIT bucket.

    The returned halo IDs index the full rank-local catalogue, whose shape is
    constant across segments. Geometry and padding remain outside differentiable
    kernels.
    """

    bucket_start = perf_counter() if profile_diagnostics is not None else None
    if rank_catalog_size < 0:
        raise ValueError("rank_catalog_size must be non-negative")
    mask = np.asarray(selected_mask, dtype=bool)
    if mask.shape != (rank_catalog_size,):
        raise ValueError("selected_mask must match the rank-local catalogue length")

    pix_id = np.asarray(stencil.pix_id, dtype=np.int32)
    local_halo_id = np.asarray(stencil.halo_id, dtype=np.int64)
    r_perp = np.asarray(stencil.r_perp)
    n_pair = int(r_perp.shape[0])
    if pix_id.shape != (n_pair,) or local_halo_id.shape != (n_pair,):
        raise ValueError("sparse stencil fields must be one-dimensional and equal length")

    selected_halo_ids = np.flatnonzero(mask)
    if local_halo_id.size and (
        np.any(local_halo_id < 0) or np.any(local_halo_id >= selected_halo_ids.size)
    ):
        raise ValueError("stencil halo IDs do not index the selected catalogue")
    rank_halo_id = selected_halo_ids[local_halo_id].astype(np.int32, copy=False)

    if stencil.pair_weight is None:
        pair_weight = np.ones(n_pair, dtype=r_perp.dtype)
    else:
        pair_weight = np.asarray(stencil.pair_weight, dtype=r_perp.dtype)
        if pair_weight.shape != (n_pair,):
            raise ValueError("stencil pair weights must match the pair count")
    if not np.all(np.isfinite(pair_weight)) or np.any(pair_weight < 0.0):
        raise ValueError("stencil pair weights must be finite and non-negative")

    bucket_size = sparse_pair_bucket_size(n_pair)
    padded_pix_id = np.zeros(bucket_size, dtype=np.int32)
    padded_halo_id = np.zeros(bucket_size, dtype=np.int32)
    padded_r_perp = np.ones(bucket_size, dtype=r_perp.dtype)
    padded_pair_weight = np.zeros(bucket_size, dtype=r_perp.dtype)
    padded_pix_id[:n_pair] = pix_id
    padded_halo_id[:n_pair] = rank_halo_id
    padded_r_perp[:n_pair] = r_perp
    padded_pair_weight[:n_pair] = pair_weight

    transfer_start = perf_counter() if profile_diagnostics is not None else None
    bucketed_stencil = LightconeSparseStencil(
        pix_id=jnp.asarray(padded_pix_id),
        halo_id=jnp.asarray(padded_halo_id),
        r_perp=jnp.asarray(padded_r_perp),
        n_pix=stencil.n_pix,
        pair_weight=jnp.asarray(padded_pair_weight),
    )
    if profile_diagnostics is not None:
        bucketed_stencil.pix_id.block_until_ready()
        bucketed_stencil.halo_id.block_until_ready()
        bucketed_stencil.r_perp.block_until_ready()
        assert bucketed_stencil.pair_weight is not None
        bucketed_stencil.pair_weight.block_until_ready()
        assert transfer_start is not None
        profile_diagnostics.jax_transfer_seconds += perf_counter() - transfer_start
        assert bucket_start is not None
        profile_diagnostics.elapsed_seconds += perf_counter() - bucket_start
    return bucketed_stencil


def build_lightcone_sparse_stencil_for_mass_map_local(
    mass_map: PinocchioMassMap,
    catalog: LightconeHaloCatalog,
    rmax_mpc_h: np.ndarray,
    *,
    collect_diagnostics: bool = False,
    pixel_index: MassMapPixelIndex | None = None,
    profile_phases: bool = False,
) -> LightconeSparseStencil | tuple[LightconeSparseStencil, StencilBuildDiagnostics]:
    """Build a HEALPix-local sparse stencil on a compact PINOCCHIO map domain.

    The returned ``pix_id`` values are compact row indices into
    ``mass_map.pixel``, not global HEALPix pixel numbers. Geometry is fixed
    outside JAX; the differentiable sparse painter receives only the retained
    local halo-pixel pairs.
    """

    try:
        import healpy as hp
    except ImportError as exc:  # pragma: no cover - exercised only without io extra
        raise RuntimeError("local sparse NFW stencil construction requires healpy") from exc

    validate_mass_map(mass_map)
    validate_catalog_for_binning(catalog)
    build_start = perf_counter()

    pixels = np.asarray(mass_map.pixel, dtype=np.int64)
    n_pix = int(pixels.shape[0])
    if pixel_index is None:
        pixel_index = MassMapPixelIndex.from_pixels(pixels)

    halo_unit_vectors = np.asarray(catalog.unit_vector)
    if not np.issubdtype(halo_unit_vectors.dtype, np.floating):
        halo_unit_vectors = halo_unit_vectors.astype(np.float64)
    geometry_dtype = halo_unit_vectors.dtype
    halo_chi = np.asarray(catalog.chi, dtype=geometry_dtype)
    n_halo = int(halo_chi.shape[0])
    if np.any(halo_chi <= 0.0):
        raise ValueError("catalog.chi values must be positive for local stencil construction")

    rmax = np.asarray(rmax_mpc_h, dtype=np.float64)
    if rmax.shape != (n_halo,):
        raise ValueError("rmax_mpc_h must have shape (n_halo,)")
    if not np.all(np.isfinite(rmax)) or np.any(rmax < 0.0):
        raise ValueError("rmax_mpc_h values must be finite and non-negative")

    pix_id_chunks: list[np.ndarray] = []
    halo_id_chunks: list[np.ndarray] = []
    r_perp_chunks: list[np.ndarray] = []
    diagnostics = StencilBuildDiagnostics()
    halo_has_kept_pair = np.zeros(n_halo, dtype=bool)
    max_pixel_radius = float(hp.max_pixrad(mass_map.nside))

    for halo_id, (halo_vector, chi_i, rmax_i) in enumerate(
        zip(halo_unit_vectors, halo_chi, rmax, strict=True)
    ):
        diagnostics.n_halos += 1
        alpha_max = 2.0 * np.arcsin(min(1.0, float(rmax_i) / (2.0 * float(chi_i))))
        if alpha_max <= max_pixel_radius:
            diagnostics.n_subpixel_radius_halos += 1
        query_radius = min(float(np.pi), float(np.nextafter(alpha_max, np.inf)))

        phase_start = perf_counter() if profile_phases else None
        queried_pixels = np.asarray(
            hp.query_disc(
                mass_map.nside,
                halo_vector.astype(np.float64, copy=False),
                query_radius,
                inclusive=False,
                nest=False,
            ),
            dtype=np.int64,
        )
        if phase_start is not None:
            diagnostics.query_disc_seconds += perf_counter() - phase_start
        diagnostics.n_query_pixels_total += int(queried_pixels.size)
        if queried_pixels.size == 0:
            continue
        diagnostics.n_halos_with_query_pixels += 1

        phase_start = perf_counter() if profile_phases else None
        rows = pixel_index.lookup(queried_pixels)
        inside_domain = rows >= 0
        n_inside = int(np.count_nonzero(inside_domain))
        if phase_start is not None:
            diagnostics.compact_lookup_seconds += perf_counter() - phase_start
        diagnostics.n_inside_domain_total += n_inside
        if not np.any(inside_domain):
            continue
        diagnostics.n_halos_with_inside_pixels += 1

        phase_start = perf_counter() if profile_phases else None
        local_pixels = queried_pixels[inside_domain]
        local_rows = rows[inside_domain]
        x, y, z = hp.pix2vec(mass_map.nside, local_pixels, nest=False)
        pixel_vectors = np.stack([x, y, z], axis=-1).astype(geometry_dtype, copy=False)
        cosang = np.clip(
            pixel_vectors[:, 0] * halo_vector[0]
            + pixel_vectors[:, 1] * halo_vector[1]
            + pixel_vectors[:, 2] * halo_vector[2],
            -1.0,
            1.0,
        )
        chord = np.sqrt(np.maximum(2.0 * (1.0 - cosang), 0.0))
        r_perp = chi_i * chord
        keep = r_perp <= float(rmax_i)
        n_keep = int(np.count_nonzero(keep))
        if phase_start is not None:
            diagnostics.pix2vec_filter_seconds += perf_counter() - phase_start
        diagnostics.n_kept_pairs_total += n_keep
        if n_keep == 0:
            continue
        diagnostics.n_halos_with_kept_pairs += 1
        halo_has_kept_pair[halo_id] = True

        pix_id_chunks.append(local_rows[keep])
        halo_id_chunks.append(np.full(n_keep, halo_id, dtype=np.int64))
        r_perp_chunks.append(r_perp[keep])

    phase_start = perf_counter() if profile_phases else None
    if pix_id_chunks:
        pix_id = np.concatenate(pix_id_chunks)
        halo_id = np.concatenate(halo_id_chunks)
        r_perp = np.concatenate(r_perp_chunks)
    else:
        pix_id = np.empty((0,), dtype=np.int64)
        halo_id = np.empty((0,), dtype=np.int64)
        r_perp = np.empty((0,), dtype=np.float64)
    if phase_start is not None:
        diagnostics.concatenate_seconds = perf_counter() - phase_start

    phase_start = perf_counter() if profile_phases else None
    stencil = LightconeSparseStencil(
        pix_id=jnp.asarray(pix_id, dtype=jnp.int32),
        halo_id=jnp.asarray(halo_id, dtype=jnp.int32),
        r_perp=jnp.asarray(r_perp, dtype=jnp.asarray(catalog.chi).dtype),
        n_pix=n_pix,
    )
    if phase_start is not None:
        stencil.pix_id.block_until_ready()
        stencil.halo_id.block_until_ready()
        stencil.r_perp.block_until_ready()
        diagnostics.jax_transfer_seconds = perf_counter() - phase_start
    diagnostics.elapsed_seconds = perf_counter() - build_start
    diagnostics.halo_has_kept_pair = halo_has_kept_pair
    if collect_diagnostics:
        if diagnostics.n_kept_pairs_total != int(pix_id.shape[0]):
            raise RuntimeError("stencil diagnostics kept-pair count does not match stencil size")
        return stencil, diagnostics
    return stencil


def print_stencil_profile(diag: StencilBuildDiagnostics) -> None:
    """Print detailed host-side stencil timing without saving it to outputs."""

    print(
        "[profile] stencil phases (s): "
        f"total {diag.elapsed_seconds:.3f}; "
        f"query_disc {diag.query_disc_seconds:.3f}; "
        f"compact lookup {diag.compact_lookup_seconds:.3f}; "
        f"pix2vec/filter {diag.pix2vec_filter_seconds:.3f}; "
        f"concatenate {diag.concatenate_seconds:.3f}; "
        f"JAX transfer {diag.jax_transfer_seconds:.3f}; "
        f"residual {diag.residual_seconds:.3f}"
    )
    print(
        "[profile] stencil counts: "
        f"halos {diag.n_halos}; queried {diag.n_query_pixels_total}; "
        f"inside {diag.n_inside_domain_total}; kept {diag.n_kept_pairs_total}; "
        f"sub-pixel radii {diag.n_subpixel_radius_halos}"
    )


def paint_bucketed_nfw_sparse_map(
    stencil: LightconeSparseStencil,
    rank_catalog: LightconeHaloCatalog,
    metadata: PinocchioRunMetadata,
    particle_mass_msun_h: float,
    pixel_area_sr: float,
    concentration_amplitude: float,
    concentration_mass_slope: float,
    concentration_redshift_slope: float,
    concentration_mass_pivot: float,
    truncation_width_fraction: float,
    mass_definition: ResolvedHaloMassDefinition | None = None,
    *,
    compute_map_derivatives: bool,
    profile: bool,
) -> tuple[jax.Array, dict[str, float | str | np.ndarray]]:
    """Paint one padded sparse bucket and optional concentration JVP maps."""

    if mass_definition is None:
        mass_definition = default_halo_mass_definition().resolve(0)

    theta = jnp.asarray(
        [
            concentration_amplitude,
            concentration_mass_slope,
            concentration_redshift_slope,
        ],
        dtype=rank_catalog.mass.dtype,
    )
    scalar_dtype = theta.dtype
    particle_mass = jnp.asarray(particle_mass_msun_h, dtype=scalar_dtype)
    pixel_area = jnp.asarray(pixel_area_sr, dtype=scalar_dtype)
    mass_pivot = jnp.asarray(concentration_mass_pivot, dtype=scalar_dtype)
    truncation_width = jnp.asarray(truncation_width_fraction, dtype=scalar_dtype)
    overdensity = jnp.asarray(
        mass_definition.profile_overdensity,
        dtype=scalar_dtype,
    )

    if compute_map_derivatives:
        with timed_stage("NFW particle map + concentration JVPs", profile):
            if stencil.size == 0:
                particle_counts = jnp.zeros((stencil.n_pix,), dtype=scalar_dtype)
                derivative_maps = jnp.zeros(
                    (theta.shape[0], stencil.n_pix),
                    dtype=scalar_dtype,
                )
            else:
                particle_counts, derivative_maps = (
                    paint_nfw_particle_count_map_and_concentration_jvps_jit(
                        stencil,
                        rank_catalog,
                        theta,
                        particle_mass,
                        pixel_area,
                        metadata.cosmology,
                        mass_pivot,
                        truncation_width,
                        overdensity,
                        overdensity_mode=mass_definition.profile_mode,
                        reference_density=mass_definition.reference_density,
                    )
                )
            jax.block_until_ready((particle_counts, derivative_maps))
    else:
        with timed_stage("NFW particle map", profile):
            if stencil.size == 0:
                particle_counts = jnp.zeros((stencil.n_pix,), dtype=scalar_dtype)
            else:
                particle_counts = paint_nfw_particle_count_map_sparse_jit(
                    stencil,
                    rank_catalog,
                    theta,
                    particle_mass,
                    pixel_area,
                    metadata.cosmology,
                    mass_pivot,
                    truncation_width,
                    overdensity,
                    overdensity_mode=mass_definition.profile_mode,
                    reference_density=mass_definition.reference_density,
                )
            particle_counts.block_until_ready()

    if not compute_map_derivatives:
        return particle_counts, {"nfw_map_derivatives": "none"}

    with timed_stage("NFW map concentration derivatives to numpy", profile):
        derivative_maps_np = np.asarray(derivative_maps)
    return particle_counts, {
        "nfw_map_derivatives": "concentration",
        "d_nfw_particle_counts_d_concentration_amplitude": derivative_maps_np[0],
        "d_nfw_particle_counts_d_concentration_mass_slope": derivative_maps_np[1],
        "d_nfw_particle_counts_d_concentration_redshift_slope": derivative_maps_np[2],
    }


def run_nfw_calibration_pipeline(
    catalog: LightconeHaloCatalog,
    mask: np.ndarray,
    mass_map: PinocchioMassMap,
    metadata: PinocchioRunMetadata,
    particle_mass_msun_h: float,
    *,
    pipeline_mode: str = "paint",
    concentration_amplitude: float = 5.71,
    concentration_mass_slope: float = -0.084,
    concentration_redshift_slope: float = -0.47,
    concentration_mass_pivot: float = 2.0e12,
    truncation_width_fraction: float = 0.05,
    chunk_size: int | None = 1024,
    taper_radius_factor: float = 10.0,
    dense_demo: bool = False,
    compute_map_derivatives: bool = False,
    profile: bool = False,
    verbose: bool = True,
    pixel_index: MassMapPixelIndex | None = None,
    stencil_profile_recorder: StencilProfileRecorder | None = None,
    mass_definition: ResolvedHaloMassDefinition | None = None,
) -> dict[str, bool | float | int | str | np.ndarray]:
    """Paint the NFW calibration map and optional concentration derivatives.

    The sparse stencil geometry, halo selection, pixel selection, and support
    radius are fixed outside the differentiated JAX paths. ``mass_pivot`` is a
    fixed concentration-relation parameter and is not part of the derivative
    vector.
    """

    if particle_mass_msun_h <= 0.0:
        raise ValueError("particle_mass_msun_h must be positive")
    if concentration_amplitude <= 0.0:
        raise ValueError("concentration_amplitude must be positive")
    if concentration_mass_pivot <= 0.0:
        raise ValueError("concentration_mass_pivot must be positive")
    if truncation_width_fraction <= 0.0:
        raise ValueError("truncation_width_fraction must be positive")
    if dense_demo and compute_map_derivatives:
        raise ValueError("map-level concentration derivatives are only supported for sparse mode")
    if chunk_size is not None and chunk_size <= 0:
        chunk_size = None
    if profile and stencil_profile_recorder is None:
        stencil_profile_recorder = StencilProfileRecorder()
    if mass_definition is None:
        mass_definition = default_halo_mass_definition().resolve(0)

    with timed_stage("NFW selected catalogue", profile):
        selected_catalog = selected_lightcone_catalog(catalog, mask)
    selected_mass = np.asarray(selected_catalog.mass)
    selected_halo_mass_msun_h = float(np.sum(selected_mass, dtype=np.float64))
    pixel_area_sr = healpix_pixel_area_sr(mass_map.nside)
    n_halo = int(selected_catalog.mass.shape[0])
    n_pix = int(np.asarray(mass_map.pixel).shape[0])
    dense_pair_count = n_halo * n_pix
    pixel_unit_vectors = None

    concentration_params = ConcentrationParams(
        amplitude=concentration_amplitude,
        mass_slope=concentration_mass_slope,
        redshift_slope=concentration_redshift_slope,
        mass_pivot=concentration_mass_pivot,
    )
    profile_params = NFWProfileParams(
        overdensity=mass_definition.profile_overdensity,
        reference_density=mass_definition.reference_density,
        truncation_width_fraction=truncation_width_fraction,
        overdensity_mode=mass_definition.profile_mode,
    )
    stencil = None
    bucketed_stencil = None
    nfw_particle_counts = None
    map_derivative_diagnostics: dict[str, float | str | np.ndarray] = {
        "nfw_map_derivatives": "none"
    }
    sparse_pair_count = dense_pair_count
    halos_with_zero_pairs = 0
    mass_in_halos_with_zero_pairs_msun_h = 0.0
    subpixel_support_halo_count = 0
    if not dense_demo:
        with timed_stage("NFW rmax", profile):
            rmax = nfw_stencil_rmax_mpc_h(
                selected_catalog,
                metadata,
                concentration_params,
                profile_params,
                taper_radius_factor,
            )
        with timed_stage("NFW local sparse stencil", profile):
            stencil_result = build_lightcone_sparse_stencil_for_mass_map_local(
                mass_map,
                selected_catalog,
                rmax,
                collect_diagnostics=True,
                pixel_index=pixel_index,
                profile_phases=profile,
            )
        stencil, collected_stencil_diag = stencil_result
        if stencil_profile_recorder is not None:
            stencil_profile_recorder.diagnostics = collected_stencil_diag
        subpixel_support_halo_count = collected_stencil_diag.n_subpixel_radius_halos
        sparse_pair_count = int(stencil.size)
        assert collected_stencil_diag.halo_has_kept_pair is not None
        zero_pair_mask = ~collected_stencil_diag.halo_has_kept_pair
        halos_with_zero_pairs = int(np.count_nonzero(zero_pair_mask))
        mass_in_halos_with_zero_pairs_msun_h = float(
            np.sum(selected_mass[zero_pair_mask], dtype=np.float64)
        )
        with timed_stage("NFW sparse JIT bucket", profile):
            bucketed_stencil = bucket_sparse_stencil_for_rank_catalog(
                stencil,
                mask,
                int(catalog.mass.shape[0]),
                profile_diagnostics=(
                    None
                    if stencil_profile_recorder is None
                    else stencil_profile_recorder.diagnostics
                ),
            )
        if (
            verbose
            and stencil_profile_recorder is not None
            and stencil_profile_recorder.diagnostics is not None
        ):
            print_stencil_profile(stencil_profile_recorder.diagnostics)
        if verbose:
            print("NFW sparse stencil:")
            print(f"  Selected halos: {n_halo}")
            print(f"  Compact pixels: {n_pix}")
            print(f"  Sparse halo-pixel pairs: {sparse_pair_count}")
            print(f"  Dense pair count: {dense_pair_count}")
            print(
                "  Sparse compression factor: "
                f"{_compression_factor(dense_pair_count, sparse_pair_count):.12g}"
            )
    else:
        with timed_stage("NFW dense pixel vectors", profile):
            pixel_unit_vectors = jnp.asarray(
                healpix_pixel_unit_vectors(mass_map.nside, np.asarray(mass_map.pixel), nest=False)
            )

    if dense_demo:
        with timed_stage("NFW particle map", profile):
            assert pixel_unit_vectors is not None
            nfw_particle_counts = paint_lightcone_particle_count_map(
                pixel_unit_vectors,
                selected_catalog,
                particle_mass_msun_h=particle_mass_msun_h,
                pixel_area_sr=pixel_area_sr,
                cosmology=metadata.cosmology,
                concentration_params=concentration_params,
                profile_params=profile_params,
                chunk_size=chunk_size,
            )
            nfw_particle_counts.block_until_ready()
    else:
        assert bucketed_stencil is not None
        nfw_particle_counts, map_derivative_diagnostics = paint_bucketed_nfw_sparse_map(
            bucketed_stencil,
            catalog,
            metadata,
            particle_mass_msun_h,
            pixel_area_sr,
            concentration_amplitude,
            concentration_mass_slope,
            concentration_redshift_slope,
            concentration_mass_pivot,
            truncation_width_fraction,
            mass_definition,
            compute_map_derivatives=compute_map_derivatives,
            profile=profile,
        )

    total_counts = jnp.sum(nfw_particle_counts)

    with timed_stage("NFW particle map to numpy", profile):
        nfw_particle_counts_np = np.asarray(nfw_particle_counts)

    painted_particle_count = float(total_counts)
    expected_particle_count = selected_halo_mass_msun_h / particle_mass_msun_h
    painted_to_expected_ratio = (
        painted_particle_count / expected_particle_count
        if expected_particle_count > 0.0
        else float("nan")
    )

    diagnostics: dict[str, bool | float | int | str | np.ndarray] = {
        "pipeline_mode": pipeline_mode,
        "particle_mass_msun_h": float(particle_mass_msun_h),
        "nfw_particle_counts": nfw_particle_counts_np,
        "nfw_paint_mode": "dense" if dense_demo else "sparse",
        "nfw_selected_halo_count": int(selected_catalog.mass.shape[0]),
        "nfw_selected_halo_mass_msun_h": selected_halo_mass_msun_h,
        "nfw_expected_particle_count": expected_particle_count,
        "nfw_painted_particle_count": painted_particle_count,
        "nfw_painted_to_expected_ratio": painted_to_expected_ratio,
        "nfw_halos_with_zero_pairs": halos_with_zero_pairs,
        "nfw_mass_in_halos_with_zero_pairs_msun_h": (
            mass_in_halos_with_zero_pairs_msun_h
        ),
        "nfw_subpixel_support_halo_count": subpixel_support_halo_count,
        "nfw_compact_pixel_count": n_pix,
        "nfw_sparse_pair_count": sparse_pair_count,
        "nfw_dense_pair_count": dense_pair_count,
        "nfw_sparse_compression_factor": _compression_factor(
            dense_pair_count, sparse_pair_count
        ),
        "nfw_sum_particle_counts": float(total_counts),
        "nfw_concentration_amplitude": float(concentration_amplitude),
        "nfw_concentration_mass_slope": float(concentration_mass_slope),
        "nfw_concentration_redshift_slope": float(concentration_redshift_slope),
        "nfw_concentration_mass_pivot": float(concentration_mass_pivot),
        "nfw_truncation_width_fraction": float(truncation_width_fraction),
        "nfw_mass_definition": mass_definition.label,
        "nfw_overdensity_mode": mass_definition.mode,
        "nfw_overdensity": (
            ""
            if mass_definition.reported_overdensity is None
            else float(mass_definition.reported_overdensity)
        ),
        "nfw_reference_density": mass_definition.reference_density,
        "nfw_overdensity_file": (
            "" if mass_definition.source is None else str(mass_definition.source)
        ),
        "nfw_mass_conversion": "none_catalog_mass_interpreted_as_profile_mass",
    }
    diagnostics.update(map_derivative_diagnostics)
    return diagnostics


def save_npz(
    output: Path,
    nfw_diagnostics: dict[str, bool | float | int | str | np.ndarray],
) -> None:
    """Save only computed training arrays in a compressed NPZ container."""

    map_key = "nfw_particle_counts"
    nfw_map = np.asarray(nfw_diagnostics[map_key])
    if nfw_map.ndim != 1:
        raise ValueError("nfw_particle_counts must be one-dimensional")

    payload = {map_key: nfw_map}
    derivative_keys = (
        "d_nfw_particle_counts_d_concentration_amplitude",
        "d_nfw_particle_counts_d_concentration_mass_slope",
        "d_nfw_particle_counts_d_concentration_redshift_slope",
    )
    if nfw_diagnostics.get("nfw_map_derivatives", "none") == "concentration":
        for key in derivative_keys:
            derivative = np.asarray(nfw_diagnostics[key])
            if derivative.shape != nfw_map.shape:
                raise ValueError(f"{key} must match nfw_particle_counts shape")
            payload[key] = derivative

    np.savez_compressed(Path(output), **payload)


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    """Write an all-segments scientific-provenance CSV manifest."""

    columns = [
        "segment_index",
        "parameter_file",
        "sheets_file",
        "plc_catalog",
        "mass_map_path",
        "output_npz",
        "z_lo",
        "z_hi",
        "chi_lo_mpc_h",
        "chi_hi_mpc_h",
        "inclusive_upper",
        "catalog_format",
        "redshift_mode",
        "bounds_mode",
        "light_plc",
        "hubble_table",
        "particle_mass_msun_h",
        "jax_precision",
        "pipeline_mode",
        "nfw_paint_mode",
        "nfw_map_derivatives",
        "nfw_mass_definition",
        "nfw_overdensity_mode",
        "nfw_overdensity",
        "nfw_reference_density",
        "nfw_overdensity_file",
        "nfw_mass_conversion",
        "nfw_concentration_amplitude",
        "nfw_concentration_mass_slope",
        "nfw_concentration_redshift_slope",
        "nfw_concentration_mass_pivot_msun_h",
        "nfw_truncation_width_fraction",
        "nfw_taper_radius_factor",
        "selected_halo_count",
        "selected_halo_mass_msun_h",
        "expected_particle_count",
        "painted_particle_count",
        "painted_to_expected_ratio",
        "sparse_pair_count",
        "halos_with_zero_pairs",
        "mass_in_halos_with_zero_pairs_msun_h",
        "subpixel_support_halo_count",
        "mpi_rank_count",
        "segment_worker_count",
        "git_commit",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _range_text(lo: float, hi: float, inclusive_upper: bool) -> str:
    right = "]" if inclusive_upper else ")"
    return f"[{lo:g}, {hi:g}{right}"


def print_segment_summary(bounds: dict[str, float], inclusive_upper: bool) -> None:
    """Print selected segment bounds."""

    print("Selected segment:")
    print(f"  sheet_index = {int(bounds['sheet_index'])}")
    print(f"  z range     = {_range_text(bounds['z_lo'], bounds['z_hi'], inclusive_upper)}")
    print(f"  a range     = {_range_text(bounds['a_lo'], bounds['a_hi'], inclusive_upper)}")
    print(
        "  chi range   = "
        f"{_range_text(bounds['chi_lo_mpc_h'], bounds['chi_hi_mpc_h'], inclusive_upper)} Mpc/h"
    )


def nfw_stage_label(mode: str) -> str:
    """Return a clear top-level profile label for the requested NFW mode."""

    return f"NFW calibration pipeline: {mode}"


def print_nfw_calibration_summary(
    nfw_diagnostics: dict[str, bool | float | int | str | np.ndarray],
) -> None:
    """Print the NFW calibration pipeline summary."""

    if nfw_diagnostics.get("nfw_map_derivatives", "none") == "concentration":
        print("NFW calibration map + derivatives:")
    else:
        print("NFW calibration map:")
    print(f"  Pipeline mode: {nfw_diagnostics['pipeline_mode']}")
    print(f"  Painter mode: {nfw_diagnostics['nfw_paint_mode']}")
    print(f"  Selected halos painted: {nfw_diagnostics['nfw_selected_halo_count']}")
    print(
        "  Selected halo mass: "
        f"{nfw_diagnostics['nfw_selected_halo_mass_msun_h']:.12g} Msun/h"
    )
    print(f"  Compact pixels: {nfw_diagnostics['nfw_compact_pixel_count']}")
    print(f"  Sparse halo-pixel pairs: {nfw_diagnostics['nfw_sparse_pair_count']}")
    print(f"  Dense pair count: {nfw_diagnostics['nfw_dense_pair_count']}")
    print(
        "  Sparse compression factor: "
        f"{nfw_diagnostics['nfw_sparse_compression_factor']:.12g}"
    )
    print(f"  NFW sum particle counts: {nfw_diagnostics['nfw_sum_particle_counts']:.12g}")
    print(
        "  Painted / expected particle count: "
        f"{nfw_diagnostics['nfw_painted_to_expected_ratio']:.12g}"
    )
    print(f"  Halos with zero pairs: {nfw_diagnostics['nfw_halos_with_zero_pairs']}")
    print(
        "  Mass in zero-pair halos: "
        f"{nfw_diagnostics['nfw_mass_in_halos_with_zero_pairs_msun_h']:.12g} Msun/h"
    )
    print(
        "  Subpixel-support halos: "
        f"{nfw_diagnostics['nfw_subpixel_support_halo_count']}"
    )
    if nfw_diagnostics.get("nfw_map_derivatives", "none") == "concentration":
        print("  Map derivatives: concentration")


def compute_calibration_for_segment(
    *,
    segment_index: int,
    mass_map_path: Path,
    output_npz: Path,
    catalog: LightconeHaloCatalog,
    sheets: Any,
    metadata: PinocchioRunMetadata,
    particle_mass: float,
    args: argparse.Namespace,
    profile: bool,
    compute_map_derivatives: bool,
    inclusive_upper: bool,
    verbose: bool = True,
    mass_definition: HaloMassDefinition | None = None,
) -> CalibrationSegmentResult:
    """Compute the NFW calibration payload for one segment without writing files."""

    stage_profile = profile and verbose
    if verbose:
        print(f"Processing segment {segment_index}: {mass_map_path}")
    with timed_stage("segment bounds", stage_profile):
        bounds = segment_bounds(sheets, segment_index)

    with timed_stage("read mass map", stage_profile):
        mass_map = read_pinocchio_mass_map_fits(mass_map_path)
        validate_mass_map(mass_map)
    with timed_stage("mass-map pixel index", stage_profile):
        pixel_index = MassMapPixelIndex.from_mass_map(mass_map)

    with timed_stage("select segment mask", stage_profile):
        mask = select_segment_mask(
            catalog,
            bounds,
            mode=args.bounds,
            inclusive_upper=inclusive_upper,
        )

    if verbose:
        print_segment_summary(bounds, inclusive_upper)
    if mass_definition is None:
        mass_definition = default_halo_mass_definition()
    resolved_mass_definition = mass_definition.resolve(segment_index)
    stencil_profile_recorder = StencilProfileRecorder() if profile else None
    with timed_stage(nfw_stage_label(args.mode), stage_profile):
        nfw_diagnostics = run_nfw_calibration_pipeline(
            catalog,
            mask,
            mass_map,
            metadata,
            particle_mass,
            pipeline_mode=args.mode,
            concentration_amplitude=args.concentration_amplitude,
            concentration_mass_slope=args.concentration_mass_slope,
            concentration_redshift_slope=args.concentration_redshift_slope,
            concentration_mass_pivot=args.concentration_mass_pivot,
            truncation_width_fraction=args.truncation_width_fraction,
            chunk_size=args.nfw_chunk_size,
            taper_radius_factor=args.nfw_taper_radius_factor,
            dense_demo=args.nfw_dense_demo,
            compute_map_derivatives=compute_map_derivatives,
            profile=stage_profile,
            verbose=verbose,
            pixel_index=pixel_index,
            stencil_profile_recorder=stencil_profile_recorder,
            mass_definition=resolved_mass_definition,
        )

    nfw_map = np.asarray(nfw_diagnostics["nfw_particle_counts"])
    if nfw_map.shape != np.asarray(mass_map.temperature).shape:
        raise RuntimeError("NFW map shape does not match mass_map.temperature")

    return CalibrationSegmentResult(
        segment_index=int(segment_index),
        mass_map_path=Path(mass_map_path),
        output_npz=Path(output_npz),
        bounds=bounds,
        inclusive_upper=bool(inclusive_upper),
        nfw_diagnostics=nfw_diagnostics,
        profile_stencil_diagnostics=(
            None if stencil_profile_recorder is None else stencil_profile_recorder.diagnostics
        ),
    )


def calibration_manifest_row(
    result: CalibrationSegmentResult,
    args: argparse.Namespace,
    metadata: PinocchioRunMetadata,
    provenance: ExecutionProvenance | None = None,
) -> dict[str, object]:
    """Return the manifest row for a computed or MPI-reduced segment."""

    nfw_diagnostics = result.nfw_diagnostics
    if provenance is None:
        provenance = ExecutionProvenance(
            mpi_rank_count=1,
            segment_worker_count=int(getattr(args, "segment_workers", 1)),
            git_commit=resolve_git_commit(),
        )
    row: dict[str, object] = {
        "segment_index": int(result.segment_index),
        "parameter_file": str(args.params),
        "sheets_file": str(args.sheets),
        "plc_catalog": str(args.plc_catalog),
        "mass_map_path": str(result.mass_map_path),
        "output_npz": str(result.output_npz),
        "z_lo": float(result.bounds["z_lo"]),
        "z_hi": float(result.bounds["z_hi"]),
        "chi_lo_mpc_h": float(result.bounds["chi_lo_mpc_h"]),
        "chi_hi_mpc_h": float(result.bounds["chi_hi_mpc_h"]),
        "inclusive_upper": bool(result.inclusive_upper),
        "catalog_format": str(args.catalog_format),
        "redshift_mode": str(args.redshift_mode),
        "bounds_mode": str(args.bounds),
        "light_plc": bool(args.light_plc),
        "hubble_table": "" if args.hubble_table is None else str(args.hubble_table),
        "particle_mass_msun_h": float(metadata.particle_mass_msun_h),
        "jax_precision": str(args.jax_precision),
        "pipeline_mode": str(args.mode),
        "nfw_paint_mode": str(nfw_diagnostics["nfw_paint_mode"]),
        "nfw_map_derivatives": str(nfw_diagnostics["nfw_map_derivatives"]),
        "nfw_mass_definition": str(nfw_diagnostics["nfw_mass_definition"]),
        "nfw_overdensity_mode": str(nfw_diagnostics["nfw_overdensity_mode"]),
        "nfw_overdensity": nfw_diagnostics["nfw_overdensity"],
        "nfw_reference_density": str(nfw_diagnostics["nfw_reference_density"]),
        "nfw_overdensity_file": str(nfw_diagnostics["nfw_overdensity_file"]),
        "nfw_mass_conversion": str(nfw_diagnostics["nfw_mass_conversion"]),
        "nfw_concentration_amplitude": float(args.concentration_amplitude),
        "nfw_concentration_mass_slope": float(args.concentration_mass_slope),
        "nfw_concentration_redshift_slope": float(args.concentration_redshift_slope),
        "nfw_concentration_mass_pivot_msun_h": float(args.concentration_mass_pivot),
        "nfw_truncation_width_fraction": float(args.truncation_width_fraction),
        "nfw_taper_radius_factor": float(args.nfw_taper_radius_factor),
        "selected_halo_count": int(nfw_diagnostics["nfw_selected_halo_count"]),
        "selected_halo_mass_msun_h": float(
            nfw_diagnostics["nfw_selected_halo_mass_msun_h"]
        ),
        "expected_particle_count": float(nfw_diagnostics["nfw_expected_particle_count"]),
        "painted_particle_count": float(nfw_diagnostics["nfw_painted_particle_count"]),
        "painted_to_expected_ratio": float(
            nfw_diagnostics["nfw_painted_to_expected_ratio"]
        ),
        "sparse_pair_count": int(nfw_diagnostics["nfw_sparse_pair_count"]),
        "halos_with_zero_pairs": int(nfw_diagnostics["nfw_halos_with_zero_pairs"]),
        "mass_in_halos_with_zero_pairs_msun_h": float(
            nfw_diagnostics["nfw_mass_in_halos_with_zero_pairs_msun_h"]
        ),
        "subpixel_support_halo_count": int(
            nfw_diagnostics["nfw_subpixel_support_halo_count"]
        ),
        "mpi_rank_count": provenance.mpi_rank_count,
        "segment_worker_count": provenance.segment_worker_count,
        "git_commit": provenance.git_commit,
    }
    return row


def write_calibration_segment_outputs(
    result: CalibrationSegmentResult,
    args: argparse.Namespace,
    metadata: PinocchioRunMetadata,
    *,
    profile: bool,
    verbose: bool = True,
    provenance: ExecutionProvenance | None = None,
) -> dict[str, object]:
    """Write one computed segment payload and return its manifest row."""

    with timed_stage("save compressed NPZ", profile):
        save_npz(result.output_npz, result.nfw_diagnostics)

    if verbose:
        print_nfw_calibration_summary(result.nfw_diagnostics)
        print(f"Wrote NPZ: {result.output_npz}")

    return calibration_manifest_row(result, args, metadata, provenance)


def run_calibration_for_segment(
    *,
    segment_index: int,
    mass_map_path: Path,
    output_npz: Path,
    catalog: LightconeHaloCatalog,
    sheets: Any,
    metadata: PinocchioRunMetadata,
    particle_mass: float,
    args: argparse.Namespace,
    profile: bool,
    compute_map_derivatives: bool,
    inclusive_upper: bool,
    mass_definition: HaloMassDefinition | None = None,
    provenance: ExecutionProvenance | None = None,
) -> dict[str, object]:
    """Run the complete NFW calibration pipeline for one mass-map segment."""

    result = compute_calibration_for_segment(
        segment_index=segment_index,
        mass_map_path=mass_map_path,
        output_npz=output_npz,
        catalog=catalog,
        sheets=sheets,
        metadata=metadata,
        particle_mass=particle_mass,
        args=args,
        profile=profile,
        compute_map_derivatives=compute_map_derivatives,
        inclusive_upper=inclusive_upper,
        verbose=True,
        mass_definition=mass_definition,
    )
    return write_calibration_segment_outputs(
        result,
        args,
        metadata,
        profile=profile,
        verbose=True,
        provenance=provenance,
    )


def _mpi_reduce_array(
    value: np.ndarray,
    mpi_context: MpiContext,
) -> np.ndarray | None:
    """Sum one numeric array onto rank 0 with MPI's buffer collective."""

    send_buffer = np.ascontiguousarray(value)
    if send_buffer.dtype.hasobject:
        raise TypeError("MPI array reduction requires a numeric NumPy dtype")
    if not mpi_context.enabled:
        return send_buffer
    if mpi_context.comm is None:
        raise RuntimeError("MPI context is enabled but has no communicator")

    receive_buffer = np.empty_like(send_buffer) if mpi_context.is_root else None
    if mpi_context.sum_op is None:
        mpi_context.comm.Reduce(send_buffer, receive_buffer, root=0)
    else:
        mpi_context.comm.Reduce(
            send_buffer,
            receive_buffer,
            op=mpi_context.sum_op,
            root=0,
        )
    return receive_buffer


def gather_segment_execution_timings(
    timing: SegmentExecutionTiming,
    mpi_context: MpiContext,
) -> np.ndarray | None:
    """Gather profile-only rank timings onto rank 0 with an MPI buffer call."""

    send_buffer = timing.as_array()
    if not mpi_context.enabled:
        return send_buffer[np.newaxis, :]
    if mpi_context.comm is None:
        raise RuntimeError("MPI context is enabled but has no communicator")

    receive_buffer = (
        np.empty((mpi_context.size, send_buffer.size), dtype=np.float64)
        if mpi_context.is_root
        else None
    )
    mpi_context.comm.Gather(send_buffer, receive_buffer, root=0)
    return receive_buffer


def _timing_min_mean_max(values: np.ndarray) -> str:
    values = np.asarray(values, dtype=np.float64)
    return f"{np.min(values):.3f}/{np.mean(values):.3f}/{np.max(values):.3f}"


def print_rank_timing_summary(label: str, rank_timings: np.ndarray) -> None:
    """Print root-only min/mean/max rank timings for one segment or run."""

    rank_timings = np.asarray(rank_timings, dtype=np.float64)
    if rank_timings.ndim != 2 or rank_timings.shape[1] != 14:
        raise ValueError("rank timings must have shape (n_rank, 14)")
    print(
        f"[profile] {label} rank min/mean/max (s): "
        f"compute {_timing_min_mean_max(rank_timings[:, 0])}; "
        f"result wait {_timing_min_mean_max(rank_timings[:, 1])}; "
        f"MPI reduce {_timing_min_mean_max(rank_timings[:, 2])}"
    )
    if not np.any(rank_timings[:, 3:]):
        return

    residual = np.maximum(
        rank_timings[:, 3] - np.sum(rank_timings[:, 4:9], axis=1),
        0.0,
    )
    print(
        f"[profile] {label} stencil rank min/mean/max (s): "
        f"total {_timing_min_mean_max(rank_timings[:, 3])}; "
        f"query_disc {_timing_min_mean_max(rank_timings[:, 4])}; "
        f"compact lookup {_timing_min_mean_max(rank_timings[:, 5])}; "
        f"pix2vec/filter {_timing_min_mean_max(rank_timings[:, 6])}; "
        f"concatenate {_timing_min_mean_max(rank_timings[:, 7])}; "
        f"JAX transfer {_timing_min_mean_max(rank_timings[:, 8])}; "
        f"residual {_timing_min_mean_max(residual)}"
    )
    counts = np.rint(np.sum(rank_timings[:, 9:14], axis=0)).astype(np.int64)
    print(
        f"[profile] {label} stencil counts (rank sum): "
        f"halos {counts[0]}; queried {counts[1]}; inside {counts[2]}; "
        f"kept {counts[3]}; sub-pixel radii {counts[4]}"
    )


def reduce_calibration_segment_result(
    local_result: CalibrationSegmentResult,
    mpi_context: MpiContext,
) -> CalibrationSegmentResult | None:
    """Sum per-rank segment maps and additive diagnostics onto rank 0."""

    if not mpi_context.enabled:
        return local_result

    reduced_nfw_counts = _mpi_reduce_array(
        np.asarray(local_result.nfw_diagnostics["nfw_particle_counts"]),
        mpi_context,
    )

    derivative_array_keys = (
        "d_nfw_particle_counts_d_concentration_amplitude",
        "d_nfw_particle_counts_d_concentration_mass_slope",
        "d_nfw_particle_counts_d_concentration_redshift_slope",
    )
    reduced_derivative_arrays = {
        key: _mpi_reduce_array(
            np.asarray(local_result.nfw_diagnostics[key]),
            mpi_context,
        )
        for key in derivative_array_keys
        if key in local_result.nfw_diagnostics
    }

    nfw_integer_sum_keys = (
        "nfw_selected_halo_count",
        "nfw_sparse_pair_count",
        "nfw_dense_pair_count",
        "nfw_halos_with_zero_pairs",
        "nfw_subpixel_support_halo_count",
    )
    present_nfw_keys = tuple(
        key for key in nfw_integer_sum_keys if key in local_result.nfw_diagnostics
    )
    integer_diagnostics = np.asarray(
        [local_result.nfw_diagnostics[key] for key in present_nfw_keys],
        dtype=np.int64,
    )
    reduced_integer_diagnostics = _mpi_reduce_array(integer_diagnostics, mpi_context)

    nfw_float_sum_keys = (
        "nfw_selected_halo_mass_msun_h",
        "nfw_mass_in_halos_with_zero_pairs_msun_h",
    )
    present_nfw_float_keys = tuple(
        key for key in nfw_float_sum_keys if key in local_result.nfw_diagnostics
    )
    float_diagnostics = np.asarray(
        [local_result.nfw_diagnostics[key] for key in present_nfw_float_keys],
        dtype=np.float64,
    )
    reduced_float_diagnostics = _mpi_reduce_array(float_diagnostics, mpi_context)

    if not mpi_context.is_root:
        return None

    assert reduced_nfw_counts is not None
    assert reduced_integer_diagnostics is not None
    assert reduced_float_diagnostics is not None

    nfw_diagnostics = dict(local_result.nfw_diagnostics)
    nfw_diagnostics["nfw_particle_counts"] = np.asarray(reduced_nfw_counts)
    for key, value in reduced_derivative_arrays.items():
        assert value is not None
        nfw_diagnostics[key] = np.asarray(value)
    for key, value in zip(
        present_nfw_keys,
        reduced_integer_diagnostics,
        strict=True,
    ):
        nfw_diagnostics[key] = int(value)
    for key, value in zip(
        present_nfw_float_keys,
        reduced_float_diagnostics,
        strict=True,
    ):
        nfw_diagnostics[key] = float(value)
    painted_particle_count = float(
        np.sum(nfw_diagnostics["nfw_particle_counts"], dtype=np.float64)
    )
    nfw_diagnostics["nfw_sum_particle_counts"] = painted_particle_count
    nfw_diagnostics["nfw_painted_particle_count"] = painted_particle_count
    if "nfw_selected_halo_mass_msun_h" in nfw_diagnostics:
        particle_mass = float(nfw_diagnostics["particle_mass_msun_h"])
        expected_particle_count = (
            float(nfw_diagnostics["nfw_selected_halo_mass_msun_h"]) / particle_mass
        )
        nfw_diagnostics["nfw_expected_particle_count"] = expected_particle_count
        nfw_diagnostics["nfw_painted_to_expected_ratio"] = (
            painted_particle_count / expected_particle_count
            if expected_particle_count > 0.0
            else float("nan")
        )
    nfw_diagnostics["nfw_sparse_compression_factor"] = _compression_factor(
        int(nfw_diagnostics["nfw_dense_pair_count"]),
        int(nfw_diagnostics["nfw_sparse_pair_count"]),
    )
    return CalibrationSegmentResult(
        segment_index=local_result.segment_index,
        mass_map_path=local_result.mass_map_path,
        output_npz=local_result.output_npz,
        bounds=local_result.bounds,
        inclusive_upper=local_result.inclusive_upper,
        nfw_diagnostics=nfw_diagnostics,
    )


def run_segment_workflow(
    args: argparse.Namespace,
    *,
    workflow: str,
    catalog: LightconeHaloCatalog,
    sheets: Any,
    metadata: PinocchioRunMetadata,
    particle_mass: float,
    profile: bool,
    compute_map_derivatives: bool,
    mpi_context: MpiContext | None = None,
    mass_definition: HaloMassDefinition | None = None,
) -> list[dict[str, object]]:
    """Run either the single-segment or all-segments workflow."""

    if mpi_context is None:
        mpi_context = MpiContext()
    if mass_definition is None:
        mass_definition = default_halo_mass_definition()
    provenance = execution_provenance(args, mpi_context)
    segment_workers = int(getattr(args, "segment_workers", 1))
    if segment_workers < 1:
        raise ValueError("--segment-workers must be at least 1")

    if workflow == "single":
        segments = [(int(args.sheet_index), Path(args.mass_map))]
        output_paths = [Path(args.output)]
        inclusive_values = [bool(args.last_segment_inclusive)]
    elif workflow == "all":
        if bool(args.last_segment_inclusive):
            raise ValueError(
                "--last-segment-inclusive is only valid in single-segment mode"
            )
        segments = discover_mass_map_segments(str(args.mass_map_glob))
        validate_mass_map_segment_batch(segments, sheets)
        output_dir = Path(args.output_dir)
        if mpi_context.is_root:
            output_dir.mkdir(parents=True, exist_ok=True)
        final_sheet_index = len(sheets) - 1
        output_paths = []
        inclusive_values = []
        for segment_index, _ in segments:
            output_paths.append(segment_output_path(output_dir, segment_index))
            inclusive_values.append(segment_index == final_sheet_index)
    else:
        raise ValueError("workflow must be 'single' or 'all'")

    segment_specs = list(
        zip(
            segments,
            output_paths,
            inclusive_values,
            strict=True,
        )
    )

    manifest_rows = []
    profile_rank_timings: list[np.ndarray] = []
    if not mpi_context.enabled and segment_workers == 1:
        for (segment_index, mass_map_path), output_npz, inclusive_upper in (
            segment_specs
        ):
            manifest_rows.append(
                run_calibration_for_segment(
                    segment_index=segment_index,
                    mass_map_path=mass_map_path,
                    output_npz=output_npz,
                    catalog=catalog,
                    sheets=sheets,
                    metadata=metadata,
                    particle_mass=particle_mass,
                    args=args,
                    profile=profile,
                    compute_map_derivatives=compute_map_derivatives,
                    inclusive_upper=inclusive_upper,
                    mass_definition=mass_definition,
                    provenance=provenance,
                )
            )
    else:

        def compute_one(spec) -> ComputedCalibrationSegment:
            (segment_index, mass_map_path), output_npz, inclusive_upper = spec
            try:
                compute_start = perf_counter() if profile else None
                result = compute_calibration_for_segment(
                    segment_index=segment_index,
                    mass_map_path=mass_map_path,
                    output_npz=output_npz,
                    catalog=catalog,
                    sheets=sheets,
                    metadata=metadata,
                    particle_mass=particle_mass,
                    args=args,
                    profile=profile,
                    compute_map_derivatives=compute_map_derivatives,
                    inclusive_upper=inclusive_upper,
                    verbose=False,
                    mass_definition=mass_definition,
                )
            except BaseException as exc:
                contextual_error = RuntimeError(
                    f"rank {mpi_context.rank} failed while computing segment "
                    f"{segment_index} from {mass_map_path}"
                )
                contextual_error.__cause__ = exc
                if mpi_context.enabled:
                    abort_mpi_job(mpi_context, contextual_error)
                raise contextual_error from exc
            compute_seconds = (
                0.0 if compute_start is None else perf_counter() - compute_start
            )
            return ComputedCalibrationSegment(
                result=result,
                compute_seconds=compute_seconds,
            )

        def reduce_and_write(
            computed: ComputedCalibrationSegment,
            *,
            result_wait_seconds: float,
        ) -> dict[str, object] | None:
            reduction_start = perf_counter() if profile else None
            reduced_result = reduce_calibration_segment_result(
                computed.result,
                mpi_context,
            )
            reduction_seconds = (
                0.0 if reduction_start is None else perf_counter() - reduction_start
            )
            if profile and mpi_context.enabled:
                rank_timings = gather_segment_execution_timings(
                    SegmentExecutionTiming(
                        compute_seconds=computed.compute_seconds,
                        result_wait_seconds=result_wait_seconds,
                        reduction_seconds=reduction_seconds,
                        stencil_diagnostics=(
                            computed.result.profile_stencil_diagnostics
                        ),
                    ),
                    mpi_context,
                )
                if rank_timings is not None:
                    print_rank_timing_summary(
                        f"segment {computed.result.segment_index:03d}",
                        rank_timings,
                    )
                    profile_rank_timings.append(rank_timings)
            if reduced_result is None:
                return None
            return write_calibration_segment_outputs(
                reduced_result,
                args,
                metadata,
                profile=profile,
                verbose=mpi_context.is_root,
                provenance=provenance,
            )

        if mpi_context.enabled:
            if segment_workers == 1:
                for spec in segment_specs:
                    result_wait_start = perf_counter() if profile else None
                    computed = compute_one(spec)
                    result_wait_seconds = (
                        0.0
                        if result_wait_start is None
                        else perf_counter() - result_wait_start
                    )
                    row = reduce_and_write(
                        computed,
                        result_wait_seconds=result_wait_seconds,
                    )
                    if row is not None:
                        manifest_rows.append(row)
            else:
                spec_iter = iter(segment_specs)
                pending = deque()
                with ThreadPoolExecutor(max_workers=segment_workers) as executor:
                    for _ in range(segment_workers):
                        try:
                            pending.append(executor.submit(compute_one, next(spec_iter)))
                        except StopIteration:
                            break

                    while pending:
                        future = pending.popleft()
                        result_wait_start = perf_counter() if profile else None
                        computed = future.result()
                        result_wait_seconds = (
                            0.0
                            if result_wait_start is None
                            else perf_counter() - result_wait_start
                        )
                        row = reduce_and_write(
                            computed,
                            result_wait_seconds=result_wait_seconds,
                        )
                        if row is not None:
                            manifest_rows.append(row)
                        try:
                            pending.append(executor.submit(compute_one, next(spec_iter)))
                        except StopIteration:
                            pass
        else:
            if segment_workers == 1:
                local_results = [compute_one(spec) for spec in segment_specs]
            else:
                with ThreadPoolExecutor(max_workers=segment_workers) as executor:
                    local_results = list(executor.map(compute_one, segment_specs))

            for computed in sorted(
                local_results,
                key=lambda item: item.result.segment_index,
            ):
                manifest_rows.append(
                    write_calibration_segment_outputs(
                        computed.result,
                        args,
                        metadata,
                        profile=profile,
                        verbose=mpi_context.is_root,
                        provenance=provenance,
                    )
                )

    if profile_rank_timings and mpi_context.is_root:
        rank_totals = np.sum(np.stack(profile_rank_timings, axis=0), axis=0)
        print_rank_timing_summary("all-segment totals", rank_totals)

    if workflow == "all" and mpi_context.is_root:
        manifest_path = Path(args.output_dir) / "painted_nfw_manifest.csv"
        with timed_stage("write manifest", profile):
            write_manifest(manifest_path, manifest_rows)
        print(f"Wrote manifest: {manifest_path}")
    return manifest_rows


def main() -> None:
    """Run the command-line calibration workflow."""

    args = parse_args()
    workflow = validate_segment_workflow_args(args)
    mpi_context = initialize_mpi_context(bool(args.mpi_plc_parts))
    try:
        warn_if_float32_precision(args, mpi_context)
        validate_mpi_workflow_args(args, workflow=workflow, mpi_context=mpi_context)
        profile = args.mode in ("profile", "derivatives-profile")
        compute_map_derivatives = args.mode in ("derivatives", "derivatives-profile")

        with timed_stage("read parameter file", profile):
            metadata = read_pinocchio_parameter_file(args.params)
        particle_mass = float(metadata.particle_mass_msun_h)
        if particle_mass <= 0.0:
            raise ValueError("particle_mass_msun_h must be positive")
        if args.concentration_amplitude <= 0.0:
            raise ValueError("--concentration-amplitude must be positive")
        if args.concentration_mass_pivot <= 0.0:
            raise ValueError("--concentration-mass-pivot must be positive")
        if args.truncation_width_fraction <= 0.0:
            raise ValueError("--truncation-width-fraction must be positive")

        with timed_stage("read sheets", profile):
            sheets = read_pinocchio_mass_sheets(args.sheets)
        mass_definition = load_halo_mass_definition(args, sheets)
        with timed_stage("load PLC catalogue", profile):
            catalog = load_rank_local_lightcone_catalog(args, mpi_context)
            validate_catalog_for_binning(catalog)

        run_segment_workflow(
            args,
            workflow=workflow,
            catalog=catalog,
            sheets=sheets,
            metadata=metadata,
            particle_mass=particle_mass,
            profile=profile,
            compute_map_derivatives=compute_map_derivatives,
            mpi_context=mpi_context,
            mass_definition=mass_definition,
        )
    except BaseException as exc:
        if mpi_context.comm is not None and mpi_context.size > 1:
            abort_mpi_job(mpi_context, exc)
        raise


if __name__ == "__main__":
    main()
