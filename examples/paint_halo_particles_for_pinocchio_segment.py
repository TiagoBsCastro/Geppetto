"""Run a PINOCCHIO-to-NFW map calibration pipeline on mass-map segments.

This diagnostic script reads a PINOCCHIO parameter file, mass-sheet table,
on-the-fly HEALPix mass-map FITS files, and a PLC halo catalogue. It can run on
one selected segment or on all existing segments discovered from a glob. For
each segment it writes two compact maps with the same pixel domain and row
ordering as the corresponding ``*.massmap.segXXX.fits`` file:

    halo_particle_counts: point-halo resolved mass / particle mass
    nfw_particle_counts: projected NFW one-halo mass / particle mass

The NFW map is intended for calibrating a concentration--mass relation against
a theoretical prediction while preserving PINOCCHIO's segment bounds and
compact pixel ordering. In derivative modes the script also saves map-level
derivatives with respect to concentration amplitude, mass slope, and redshift
slope. HEALPix stencil construction is fixed geometry and is not differentiated.
In all-segments mode the output is one segment-local NPZ and one compact NFW
FITS map per input segment; no global light-cone map is merged in this script.

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
  --output path/to/halo_particles.seg000.npz

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
    --mode derivatives
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from geppetto import (
    ConcentrationParams,
    NFWProfileParams,
    paint_lightcone_particle_count_map,
    paint_lightcone_particle_count_map_sparse,
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
from geppetto.profiles import nfw_projected_surface_density, nfw_scale_radius_and_density

# Kept as a module attribute for regression tests proving the default sparse
# calibration path never calls the dense validation builder.
_BRUTE_FORCE_STENCIL_BUILDER_REGRESSION_SENTINEL = build_lightcone_sparse_stencil_bruteforce
_SEGMENT_RE = re.compile(r"seg(\d+)")


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


@dataclass
class CalibrationSegmentResult:
    """Computed segment payload, before any output files are written."""

    segment_index: int
    mass_map_path: Path
    output_npz: Path
    output_fits: Path | None
    bounds: dict[str, float]
    inclusive_upper: bool
    mass_map: PinocchioMassMap
    halo_particle_counts: np.ndarray
    diagnostics: dict[str, float | int]
    nfw_diagnostics: dict[str, bool | float | int | str | np.ndarray]


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
    query_mode: str = "inclusive"
    elapsed_seconds: float = 0.0

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
        help="Directory for all-segments painted NFW NPZ/FITS outputs and manifest",
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
    parser.add_argument("--output-fits", type=Path, help="Optional output HEALPix FITS table")
    parser.add_argument(
        "--last-segment-inclusive",
        action="store_true",
        help=(
            "In single-segment mode, use an inclusive upper segment bound. "
            "In all-segments mode, intermediate segments are half-open and the final "
            "discovered segment is inclusive automatically."
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
        "--stencil-query-mode",
        choices=("inclusive", "center"),
        default="inclusive",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--stencil-diagnostics",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--stencil-compare-query-modes",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--mpi-plc-parts",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--mpi-output-mode",
        choices=("reduce", "rank-local"),
        default="reduce",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--segment-workers",
        type=int,
        default=1,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


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
    if args.output_fits is not None and any(all_segment_args):
        raise ValueError("--output-fits is only supported in single-segment mode")
    if all(single_segment_args) and not any(all_segment_args):
        return "single"
    if all(all_segment_args) and not any(single_segment_args):
        if args.stencil_compare_query_modes:
            raise ValueError(
                "--stencil-compare-query-modes is currently supported only in single-segment mode"
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
    mpi_output_mode = getattr(args, "mpi_output_mode", "reduce")
    if mpi_output_mode == "rank-local" and not bool(getattr(args, "mpi_plc_parts", False)):
        raise ValueError("--mpi-output-mode rank-local requires --mpi-plc-parts")
    if mpi_context.size > 1 and not bool(getattr(args, "mpi_plc_parts", False)):
        raise ValueError("MPI world size > 1 requires --mpi-plc-parts")
    if mpi_context.enabled and bool(getattr(args, "stencil_compare_query_modes", False)):
        raise ValueError("--stencil-compare-query-modes is not supported with --mpi-plc-parts")
    if mpi_context.enabled:
        parts = discover_plc_catalog_parts(Path(args.plc_catalog))
        validate_mpi_plc_part_count(parts, mpi_context.size)


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


def segment_output_paths(output_dir: Path, segment_index: int) -> dict[str, Path]:
    """Return all-segments output paths for one segment index."""

    tag = f"seg{segment_index:03d}"
    return {
        "npz": output_dir / f"painted_nfw.{tag}.npz",
        "fits": output_dir / f"painted_nfw.{tag}.fits",
    }


def rank_suffix(rank: int) -> str:
    """Return the stable filename suffix for one MPI rank."""

    if rank < 0:
        raise ValueError("rank must be non-negative")
    return f"rank{rank:03d}"


def rank_local_output_path(path: Path | None, rank: int) -> Path | None:
    """Insert ``.rankNNN`` before the output path suffix."""

    if path is None:
        return None
    suffix = rank_suffix(rank)
    path = Path(path)
    return path.with_name(f"{path.stem}.{suffix}{path.suffix}")


def rank_local_output_specs(
    output_specs: list[tuple[Path, Path | None]],
    rank: int,
) -> list[tuple[Path, Path | None]]:
    """Return rank-suffixed NPZ/FITS output specs."""

    return [
        (rank_local_output_path(output_npz, rank), rank_local_output_path(output_fits, rank))
        for output_npz, output_fits in output_specs
    ]


def rank_local_manifest_path(output_dir: Path, rank: int) -> Path:
    """Return the rank-local all-segments manifest path."""

    return output_dir / f"painted_nfw_manifest.{rank_suffix(rank)}.csv"


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


def halo_rows_in_mass_map(
    catalog: LightconeHaloCatalog,
    mask: np.ndarray,
    mass_map: PinocchioMassMap,
) -> tuple[np.ndarray, np.ndarray]:
    """Map selected halo directions to compact mass-map rows."""

    try:
        import healpy as hp
    except ImportError as exc:  # pragma: no cover - exercised only without io extra
        raise RuntimeError("halo pixel binning requires healpy; install geppetto[io]") from exc

    validate_mass_map(mass_map)
    validate_catalog_for_binning(catalog)

    mass = np.asarray(catalog.mass)
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 1 or mask.shape[0] != mass.shape[0]:
        raise ValueError("mask must have shape (n_halo,)")

    uv = np.asarray(catalog.unit_vector)[mask]
    if uv.shape[0] == 0:
        empty_rows = np.empty((0,), dtype=np.int64)
        empty_inside = np.empty((0,), dtype=bool)
        return empty_rows, empty_inside

    halo_pix = hp.vec2pix(mass_map.nside, uv[:, 0], uv[:, 1], uv[:, 2], nest=False)
    pixel_to_row = {int(pixel): row for row, pixel in enumerate(np.asarray(mass_map.pixel))}
    rows = np.array([pixel_to_row.get(int(pixel), -1) for pixel in halo_pix], dtype=np.int64)
    inside_pixel_domain = rows >= 0
    return rows, inside_pixel_domain


def _accumulate_halo_particle_counts(
    catalog: LightconeHaloCatalog,
    mask: np.ndarray,
    mass_map: PinocchioMassMap,
    particle_mass_msun_h: float,
    rows: np.ndarray,
    inside_pixel_domain: np.ndarray,
) -> np.ndarray:
    if particle_mass_msun_h <= 0.0:
        raise ValueError("particle_mass_msun_h must be positive")

    out = np.zeros(np.asarray(mass_map.temperature).shape, dtype=np.float64)
    selected_masses = np.asarray(catalog.mass)[mask][inside_pixel_domain]
    selected_rows = rows[inside_pixel_domain]
    np.add.at(out, selected_rows, selected_masses / particle_mass_msun_h)
    return out


def build_halo_particle_count_map(
    catalog: LightconeHaloCatalog,
    mask: np.ndarray,
    mass_map: PinocchioMassMap,
    particle_mass_msun_h: float,
) -> np.ndarray:
    """Build the compact halo-particle-count map for selected halos."""

    rows, inside_pixel_domain = halo_rows_in_mass_map(catalog, mask, mass_map)
    return _accumulate_halo_particle_counts(
        catalog,
        np.asarray(mask, dtype=bool),
        mass_map,
        particle_mass_msun_h,
        rows,
        inside_pixel_domain,
    )


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


def build_lightcone_sparse_stencil_for_mass_map_local(
    mass_map: PinocchioMassMap,
    catalog: LightconeHaloCatalog,
    rmax_mpc_h: np.ndarray,
    *,
    query_mode: str = "inclusive",
    collect_diagnostics: bool = False,
) -> LightconeSparseStencil | tuple[LightconeSparseStencil, StencilBuildDiagnostics]:
    """Build a HEALPix-local sparse stencil on a compact PINOCCHIO map domain.

    The returned ``pix_id`` values are compact row indices into
    ``mass_map.pixel``, not global HEALPix pixel numbers. Geometry is fixed
    outside JAX; the differentiable sparse painter receives only the retained
    local halo-pixel pairs.
    """

    if query_mode not in ("inclusive", "center"):
        raise ValueError("query_mode must be 'inclusive' or 'center'")

    try:
        import healpy as hp
    except ImportError as exc:  # pragma: no cover - exercised only without io extra
        raise RuntimeError("local sparse NFW stencil construction requires healpy") from exc

    validate_mass_map(mass_map)
    validate_catalog_for_binning(catalog)

    pixels = np.asarray(mass_map.pixel, dtype=np.int64)
    n_pix = int(pixels.shape[0])
    pixel_to_row = {int(pixel): row for row, pixel in enumerate(pixels)}

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
    diagnostics = StencilBuildDiagnostics(query_mode=query_mode)
    inclusive = query_mode == "inclusive"
    t0 = perf_counter()

    for halo_id, (halo_vector, chi_i, rmax_i) in enumerate(
        zip(halo_unit_vectors, halo_chi, rmax, strict=True)
    ):
        diagnostics.n_halos += 1
        alpha_max = 2.0 * np.arcsin(min(1.0, float(rmax_i) / (2.0 * float(chi_i))))
        queried_pixels = np.asarray(
            hp.query_disc(
                mass_map.nside,
                halo_vector.astype(np.float64, copy=False),
                alpha_max,
                inclusive=inclusive,
                nest=False,
            ),
            dtype=np.int64,
        )
        diagnostics.n_query_pixels_total += int(queried_pixels.size)
        if queried_pixels.size == 0:
            continue
        diagnostics.n_halos_with_query_pixels += 1

        rows = np.array(
            [pixel_to_row.get(int(pixel), -1) for pixel in queried_pixels],
            dtype=np.int64,
        )
        inside_domain = rows >= 0
        n_inside = int(np.count_nonzero(inside_domain))
        diagnostics.n_inside_domain_total += n_inside
        if not np.any(inside_domain):
            continue
        diagnostics.n_halos_with_inside_pixels += 1

        local_pixels = queried_pixels[inside_domain]
        local_rows = rows[inside_domain]
        x, y, z = hp.pix2vec(mass_map.nside, local_pixels, nest=False)
        pixel_vectors = np.stack([x, y, z], axis=-1).astype(geometry_dtype, copy=False)
        cosang = np.clip(pixel_vectors @ halo_vector, -1.0, 1.0)
        chord = np.sqrt(np.maximum(2.0 * (1.0 - cosang), 0.0))
        r_perp = chi_i * chord
        keep = r_perp <= float(rmax_i)
        n_keep = int(np.count_nonzero(keep))
        diagnostics.n_kept_pairs_total += n_keep
        if not np.any(keep):
            continue
        diagnostics.n_halos_with_kept_pairs += 1

        pix_id_chunks.append(local_rows[keep])
        halo_id_chunks.append(np.full(n_keep, halo_id, dtype=np.int64))
        r_perp_chunks.append(r_perp[keep])
    diagnostics.elapsed_seconds = perf_counter() - t0

    if pix_id_chunks:
        pix_id = np.concatenate(pix_id_chunks)
        halo_id = np.concatenate(halo_id_chunks)
        r_perp = np.concatenate(r_perp_chunks)
    else:
        pix_id = np.empty((0,), dtype=np.int64)
        halo_id = np.empty((0,), dtype=np.int64)
        r_perp = np.empty((0,), dtype=np.float64)

    stencil = LightconeSparseStencil(
        pix_id=jnp.asarray(pix_id, dtype=jnp.int32),
        halo_id=jnp.asarray(halo_id, dtype=jnp.int32),
        r_perp=jnp.asarray(r_perp, dtype=jnp.asarray(catalog.chi).dtype),
        n_pix=n_pix,
    )
    if collect_diagnostics:
        if diagnostics.n_kept_pairs_total != int(pix_id.shape[0]):
            raise RuntimeError("stencil diagnostics kept-pair count does not match stencil size")
        return stencil, diagnostics
    return stencil


def stencil_diagnostics_to_dict(diag: StencilBuildDiagnostics) -> dict[str, float | int | str]:
    """Return NPZ-safe scalar diagnostics for a stencil build."""

    return {
        "stencil_query_mode": diag.query_mode,
        "stencil_query_pixels_total": int(diag.n_query_pixels_total),
        "stencil_inside_domain_total": int(diag.n_inside_domain_total),
        "stencil_kept_pairs_total": int(diag.n_kept_pairs_total),
        "stencil_inside_over_query": float(diag.inside_over_query),
        "stencil_kept_over_query": float(diag.kept_over_query),
        "stencil_kept_over_inside": float(diag.kept_over_inside),
        "stencil_build_seconds": float(diag.elapsed_seconds),
    }


def print_stencil_diagnostics(diag: StencilBuildDiagnostics) -> None:
    """Print HEALPix query diagnostics for one sparse-stencil build."""

    print("Sparse-stencil HEALPix query diagnostics:")
    print(f"  Query mode: {diag.query_mode}")
    print(f"  Halos processed: {diag.n_halos}")
    print(f"  Halos with query pixels: {diag.n_halos_with_query_pixels}")
    print(f"  Halos with compact-domain pixels: {diag.n_halos_with_inside_pixels}")
    print(f"  Halos with kept pairs: {diag.n_halos_with_kept_pairs}")
    print(f"  Queried HEALPix pixels: {diag.n_query_pixels_total}")
    print(f"  Inside compact domain: {diag.n_inside_domain_total}")
    print(f"  Kept halo-pixel pairs: {diag.n_kept_pairs_total}")
    print(f"  Inside/query fraction: {diag.inside_over_query:.6g}")
    print(f"  Kept/query fraction: {diag.kept_over_query:.6g}")
    print(f"  Kept/inside fraction: {diag.kept_over_inside:.6g}")
    print(f"  Stencil build time [s]: {diag.elapsed_seconds:.6g}")


def print_stencil_query_mode_comparison(
    comparison: dict[str, bool | float | int | str],
) -> None:
    """Print an inclusive-vs-center sparse-stencil comparison report."""

    print("Stencil query-mode comparison:")
    print(f"  inclusive stencil time [s]: {comparison['inclusive_stencil_seconds']:.6g}")
    print(f"  center stencil time [s]: {comparison['center_stencil_seconds']:.6g}")
    print(f"  inclusive queried pixels: {comparison['inclusive_query_pixels_total']}")
    print(f"  center queried pixels: {comparison['center_query_pixels_total']}")
    print(f"  inclusive kept pairs: {comparison['inclusive_kept_pairs_total']}")
    print(f"  center kept pairs: {comparison['center_kept_pairs_total']}")
    print(f"  max abs map difference: {comparison['max_abs_map_difference']:.12g}")
    print(f"  sum abs map difference: {comparison['sum_abs_map_difference']:.12g}")
    print(f"  relative sum difference: {comparison['relative_sum_difference']:.12g}")
    print(f"  differing pixels: {comparison['differing_pixels']}")
    print(f"  maps allclose: {comparison['maps_allclose']}")


def compare_stencil_query_modes(
    mass_map: PinocchioMassMap,
    selected_catalog: LightconeHaloCatalog,
    rmax_mpc_h: np.ndarray,
    metadata: PinocchioRunMetadata,
    particle_mass_msun_h: float,
    pixel_area_sr: float,
    concentration_params: ConcentrationParams,
    profile_params: NFWProfileParams,
    *,
    profile: bool = False,
) -> tuple[jax.Array, LightconeSparseStencil, dict[str, bool | float | int | str]]:
    """Build inclusive and center stencils, paint both maps, and compare them."""

    stencils: dict[str, LightconeSparseStencil] = {}
    diagnostics_by_mode: dict[str, StencilBuildDiagnostics] = {}
    maps: dict[str, jax.Array] = {}

    for query_mode in ("inclusive", "center"):
        with timed_stage(f"NFW local sparse stencil ({query_mode})", profile):
            stencil, diag = build_lightcone_sparse_stencil_for_mass_map_local(
                mass_map,
                selected_catalog,
                rmax_mpc_h,
                query_mode=query_mode,
                collect_diagnostics=True,
            )
        with timed_stage(f"NFW particle map ({query_mode})", profile):
            counts = paint_lightcone_particle_count_map_sparse(
                stencil,
                selected_catalog,
                particle_mass_msun_h=particle_mass_msun_h,
                pixel_area_sr=pixel_area_sr,
                cosmology=metadata.cosmology,
                concentration_params=concentration_params,
                profile_params=profile_params,
            )
            counts.block_until_ready()
        stencils[query_mode] = stencil
        diagnostics_by_mode[query_mode] = diag
        maps[query_mode] = counts

    inclusive_map = np.asarray(maps["inclusive"])
    center_map = np.asarray(maps["center"])
    abs_diff = np.abs(inclusive_map - center_map)
    inclusive_sum = float(np.sum(inclusive_map))
    center_sum = float(np.sum(center_map))
    relative_sum_difference = abs(inclusive_sum - center_sum) / max(abs(inclusive_sum), 1.0)
    inclusive_diag = diagnostics_by_mode["inclusive"]
    center_diag = diagnostics_by_mode["center"]

    comparison: dict[str, bool | float | int | str] = {
        "inclusive_stencil_seconds": float(inclusive_diag.elapsed_seconds),
        "center_stencil_seconds": float(center_diag.elapsed_seconds),
        "inclusive_query_pixels_total": int(inclusive_diag.n_query_pixels_total),
        "center_query_pixels_total": int(center_diag.n_query_pixels_total),
        "inclusive_inside_domain_total": int(inclusive_diag.n_inside_domain_total),
        "center_inside_domain_total": int(center_diag.n_inside_domain_total),
        "inclusive_kept_pairs_total": int(inclusive_diag.n_kept_pairs_total),
        "center_kept_pairs_total": int(center_diag.n_kept_pairs_total),
        "inclusive_inside_over_query": float(inclusive_diag.inside_over_query),
        "center_inside_over_query": float(center_diag.inside_over_query),
        "inclusive_kept_over_query": float(inclusive_diag.kept_over_query),
        "center_kept_over_query": float(center_diag.kept_over_query),
        "inclusive_kept_over_inside": float(inclusive_diag.kept_over_inside),
        "center_kept_over_inside": float(center_diag.kept_over_inside),
        "max_abs_map_difference": float(np.max(abs_diff)) if abs_diff.size else 0.0,
        "sum_abs_map_difference": float(np.sum(abs_diff)),
        "relative_sum_difference": float(relative_sum_difference),
        "differing_pixels": int(np.count_nonzero(inclusive_map != center_map)),
        "maps_allclose": bool(
            np.allclose(
                inclusive_map,
                center_map,
                rtol=1.0e-6,
                atol=1.0e-10,
            )
        ),
    }
    return maps["inclusive"], stencils["inclusive"], comparison


def nfw_sparse_total_particle_count(
    stencil: LightconeSparseStencil,
    catalog: LightconeHaloCatalog,
    particle_mass_msun_h: float,
    pixel_area_sr: float,
    cosmology: Any,
    concentration_params: ConcentrationParams,
    profile_params: NFWProfileParams,
):
    """Sum sparse NFW projected mass directly in PINOCCHIO particle-count units.

    The stencil contains fixed halo-pixel geometry with ``r_perp`` in comoving
    ``Mpc/h``. Halo masses are ``Msun/h`` and radial distances are comoving
    ``Mpc/h``. The returned scalar is ``sum(Sigma * chi**2 * pixel_area_sr)``
    divided by ``particle_mass_msun_h``. It is differentiable with respect to
    halo/profile quantities and profile/concentration parameters, while the
    retained sparse pair set remains fixed.
    """

    if particle_mass_msun_h <= 0.0:
        raise ValueError("particle_mass_msun_h must be positive")
    if pixel_area_sr <= 0.0:
        raise ValueError("pixel_area_sr must be positive")

    halo_id = jnp.asarray(stencil.halo_id, dtype=jnp.int32)
    mass = catalog.mass[halo_id]
    redshift = catalog.redshift[halo_id]
    chi = catalog.chi[halo_id]
    sigma = nfw_projected_surface_density(
        stencil.r_perp,
        mass,
        redshift,
        cosmology,
        concentration_params,
        profile_params,
    )
    return jnp.sum(sigma * (chi**2) * pixel_area_sr / particle_mass_msun_h)


def nfw_concentration_map_derivatives(
    stencil: LightconeSparseStencil,
    selected_catalog: LightconeHaloCatalog,
    mass_map: PinocchioMassMap,
    metadata: PinocchioRunMetadata,
    particle_mass_msun_h: float,
    concentration_amplitude: float,
    concentration_mass_slope: float,
    concentration_redshift_slope: float,
    concentration_mass_pivot: float,
    truncation_width_fraction: float,
    profile: bool = False,
) -> dict[str, float | str | np.ndarray]:
    """Return compact-map JVP derivatives with respect to concentration parameters.

    The sparse stencil geometry and retained pair set are fixed. Derivatives are
    taken only with respect to concentration amplitude, mass slope, and redshift
    slope; ``mass_pivot`` remains fixed.
    """

    if particle_mass_msun_h <= 0.0:
        raise ValueError("particle_mass_msun_h must be positive")
    if concentration_mass_pivot <= 0.0:
        raise ValueError("concentration_mass_pivot must be positive")

    pixel_area_sr = healpix_pixel_area_sr(mass_map.nside)
    theta = jnp.asarray(
        [
            concentration_amplitude,
            concentration_mass_slope,
            concentration_redshift_slope,
        ],
        dtype=selected_catalog.mass.dtype,
    )

    def paint_map_from_theta(theta):
        concentration_params = ConcentrationParams(
            amplitude=theta[0],
            mass_slope=theta[1],
            redshift_slope=theta[2],
            mass_pivot=concentration_mass_pivot,
        )
        profile_params = NFWProfileParams(
            truncation_width_fraction=truncation_width_fraction,
        )
        return paint_lightcone_particle_count_map_sparse(
            stencil,
            selected_catalog,
            particle_mass_msun_h=particle_mass_msun_h,
            pixel_area_sr=pixel_area_sr,
            cosmology=metadata.cosmology,
            concentration_params=concentration_params,
            profile_params=profile_params,
        )

    basis = jnp.eye(theta.shape[0], dtype=theta.dtype)
    with timed_stage("NFW map concentration JVPs", profile):
        dmaps = jax.vmap(
            lambda direction: jax.jvp(
                paint_map_from_theta,
                (theta,),
                (direction,),
            )[1]
        )(basis)
        dmaps.block_until_ready()

    d_amp = dmaps[0]
    d_mass_slope = dmaps[1]
    d_redshift_slope = dmaps[2]
    sum_d_amp = float(jnp.sum(d_amp))
    sum_d_mass_slope = float(jnp.sum(d_mass_slope))
    sum_d_redshift_slope = float(jnp.sum(d_redshift_slope))

    with timed_stage("NFW map concentration derivatives to numpy", profile):
        d_amp_np = np.asarray(d_amp)
        d_mass_slope_np = np.asarray(d_mass_slope)
        d_redshift_slope_np = np.asarray(d_redshift_slope)

    return {
        "nfw_map_derivatives": "concentration",
        "d_nfw_particle_counts_d_concentration_amplitude": d_amp_np,
        "d_nfw_particle_counts_d_concentration_mass_slope": d_mass_slope_np,
        "d_nfw_particle_counts_d_concentration_redshift_slope": d_redshift_slope_np,
        "sum_d_nfw_particle_counts_d_concentration_amplitude": sum_d_amp,
        "sum_d_nfw_particle_counts_d_concentration_mass_slope": sum_d_mass_slope,
        "sum_d_nfw_particle_counts_d_concentration_redshift_slope": sum_d_redshift_slope,
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
    stencil_query_mode: str = "inclusive",
    stencil_diagnostics: bool = False,
    stencil_compare_query_modes: bool = False,
    verbose: bool = True,
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
    if dense_demo and (stencil_diagnostics or stencil_compare_query_modes):
        raise ValueError("stencil diagnostics are only supported for sparse mode")
    if chunk_size is not None and chunk_size <= 0:
        chunk_size = None

    with timed_stage("NFW selected catalogue", profile):
        selected_catalog = selected_lightcone_catalog(catalog, mask)
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
        truncation_width_fraction=truncation_width_fraction
    )
    stencil = None
    stencil_diag = None
    comparison_diagnostics: dict[str, bool | float | int | str] = {}
    nfw_particle_counts = None
    sparse_pair_count = dense_pair_count
    if not dense_demo:
        with timed_stage("NFW rmax", profile):
            rmax = nfw_stencil_rmax_mpc_h(
                selected_catalog,
                metadata,
                concentration_params,
                profile_params,
                taper_radius_factor,
            )
        if stencil_compare_query_modes:
            nfw_particle_counts, stencil, comparison_diagnostics = compare_stencil_query_modes(
                mass_map,
                selected_catalog,
                rmax,
                metadata,
                particle_mass_msun_h,
                pixel_area_sr,
                concentration_params,
                profile_params,
                profile=profile,
            )
        else:
            with timed_stage("NFW local sparse stencil", profile):
                stencil_result = build_lightcone_sparse_stencil_for_mass_map_local(
                    mass_map,
                    selected_catalog,
                    rmax,
                    query_mode=stencil_query_mode,
                    collect_diagnostics=stencil_diagnostics,
                )
            if stencil_diagnostics:
                stencil, stencil_diag = stencil_result
            else:
                stencil = stencil_result
        sparse_pair_count = int(stencil.size)
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
            if stencil_diag is not None:
                print_stencil_diagnostics(stencil_diag)
            if comparison_diagnostics:
                print_stencil_query_mode_comparison(comparison_diagnostics)
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
    elif nfw_particle_counts is None:
        with timed_stage("NFW particle map", profile):
            assert stencil is not None
            nfw_particle_counts = paint_lightcone_particle_count_map_sparse(
                stencil,
                selected_catalog,
                particle_mass_msun_h=particle_mass_msun_h,
                pixel_area_sr=pixel_area_sr,
                cosmology=metadata.cosmology,
                concentration_params=concentration_params,
                profile_params=profile_params,
            )
            nfw_particle_counts.block_until_ready()
    else:
        nfw_particle_counts.block_until_ready()

    total_counts = jnp.sum(nfw_particle_counts)

    with timed_stage("NFW particle map to numpy", profile):
        nfw_particle_counts_np = np.asarray(nfw_particle_counts)

    map_derivative_diagnostics: dict[str, float | str | np.ndarray] = {
        "nfw_map_derivatives": "none"
    }
    if compute_map_derivatives:
        assert stencil is not None
        map_derivative_diagnostics = nfw_concentration_map_derivatives(
            stencil,
            selected_catalog,
            mass_map,
            metadata,
            particle_mass_msun_h,
            concentration_amplitude=concentration_amplitude,
            concentration_mass_slope=concentration_mass_slope,
            concentration_redshift_slope=concentration_redshift_slope,
            concentration_mass_pivot=concentration_mass_pivot,
            truncation_width_fraction=truncation_width_fraction,
            profile=profile,
        )

    diagnostics: dict[str, bool | float | int | str | np.ndarray] = {
        "pipeline_mode": pipeline_mode,
        "particle_mass_msun_h": float(particle_mass_msun_h),
        "nfw_particle_counts": nfw_particle_counts_np,
        "nfw_paint_mode": "dense" if dense_demo else "sparse",
        "nfw_selected_halo_count": int(selected_catalog.mass.shape[0]),
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
    }
    if stencil_diag is not None:
        diagnostics.update(stencil_diagnostics_to_dict(stencil_diag))
    if comparison_diagnostics:
        diagnostics.update(comparison_diagnostics)
        if stencil_diagnostics:
            diagnostics.update(
                {
                    "stencil_query_mode": "inclusive",
                    "stencil_query_pixels_total": int(
                        comparison_diagnostics["inclusive_query_pixels_total"]
                    ),
                    "stencil_inside_domain_total": int(
                        comparison_diagnostics["inclusive_inside_domain_total"]
                    ),
                    "stencil_kept_pairs_total": int(
                        comparison_diagnostics["inclusive_kept_pairs_total"]
                    ),
                    "stencil_inside_over_query": float(
                        comparison_diagnostics["inclusive_inside_over_query"]
                    ),
                    "stencil_kept_over_query": float(
                        comparison_diagnostics["inclusive_kept_over_query"]
                    ),
                    "stencil_kept_over_inside": float(
                        comparison_diagnostics["inclusive_kept_over_inside"]
                    ),
                    "stencil_build_seconds": float(
                        comparison_diagnostics["inclusive_stencil_seconds"]
                    ),
                }
            )
    diagnostics.update(map_derivative_diagnostics)
    return diagnostics


def diagnostics_for_map(
    catalog: LightconeHaloCatalog,
    mask: np.ndarray,
    mass_map: PinocchioMassMap,
    out: np.ndarray,
    particle_mass_msun_h: float,
    inside_pixel_domain: np.ndarray,
) -> dict[str, float | int]:
    """Return scalar diagnostics for printed and saved summaries."""

    return {
        "particle_mass_msun_h": float(particle_mass_msun_h),
        "n_halos_total": int(np.asarray(catalog.mass).shape[0]),
        "n_halos_in_segment": int(np.count_nonzero(mask)),
        "n_halos_in_segment_and_pixels": int(np.count_nonzero(inside_pixel_domain)),
        "sum_halo_particle_counts": float(np.sum(out)),
        "sum_pinocchio_mass_map_values": float(np.sum(mass_map.temperature)),
    }


def _resolve_output_path(output: Path | argparse.Namespace) -> Path:
    if isinstance(output, Path):
        return output
    return Path(output.output)


def save_npz(
    output: Path | argparse.Namespace,
    out: np.ndarray,
    mass_map: PinocchioMassMap,
    bounds: dict[str, float],
    metadata: PinocchioRunMetadata,
    diagnostics: dict[str, float | int],
    nfw_diagnostics: dict[str, bool | float | int | str | np.ndarray] | None = None,
) -> None:
    """Save the diagnostic map and metadata to the requested ``.npz`` file."""

    payload = {
        "halo_particle_counts": out,
        "pinocchio_mass_map_values": np.asarray(mass_map.temperature),
        "pixel": np.asarray(mass_map.pixel),
        "nside": int(mass_map.nside),
        "ordering": mass_map.ordering,
        "sheet_index": int(bounds["sheet_index"]),
        "z_lo": float(bounds["z_lo"]),
        "z_hi": float(bounds["z_hi"]),
        "a_lo": float(bounds["a_lo"]),
        "a_hi": float(bounds["a_hi"]),
        "chi_lo_mpc_h": float(bounds["chi_lo_mpc_h"]),
        "chi_hi_mpc_h": float(bounds["chi_hi_mpc_h"]),
        "particle_mass_msun_h": float(metadata.particle_mass_msun_h),
        "n_halos_total": int(diagnostics["n_halos_total"]),
        "n_halos_in_segment": int(diagnostics["n_halos_in_segment"]),
        "n_halos_in_segment_and_pixels": int(diagnostics["n_halos_in_segment_and_pixels"]),
        "sum_halo_particle_counts": float(diagnostics["sum_halo_particle_counts"]),
        "sum_pinocchio_mass_map_values": float(diagnostics["sum_pinocchio_mass_map_values"]),
    }
    if nfw_diagnostics is not None:
        payload.update(nfw_diagnostics)

    np.savez(_resolve_output_path(output), **payload)


def _pixel_column_format(pixels: np.ndarray) -> tuple[str, np.ndarray]:
    if pixels.size and (
        np.min(pixels) < np.iinfo(np.int32).min or np.max(pixels) > np.iinfo(np.int32).max
    ):
        return "1K", pixels.astype(np.int64)
    return "1J", pixels.astype(np.int32)


def write_output_fits(
    path: Path,
    out: np.ndarray,
    mass_map: PinocchioMassMap,
    bounds: dict[str, float],
    diagnostics: dict[str, float | int],
) -> None:
    """Write an optional HEALPix FITS binary table matching the compact domain."""

    try:
        from astropy.io import fits
    except ImportError as exc:  # pragma: no cover - exercised only without io extra
        raise RuntimeError("FITS output requires astropy; install geppetto[io]") from exc

    pixels = np.asarray(mass_map.pixel, dtype=np.int64)
    pixel_format, pixel_values = _pixel_column_format(pixels)
    table = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="PIXEL", format=pixel_format, array=pixel_values),
            fits.Column(name="TEMPERATURE", format="1D", array=out.astype(np.float64)),
        ],
        name="HEALPIX",
    )

    reserved_keys = {
        "XTENSION",
        "BITPIX",
        "NAXIS",
        "NAXIS1",
        "NAXIS2",
        "PCOUNT",
        "GCOUNT",
        "TFIELDS",
        "EXTNAME",
        "CHECKSUM",
        "DATASUM",
    }
    reserved_prefixes = ("TTYPE", "TFORM", "TUNIT", "TDIM", "TSCAL", "TZERO", "TNULL", "TDISP")
    for key, value in mass_map.header.items():
        key = str(key).upper()
        if key in reserved_keys or any(key.startswith(prefix) for prefix in reserved_prefixes):
            continue
        try:
            table.header[key] = value
        except (TypeError, ValueError):
            continue

    table.header["PIXTYPE"] = "HEALPIX"
    table.header["ORDERING"] = mass_map.ordering
    table.header["NSIDE"] = int(mass_map.nside)
    table.header["INDXSCHM"] = mass_map.index_scheme or "EXPLICIT"
    table.header["SHEETIDX"] = int(bounds["sheet_index"])
    table.header["ZLO"] = float(bounds["z_lo"])
    table.header["ZHI"] = float(bounds["z_hi"])
    table.header["ALO"] = float(bounds["a_lo"])
    table.header["AHI"] = float(bounds["a_hi"])
    table.header["CHILO"] = float(bounds["chi_lo_mpc_h"])
    table.header["CHIHI"] = float(bounds["chi_hi_mpc_h"])
    table.header["PMASS"] = float(diagnostics["particle_mass_msun_h"])
    table.header["NHALSEG"] = int(diagnostics["n_halos_in_segment"])
    table.header["NHALPIX"] = int(diagnostics["n_halos_in_segment_and_pixels"])
    table.header["MAPTYPE"] = "HALO_PARTICLE_COUNT"
    table.header["COMMENT"] = "Diagnostic halo-catalog particle-count map."
    table.header["COMMENT"] = "TEMPERATURE is halo mass / PINOCCHIO particle mass."
    table.header["COMMENT"] = "This is not the original PINOCCHIO on-the-fly mass map."
    fits.HDUList([fits.PrimaryHDU(), table]).writeto(path, overwrite=True)


def write_nfw_painted_fits(
    path: Path,
    nfw_particle_counts: np.ndarray,
    mass_map: PinocchioMassMap,
    bounds: dict[str, float],
    nfw_diagnostics: dict[str, bool | float | int | str | np.ndarray],
) -> None:
    """Write a compact HEALPix FITS table for the painted NFW map."""

    try:
        from astropy.io import fits
    except ImportError as exc:  # pragma: no cover - exercised only without io extra
        raise RuntimeError("FITS output requires astropy; install geppetto[io]") from exc

    pixels = np.asarray(mass_map.pixel, dtype=np.int64)
    nfw_particle_counts = np.asarray(nfw_particle_counts, dtype=np.float64)
    if nfw_particle_counts.shape != pixels.shape:
        raise RuntimeError("NFW map shape does not match mass_map.pixel")

    pixel_format, pixel_values = _pixel_column_format(pixels)
    table = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="PIXEL", format=pixel_format, array=pixel_values),
            fits.Column(
                name="TEMPERATURE",
                format="1D",
                array=nfw_particle_counts.astype(np.float64),
            ),
        ],
        name="HEALPIX",
    )

    reserved_keys = {
        "XTENSION",
        "BITPIX",
        "NAXIS",
        "NAXIS1",
        "NAXIS2",
        "PCOUNT",
        "GCOUNT",
        "TFIELDS",
        "EXTNAME",
        "CHECKSUM",
        "DATASUM",
    }
    reserved_prefixes = ("TTYPE", "TFORM", "TUNIT", "TDIM", "TSCAL", "TZERO", "TNULL", "TDISP")
    for key, value in mass_map.header.items():
        key = str(key).upper()
        if key in reserved_keys or any(key.startswith(prefix) for prefix in reserved_prefixes):
            continue
        try:
            table.header[key] = value
        except (TypeError, ValueError):
            continue

    table.header["PIXTYPE"] = "HEALPIX"
    table.header["ORDERING"] = mass_map.ordering
    table.header["NSIDE"] = int(mass_map.nside)
    table.header["INDXSCHM"] = mass_map.index_scheme or "EXPLICIT"
    table.header["SHEETIDX"] = int(bounds["sheet_index"])
    table.header["ZLO"] = float(bounds["z_lo"])
    table.header["ZHI"] = float(bounds["z_hi"])
    table.header["ALO"] = float(bounds["a_lo"])
    table.header["AHI"] = float(bounds["a_hi"])
    table.header["CHILO"] = float(bounds["chi_lo_mpc_h"])
    table.header["CHIHI"] = float(bounds["chi_hi_mpc_h"])
    table.header["PMASS"] = float(nfw_diagnostics["particle_mass_msun_h"])
    table.header["MAPTYPE"] = "NFW_PARTICLE_COUNT"
    table.header["CONCAMP"] = float(nfw_diagnostics["nfw_concentration_amplitude"])
    table.header["CONCMSLP"] = float(nfw_diagnostics["nfw_concentration_mass_slope"])
    table.header["CONCZSLP"] = float(nfw_diagnostics["nfw_concentration_redshift_slope"])
    table.header["CONCPIV"] = float(nfw_diagnostics["nfw_concentration_mass_pivot"])
    table.header["TRUNCW"] = float(nfw_diagnostics["nfw_truncation_width_fraction"])
    table.header["COMMENT"] = "TEMPERATURE contains painted NFW particle-count-equivalent values."
    table.header["COMMENT"] = (
        "Compact pixel domain matches the corresponding PINOCCHIO mass-map segment."
    )
    fits.HDUList([fits.PrimaryHDU(), table]).writeto(path, overwrite=True)


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    """Write an all-segments CSV manifest."""

    base_columns = [
        "segment_index",
        "mass_map_path",
        "output_npz",
        "output_fits",
        "z_lo",
        "z_hi",
        "chi_lo_mpc_h",
        "chi_hi_mpc_h",
        "inclusive_upper",
        "n_halos_in_segment",
        "n_halos_in_segment_and_pixels",
        "nfw_selected_halo_count",
        "nfw_compact_pixel_count",
        "nfw_sparse_pair_count",
        "nfw_sum_particle_counts",
        "nfw_map_derivatives",
    ]
    derivative_columns = [
        "sum_d_nfw_particle_counts_d_concentration_amplitude",
        "sum_d_nfw_particle_counts_d_concentration_mass_slope",
        "sum_d_nfw_particle_counts_d_concentration_redshift_slope",
    ]
    columns = list(base_columns)
    if any(any(column in row for column in derivative_columns) for row in rows):
        columns.extend(derivative_columns)

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


def print_output_summary(
    diagnostics: dict[str, float | int],
    out: np.ndarray,
    mass_map: PinocchioMassMap,
) -> None:
    """Print the final comparison summary."""

    print("Halo particle-count map summary:")
    print(f"  Read {diagnostics['n_halos_total']} total halos")
    print(f"  Selected {diagnostics['n_halos_in_segment']} halos in segment bounds")
    print(f"  Kept {diagnostics['n_halos_in_segment_and_pixels']} halos inside map pixel domain")
    print(f"  Output pixels: {len(out)}")
    print(f"  PINOCCHIO map pixels: {len(mass_map.temperature)}")
    print(f"  Sum halo particle counts: {diagnostics['sum_halo_particle_counts']:.12g}")
    print(
        "  Sum PINOCCHIO on-the-fly map values: "
        f"{diagnostics['sum_pinocchio_mass_map_values']:.12g}"
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
    print(f"  Compact pixels: {nfw_diagnostics['nfw_compact_pixel_count']}")
    print(f"  Sparse halo-pixel pairs: {nfw_diagnostics['nfw_sparse_pair_count']}")
    print(f"  Dense pair count: {nfw_diagnostics['nfw_dense_pair_count']}")
    print(
        "  Sparse compression factor: "
        f"{nfw_diagnostics['nfw_sparse_compression_factor']:.12g}"
    )
    print(f"  NFW sum particle counts: {nfw_diagnostics['nfw_sum_particle_counts']:.12g}")
    if nfw_diagnostics.get("nfw_map_derivatives", "none") == "concentration":
        print("  Map derivatives: concentration")
        print(
            "  Sum d(map)/d concentration amplitude: "
            f"{nfw_diagnostics['sum_d_nfw_particle_counts_d_concentration_amplitude']:.12g}"
        )
        print(
            "  Sum d(map)/d concentration mass slope: "
            f"{nfw_diagnostics['sum_d_nfw_particle_counts_d_concentration_mass_slope']:.12g}"
        )
        print(
            "  Sum d(map)/d concentration redshift slope: "
            f"{nfw_diagnostics['sum_d_nfw_particle_counts_d_concentration_redshift_slope']:.12g}"
        )


def compute_calibration_for_segment(
    *,
    segment_index: int,
    mass_map_path: Path,
    output_npz: Path,
    output_fits: Path | None,
    catalog: LightconeHaloCatalog,
    sheets: Any,
    metadata: PinocchioRunMetadata,
    particle_mass: float,
    args: argparse.Namespace,
    profile: bool,
    compute_map_derivatives: bool,
    inclusive_upper: bool,
    verbose: bool = True,
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

    with timed_stage("select segment mask", stage_profile):
        mask = select_segment_mask(
            catalog,
            bounds,
            mode=args.bounds,
            inclusive_upper=inclusive_upper,
        )

    if verbose:
        print_segment_summary(bounds, inclusive_upper)
    with timed_stage("point-halo rows", stage_profile):
        rows, inside_pixel_domain = halo_rows_in_mass_map(catalog, mask, mass_map)
    with timed_stage("point-halo accumulation", stage_profile):
        out = _accumulate_halo_particle_counts(
            catalog,
            mask,
            mass_map,
            particle_mass,
            rows,
            inside_pixel_domain,
        )

    if out.shape != np.asarray(mass_map.temperature).shape:
        raise RuntimeError("output map shape does not match mass_map.temperature")
    if len(out) != len(mass_map.pixel):
        raise RuntimeError("output map length does not match mass_map.pixel")

    with timed_stage("point-halo diagnostics", stage_profile):
        diagnostics = diagnostics_for_map(
            catalog,
            mask,
            mass_map,
            out,
            particle_mass,
            inside_pixel_domain,
        )
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
            stencil_query_mode=args.stencil_query_mode,
            stencil_diagnostics=args.stencil_diagnostics,
            stencil_compare_query_modes=args.stencil_compare_query_modes,
            verbose=verbose,
        )

    return CalibrationSegmentResult(
        segment_index=int(segment_index),
        mass_map_path=Path(mass_map_path),
        output_npz=Path(output_npz),
        output_fits=None if output_fits is None else Path(output_fits),
        bounds=bounds,
        inclusive_upper=bool(inclusive_upper),
        mass_map=mass_map,
        halo_particle_counts=out,
        diagnostics=diagnostics,
        nfw_diagnostics=nfw_diagnostics,
    )


def calibration_manifest_row(result: CalibrationSegmentResult) -> dict[str, object]:
    """Return the manifest row for a computed or MPI-reduced segment."""

    diagnostics = result.diagnostics
    nfw_diagnostics = result.nfw_diagnostics
    row: dict[str, object] = {
        "segment_index": int(result.segment_index),
        "mass_map_path": str(result.mass_map_path),
        "output_npz": str(result.output_npz),
        "output_fits": "" if result.output_fits is None else str(result.output_fits),
        "z_lo": float(result.bounds["z_lo"]),
        "z_hi": float(result.bounds["z_hi"]),
        "chi_lo_mpc_h": float(result.bounds["chi_lo_mpc_h"]),
        "chi_hi_mpc_h": float(result.bounds["chi_hi_mpc_h"]),
        "inclusive_upper": bool(result.inclusive_upper),
        "n_halos_in_segment": int(diagnostics["n_halos_in_segment"]),
        "n_halos_in_segment_and_pixels": int(diagnostics["n_halos_in_segment_and_pixels"]),
        "nfw_selected_halo_count": int(nfw_diagnostics["nfw_selected_halo_count"]),
        "nfw_compact_pixel_count": int(nfw_diagnostics["nfw_compact_pixel_count"]),
        "nfw_sparse_pair_count": int(nfw_diagnostics["nfw_sparse_pair_count"]),
        "nfw_sum_particle_counts": float(nfw_diagnostics["nfw_sum_particle_counts"]),
        "nfw_map_derivatives": str(nfw_diagnostics["nfw_map_derivatives"]),
    }
    for key in (
        "sum_d_nfw_particle_counts_d_concentration_amplitude",
        "sum_d_nfw_particle_counts_d_concentration_mass_slope",
        "sum_d_nfw_particle_counts_d_concentration_redshift_slope",
    ):
        if key in nfw_diagnostics:
            row[key] = float(nfw_diagnostics[key])
    return row


def write_calibration_segment_outputs(
    result: CalibrationSegmentResult,
    metadata: PinocchioRunMetadata,
    *,
    profile: bool,
    verbose: bool = True,
) -> dict[str, object]:
    """Write one computed segment payload and return its manifest row."""

    with timed_stage("save NPZ", profile):
        save_npz(
            result.output_npz,
            result.halo_particle_counts,
            result.mass_map,
            result.bounds,
            metadata,
            result.diagnostics,
            result.nfw_diagnostics,
        )
    if result.output_fits is not None:
        with timed_stage("write NFW FITS", profile):
            write_nfw_painted_fits(
                result.output_fits,
                np.asarray(result.nfw_diagnostics["nfw_particle_counts"]),
                result.mass_map,
                result.bounds,
                result.nfw_diagnostics,
            )

    if verbose:
        print_output_summary(result.diagnostics, result.halo_particle_counts, result.mass_map)
        print_nfw_calibration_summary(result.nfw_diagnostics)
        print(f"Wrote NPZ: {result.output_npz}")
        if result.output_fits is not None:
            print(f"Wrote NFW FITS: {result.output_fits}")

    return calibration_manifest_row(result)


def run_calibration_for_segment(
    *,
    segment_index: int,
    mass_map_path: Path,
    output_npz: Path,
    output_fits: Path | None,
    catalog: LightconeHaloCatalog,
    sheets: Any,
    metadata: PinocchioRunMetadata,
    particle_mass: float,
    args: argparse.Namespace,
    profile: bool,
    compute_map_derivatives: bool,
    inclusive_upper: bool,
) -> dict[str, object]:
    """Run the complete NFW calibration pipeline for one mass-map segment."""

    result = compute_calibration_for_segment(
        segment_index=segment_index,
        mass_map_path=mass_map_path,
        output_npz=output_npz,
        output_fits=output_fits,
        catalog=catalog,
        sheets=sheets,
        metadata=metadata,
        particle_mass=particle_mass,
        args=args,
        profile=profile,
        compute_map_derivatives=compute_map_derivatives,
        inclusive_upper=inclusive_upper,
        verbose=True,
    )
    return write_calibration_segment_outputs(
        result,
        metadata,
        profile=profile,
        verbose=True,
    )


def _mpi_sum(value: Any, mpi_context: MpiContext) -> Any:
    if not mpi_context.enabled:
        return value
    if mpi_context.comm is None:
        raise RuntimeError("MPI context is enabled but has no communicator")
    if mpi_context.sum_op is None:
        return mpi_context.comm.reduce(value, root=0)
    return mpi_context.comm.reduce(value, op=mpi_context.sum_op, root=0)


def _as_reduced_number(value: Any, *, integer: bool) -> int | float:
    array = np.asarray(value)
    scalar = array.item() if array.shape == () else value
    if integer:
        return int(scalar)
    return float(scalar)


def reduce_calibration_segment_result(
    local_result: CalibrationSegmentResult,
    mpi_context: MpiContext,
) -> CalibrationSegmentResult | None:
    """Sum rank-local segment maps and additive diagnostics onto rank 0."""

    if not mpi_context.enabled:
        return local_result

    reduced_halo_counts = _mpi_sum(local_result.halo_particle_counts, mpi_context)
    reduced_nfw_counts = _mpi_sum(
        np.asarray(local_result.nfw_diagnostics["nfw_particle_counts"]),
        mpi_context,
    )

    derivative_array_keys = (
        "d_nfw_particle_counts_d_concentration_amplitude",
        "d_nfw_particle_counts_d_concentration_mass_slope",
        "d_nfw_particle_counts_d_concentration_redshift_slope",
    )
    reduced_derivative_arrays = {
        key: _mpi_sum(np.asarray(local_result.nfw_diagnostics[key]), mpi_context)
        for key in derivative_array_keys
        if key in local_result.nfw_diagnostics
    }

    diagnostic_sum_keys = (
        "n_halos_total",
        "n_halos_in_segment",
        "n_halos_in_segment_and_pixels",
        "sum_halo_particle_counts",
    )
    reduced_diagnostics = {
        key: _mpi_sum(local_result.diagnostics[key], mpi_context)
        for key in diagnostic_sum_keys
        if key in local_result.diagnostics
    }

    nfw_sum_keys = (
        "nfw_selected_halo_count",
        "nfw_sparse_pair_count",
        "nfw_dense_pair_count",
        "nfw_sum_particle_counts",
        "sum_d_nfw_particle_counts_d_concentration_amplitude",
        "sum_d_nfw_particle_counts_d_concentration_mass_slope",
        "sum_d_nfw_particle_counts_d_concentration_redshift_slope",
        "stencil_query_pixels_total",
        "stencil_inside_domain_total",
        "stencil_kept_pairs_total",
    )
    reduced_nfw_scalars = {
        key: _mpi_sum(local_result.nfw_diagnostics[key], mpi_context)
        for key in nfw_sum_keys
        if key in local_result.nfw_diagnostics
    }

    if not mpi_context.is_root:
        return None

    diagnostics = dict(local_result.diagnostics)
    for key, value in reduced_diagnostics.items():
        diagnostics[key] = _as_reduced_number(
            value,
            integer=key.startswith("n_halos"),
        )
    diagnostics["sum_halo_particle_counts"] = float(np.sum(reduced_halo_counts))

    nfw_diagnostics = dict(local_result.nfw_diagnostics)
    nfw_diagnostics["nfw_particle_counts"] = np.asarray(reduced_nfw_counts)
    for key, value in reduced_derivative_arrays.items():
        nfw_diagnostics[key] = np.asarray(value)
    integer_nfw_keys = {
        "nfw_selected_halo_count",
        "nfw_sparse_pair_count",
        "nfw_dense_pair_count",
        "stencil_query_pixels_total",
        "stencil_inside_domain_total",
        "stencil_kept_pairs_total",
    }
    for key, value in reduced_nfw_scalars.items():
        nfw_diagnostics[key] = _as_reduced_number(
            value,
            integer=key in integer_nfw_keys,
        )
    nfw_diagnostics["nfw_sum_particle_counts"] = float(
        np.sum(nfw_diagnostics["nfw_particle_counts"])
    )
    nfw_diagnostics["nfw_sparse_compression_factor"] = _compression_factor(
        int(nfw_diagnostics["nfw_dense_pair_count"]),
        int(nfw_diagnostics["nfw_sparse_pair_count"]),
    )
    if {
        "stencil_query_pixels_total",
        "stencil_inside_domain_total",
        "stencil_kept_pairs_total",
    }.issubset(nfw_diagnostics):
        query = int(nfw_diagnostics["stencil_query_pixels_total"])
        inside = int(nfw_diagnostics["stencil_inside_domain_total"])
        kept = int(nfw_diagnostics["stencil_kept_pairs_total"])
        nfw_diagnostics["stencil_inside_over_query"] = inside / query if query else 0.0
        nfw_diagnostics["stencil_kept_over_query"] = kept / query if query else 0.0
        nfw_diagnostics["stencil_kept_over_inside"] = kept / inside if inside else 0.0

    return CalibrationSegmentResult(
        segment_index=local_result.segment_index,
        mass_map_path=local_result.mass_map_path,
        output_npz=local_result.output_npz,
        output_fits=local_result.output_fits,
        bounds=local_result.bounds,
        inclusive_upper=local_result.inclusive_upper,
        mass_map=local_result.mass_map,
        halo_particle_counts=np.asarray(reduced_halo_counts),
        diagnostics=diagnostics,
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
) -> list[dict[str, object]]:
    """Run either the single-segment or all-segments workflow."""

    if mpi_context is None:
        mpi_context = MpiContext()
    segment_workers = int(getattr(args, "segment_workers", 1))
    if segment_workers < 1:
        raise ValueError("--segment-workers must be at least 1")
    mpi_output_mode = getattr(args, "mpi_output_mode", "reduce")
    rank_local_outputs = mpi_context.enabled and mpi_output_mode == "rank-local"

    if workflow == "single":
        segments = [(int(args.sheet_index), Path(args.mass_map))]
        output_specs = [(Path(args.output), args.output_fits)]
        inclusive_values = [bool(args.last_segment_inclusive)]
    elif workflow == "all":
        segments = discover_mass_map_segments(str(args.mass_map_glob))
        output_dir = Path(args.output_dir)
        if mpi_context.is_root or rank_local_outputs:
            output_dir.mkdir(parents=True, exist_ok=True)
        last_segment_index = max(segment_index for segment_index, _ in segments)
        output_specs = []
        inclusive_values = []
        for segment_index, _ in segments:
            paths = segment_output_paths(output_dir, segment_index)
            output_specs.append((paths["npz"], paths["fits"]))
            inclusive_values.append(segment_index == last_segment_index)
    else:
        raise ValueError("workflow must be 'single' or 'all'")

    if rank_local_outputs:
        output_specs = rank_local_output_specs(output_specs, mpi_context.rank)

    segment_specs = list(
        zip(
            segments,
            output_specs,
            inclusive_values,
            strict=True,
        )
    )

    manifest_rows = []
    if not mpi_context.enabled and segment_workers == 1:
        for (segment_index, mass_map_path), (output_npz, output_fits), inclusive_upper in segment_specs:
            manifest_rows.append(
                run_calibration_for_segment(
                    segment_index=segment_index,
                    mass_map_path=mass_map_path,
                    output_npz=output_npz,
                    output_fits=output_fits,
                    catalog=catalog,
                    sheets=sheets,
                    metadata=metadata,
                    particle_mass=particle_mass,
                    args=args,
                    profile=profile,
                    compute_map_derivatives=compute_map_derivatives,
                    inclusive_upper=inclusive_upper,
                )
            )
    else:

        def compute_one(spec):
            (segment_index, mass_map_path), (output_npz, output_fits), inclusive_upper = spec
            return compute_calibration_for_segment(
                segment_index=segment_index,
                mass_map_path=mass_map_path,
                output_npz=output_npz,
                output_fits=output_fits,
                catalog=catalog,
                sheets=sheets,
                metadata=metadata,
                particle_mass=particle_mass,
                args=args,
                profile=profile,
                compute_map_derivatives=compute_map_derivatives,
                inclusive_upper=inclusive_upper,
                verbose=False,
            )

        if segment_workers == 1:
            local_results = [compute_one(spec) for spec in segment_specs]
        else:
            with ThreadPoolExecutor(max_workers=segment_workers) as executor:
                local_results = list(executor.map(compute_one, segment_specs))

        for local_result in sorted(local_results, key=lambda result: result.segment_index):
            if rank_local_outputs:
                manifest_rows.append(
                    write_calibration_segment_outputs(
                        local_result,
                        metadata,
                        profile=profile,
                        verbose=mpi_context.is_root,
                    )
                )
                continue
            reduced_result = reduce_calibration_segment_result(local_result, mpi_context)
            if reduced_result is None:
                continue
            manifest_rows.append(
                write_calibration_segment_outputs(
                    reduced_result,
                    metadata,
                    profile=profile,
                    verbose=mpi_context.is_root,
                )
            )

    if workflow == "all" and rank_local_outputs:
        manifest_path = rank_local_manifest_path(Path(args.output_dir), mpi_context.rank)
        with timed_stage("write manifest", profile and mpi_context.is_root):
            write_manifest(manifest_path, manifest_rows)
        if mpi_context.is_root:
            print(f"Wrote manifest: {manifest_path}")
    elif workflow == "all" and mpi_context.is_root:
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
    )


if __name__ == "__main__":
    main()
