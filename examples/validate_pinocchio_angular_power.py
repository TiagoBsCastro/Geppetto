#!/usr/bin/env python3
"""Compare PINOCCHIO+GEPPETTO shell maps with linear-plus-one-halo theory.

The original PINOCCHIO FITS map supplies compact RING pixel IDs and
uncollapsed-particle counts. The lean GEPPETTO NPZ supplies painted one-halo
counts in exactly the same row order. This script sums those components,
estimates cut-sky pseudo-C_ell/f_sky spectra, and writes the matching theory
components without duplicating map arrays.
"""

from __future__ import annotations

import argparse
import csv
import gc
import glob
import hashlib
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from geppetto.concentration import ConcentrationParams
from geppetto.cosmology import Cosmology
from geppetto.io import (
    PinocchioCatalogError,
    healpix_pixel_area_sr,
    read_pinocchio_cosmology_table,
    read_pinocchio_mass_function_series,
    read_pinocchio_mass_map_fits,
    read_pinocchio_parameter_file,
)
from geppetto.profiles import NFWProfileParams
from geppetto.theory import (
    LinearTheoryTable,
    comoving_distance_mpc_h,
    exact_linear_shell_cls,
    hybrid_angular_power_spectra,
    linear_matter_power,
    one_halo_matter_power,
    resolved_halo_mass_fraction,
    sigma8_from_linear_power,
)

VALIDATION_SCHEMA_VERSION = 2
EXACT_CHECKPOINT_SCHEMA_VERSION = 1
THEORY_COMPONENTS = ("linear", "one_halo", "particle_shot_noise")


@dataclass(frozen=True)
class MaskCoupling:
    """NaMaster objects defining the validation pseudo-spectrum convention."""

    module: Any
    mask: np.ndarray
    template: np.ndarray
    reference_field: Any
    workspace: Any
    ell_full: np.ndarray
    f_sky: float
    nside: int
    n_iter: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate painted PINOCCHIO angular maps against halo-model C_ell theory"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--cosmology-table", type=Path, required=True)
    parser.add_argument(
        "--hmf-glob",
        required=True,
        help="Glob matching PINOCCHIO *.mf.out files at every shell boundary",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ell-max", type=int, default=None)
    parser.add_argument("--ell-min-compare", type=int, default=20)
    parser.add_argument("--ell-bin-width", type=int, default=20)
    parser.add_argument("--ell-exact-cap", type=int, default=512)
    parser.add_argument("--limber-match-rtol", type=float, default=0.01)
    parser.add_argument("--limber-match-width", type=int, default=20)
    parser.add_argument("--exact-batch-size", type=int, default=20)
    parser.add_argument("--exact-workers", type=int, default=1)
    parser.add_argument("--radial-order", type=int, default=64)
    parser.add_argument("--profile-order", type=int, default=64)
    parser.add_argument("--exact-relative-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--sigma8-rtol", type=float, default=0.01)
    parser.add_argument("--mask-sht-iterations", type=int, default=3)
    parser.add_argument(
        "--jax-precision",
        choices=("float32", "float64"),
        default="float64",
    )
    return parser.parse_args()


def exact_checkpoint_fingerprint(
    z_lo: np.ndarray,
    z_hi: np.ndarray,
    shell_weights: np.ndarray,
    linear_theory: LinearTheoryTable,
    *,
    radial_order: int,
    relative_tolerance: float,
) -> str:
    """Return a deterministic fingerprint for exact-projection inputs."""

    digest = hashlib.sha256()
    digest.update(f"schema={EXACT_CHECKPOINT_SCHEMA_VERSION}".encode())
    digest.update(f"radial_order={radial_order}".encode())
    digest.update(f"relative_tolerance={relative_tolerance:.17g}".encode())
    for value in (
        z_lo,
        z_hi,
        shell_weights,
        np.asarray(linear_theory.h),
        np.asarray(linear_theory.omega_m0),
        linear_theory.scale_factor,
        linear_theory.chi_mpc_h,
        linear_theory.growth,
        linear_theory.k_h_mpc,
        linear_theory.power_mpc_h3,
    ):
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(array.dtype.str.encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.view(np.uint8))
    return digest.hexdigest()


def exact_batch_with_checkpoint(
    checkpoint_path: Path,
    fingerprint: str,
    ell: np.ndarray,
    n_shell: int,
    compute: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Load or compute one exact batch and atomically extend its checkpoint.

    ``compute`` receives only missing multipoles. The boolean result reports
    whether the complete requested batch came from the checkpoint.
    """

    requested = np.asarray(ell, dtype=np.int64)
    if requested.ndim != 1 or requested.size == 0 or np.unique(requested).size != requested.size:
        raise ValueError("exact checkpoint requests must contain unique multipoles")
    cached_ell = np.empty(0, dtype=np.int64)
    cached_shell = np.empty((n_shell, 0), dtype=np.float64)
    cached_sum = np.empty(0, dtype=np.float64)
    if checkpoint_path.exists():
        try:
            with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
                stored_fingerprint = str(checkpoint["fingerprint"].item())
                stored_schema = int(checkpoint["schema_version"])
                if (
                    stored_schema == EXACT_CHECKPOINT_SCHEMA_VERSION
                    and stored_fingerprint == fingerprint
                ):
                    cached_ell = np.asarray(checkpoint["ell"], dtype=np.int64)
                    cached_shell = np.asarray(checkpoint["shell_linear"], dtype=np.float64)
                    cached_sum = np.asarray(checkpoint["summed_linear"], dtype=np.float64)
        except (OSError, KeyError, ValueError) as exc:
            raise ValueError(f"cannot read exact-projection checkpoint: {checkpoint_path}") from exc
    if cached_shell.shape != (n_shell, cached_ell.size) or cached_sum.shape != cached_ell.shape:
        raise ValueError(f"exact-projection checkpoint has inconsistent shapes: {checkpoint_path}")
    if cached_ell.size and (
        np.unique(cached_ell).size != cached_ell.size
        or not np.all(np.isfinite(cached_shell))
        or not np.all(np.isfinite(cached_sum))
    ):
        raise ValueError(f"exact-projection checkpoint contains invalid values: {checkpoint_path}")

    cached_lookup = {int(value): index for index, value in enumerate(cached_ell)}
    missing = np.asarray(
        [value for value in requested if int(value) not in cached_lookup], dtype=np.int64
    )
    cache_hit = missing.size == 0
    if missing.size:
        missing_shell, missing_sum = compute(missing)
        missing_shell = np.asarray(missing_shell, dtype=np.float64)
        missing_sum = np.asarray(missing_sum, dtype=np.float64)
        if missing_shell.shape != (n_shell, missing.size) or missing_sum.shape != missing.shape:
            raise ValueError("exact-projection compute callback returned inconsistent shapes")
        if not np.all(np.isfinite(missing_shell)) or not np.all(np.isfinite(missing_sum)):
            raise ValueError("exact-projection compute callback returned non-finite values")
        combined_ell = np.concatenate((cached_ell, missing))
        combined_shell = np.concatenate((cached_shell, missing_shell), axis=1)
        combined_sum = np.concatenate((cached_sum, missing_sum))
        order = np.argsort(combined_ell)
        cached_ell = combined_ell[order]
        cached_shell = combined_shell[:, order]
        cached_sum = combined_sum[order]
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=checkpoint_path.parent,
                prefix=f".{checkpoint_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                np.savez_compressed(
                    handle,
                    schema_version=np.asarray(EXACT_CHECKPOINT_SCHEMA_VERSION, dtype=np.int64),
                    fingerprint=np.asarray(fingerprint),
                    ell=cached_ell,
                    shell_linear=cached_shell,
                    summed_linear=cached_sum,
                )
            os.replace(temporary_path, checkpoint_path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        cached_lookup = {int(value): index for index, value in enumerate(cached_ell)}

    requested_indices = np.asarray([cached_lookup[int(value)] for value in requested], dtype=np.int64)
    return cached_shell[:, requested_indices], cached_sum[requested_indices], cache_hit


def load_manifest(path: Path) -> list[dict[str, str]]:
    """Load and sort painting-manifest rows by segment index."""

    try:
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise ValueError(f"cannot read painting manifest: {path}") from exc
    if not rows:
        raise ValueError(f"painting manifest is empty: {path}")
    required = {
        "segment_index",
        "mass_map_path",
        "output_npz",
        "z_lo",
        "z_hi",
        "nfw_concentration_amplitude",
        "nfw_concentration_mass_slope",
        "nfw_concentration_redshift_slope",
        "nfw_concentration_mass_pivot_msun_h",
        "nfw_overdensity_mode",
        "nfw_overdensity",
        "nfw_reference_density",
        "theta_resolution_rad",
    }
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"painting manifest is missing columns: {sorted(missing)}")
    try:
        rows.sort(key=lambda row: int(row["segment_index"]))
    except ValueError as exc:
        raise ValueError("manifest segment_index values must be integers") from exc
    if "nfw_profile_support" in rows[0]:
        support = {row["nfw_profile_support"].strip() for row in rows}
        if support != {"hard_3d_r_delta_los_projection"}:
            raise ValueError(
                "theory validation requires hard_3d_r_delta_los_projection maps"
            )
    if "nfw_paint_mode" in rows[0]:
        paint_mode = {row["nfw_paint_mode"].strip() for row in rows}
        if paint_mode != {"adaptive_global_support"}:
            raise ValueError("theory validation requires adaptive_global_support maps")
    return rows


def _resolve_input_path(value: str, manifest_path: Path) -> Path:
    candidate = Path(value)
    if candidate.exists() or candidate.is_absolute():
        return candidate
    relative = manifest_path.parent / candidate
    return relative if relative.exists() else candidate


def _consistent_float(rows: list[dict[str, str]], key: str) -> float:
    try:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    except (KeyError, ValueError) as exc:
        raise ValueError(f"manifest column {key} must contain numeric values") from exc
    if not np.allclose(values, values[0], rtol=1.0e-12, atol=0.0):
        raise ValueError(f"manifest column {key} must be consistent across segments")
    return float(values[0])


def concentration_from_manifest(rows: list[dict[str, str]]) -> ConcentrationParams:
    """Return the common concentration relation recorded by the painter."""

    return ConcentrationParams(
        amplitude=_consistent_float(rows, "nfw_concentration_amplitude"),
        mass_slope=_consistent_float(rows, "nfw_concentration_mass_slope"),
        redshift_slope=_consistent_float(rows, "nfw_concentration_redshift_slope"),
        mass_pivot=_consistent_float(rows, "nfw_concentration_mass_pivot_msun_h"),
    )


def profiles_from_manifest(rows: list[dict[str, str]]) -> tuple[NFWProfileParams, ...]:
    """Resolve the per-segment NFW mass definitions used during painting."""

    profiles: list[NFWProfileParams] = []
    for row in rows:
        mode = row["nfw_overdensity_mode"].strip().lower()
        reference = row["nfw_reference_density"].strip().lower()
        if reference not in {"critical", "mean"}:
            raise ValueError(f"unsupported NFW reference density: {reference!r}")
        if mode == "bryan_norman":
            profiles.append(
                NFWProfileParams(
                    reference_density=reference,
                    overdensity_mode="bryan_norman",
                )
            )
        elif mode in {"constant", "per_segment"}:
            try:
                overdensity = float(row["nfw_overdensity"])
            except ValueError as exc:
                raise ValueError(
                    f"segment {row['segment_index']} has no numerical NFW overdensity"
                ) from exc
            profiles.append(
                NFWProfileParams(
                    overdensity=overdensity,
                    reference_density=reference,
                    overdensity_mode="constant",
                )
            )
        else:
            raise ValueError(f"unsupported NFW overdensity mode: {mode!r}")
    return tuple(profiles)


def build_mask_coupling(
    pixels: np.ndarray,
    nside: int,
    lmax: int,
    *,
    bin_width: int,
    n_iter: int,
) -> MaskCoupling:
    """Build one exact NaMaster workspace for the compact binary mask."""

    try:
        import pymaster as nmt
    except ImportError as exc:  # pragma: no cover - optional install
        raise ImportError(
            "mask-coupled validation requires NaMaster; install geppetto[validation] "
            "or conda install -c conda-forge namaster"
        ) from exc
    if n_iter < 0:
        raise ValueError("mask SHT iterations must be non-negative")
    npix = 12 * nside**2
    mask = np.zeros(npix, dtype=np.float64)
    mask[pixels] = 1.0
    template = mask[None, None, :]
    lmax_mask = min(2 * lmax, 3 * nside - 1)
    reference_field = nmt.NmtField(
        mask,
        None,
        spin=0,
        templates=template,
        n_iter=n_iter,
        n_iter_mask=n_iter,
        lmax=lmax,
        lmax_mask=lmax_mask,
        masked_on_input=True,
    )
    bins = nmt.NmtBin.from_lmax_linear(lmax, nlb=max(1, bin_width))
    workspace = nmt.NmtWorkspace.from_fields(reference_field, reference_field, bins)
    return MaskCoupling(
        module=nmt,
        mask=mask,
        template=template,
        reference_field=reference_field,
        workspace=workspace,
        ell_full=np.arange(lmax + 1, dtype=np.int64),
        f_sky=float(np.mean(mask)),
        nside=nside,
        n_iter=n_iter,
    )


def estimate_pseudo_cls(
    compact_counts: np.ndarray,
    pixels: np.ndarray,
    coupling: MaskCoupling,
    *,
    full_sky_buffer: np.ndarray | None = None,
) -> np.ndarray:
    """Estimate constant-deprojected pseudo-``C_ell / f_sky`` with NaMaster."""

    mean_count = float(np.mean(compact_counts))
    if not np.isfinite(mean_count) or mean_count <= 0.0:
        raise ValueError("compact count map must have a positive finite mean")
    if full_sky_buffer is None:
        normalized_counts = np.zeros_like(coupling.mask)
    else:
        normalized_counts = full_sky_buffer
        if normalized_counts.shape != coupling.mask.shape:
            raise ValueError("full-sky scratch buffer has the wrong shape")
        if normalized_counts.dtype != np.float64:
            raise ValueError("full-sky scratch buffer must use float64")
        normalized_counts.fill(0.0)
    normalized_counts[pixels] = compact_counts / mean_count
    field = coupling.module.NmtField(
        coupling.mask,
        normalized_counts[None, :],
        spin=0,
        templates=coupling.template,
        n_iter=coupling.n_iter,
        n_iter_mask=coupling.n_iter,
        lmax=int(coupling.ell_full[-1]),
        lmax_mask=min(2 * int(coupling.ell_full[-1]), 3 * coupling.nside - 1),
        masked_on_input=True,
        lite=True,
    )
    coupled = coupling.module.compute_coupled_cell(field, field)[0]
    result = np.asarray(coupled, dtype=np.float64) / coupling.f_sky
    del field
    gc.collect()
    return result


def couple_theory_component(
    spectrum: np.ndarray,
    ell: np.ndarray,
    coupling: MaskCoupling,
) -> np.ndarray:
    """Forward-couple one full-sky component using the measured-map convention."""

    values = np.asarray(spectrum, dtype=np.float64)
    ell_values = np.asarray(ell, dtype=np.int64)
    if values.shape[-1] != ell_values.size:
        raise ValueError("theory component and ell dimensions disagree")

    def couple_one(row: np.ndarray) -> np.ndarray:
        full_spectrum = np.zeros(coupling.ell_full.size, dtype=np.float64)
        full_spectrum[ell_values] = row
        component = full_spectrum[None, :]
        coupled = coupling.workspace.couple_cell(component)[0]
        deprojection = coupling.module.deprojection_bias(
            coupling.reference_field,
            coupling.reference_field,
            component,
            n_iter=coupling.n_iter,
        )[0]
        return np.asarray(coupled + deprojection, dtype=np.float64)[ell_values] / coupling.f_sky

    if values.ndim == 1:
        return couple_one(values)
    if values.ndim == 2:
        return np.stack([couple_one(row) for row in values])
    raise ValueError("theory component must be one- or two-dimensional")


def sigma8_reference(
    sigma8_input: float,
    mass_map_headers: list[dict[str, Any]],
    *,
    reconstructed_sigma8: float,
) -> tuple[float, str]:
    """Resolve effective PINOCCHIO sigma8 and validate available metadata.

    Older mass-map files may omit ``COS_S8`` when the parameter-file value is
    zero. In that case the value reconstructed from the PINOCCHIO cosmology
    table is used as a non-independent fallback.
    """

    header_values: list[float] = []
    for header in mass_map_headers:
        value = header.get("COS_S8")
        if value is not None:
            header_values.append(float(value))
    if header_values and len(header_values) != len(mass_map_headers):
        raise ValueError("COS_S8 must be present in either every mass-map header or none")
    if header_values:
        values = np.asarray(header_values, dtype=np.float64)
        if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("mass-map COS_S8 values must be positive and finite")
        if not np.allclose(values, values[0], rtol=1.0e-8, atol=0.0):
            raise ValueError("mass-map COS_S8 values disagree across shells")
        header_sigma8 = float(values[0])
    else:
        header_sigma8 = np.nan

    if sigma8_input > 0.0:
        if np.isfinite(header_sigma8) and not np.isclose(
            header_sigma8, sigma8_input, rtol=1.0e-6, atol=0.0
        ):
            raise ValueError(
                "PINOCCHIO parameter Sigma8 and mass-map COS_S8 disagree: "
                f"parameter={sigma8_input:.12g}, header={header_sigma8:.12g}"
            )
        return sigma8_input, "parameter_file"
    if not np.isfinite(header_sigma8):
        if not np.isfinite(reconstructed_sigma8) or reconstructed_sigma8 <= 0.0:
            raise ValueError(
                "cosmology-table sigma8 fallback must be positive and finite"
            )
        return reconstructed_sigma8, "cosmology_power_spectrum"
    return header_sigma8, "mass_map_COS_S8"


def _binned_rows(
    label: str,
    ell: np.ndarray,
    measured: np.ndarray,
    linear: np.ndarray,
    one_halo: np.ndarray,
    shot: np.ndarray,
    *,
    ell_min: int,
    bin_width: int,
    f_sky: float,
    shell_weight: float,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    first = max(int(ell[0]), ell_min)
    for lower in range(first, int(ell[-1]) + 1, bin_width):
        upper = min(lower + bin_width, int(ell[-1]) + 1)
        selected = (ell >= lower) & (ell < upper)
        if not np.any(selected):
            continue
        weights = 2.0 * ell[selected] + 1.0

        def average(
            values: np.ndarray,
            selected_values: np.ndarray = selected,
            bin_weights: np.ndarray = weights,
        ) -> float:
            return float(np.average(values[selected_values], weights=bin_weights))

        measured_bin = average(measured)
        linear_bin = average(linear)
        one_halo_bin = average(one_halo)
        shot_bin = average(shot)
        clustering = linear_bin + one_halo_bin
        total = clustering + shot_bin
        rows.append(
            {
                "map": label,
                "ell_min": lower,
                "ell_max": upper - 1,
                "ell_effective": float(np.average(ell[selected], weights=weights)),
                "measured": measured_bin,
                "linear": linear_bin,
                "one_halo": one_halo_bin,
                "particle_shot_noise": shot_bin,
                "clustering": clustering,
                "total": total,
                "measured_over_total": measured_bin / total if total != 0.0 else np.nan,
                "f_sky": f_sky,
                "shell_weight": shell_weight,
                "theory_convention": "constant_deprojected_pseudo_cl_over_f_sky",
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_validation(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    """Run the map/theory comparison and return its three output paths."""

    print("[theory] loading manifest and PINOCCHIO tables", flush=True)
    rows = load_manifest(args.manifest)
    concentration = concentration_from_manifest(rows)
    profiles = profiles_from_manifest(rows)
    z_lo = np.asarray([float(row["z_lo"]) for row in rows], dtype=np.float64)
    z_hi = np.asarray([float(row["z_hi"]) for row in rows], dtype=np.float64)
    if np.any(z_lo < 0.0) or np.any(z_hi <= z_lo):
        raise ValueError("manifest shell bounds must satisfy 0 <= z_lo < z_hi")
    required_redshifts = np.unique(np.concatenate([z_lo, z_hi]))

    run_metadata = read_pinocchio_parameter_file(args.params)
    linear_theory = read_pinocchio_cosmology_table(args.cosmology_table)
    if not np.isclose(run_metadata.cosmology.h, linear_theory.h, rtol=1.0e-6, atol=0.0):
        raise ValueError("parameter-file and cosmology-table Hubble100 values disagree")
    if not np.isclose(
        run_metadata.cosmology.omega_m,
        linear_theory.omega_m0,
        rtol=1.0e-6,
        atol=0.0,
    ):
        raise ValueError("parameter-file and cosmology-table Omega0 values disagree")
    reconstructed_sigma8 = float(np.asarray(sigma8_from_linear_power(linear_theory)))
    if not np.isfinite(reconstructed_sigma8) or reconstructed_sigma8 <= 0.0:
        raise ValueError("reconstructed sigma8 must be positive and finite")
    hmf_paths = tuple(Path(path) for path in sorted(glob.glob(args.hmf_glob)))
    mass_function = read_pinocchio_mass_function_series(
        hmf_paths,
        required_redshifts=required_redshifts,
    )
    table_z_max = 1.0 / float(np.min(np.asarray(linear_theory.scale_factor))) - 1.0
    if float(np.max(z_hi)) > table_z_max:
        raise ValueError("shell redshift exceeds the PINOCCHIO cosmology table")

    first_mass_map_path = _resolve_input_path(rows[0]["mass_map_path"], args.manifest)
    first_mass_map = read_pinocchio_mass_map_fits(first_mass_map_path)
    if first_mass_map.ordering.strip().upper() != "RING":
        raise ValueError(f"theory validation requires RING maps: {first_mass_map_path}")
    compact_pixels = np.array(first_mass_map.pixel, dtype=np.int64, copy=True)
    nside = first_mass_map.nside
    if np.unique(compact_pixels).size != compact_pixels.size:
        raise ValueError("compact map contains duplicate HEALPix pixels")
    npix = 12 * nside**2
    if np.any(compact_pixels < 0) or np.any(compact_pixels >= npix):
        raise ValueError("compact map contains out-of-range HEALPix pixels")
    lmax = 2 * nside if args.ell_max is None else args.ell_max
    if lmax < 2 or lmax > 3 * nside - 1:
        raise ValueError("ell_max must satisfy 2 <= ell_max <= 3*nside-1")
    if args.ell_bin_width < 1 or args.radial_order < 2 or args.profile_order < 2:
        raise ValueError("bin width and quadrature orders must be positive")
    if (
        args.ell_exact_cap < 0
        or args.limber_match_rtol <= 0.0
        or args.limber_match_width < 1
        or args.exact_batch_size < 1
        or args.exact_workers < 1
        or args.sigma8_rtol <= 0.0
        or args.mask_sht_iterations < 0
    ):
        raise ValueError("adaptive projection, sigma8, and mask controls are inconsistent")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    exact_checkpoint_path = args.output_dir / "angular_power_exact_checkpoint.npz"

    print("[theory] building NaMaster mask-coupling workspace", flush=True)
    coupling = build_mask_coupling(
        compact_pixels,
        nside,
        lmax,
        bin_width=args.ell_bin_width,
        n_iter=args.mask_sht_iterations,
    )
    f_sky = coupling.f_sky

    print("[theory] measuring constant-deprojected pseudo-spectra", flush=True)
    ell = np.arange(2, lmax + 1, dtype=np.int64)
    observed_shell = np.empty((len(rows), ell.size), dtype=np.float64)
    mean_uncollapsed = np.empty(len(rows), dtype=np.float64)
    mean_total = np.empty(len(rows), dtype=np.float64)
    summed_counts = np.zeros(compact_pixels.size, dtype=np.float64)
    full_sky_buffer = np.zeros(npix, dtype=np.float64)
    mass_map_headers: list[dict[str, Any]] = []
    measurement_started = perf_counter()
    for index, row in enumerate(rows):
        shell_started = perf_counter()
        mass_map_path = _resolve_input_path(row["mass_map_path"], args.manifest)
        if index == 0:
            mass_map = first_mass_map
            first_mass_map = None
        else:
            mass_map = read_pinocchio_mass_map_fits(mass_map_path)
        if mass_map.ordering.strip().upper() != "RING":
            raise ValueError(f"theory validation requires RING maps: {mass_map_path}")
        if mass_map.nside != nside or not np.array_equal(mass_map.pixel, compact_pixels):
            raise ValueError("all segments must use the same NSIDE and compact RING pixel rows")
        mass_map_headers.append({"COS_S8": mass_map.header.get("COS_S8")})

        output_path = _resolve_input_path(row["output_npz"], args.manifest)
        try:
            with np.load(output_path, allow_pickle=False) as painted_file:
                total_counts = np.array(
                    painted_file["nfw_particle_counts"],
                    dtype=np.float64,
                    copy=True,
                    order="C",
                )
        except (OSError, KeyError) as exc:
            raise ValueError(f"cannot read painted NFW counts: {output_path}") from exc
        if total_counts.shape != mass_map.temperature.shape:
            raise ValueError(f"painted and PINOCCHIO map shapes differ: {output_path}")
        mean_uncollapsed[index] = np.mean(mass_map.temperature, dtype=np.float64)
        np.add(total_counts, mass_map.temperature, out=total_counts)
        mean_total[index] = np.mean(total_counts, dtype=np.float64)
        if not np.isfinite(mean_total[index]) or mean_total[index] <= 0.0:
            raise ValueError(f"total shell map must have a positive finite mean: {output_path}")
        summed_counts += total_counts
        del mass_map
        gc.collect()
        observed_shell[index] = estimate_pseudo_cls(
            total_counts,
            compact_pixels,
            coupling,
            full_sky_buffer=full_sky_buffer,
        )[ell]
        del total_counts
        print(
            f"[theory] measured shell {index + 1}/{len(rows)} "
            f"in {perf_counter() - shell_started:.1f}s "
            f"(total {perf_counter() - measurement_started:.1f}s)",
            flush=True,
        )

    print("[theory] measuring summed map", flush=True)
    observed_sum = estimate_pseudo_cls(
        summed_counts,
        compact_pixels,
        coupling,
        full_sky_buffer=full_sky_buffer,
    )[ell]
    del summed_counts, full_sky_buffer
    gc.collect()

    reference_sigma8, sigma8_source = sigma8_reference(
        run_metadata.sigma8_input,
        mass_map_headers,
        reconstructed_sigma8=reconstructed_sigma8,
    )
    sigma8_relative_error = abs(reconstructed_sigma8 - reference_sigma8) / reference_sigma8
    print(
        "[theory] sigma8 closure: "
        f"reconstructed={reconstructed_sigma8:.8g}, reference={reference_sigma8:.8g} "
        f"({sigma8_source}), relative_error={sigma8_relative_error:.3e}",
        flush=True,
    )
    if sigma8_relative_error > args.sigma8_rtol:
        raise ValueError(
            "PINOCCHIO power-spectrum sigma8 closure failed: "
            f"reconstructed={reconstructed_sigma8:.12g}, reference={reference_sigma8:.12g}, "
            f"relative_error={sigma8_relative_error:.6g}, tolerance={args.sigma8_rtol:.6g}"
        )

    shell_weights = mean_total / np.sum(mean_total)
    chi_lo = np.asarray(comoving_distance_mpc_h(jnp.asarray(z_lo), linear_theory))
    chi_hi = np.asarray(comoving_distance_mpc_h(jnp.asarray(z_hi), linear_theory))
    shell_volumes = chi_hi**3 - chi_lo**3
    volume_weights = shell_volumes / np.sum(shell_volumes)

    try:
        import healpy as hp
    except ImportError as exc:  # pragma: no cover - optional install
        raise ImportError("map validation requires healpy; install geppetto[io]") from exc
    try:
        # FITS-backed healpy tables use big-endian floats. JAX requires native
        # endian arrays, so force a native contiguous copy at the I/O boundary.
        pixel_window = np.array(
            hp.pixwin(nside, lmax=lmax),
            dtype=np.float64,
            copy=True,
            order="C",
        )[ell]
    except (OSError, KeyError) as exc:
        raise ValueError(
            "HEALPix pixel-window data are unavailable; install/cache the healpy-data "
            f"pixel window for NSIDE={nside} before validation"
        ) from exc
    theta_resolution = _consistent_float(rows, "theta_resolution_rad")
    theory_started = perf_counter()
    exact_batch_started = theory_started
    exact_fingerprint = exact_checkpoint_fingerprint(
        z_lo,
        z_hi,
        shell_weights,
        linear_theory,
        radial_order=args.radial_order,
        relative_tolerance=args.exact_relative_tolerance,
    )

    def report_exact_batch(event: str, ell_min: int, ell_max: int) -> None:
        nonlocal exact_batch_started
        if event == "start":
            exact_batch_started = perf_counter()
            print(
                f"[theory] exact multipoles {ell_min}-{ell_max} started",
                flush=True,
            )
        elif event == "complete":
            print(
                f"[theory] exact multipoles {ell_min}-{ell_max} completed in "
                f"{perf_counter() - exact_batch_started:.1f}s",
                flush=True,
            )

    def compute_exact_batch(batch_ell: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return exact_linear_shell_cls(
            batch_ell,
            z_lo,
            z_hi,
            linear_theory,
            shell_weights=shell_weights,
            radial_order=args.radial_order,
            relative_tolerance=args.exact_relative_tolerance,
            workers=args.exact_workers,
        )

    def evaluate_exact_batch(batch_ell: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        shell_result, summed_result, cache_hit = exact_batch_with_checkpoint(
            exact_checkpoint_path,
            exact_fingerprint,
            batch_ell,
            len(rows),
            compute_exact_batch,
        )
        action = "restored from" if cache_hit else "saved to"
        print(
            f"[theory] exact multipoles {int(batch_ell[0])}-{int(batch_ell[-1])} "
            f"{action} {exact_checkpoint_path}",
            flush=True,
        )
        return shell_result, summed_result

    print(
        "[theory] computing full-sky exact/Limber and one-halo spectra "
        f"with {args.exact_workers} exact workers",
        flush=True,
    )
    theory = hybrid_angular_power_spectra(
        jnp.asarray(ell),
        z_lo,
        z_hi,
        linear_theory,
        mass_function,
        concentration,
        profiles,
        shell_weights=jnp.asarray(shell_weights),
        pixel_window=jnp.asarray(pixel_window),
        mean_uncollapsed_counts_per_pixel=jnp.asarray(mean_uncollapsed),
        mean_total_counts_per_pixel=jnp.asarray(mean_total),
        pixel_area_sr=healpix_pixel_area_sr(nside),
        theta_resolution_rad=theta_resolution,
        ell_exact_cap=args.ell_exact_cap,
        limber_match_rtol=args.limber_match_rtol,
        limber_match_width=args.limber_match_width,
        exact_batch_size=args.exact_batch_size,
        exact_workers=args.exact_workers,
        exact_batch_callback=report_exact_batch,
        exact_batch_evaluator=evaluate_exact_batch,
        radial_order=args.radial_order,
        profile_order=args.profile_order,
        exact_relative_tolerance=args.exact_relative_tolerance,
    )

    theory_np = {field: np.asarray(getattr(theory, field)) for field in theory._fields}
    print(
        "[theory] Limber starts at "
        f"ell={int(theory_np['summed_ell_limber_start'])} for the summed spectrum and "
        f"ell={int(np.min(theory_np['shell_ell_limber_start']))}-"
        f"{int(np.max(theory_np['shell_ell_limber_start']))} across shells; "
        f"theory completed in {perf_counter() - theory_started:.1f}s",
        flush=True,
    )
    print("[theory] forward-coupling theory components through the mask", flush=True)
    pseudo_theory: dict[str, np.ndarray] = {}
    for scope in ("shell", "summed"):
        for component in THEORY_COMPONENTS:
            source_name = f"{scope}_{component}"
            output_name = f"{source_name}_pseudo_over_fsky"
            pseudo_theory[output_name] = couple_theory_component(
                theory_np[source_name],
                ell,
                coupling,
            )
    midpoint_z = 0.5 * (z_lo + z_hi)
    cosmology = Cosmology(omega_m=linear_theory.omega_m0, h=linear_theory.h)
    mass_fraction = np.asarray(
        [resolved_halo_mass_fraction(z, mass_function, cosmology) for z in midpoint_z]
    )
    diagnostic_k = float(np.asarray(linear_theory.k_h_mpc)[0])
    low_k_one_halo = np.asarray(
        [
            one_halo_matter_power(
                jnp.asarray(diagnostic_k),
                z,
                linear_theory,
                mass_function,
                concentration,
                profile,
                theta_resolution_rad=theta_resolution,
            )
            for z, profile in zip(midpoint_z, profiles, strict=True)
        ]
    )
    low_k_linear = np.asarray(
        [linear_matter_power(jnp.asarray(diagnostic_k), z, linear_theory) for z in midpoint_z]
    )

    theory_path = args.output_dir / "angular_power_theory.npz"
    print(f"[theory] writing {theory_path}", flush=True)
    np.savez_compressed(
        theory_path,
        validation_schema_version=np.asarray(VALIDATION_SCHEMA_VERSION, dtype=np.int64),
        observed_shell=observed_shell,
        observed_sum=observed_sum,
        ell=theory_np["ell"],
        shell_linear=theory_np["shell_linear"],
        shell_one_halo=theory_np["shell_one_halo"],
        shell_particle_shot_noise=theory_np["shell_particle_shot_noise"],
        summed_linear=theory_np["summed_linear"],
        summed_one_halo=theory_np["summed_one_halo"],
        summed_particle_shot_noise=theory_np["summed_particle_shot_noise"],
        shell_weights=theory_np["shell_weights"],
        shell_ell_limber_start=theory_np["shell_ell_limber_start"],
        summed_ell_limber_start=theory_np["summed_ell_limber_start"],
        ell_limber_start=theory_np["ell_limber_start"],
        limber_match_shell_relative_error=theory_np[
            "limber_match_shell_relative_error"
        ],
        limber_match_summed_relative_error=theory_np[
            "limber_match_summed_relative_error"
        ],
        reference_sigma8=np.asarray(reference_sigma8),
        reconstructed_sigma8=np.asarray(reconstructed_sigma8),
        sigma8_relative_error=np.asarray(sigma8_relative_error),
        sigma8_reference_source=np.asarray(sigma8_source),
        sigma8_relative_tolerance=np.asarray(args.sigma8_rtol),
        limber_match_relative_tolerance=np.asarray(args.limber_match_rtol),
        limber_match_width=np.asarray(args.limber_match_width, dtype=np.int64),
        ell_exact_cap=np.asarray(args.ell_exact_cap, dtype=np.int64),
        exact_batch_size=np.asarray(args.exact_batch_size, dtype=np.int64),
        exact_workers=np.asarray(args.exact_workers, dtype=np.int64),
        exact_relative_tolerance=np.asarray(args.exact_relative_tolerance),
        mask_sht_iterations=np.asarray(args.mask_sht_iterations, dtype=np.int64),
        mask_pixel_sha256=np.asarray(
            hashlib.sha256(np.ascontiguousarray(compact_pixels).view(np.uint8)).hexdigest()
        ),
        **pseudo_theory,
    )

    binned_rows: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        binned_rows.extend(
            _binned_rows(
                f"segment_{row['segment_index']}",
                ell,
                observed_shell[index],
                pseudo_theory["shell_linear_pseudo_over_fsky"][index],
                pseudo_theory["shell_one_halo_pseudo_over_fsky"][index],
                pseudo_theory["shell_particle_shot_noise_pseudo_over_fsky"][index],
                ell_min=args.ell_min_compare,
                bin_width=args.ell_bin_width,
                f_sky=f_sky,
                shell_weight=float(shell_weights[index]),
            )
        )
    binned_rows.extend(
        _binned_rows(
            "summed",
            ell,
            observed_sum,
            pseudo_theory["summed_linear_pseudo_over_fsky"],
            pseudo_theory["summed_one_halo_pseudo_over_fsky"],
            pseudo_theory["summed_particle_shot_noise_pseudo_over_fsky"],
            ell_min=args.ell_min_compare,
            bin_width=args.ell_bin_width,
            f_sky=f_sky,
            shell_weight=1.0,
        )
    )
    binned_path = args.output_dir / "angular_power_binned.csv"
    _write_csv(binned_path, binned_rows)

    diagnostic_rows: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        diagnostic_rows.append(
            {
                "segment_index": int(row["segment_index"]),
                "z_lo": z_lo[index],
                "z_hi": z_hi[index],
                "chi_lo_mpc_h": chi_lo[index],
                "chi_hi_mpc_h": chi_hi[index],
                "mean_uncollapsed_counts_per_pixel": mean_uncollapsed[index],
                "mean_total_counts_per_pixel": mean_total[index],
                "measured_shell_weight": shell_weights[index],
                "volume_shell_weight": volume_weights[index],
                "resolved_hmf_mass_fraction": mass_fraction[index],
                "diagnostic_k_h_mpc": diagnostic_k,
                "one_halo_over_linear_at_diagnostic_k": (
                    low_k_one_halo[index] / low_k_linear[index]
                ),
                "f_sky": f_sky,
                "nside": nside,
                "theta_resolution_rad": theta_resolution,
                "reference_sigma8": reference_sigma8,
                "reconstructed_sigma8": reconstructed_sigma8,
                "sigma8_relative_error": sigma8_relative_error,
                "sigma8_reference_source": sigma8_source,
                "ell_limber_start": int(theory_np["summed_ell_limber_start"]),
                "shell_ell_limber_start": int(theory_np["shell_ell_limber_start"][index]),
                "summed_ell_limber_start": int(theory_np["summed_ell_limber_start"]),
                "limber_match_shell_relative_error": theory_np[
                    "limber_match_shell_relative_error"
                ][index],
                "limber_match_summed_relative_error": theory_np[
                    "limber_match_summed_relative_error"
                ],
                "ell_exact_cap": args.ell_exact_cap,
                "exact_batch_size": args.exact_batch_size,
                "exact_workers": args.exact_workers,
                "exact_relative_tolerance": args.exact_relative_tolerance,
                "mask_sht_iterations": args.mask_sht_iterations,
                "theory_convention": "constant_deprojected_pseudo_cl_over_f_sky",
            }
        )
    diagnostics_path = args.output_dir / "angular_power_diagnostics.csv"
    _write_csv(diagnostics_path, diagnostic_rows)
    exact_checkpoint_path.unlink(missing_ok=True)
    return theory_path, binned_path, diagnostics_path


def main() -> None:
    args = parse_args()
    jax.config.update("jax_enable_x64", args.jax_precision == "float64")
    try:
        outputs = run_validation(args)
    except (PinocchioCatalogError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc
    for output in outputs:
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
