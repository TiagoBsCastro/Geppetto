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
import glob
from pathlib import Path

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
)
from geppetto.profiles import NFWProfileParams
from geppetto.theory import (
    comoving_distance_mpc_h,
    hybrid_angular_power_spectra,
    linear_matter_power,
    one_halo_matter_power,
    resolved_halo_mass_fraction,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate painted PINOCCHIO angular maps against halo-model C_ell theory"
    )
    parser.add_argument("--manifest", type=Path, required=True)
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
    parser.add_argument("--ell-exact-max", type=int, default=100)
    parser.add_argument("--radial-order", type=int, default=64)
    parser.add_argument("--profile-order", type=int, default=64)
    parser.add_argument("--exact-relative-tolerance", type=float, default=1.0e-4)
    parser.add_argument(
        "--jax-precision",
        choices=("float32", "float64"),
        default="float64",
    )
    return parser.parse_args()


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


def estimate_pseudo_cls(
    compact_counts: np.ndarray,
    pixels: np.ndarray,
    nside: int,
    lmax: int,
) -> np.ndarray:
    """Estimate binary-mask pseudo-C_ell/f_sky from compact count rows."""

    try:
        import healpy as hp
    except ImportError as exc:  # pragma: no cover - optional install
        raise ImportError("map validation requires healpy; install geppetto[io]") from exc

    mean_count = float(np.mean(compact_counts))
    if not np.isfinite(mean_count) or mean_count <= 0.0:
        raise ValueError("compact count map must have a positive finite mean")
    npix = hp.nside2npix(nside)
    delta = np.zeros(npix, dtype=np.float64)
    delta[pixels] = compact_counts / mean_count - 1.0
    f_sky = pixels.size / npix
    return np.asarray(hp.anafast(delta, lmax=lmax, iter=0), dtype=np.float64) / f_sky


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

    rows = load_manifest(args.manifest)
    concentration = concentration_from_manifest(rows)
    profiles = profiles_from_manifest(rows)
    z_lo = np.asarray([float(row["z_lo"]) for row in rows], dtype=np.float64)
    z_hi = np.asarray([float(row["z_hi"]) for row in rows], dtype=np.float64)
    if np.any(z_lo < 0.0) or np.any(z_hi <= z_lo):
        raise ValueError("manifest shell bounds must satisfy 0 <= z_lo < z_hi")
    required_redshifts = np.unique(np.concatenate([z_lo, z_hi]))

    linear_theory = read_pinocchio_cosmology_table(args.cosmology_table)
    hmf_paths = tuple(Path(path) for path in sorted(glob.glob(args.hmf_glob)))
    mass_function = read_pinocchio_mass_function_series(
        hmf_paths,
        required_redshifts=required_redshifts,
    )
    table_z_max = 1.0 / float(np.min(np.asarray(linear_theory.scale_factor))) - 1.0
    if float(np.max(z_hi)) > table_z_max:
        raise ValueError("shell redshift exceeds the PINOCCHIO cosmology table")

    compact_pixels: np.ndarray | None = None
    nside: int | None = None
    uncollapsed_maps: list[np.ndarray] = []
    total_maps: list[np.ndarray] = []
    for row in rows:
        mass_map_path = _resolve_input_path(row["mass_map_path"], args.manifest)
        output_path = _resolve_input_path(row["output_npz"], args.manifest)
        mass_map = read_pinocchio_mass_map_fits(mass_map_path)
        if mass_map.ordering.strip().upper() != "RING":
            raise ValueError(f"theory validation requires RING maps: {mass_map_path}")
        try:
            with np.load(output_path, allow_pickle=False) as painted_file:
                painted = np.asarray(painted_file["nfw_particle_counts"], dtype=np.float64)
        except (OSError, KeyError) as exc:
            raise ValueError(f"cannot read painted NFW counts: {output_path}") from exc
        if painted.shape != mass_map.temperature.shape:
            raise ValueError(f"painted and PINOCCHIO map shapes differ: {output_path}")
        if compact_pixels is None:
            compact_pixels = np.asarray(mass_map.pixel, dtype=np.int64)
            nside = mass_map.nside
        elif mass_map.nside != nside or not np.array_equal(mass_map.pixel, compact_pixels):
            raise ValueError("all segments must use the same NSIDE and compact RING pixel rows")
        uncollapsed = np.asarray(mass_map.temperature, dtype=np.float64)
        uncollapsed_maps.append(uncollapsed)
        total_maps.append(uncollapsed + painted)

    assert compact_pixels is not None and nside is not None
    if np.unique(compact_pixels).size != compact_pixels.size:
        raise ValueError("compact map contains duplicate HEALPix pixels")
    npix = 12 * nside**2
    if np.any(compact_pixels < 0) or np.any(compact_pixels >= npix):
        raise ValueError("compact map contains out-of-range HEALPix pixels")
    f_sky = compact_pixels.size / npix
    lmax = 2 * nside if args.ell_max is None else args.ell_max
    if lmax < 2 or lmax > 3 * nside - 1:
        raise ValueError("ell_max must satisfy 2 <= ell_max <= 3*nside-1")
    if args.ell_bin_width < 1 or args.radial_order < 2 or args.profile_order < 2:
        raise ValueError("bin width and quadrature orders must be positive")

    observed_shell_full = np.stack(
        [
            estimate_pseudo_cls(counts, compact_pixels, nside, lmax)
            for counts in total_maps
        ]
    )
    observed_sum_full = estimate_pseudo_cls(
        np.sum(np.stack(total_maps), axis=0),
        compact_pixels,
        nside,
        lmax,
    )
    ell = np.arange(2, lmax + 1, dtype=np.int64)
    observed_shell = observed_shell_full[:, ell]
    observed_sum = observed_sum_full[ell]

    mean_uncollapsed = np.asarray([np.mean(values) for values in uncollapsed_maps])
    mean_total = np.asarray([np.mean(values) for values in total_maps])
    if np.any(mean_total <= 0.0):
        raise ValueError("every total shell map must have a positive mean")
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
        pixel_window = np.asarray(hp.pixwin(nside, lmax=lmax))[ell]
    except (OSError, KeyError) as exc:
        raise ValueError(
            "HEALPix pixel-window data are unavailable; install/cache the healpy-data "
            f"pixel window for NSIDE={nside} before validation"
        ) from exc
    theta_resolution = _consistent_float(rows, "theta_resolution_rad")
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
        ell_exact_max=args.ell_exact_max,
        radial_order=args.radial_order,
        profile_order=args.profile_order,
        exact_relative_tolerance=args.exact_relative_tolerance,
    )

    theory_np = {field: np.asarray(getattr(theory, field)) for field in theory._fields}
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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    theory_path = args.output_dir / "angular_power_theory.npz"
    np.savez_compressed(
        theory_path,
        observed_shell=observed_shell,
        observed_sum=observed_sum,
        **theory_np,
    )

    binned_rows: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        binned_rows.extend(
            _binned_rows(
                f"segment_{row['segment_index']}",
                ell,
                observed_shell[index],
                theory_np["shell_linear"][index],
                theory_np["shell_one_halo"][index],
                theory_np["shell_particle_shot_noise"][index],
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
            theory_np["summed_linear"],
            theory_np["summed_one_halo"],
            theory_np["summed_particle_shot_noise"],
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
            }
        )
    diagnostics_path = args.output_dir / "angular_power_diagnostics.csv"
    _write_csv(diagnostics_path, diagnostic_rows)
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
