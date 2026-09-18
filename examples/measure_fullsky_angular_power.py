#!/usr/bin/env python3
"""Measure full-sky PINOCCHIO+GEPPETTO shell angular power spectra."""

from __future__ import annotations

import argparse
import csv
import gc
from pathlib import Path
from time import perf_counter

import numpy as np

from geppetto.io import (
    read_pinocchio_cosmology_table,
    read_pinocchio_mass_map_fits,
    read_pinocchio_parameter_file,
)

FULLSKY_OBSERVED_SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--cosmology-table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ell-max", type=int, default=None)
    parser.add_argument("--sht-iterations", type=int, default=3)
    return parser.parse_args()


def _read_manifest(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise ValueError(f"cannot read painting manifest: {path}") from exc
    required = {"segment_index", "z_lo", "z_hi", "mass_map_path", "output_npz"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"painting manifest is empty or incomplete: {path}")
    try:
        rows.sort(key=lambda row: int(row["segment_index"]))
    except ValueError as exc:
        raise ValueError("manifest segment indices must be integers") from exc
    return rows


def _resolve_input_path(value: str, manifest: Path) -> Path:
    candidate = Path(value)
    if candidate.exists() or candidate.is_absolute():
        return candidate
    relative = manifest.parent / candidate
    return relative if relative.exists() else candidate


def theoretical_shell_mean_counts(
    z_lo: np.ndarray,
    z_hi: np.ndarray,
    nside: int,
    *,
    params: Path,
    cosmology_table: Path,
) -> np.ndarray:
    """Return homogeneous particle counts per full-sky shell pixel."""

    run = read_pinocchio_parameter_file(params)
    theory = read_pinocchio_cosmology_table(cosmology_table)
    if not np.isclose(run.cosmology.h, theory.h, rtol=1.0e-6, atol=0.0):
        raise ValueError("parameter and cosmology tables have inconsistent Hubble constants")
    scale_factor = np.asarray(theory.scale_factor, dtype=np.float64)
    chi_table = np.asarray(theory.chi_mpc_h, dtype=np.float64)
    chi_lo = np.interp(1.0 / (1.0 + z_lo), scale_factor, chi_table)
    chi_hi = np.interp(1.0 / (1.0 + z_hi), scale_factor, chi_table)
    particle_density = (run.grid_size / run.box_size_mpc_h) ** 3
    pixel_area = 4.0 * np.pi / (12 * nside**2)
    result = particle_density * pixel_area * (chi_hi**3 - chi_lo**3) / 3.0
    if np.any(~np.isfinite(result)) or np.any(result <= 0.0):
        raise ValueError("theoretical shell means must be positive and finite")
    return result


def _fullsky_cl(
    counts: np.ndarray,
    normalization: float,
    *,
    lmax: int,
    iterations: int,
) -> np.ndarray:
    try:
        import healpy as hp
    except ImportError as exc:  # pragma: no cover - optional install
        raise ImportError("full-sky measurement requires healpy") from exc
    if not np.isfinite(normalization) or normalization <= 0.0:
        raise ValueError("map normalization must be positive and finite")
    values = np.asarray(counts, dtype=np.float64)
    # Remove only ell=0 before the SHT, while retaining the requested
    # theoretical or measured normalization for every ell > 0 mode.
    overdensity = (values - np.mean(values, dtype=np.float64)) / normalization
    return np.asarray(
        hp.anafast(overdensity, lmax=lmax, iter=iterations),
        dtype=np.float64,
    )


def run_measurement(args: argparse.Namespace) -> Path:
    """Measure both theoretical-mean and measured-mean full-sky spectra."""

    if args.sht_iterations < 0:
        raise ValueError("sht_iterations must be non-negative")
    rows = _read_manifest(args.manifest)
    segment_index = np.asarray([int(row["segment_index"]) for row in rows], dtype=np.int64)
    z_lo = np.asarray([float(row["z_lo"]) for row in rows], dtype=np.float64)
    z_hi = np.asarray([float(row["z_hi"]) for row in rows], dtype=np.float64)
    if np.any(z_lo < 0.0) or np.any(z_hi <= z_lo):
        raise ValueError("manifest shell bounds must satisfy 0 <= z_lo < z_hi")

    first_map = read_pinocchio_mass_map_fits(
        _resolve_input_path(rows[0]["mass_map_path"], args.manifest)
    )
    nside = first_map.nside
    npix = 12 * nside**2
    expected_pixels = np.arange(npix, dtype=np.int64)
    if first_map.ordering.strip().upper() != "RING" or not np.array_equal(
        first_map.pixel, expected_pixels
    ):
        raise ValueError("full-sky validation requires every native RING pixel in row order")
    lmax = 2 * nside if args.ell_max is None else args.ell_max
    if lmax < 2 or lmax > 3 * nside - 1:
        raise ValueError("ell_max must satisfy 2 <= ell_max <= 3*nside-1")
    ell = np.arange(2, lmax + 1, dtype=np.int64)
    theoretical_mean = theoretical_shell_mean_counts(
        z_lo,
        z_hi,
        nside,
        params=args.params,
        cosmology_table=args.cosmology_table,
    )
    measured_mean = np.empty(len(rows), dtype=np.float64)
    shell_theoretical_mean = np.empty((len(rows), ell.size), dtype=np.float64)
    shell_measured_mean = np.empty((len(rows), ell.size), dtype=np.float64)
    summed_counts = np.zeros(npix, dtype=np.float64)
    started = perf_counter()
    for index, row in enumerate(rows):
        shell_started = perf_counter()
        mass_map = first_map if index == 0 else read_pinocchio_mass_map_fits(
            _resolve_input_path(row["mass_map_path"], args.manifest)
        )
        if mass_map.nside != nside or not np.array_equal(mass_map.pixel, expected_pixels):
            raise ValueError("all mass maps must share complete native RING rows")
        output_path = _resolve_input_path(row["output_npz"], args.manifest)
        try:
            with np.load(output_path, allow_pickle=False) as painted:
                counts = np.array(
                    painted["nfw_particle_counts"],
                    dtype=np.float64,
                    copy=True,
                    order="C",
                )
        except (OSError, KeyError) as exc:
            raise ValueError(f"cannot read painted NFW counts: {output_path}") from exc
        if counts.shape != mass_map.temperature.shape:
            raise ValueError(f"painted and PINOCCHIO map shapes differ: {output_path}")
        np.add(counts, mass_map.temperature, out=counts)
        measured_mean[index] = np.mean(counts, dtype=np.float64)
        summed_counts += counts
        shell_theoretical_mean[index] = _fullsky_cl(
            counts,
            theoretical_mean[index],
            lmax=lmax,
            iterations=args.sht_iterations,
        )[ell]
        # Changing a full-sky constant normalization only rescales ell > 0.
        shell_measured_mean[index] = shell_theoretical_mean[index] * (
            theoretical_mean[index] / measured_mean[index]
        ) ** 2
        del mass_map, counts
        gc.collect()
        print(
            f"[fullsky] measured shell {index + 1}/{len(rows)} in "
            f"{perf_counter() - shell_started:.1f}s "
            f"(total {perf_counter() - started:.1f}s)",
            flush=True,
        )
    print("[fullsky] measuring summed map", flush=True)
    summed_theoretical_mean = _fullsky_cl(
        summed_counts,
        float(np.sum(theoretical_mean)),
        lmax=lmax,
        iterations=args.sht_iterations,
    )[ell]
    summed_measured_mean = summed_theoretical_mean * (
        np.sum(theoretical_mean) / np.sum(measured_mean)
    ) ** 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        schema_version=np.asarray(FULLSKY_OBSERVED_SCHEMA_VERSION, dtype=np.int64),
        ell=ell,
        shell_cl_theoretical_mean=shell_theoretical_mean,
        shell_cl_measured_mean=shell_measured_mean,
        summed_cl_theoretical_mean=summed_theoretical_mean,
        summed_cl_measured_mean=summed_measured_mean,
        theoretical_mean_counts_per_pixel=theoretical_mean,
        measured_mean_counts_per_pixel=measured_mean,
        segment_index=segment_index,
        z_lo=z_lo,
        z_hi=z_hi,
        nside=np.asarray(nside, dtype=np.int64),
        sht_iterations=np.asarray(args.sht_iterations, dtype=np.int64),
    )
    print(f"Wrote {args.output}", flush=True)
    return args.output


def main() -> None:
    args = parse_args()
    try:
        run_measurement(args)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    main()
