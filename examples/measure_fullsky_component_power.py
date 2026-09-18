#!/usr/bin/env python3
"""Measure full-sky uncollapsed, painted-halo, and cross spectra."""

from __future__ import annotations

import argparse
import csv
import gc
from pathlib import Path
from time import perf_counter

import numpy as np

from geppetto.io import read_pinocchio_mass_map_fits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--observed-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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
    rows.sort(key=lambda row: int(row["segment_index"]))
    return rows


def _resolve_input_path(value: str, manifest: Path) -> Path:
    candidate = Path(value)
    if candidate.exists() or candidate.is_absolute():
        return candidate
    relative = manifest.parent / candidate
    return relative if relative.exists() else candidate


def component_cls(
    uncollapsed_counts: np.ndarray,
    halo_counts: np.ndarray,
    normalization: float,
    *,
    lmax: int,
    iterations: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``UU``, ``HH``, ``UH``, and their exact total-map closure.

    Both inputs are particle counts per native HEALPix pixel. The component
    monopoles are removed separately and both fields are divided by the same
    theoretical total-matter mean count per pixel. Consequently
    ``total = UU + HH + 2 UH`` is the spectrum of the summed count map.
    """

    try:
        import healpy as hp
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("component measurement requires healpy") from exc
    uncollapsed = np.asarray(uncollapsed_counts, dtype=np.float64)
    halo = np.asarray(halo_counts, dtype=np.float64)
    if uncollapsed.shape != halo.shape or uncollapsed.ndim != 1:
        raise ValueError("component count maps must be matching one-dimensional arrays")
    if not np.isfinite(normalization) or normalization <= 0.0:
        raise ValueError("component normalization must be positive and finite")
    if iterations < 0 or lmax < 2:
        raise ValueError("SHT iterations and lmax must be non-negative and at least two")

    delta_uncollapsed = (
        uncollapsed - np.mean(uncollapsed, dtype=np.float64)
    ) / normalization
    delta_halo = (halo - np.mean(halo, dtype=np.float64)) / normalization
    alm_uncollapsed = hp.map2alm(delta_uncollapsed, lmax=lmax, iter=iterations)
    alm_halo = hp.map2alm(delta_halo, lmax=lmax, iter=iterations)
    cl_uncollapsed = np.asarray(hp.alm2cl(alm_uncollapsed), dtype=np.float64)
    cl_halo = np.asarray(hp.alm2cl(alm_halo), dtype=np.float64)
    cl_cross = np.asarray(hp.alm2cl(alm_uncollapsed, alm_halo), dtype=np.float64)
    cl_total = cl_uncollapsed + cl_halo + 2.0 * cl_cross
    return cl_uncollapsed, cl_halo, cl_cross, cl_total


def run_measurement(args: argparse.Namespace) -> Path:
    """Measure component spectra and verify total-map spectral closure."""

    if args.sht_iterations < 0:
        raise ValueError("sht_iterations must be non-negative")
    rows = _read_manifest(args.manifest)
    try:
        with np.load(args.observed_cache, allow_pickle=False) as observed:
            ell = np.array(observed["ell"], dtype=np.int64, copy=True)
            total_reference = np.array(
                observed["shell_cl_theoretical_mean"], dtype=np.float64, copy=True
            )
            theoretical_mean = np.array(
                observed["theoretical_mean_counts_per_pixel"],
                dtype=np.float64,
                copy=True,
            )
            cache_segment = np.array(observed["segment_index"], dtype=np.int64, copy=True)
            cache_z_lo = np.array(observed["z_lo"], dtype=np.float64, copy=True)
            cache_z_hi = np.array(observed["z_hi"], dtype=np.float64, copy=True)
            nside = int(np.asarray(observed["nside"]))
    except (OSError, KeyError) as exc:
        raise ValueError(f"cannot read observed full-sky cache: {args.observed_cache}") from exc
    segment_index = np.asarray([int(row["segment_index"]) for row in rows])
    z_lo = np.asarray([float(row["z_lo"]) for row in rows])
    z_hi = np.asarray([float(row["z_hi"]) for row in rows])
    if not (
        np.array_equal(segment_index, cache_segment)
        and np.allclose(z_lo, cache_z_lo, rtol=0.0, atol=1.0e-10)
        and np.allclose(z_hi, cache_z_hi, rtol=0.0, atol=1.0e-10)
    ):
        raise ValueError("manifest and observed cache shell definitions differ")
    expected_shape = (len(rows), ell.size)
    if total_reference.shape != expected_shape or theoretical_mean.shape != (len(rows),):
        raise ValueError("observed cache arrays have inconsistent shapes")
    lmax = int(ell[-1])
    if not np.array_equal(ell, np.arange(2, lmax + 1)):
        raise ValueError("observed cache must contain every multipole from 2 through lmax")

    component_uncollapsed = np.empty(expected_shape, dtype=np.float64)
    component_halo = np.empty(expected_shape, dtype=np.float64)
    component_cross = np.empty(expected_shape, dtype=np.float64)
    component_total = np.empty(expected_shape, dtype=np.float64)
    mean_uncollapsed = np.empty(len(rows), dtype=np.float64)
    mean_halo = np.empty(len(rows), dtype=np.float64)
    closure_error = np.empty(len(rows), dtype=np.float64)
    npix = 12 * nside**2
    started = perf_counter()
    for index, row in enumerate(rows):
        shell_started = perf_counter()
        mass_map = read_pinocchio_mass_map_fits(
            _resolve_input_path(row["mass_map_path"], args.manifest)
        )
        if (
            mass_map.nside != nside
            or mass_map.ordering.strip().upper() != "RING"
            or mass_map.pixel.size != npix
            or not np.array_equal(mass_map.pixel, np.arange(npix, dtype=np.int64))
        ):
            raise ValueError("component diagnostics require complete native RING maps")
        output_path = _resolve_input_path(row["output_npz"], args.manifest)
        try:
            with np.load(output_path, allow_pickle=False) as painted:
                halo = np.array(
                    painted["nfw_particle_counts"], dtype=np.float64, copy=True
                )
        except (OSError, KeyError) as exc:
            raise ValueError(f"cannot read painted halo map: {output_path}") from exc
        uncollapsed = np.asarray(mass_map.temperature, dtype=np.float64)
        if halo.shape != uncollapsed.shape:
            raise ValueError(f"component map shapes differ: {output_path}")
        mean_uncollapsed[index] = np.mean(uncollapsed, dtype=np.float64)
        mean_halo[index] = np.mean(halo, dtype=np.float64)
        uu, hh, uh, total = component_cls(
            uncollapsed,
            halo,
            theoretical_mean[index],
            lmax=lmax,
            iterations=args.sht_iterations,
        )
        component_uncollapsed[index] = uu[ell]
        component_halo[index] = hh[ell]
        component_cross[index] = uh[ell]
        component_total[index] = total[ell]
        denominator = np.maximum(np.abs(total_reference[index]), np.finfo(float).tiny)
        closure_error[index] = np.max(
            np.abs(component_total[index] - total_reference[index]) / denominator
        )
        if closure_error[index] > 1.0e-8:
            raise ValueError(
                f"component closure failed for segment {segment_index[index]}: "
                f"maximum relative error={closure_error[index]:.6g}"
            )
        del mass_map, halo, uncollapsed, uu, hh, uh, total
        gc.collect()
        print(
            f"[components] measured shell {index + 1}/{len(rows)} in "
            f"{perf_counter() - shell_started:.1f}s "
            f"(total {perf_counter() - started:.1f}s)",
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        ell=ell,
        shell_cl_uncollapsed=component_uncollapsed,
        shell_cl_halo=component_halo,
        shell_cl_cross=component_cross,
        shell_cl_total_closure=component_total,
        shell_cl_total_reference=total_reference,
        theoretical_mean_counts_per_pixel=theoretical_mean,
        measured_mean_uncollapsed_counts_per_pixel=mean_uncollapsed,
        measured_mean_halo_counts_per_pixel=mean_halo,
        maximum_relative_closure_error=closure_error,
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
