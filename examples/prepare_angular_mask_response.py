#!/usr/bin/env python3
"""Precompute mode-count-binned mask/mean-removal response for shell theory.

The binary mask's constant-deprojection operator is self-adjoint. Its angular
power kernel obeys (2l+1) K[l,L] = (2L+1) K[L,l]. Applying the existing forward
model to one indicator per OUTPUT band therefore builds the transpose action
without thousands of individual input-multipole deprojection calculations.
The resulting response is checked against new direct calls. Differences from
archived forward-coupled theory are reported separately, since older archives
used a mask-only reference that silently skipped template deprojection.
No measured painted spectrum enters this cache.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from time import perf_counter

import numpy as np

import validate_pinocchio_angular_power as validation
from geppetto.io import read_pinocchio_mass_map_fits


def mode_count_binning(ell: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Return a normalized (n_band,n_ell) operator for dimensionless C_ell."""

    ell, edges = np.asarray(ell), np.asarray(edges)
    if (
        ell.ndim != 1 or ell.size == 0 or not np.all(np.isfinite(ell))
        or not np.all(np.isfinite(edges)) or np.any(ell < 0) or np.any(np.diff(ell) <= 0)
        or np.any(ell != np.floor(ell)) or edges.ndim != 1 or edges.size < 2
        or np.any(np.diff(edges) <= 0)
    ):
        raise ValueError("require ordered non-negative integer multipoles and increasing bin edges")
    selection = (ell[None, :] >= edges[:-1, None]) & (ell[None, :] < edges[1:, None])
    weighted = selection * (2 * ell[None, :] + 1)
    totals = np.sum(weighted, axis=1)
    if np.any(totals == 0):
        raise ValueError("each band must contain at least one input multipole")
    return weighted / totals[:, None]


def binned_self_adjoint_response(
    forward: Callable[[np.ndarray], np.ndarray],
    ell: np.ndarray,
    edges: np.ndarray,
    progress: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Transpose a self-adjoint field operator's isotropic power response.

    ``forward`` must act on dimensionless C_ell with the stated reciprocal
    mode-count weighting. This is not valid for non-self-adjoint template
    weighting or an arbitrary pseudo-spectrum cross-estimator. The binary
    auto-mask/constant-template convention is validated by the caller.
    """

    binning = mode_count_binning(ell, edges)
    modes = 2 * ell + 1
    response = np.empty_like(binning)
    for index, row in enumerate(binning):
        selected = row != 0
        values = np.asarray(forward(selected.astype(np.float64)), dtype=np.float64)
        if values.shape != ell.shape or not np.all(np.isfinite(values)):
            raise ValueError("forward response has invalid shape or non-finite values")
        response[index] = values * modes / np.sum(modes[selected])
        if progress is not None:
            progress(index + 1, len(binning))
    return response


def prepare(args: argparse.Namespace) -> dict:
    if not np.isfinite(args.check_rtol) or args.check_rtol <= 0:
        raise ValueError("check_rtol must be finite and positive")
    started = perf_counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with (args.theory_dir / "angular_power_diagnostics.csv").open(newline="") as stream:
        diagnostics = list(csv.DictReader(stream))
    with np.load(args.theory_dir / "angular_power_theory.npz", allow_pickle=False) as source:
        if str(source["mask_pixel_sha256"].item()) == "":
            raise ValueError("reference archive has no mask fingerprint")
        ell = np.asarray(source["ell"], dtype=np.int64)
        expected_hash = str(source["mask_pixel_sha256"].item())
        n_iter = int(source["mask_sht_iterations"])
        reference = {key: np.array(source[key]) for key in (
            "shell_linear", "shell_two_halo", "shell_linear_pseudo_over_fsky",
            "shell_two_halo_pseudo_over_fsky",
        )}
    if not diagnostics or args.ell_min < ell[0] or args.ell_stop > ell[-1] + 1:
        raise ValueError("requested comparison range is outside the reference archive")
    if args.bin_width < 1 or args.ell_stop <= args.ell_min:
        raise ValueError("invalid bandpower controls")
    edges = np.r_[np.arange(args.ell_min, args.ell_stop, args.bin_width), args.ell_stop]
    binning = mode_count_binning(ell, edges)
    mass_map = read_pinocchio_mass_map_fits(args.mass_map)
    pixel_hash = hashlib.sha256(np.ascontiguousarray(mass_map.pixel, dtype=np.int64).tobytes()).hexdigest()
    if (pixel_hash != expected_hash or mass_map.ordering != "RING"
            or mass_map.nside != int(diagnostics[0]["nside"])):
        raise ValueError("mass-map mask, NSIDE, or ordering differs from reference")
    print("[mask-response] building reference workspace", flush=True)
    coupling = validation.build_mask_coupling(
        mass_map.pixel, mass_map.nside, int(ell[-1]), bin_width=args.bin_width, n_iter=n_iter
    )

    def forward(values):
        return validation.couple_theory_component(values, ell, coupling)

    def progress(done, total):
        if done == 1 or done % 10 == 0 or done == total:
            print(f"[mask-response] band {done}/{total}; elapsed={perf_counter()-started:.1f}s", flush=True)

    response = binned_self_adjoint_response(forward, ell, edges, progress)
    errors, archive_differences = {}, {}
    for name in ("linear", "two_halo"):
        direct = reference[f"shell_{name}_pseudo_over_fsky"] @ binning.T
        cached = reference[f"shell_{name}"] @ response.T
        archive_differences[name] = float(np.max(np.abs(cached / direct - 1)))
    probes = {
        "white": np.ones_like(ell, dtype=np.float64),
        "localized": .001 + np.exp(-.5 * ((ell - 700.) / 80.) ** 2),
        "red": 1. / (ell + 1.) ** 1.8,
    }
    for name, spectrum in probes.items():
        direct = binning @ forward(spectrum)
        cached = response @ spectrum
        errors[f"direct_{name}"] = float(np.max(np.abs(cached / direct - 1)))
    audit = dict(
        mask_pixel_sha256=pixel_hash, nside=mass_map.nside, mask_sht_iterations=n_iter,
        f_sky=coupling.f_sky, ell_min=args.ell_min, ell_stop=args.ell_stop,
        bin_width=args.bin_width, maximum_relative_errors=errors,
        maximum_relative_archive_differences=archive_differences,
        reference_template_count=coupling.reference_field.n_temp,
        check_rtol=args.check_rtol, elapsed_seconds=perf_counter() - started,
        theory_reference=str(args.theory_dir), mass_map=str(args.mass_map),
        convention="constant_deprojected_pseudo_cl_over_f_sky",
    )
    if max(errors.values()) > args.check_rtol:
        args.output.with_suffix(".failed.json").write_text(json.dumps(audit, indent=2) + "\n")
        raise ValueError(f"reciprocity/forward comparison failed: {errors}")
    np.savez_compressed(
        args.output, ell=ell, edges=edges, binning=binning, response=response,
        mask_pixel_sha256=np.asarray(pixel_hash), mask_sht_iterations=np.asarray(n_iter),
        f_sky=np.asarray(coupling.f_sky), nside=np.asarray(mass_map.nside),
        reference_template_count=np.asarray(coupling.reference_field.n_temp),
    )
    args.output.with_suffix(".json").write_text(json.dumps(audit, indent=2) + "\n")
    print(f"[mask-response] saved {args.output}: {errors}", flush=True)
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theory-dir", type=Path, required=True)
    parser.add_argument("--mass-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ell-min", type=int, default=20)
    parser.add_argument("--ell-stop", type=int, default=2000)
    parser.add_argument("--bin-width", type=int, default=20)
    parser.add_argument("--check-rtol", type=float, default=5e-4)
    prepare(parser.parse_args())
