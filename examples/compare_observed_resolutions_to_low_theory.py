#!/usr/bin/env python3
"""Compare two measured resolutions against the completed low-resolution theory."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from geppetto.io import read_pinocchio_mass_map_fits


@dataclass(frozen=True)
class LowResolutionReference:
    """Low-resolution measured/theory spectra and common map metadata."""

    ell: np.ndarray
    observed_shell: np.ndarray
    observed_sum: np.ndarray
    theory_shell: np.ndarray
    theory_sum: np.ndarray
    mean_total_counts_per_pixel: np.ndarray
    segment_index: np.ndarray
    z_lo: np.ndarray
    z_hi: np.ndarray
    nside: int
    f_sky: float
    mask_pixel_sha256: str
    mask_sht_iterations: int


@dataclass(frozen=True)
class ObservedResolution:
    """Measured shell and summed spectra for one map resolution."""

    ell: np.ndarray
    observed_shell: np.ndarray
    observed_sum: np.ndarray
    mean_total_counts_per_pixel: np.ndarray
    segment_index: np.ndarray
    z_lo: np.ndarray
    z_hi: np.ndarray
    nside: int
    f_sky: float
    mask_pixel_sha256: str


@dataclass(frozen=True)
class BinnedTheoryResolutionComparison:
    """Common bins for low theory and low/high measured maps."""

    ell_min: np.ndarray
    ell_max: np.ndarray
    ell_effective: np.ndarray
    theory_shell: np.ndarray
    low_shell: np.ndarray
    high_shell: np.ndarray
    theory_sum: np.ndarray
    low_sum: np.ndarray
    high_sum: np.ndarray
    segment_index: np.ndarray
    z_lo: np.ndarray
    z_hi: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--low-dir", type=Path, required=True)
    parser.add_argument("--high-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ell-min", type=int, default=20)
    parser.add_argument("--bin-width", type=int, default=20)
    parser.add_argument("--representative-segment", type=int, default=23)
    parser.add_argument("--png-dpi", type=int, default=300)
    parser.add_argument(
        "--reuse-cache",
        action="store_true",
        help="reuse a compatible high-resolution observed-spectrum cache",
    )
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise ValueError(f"cannot read CSV file: {path}") from exc
    if not rows:
        raise ValueError(f"CSV file is empty: {path}")
    return rows


def load_low_resolution_reference(path: Path) -> LowResolutionReference:
    """Load the completed schema-v3 low-resolution validation product."""

    archive_path = path / "angular_power_theory.npz"
    required = {
        "validation_schema_version",
        "ell",
        "observed_shell",
        "observed_sum",
        "shell_linear_pseudo_over_fsky",
        "shell_one_halo_pseudo_over_fsky",
        "shell_particle_shot_noise_pseudo_over_fsky",
        "summed_linear_pseudo_over_fsky",
        "summed_one_halo_pseudo_over_fsky",
        "summed_particle_shot_noise_pseudo_over_fsky",
        "mask_pixel_sha256",
        "mask_sht_iterations",
    }
    try:
        with np.load(archive_path, allow_pickle=False) as source:
            missing = required - set(source.files)
            if missing:
                raise ValueError(f"{archive_path} is missing arrays: {sorted(missing)}")
            if int(np.asarray(source["validation_schema_version"])) != 3:
                raise ValueError(f"{archive_path} is not a schema-v3 validation archive")
            ell = np.array(source["ell"], dtype=np.int64, copy=True)
            observed_shell = np.array(source["observed_shell"], dtype=np.float64, copy=True)
            observed_sum = np.array(source["observed_sum"], dtype=np.float64, copy=True)
            theory_shell = sum(
                np.array(source[key], dtype=np.float64, copy=True)
                for key in (
                    "shell_linear_pseudo_over_fsky",
                    "shell_one_halo_pseudo_over_fsky",
                    "shell_particle_shot_noise_pseudo_over_fsky",
                )
            )
            theory_sum = sum(
                np.array(source[key], dtype=np.float64, copy=True)
                for key in (
                    "summed_linear_pseudo_over_fsky",
                    "summed_one_halo_pseudo_over_fsky",
                    "summed_particle_shot_noise_pseudo_over_fsky",
                )
            )
            mask_hash = str(np.asarray(source["mask_pixel_sha256"]).item())
            mask_iterations = int(np.asarray(source["mask_sht_iterations"]))
    except OSError as exc:
        raise ValueError(f"cannot read angular-power archive: {archive_path}") from exc

    rows = _read_csv(path / "angular_power_diagnostics.csv")
    segment_index = np.asarray([int(row["segment_index"]) for row in rows], dtype=np.int64)
    z_lo = np.asarray([float(row["z_lo"]) for row in rows], dtype=np.float64)
    z_hi = np.asarray([float(row["z_hi"]) for row in rows], dtype=np.float64)
    mean_total = np.asarray(
        [float(row["mean_total_counts_per_pixel"]) for row in rows],
        dtype=np.float64,
    )
    nside_values = {int(row["nside"]) for row in rows}
    f_sky_values = np.asarray([float(row["f_sky"]) for row in rows], dtype=np.float64)
    expected_shell_shape = (segment_index.size, ell.size)
    if len(nside_values) != 1 or not np.allclose(f_sky_values, f_sky_values[0]):
        raise ValueError("low-resolution NSIDE and f_sky must be constant across shells")
    if observed_shell.shape != expected_shell_shape or theory_shell.shape != expected_shell_shape:
        raise ValueError("low-resolution shell spectra have inconsistent shapes")
    if observed_sum.shape != ell.shape or theory_sum.shape != ell.shape:
        raise ValueError("low-resolution summed spectra have inconsistent shapes")
    arrays = (observed_shell, observed_sum, theory_shell, theory_sum)
    if any(not np.all(np.isfinite(values)) for values in arrays):
        raise ValueError("low-resolution validation contains non-finite spectra")
    return LowResolutionReference(
        ell=ell,
        observed_shell=observed_shell,
        observed_sum=observed_sum,
        theory_shell=theory_shell,
        theory_sum=theory_sum,
        mean_total_counts_per_pixel=mean_total,
        segment_index=segment_index,
        z_lo=z_lo,
        z_hi=z_hi,
        nside=nside_values.pop(),
        f_sky=float(f_sky_values[0]),
        mask_pixel_sha256=mask_hash,
        mask_sht_iterations=mask_iterations,
    )


def _resolve_input_path(value: str, manifest: Path) -> Path:
    candidate = Path(value)
    if candidate.exists() or candidate.is_absolute():
        return candidate
    relative = manifest.parent / candidate
    return relative if relative.exists() else candidate


def _load_high_manifest(
    path: Path,
    low: LowResolutionReference,
) -> list[dict[str, str]]:
    rows = _read_csv(path)
    required = {"segment_index", "z_lo", "z_hi", "mass_map_path", "output_npz"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"high-resolution manifest is missing columns: {sorted(missing)}")
    by_segment = {int(row["segment_index"]): row for row in rows}
    if set(by_segment) != set(low.segment_index):
        raise ValueError("low- and high-resolution segment indices differ")
    ordered = [by_segment[int(segment)] for segment in low.segment_index]
    high_z_lo = np.asarray([float(row["z_lo"]) for row in ordered])
    high_z_hi = np.asarray([float(row["z_hi"]) for row in ordered])
    if not np.allclose(low.z_lo, high_z_lo, rtol=0.0, atol=1.0e-8) or not np.allclose(
        low.z_hi, high_z_hi, rtol=0.0, atol=1.0e-8
    ):
        raise ValueError("low- and high-resolution shell boundaries differ")
    return ordered


def _estimate_pseudo_cls(
    compact_counts: np.ndarray,
    pixels: np.ndarray,
    *,
    module: Any,
    mask: np.ndarray,
    template: np.ndarray,
    f_sky: float,
    nside: int,
    lmax: int,
    n_iter: int,
    full_sky_buffer: np.ndarray,
) -> np.ndarray:
    mean_count = float(np.mean(compact_counts))
    if not np.isfinite(mean_count) or mean_count <= 0.0:
        raise ValueError("compact count map must have a positive finite mean")
    full_sky_buffer.fill(0.0)
    full_sky_buffer[pixels] = compact_counts / mean_count
    field = module.NmtField(
        mask,
        full_sky_buffer[None, :],
        spin=0,
        templates=template,
        n_iter=n_iter,
        n_iter_mask=n_iter,
        lmax=lmax,
        lmax_mask=min(2 * lmax, 3 * nside - 1),
        masked_on_input=True,
        lite=True,
    )
    result = np.asarray(module.compute_coupled_cell(field, field)[0], dtype=np.float64)
    del field
    gc.collect()
    return result / f_sky


def measure_high_resolution(
    manifest: Path,
    low: LowResolutionReference,
) -> ObservedResolution:
    """Measure high-resolution maps using the low-validation pseudo-Cl convention."""

    try:
        import pymaster as nmt
    except ImportError as exc:  # pragma: no cover - optional install
        raise ImportError("observed comparison requires NaMaster") from exc
    rows = _load_high_manifest(manifest, low)
    first_path = _resolve_input_path(rows[0]["mass_map_path"], manifest)
    first_map = read_pinocchio_mass_map_fits(first_path)
    pixels = np.array(first_map.pixel, dtype=np.int64, copy=True)
    if first_map.ordering.strip().upper() != "RING":
        raise ValueError("high-resolution validation requires RING maps")
    if first_map.nside != low.nside:
        raise ValueError("low- and high-resolution map NSIDE values differ")
    mask_hash = hashlib.sha256(np.ascontiguousarray(pixels).view(np.uint8)).hexdigest()
    if mask_hash != low.mask_pixel_sha256:
        raise ValueError("low- and high-resolution compact masks differ")

    npix = 12 * low.nside**2
    mask = np.zeros(npix, dtype=np.float64)
    mask[pixels] = 1.0
    f_sky = float(np.mean(mask))
    if not np.isclose(f_sky, low.f_sky, rtol=0.0, atol=1.0e-12):
        raise ValueError("low- and high-resolution f_sky values differ")
    template = mask[None, None, :]
    full_sky_buffer = np.zeros(npix, dtype=np.float64)
    observed_shell = np.empty((len(rows), low.ell.size), dtype=np.float64)
    mean_total = np.empty(len(rows), dtype=np.float64)
    summed_counts = np.zeros(pixels.size, dtype=np.float64)
    lmax = int(low.ell[-1])
    if not np.array_equal(low.ell, np.arange(2, lmax + 1)):
        raise ValueError("low-resolution multipoles must be contiguous from ell=2")

    measurement_started = perf_counter()
    for index, row in enumerate(rows):
        shell_started = perf_counter()
        mass_map = first_map if index == 0 else read_pinocchio_mass_map_fits(
            _resolve_input_path(row["mass_map_path"], manifest)
        )
        if mass_map.nside != low.nside or not np.array_equal(mass_map.pixel, pixels):
            raise ValueError("high-resolution mass maps do not share compact RING rows")
        output_path = _resolve_input_path(row["output_npz"], manifest)
        try:
            with np.load(output_path, allow_pickle=False) as painted:
                total_counts = np.array(
                    painted["nfw_particle_counts"], dtype=np.float64, copy=True, order="C"
                )
        except (OSError, KeyError) as exc:
            raise ValueError(f"cannot read painted NFW counts: {output_path}") from exc
        if total_counts.shape != mass_map.temperature.shape:
            raise ValueError(f"painted and PINOCCHIO map shapes differ: {output_path}")
        np.add(total_counts, mass_map.temperature, out=total_counts)
        mean_total[index] = np.mean(total_counts, dtype=np.float64)
        summed_counts += total_counts
        observed_shell[index] = _estimate_pseudo_cls(
            total_counts,
            pixels,
            module=nmt,
            mask=mask,
            template=template,
            f_sky=f_sky,
            nside=low.nside,
            lmax=lmax,
            n_iter=low.mask_sht_iterations,
            full_sky_buffer=full_sky_buffer,
        )[low.ell]
        del mass_map, total_counts
        print(
            f"[quick-compare] measured shell {index + 1}/{len(rows)} in "
            f"{perf_counter() - shell_started:.1f}s "
            f"(total {perf_counter() - measurement_started:.1f}s)",
            flush=True,
        )
    print("[quick-compare] measuring summed map", flush=True)
    observed_sum = _estimate_pseudo_cls(
        summed_counts,
        pixels,
        module=nmt,
        mask=mask,
        template=template,
        f_sky=f_sky,
        nside=low.nside,
        lmax=lmax,
        n_iter=low.mask_sht_iterations,
        full_sky_buffer=full_sky_buffer,
    )[low.ell]
    return ObservedResolution(
        ell=low.ell,
        observed_shell=observed_shell,
        observed_sum=observed_sum,
        mean_total_counts_per_pixel=mean_total,
        segment_index=low.segment_index,
        z_lo=low.z_lo,
        z_hi=low.z_hi,
        nside=low.nside,
        f_sky=f_sky,
        mask_pixel_sha256=mask_hash,
    )


def save_observed_cache(data: ObservedResolution, path: Path) -> Path:
    """Save lean measured spectra so plotting can be repeated without SHTs."""

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        schema_version=np.asarray(2, dtype=np.int64),
        ell=data.ell,
        observed_shell=data.observed_shell,
        observed_sum=data.observed_sum,
        mean_total_counts_per_pixel=data.mean_total_counts_per_pixel,
        segment_index=data.segment_index,
        z_lo=data.z_lo,
        z_hi=data.z_hi,
        nside=np.asarray(data.nside, dtype=np.int64),
        f_sky=np.asarray(data.f_sky),
        mask_pixel_sha256=np.asarray(data.mask_pixel_sha256),
    )
    return path


def load_observed_cache(path: Path, low: LowResolutionReference) -> ObservedResolution:
    """Load and validate a high-resolution observed-spectrum cache."""

    with np.load(path, allow_pickle=False) as source:
        schema = int(np.asarray(source["schema_version"]))
        if schema not in {1, 2}:
            raise ValueError("unsupported observed-resolution cache schema")
        observed_shell = np.array(source["observed_shell"], dtype=np.float64, copy=True)
        if schema == 2:
            mean_total = np.array(
                source["mean_total_counts_per_pixel"], dtype=np.float64, copy=True
            )
        else:
            mean_total = np.full(observed_shell.shape[0], np.nan, dtype=np.float64)
        data = ObservedResolution(
            ell=np.array(source["ell"], dtype=np.int64, copy=True),
            observed_shell=observed_shell,
            observed_sum=np.array(source["observed_sum"], dtype=np.float64, copy=True),
            mean_total_counts_per_pixel=mean_total,
            segment_index=np.array(source["segment_index"], dtype=np.int64, copy=True),
            z_lo=np.array(source["z_lo"], dtype=np.float64, copy=True),
            z_hi=np.array(source["z_hi"], dtype=np.float64, copy=True),
            nside=int(np.asarray(source["nside"])),
            f_sky=float(np.asarray(source["f_sky"])),
            mask_pixel_sha256=str(np.asarray(source["mask_pixel_sha256"]).item()),
        )
    _validate_high_resolution(low, data)
    return data


def _validate_high_resolution(
    low: LowResolutionReference,
    high: ObservedResolution,
) -> None:
    if not np.array_equal(low.ell, high.ell):
        raise ValueError("low- and high-resolution multipoles differ")
    if not np.array_equal(low.segment_index, high.segment_index):
        raise ValueError("low- and high-resolution segment order differs")
    if low.nside != high.nside or low.mask_pixel_sha256 != high.mask_pixel_sha256:
        raise ValueError("low- and high-resolution map geometry differs")
    if not np.isclose(low.f_sky, high.f_sky, rtol=0.0, atol=1.0e-12):
        raise ValueError("low- and high-resolution f_sky values differ")
    if not np.allclose(low.z_lo, high.z_lo) or not np.allclose(low.z_hi, high.z_hi):
        raise ValueError("low- and high-resolution shell boundaries differ")
    if high.mean_total_counts_per_pixel.shape != low.mean_total_counts_per_pixel.shape:
        raise ValueError("low- and high-resolution shell mean arrays differ")


def bin_comparison(
    low: LowResolutionReference,
    high: ObservedResolution,
    *,
    ell_min: int,
    bin_width: int,
) -> BinnedTheoryResolutionComparison:
    """Mode-count bin low theory and low/high measured spectra."""

    if ell_min < 0 or bin_width < 1:
        raise ValueError("ell_min must be non-negative and bin_width must be positive")
    _validate_high_resolution(low, high)
    lower_edges: list[int] = []
    upper_edges: list[int] = []
    effective: list[float] = []
    theory_shell: list[np.ndarray] = []
    low_shell: list[np.ndarray] = []
    high_shell: list[np.ndarray] = []
    theory_sum: list[float] = []
    low_sum: list[float] = []
    high_sum: list[float] = []
    first = max(ell_min, int(low.ell[0]))
    for lower in range(first, int(low.ell[-1]) + 1, bin_width):
        upper = min(lower + bin_width, int(low.ell[-1]) + 1)
        selected = (low.ell >= lower) & (low.ell < upper)
        if not np.any(selected):
            continue
        weights = 2.0 * low.ell[selected] + 1.0
        lower_edges.append(lower)
        upper_edges.append(upper - 1)
        effective.append(float(np.average(low.ell[selected], weights=weights)))
        theory_shell.append(np.average(low.theory_shell[:, selected], axis=1, weights=weights))
        low_shell.append(np.average(low.observed_shell[:, selected], axis=1, weights=weights))
        high_shell.append(np.average(high.observed_shell[:, selected], axis=1, weights=weights))
        theory_sum.append(float(np.average(low.theory_sum[selected], weights=weights)))
        low_sum.append(float(np.average(low.observed_sum[selected], weights=weights)))
        high_sum.append(float(np.average(high.observed_sum[selected], weights=weights)))
    if not lower_edges:
        raise ValueError("no multipoles remain after applying ell_min")
    return BinnedTheoryResolutionComparison(
        ell_min=np.asarray(lower_edges),
        ell_max=np.asarray(upper_edges),
        ell_effective=np.asarray(effective),
        theory_shell=np.stack(theory_shell, axis=1),
        low_shell=np.stack(low_shell, axis=1),
        high_shell=np.stack(high_shell, axis=1),
        theory_sum=np.asarray(theory_sum),
        low_sum=np.asarray(low_sum),
        high_sum=np.asarray(high_sum),
        segment_index=low.segment_index,
        z_lo=low.z_lo,
        z_hi=low.z_hi,
    )


def write_comparison_csv(data: BinnedTheoryResolutionComparison, path: Path) -> Path:
    """Write theory and both measured resolutions for every shell and sum."""

    rows: list[dict[str, object]] = []
    maps = [("summed", None, data.theory_sum, data.low_sum, data.high_sum)]
    maps.extend(
        (
            f"segment_{segment}",
            index,
            data.theory_shell[index],
            data.low_shell[index],
            data.high_shell[index],
        )
        for index, segment in enumerate(data.segment_index)
    )
    for name, shell_index, theory, low, high in maps:
        for index, ell in enumerate(data.ell_effective):
            rows.append(
                {
                    "map": name,
                    "z_lo": "" if shell_index is None else data.z_lo[shell_index],
                    "z_hi": "" if shell_index is None else data.z_hi[shell_index],
                    "ell_min": data.ell_min[index],
                    "ell_max": data.ell_max[index],
                    "ell_effective": ell,
                    "low_theory_cl": theory[index],
                    "low_observed_cl": low[index],
                    "high_observed_cl": high[index],
                    "low_observed_over_theory": low[index] / theory[index],
                    "high_observed_over_theory": high[index] / theory[index],
                    "high_over_low_observed": high[index] / low[index],
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def render_comparison(
    data: BinnedTheoryResolutionComparison,
    output_dir: Path,
    *,
    representative_segment: int,
    png_dpi: int,
) -> list[Path]:
    """Render summed and representative-shell spectra with theory ratios."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional install
        raise ImportError("comparison plotting requires matplotlib") from exc
    matches = np.flatnonzero(data.segment_index == representative_segment)
    if matches.size != 1:
        raise ValueError(f"representative segment {representative_segment} is unavailable")
    shell = int(matches[0])
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(7.2, 4.8),
        sharex="col",
        gridspec_kw={"height_ratios": (3.0, 1.0), "hspace": 0.05, "wspace": 0.28},
    )
    spectra = (
        (data.theory_sum, data.low_sum, data.high_sum, "Summed map"),
        (
            data.theory_shell[shell],
            data.low_shell[shell],
            data.high_shell[shell],
            (
                f"Segment {representative_segment}: "
                f"{data.z_lo[shell]:.3f} < z < {data.z_hi[shell]:.3f}"
            ),
        ),
    )
    ell = data.ell_effective
    factor = ell * (ell + 1.0) / (2.0 * np.pi)
    for column, (theory, low, high, title) in enumerate(spectra):
        axes[0, column].loglog(ell, factor * theory, color="black", label="2160³ theory")
        axes[0, column].loglog(ell, factor * low, color="#0072B2", label="2160³ map")
        axes[0, column].loglog(ell, factor * high, color="#D55E00", label="4096³ map")
        axes[0, column].set_title(title)
        axes[0, column].legend(frameon=False, fontsize=8)
        axes[1, column].semilogx(ell, low / theory, color="#0072B2")
        axes[1, column].semilogx(ell, high / theory, color="#D55E00")
        axes[1, column].axhline(1.0, color="0.25", linewidth=0.8)
        axes[1, column].set_xlabel(r"Multipole $\ell$")
        ratios = np.concatenate((low / theory, high / theory))
        finite = ratios[np.isfinite(ratios)]
        extent = max(0.1, float(np.nanpercentile(np.abs(finite - 1.0), 98)))
        axes[1, column].set_ylim(max(0.0, 1.0 - 1.1 * extent), 1.0 + 1.1 * extent)
    axes[0, 0].set_ylabel(r"$\ell(\ell+1)C_\ell/(2\pi)$")
    axes[1, 0].set_ylabel("Map / low theory")
    base = output_dir / "angular_power_low_theory_resolution_check"
    outputs = [base.with_suffix(".pdf"), base.with_suffix(".png")]
    figure.savefig(outputs[0], bbox_inches="tight")
    figure.savefig(outputs[1], dpi=png_dpi, bbox_inches="tight")
    plt.close(figure)
    return outputs


def render_all_shells(
    data: BinnedTheoryResolutionComparison,
    output_dir: Path,
    *,
    png_dpi: int,
) -> list[Path]:
    """Render measured-to-theory ratios for every shell on common axes."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional install
        raise ImportError("comparison plotting requires matplotlib") from exc
    n_columns = 5
    n_rows = int(np.ceil(data.segment_index.size / n_columns))
    figure, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(11.0, 2.0 * n_rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    ell = data.ell_effective
    for shell, axis in enumerate(axes.flat):
        if shell >= data.segment_index.size:
            axis.set_visible(False)
            continue
        axis.semilogx(
            ell,
            data.low_shell[shell] / data.theory_shell[shell],
            color="#0072B2",
            linewidth=1.0,
            label="2160³ map",
        )
        axis.semilogx(
            ell,
            data.high_shell[shell] / data.theory_shell[shell],
            color="#D55E00",
            linewidth=1.0,
            label="4096³ map",
        )
        axis.axhline(1.0, color="0.25", linewidth=0.7)
        axis.set_ylim(0.0, 1.15)
        axis.set_title(
            f"{data.z_lo[shell]:.3f} < z < {data.z_hi[shell]:.3f}",
            fontsize=8,
        )
        axis.tick_params(labelsize=7)
    axes[0, 0].legend(frameon=False, fontsize=7, loc="lower left")
    figure.supxlabel(r"Multipole $\ell$", fontsize=10)
    figure.supylabel("Map / low-resolution theory", fontsize=10)
    figure.subplots_adjust(left=0.07, right=0.995, bottom=0.06, top=0.97, hspace=0.3)
    base = output_dir / "angular_power_low_theory_all_shells"
    outputs = [base.with_suffix(".pdf"), base.with_suffix(".png")]
    figure.savefig(outputs[0], bbox_inches="tight")
    figure.savefig(outputs[1], dpi=png_dpi, bbox_inches="tight")
    plt.close(figure)
    return outputs


def main() -> None:
    args = parse_args()
    low = load_low_resolution_reference(args.low_dir)
    cache = args.output_dir / "high_observed_spectra.npz"
    if args.reuse_cache and cache.exists():
        print(f"[quick-compare] reusing {cache}", flush=True)
        high = load_observed_cache(cache, low)
    else:
        high = measure_high_resolution(args.high_manifest, low)
        save_observed_cache(high, cache)
        print(f"Wrote {cache}", flush=True)
    comparison = bin_comparison(
        low,
        high,
        ell_min=args.ell_min,
        bin_width=args.bin_width,
    )
    csv_path = write_comparison_csv(
        comparison,
        args.output_dir / "angular_power_low_theory_resolution_check.csv",
    )
    figures = render_comparison(
        comparison,
        args.output_dir,
        representative_segment=args.representative_segment,
        png_dpi=args.png_dpi,
    )
    figures.extend(
        render_all_shells(
            comparison,
            args.output_dir,
            png_dpi=args.png_dpi,
        )
    )
    print(f"Wrote {csv_path}")
    for path in figures:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
