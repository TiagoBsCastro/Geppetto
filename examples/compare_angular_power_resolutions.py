#!/usr/bin/env python3
"""Compare measured angular spectra from phase-matched PINOCCHIO resolutions."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ResolutionSpectra:
    """Measured spectra and shell metadata from one validation run."""

    ell: np.ndarray
    observed_shell: np.ndarray
    observed_sum: np.ndarray
    segment_index: np.ndarray
    z_lo: np.ndarray
    z_hi: np.ndarray
    nside: int
    f_sky: float
    mask_pixel_sha256: str


@dataclass(frozen=True)
class BinnedResolutionComparison:
    """Common binned spectra for two phase-matched resolutions."""

    ell_min: np.ndarray
    ell_max: np.ndarray
    ell_effective: np.ndarray
    low_shell: np.ndarray
    high_shell: np.ndarray
    shell_ratio: np.ndarray
    low_sum: np.ndarray
    high_sum: np.ndarray
    summed_ratio: np.ndarray
    segment_index: np.ndarray
    z_lo: np.ndarray
    z_hi: np.ndarray
    nside: int
    f_sky: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--low-dir", type=Path, required=True)
    parser.add_argument("--high-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ell-min", type=int, default=20)
    parser.add_argument("--bin-width", type=int, default=20)
    parser.add_argument("--representative-segment", type=int, default=23)
    parser.add_argument("--png-dpi", type=int, default=300)
    return parser.parse_args()


def _read_diagnostics(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise ValueError(f"cannot read angular-power diagnostics: {path}") from exc
    required = {"segment_index", "z_lo", "z_hi", "nside", "f_sky"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"angular-power diagnostics are empty or incomplete: {path}")
    return rows


def load_resolution_spectra(path: Path) -> ResolutionSpectra:
    """Load one schema-v3 angular validation product."""

    archive_path = path / "angular_power_theory.npz"
    try:
        with np.load(archive_path, allow_pickle=False) as source:
            required = {
                "validation_schema_version",
                "ell",
                "observed_shell",
                "observed_sum",
                "mask_pixel_sha256",
            }
            missing = required - set(source.files)
            if missing:
                raise ValueError(f"{archive_path} is missing arrays: {sorted(missing)}")
            schema = np.asarray(source["validation_schema_version"])
            if schema.shape != () or int(schema) != 3:
                raise ValueError(f"{archive_path} is not a schema-v3 validation archive")
            ell = np.array(source["ell"], dtype=np.int64, copy=True)
            observed_shell = np.array(source["observed_shell"], dtype=np.float64, copy=True)
            observed_sum = np.array(source["observed_sum"], dtype=np.float64, copy=True)
            mask_hash = str(np.asarray(source["mask_pixel_sha256"]).item())
    except OSError as exc:
        raise ValueError(f"cannot read angular-power archive: {archive_path}") from exc

    if ell.ndim != 1 or ell.size == 0 or np.any(np.diff(ell) <= 0):
        raise ValueError(f"{archive_path} has invalid multipoles")
    if observed_shell.ndim != 2 or observed_shell.shape[1] != ell.size:
        raise ValueError(f"{archive_path} has an invalid observed_shell shape")
    if observed_sum.shape != ell.shape:
        raise ValueError(f"{archive_path} has an invalid observed_sum shape")
    if not np.all(np.isfinite(observed_shell)) or not np.all(np.isfinite(observed_sum)):
        raise ValueError(f"{archive_path} contains non-finite measured spectra")

    rows = _read_diagnostics(path / "angular_power_diagnostics.csv")
    if len(rows) != observed_shell.shape[0]:
        raise ValueError("diagnostic rows do not match the observed shell count")
    segment_index = np.asarray([int(row["segment_index"]) for row in rows], dtype=np.int64)
    if np.unique(segment_index).size != segment_index.size:
        raise ValueError("segment indices must be unique")
    z_lo = np.asarray([float(row["z_lo"]) for row in rows], dtype=np.float64)
    z_hi = np.asarray([float(row["z_hi"]) for row in rows], dtype=np.float64)
    nside_values = {int(row["nside"]) for row in rows}
    f_sky_values = np.asarray([float(row["f_sky"]) for row in rows], dtype=np.float64)
    if len(nside_values) != 1 or not np.allclose(f_sky_values, f_sky_values[0]):
        raise ValueError("NSIDE and f_sky must be constant across shells")
    return ResolutionSpectra(
        ell=ell,
        observed_shell=observed_shell,
        observed_sum=observed_sum,
        segment_index=segment_index,
        z_lo=z_lo,
        z_hi=z_hi,
        nside=nside_values.pop(),
        f_sky=float(f_sky_values[0]),
        mask_pixel_sha256=mask_hash,
    )


def _align_high_resolution(
    low: ResolutionSpectra,
    high: ResolutionSpectra,
) -> ResolutionSpectra:
    if not np.array_equal(low.ell, high.ell):
        raise ValueError("low- and high-resolution multipoles differ")
    if low.nside != high.nside:
        raise ValueError("low- and high-resolution HEALPix NSIDE values differ")
    if not np.isclose(low.f_sky, high.f_sky, rtol=0.0, atol=1.0e-12):
        raise ValueError("low- and high-resolution f_sky values differ")
    if low.mask_pixel_sha256 != high.mask_pixel_sha256:
        raise ValueError("low- and high-resolution compact masks differ")
    if set(low.segment_index) != set(high.segment_index):
        raise ValueError("low- and high-resolution segment indices differ")
    high_lookup = {segment: index for index, segment in enumerate(high.segment_index)}
    order = np.asarray([high_lookup[segment] for segment in low.segment_index], dtype=np.int64)
    if not np.allclose(low.z_lo, high.z_lo[order], rtol=0.0, atol=1.0e-8) or not np.allclose(
        low.z_hi, high.z_hi[order], rtol=0.0, atol=1.0e-8
    ):
        raise ValueError("low- and high-resolution shell boundaries differ")
    return ResolutionSpectra(
        ell=high.ell,
        observed_shell=high.observed_shell[order],
        observed_sum=high.observed_sum,
        segment_index=high.segment_index[order],
        z_lo=high.z_lo[order],
        z_hi=high.z_hi[order],
        nside=high.nside,
        f_sky=high.f_sky,
        mask_pixel_sha256=high.mask_pixel_sha256,
    )


def compare_resolutions(
    low: ResolutionSpectra,
    high: ResolutionSpectra,
    *,
    ell_min: int = 20,
    bin_width: int = 20,
) -> BinnedResolutionComparison:
    """Return common mode-count-weighted bins for two validation runs."""

    if ell_min < 0 or bin_width < 1:
        raise ValueError("ell_min must be non-negative and bin_width must be positive")
    high = _align_high_resolution(low, high)
    lower_edges: list[int] = []
    upper_edges: list[int] = []
    effective: list[float] = []
    low_shell_bins: list[np.ndarray] = []
    high_shell_bins: list[np.ndarray] = []
    low_sum_bins: list[float] = []
    high_sum_bins: list[float] = []
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
        low_shell_bins.append(np.average(low.observed_shell[:, selected], axis=1, weights=weights))
        high_shell_bins.append(
            np.average(high.observed_shell[:, selected], axis=1, weights=weights)
        )
        low_sum_bins.append(float(np.average(low.observed_sum[selected], weights=weights)))
        high_sum_bins.append(float(np.average(high.observed_sum[selected], weights=weights)))
    if not lower_edges:
        raise ValueError("no multipoles remain after applying ell_min")

    low_shell_values = np.stack(low_shell_bins, axis=1)
    high_shell_values = np.stack(high_shell_bins, axis=1)
    low_sum_values = np.asarray(low_sum_bins)
    high_sum_values = np.asarray(high_sum_bins)
    return BinnedResolutionComparison(
        ell_min=np.asarray(lower_edges, dtype=np.int64),
        ell_max=np.asarray(upper_edges, dtype=np.int64),
        ell_effective=np.asarray(effective),
        low_shell=low_shell_values,
        high_shell=high_shell_values,
        shell_ratio=np.divide(high_shell_values, low_shell_values),
        low_sum=low_sum_values,
        high_sum=high_sum_values,
        summed_ratio=np.divide(high_sum_values, low_sum_values),
        segment_index=low.segment_index,
        z_lo=low.z_lo,
        z_hi=low.z_hi,
        nside=low.nside,
        f_sky=low.f_sky,
    )


def write_comparison_csv(comparison: BinnedResolutionComparison, path: Path) -> Path:
    """Write all shell and summed resolution ratios."""

    rows: list[dict[str, object]] = []
    for shell_row, segment in enumerate(comparison.segment_index):
        for ell_row, ell_value in enumerate(comparison.ell_effective):
            rows.append(
                {
                    "map": f"segment_{segment}",
                    "z_lo": comparison.z_lo[shell_row],
                    "z_hi": comparison.z_hi[shell_row],
                    "ell_min": comparison.ell_min[ell_row],
                    "ell_max": comparison.ell_max[ell_row],
                    "ell_effective": ell_value,
                    "low_cl": comparison.low_shell[shell_row, ell_row],
                    "high_cl": comparison.high_shell[shell_row, ell_row],
                    "high_over_low": comparison.shell_ratio[shell_row, ell_row],
                }
            )
    for ell_row, ell_value in enumerate(comparison.ell_effective):
        rows.append(
            {
                "map": "summed",
                "z_lo": "",
                "z_hi": "",
                "ell_min": comparison.ell_min[ell_row],
                "ell_max": comparison.ell_max[ell_row],
                "ell_effective": ell_value,
                "low_cl": comparison.low_sum[ell_row],
                "high_cl": comparison.high_sum[ell_row],
                "high_over_low": comparison.summed_ratio[ell_row],
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _save_figure(figure: object, base: Path, png_dpi: int) -> list[Path]:
    outputs = [base.with_suffix(".pdf"), base.with_suffix(".png")]
    figure.savefig(outputs[0], bbox_inches="tight")
    figure.savefig(outputs[1], dpi=png_dpi, bbox_inches="tight")
    return outputs


def render_comparison_figures(
    comparison: BinnedResolutionComparison,
    output_dir: Path,
    *,
    representative_segment: int = 23,
    png_dpi: int = 300,
) -> list[Path]:
    """Render summed, representative-shell, and shell-ratio figures."""

    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm
    except ImportError as exc:  # pragma: no cover - optional plotting dependency
        raise ImportError("resolution comparison plotting requires geppetto[plot]") from exc
    matches = np.flatnonzero(comparison.segment_index == representative_segment)
    if matches.size != 1:
        raise ValueError(f"representative segment {representative_segment} is unavailable")
    shell_row = int(matches[0])
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(7.0, 4.8),
        sharex="col",
        gridspec_kw={"height_ratios": (3.0, 1.0), "hspace": 0.05, "wspace": 0.27},
    )
    ell = comparison.ell_effective
    spectra = (
        (comparison.low_sum, comparison.high_sum, comparison.summed_ratio, "Summed map"),
        (
            comparison.low_shell[shell_row],
            comparison.high_shell[shell_row],
            comparison.shell_ratio[shell_row],
            (
                f"Segment {representative_segment}: "
                f"{comparison.z_lo[shell_row]:.3f} < z < {comparison.z_hi[shell_row]:.3f}"
            ),
        ),
    )
    for column, (low_values, high_values, ratio, title) in enumerate(spectra):
        factor = ell * (ell + 1.0) / (2.0 * np.pi)
        axes[0, column].loglog(ell, factor * low_values, color="#0072B2", label="2160³")
        axes[0, column].loglog(ell, factor * high_values, color="#D55E00", label="4096³")
        axes[0, column].set_title(title)
        axes[0, column].legend(frameon=False)
        axes[1, column].semilogx(ell, ratio, color="#6A3D9A", marker=".", markersize=2)
        axes[1, column].axhline(1.0, color="0.25", linewidth=0.8)
        axes[1, column].set_xlabel(r"Multipole $\ell$")
        finite = ratio[np.isfinite(ratio)]
        extent = max(0.05, float(np.nanpercentile(np.abs(finite - 1.0), 98)))
        axes[1, column].set_ylim(1.0 - 1.1 * extent, 1.0 + 1.1 * extent)
    axes[0, 0].set_ylabel(r"$\ell(\ell+1)C_\ell/(2\pi)$")
    axes[1, 0].set_ylabel("4096³ / 2160³")
    outputs = _save_figure(
        figure, output_dir / "angular_power_resolution_comparison", png_dpi
    )
    plt.close(figure)

    order = np.argsort(comparison.z_lo)
    shell_edges = np.concatenate(
        [comparison.z_lo[order], np.asarray([comparison.z_hi[order][-1]])]
    )
    ell_edges = np.concatenate(
        [comparison.ell_min, np.asarray([comparison.ell_max[-1] + 1])]
    )
    figure, axis = plt.subplots(figsize=(7.0, 3.7))
    image = axis.pcolormesh(
        ell_edges,
        shell_edges,
        comparison.shell_ratio[order],
        shading="auto",
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=0.8, vcenter=1.0, vmax=1.2),
    )
    axis.set_xscale("log")
    axis.set_xlabel(r"Multipole $\ell$")
    axis.set_ylabel("Redshift")
    colorbar = figure.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label("4096³ / 2160³")
    outputs.extend(
        _save_figure(figure, output_dir / "angular_power_resolution_shells", png_dpi)
    )
    plt.close(figure)
    return outputs


def main() -> None:
    args = parse_args()
    comparison = compare_resolutions(
        load_resolution_spectra(args.low_dir),
        load_resolution_spectra(args.high_dir),
        ell_min=args.ell_min,
        bin_width=args.bin_width,
    )
    csv_path = write_comparison_csv(
        comparison, args.output_dir / "angular_power_resolution_comparison.csv"
    )
    figures = render_comparison_figures(
        comparison,
        args.output_dir,
        representative_segment=args.representative_segment,
        png_dpi=args.png_dpi,
    )
    print(f"Wrote {csv_path}")
    for path in figures:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
