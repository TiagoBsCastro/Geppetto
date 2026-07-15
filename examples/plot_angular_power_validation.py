#!/usr/bin/env python3
"""Create paper-ready figures from angular-power validation products."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

NPZ_KEYS = (
    "validation_schema_version",
    "observed_shell",
    "observed_sum",
    "ell",
    "shell_linear_pseudo_over_fsky",
    "shell_one_halo_pseudo_over_fsky",
    "shell_particle_shot_noise_pseudo_over_fsky",
    "summed_linear_pseudo_over_fsky",
    "summed_one_halo_pseudo_over_fsky",
    "summed_particle_shot_noise_pseudo_over_fsky",
    "shell_weights",
    "reference_sigma8",
    "reconstructed_sigma8",
    "sigma8_relative_error",
    "ell_limber_start",
)

BINNED_COLUMNS = (
    "map",
    "ell_min",
    "ell_max",
    "ell_effective",
    "measured",
    "linear",
    "one_halo",
    "particle_shot_noise",
    "clustering",
    "total",
    "measured_over_total",
    "f_sky",
    "shell_weight",
    "theory_convention",
)

DIAGNOSTIC_COLUMNS = (
    "segment_index",
    "z_lo",
    "z_hi",
    "f_sky",
    "nside",
    "reference_sigma8",
    "reconstructed_sigma8",
    "sigma8_relative_error",
    "ell_limber_start",
    "theory_convention",
)


@dataclass(frozen=True)
class AngularPowerValidationData:
    """Validated arrays needed by the angular-power figures."""

    ell: np.ndarray
    observed_sum: np.ndarray
    summed_linear: np.ndarray
    summed_one_halo: np.ndarray
    summed_particle_shot_noise: np.ndarray
    summed_total: np.ndarray
    bin_ell_min: np.ndarray
    bin_ell_max: np.ndarray
    bin_ell_effective: np.ndarray
    binned_observed_sum: np.ndarray
    binned_ratio_sum: np.ndarray
    shell_segment_index: np.ndarray
    shell_z_lo: np.ndarray
    shell_z_hi: np.ndarray
    shell_z_edges: np.ndarray
    observed_shell: np.ndarray
    shell_linear: np.ndarray
    shell_one_halo: np.ndarray
    shell_particle_shot_noise: np.ndarray
    shell_total: np.ndarray
    binned_observed_shell: np.ndarray
    shell_ratios: np.ndarray
    f_sky: float
    nside: int
    reference_sigma8: float
    reconstructed_sigma8: float
    sigma8_relative_error: float
    ell_limber_start: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).with_name("angular_power_validation"),
        help="directory containing angular_power_theory.npz and validation CSV files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parents[1] / "manuscript" / "figures",
        help="directory for PDF and PNG figures",
    )
    parser.add_argument(
        "--ratio-half-range",
        type=float,
        default=0.25,
        help="symmetric shell-ratio color range around unity",
    )
    parser.add_argument(
        "--shell-redshifts",
        type=float,
        nargs=4,
        default=(0.2, 0.8, 1.5, 1.9),
        metavar=("Z1", "Z2", "Z3", "Z4"),
        help="target redshifts for the four representative-shell panels",
    )
    parser.add_argument("--png-dpi", type=int, default=300)
    return parser.parse_args()


def _read_csv(path: Path, required_columns: tuple[str, ...]) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            columns = set(reader.fieldnames or ())
            missing = set(required_columns) - columns
            if missing:
                raise ValueError(f"{path} is missing columns: {sorted(missing)}")
            rows = list(reader)
    except OSError as exc:
        raise ValueError(f"cannot read validation table: {path}") from exc
    if not rows:
        raise ValueError(f"validation table is empty: {path}")
    return rows


def _finite_float(row: dict[str, str], key: str, path: Path) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{path} contains a non-numeric {key!r} value") from exc
    if not np.isfinite(value):
        raise ValueError(f"{path} contains a non-finite {key!r} value")
    return value


def _integer(row: dict[str, str], key: str, path: Path) -> int:
    value = _finite_float(row, key, path)
    integer = int(value)
    if value != integer:
        raise ValueError(f"{path} contains a non-integer {key!r} value")
    return integer


def _require_vector(name: str, values: np.ndarray, size: int | None = None) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 1 or (size is not None and result.size != size):
        expected = "one-dimensional" if size is None else f"shape ({size},)"
        raise ValueError(f"NPZ array {name!r} must have {expected}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"NPZ array {name!r} contains non-finite values")
    return result


def load_validation_data(input_dir: Path) -> AngularPowerValidationData:
    """Load and cross-check the three angular-power validation products."""

    theory_path = input_dir / "angular_power_theory.npz"
    binned_path = input_dir / "angular_power_binned.csv"
    diagnostics_path = input_dir / "angular_power_diagnostics.csv"
    try:
        with np.load(theory_path, allow_pickle=False) as source:
            if "validation_schema_version" not in source.files:
                raise ValueError(
                    f"{theory_path} is a legacy validation archive; rerun angular validation"
                )
            schema = np.asarray(source["validation_schema_version"])
            if schema.shape != () or int(schema) != 2:
                raise ValueError(
                    f"{theory_path} uses an unsupported validation schema; "
                    "rerun angular validation"
                )
            missing = set(NPZ_KEYS) - set(source.files)
            if missing:
                raise ValueError(f"{theory_path} is missing arrays: {sorted(missing)}")
            arrays = {key: np.array(source[key], copy=True) for key in NPZ_KEYS}
    except OSError as exc:
        raise ValueError(f"cannot read validation archive: {theory_path}") from exc

    ell = _require_vector("ell", arrays["ell"])
    if (
        ell.size == 0
        or np.any(ell < 0)
        or np.any(ell != np.rint(ell))
        or np.any(np.diff(ell) <= 0)
    ):
        raise ValueError("NPZ multipoles must be non-negative increasing integers")
    n_ell = ell.size
    summed = {
        "observed_sum": _require_vector("observed_sum", arrays["observed_sum"], n_ell),
        "summed_linear": _require_vector(
            "summed_linear_pseudo_over_fsky",
            arrays["summed_linear_pseudo_over_fsky"],
            n_ell,
        ),
        "summed_one_halo": _require_vector(
            "summed_one_halo_pseudo_over_fsky",
            arrays["summed_one_halo_pseudo_over_fsky"],
            n_ell,
        ),
        "summed_particle_shot_noise": _require_vector(
            "summed_particle_shot_noise_pseudo_over_fsky",
            arrays["summed_particle_shot_noise_pseudo_over_fsky"],
            n_ell,
        ),
    }
    summed["summed_total"] = (
        summed["summed_linear"]
        + summed["summed_one_halo"]
        + summed["summed_particle_shot_noise"]
    )
    shell_weights = _require_vector("shell_weights", arrays["shell_weights"])
    expected_shell_shape = (shell_weights.size, n_ell)
    for key in (
        "observed_shell",
        "shell_linear_pseudo_over_fsky",
        "shell_one_halo_pseudo_over_fsky",
        "shell_particle_shot_noise_pseudo_over_fsky",
    ):
        values = np.asarray(arrays[key])
        if values.shape != expected_shell_shape or not np.all(np.isfinite(values)):
            raise ValueError(
                f"NPZ array {key!r} must be finite with shape {expected_shell_shape}"
            )

    shell_linear = arrays["shell_linear_pseudo_over_fsky"]
    shell_one_halo = arrays["shell_one_halo_pseudo_over_fsky"]
    shell_shot = arrays["shell_particle_shot_noise_pseudo_over_fsky"]
    shell_total = shell_linear + shell_one_halo + shell_shot

    binned_rows = _read_csv(binned_path, BINNED_COLUMNS)
    convention = "constant_deprojected_pseudo_cl_over_f_sky"
    if any(row["theory_convention"] != convention for row in binned_rows):
        raise ValueError(f"{binned_path} contains an unsupported theory convention")
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in binned_rows:
        grouped.setdefault(row["map"], []).append(row)
    if "summed" not in grouped:
        raise ValueError(f"{binned_path} has no 'summed' rows")

    summed_rows = grouped.pop("summed")
    bin_ell_min = np.asarray(
        [_integer(row, "ell_min", binned_path) for row in summed_rows], dtype=np.int64
    )
    bin_ell_max = np.asarray(
        [_integer(row, "ell_max", binned_path) for row in summed_rows], dtype=np.int64
    )
    bin_ell_effective = np.asarray(
        [_finite_float(row, "ell_effective", binned_path) for row in summed_rows]
    )
    if np.any(bin_ell_max < bin_ell_min) or np.any(np.diff(bin_ell_effective) <= 0.0):
        raise ValueError("summed multipole bins must be ordered and non-empty")
    if np.any(bin_ell_effective < bin_ell_min) or np.any(
        bin_ell_effective >= bin_ell_max + 1
    ):
        raise ValueError("effective multipoles must lie inside their bins")
    f_sky_values = np.asarray(
        [_finite_float(row, "f_sky", binned_path) for row in summed_rows]
    )
    if not np.allclose(f_sky_values, f_sky_values[0], rtol=0.0, atol=1.0e-12):
        raise ValueError("binned rows do not share one f_sky value")
    f_sky = float(f_sky_values[0])
    if not 0.0 < f_sky <= 1.0:
        raise ValueError("f_sky must lie in (0, 1]")

    scalar_diagnostics: dict[str, float] = {}
    for key in ("reference_sigma8", "reconstructed_sigma8", "sigma8_relative_error"):
        value = np.asarray(arrays[key])
        if value.shape != () or not np.isfinite(float(value)):
            raise ValueError(f"NPZ scalar {key!r} must be finite")
        scalar_diagnostics[key] = float(value)
    limber_start_value = np.asarray(arrays["ell_limber_start"])
    if limber_start_value.shape != () or int(limber_start_value) < 0:
        raise ValueError("NPZ ell_limber_start must be a non-negative scalar")
    ell_limber_start = int(limber_start_value)

    diagnostics = _read_csv(diagnostics_path, DIAGNOSTIC_COLUMNS)
    shell_rows: list[tuple[float, float, int, int]] = []
    nside_values: list[int] = []
    for archive_index, row in enumerate(diagnostics):
        segment_index = _integer(row, "segment_index", diagnostics_path)
        z_lo = _finite_float(row, "z_lo", diagnostics_path)
        z_hi = _finite_float(row, "z_hi", diagnostics_path)
        if z_lo < 0.0 or z_hi <= z_lo:
            raise ValueError("diagnostic shell bounds must satisfy 0 <= z_lo < z_hi")
        if not np.isclose(
            _finite_float(row, "f_sky", diagnostics_path), f_sky, rtol=0.0, atol=1.0e-12
        ):
            raise ValueError("diagnostic and binned f_sky values disagree")
        if row["theory_convention"] != convention:
            raise ValueError("diagnostic table uses an unsupported theory convention")
        for key in ("reference_sigma8", "reconstructed_sigma8", "sigma8_relative_error"):
            if not np.isclose(
                _finite_float(row, key, diagnostics_path),
                scalar_diagnostics[key],
                rtol=1.0e-12,
                atol=0.0,
            ):
                raise ValueError(f"diagnostic and NPZ {key} values disagree")
        if _integer(row, "ell_limber_start", diagnostics_path) != ell_limber_start:
            raise ValueError("diagnostic and NPZ Limber transition values disagree")
        shell_rows.append((z_lo, z_hi, segment_index, archive_index))
        nside_values.append(_integer(row, "nside", diagnostics_path))
    if len(set(nside_values)) != 1 or nside_values[0] <= 0:
        raise ValueError("diagnostic rows must share one positive NSIDE")
    if len(shell_rows) != shell_weights.size:
        raise ValueError("diagnostic and NPZ shell counts disagree")

    shell_rows.sort()
    if any(
        not np.isclose(previous[1], current[0], rtol=0.0, atol=1.0e-8)
        for previous, current in zip(shell_rows[:-1], shell_rows[1:], strict=True)
    ):
        raise ValueError("diagnostic redshift shells must be contiguous")
    shell_z_lo = np.asarray([row[0] for row in shell_rows])
    shell_z_hi = np.asarray([row[1] for row in shell_rows])
    shell_segment_index = np.asarray([row[2] for row in shell_rows], dtype=np.int64)
    shell_archive_order = np.asarray([row[3] for row in shell_rows], dtype=np.int64)
    shell_z_edges = np.concatenate((shell_z_lo[:1], shell_z_hi))

    shell_ratios: list[np.ndarray] = []
    binned_observed_shell: list[np.ndarray] = []
    expected_bins = np.column_stack((bin_ell_min, bin_ell_max, bin_ell_effective))
    for _, _, segment_index, _ in shell_rows:
        label = f"segment_{segment_index}"
        try:
            rows = grouped.pop(label)
        except KeyError as exc:
            raise ValueError(f"{binned_path} has no rows for {label}") from exc
        bins = np.asarray(
            [
                (
                    _integer(row, "ell_min", binned_path),
                    _integer(row, "ell_max", binned_path),
                    _finite_float(row, "ell_effective", binned_path),
                )
                for row in rows
            ]
        )
        if bins.shape != expected_bins.shape or not np.allclose(
            bins, expected_bins, rtol=0.0, atol=1.0e-10
        ):
            raise ValueError(f"{label} does not use the summed-map multipole bins")
        if any(
            not np.isclose(
                _finite_float(row, "f_sky", binned_path),
                f_sky,
                rtol=0.0,
                atol=1.0e-12,
            )
            for row in rows
        ):
            raise ValueError(f"{label} does not use the common f_sky value")
        shell_ratios.append(
            np.asarray([_finite_float(row, "measured_over_total", binned_path) for row in rows])
        )
        binned_observed_shell.append(
            np.asarray([_finite_float(row, "measured", binned_path) for row in rows])
        )
    if grouped:
        raise ValueError(f"binned table contains unknown map labels: {sorted(grouped)}")

    return AngularPowerValidationData(
        ell=ell,
        observed_sum=summed["observed_sum"],
        summed_linear=summed["summed_linear"],
        summed_one_halo=summed["summed_one_halo"],
        summed_particle_shot_noise=summed["summed_particle_shot_noise"],
        summed_total=summed["summed_total"],
        bin_ell_min=bin_ell_min,
        bin_ell_max=bin_ell_max,
        bin_ell_effective=bin_ell_effective,
        binned_observed_sum=np.asarray(
            [_finite_float(row, "measured", binned_path) for row in summed_rows]
        ),
        binned_ratio_sum=np.asarray(
            [_finite_float(row, "measured_over_total", binned_path) for row in summed_rows]
        ),
        shell_segment_index=shell_segment_index,
        shell_z_lo=shell_z_lo,
        shell_z_hi=shell_z_hi,
        shell_z_edges=shell_z_edges,
        observed_shell=arrays["observed_shell"][shell_archive_order],
        shell_linear=shell_linear[shell_archive_order],
        shell_one_halo=shell_one_halo[shell_archive_order],
        shell_particle_shot_noise=shell_shot[shell_archive_order],
        shell_total=shell_total[shell_archive_order],
        binned_observed_shell=np.stack(binned_observed_shell),
        shell_ratios=np.stack(shell_ratios),
        f_sky=f_sky,
        nside=nside_values[0],
        reference_sigma8=scalar_diagnostics["reference_sigma8"],
        reconstructed_sigma8=scalar_diagnostics["reconstructed_sigma8"],
        sigma8_relative_error=scalar_diagnostics["sigma8_relative_error"],
        ell_limber_start=ell_limber_start,
    )


def gaussian_mode_counting_fraction(
    ell_min: np.ndarray,
    ell_max: np.ndarray,
    f_sky: float,
) -> np.ndarray:
    """Return the Gaussian ``sigma(C_b) / C_b`` mode-counting guide."""

    lower = np.asarray(ell_min, dtype=np.float64)
    upper = np.asarray(ell_max, dtype=np.float64)
    if lower.shape != upper.shape or np.any(upper < lower):
        raise ValueError("ell_min and ell_max must have matching valid bins")
    if not 0.0 < f_sky <= 1.0:
        raise ValueError("f_sky must lie in (0, 1]")
    mode_count = f_sky * ((upper + 1.0) ** 2 - lower**2)
    return np.sqrt(2.0 / mode_count)


def representative_shell_indices(
    data: AngularPowerValidationData,
    target_redshifts: tuple[float, float, float, float] | list[float],
) -> np.ndarray:
    """Return four unique shell rows nearest the requested redshifts."""

    targets = np.asarray(target_redshifts, dtype=np.float64)
    if targets.shape != (4,) or not np.all(np.isfinite(targets)):
        raise ValueError("representative shell redshifts must contain four finite values")
    midpoint = 0.5 * (data.shell_z_lo + data.shell_z_hi)
    selected = np.asarray(
        [int(np.argmin(np.abs(midpoint - target))) for target in targets],
        dtype=np.int64,
    )
    if np.unique(selected).size != 4:
        raise ValueError("representative redshifts must select four distinct shells")
    return selected


def _load_plotting() -> tuple[Any, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm
    except ImportError as exc:  # pragma: no cover - optional install
        raise ImportError(
            "figure generation requires matplotlib; install geppetto[plot]"
        ) from exc
    return plt, TwoSlopeNorm


def _paper_style() -> dict[str, object]:
    return {
        "font.family": "serif",
        "font.size": 9,
        "mathtext.fontset": "stix",
        "axes.labelsize": 10,
        "axes.linewidth": 0.7,
        "legend.fontsize": 8,
        "legend.handlelength": 2.4,
        "lines.linewidth": 1.3,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "savefig.facecolor": "white",
    }


def _save_figure(figure: Any, output_stem: Path, png_dpi: int) -> tuple[Path, Path]:
    pdf_path = output_stem.with_suffix(".pdf")
    png_path = output_stem.with_suffix(".png")
    metadata = {"Creator": "GEPPETTO", "Title": output_stem.name.replace("_", " ")}
    figure.savefig(pdf_path, bbox_inches="tight", metadata=metadata)
    figure.savefig(png_path, dpi=png_dpi, bbox_inches="tight", metadata=metadata)
    return pdf_path, png_path


def render_validation_figures(
    data: AngularPowerValidationData,
    output_dir: Path,
    *,
    ratio_half_range: float = 0.25,
    representative_redshifts: tuple[float, float, float, float] | list[float] = (
        0.2,
        0.8,
        1.5,
        1.9,
    ),
    png_dpi: int = 300,
) -> tuple[Path, ...]:
    """Render summed, representative-shell, and shell-residual figures."""

    if not np.isfinite(ratio_half_range) or ratio_half_range <= 0.0:
        raise ValueError("ratio_half_range must be finite and positive")
    if png_dpi < 72:
        raise ValueError("png_dpi must be at least 72")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt, two_slope_norm = _load_plotting()
    outputs: list[Path] = []

    with plt.rc_context(_paper_style()):
        ell = data.ell.astype(np.float64)
        ell_bin = data.bin_ell_effective
        scale = ell * (ell + 1.0) / (2.0 * np.pi)
        bin_scale = ell_bin * (ell_bin + 1.0) / (2.0 * np.pi)
        figure, (spectrum_axis, ratio_axis) = plt.subplots(
            2,
            1,
            figsize=(7.1, 5.0),
            sharex=True,
            gridspec_kw={"height_ratios": (3.0, 1.0), "hspace": 0.05},
        )
        spectrum_axis.loglog(ell, scale * data.summed_total, color="black", label="Total theory")
        spectrum_axis.loglog(
            ell, scale * data.summed_linear, color="#0072B2", linestyle="--", label="Linear"
        )
        spectrum_axis.loglog(
            ell,
            scale * data.summed_one_halo,
            color="#D55E00",
            linestyle="-.",
            label="One halo",
        )
        spectrum_axis.loglog(
            ell,
            scale * data.summed_particle_shot_noise,
            color="#009E73",
            linestyle=":",
            label="Particle shot noise",
        )
        spectrum_axis.scatter(
            ell_bin,
            bin_scale * data.binned_observed_sum,
            s=9,
            facecolor="white",
            edgecolor="black",
            linewidth=0.55,
            zorder=5,
            label="PINOCCHIO + GEPPETTO",
        )
        spectrum_axis.set_ylabel(r"$\ell(\ell+1)C_\ell/(2\pi)$")
        spectrum_axis.legend(
            ncol=3,
            frameon=False,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
        )
        visible = ell >= data.bin_ell_min[0]
        positive_components = np.concatenate(
            (
                scale[visible] * data.summed_linear[visible],
                scale[visible] * data.summed_one_halo[visible],
                scale[visible] * data.summed_particle_shot_noise[visible],
            )
        )
        spectrum_axis.set_ylim(
            0.7 * float(np.min(positive_components[positive_components > 0.0])),
            1.4 * float(np.max(bin_scale * data.binned_observed_sum)),
        )
        spectrum_axis.text(
            0.015, 0.96, "(a)", transform=spectrum_axis.transAxes, va="top", weight="bold"
        )

        gaussian_fraction = gaussian_mode_counting_fraction(
            data.bin_ell_min, data.bin_ell_max, data.f_sky
        )
        ratio_axis.fill_between(
            ell_bin,
            1.0 - gaussian_fraction,
            1.0 + gaussian_fraction,
            color="0.8",
            label="Gaussian mode-counting guide",
        )
        ratio_axis.axhline(1.0, color="black", linewidth=0.8)
        ratio_axis.plot(
            ell_bin,
            data.binned_ratio_sum,
            color="#7A3E9D",
            marker="o",
            markersize=2.0,
            linewidth=0.8,
        )
        ratio_axis.set_xscale("log")
        ratio_axis.set_xlim(float(data.bin_ell_min[0]), float(data.bin_ell_max[-1] + 1))
        ratio_extent = max(
            0.1,
            float(np.max(np.abs(data.binned_ratio_sum - 1.0))) * 1.15,
            float(np.max(gaussian_fraction)) * 1.15,
        )
        ratio_axis.set_ylim(1.0 - ratio_extent, 1.0 + ratio_extent)
        ratio_axis.set_xlabel(r"Multipole $\ell$")
        ratio_axis.set_ylabel("Measured / theory")
        ratio_axis.legend(frameon=False, loc="lower left")
        ratio_axis.text(
            0.015, 0.91, "(b)", transform=ratio_axis.transAxes, va="top", weight="bold"
        )
        outputs.extend(
            _save_figure(figure, output_dir / "angular_power_summed", png_dpi)
        )
        plt.close(figure)

        selected_shells = representative_shell_indices(data, representative_redshifts)
        representative_ratio_extent = max(
            0.15,
            float(np.max(np.abs(data.shell_ratios[selected_shells] - 1.0))) * 1.08,
            float(np.max(gaussian_fraction)) * 1.15,
        )
        figure = plt.figure(figsize=(7.1, 7.0))
        outer_grid = figure.add_gridspec(
            2,
            2,
            left=0.10,
            right=0.98,
            bottom=0.08,
            top=0.88,
            hspace=0.28,
            wspace=0.24,
        )
        legend_handles: list[Any] = []
        legend_labels: list[str] = []
        panel_letters = ("(a)", "(b)", "(c)", "(d)")
        for panel_index, (shell_index, grid_cell) in enumerate(
            zip(selected_shells, outer_grid, strict=True)
        ):
            panel_grid = grid_cell.subgridspec(
                2, 1, height_ratios=(2.7, 1.0), hspace=0.05
            )
            spectrum_axis = figure.add_subplot(panel_grid[0])
            ratio_axis = figure.add_subplot(panel_grid[1], sharex=spectrum_axis)
            spectrum_axis.loglog(
                ell,
                scale * data.shell_total[shell_index],
                color="black",
                label="Total theory",
            )
            spectrum_axis.loglog(
                ell,
                scale * data.shell_linear[shell_index],
                color="#0072B2",
                linestyle="--",
                label="Linear",
            )
            spectrum_axis.loglog(
                ell,
                scale * data.shell_one_halo[shell_index],
                color="#D55E00",
                linestyle="-.",
                label="One halo",
            )
            spectrum_axis.loglog(
                ell,
                scale * data.shell_particle_shot_noise[shell_index],
                color="#009E73",
                linestyle=":",
                label="Particle shot noise",
            )
            spectrum_axis.scatter(
                ell_bin,
                bin_scale * data.binned_observed_shell[shell_index],
                s=7,
                facecolor="white",
                edgecolor="black",
                linewidth=0.5,
                zorder=5,
                label="PINOCCHIO + GEPPETTO",
            )
            panel_components = np.concatenate(
                (
                    scale[visible] * data.shell_linear[shell_index, visible],
                    scale[visible] * data.shell_one_halo[shell_index, visible],
                    scale[visible]
                    * data.shell_particle_shot_noise[shell_index, visible],
                )
            )
            spectrum_axis.set_ylim(
                0.7 * float(np.min(panel_components[panel_components > 0.0])),
                1.4
                * float(
                    np.max(bin_scale * data.binned_observed_shell[shell_index])
                ),
            )
            spectrum_axis.tick_params(labelbottom=False)
            spectrum_axis.set_title(
                rf"${data.shell_z_lo[shell_index]:.3f}<z<"
                rf"{data.shell_z_hi[shell_index]:.3f}$",
                fontsize=9,
                pad=3,
            )
            spectrum_axis.text(
                0.025,
                0.94,
                panel_letters[panel_index],
                transform=spectrum_axis.transAxes,
                va="top",
                weight="bold",
            )
            if panel_index % 2 == 0:
                spectrum_axis.set_ylabel(r"$\ell(\ell+1)C_\ell/(2\pi)$")

            guide = ratio_axis.fill_between(
                ell_bin,
                1.0 - gaussian_fraction,
                1.0 + gaussian_fraction,
                color="0.8",
                label="Gaussian mode-counting guide",
            )
            ratio_axis.axhline(1.0, color="black", linewidth=0.8)
            ratio_axis.plot(
                ell_bin,
                data.shell_ratios[shell_index],
                color="#7A3E9D",
                marker="o",
                markersize=1.7,
                linewidth=0.7,
            )
            ratio_axis.set_xscale("log")
            ratio_axis.set_xlim(
                float(data.bin_ell_min[0]), float(data.bin_ell_max[-1] + 1)
            )
            ratio_axis.set_ylim(
                1.0 - representative_ratio_extent,
                1.0 + representative_ratio_extent,
            )
            if panel_index % 2 == 0:
                ratio_axis.set_ylabel("Measured / theory", fontsize=8)
            if panel_index == 0:
                legend_handles, legend_labels = spectrum_axis.get_legend_handles_labels()
                legend_handles.append(guide)
                legend_labels.append("Gaussian mode-counting guide")

        figure.legend(
            legend_handles,
            legend_labels,
            ncol=3,
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.54, 0.995),
        )
        figure.supxlabel(r"Multipole $\ell$", y=0.015, fontsize=10)
        outputs.extend(
            _save_figure(
                figure,
                output_dir / "angular_power_representative_shells",
                png_dpi,
            )
        )
        plt.close(figure)

        x_edges = np.concatenate(
            ([float(data.bin_ell_min[0])], data.bin_ell_max.astype(np.float64) + 1.0)
        )
        figure, axis = plt.subplots(figsize=(7.1, 3.5))
        image = axis.pcolormesh(
            x_edges,
            data.shell_z_edges,
            data.shell_ratios,
            shading="flat",
            cmap="RdBu_r",
            norm=two_slope_norm(
                vmin=1.0 - ratio_half_range,
                vcenter=1.0,
                vmax=1.0 + ratio_half_range,
            ),
            rasterized=True,
        )
        axis.set_xscale("log")
        axis.set_xlim(x_edges[0], x_edges[-1])
        axis.set_xlabel(r"Multipole $\ell$")
        axis.set_ylabel("Redshift $z$")
        colorbar = figure.colorbar(image, ax=axis, pad=0.02, extend="both")
        colorbar.set_label("Measured / theory")
        outputs.extend(
            _save_figure(figure, output_dir / "angular_power_shell_residuals", png_dpi)
        )
        plt.close(figure)

    return tuple(outputs)


def main() -> None:
    args = parse_args()
    try:
        data = load_validation_data(args.input_dir)
        outputs = render_validation_figures(
            data,
            args.output_dir,
            ratio_half_range=args.ratio_half_range,
            representative_redshifts=args.shell_redshifts,
            png_dpi=args.png_dpi,
        )
    except (ImportError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc
    for output in outputs:
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
