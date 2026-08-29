#!/usr/bin/env python3
"""Plot autodiff-versus-finite-difference concentration derivative checks."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

PARAMETERS = (
    "concentration_amplitude",
    "concentration_mass_slope",
    "concentration_redshift_slope",
)
PARAMETER_LABELS = {
    "concentration_amplitude": r"Amplitude $A$",
    "concentration_mass_slope": r"Mass slope $B$",
    "concentration_redshift_slope": r"Redshift slope $C$",
}
DETAIL_COLUMNS = (
    "segment_index",
    "z_lo",
    "z_hi",
    "parameter",
    "step_label",
    "relative_l2_error",
    "cosine_similarity",
)
SUMMARY_COLUMNS = (
    "parameter",
    "coarse_relative_l2_error",
    "fine_relative_l2_error",
    "maximum_shell_fine_relative_l2_error",
    "fine_cosine_similarity",
    "fine_best_fit_slope",
    "global_rtol",
    "shell_rtol",
    "passed",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).with_name("concentration_derivative_validation"),
        help="directory containing the derivative-validation CSV files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="figure directory; defaults to --input-dir",
    )
    parser.add_argument("--png-dpi", type=int, default=300)
    return parser.parse_args()


def _read_csv(path: Path, required: tuple[str, ...]) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = set(required) - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"{path} is missing columns: {sorted(missing)}")
            rows = list(reader)
    except OSError as exc:
        raise ValueError(f"cannot read derivative-validation table: {path}") from exc
    if not rows:
        raise ValueError(f"derivative-validation table is empty: {path}")
    return rows


def _finite(row: dict[str, str], key: str, path: Path) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{path} contains a non-numeric {key!r} value") from exc
    if not np.isfinite(value):
        raise ValueError(f"{path} contains a non-finite {key!r} value")
    return value


def _boolean(row: dict[str, str], key: str, path: Path) -> bool:
    value = row.get(key, "").strip().lower()
    if value in {"true", "1"}:
        return True
    if value in {"false", "0"}:
        return False
    raise ValueError(f"{path} contains an invalid Boolean {key!r} value")


def load_validation_tables(
    input_dir: Path,
) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    """Load and validate per-shell and aggregate derivative diagnostics."""

    detail_path = input_dir / "painted_nfw_derivative_validation.csv"
    summary_path = input_dir / "painted_nfw_derivative_validation_summary.csv"
    detail_source = _read_csv(detail_path, DETAIL_COLUMNS)
    summary_source = _read_csv(summary_path, SUMMARY_COLUMNS)

    details: list[dict[str, object]] = []
    seen: set[tuple[int, str, str]] = set()
    for source in detail_source:
        parameter = source["parameter"]
        step_label = source["step_label"]
        if parameter not in PARAMETERS:
            raise ValueError(f"{detail_path} contains an unknown parameter {parameter!r}")
        if step_label not in {"h", "h_over_2"}:
            raise ValueError(f"{detail_path} contains an unknown step label {step_label!r}")
        segment_index = int(_finite(source, "segment_index", detail_path))
        key = (segment_index, parameter, step_label)
        if key in seen:
            raise ValueError(f"{detail_path} contains duplicate row {key}")
        seen.add(key)
        z_lo = _finite(source, "z_lo", detail_path)
        z_hi = _finite(source, "z_hi", detail_path)
        if z_hi <= z_lo:
            raise ValueError(f"{detail_path} contains non-increasing redshift bounds")
        details.append(
            {
                "segment_index": segment_index,
                "z_mid": 0.5 * (z_lo + z_hi),
                "parameter": parameter,
                "step_label": step_label,
                "relative_l2_error": _finite(source, "relative_l2_error", detail_path),
                "cosine_similarity": _finite(source, "cosine_similarity", detail_path),
            }
        )

    summaries: dict[str, dict[str, object]] = {}
    for source in summary_source:
        parameter = source["parameter"]
        if parameter not in PARAMETERS or parameter in summaries:
            raise ValueError(f"{summary_path} has an invalid parameter row {parameter!r}")
        summaries[parameter] = {
            key: _finite(source, key, summary_path)
            for key in SUMMARY_COLUMNS
            if key not in {"parameter", "passed"}
        }
        summaries[parameter]["passed"] = _boolean(source, "passed", summary_path)
    if set(summaries) != set(PARAMETERS):
        raise ValueError(f"{summary_path} must contain all three concentration parameters")
    return details, summaries


def _save(figure, base: Path, png_dpi: int) -> tuple[Path, Path]:
    pdf = base.with_suffix(".pdf")
    png = base.with_suffix(".png")
    metadata = {"Creator": "GEPPETTO concentration derivative validation"}
    figure.savefig(pdf, bbox_inches="tight", metadata=metadata)
    figure.savefig(png, dpi=png_dpi, bbox_inches="tight", metadata=metadata)
    return pdf, png


def render_validation_figures(
    details: list[dict[str, object]],
    summaries: dict[str, dict[str, object]],
    output_dir: Path,
    *,
    png_dpi: int = 300,
) -> tuple[Path, ...]:
    """Render shell-level convergence and aggregate closure figures."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ValueError("matplotlib is required to render derivative figures") from exc

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "savefig.facecolor": "white",
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    figure, axes = plt.subplots(1, 3, figsize=(7.2, 2.55), sharey=True)
    for axis, parameter, panel in zip(axes, PARAMETERS, "abc", strict=True):
        for step_label, label, color, marker in (
            ("h", r"$h$", "#D95F02", "s"),
            ("h_over_2", r"$h/2$", "#0072B2", "o"),
        ):
            selected = sorted(
                (
                    row
                    for row in details
                    if row["parameter"] == parameter and row["step_label"] == step_label
                ),
                key=lambda row: int(row["segment_index"]),
            )
            axis.plot(
                [float(row["z_mid"]) for row in selected],
                [max(float(row["relative_l2_error"]), 1.0e-16) for row in selected],
                color=color,
                marker=marker,
                markersize=2.7,
                linewidth=0.8,
                label=label,
            )
        shell_rtol = float(summaries[parameter]["shell_rtol"])
        axis.axhline(shell_rtol, color="0.35", linestyle=":", linewidth=0.8)
        axis.set_yscale("log")
        axis.set_xlabel("Shell redshift")
        axis.set_title(PARAMETER_LABELS[parameter])
        axis.text(0.04, 0.94, f"({panel})", transform=axis.transAxes, va="top", weight="bold")
        axis.grid(axis="y", color="0.9", linewidth=0.5)
    axes[0].set_ylabel(r"$\Vert D_{\rm FD}-D_{\rm AD}\Vert_2/\Vert D_{\rm AD}\Vert_2$")
    axes[0].legend(frameon=False, loc="best")
    figure.tight_layout()
    outputs.extend(
        _save(
            figure,
            output_dir / "concentration_derivative_validation_by_shell",
            png_dpi,
        )
    )
    plt.close(figure)

    x = np.arange(len(PARAMETERS), dtype=np.float64)
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.75))
    coarse = np.asarray(
        [float(summaries[parameter]["coarse_relative_l2_error"]) for parameter in PARAMETERS]
    )
    fine = np.asarray(
        [float(summaries[parameter]["fine_relative_l2_error"]) for parameter in PARAMETERS]
    )
    width = 0.34
    axes[0].bar(x - width / 2, np.maximum(coarse, 1.0e-16), width, color="#D95F02", label=r"$h$")
    axes[0].bar(x + width / 2, np.maximum(fine, 1.0e-16), width, color="#0072B2", label=r"$h/2$")
    axes[0].axhline(
        float(summaries[PARAMETERS[0]]["global_rtol"]),
        color="black",
        linestyle=":",
        linewidth=0.8,
        label="Acceptance threshold",
    )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Aggregate relative L2 error")
    axes[0].legend(frameon=False)
    axes[0].text(0.03, 0.95, "(a)", transform=axes[0].transAxes, va="top", weight="bold")

    slopes = np.asarray(
        [float(summaries[parameter]["fine_best_fit_slope"]) for parameter in PARAMETERS]
    )
    slope_residual = slopes - 1.0
    cosine_deficit = np.asarray(
        [
            max(1.0 - float(summaries[parameter]["fine_cosine_similarity"]), 0.0)
            for parameter in PARAMETERS
        ]
    )
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].plot(
        x,
        slope_residual,
        color="#009E73",
        marker="o",
        label="Best-fit slope minus one",
    )
    cosine_axis = axes[1].twinx()
    cosine_axis.plot(
        x,
        np.maximum(cosine_deficit, 1.0e-16),
        color="#7A3E9D",
        marker="s",
        linestyle="--",
        label=r"$1-\cos(D_{\rm AD},D_{\rm FD})$",
    )
    cosine_axis.set_yscale("log")
    axes[1].set_ylabel("Fine-step best-fit slope minus one")
    cosine_axis.set_ylabel("Fine-step cosine deficit")
    handles, labels = axes[1].get_legend_handles_labels()
    second_handles, second_labels = cosine_axis.get_legend_handles_labels()
    axes[1].legend(handles + second_handles, labels + second_labels, frameon=False)
    axes[1].text(0.03, 0.95, "(b)", transform=axes[1].transAxes, va="top", weight="bold")

    tick_labels = ["Amplitude", "Mass slope", "Redshift slope"]
    for axis in axes:
        axis.set_xticks(x, tick_labels)
        axis.tick_params(axis="x", labelrotation=18)
        axis.grid(axis="y", color="0.9", linewidth=0.5)
    figure.tight_layout()
    outputs.extend(
        _save(
            figure,
            output_dir / "concentration_derivative_validation_summary",
            png_dpi,
        )
    )
    plt.close(figure)
    return tuple(outputs)


def main() -> None:
    args = parse_args()
    output_dir = args.input_dir if args.output_dir is None else args.output_dir
    try:
        details, summaries = load_validation_tables(args.input_dir)
        outputs = render_validation_figures(
            details,
            summaries,
            output_dir,
            png_dpi=args.png_dpi,
        )
    except (ImportError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc
    for output in outputs:
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
