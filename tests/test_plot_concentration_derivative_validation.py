from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest

EXAMPLE_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "plot_concentration_derivative_validation.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "plot_concentration_derivative_validation",
        EXAMPLE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_tables(path: Path) -> None:
    path.mkdir()
    parameters = (
        "concentration_amplitude",
        "concentration_mass_slope",
        "concentration_redshift_slope",
    )
    with (path / "painted_nfw_derivative_validation.csv").open(
        "w",
        newline="",
    ) as handle:
        columns = (
            "segment_index",
            "z_lo",
            "z_hi",
            "parameter",
            "step_label",
            "relative_l2_error",
            "cosine_similarity",
        )
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for segment_index in range(2):
            for parameter_index, parameter in enumerate(parameters):
                for step_label, scale in (("h", 1.0), ("h_over_2", 0.25)):
                    writer.writerow(
                        {
                            "segment_index": segment_index,
                            "z_lo": 0.1 + 0.1 * segment_index,
                            "z_hi": 0.2 + 0.1 * segment_index,
                            "parameter": parameter,
                            "step_label": step_label,
                            "relative_l2_error": scale * (parameter_index + 1) * 1.0e-6,
                            "cosine_similarity": 1.0 - scale * 1.0e-10,
                        }
                    )
    with (path / "painted_nfw_derivative_validation_summary.csv").open(
        "w",
        newline="",
    ) as handle:
        columns = (
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
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for parameter_index, parameter in enumerate(parameters):
            writer.writerow(
                {
                    "parameter": parameter,
                    "coarse_relative_l2_error": (parameter_index + 1) * 1.0e-6,
                    "fine_relative_l2_error": (parameter_index + 1) * 2.5e-7,
                    "maximum_shell_fine_relative_l2_error": (
                        parameter_index + 1
                    )
                    * 3.0e-7,
                    "fine_cosine_similarity": 1.0 - 1.0e-11,
                    "fine_best_fit_slope": 1.0 + 1.0e-8,
                    "global_rtol": 1.0e-4,
                    "shell_rtol": 1.0e-3,
                    "passed": True,
                }
            )


def test_derivative_validation_plotter_loads_and_renders(tmp_path):
    pytest.importorskip("matplotlib")
    module = _load_module()
    input_dir = tmp_path / "validation"
    _write_tables(input_dir)

    details, summaries = module.load_validation_tables(input_dir)
    outputs = module.render_validation_figures(
        details,
        summaries,
        tmp_path / "figures",
        png_dpi=72,
    )

    assert len(details) == 12
    assert set(summaries) == set(module.PARAMETERS)
    assert len(outputs) == 4
    assert all(output.exists() and output.stat().st_size > 0 for output in outputs)


def test_derivative_validation_plotter_rejects_missing_columns(tmp_path):
    module = _load_module()
    input_dir = tmp_path / "validation"
    input_dir.mkdir()
    (input_dir / "painted_nfw_derivative_validation.csv").write_text(
        "segment_index\n0\n",
        encoding="utf-8",
    )
    (input_dir / "painted_nfw_derivative_validation_summary.csv").write_text(
        "parameter\nconcentration_amplitude\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing columns"):
        module.load_validation_tables(input_dir)
