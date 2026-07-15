import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_example_module():
    path = (
        Path(__file__).parents[1]
        / "examples"
        / "plot_angular_power_validation.py"
    )
    spec = importlib.util.spec_from_file_location("plot_angular_power_validation", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_validation_products(path):
    ell = np.arange(2, 10, dtype=np.int64)
    shell_linear = np.vstack([(2.0 + index) / ell for index in range(4)])
    shell_one_halo = np.full((4, ell.size), 0.02)
    shell_shot = np.full((4, ell.size), 0.01)
    shell_total = shell_linear + shell_one_halo + shell_shot
    shell_weights = np.array([0.1, 0.2, 0.3, 0.4])
    summed_linear = np.sum(shell_weights[:, None] * shell_linear, axis=0)
    summed_one_halo = np.full(ell.size, 0.02)
    summed_shot = np.full(ell.size, 0.01)
    summed_total = summed_linear + summed_one_halo + summed_shot
    np.savez_compressed(
        path / "angular_power_theory.npz",
        validation_schema_version=np.asarray(2),
        observed_shell=shell_total * np.array([[0.9], [0.95], [1.05], [1.1]]),
        observed_sum=summed_total,
        ell=ell,
        shell_linear_pseudo_over_fsky=shell_linear,
        shell_one_halo_pseudo_over_fsky=shell_one_halo,
        shell_particle_shot_noise_pseudo_over_fsky=shell_shot,
        summed_linear_pseudo_over_fsky=summed_linear,
        summed_one_halo_pseudo_over_fsky=summed_one_halo,
        summed_particle_shot_noise_pseudo_over_fsky=summed_shot,
        shell_weights=shell_weights,
        reference_sigma8=np.asarray(0.81),
        reconstructed_sigma8=np.asarray(0.809),
        sigma8_relative_error=np.asarray(abs(0.809 / 0.81 - 1.0)),
        ell_limber_start=np.asarray(6),
    )

    rows = []
    for label, ratios in (
        ("segment_0", (0.9, 0.95)),
        ("segment_1", (0.95, 0.98)),
        ("segment_2", (1.05, 1.02)),
        ("segment_3", (1.1, 1.05)),
        ("summed", (1.0, 1.0)),
    ):
        for index, (lower, upper, effective) in enumerate(((2, 5, 3.5), (6, 9, 7.5))):
            linear = float(np.mean(summed_linear[lower - 2 : upper - 1]))
            one_halo = 0.02
            shot = 0.01
            total = linear + one_halo + shot
            rows.append(
                {
                    "map": label,
                    "ell_min": lower,
                    "ell_max": upper,
                    "ell_effective": effective,
                    "measured": ratios[index] * total,
                    "linear": linear,
                    "one_halo": one_halo,
                    "particle_shot_noise": shot,
                    "clustering": linear + one_halo,
                    "total": total,
                    "measured_over_total": ratios[index],
                    "f_sky": 0.5,
                    "shell_weight": 1.0 if label == "summed" else 0.5,
                    "theory_convention": "constant_deprojected_pseudo_cl_over_f_sky",
                }
            )
    _write_csv(path / "angular_power_binned.csv", rows)
    # Segment numbering is deliberately reversed relative to redshift order.
    _write_csv(
        path / "angular_power_diagnostics.csv",
        [
            {
                "segment_index": index,
                "z_lo": z_lo,
                "z_hi": z_hi,
                "f_sky": 0.5,
                "nside": 8,
                "reference_sigma8": 0.81,
                "reconstructed_sigma8": 0.809,
                "sigma8_relative_error": abs(0.809 / 0.81 - 1.0),
                "ell_limber_start": 6,
                "theory_convention": "constant_deprojected_pseudo_cl_over_f_sky",
            }
            for index, (z_lo, z_hi) in enumerate(
                ((1.5, 2.0), (1.0, 1.5), (0.5, 1.0), (0.0, 0.5))
            )
        ],
    )


def test_validation_loader_orders_shells_by_redshift(tmp_path):
    module = _load_example_module()
    _write_validation_products(tmp_path)

    data = module.load_validation_data(tmp_path)

    np.testing.assert_array_equal(data.shell_z_edges, [0.0, 0.5, 1.0, 1.5, 2.0])
    np.testing.assert_array_equal(data.shell_segment_index, [3, 2, 1, 0])
    np.testing.assert_allclose(data.shell_ratios[0], [1.1, 1.05])
    np.testing.assert_allclose(data.shell_ratios[-1], [0.9, 0.95])
    assert data.nside == 8
    assert data.f_sky == pytest.approx(0.5)

    np.testing.assert_array_equal(
        module.representative_shell_indices(data, [0.2, 0.8, 1.3, 1.9]),
        [0, 1, 2, 3],
    )


def test_gaussian_mode_counting_fraction():
    module = _load_example_module()
    result = module.gaussian_mode_counting_fraction(
        np.array([2, 6]), np.array([5, 9]), f_sky=0.5
    )
    expected_modes = 0.5 * np.array([6**2 - 2**2, 10**2 - 6**2])
    np.testing.assert_allclose(result, np.sqrt(2.0 / expected_modes))


def test_render_validation_figures(tmp_path):
    pytest.importorskip("matplotlib")
    module = _load_example_module()
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    _write_validation_products(input_dir)

    outputs = module.render_validation_figures(
        module.load_validation_data(input_dir), output_dir, png_dpi=72
    )

    assert {path.name for path in outputs} == {
        "angular_power_summed.pdf",
        "angular_power_summed.png",
        "angular_power_representative_shells.pdf",
        "angular_power_representative_shells.png",
        "angular_power_shell_residuals.pdf",
        "angular_power_shell_residuals.png",
    }
    assert all(path.stat().st_size > 0 for path in outputs)


def test_validation_loader_rejects_inconsistent_shell_bins(tmp_path):
    module = _load_example_module()
    _write_validation_products(tmp_path)
    rows = list(csv.DictReader((tmp_path / "angular_power_binned.csv").open()))
    rows[0]["ell_effective"] = "4.0"
    _write_csv(tmp_path / "angular_power_binned.csv", rows)

    with pytest.raises(ValueError, match="does not use the summed-map multipole bins"):
        module.load_validation_data(tmp_path)


def test_validation_loader_rejects_legacy_archive(tmp_path):
    module = _load_example_module()
    np.savez_compressed(tmp_path / "angular_power_theory.npz", ell=np.arange(3))

    with pytest.raises(ValueError, match="legacy validation archive"):
        module.load_validation_data(tmp_path)
