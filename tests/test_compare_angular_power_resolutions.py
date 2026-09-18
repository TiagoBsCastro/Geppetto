import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_module():
    path = Path(__file__).parents[1] / "examples" / "compare_angular_power_resolutions.py"
    spec = importlib.util.spec_from_file_location("compare_angular_power_resolutions", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_products(path, scale, order=(0, 1)):
    path.mkdir()
    ell = np.arange(2, 42)
    base_shell = np.vstack((1.0 / ell, 2.0 / ell))
    observed_shell = scale * base_shell[np.asarray(order)]
    observed_sum = scale * 1.5 / ell
    np.savez_compressed(
        path / "angular_power_theory.npz",
        validation_schema_version=np.asarray(3),
        ell=ell,
        observed_shell=observed_shell,
        observed_sum=observed_sum,
        mask_pixel_sha256=np.asarray("same-mask"),
    )
    metadata = {
        0: (0.0, 0.5),
        1: (0.5, 1.0),
    }
    rows = [
        {
            "segment_index": segment,
            "z_lo": metadata[segment][0],
            "z_hi": metadata[segment][1],
            "nside": 8,
            "f_sky": 0.5,
        }
        for segment in order
    ]
    with (path / "angular_power_diagnostics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_comparison_aligns_shells_and_bins_ratios(tmp_path):
    module = _load_module()
    low_path = tmp_path / "low"
    high_path = tmp_path / "high"
    _write_products(low_path, 1.0)
    _write_products(high_path, 1.25, order=(1, 0))

    comparison = module.compare_resolutions(
        module.load_resolution_spectra(low_path),
        module.load_resolution_spectra(high_path),
        ell_min=2,
        bin_width=20,
    )

    np.testing.assert_array_equal(comparison.segment_index, [0, 1])
    np.testing.assert_allclose(comparison.shell_ratio, 1.25)
    np.testing.assert_allclose(comparison.summed_ratio, 1.25)


def test_comparison_rejects_different_masks(tmp_path):
    module = _load_module()
    low_path = tmp_path / "low"
    high_path = tmp_path / "high"
    _write_products(low_path, 1.0)
    _write_products(high_path, 1.0)
    archive = dict(np.load(high_path / "angular_power_theory.npz"))
    archive["mask_pixel_sha256"] = np.asarray("different-mask")
    np.savez_compressed(high_path / "angular_power_theory.npz", **archive)

    with pytest.raises(ValueError, match="compact masks differ"):
        module.compare_resolutions(
            module.load_resolution_spectra(low_path),
            module.load_resolution_spectra(high_path),
        )


def test_render_resolution_comparison(tmp_path):
    pytest.importorskip("matplotlib")
    module = _load_module()
    low_path = tmp_path / "low"
    high_path = tmp_path / "high"
    output_path = tmp_path / "output"
    _write_products(low_path, 1.0)
    _write_products(high_path, 1.1)
    comparison = module.compare_resolutions(
        module.load_resolution_spectra(low_path),
        module.load_resolution_spectra(high_path),
        ell_min=2,
        bin_width=20,
    )

    outputs = module.render_comparison_figures(
        comparison, output_path, representative_segment=0, png_dpi=72
    )

    assert {path.name for path in outputs} == {
        "angular_power_resolution_comparison.pdf",
        "angular_power_resolution_comparison.png",
        "angular_power_resolution_shells.pdf",
        "angular_power_resolution_shells.png",
    }
    assert all(path.stat().st_size > 0 for path in outputs)
