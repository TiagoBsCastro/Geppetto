import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_module():
    path = (
        Path(__file__).parents[1]
        / "examples"
        / "compare_observed_resolutions_to_low_theory.py"
    )
    spec = importlib.util.spec_from_file_location(
        "compare_observed_resolutions_to_low_theory", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _products(module):
    ell = np.arange(2, 42)
    theory_shell = np.vstack((1.0 / ell, 2.0 / ell))
    theory_sum = 1.5 / ell
    low = module.LowResolutionReference(
        ell=ell,
        observed_shell=1.1 * theory_shell,
        observed_sum=1.1 * theory_sum,
        theory_shell=theory_shell,
        theory_sum=theory_sum,
        mean_total_counts_per_pixel=np.asarray([1.0, 2.0]),
        segment_index=np.asarray([0, 1]),
        z_lo=np.asarray([0.0, 0.5]),
        z_hi=np.asarray([0.5, 1.0]),
        nside=8,
        f_sky=0.5,
        mask_pixel_sha256="mask",
        mask_sht_iterations=3,
    )
    high = module.ObservedResolution(
        ell=ell,
        observed_shell=1.2 * theory_shell,
        observed_sum=1.2 * theory_sum,
        mean_total_counts_per_pixel=np.asarray([1.1, 2.1]),
        segment_index=low.segment_index,
        z_lo=low.z_lo,
        z_hi=low.z_hi,
        nside=low.nside,
        f_sky=low.f_sky,
        mask_pixel_sha256=low.mask_pixel_sha256,
    )
    return low, high


def test_bins_low_theory_and_both_observed_resolutions():
    module = _load_module()
    low, high = _products(module)

    comparison = module.bin_comparison(low, high, ell_min=2, bin_width=20)

    np.testing.assert_allclose(comparison.low_shell / comparison.theory_shell, 1.1)
    np.testing.assert_allclose(comparison.high_shell / comparison.theory_shell, 1.2)
    np.testing.assert_allclose(comparison.low_sum / comparison.theory_sum, 1.1)
    np.testing.assert_allclose(comparison.high_sum / comparison.theory_sum, 1.2)


def test_observed_cache_round_trip(tmp_path):
    module = _load_module()
    low, high = _products(module)
    path = tmp_path / "observed.npz"

    module.save_observed_cache(high, path)
    restored = module.load_observed_cache(path, low)

    np.testing.assert_array_equal(restored.ell, high.ell)
    np.testing.assert_allclose(restored.observed_shell, high.observed_shell)
    np.testing.assert_allclose(restored.observed_sum, high.observed_sum)
    np.testing.assert_allclose(
        restored.mean_total_counts_per_pixel,
        high.mean_total_counts_per_pixel,
    )


def test_render_low_theory_resolution_comparison(tmp_path):
    pytest.importorskip("matplotlib")
    module = _load_module()
    low, high = _products(module)
    comparison = module.bin_comparison(low, high, ell_min=2, bin_width=20)

    outputs = module.render_comparison(
        comparison,
        tmp_path,
        representative_segment=0,
        png_dpi=72,
    )

    assert {path.name for path in outputs} == {
        "angular_power_low_theory_resolution_check.pdf",
        "angular_power_low_theory_resolution_check.png",
    }
    assert all(path.stat().st_size > 0 for path in outputs)


def test_render_all_shells(tmp_path):
    pytest.importorskip("matplotlib")
    module = _load_module()
    low, high = _products(module)
    comparison = module.bin_comparison(low, high, ell_min=2, bin_width=20)

    outputs = module.render_all_shells(comparison, tmp_path, png_dpi=72)

    assert {path.name for path in outputs} == {
        "angular_power_low_theory_all_shells.pdf",
        "angular_power_low_theory_all_shells.png",
    }
    assert all(path.stat().st_size > 0 for path in outputs)
