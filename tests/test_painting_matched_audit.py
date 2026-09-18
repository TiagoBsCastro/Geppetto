import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    return importlib.import_module("plot_painting_matched_audit")


def test_only_distinct_correction_gets_extra_pixel_window(module):
    terms = module.candidate_components(
        np.ones((2, 3)), -np.ones((2, 3)), np.array([2., 4.]),
        np.full((2, 3), 3.), np.full(3, .5),
    )
    np.testing.assert_array_equal(terms["total"], [[5.75]*3, [7.75]*3])
    np.testing.assert_array_equal(terms["halo_self"], 3.)
    np.testing.assert_array_equal(terms["distinct_correction"], -.25)


def test_candidate_rejects_wrong_shapes_and_nonfinite_values(module):
    inputs = [np.ones((2, 3)), np.ones((2, 3)), np.ones(2), np.ones((2, 3)), np.ones(3)]
    with pytest.raises(ValueError, match="dimensions"):
        module.candidate_components(*inputs[:-1], np.ones(4))
    inputs[1][0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        module.candidate_components(*inputs)


def test_rms_metric_uses_measured_over_predicted_not_inverse(module):
    np.testing.assert_allclose(module.fractional_rms(np.array([[2., 4.]]), np.ones((1, 2))), np.sqrt(5))
    with pytest.raises(ValueError, match="positive"):
        module.fractional_rms(np.ones((1, 2)), np.zeros((1, 2)))


def test_portable_covariance_keeps_linear_replacement_separate(module):
    parts = np.array([[[1., 2.], [-.1, .2], [-.5, .4], [2., 3.], [2.4, 5.6]]])
    correction = np.array([[.2, .3]])
    data = dict(stationary_components=parts, linear_projection_correction=correction,
                shell_total=parts[:, -1]+correction)
    terms = module.covariance_candidate_components(data)
    np.testing.assert_array_equal(terms["uncollapsed"], parts[:, 0])
    np.testing.assert_array_equal(terms["linear_projection_correction"], correction)
    np.testing.assert_allclose(sum(value for key, value in terms.items() if key != "total"), terms["total"])
    data["shell_total"] *= 2
    with pytest.raises(ValueError, match="sum"):
        module.covariance_candidate_components(data)
