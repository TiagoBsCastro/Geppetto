import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    return importlib.import_module("prepare_angular_mask_response")


def test_reciprocal_response_matches_direct_binned_operator(module):
    ell = np.arange(2, 80)
    modes = 2 * ell + 1
    rng = np.random.default_rng(16)
    matrix = rng.uniform(size=(len(ell), len(ell)))
    matrix = (matrix + matrix.T) / 2 / modes[:, None]
    edges = np.array([20, 30, 40, 60])
    actual = module.binned_self_adjoint_response(lambda cl: matrix @ cl, ell, edges)
    expected = module.mode_count_binning(ell, edges) @ matrix
    np.testing.assert_allclose(actual, expected, rtol=5e-15)


def test_full_sky_response_is_just_binning(module):
    ell, edges = np.arange(2, 10), np.array([2, 5, 10])
    calls = []
    response = module.binned_self_adjoint_response(lambda cl: cl, ell, edges,
                                                  lambda *args: calls.append(args))
    np.testing.assert_allclose(response, module.mode_count_binning(ell, edges))
    assert calls == [(1, 2), (2, 2)]


@pytest.mark.parametrize("ell,edges", [([1, 1], [1, 3]), ([1, 2], [3, 4]),
                                       ([1.5, 2], [1, 3]), ([1, 2], [3, 1]),
                                       ([1, np.inf], [1, 3]), ([1, 2], [1, np.nan])])
def test_invalid_binning_fails(module, ell, edges):
    with pytest.raises(ValueError):
        module.mode_count_binning(np.array(ell), np.array(edges))


def test_invalid_forward_action_fails(module):
    with pytest.raises(ValueError, match="forward"):
        module.binned_self_adjoint_response(lambda cl: cl * np.nan, np.arange(2, 8), np.array([2, 8]))
