"""Audit-only numerical references: no halo/HOD fit or large fixture needed."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")

SPEC = importlib.util.spec_from_file_location("galaxy_diagnostics", Path(__file__).parents[1]/"examples/diagnose_pinocchio_galaxies.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_half_open_histogram_does_not_clip_outliers():
    result = MODULE.selected_bins(np.array([-2., 0., .5, 1., 2., 3.]), [0., 1., 2.])
    np.testing.assert_array_equal(result, [2, 1])


def test_conditional_histogram_variance_is_not_poisson():
    edges = np.array([-100., 0., 100.])
    components = (np.zeros(10), np.zeros(10), np.ones(10), np.zeros(10), np.ones(10))
    expected, variance = MODULE.probability_histogram(edges,
        lambda x, part: MODULE.mixture_cdf(x, tuple(v[part] for v in components)), 10)
    np.testing.assert_allclose(expected, [5., 5.])
    np.testing.assert_allclose(variance, [2.5, 2.5])


def test_piecewise_apparent_cdf_accounts_for_clipped_tails_and_negative_slope():
    from scipy.special import ndtr
    knots = np.array([-1., 0., 1.])
    components = (np.array([.4]), np.array([.2]), np.array([.7]), np.array([-.3]), np.array([1.2]))
    edges = np.array([-100., -.8, -.2, .2, .8, 100.])
    positive = MODULE.piecewise_colour_cdf(edges, knots, knots[:, None], components)[:, 0]
    reference = .4*ndtr((edges-.2)/.7)+.6*ndtr((edges+.3)/1.2)
    np.testing.assert_allclose(positive, reference)
    negative = MODULE.piecewise_colour_cdf(edges, knots, -knots[:, None], components)[:, 0]
    np.testing.assert_allclose(negative[1:-1], 1-(.4*ndtr((-edges[1:-1]-.2)/.7)+.6*ndtr((-edges[1:-1]+.3)/1.2)))
    constant = MODULE.piecewise_colour_cdf([-1., 0., 1.], knots, np.zeros((3, 1)), components)[:, 0]
    np.testing.assert_allclose(constant, [0., 0., 1.])


def test_landy_szalay_pair_normalizations_match_explicit_pairs():
    from scipy.spatial.distance import cdist
    points = MODULE.angular_vectors(np.array([0., 30., 90., 150.]), np.zeros(4))
    random = MODULE.angular_vectors(np.array([15., 45., 75., 120., 180.]), np.zeros(5))
    edges = np.array([.01, .6, 1.3, 2.01])
    value, dd, dr, rr = MODULE.landy_szalay(points, random, edges)
    expected_dd = np.histogram(cdist(points, points).ravel(), edges)[0]/12
    expected_dr = np.histogram(cdist(points, random).ravel(), edges)[0]/20
    expected_rr = np.histogram(cdist(random, random).ravel(), edges)[0]/20
    np.testing.assert_allclose(dd, expected_dd)
    np.testing.assert_allclose(dr, expected_dr)
    np.testing.assert_allclose(rr, expected_rr)
    np.testing.assert_allclose(value, (expected_dd-2*expected_dr+expected_rr)/expected_rr)


def test_plot_only_never_loads_or_recalibrates_the_model(tmp_path, monkeypatch):
    output = tmp_path/"diagnostics"
    output.mkdir()
    settings = {"output_dir": str(output)}
    config = tmp_path/"config.json"
    config.write_text(json.dumps(settings))
    (output/"manifest.json").write_text(json.dumps({"settings": settings}))
    calls = []
    monkeypatch.setattr(MODULE, "compute", lambda *_: pytest.fail("Plot-only must not compute predictions"))
    monkeypatch.setattr(MODULE, "plot_all", lambda *_: calls.append("plot"))
    monkeypatch.setattr(MODULE, "write_report", lambda *_: calls.append("report"))
    monkeypatch.setattr(sys, "argv", ["diagnostics", "--config", str(config), "--plot-only"])
    MODULE.main()
    assert calls == ["plot", "report"]
    assert (output/"code"/"diagnose_pinocchio_galaxies.py").exists()
