"""Small differentiability, unit, luminosity and radial-sampling checks."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from geppetto.galaxies.models import (
    SatelliteParams,
    ThresholdParams,
    central_probability,
    nfw_inverse_cdf,
    nfw_radial_cdf,
    occupation,
    satellite_scales,
)

jax.config.update("jax_enable_x64", True)


def test_occupation_shape_and_finite_nonzero_shift_gradient():
    threshold = ThresholdParams(12., 11., 13., .3, 1.1)
    mass = jnp.array([10., 11.8, 12., 13., 14.])
    central, satellite = occupation(mass, threshold)
    assert central.shape == satellite.shape == (5,)
    assert np.all((central >= 0) & (central <= 1))
    assert np.all(satellite >= 0)
    gradient = jax.grad(lambda shift: jnp.sum(sum(occupation(mass, threshold, shift))))(.05)
    assert np.isfinite(gradient) and gradient < 0
    assert jax.grad(lambda x: central_probability(x, threshold))(12.) > 0


def test_compact_central_tails_are_exact_and_nested():
    params = ThresholdParams(jnp.array([13., 12.5, 12.]), 11., jnp.array([14., 13.5, 13.]), .3, 1.)
    mass = jnp.linspace(9., 16., 1001)[:, None]
    central, satellite = jax.jit(occupation)(mass, params)
    assert np.all(np.diff(central, axis=1) >= -1.e-15)
    assert np.all(np.diff(satellite, axis=1) >= -1.e-15)
    np.testing.assert_array_equal(central[0], np.zeros(3))
    np.testing.assert_array_equal(central[-1], np.ones(3))
    np.testing.assert_allclose(central_probability(13.-np.sqrt(6)*.3+1.e-5, ThresholdParams(13., 12., 14., .3, 1.)), (2/3)*(1.e-5/(np.sqrt(6)*.3))**4, rtol=1.e-8)


def test_effective_satellite_units_and_gradient():
    mass = jnp.array([1.e12, 1.e14])
    z = jnp.array([0., 1.])
    density = 2.775499745e11*.3
    radius, c, sigma = satellite_scales(mass, z, density)
    np.testing.assert_allclose(4*np.pi/3*radius**3*200*density, mass)
    physical = radius/(1+z)
    np.testing.assert_allclose(sigma**2, SatelliteParams().gravitational_constant*mass/(2*physical))
    assert radius.shape == c.shape == sigma.shape == (2,)
    derivative = jax.jacfwd(lambda factor: satellite_scales(mass, z, density, SatelliteParams(radius_factor=factor))[0])(1.)
    np.testing.assert_allclose(derivative, radius)


def test_nfw_inverse_cdf_distribution_endpoints_and_gradient():
    rng = np.random.default_rng(183)
    u = jnp.asarray(rng.random(20000))
    radius = nfw_inverse_cdf(u, 5.)
    assert radius.shape == u.shape
    np.testing.assert_allclose(nfw_radial_cdf(radius, 5.), u, atol=5.e-15)
    np.testing.assert_array_equal(nfw_inverse_cdf(jnp.array([0., 1.]), 5.), [0., 1.])
    bins = jnp.linspace(0., 1., 21)
    measured = np.array([np.mean(radius <= x) for x in bins])
    np.testing.assert_allclose(measured, nfw_radial_cdf(bins, 5.), atol=.012)
    derivative = jax.jacfwd(lambda c: nfw_inverse_cdf(jnp.array([.1, .5, .9]), c))(5.)
    finite_difference = (nfw_inverse_cdf(jnp.array([.1, .5, .9]), 5.001)-nfw_inverse_cdf(jnp.array([.1, .5, .9]), 4.999))/.002
    np.testing.assert_allclose(derivative, finite_difference, rtol=1.e-7, atol=1.e-9)
    assert np.all(derivative < 0)


def test_native_velocity_convention_and_sky_basis():
    pytest.importorskip("scipy")
    from geppetto.galaxies.adapters import cone_basis, observed_redshift, sky_coordinates

    basis = cone_basis([-1., 2., -3.])
    np.testing.assert_allclose(basis@basis.T, np.eye(3), atol=1.e-15)
    positions = np.array([[100., 0., 0.], [0., 100., 0.]]) @ basis
    lon, lat = sky_coordinates(positions, basis)
    np.testing.assert_allclose(lon % 360, [0., 90.], atol=1.e-12)
    np.testing.assert_allclose(lat, [0., 0.], atol=1.e-12)
    np.testing.assert_allclose(observed_redshift(np.array([[100., 0., 0.]]), np.array([[299.792458, 50., 60.]]), np.array([.2])), [.2012])


def test_luminosity_sampler_reproducibility_and_nonnested_rejection():
    pytest.importorskip("h5py")
    pytest.importorskip("scipy")
    from geppetto.galaxies.pipeline import inverse_cumulative_rows, sample_luminosities

    mags = np.array([-25., -23., -22., -21.])
    central = np.tile([0., .1, .5, .95], (3000, 1))
    satellite = np.tile([0., .05, .2, .5], (3000, 1))
    first = sample_luminosities(mags, central, satellite, np.random.default_rng(45))
    second = sample_luminosities(mags, central, satellite, np.random.default_rng(45))
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, b)
    hosts, magnitude, is_central = first
    assert len(np.unique(hosts[is_central])) == np.sum(is_central)
    assert np.all((magnitude >= -25) & (magnitude < -21))
    bad = np.array([[0., 1., .5, 1.1]])
    with pytest.raises(ValueError, match="non-nested"):
        inverse_cumulative_rows(mags, bad, np.array([.2]))


def test_native_hmf_weights_are_counts_per_volume():
    pytest.importorskip("scipy")
    from geppetto.galaxies.calibration import native_hmf_quadrature

    mass = np.array([1.e12, 2.e12, 2.e12, 1.e13])
    centers, weights = native_hmf_quadrature(mass, np.array([True, True, True, False]), [11.9, 12.1, 12.5, 13.1], 1000.)
    np.testing.assert_allclose(weights, [.001, .002, 0.])
    np.testing.assert_allclose(centers[:2], np.log10([1.e12, 2.e12]))


def test_upstream_lf_is_independent_and_integrates_consistently():
    pytest.importorskip("scipy")
    from scipy.integrate import quad

    from geppetto.galaxies.adapters import IndependentLuminosityFunction

    root = Path(__file__).parents[1]/"outputs/galaxy_lightcone/vendor/hodpy"
    if not root.exists():
        pytest.skip("Fetch pinned optional hodpy sources with scripts/fetch_hodpy.sh")
    lf = IndependentLuminosityFunction(root)
    assert not any("target_lf.dat" in name for name in lf.input_hashes)
    z = .25
    a, b = -22.8, -21.9
    differential = quad(lambda m: float(lf.gama.Phi(m, z)), a, b, epsabs=1.e-12)[0]
    cumulative = lf.gama.Phi_cumulative(b, z)-lf.gama.Phi_cumulative(a, z)
    np.testing.assert_allclose(differential, cumulative, rtol=1.e-10)
    assert np.all(np.diff(lf.cumulative(np.linspace(-24, -19, 50), .15)) > 0)
