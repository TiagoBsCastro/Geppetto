import importlib
from pathlib import Path

import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from geppetto.catalog import AngularAssignmentParams, LightconeHaloCatalog
from geppetto.concentration import ConcentrationParams
from geppetto.cosmology import Cosmology, halo_radius_delta_comoving
from geppetto.painters import paint_lightcone_particle_count_map_sparse
from geppetto.painting_theory import angular_assignment_moments
from geppetto.profiles import NFWProfileParams, nfw_halo_overdensity


@pytest.fixture(autouse=True)
def double_precision():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    return importlib.import_module("painting_theory_population")


def _arguments():
    nside = 16
    radius = float(halo_radius_delta_comoving(1.e14, .3, Cosmology()))
    chi = radius/(2*np.sin(1.5*np.sqrt(hp.nside2pixarea(nside))/2))
    axes = np.array([hp.ang2vec(1.13, .73), hp.ang2vec(.89, 2.19), hp.ang2vec(2.1, 1.73)])
    return dict(mass_msun_h=np.array([1.e11, 1.e14, 1.e16]), redshift=.3, chi_mpc_h=chi,
                orientation_unit_vectors=axes[:, None, :], nside=nside, particle_mass_msun_h=1.e10,
                cosmology=Cosmology(), profile=NFWProfileParams(), assignment=AngularAssignmentParams())


def test_ngp_geometry_does_not_query_and_has_zero_concentration_derivatives(module, monkeypatch):
    args = _arguments()
    args.update(mass_msun_h=args["mass_msun_h"][:1],
                orientation_unit_vectors=args["orientation_unit_vectors"][:1])
    monkeypatch.setattr(hp, "query_disc", lambda *a, **k: pytest.fail("NGP must not query"))
    geometry = module.build_population_geometry(**args)
    assert geometry.stencil is None
    value = module.population_assignment_moments(ConcentrationParams(), geometry, lmax=24)
    assert value.response.shape == (1, 25)
    np.testing.assert_array_equal(value, 1.)
    derivative = jax.jacfwd(lambda p: jnp.stack(module.population_assignment_moments(
        ConcentrationParams(*p, 2.e12), geometry, lmax=24,
    )))(jnp.array([5.71, -.084, -.47]))
    np.testing.assert_array_equal(derivative, 0.)


@pytest.mark.parametrize("overdensity_mode", ["constant", "bryan_norman"])
def test_population_moments_match_independent_fullsky_painting(module, overdensity_mode):
    args = _arguments()
    args["profile"] = NFWProfileParams(overdensity_mode=overdensity_mode)
    geometry = module.build_population_geometry(**args)
    result = module.population_assignment_moments(ConcentrationParams(), geometry, lmax=24)
    assert result.response.shape == (3, 25)
    np.testing.assert_allclose(np.asarray(result)[:, :, 0], 1., atol=1.e-12)
    operator = importlib.import_module("validate_painting_theory_operator")
    domain = operator.full_sky_domain(args["nside"])
    for group, mass in enumerate(args["mass_msun_h"]):
        axis = args["orientation_unit_vectors"][group, 0]
        catalog = LightconeHaloCatalog(jnp.asarray(axis[None, :]), jnp.array([args["chi_mpc_h"]]),
                                       jnp.array([mass]), jnp.array([args["redshift"]]))
        radius = np.asarray(halo_radius_delta_comoving(
            catalog.mass, catalog.redshift, args["cosmology"],
            overdensity=nfw_halo_overdensity(catalog.redshift, args["cosmology"], args["profile"]),
        ))
        stencil = module.build_adaptive_lightcone_stencil_for_mass_map(domain, catalog, radius, args["assignment"])
        counts = np.asarray(paint_lightcone_particle_count_map_sparse(
            stencil, catalog, particle_mass_msun_h=args["particle_mass_msun_h"],
            cosmology=args["cosmology"], profile_params=args["profile"], sample_chunk_size=256,
        ))
        active = np.flatnonzero(counts > 0)
        reference = np.asarray(hp.pix2vec(args["nside"], hp.vec2pix(args["nside"], *axis)))
        expected = angular_assignment_moments(np.asarray(hp.pix2vec(args["nside"], active)).T,
                                              counts[active]/(mass/args["particle_mass_msun_h"]),
                                              reference, lmax=24)
        np.testing.assert_allclose(np.asarray(result)[:, group], expected, rtol=2.e-6, atol=2.e-7)


def test_all_three_population_concentration_jacobians_match_finite_differences(module):
    geometry = module.build_population_geometry(**_arguments())
    def evaluate(p):
        return jnp.stack(module.population_assignment_moments(
            ConcentrationParams(*p, 2.e12), geometry, lmax=16,
        ))
    parameters = jnp.array([5.71, -.084, -.47])
    derivative = np.asarray(jax.jit(jax.jacfwd(evaluate))(parameters))
    value = jax.jit(evaluate)
    finite = []
    for step in (1.e-3, 5.e-4):
        finite.append(np.stack([(np.asarray(value(parameters.at[i].add(step)))
                                 - np.asarray(value(parameters.at[i].add(-step))))/(2*step)
                                for i in range(3)], axis=-1))
    richardson = (4*finite[1]-finite[0])/3
    np.testing.assert_allclose(derivative, richardson, rtol=3.e-4, atol=5.e-8)
    np.testing.assert_allclose(derivative[:, :, 0], 0., atol=1.e-10)
    np.testing.assert_array_equal(derivative[:, 0], 0.)
    assert np.all(np.isfinite(derivative))
    assert np.max(abs(derivative[:, 1:])) > .001


def test_geometry_budget_and_assignment_validation(module):
    with pytest.raises(ValueError, match="budget"):
        module.build_population_geometry(**_arguments(), max_pair_count=1)
    args = _arguments()
    args["assignment"] = AngularAssignmentParams(n_resolution=1.5)
    with pytest.raises(ValueError, match="integer"):
        module.build_population_geometry(**args)
    args["assignment"] = AngularAssignmentParams(theta_resolution_rad=0.)
    with pytest.raises(ValueError, match="positive"):
        module.build_population_geometry(**args)


def test_harmonic_sum_matches_explicit_native_pixel_pairs_and_weight_jvp(module):
    pytest.importorskip("ducc0")
    rng = np.random.default_rng(42)
    vectors = rng.normal(size=(33, 3))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    weights = rng.uniform(size=33)
    weights /= weights.sum()
    tangent = rng.normal(size=33)
    tangent -= tangent.mean()
    reference = np.array([0., 0., 1.])
    values, jacobian = module.discrete_harmonic_moments(
        vectors, weights, reference, lmax=32, weight_jacobian=tangent[:, None],
    )
    exact = angular_assignment_moments(vectors, weights, reference, lmax=32)
    exact_jvp = jax.jvp(lambda w: jnp.stack(angular_assignment_moments(
        vectors, w, reference, lmax=32)), (jnp.asarray(weights),), (jnp.asarray(tangent),))[1]
    np.testing.assert_allclose(values, exact, rtol=1.e-8, atol=1.e-10)
    np.testing.assert_allclose(jacobian[:, :, 0], exact_jvp, rtol=1.e-8, atol=1.e-9)
    np.testing.assert_allclose(values[:, 0], 1., atol=1.e-10)
    np.testing.assert_allclose(jacobian[:, 0], 0., atol=1.e-9)


def test_auto_harmonic_population_preserves_painter_and_concentration_derivatives(module):
    pytest.importorskip("ducc0")
    args = _arguments()
    regular = module.build_population_geometry(**args)
    harmonic = module.build_population_geometry(**args, moment_backend="auto", max_pair_count=1)
    assert harmonic.histogram is None
    assert harmonic.harmonic is not None
    with pytest.raises(ValueError, match="host"):
        module.population_assignment_moments(ConcentrationParams(), harmonic, lmax=24)
    values, jacobian = module.population_harmonic_moments_on_host(
        ConcentrationParams(), harmonic, lmax=24, derivatives=True,
    )
    def evaluate(p):
        return jnp.stack(module.population_assignment_moments(
            ConcentrationParams(*p, 2.e12), regular, lmax=24,
        ))
    parameters = jnp.array([5.71, -.084, -.47])
    np.testing.assert_allclose(values, evaluate(parameters), rtol=5.e-6, atol=2.e-7)
    np.testing.assert_allclose(jacobian, jax.jacfwd(evaluate)(parameters), rtol=5.e-6, atol=2.e-7)
    np.testing.assert_array_equal(jacobian[:, 0], 0.)
    with pytest.raises(ValueError, match="native pixel budget"):
        module.build_population_geometry(**args, moment_backend="auto", max_native_pixels=1)
