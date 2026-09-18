from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad

from geppetto.concentration import ConcentrationParams
from geppetto.cosmology import RHO_CRIT0_MSUNH_PER_MPCH3
from geppetto.hmf import (
    HMFIntegrationParams,
    NormalizedHMFFit,
    NormalizedHMFFitParams,
    average_normalized_hmf,
    fit_normalized_hmf,
    multiplicity_tail_integrals,
    normalized_multiplicity,
    peak_height_from_power,
    tabulate_normalized_hmf,
)
from geppetto.io import PinocchioMassFunction
from geppetto.profiles import NFWProfileParams
from geppetto.theory import (
    LinearPowerEvolutionTable,
    LinearTheoryTable,
    NormalizedHaloPowerTable,
    NormalizedHaloTable,
    gauss_legendre_rule,
    normalized_halo_matter_power,
    normalized_halo_power_grid,
    normalized_one_halo_shell_cls,
)


@pytest.fixture(autouse=True)
def x64():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


def linear_table():
    k = jnp.geomspace(1e-5, 1e3, 4096)
    return LinearTheoryTable(
        0.7,
        0.3,
        jnp.array([0.5, 1.0]),
        jnp.array([1800.0, 0.0]),
        jnp.array([0.75, 0.3]),
        jnp.array([0.5, 1.0]),
        k,
        20 / k**2,
    )


def synthetic_hmf(z, theory, shape=None, collapse_threshold=1.686):
    if shape is None:
        shape = (np.log(0.8), 0.25, 0.7)
    mass = np.geomspace(1e10, 1e15, 32)
    nu, slope = peak_height_from_power(mass, z, theory, collapse_threshold=collapse_threshold)
    density = RHO_CRIT0_MSUNH_PER_MPCH3 * 0.3
    dndm = normalized_multiplicity(nu, *shape) * density * slope / mass**2
    return PinocchioMassFunction(
        mass, dndm, dndm, dndm, np.full(32, 1000), dndm, nu, Path(f"synthetic-{z}.mf.out"), z
    )


def test_analytic_mass_and_pbs_tail_closure():
    for shape in ((np.log(0.8), 0.25, 0.7), (0.0, 1.0, 0.03), (np.log(4.0), 0.0, 1.685)):
        integral = quad(lambda x, shape=shape: normalized_multiplicity(np.exp(x), *shape), -60, 4)[
            0
        ]
        integral += multiplicity_tail_integrals(np.exp(-60), *shape)[0]
        np.testing.assert_allclose(integral, 1, atol=1e-10)
        lo = multiplicity_tail_integrals(0.9, *shape)
        hi = multiplicity_tail_integrals(0.9, *shape, upper=True)
        np.testing.assert_allclose(np.array(lo) + hi, [1.0, 1.0], atol=1e-13)


def test_normalized_fit_recovers_shape_and_rejects_bad_peak_heights():
    theory = linear_table()
    tables = [synthetic_hmf(z, theory) for z in (0.0, 1.0)]
    fits = fit_normalized_hmf(tables, theory)
    for fit in fits:
        np.testing.assert_allclose([fit.log_a, fit.p, fit.s], [np.log(0.8), 0.25, 0.7], atol=1e-6)
        assert fit.weighted_log_residual < 1e-10
    tables[0] = replace(tables[0], peak_height_nu=tables[0].peak_height_nu * 1.1)
    with pytest.raises(ValueError, match="peak-height closure"):
        fit_normalized_hmf(tables, theory)


def test_peak_height_uses_scale_dependent_power_not_background_growth():
    theory = linear_table()
    evolution = LinearPowerEvolutionTable(
        theory.scale_factor,
        theory.k_h_mpc,
        jnp.stack([0.64 * theory.power_mpc_h3, theory.power_mpc_h3]),
    )
    mass = np.array([1e11, 1e13])
    now, _ = peak_height_from_power(mass, 0.0, theory)
    past, _ = peak_height_from_power(mass, 1.0, theory, evolution)
    np.testing.assert_allclose(past / now, 1 / 0.8, rtol=1e-12)


def test_fit_starting_points_respect_nondefault_collapse_threshold():
    theory = linear_table()
    shape = (np.log(0.8), 0.25, 0.3)
    table = synthetic_hmf(0.0, theory, shape, collapse_threshold=0.8)
    (fit,) = fit_normalized_hmf(
        [table], theory, params=NormalizedHMFFitParams(collapse_threshold=0.8)
    )
    np.testing.assert_allclose([fit.log_a, fit.p, fit.s], shape, atol=1e-6)


@pytest.mark.parametrize(
    "controls",
    [
        HMFIntegrationParams(minimum_mass=np.nan),
        HMFIntegrationParams(maximum_mass=np.inf),
        HMFIntegrationParams(closure_rtol=np.nan),
        HMFIntegrationParams(maximum_high_tail=np.nan),
    ],
)
def test_integration_rejects_nonfinite_controls(controls):
    with pytest.raises(ValueError, match="invalid HMF integration controls"):
        tabulate_normalized_hmf([], np.array([0.0, 1.0]), linear_table(), params=controls)


def test_bias_positivity_is_checked_beyond_low_peak_height_limit():
    fits = [NormalizedHMFFit("synthetic", z, 0.0, 2.0, 1.5, 32, 0.0, 0.0) for z in (0.0, 1.0)]
    with pytest.raises(ValueError, match="invalid Castro/PBS bias"):
        tabulate_normalized_hmf(
            fits, np.array([0.0, 1.0]), linear_table(), correction_function=lambda *args: 1.0
        )


def test_mass_bias_closure_and_tail_bound_at_intermediate_redshifts():
    theory = linear_table()
    fits = [
        NormalizedHMFFit("synthetic", z, np.log(0.8), 0.25, 0.7, 32, 0.0, 0.0) for z in (0.0, 1.0)
    ]
    table, diagnostics = tabulate_normalized_hmf(
        fits,
        np.array([0.0, 0.2, 0.5, 1.0]),
        theory,
        params=HMFIntegrationParams(mass_order=256),
        correction_function=lambda omega, slope, s8: 1.2 + 0.1 * slope**2,
    )
    assert table.mass_fraction_weight.shape == (4, 256)
    np.testing.assert_allclose(
        np.sum(table.mass_fraction_weight, axis=1) + table.tail_mass_fraction, 1.0, atol=1e-5
    )
    np.testing.assert_allclose(
        np.sum(table.biased_mass_fraction_weight, axis=1) + table.tail_biased_mass_fraction,
        1.0,
        atol=2e-15,
    )
    assert all(row.bias_renormalization != 1 for row in diagnostics)
    assert all(row.low_mass_one_halo_upper_bound_mpc_h3 > 0 for row in diagnostics)
    with pytest.raises(ValueError, match="upper HMF tail"):
        tabulate_normalized_hmf(
            fits,
            np.array([0.0, 1.0]),
            theory,
            params=HMFIntegrationParams(maximum_mass=1e13),
            correction_function=lambda *args: 1.0,
        )


def small_halos():
    mass = jnp.array([1e11, 1e13, 1e14])
    return NormalizedHaloTable(
        jnp.array([0.5, 1.0]),
        mass,
        jnp.array([[0.1, 0.2, 0.3], [0.2, 0.25, 0.35]]),
        jnp.array([[0.1, 0.3, 0.5], [0.2, 0.3, 0.4]]),
        jnp.array([0.4, 0.2]),
        jnp.array([0.1, 0.1]),
    )


def test_standard_low_k_limit_and_compensated_k4():
    theory, halos = linear_table(), small_halos()
    power = normalized_halo_matter_power(
        jnp.array([0.0, 1e-4, 2e-4]), 0.3, theory, halos, ConcentrationParams()
    )
    np.testing.assert_allclose(power.response**2, 1.0, atol=2e-7)
    assert power.one_halo_standard[0] > 0
    np.testing.assert_allclose(power.one_halo_standard, power.one_halo_standard[0], rtol=1e-7)
    assert power.one_halo_compensated[0] == 0
    np.testing.assert_allclose(
        power.one_halo_compensated[2] / power.one_halo_compensated[1], 16.0, rtol=2e-6
    )


@pytest.mark.parametrize("parameter", [0, 1, 2])
def test_normalized_power_gradients_match_finite_differences(parameter):
    theory, halos = linear_table(), small_halos()
    parameters = jnp.array([5.71, -0.084, -0.47])

    def evaluate(values):
        result = normalized_halo_matter_power(
            jnp.array([0.1, 1.0, 4.0]),
            0.3,
            theory,
            halos,
            ConcentrationParams(*values),
            profile_quadrature=gauss_legendre_rule(32),
        )
        return jnp.stack(result)

    automatic = jax.jacfwd(evaluate)(parameters)[..., parameter]
    step = 1e-4
    offset = jnp.zeros(3).at[parameter].set(step)
    finite = (evaluate(parameters + offset) - evaluate(parameters - offset)) / (2 * step)
    assert automatic.shape == (3, 3)
    assert np.all(np.isfinite(automatic))
    np.testing.assert_allclose(automatic, finite, rtol=2e-5, atol=1e-9)


def test_ngp_has_zero_concentration_derivatives_and_grid_projection_shapes():
    theory, halos = linear_table(), small_halos()
    k = jnp.geomspace(1e-3, 20, 32)
    z = jnp.array([0.6, 0.3, 0.1])
    params, rule = NFWProfileParams(), gauss_legendre_rule(16)

    def grid(amplitude):
        return normalized_halo_power_grid(
            k,
            z,
            theory,
            halos,
            ConcentrationParams(amplitude=amplitude),
            params,
            theta_resolution_rad=3.0,
            profile_quadrature=rule,
        )

    values = grid(5.71)
    assert all(value.shape == (3, 32) for value in values)
    assert all(np.all(np.asarray(value) == 0) for value in jax.jacfwd(grid)(5.71))
    power_table = NormalizedHaloPowerTable(1 / (1 + z), k, *values)
    spectra = normalized_one_halo_shell_cls(
        jnp.arange(2, 30), 0.2, 0.5, theory, power_table, radial_quadrature=rule
    )
    assert all(value.shape == (28,) for value in spectra)
    assert all(np.all(np.isfinite(value) & (value >= 0)) for value in spectra)


def test_ensemble_averages_bias_weighted_abundance_before_squaring():
    first = small_halos()
    second = first._replace(biased_mass_fraction_weight=first.biased_mass_fraction_weight[:, ::-1])
    averaged = average_normalized_hmf([first, second])
    theory = linear_table()

    def f(table):
        return normalized_halo_matter_power(
            jnp.array([1.0, 5.0]), 0.3, theory, table, ConcentrationParams()
        ).response

    np.testing.assert_allclose(f(averaged), (f(first) + f(second)) / 2, rtol=1e-14)
    assert not np.allclose(f(averaged) ** 2, (f(first) ** 2 + f(second) ** 2) / 2, rtol=1e-7)
    with pytest.raises(ValueError, match="quadratures must match"):
        average_normalized_hmf([first, first._replace(mass_msun_h=first.mass_msun_h * 2)])
