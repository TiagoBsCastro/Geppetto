from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from geppetto.cosmology import RHO_CRIT0_MSUNH_PER_MPCH3
from geppetto.halo_bias import NumericalPBSFitParams, fit_pinocchio_numerical_halo_bias
from geppetto.io import PinocchioMassFunction, pinocchio_mass_function_series_from_tables
from geppetto.theory import LinearTheoryTable


def _linear_theory() -> LinearTheoryTable:
    return LinearTheoryTable(
        h=0.7,
        omega_m0=0.3,
        scale_factor=jnp.asarray([0.5, 1.0]),
        chi_mpc_h=jnp.asarray([1800.0, 0.0]),
        omega_m=jnp.asarray([0.75, 0.3]),
        growth=jnp.asarray([0.5, 1.0]),
        k_h_mpc=jnp.logspace(-3, 1, 64),
        power_mpc_h3=jnp.ones(64),
    )


def _synthetic_hmf(redshift: float, *, n_mass: int = 20) -> PinocchioMassFunction:
    mass = np.geomspace(1.0e11, 3.0e15, n_mass)
    logarithmic_nu_slope = 0.28
    peak_height = 0.45 * (mass / 1.0e11) ** logarithmic_nu_slope
    amplitude = 0.27
    a = 0.82
    p = 0.21
    q = 1.3
    a_nu_squared = a * peak_height**2
    multiplicity = (
        amplitude
        * peak_height**q
        * np.exp(-0.5 * a_nu_squared)
        * (1.0 + a_nu_squared ** (-p))
    )
    mean_density = RHO_CRIT0_MSUNH_PER_MPCH3 * 0.3
    dndm = multiplicity * mean_density * logarithmic_nu_slope / mass**2
    counts = np.full(n_mass, 1000, dtype=np.int64)
    return PinocchioMassFunction(
        mass_msun_h=mass,
        number_density=dndm,
        number_density_plus_1sigma=dndm,
        number_density_minus_1sigma=dndm,
        halo_counts=counts,
        analytic_number_density=dndm,
        peak_height_nu=peak_height,
        source=Path(f"synthetic-z{redshift:g}.mf.out"),
        redshift=redshift,
    )


def test_numerical_hmf_pbs_fit_recovers_bias_shape_and_applies_correction():
    tables = (_synthetic_hmf(0.0), _synthetic_hmf(1.0))
    mass_function = pinocchio_mass_function_series_from_tables(tables)

    def correction(omega_m_z, dlnsigma_dlnr, s8):
        del omega_m_z, s8
        return 1.1 + 0.0 * dlnsigma_dlnr

    bias, diagnostics = fit_pinocchio_numerical_halo_bias(
        tables,
        mass_function,
        _linear_theory(),
        correction_function=correction,
    )

    assert bias.linear_bias.shape == (2, 20)
    np.testing.assert_allclose(bias.correction, 1.1)
    np.testing.assert_allclose(bias.linear_bias, 1.1 * bias.pbs_bias, rtol=2.0e-7)
    assert np.all(np.isfinite(bias.linear_bias))
    assert np.all(np.diff(np.asarray(bias.pbs_bias), axis=1) > 0.0)
    assert len(diagnostics) == 2
    assert max(item.weighted_log_residual for item in diagnostics) < 1.0e-10


def test_numerical_hmf_pbs_fit_accepts_float32_scale_factor_roundoff():
    tables = (_synthetic_hmf(0.137969), _synthetic_hmf(0.629465))
    mass_function = pinocchio_mass_function_series_from_tables(tables)

    bias, diagnostics = fit_pinocchio_numerical_halo_bias(
        tables,
        mass_function,
        _linear_theory(),
        correction_function=lambda *args: 1.0,
    )

    assert bias.linear_bias.shape == (2, 20)
    assert len(diagnostics) == 2


def test_numerical_hmf_pbs_fit_rejects_too_few_populated_bins():
    tables = (_synthetic_hmf(0.0, n_mass=7),)
    mass_function = pinocchio_mass_function_series_from_tables(tables)

    with pytest.raises(ValueError, match="too few populated bins|invalid numerical HMF"):
        fit_pinocchio_numerical_halo_bias(
            tables,
            mass_function,
            _linear_theory(),
            params=NumericalPBSFitParams(minimum_populated_bins=8),
            correction_function=lambda *args: 1.0,
        )


def test_numerical_hmf_pbs_fit_rejects_large_weighted_shape_residual():
    table = _synthetic_hmf(0.0)
    modulation = np.where(np.arange(len(table)) % 2 == 0, 0.5, 2.0)
    table = replace(table, number_density=table.number_density * modulation)
    mass_function = pinocchio_mass_function_series_from_tables((table,))

    with pytest.raises(ValueError, match="weighted_log_residual"):
        fit_pinocchio_numerical_halo_bias(
            (table,),
            mass_function,
            _linear_theory(),
            params=NumericalPBSFitParams(maximum_weighted_log_residual=0.05),
            correction_function=lambda *args: 1.0,
        )
