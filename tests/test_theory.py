from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from geppetto.concentration import ConcentrationParams
from geppetto.cosmology import Cosmology
from geppetto.io import (
    read_pinocchio_cosmology_table,
    read_pinocchio_mass_function_series,
)
from geppetto.profiles import NFWProfileParams, nfw_projected_surface_density
from geppetto.theory import (
    HaloMassFunctionTable,
    LinearTheoryTable,
    exact_linear_shell_cls,
    gauss_legendre_rule,
    hybrid_angular_power_spectra,
    limber_shell_cls,
    linear_matter_power,
    nfw_fourier_profile,
    one_halo_matter_power,
    particle_count_shot_noise,
    resolved_halo_mass_fraction,
)


def _linear_theory() -> LinearTheoryTable:
    return LinearTheoryTable(
        h=0.7,
        omega_m0=0.3,
        scale_factor=jnp.asarray([0.5, 0.75, 1.0]),
        chi_mpc_h=jnp.asarray([1800.0, 800.0, 0.0]),
        omega_m=jnp.asarray([0.75, 0.5, 0.3]),
        growth=jnp.asarray([0.5, 0.75, 1.0]),
        k_h_mpc=jnp.asarray([1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0]),
        power_mpc_h3=jnp.asarray([1.0e3, 5.0e3, 1.0e3, 50.0, 1.0]),
    )


def _mass_function() -> HaloMassFunctionTable:
    mass = jnp.asarray([1.0e12, 3.0e12, 1.0e13, 3.0e13, 1.0e14])
    shape = jnp.asarray([2.0e-3, 1.0e-3, 4.0e-4, 8.0e-5, 1.0e-5])
    return HaloMassFunctionTable(
        scale_factor=jnp.asarray([0.5, 1.0]),
        log_mass_msun_h=jnp.log(mass),
        dndlnm_mpc_h3=jnp.stack([0.7 * shape, shape]),
    )


def test_linear_power_uses_pinocchio_growth_squared():
    theory = _linear_theory()
    present = linear_matter_power(jnp.asarray(0.1), 0.0, theory)
    redshift_one = linear_matter_power(jnp.asarray(0.1), 1.0, theory)

    assert present == pytest.approx(1000.0)
    assert redshift_one == pytest.approx(250.0)


def test_nfw_fourier_profile_is_normalized_and_differentiable():
    mass = jnp.asarray([1.0e13, 1.0e14])
    quadrature = gauss_legendre_rule(32)
    profile = nfw_fourier_profile(
        jnp.asarray([0.0, 0.5]),
        mass,
        jnp.asarray(0.3),
        Cosmology(),
        ConcentrationParams(),
        quadrature=quadrature,
    )

    assert profile.shape == (2, 2)
    np.testing.assert_allclose(profile[0], 1.0, rtol=0.0, atol=2.0e-7)
    gradient = jax.grad(
        lambda amplitude: jnp.sum(
            nfw_fourier_profile(
                jnp.asarray(0.5),
                mass,
                jnp.asarray(0.3),
                Cosmology(),
                ConcentrationParams(amplitude=amplitude),
                quadrature=quadrature,
            )
        )
    )(5.71)
    assert jnp.isfinite(gradient)


def test_one_halo_power_shape_gradient_and_ngp_independence():
    theory = _linear_theory()
    hmf = _mass_function()
    k = jnp.asarray([0.05, 0.5])
    power = one_halo_matter_power(
        k,
        0.2,
        theory,
        hmf,
        ConcentrationParams(),
        profile_quadrature=gauss_legendre_rule(16),
    )

    assert power.shape == (2,)
    assert jnp.all(jnp.isfinite(power))
    assert jnp.all(power > 0.0)
    gradient = jax.grad(
        lambda amplitude: jnp.sum(
            one_halo_matter_power(
                k,
                0.2,
                theory,
                hmf,
                ConcentrationParams(amplitude=amplitude),
                profile_quadrature=gauss_legendre_rule(16),
            )
        )
    )(5.71)
    assert jnp.isfinite(gradient)

    ngp_gradient = jax.grad(
        lambda amplitude: one_halo_matter_power(
            jnp.asarray(0.5),
            0.2,
            theory,
            hmf,
            ConcentrationParams(amplitude=amplitude),
            theta_resolution_rad=np.pi,
            profile_quadrature=gauss_legendre_rule(16),
        )
    )(5.71)
    assert ngp_gradient == pytest.approx(0.0, abs=1.0e-8)


def test_resolved_mass_fraction_uses_only_measured_hmf_support():
    fraction = resolved_halo_mass_fraction(0.0, _mass_function(), Cosmology(omega_m=0.3))
    assert jnp.isfinite(fraction)
    assert 0.0 < fraction < 1.0


def test_limber_shell_cls_shape_and_concentration_gradient():
    ell = jnp.asarray([20, 40, 60])
    radial_rule = gauss_legendre_rule(8)
    profile_rule = gauss_legendre_rule(12)

    def one_halo_sum(amplitude):
        _, one_halo = limber_shell_cls(
            ell,
            0.1,
            0.2,
            _linear_theory(),
            _mass_function(),
            ConcentrationParams(amplitude=amplitude),
            radial_quadrature=radial_rule,
            profile_quadrature=profile_rule,
        )
        return jnp.sum(one_halo)

    linear, one_halo = limber_shell_cls(
        ell,
        0.1,
        0.2,
        _linear_theory(),
        _mass_function(),
        ConcentrationParams(),
        radial_quadrature=radial_rule,
        profile_quadrature=profile_rule,
    )
    assert linear.shape == one_halo.shape == (3,)
    assert jnp.all(jnp.isfinite(linear))
    assert jnp.all(jnp.isfinite(one_halo))
    assert jnp.isfinite(jax.grad(one_halo_sum)(5.71))


def test_hybrid_spectra_shapes_weighted_one_halo_and_shot_noise():
    ell = jnp.asarray([20, 40])
    shell_weights = jnp.asarray([0.25, 0.75])
    result = hybrid_angular_power_spectra(
        ell,
        [0.1, 0.2],
        [0.2, 0.3],
        _linear_theory(),
        _mass_function(),
        ConcentrationParams(),
        [NFWProfileParams(), NFWProfileParams()],
        shell_weights=shell_weights,
        mean_uncollapsed_counts_per_pixel=jnp.asarray([5.0, 10.0]),
        mean_total_counts_per_pixel=jnp.asarray([10.0, 20.0]),
        pixel_area_sr=0.1,
        ell_exact_max=0,
        radial_order=8,
        profile_order=12,
    )

    assert result.shell_total.shape == (2, 2)
    assert result.summed_total.shape == (2,)
    np.testing.assert_allclose(
        result.summed_one_halo,
        np.sum(np.asarray(shell_weights)[:, None] ** 2 * result.shell_one_halo, axis=0),
    )
    np.testing.assert_allclose(result.shell_particle_shot_noise[:, 0], [0.005, 0.0025])
    np.testing.assert_allclose(result.summed_particle_shot_noise, 0.1 * 15.0 / 30.0**2)


def test_particle_count_shot_noise_formula():
    noise = particle_count_shot_noise(jnp.asarray(4.0), jnp.asarray(10.0), 0.2)
    assert noise == pytest.approx(0.008)


def test_exact_linear_projection_converges_to_limber_at_switch():
    pytest.importorskip("scipy")
    case = Path(__file__).parents[1] / "examples" / "pinocchio_geppetto_case"
    theory = read_pinocchio_cosmology_table(case / "pinocchio.example.cosmology.out")
    hmf = read_pinocchio_mass_function_series(tuple(sorted(case.glob("*.mf.out"))))
    exact, exact_sum = exact_linear_shell_cls(
        np.asarray([100]),
        np.asarray([0.05]),
        np.asarray([0.1]),
        theory,
        radial_order=64,
        relative_tolerance=5.0e-4,
    )
    limber, _ = limber_shell_cls(
        jnp.asarray([100]),
        0.05,
        0.1,
        theory,
        hmf,
        ConcentrationParams(),
        radial_quadrature=gauss_legendre_rule(32),
        profile_quadrature=gauss_legendre_rule(8),
    )

    assert exact.shape == (1, 1)
    np.testing.assert_allclose(exact_sum, exact[0], rtol=1.0e-8)
    np.testing.assert_allclose(exact[0], limber, rtol=0.1)


def test_nfw_3d_transform_matches_projected_profile_hankel_transform():
    scipy = pytest.importorskip("scipy.special")
    mass = jnp.asarray(1.0e14)
    redshift = jnp.asarray(0.3)
    concentration = ConcentrationParams()
    profile_params = NFWProfileParams()
    cosmology = Cosmology()
    from geppetto.profiles import nfw_scale_radius_and_density

    r_delta, _, _, _ = nfw_scale_radius_and_density(
        mass,
        redshift,
        cosmology,
        concentration,
        profile_params,
    )
    radius = jnp.linspace(0.0, r_delta, 20_001)
    sigma = nfw_projected_surface_density(
        radius,
        mass,
        redshift,
        cosmology,
        concentration,
        profile_params,
    )
    k = 0.5
    radial_weight = np.asarray(radius * sigma)
    normalization = np.trapezoid(radial_weight, np.asarray(radius))
    hankel = np.trapezoid(
        radial_weight * scipy.j0(k * np.asarray(radius)),
        np.asarray(radius),
    ) / normalization
    transform = nfw_fourier_profile(
        jnp.asarray(k),
        jnp.asarray([mass]),
        redshift,
        cosmology,
        concentration,
        profile_params,
        quadrature=gauss_legendre_rule(64),
    )
    np.testing.assert_allclose(transform[0], hankel, rtol=2.0e-3)
