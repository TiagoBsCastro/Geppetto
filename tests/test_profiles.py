import jax
import jax.numpy as jnp

from geppetto.concentration import ConcentrationParams, concentration_power_law, duffy08_all_200c
from geppetto.cosmology import (
    Cosmology,
    bryan_norman_virial_overdensity,
    omega_m_at_redshift,
    rho_crit_physical,
    rho_mean_comoving,
)
from geppetto.profiles import (
    NFWProfileParams,
    TabulatedProjectedProfileParams,
    nfw_density,
    nfw_projected_surface_density,
    nfw_scale_radius_and_density,
    tabulated_projected_surface_density,
)


def test_duffy_like_concentration_shape_and_gradient():
    mass = jnp.array([1.0e13, 1.0e14])
    redshift = jnp.array([0.0, 0.5])

    def total_c(amplitude):
        params = duffy08_all_200c()._replace(amplitude=amplitude)
        return jnp.sum(concentration_power_law(mass, redshift, params))

    grad = jax.grad(total_c)(5.71)
    assert jnp.isfinite(grad)
    assert concentration_power_law(mass, redshift, duffy08_all_200c()).shape == (2,)


def test_nfw_density_finite_gradient_wrt_amplitude():
    r = jnp.array([0.1, 0.5, 1.0])
    mass = jnp.array([1.0e14])
    redshift = jnp.array([0.3])

    def total_density(amplitude):
        params = ConcentrationParams(amplitude=amplitude)
        return jnp.sum(nfw_density(r, mass, redshift, Cosmology(), params, NFWProfileParams()))

    grad = jax.grad(total_density)(5.71)
    assert jnp.isfinite(grad)


def test_bryan_norman_overdensity_reference_conventions_are_equivalent():
    cosmology = Cosmology()
    redshift = jnp.array([0.0, 0.5, 1.0])
    omega_m_z = omega_m_at_redshift(redshift, cosmology)
    x = omega_m_z - 1.0
    expected_critical = 18.0 * jnp.pi**2 + 82.0 * x - 39.0 * x**2
    delta_critical = bryan_norman_virial_overdensity(
        redshift,
        cosmology,
        reference_density="critical",
    )
    delta_mean = bryan_norman_virial_overdensity(
        redshift,
        cosmology,
        reference_density="mean",
    )

    assert jnp.allclose(delta_critical, expected_critical)
    assert jnp.allclose(delta_mean, delta_critical / omega_m_z)
    assert jnp.allclose(
        delta_critical * rho_crit_physical(redshift, cosmology),
        delta_mean * rho_mean_comoving(cosmology) * (1.0 + redshift) ** 3,
    )
    assert jnp.isfinite(
        jax.grad(
            lambda z: bryan_norman_virial_overdensity(
                z,
                cosmology,
                reference_density="critical",
            )
        )(0.4)
    )


def test_bryan_norman_nfw_radius_is_reference_density_independent():
    mass = jnp.array([1.0e13, 1.0e14])
    redshift = jnp.array([0.2, 0.8])
    concentration = ConcentrationParams()
    critical = NFWProfileParams(
        reference_density="critical",
        overdensity_mode="bryan_norman",
    )
    mean = NFWProfileParams(
        reference_density="mean",
        overdensity_mode="bryan_norman",
    )

    critical_values = nfw_scale_radius_and_density(
        mass,
        redshift,
        Cosmology(),
        concentration,
        critical,
    )
    mean_values = nfw_scale_radius_and_density(
        mass,
        redshift,
        Cosmology(),
        concentration,
        mean,
    )

    for critical_value, mean_value in zip(critical_values, mean_values, strict=True):
        assert jnp.allclose(critical_value, mean_value, rtol=1.0e-6)


def test_projected_nfw_surface_density_is_finite():
    r_perp = jnp.array([0.01, 0.2, 1.0])
    sigma = nfw_projected_surface_density(
        r_perp,
        jnp.array([1.0e14]),
        jnp.array([0.3]),
        Cosmology(),
        duffy08_all_200c(),
    )
    assert sigma.shape == (3,)
    assert jnp.all(jnp.isfinite(sigma))


def test_tabulated_projected_surface_density_shape_support_and_normalization():
    x_grid = jnp.linspace(0.0, 1.0, 8)
    params = TabulatedProjectedProfileParams(
        x=x_grid,
        log_shape=jnp.zeros_like(x_grid),
    )
    mass = jnp.array([1.0e14])
    rmax = jnp.array([2.0])
    r = jnp.linspace(0.0, 2.0, 256)

    sigma = tabulated_projected_surface_density(r, mass, rmax, params)
    outside = tabulated_projected_surface_density(
        jnp.array([2.01, 3.0]),
        mass,
        rmax,
        params,
    )
    expected_sigma = mass[0] / (jnp.pi * rmax[0] ** 2)
    mass_integrand = 2.0 * jnp.pi * r * sigma
    recovered_mass = jnp.sum(
        0.5 * (mass_integrand[1:] + mass_integrand[:-1]) * (r[1:] - r[:-1])
    )

    assert sigma.shape == (256,)
    assert jnp.all(jnp.isfinite(sigma))
    assert jnp.all(sigma >= 0.0)
    assert jnp.allclose(sigma, expected_sigma, rtol=1.0e-6)
    assert jnp.all(outside == 0.0)
    assert jnp.allclose(recovered_mass, mass[0], rtol=5.0e-4)


def test_tabulated_projected_surface_density_gradient_wrt_log_shape_is_finite():
    x_grid = jnp.linspace(0.0, 1.0, 6)
    r = jnp.array([0.0, 0.2, 0.6, 1.0])
    mass = jnp.array([1.0e14])
    rmax = jnp.array([1.5])

    def objective(log_shape):
        params = TabulatedProjectedProfileParams(x=x_grid, log_shape=log_shape)
        return jnp.sum(tabulated_projected_surface_density(r, mass, rmax, params))

    grad = jax.grad(objective)(jnp.linspace(0.0, -1.0, x_grid.shape[0]))
    assert grad.shape == x_grid.shape
    assert jnp.all(jnp.isfinite(grad))


def test_tabulated_projected_surface_density_stops_rmax_gradient():
    x_grid = jnp.linspace(0.0, 1.0, 6)
    params = TabulatedProjectedProfileParams(
        x=x_grid,
        log_shape=jnp.zeros_like(x_grid),
    )

    def objective(rmax):
        return jnp.sum(
            tabulated_projected_surface_density(
                jnp.array([0.1, 0.2]),
                jnp.array([1.0e14]),
                rmax,
                params,
            )
        )

    assert jax.grad(objective)(2.0) == 0.0


def test_tabulated_projected_surface_density_stops_x_grid_gradient():
    params = TabulatedProjectedProfileParams(
        x=jnp.linspace(0.0, 1.0, 6),
        log_shape=jnp.linspace(0.0, -1.0, 6),
    )

    def objective(profile_params):
        return jnp.sum(
            tabulated_projected_surface_density(
                jnp.array([0.1, 0.2, 0.7]),
                jnp.array([1.0e14]),
                2.0,
                profile_params,
            )
        )

    grad = jax.grad(objective)(params)
    assert jnp.all(grad.x == 0.0)
    assert grad.log_shape.shape == params.log_shape.shape
    assert jnp.all(jnp.isfinite(grad.log_shape))
