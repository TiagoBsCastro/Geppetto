import jax
import jax.numpy as jnp

from geppetto.concentration import ConcentrationParams, concentration_power_law, duffy08_all_200c
from geppetto.cosmology import Cosmology
from geppetto.profiles import NFWProfileParams, nfw_density, nfw_projected_surface_density


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
