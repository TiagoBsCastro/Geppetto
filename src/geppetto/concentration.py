"""Differentiable concentration--mass prescriptions."""

from __future__ import annotations

from typing import NamedTuple

from geppetto.types import Array


class ConcentrationParams(NamedTuple):
    """Power-law concentration--mass relation.

    The implemented form is

    ``c(M, z) = amplitude * (M / mass_pivot)**mass_slope * (1 + z)**redshift_slope``.

    This covers Duffy-like relations while keeping all parameters free for
    inference or calibration.
    """

    amplitude: float = 5.71
    mass_slope: float = -0.084
    redshift_slope: float = -0.47
    mass_pivot: float = 2.0e12


def concentration_power_law(mass: Array, redshift: Array, params: ConcentrationParams) -> Array:
    """Evaluate the differentiable power-law concentration relation."""

    return params.amplitude * (mass / params.mass_pivot) ** params.mass_slope * (1.0 + redshift) ** params.redshift_slope


def duffy08_all_200c() -> ConcentrationParams:
    """Duffy et al.-like 200c relation for the full halo sample.

    Returned values are intended as a starting point only. They remain ordinary
    parameters and can be differentiated, sampled or calibrated.
    """

    return ConcentrationParams(amplitude=5.71, mass_slope=-0.084, redshift_slope=-0.47, mass_pivot=2.0e12)


def duffy08_relaxed_200c() -> ConcentrationParams:
    """Duffy et al.-like 200c relation for relaxed haloes."""

    return ConcentrationParams(amplitude=6.71, mass_slope=-0.091, redshift_slope=-0.44, mass_pivot=2.0e12)
