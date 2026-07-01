"""Baryonification extension point.

This file deliberately contains no placeholder physics disguised as a model. The
long-term goal is to implement an Aricò-style baryonification prescription as a
composition of differentiable transforms:

1. split the target mass profile into dark matter, gas, central-galaxy and
   ejected-gas components;
2. enforce mass bookkeeping at fixed halo mass and cosmology;
3. expose all baryonic parameters as JAX pytrees;
4. provide both 3D profile and projected-profile kernels;
5. validate against power-spectrum and profile-level reference cases.

The public API should eventually mirror ``profiles.nfw_density`` and
``profiles.nfw_projected_surface_density`` so painters can swap the profile
prescription without changing geometry code.
"""

from __future__ import annotations

from typing import NamedTuple


class AricoBaryonificationParams(NamedTuple):
    """Draft parameter container for a future Aricò-style baryonification model.

    Names are intentionally generic until the exact prescription and calibration
    target are selected.
    """

    gas_fraction_amplitude: float = 0.0
    gas_ejection_radius: float = 0.0
    gas_core_radius: float = 0.0
    stellar_fraction_amplitude: float = 0.0
    stellar_scale_radius: float = 0.0
    dark_matter_relaxation: float = 0.0
    outer_compensation_radius: float = 0.0


def not_implemented_message() -> str:
    """Return the current baryonification status."""

    return (
        "Aricò-style baryonification is a planned GEPPETTO extension. "
        "The current release includes differentiable NFW one-halo painting only."
    )
