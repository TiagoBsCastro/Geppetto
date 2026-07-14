"""GEPPETTO: differentiable one-halo profile painting for PINOCCHIO catalogues."""

from geppetto.catalog import (
    AdaptiveLightconeStencil,
    AngularAssignmentParams,
    HaloCatalog,
    LightconeHaloCatalog,
    LightconeSparseStencil,
    from_spherical_lightcone,
)
from geppetto.concentration import ConcentrationParams, duffy08_all_200c, duffy08_relaxed_200c
from geppetto.cosmology import (
    Cosmology,
    bryan_norman_virial_overdensity,
    omega_m_at_redshift,
)
from geppetto.painters import (
    density_at_points,
    density_at_points_chunked,
    paint_box_density_grid,
    paint_lightcone_particle_count_map_sparse,
    paint_lightcone_particle_count_map_tabulated_sparse,
    paint_lightcone_surface_density,
    paint_lightcone_surface_density_sparse,
    paint_lightcone_surface_density_tabulated_sparse,
)
from geppetto.profiles import NFWProfileParams, TabulatedProjectedProfileParams
from geppetto.theory import (
    AngularPowerSpectra,
    HaloMassFunctionTable,
    LinearTheoryTable,
    exact_linear_shell_cls,
    hybrid_angular_power_spectra,
    limber_shell_cls,
    linear_matter_power,
    nfw_fourier_profile,
    one_halo_matter_power,
)

__all__ = [
    "AdaptiveLightconeStencil",
    "AngularAssignmentParams",
    "AngularPowerSpectra",
    "ConcentrationParams",
    "Cosmology",
    "HaloCatalog",
    "HaloMassFunctionTable",
    "LightconeHaloCatalog",
    "LightconeSparseStencil",
    "LinearTheoryTable",
    "NFWProfileParams",
    "TabulatedProjectedProfileParams",
    "bryan_norman_virial_overdensity",
    "density_at_points",
    "density_at_points_chunked",
    "duffy08_all_200c",
    "duffy08_relaxed_200c",
    "exact_linear_shell_cls",
    "from_spherical_lightcone",
    "hybrid_angular_power_spectra",
    "limber_shell_cls",
    "linear_matter_power",
    "nfw_fourier_profile",
    "omega_m_at_redshift",
    "paint_box_density_grid",
    "paint_lightcone_particle_count_map_sparse",
    "paint_lightcone_particle_count_map_tabulated_sparse",
    "paint_lightcone_surface_density",
    "paint_lightcone_surface_density_sparse",
    "paint_lightcone_surface_density_tabulated_sparse",
    "one_halo_matter_power",
]

__version__ = "0.1.0"
