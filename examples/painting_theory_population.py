"""Bounded host geometry and JAX moments of the production painting rule.

This helper deliberately calls the calibration script's production stencil
builder. It does not maintain a second NFW pixelization implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path

import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np

from geppetto.catalog import AdaptiveLightconeStencil, AngularAssignmentParams, LightconeHaloCatalog
from geppetto.concentration import ConcentrationParams
from geppetto.cosmology import Cosmology, halo_radius_delta_comoving
from geppetto.io import PinocchioMassMap, build_angular_histogram_geometry
from geppetto.painters import paint_lightcone_particle_count_map_sparse
from geppetto.painting_theory import (
    AngularAssignmentMoments,
    HistogramAngularGeometry,
    histogram_angular_assignment_moments,
)
from geppetto.profiles import NFWProfileParams, nfw_halo_overdensity
from paint_halo_particles_for_pinocchio_segment import (
    MassMapPixelIndex,
    build_adaptive_lightcone_stencil_for_mass_map,
    resolve_angular_assignment,
)


@dataclass(frozen=True)
class HarmonicPopulationGeometry:
    """Native global rows grouped by halo for a host spherical-harmonic sum."""

    vectors: np.ndarray
    offsets: np.ndarray
    references: np.ndarray
    groups: np.ndarray
    orientation_weight: float
    analytic_ngp: np.ndarray


@dataclass(frozen=True)
class PaintingPopulationGeometry:
    """Fixed, concentration-independent inputs for one bounded mass block.

    Distances are comoving Mpc/h, masses Msun/h, and native_halo_counts are
    M/m_particle. Global native rows cover every normalization sample; no
    survey mask is present. Analytic NGP groups need no catalogue or queries.
    """

    stencil: AdaptiveLightconeStencil | None
    catalog: LightconeHaloCatalog | None
    histogram: HistogramAngularGeometry | None
    native_indices: jax.Array
    native_halo_counts: jax.Array
    n_mass: int
    particle_mass_msun_h: float
    cosmology: Cosmology
    profile: NFWProfileParams
    harmonic: HarmonicPopulationGeometry | None = None


def build_population_geometry(
    mass_msun_h: np.ndarray,
    redshift: float,
    chi_mpc_h: float,
    orientation_unit_vectors: np.ndarray,
    *,
    nside: int,
    particle_mass_msun_h: float,
    cosmology: Cosmology,
    profile: NFWProfileParams,
    assignment: AngularAssignmentParams,
    angle_nodes: int = 8193,
    pair_chunk_size: int = 64,
    max_pair_count: int = 10_000_000,
    moment_backend: str = "pairs",
    max_native_pixels: int = 32_000_000,
) -> PaintingPopulationGeometry:
    """Build global stencils and native pair brackets for a bounded mass block.

    Masses have shape (n_mass,), orientations (n_mass,n_orientation,3), with
    uniform orientation weights. All selections depend on mass/redshift and
    fixed angular parameters, never concentration. The caller should loop
    over mass/orientation blocks instead of retaining a whole large population.
    ``max_pair_count`` limits retained native ordered pairs and fails explicitly
    rather than silently changing the painting or sampling density.
    ``moment_backend='auto'`` replaces excessive pair storage with the
    spherical-harmonic addition theorem, evaluated by the host helper below.
    This changes the numerical algorithm, not the profile or pixel weights.
    """

    mass = np.asarray(mass_msun_h, dtype=float)
    axes = np.asarray(orientation_unit_vectors, dtype=float)
    if not jax.config.jax_enable_x64:
        raise ValueError("enable JAX float64 for high-ell population geometry")
    if moment_backend not in {"pairs", "auto", "harmonic"}:
        raise ValueError("moment_backend must be pairs, auto, or harmonic")
    if mass.ndim != 1 or np.any(mass <= 0) or not np.all(np.isfinite(mass)):
        raise ValueError("population masses must be a positive finite vector")
    if (axes.ndim != 3 or axes.shape[0] != mass.size or axes.shape[1] == 0 or axes.shape[2] != 3
            or not np.all(np.isfinite(axes)) or not np.allclose(np.linalg.norm(axes, axis=-1), 1., atol=1.e-12, rtol=0)):
        raise ValueError("orientations must be unit vectors with shape (n_mass,n_orientation,3)")
    if (not hp.isnsideok(nside, nest=True) or not np.isfinite(chi_mpc_h) or chi_mpc_h <= 0
            or not np.isfinite(redshift) or redshift < 0
            or not np.isfinite(particle_mass_msun_h) or particle_mass_msun_h <= 0):
        raise ValueError("invalid NSIDE, distance, redshift, or particle mass")
    for name, value, minimum in (("angle_nodes", angle_nodes, 2), ("pair_chunk_size", pair_chunk_size, 1),
                                 ("max_pair_count", max_pair_count, 1),
                                 ("max_native_pixels", max_native_pixels, 1)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if isinstance(assignment.n_resolution, bool) or not isinstance(assignment.n_resolution, (int, np.integer)):
        raise ValueError("n_resolution must be an integer")
    # Validate even all-NGP populations, where the production builder is skipped.
    settings = resolve_angular_assignment(nside, assignment)
    radius = np.asarray(halo_radius_delta_comoving(
        jnp.asarray(mass), redshift, cosmology,
        overdensity=nfw_halo_overdensity(redshift, cosmology, profile),
        reference_density=profile.reference_density,
    ))
    theta = 2*np.arcsin(np.minimum(1., radius/(2*chi_mpc_h)))
    resolved = theta >= settings.theta_resolution_rad
    if not np.any(resolved):
        return PaintingPopulationGeometry(
            None, None, None, jnp.empty(0, dtype=int), jnp.empty(0), len(mass),
            particle_mass_msun_h, cosmology, profile,
        )
    empty_map = PinocchioMassMap(
        pixel=np.empty(0, dtype=int), temperature=np.empty(0), source=Path("global_theory_geometry"),
        header={}, nside=nside, ordering="RING", index_scheme="EXPLICIT", first_pixel=None,
        last_pixel=None, aperture_deg=180., selection_type=None, axis_vector=None, filter_name=None,
        filter_considered=None, filter_excluded=None, filter_included=None, filter_excluded_fraction=None,
    )
    stencils, vectors, halo_masses, group_ids = [], [], [], []
    native_indices, native_vectors, reference_vectors, native_counts = [], [], [], []
    pixel_offsets, row_offsets = [0], [0]
    pairs = 0
    max_radius = hp.max_pixrad(nside)
    for group in np.flatnonzero(resolved):
        for axis in axes[group]:
            query_radius = min(np.pi, theta[group]+2*max_radius)
            estimated_rows = hp.nside2npix(nside)*np.sin(query_radius/2)**2
            if moment_backend == "pairs" and estimated_rows > np.sqrt(max_pair_count)*4:
                raise ValueError("global halo footprint exceeds the requested pair geometry budget")
            if pixel_offsets[-1]+estimated_rows > max_native_pixels:
                raise ValueError("native pixel budget exceeded; reduce mass/orientation block size")
            pixels = hp.query_disc(nside, axis, query_radius, inclusive=False, nest=False)
            catalogue = LightconeHaloCatalog(jnp.asarray(axis[None, :]), jnp.array([chi_mpc_h]),
                                             jnp.asarray(mass[group:group+1]), jnp.array([redshift]))
            domain = replace(empty_map, pixel=pixels, temperature=np.zeros(pixels.size))
            stencil = build_adaptive_lightcone_stencil_for_mass_map(
                domain, catalogue, radius[group:group+1], assignment,
                pixel_index=MassMapPixelIndex.from_pixels(pixels, max_dense_bytes=0),
            )
            valid = np.asarray(stencil.sample_valid)
            ngp = np.asarray(stencil.ngp_active)
            if (np.any(valid & ~np.asarray(stencil.sample_in_compact))
                    or np.any(ngp & ~np.asarray(stencil.ngp_in_compact))):
                raise ValueError("the theory domain clipped a global painting sample")
            active = np.unique(np.r_[np.asarray(stencil.sample_compact_row)[valid],
                                      np.asarray(stencil.ngp_compact_row)[ngp]])
            if not active.size or np.any(active < 0):
                raise ValueError("the global painting geometry has no valid native rows")
            pairs += active.size**2
            if moment_backend == "pairs" and pairs > max_pair_count:
                raise ValueError("native pair geometry budget exceeded; use smaller mass/orientation blocks")
            if pixel_offsets[-1]+pixels.size > max_native_pixels:
                raise ValueError("native pixel budget exceeded; reduce mass/orientation block size")
            native_indices.append(pixel_offsets[-1]+active)
            native_vectors.append(np.asarray(hp.pix2vec(nside, pixels[active])).T)
            reference_vectors.append(hp.pix2vec(nside, hp.vec2pix(nside, *axis)))
            native_counts.append(np.full(active.size, mass[group]/particle_mass_msun_h))
            row_offsets.append(row_offsets[-1]+active.size)
            pixel_offsets.append(pixel_offsets[-1]+pixels.size)
            stencils.append(stencil)
            vectors.append(axis)
            halo_masses.append(mass[group])
            group_ids.append(group)
    combined_fields = {}
    for field in fields(AdaptiveLightconeStencil):
        if field.name == "n_pix":
            continue
        pieces = []
        for halo, stencil in enumerate(stencils):
            value = np.asarray(getattr(stencil, field.name))
            if field.name in ("sample_compact_row", "ngp_compact_row"):
                value = np.where(value >= 0, value+pixel_offsets[halo], value)
            elif field.name == "sample_halo_id":
                value = value+halo
            pieces.append(value)
        combined_fields[field.name] = jnp.asarray(np.concatenate(pieces))
    combined = AdaptiveLightconeStencil(**combined_fields, n_pix=pixel_offsets[-1])
    catalogue = LightconeHaloCatalog(jnp.asarray(vectors), jnp.full(len(vectors), chi_mpc_h),
                                     jnp.asarray(halo_masses), jnp.full(len(vectors), redshift))
    largest_angle = min(np.pi, 2*(theta[resolved].max()+2*max_radius))
    harmonic = None
    if moment_backend == "harmonic" or (moment_backend == "auto" and pairs > max_pair_count):
        histogram = None
        harmonic = HarmonicPopulationGeometry(
            np.concatenate(native_vectors), np.asarray(row_offsets), np.asarray(reference_vectors),
            np.asarray(group_ids), 1/axes.shape[1], (~resolved).astype(float),
        )
    else:
        histogram = build_angular_histogram_geometry(
            np.concatenate(native_vectors), np.asarray(row_offsets), np.asarray(reference_vectors),
            np.asarray(group_ids), np.full(len(vectors), 1/axes.shape[1]), (~resolved).astype(float),
            np.linspace(0., largest_angle, angle_nodes), pair_chunk_size=pair_chunk_size,
        )
    return PaintingPopulationGeometry(
        combined, catalogue, histogram, jnp.asarray(np.concatenate(native_indices)),
        jnp.asarray(np.concatenate(native_counts)), len(mass), particle_mass_msun_h, cosmology, profile, harmonic,
    )


def population_assignment_moments(
    concentration: ConcentrationParams,
    geometry: PaintingPopulationGeometry,
    *,
    lmax: int,
    sample_chunk_size: int = 4096,
    pair_chunk_size: int = 65536,
) -> AngularAssignmentMoments:
    """Evaluate A,D through the production JAX painter at fixed geometry.

    Returns dimensionless arrays (n_mass,lmax+1). Profile evaluation, global
    normalization, native child aggregation and angular pair products remain
    differentiable in all concentration parameters. No host conversion occurs
    here. NGP-only populations are analytic with exactly zero derivatives.
    """

    if lmax < 0 or sample_chunk_size < 1 or pair_chunk_size < 1:
        raise ValueError("lmax must be nonnegative and chunk sizes positive")
    if geometry.harmonic is not None:
        raise ValueError("harmonic geometry requires population_harmonic_moments_on_host")
    if geometry.stencil is None:
        unit = jnp.ones((geometry.n_mass, lmax+1), dtype=jnp.float64)
        return AngularAssignmentMoments(unit, unit)
    counts = paint_lightcone_particle_count_map_sparse(
        geometry.stencil, geometry.catalog, particle_mass_msun_h=geometry.particle_mass_msun_h,
        cosmology=geometry.cosmology, concentration_params=concentration,
        profile_params=geometry.profile, sample_chunk_size=sample_chunk_size,
    )
    return histogram_angular_assignment_moments(
        counts[geometry.native_indices]/geometry.native_halo_counts,
        geometry.histogram, lmax=lmax, pair_chunk_size=pair_chunk_size,
    )


def discrete_harmonic_moments(
    vectors: np.ndarray, weights: np.ndarray, reference: np.ndarray, *, lmax: int,
    weight_jacobian: np.ndarray | None = None, nthreads: int = 1, epsilon: float = 1.e-10,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Evaluate exact discrete A,D via the addition theorem, without pair storage.

    Unit vectors and dimensionless globally normalized native weights enter
    the adjoint spherical-harmonic transform, NOT iterative map analysis.
    Accuracy is controlled by DUCC's NUFFT epsilon. Optional weight Jacobians
    have shape (n_pixel,n_parameter); the returned moment Jacobian has shape
    (2,lmax+1,n_parameter). This host function is not a JAX kernel/callback.
    """

    from ducc0.sht import adjoint_synthesis_general

    vectors, weights, reference = map(np.asarray, (vectors, weights, reference))
    if (vectors.shape != (weights.size, 3) or weights.ndim != 1 or reference.shape != (3,)
            or not np.all(np.isfinite(vectors)) or not np.all(np.isfinite(weights))
            or not np.allclose(np.linalg.norm(vectors, axis=1), 1., rtol=0., atol=1.e-12)
            or not np.isclose(np.linalg.norm(reference), 1., rtol=0., atol=1.e-12)):
        raise ValueError("harmonic moments need unit vectors and finite matching native weights")
    if not isinstance(lmax, (int, np.integer)) or lmax < 0 or nthreads < 1 or not 2.e-13 < epsilon < 1.e-3:
        raise ValueError("invalid harmonic transform lmax, threads, or epsilon")
    if weight_jacobian is not None:
        weight_jacobian = np.asarray(weight_jacobian, dtype=float)
        if (weight_jacobian.ndim != 2 or weight_jacobian.shape[0] != weights.size
                or not np.all(np.isfinite(weight_jacobian))):
            raise ValueError("weight Jacobian must have shape (n_pixel,n_parameter) and be finite")
    anchor = np.array([0., 0., 1.]) if abs(reference[2]) < .9 else np.array([1., 0., 0.])
    x = np.cross(anchor, reference)
    x /= np.linalg.norm(x)
    y = np.cross(reference, x)
    px, py, pz = vectors @ x, vectors @ y, vectors @ reference
    locations = np.column_stack((np.arctan2(np.hypot(px, py), pz), np.mod(np.arctan2(py, px), 2*np.pi)))
    def transform(w):
        return adjoint_synthesis_general(
            map=np.asarray(w, dtype=np.float64)[None], loc=locations, spin=0,
            lmax=lmax, epsilon=epsilon, nthreads=nthreads,
        )[0]
    coefficients = transform(weights)
    response_factor = np.sqrt(4*np.pi/(2*np.arange(lmax+1)+1))
    result = np.stack((response_factor*coefficients[:lmax+1].real, 4*np.pi*hp.alm2cl(coefficients)))
    derivative = None
    if weight_jacobian is not None:
        derivative = np.empty((2, lmax+1, weight_jacobian.shape[1]))
        for parameter in range(weight_jacobian.shape[1]):
            tangent = transform(weight_jacobian[:, parameter])
            derivative[:, :, parameter] = np.stack((response_factor*tangent[:lmax+1].real,
                                                    8*np.pi*hp.alm2cl(coefficients, tangent)))
    return result, derivative


def population_harmonic_moments_on_host(
    concentration: ConcentrationParams, geometry: PaintingPopulationGeometry, *,
    lmax: int, derivatives: bool = False, nthreads: int = 1,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Paint with JAX, then evaluate native angular moments with host DUCC.

    The optional three concentration derivatives include the production
    painter's global normalization and are propagated through the linear
    harmonic transform and quadratic self power analytically. No host
    callbacks occur inside a differentiated/JIT-compiled function.
    """

    angular = geometry.harmonic
    if angular is None:
        raise ValueError("population harmonic evaluation needs harmonic geometry")
    def evaluate(p):
        counts = paint_lightcone_particle_count_map_sparse(
            geometry.stencil, geometry.catalog, particle_mass_msun_h=geometry.particle_mass_msun_h,
            cosmology=geometry.cosmology, concentration_params=ConcentrationParams(*p, concentration.mass_pivot),
            profile_params=geometry.profile, sample_chunk_size=4096,
        )
        return counts[geometry.native_indices]/geometry.native_halo_counts
    parameters = jnp.asarray(concentration[:3])
    weights = np.asarray(jax.jit(evaluate)(parameters))
    weight_jacobian = np.asarray(jax.jit(jax.jacfwd(evaluate))(parameters)) if derivatives else None
    result = np.broadcast_to(angular.analytic_ngp[None, :, None], (2, geometry.n_mass, lmax+1)).copy()
    jacobian = np.zeros((*result.shape, 3)) if derivatives else None
    for halo, group in enumerate(angular.groups):
        first, last = angular.offsets[halo:halo+2]
        values, tangent = discrete_harmonic_moments(
            angular.vectors[first:last], weights[first:last], angular.references[halo], lmax=lmax,
            weight_jacobian=None if weight_jacobian is None else weight_jacobian[first:last], nthreads=nthreads,
        )
        result[:, group] += angular.orientation_weight*values
        if derivatives:
            jacobian[:, group] += angular.orientation_weight*tangent
    return result, jacobian
