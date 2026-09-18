#!/usr/bin/env python3
"""Predict experimental painting-matched shell power from PINOCCHIO inputs.

This computes the particle-pair backbone, native-count halo weights, and actual
adaptive NFW assignment moments. Only the independent linear projection is
reused from a supplied theory archive; measured spectra in that archive are
never read. No target C_ell is used to fit a transition or amplitude.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np

from geppetto.catalog import AngularAssignmentParams
from geppetto.concentration import ConcentrationParams
from geppetto.cosmology import rho_mean_comoving
from geppetto.halo_bias import CCTOOLKIT_REVISION, fit_pinocchio_numerical_halo_bias
from geppetto.io import (
    pinocchio_mass_function_series_from_tables,
    read_pinocchio_cosmology_table,
    read_pinocchio_halo_count_quadrature,
    read_pinocchio_linear_power_evolution,
    read_pinocchio_lpt_growth_ratios,
    read_pinocchio_mass_function,
    read_pinocchio_parameter_file,
)
from geppetto.lpt_backbone import LPTBackboneParams, build_lpt_particle_backbone
from geppetto.painting_theory import (
    project_constrained_painting_node,
    resolved_painting_population,
    stationary_shell_average,
)
from geppetto.theory import (
    comoving_distance_mpc_h,
    halo_bias_at_redshift,
    halo_count_weights,
    linear_matter_power,
    redshift_at_comoving_distance,
    sigma8_from_linear_power,
)
from painting_theory_population import (
    build_population_geometry,
    population_assignment_moments,
    population_harmonic_moments_on_host,
)
from project_painting_covariance import line_of_sight_rule
from validate_pinocchio_angular_power import (
    concentration_from_manifest,
    load_manifest,
    profiles_from_manifest,
)


def _native(value):
    array = np.asarray(value)
    return array.astype(array.dtype.newbyteorder("="), copy=False)


def load_linear_reference(
    path: Path, rows: list[dict], segments: list[int], *,
    sigma8: float | None = None, evolution: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Read only theoretical linear fields from a previous validation archive.

    Schema-5 validation arrays follow the sorted manifest order. A standalone
    linear archive can instead supply explicit segment_indices. An archive
    without enough indexing information is rejected, never reordered by guess.
    """

    with np.load(path, allow_pickle=False) as source:
        if sigma8 is not None:
            if "reconstructed_sigma8" not in source or not np.isclose(
                float(source["reconstructed_sigma8"]), sigma8, rtol=1.e-6, atol=0.,
            ):
                raise ValueError("linear reference sigma8 disagrees with the cosmology spectrum")
        if evolution is not None:
            if "linear_power_evolution" not in source or str(source["linear_power_evolution"]) != evolution:
                raise ValueError("linear reference uses a different or unspecified growth prescription")
        ell, linear = _native(source["ell"]), _native(source["shell_linear"])
        if "segment_indices" in source:
            indices = _native(source["segment_indices"])
        else:
            indices = np.array([int(row["segment_index"]) for row in rows])
        if (ell.ndim != 1 or not ell.size or np.any(~np.isfinite(ell)) or np.any(ell != np.floor(ell))
                or np.any(ell < 2) or np.any(np.diff(ell) <= 0)):
            raise ValueError("linear reference ell must be increasing integers >= 2")
        if (indices.ndim != 1 or np.any(indices != np.floor(indices)) or len(np.unique(indices)) != len(indices)
                or linear.shape != (len(indices), len(ell))
                or not np.all(np.isfinite(linear)) or np.any(linear < 0)):
            raise ValueError("linear reference shell indexing/values are inconsistent")
        lookup = {int(segment): index for index, segment in enumerate(indices)}
        if any(segment not in lookup for segment in segments):
            raise ValueError("linear reference is missing a requested segment")
        return ell.astype(int), linear[[lookup[segment] for segment in segments]]


def manifest_nside(rows: list[dict], requested: int | None) -> int:
    """Infer NSIDE from the manifest's native angular pixel scale, not chi fields."""

    scales = np.array([float(row["theta_map_rad"]) for row in rows])
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("manifest needs positive finite theta_map_rad values")
    inferred = int(round(np.sqrt(np.pi/3)/scales[0]))
    nside = inferred if requested is None else requested
    if not hp.isnsideok(nside, nest=True) or not np.allclose(
        scales, np.sqrt(hp.nside2pixarea(nside)), rtol=1.e-10, atol=0.,
    ):
        raise ValueError("requested NSIDE disagrees with the painting manifest")
    return nside


def validate_painting_selection(rows: list[dict]) -> None:
    """Restrict this forecast to real-space, catalogue-mass redshift shells."""

    for key, required in (("bounds_mode", "z"), ("redshift_mode", "true"),
                          ("nfw_mass_conversion", "none_catalog_mass_interpreted_as_profile_mass")):
        if any(row.get(key) != required for row in rows):
            raise ValueError(f"painting-matched prediction requires manifest {key}={required}")


def run(args: argparse.Namespace) -> Path:
    if not jax.config.jax_enable_x64:
        raise ValueError("enable JAX float64 for the painting-matched prediction")
    if args.output.suffix != ".npz":
        raise ValueError("output needs a .npz suffix")
    for name in ("radial_order", "orientations", "mass_batch_size", "angle_nodes", "threads"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if args.angle_nodes < 2 or args.seed < 0:
        raise ValueError("angle_nodes must be >= 2 and seed nonnegative")
    started = perf_counter()
    metadata = read_pinocchio_parameter_file(args.params)
    linear = read_pinocchio_cosmology_table(args.cosmology_table)
    evolution = read_pinocchio_linear_power_evolution(metadata)
    if not np.isclose(metadata.cosmology.h, linear.h, rtol=1.e-5) or not np.isclose(
        metadata.cosmology.omega_m, linear.omega_m0, rtol=1.e-5,
    ):
        raise ValueError("parameter file and cosmology table disagree")
    files = [Path(path) for path in sorted(glob.glob(args.hmf_glob))]
    if not files:
        raise ValueError("HMF glob matched no files")
    inputs = [args.manifest, args.params, args.cosmology_table, args.linear_reference, *files]
    if args.output.resolve() in {path.resolve() for path in inputs}:
        raise ValueError("output must not overwrite an input")
    native = tuple(read_pinocchio_mass_function(path) for path in files)
    hmf = pinocchio_mass_function_series_from_tables(native)
    counts = read_pinocchio_halo_count_quadrature(files, box_size_mpc_h=metadata.box_size_mpc_h)
    if not np.array_equal(counts.log_mass_msun_h, hmf.log_mass_msun_h):
        raise ValueError("HMF bias and native-count mass grids disagree")
    all_rows = load_manifest(args.manifest)
    by_index = {int(row["segment_index"]): row for row in all_rows}
    if len(by_index) != len(all_rows) or len(set(args.segments)) != len(args.segments):
        raise ValueError("duplicate manifest/requested segment indices")
    if any(segment not in by_index for segment in args.segments):
        raise ValueError("a requested segment is missing from the manifest")
    rows = [by_index[index] for index in args.segments]
    validate_painting_selection(rows)
    concentration = concentration_from_manifest(rows)
    profiles = profiles_from_manifest(rows)
    nside = manifest_nside(rows, args.nside)
    sigma8 = float(sigma8_from_linear_power(linear))
    growth_source = "scale_dependent_camb" if evolution is not None else "scalar_growth"
    ell, linear_reference = load_linear_reference(
        args.linear_reference, all_rows, args.segments, sigma8=sigma8, evolution=growth_source,
    )
    ell_max = getattr(args, "ell_max", None)
    if ell_max is not None:
        if ell_max < ell[0] or ell_max > ell[-1]:
            raise ValueError("ell_max must lie inside the linear reference grid")
        selected_ell = ell <= ell_max
        ell, linear_reference = ell[selected_ell], linear_reference[:, selected_ell]
    if ell[-1] > 3*nside-1:
        raise ValueError("linear reference exceeds the native map's harmonic range")
    pixel_window = np.asarray(hp.pixwin(nside, lmax=int(ell[-1])), dtype=float)[ell]
    particle_mass = float(metadata.particle_mass_msun_h)
    for row in rows:
        if not np.isclose(float(row["particle_mass_msun_h"]), particle_mass, rtol=1.e-10):
            raise ValueError("manifest and parameters disagree on particle mass")
    print("[painting model] fitting Castro-corrected bias from native HMF tables", flush=True)
    bias, bias_diagnostics = fit_pinocchio_numerical_halo_bias(native, hmf, linear)
    rho = float(rho_mean_comoving(metadata.cosmology))
    all_mass = np.exp(np.asarray(counts.log_mass_msun_h))
    radial_nodes, radial_weights = np.polynomial.legendre.leggauss(args.radial_order)
    los = line_of_sight_rule(args.los_periods, args.los_order)
    average = jax.jit(stationary_shell_average)
    project = jax.jit(project_constrained_painting_node)
    settings = replace(LPTBackboneParams(), threads=args.threads)
    results, node_records = [], []
    powers_k = np.asarray(linear.k_h_mpc if evolution is None else evolution.k_h_mpc)
    parameter_vector = jnp.asarray(concentration[:3])
    harmonic_blocks = 0
    for segment, row, profile in zip(args.segments, rows, profiles, strict=True):
        lo, hi = [float(comoving_distance_mpc_h(float(row[key]), linear)) for key in ("z_lo", "z_hi")]
        if not 0 <= lo < hi:
            raise ValueError("shell redshift bounds do not define a positive radial volume")
        chi_nodes = .5*(hi+lo)+.5*(hi-lo)*radial_nodes
        redshifts = np.asarray(redshift_at_comoving_distance(jnp.asarray(chi_nodes), linear))
        assignment = AngularAssignmentParams(float(row["theta_resolution_rad"]), int(row["n_resolution"]))
        components, stationary_linear = np.zeros((5, len(ell))), np.zeros(len(ell))
        component_jacobian = np.zeros((3, 5, len(ell)))
        for ordinal, (redshift, chi, quadrature_weight) in enumerate(zip(redshifts, chi_nodes, radial_weights, strict=True)):
            scale = 1/(1+redshift)
            if not counts.scale_factor[0] <= scale <= counts.scale_factor[-1]:
                raise ValueError("HMF redshifts do not cover a requested radial node")
            if evolution is not None and not evolution.scale_factor[0] <= scale <= evolution.scale_factor[-1]:
                raise ValueError("CAMB power evolution does not cover a requested radial node")
            number = np.asarray(halo_count_weights(redshift, counts))
            chosen = number > 0
            mass, number = all_mass[chosen], number[chosen]
            growth = read_pinocchio_lpt_growth_ratios(args.cosmology_table, float(redshift))
            print(f"[painting model] segment={segment}, node={ordinal+1}/{args.radial_order}: backbone", flush=True)
            backbone = build_lpt_particle_backbone(
                powers_k, np.asarray(linear_matter_power(powers_k, redshift, linear, evolution)),
                mass, number, mean_density_msun_h_mpch3=rho, particle_mass_msun_h=particle_mass,
                grid_spacing_mpc_h=metadata.box_size_mpc_h/metadata.grid_size, growth=growth, params=settings,
            )
            response, self_pair = np.ones((len(ell), len(mass))), np.ones((len(ell), len(mass)))
            response_jacobian = np.zeros((3, len(ell), len(mass))) if args.derivatives else None
            self_jacobian = np.zeros_like(response_jacobian) if args.derivatives else None
            rng = np.random.default_rng(args.seed+segment)
            axes = rng.normal(size=(len(all_mass), args.orientations, 3))
            axes /= np.linalg.norm(axes, axis=-1, keepdims=True)
            axes = axes[chosen]
            for first in range(0, len(mass), args.mass_batch_size):
                last = min(first+args.mass_batch_size, len(mass))
                geometry = build_population_geometry(
                    mass[first:last], float(redshift), float(chi), axes[first:last], nside=nside,
                    particle_mass_msun_h=particle_mass, cosmology=metadata.cosmology, profile=profile,
                    assignment=assignment, angle_nodes=args.angle_nodes,
                    moment_backend=getattr(args, "moment_backend", "pairs"),
                )
                def evaluate(p, geometry=geometry):
                    return jnp.stack(population_assignment_moments(
                        ConcentrationParams(*p, concentration.mass_pivot), geometry, lmax=int(ell[-1]),
                    ))
                harmonic = getattr(geometry, "harmonic", None) is not None
                if harmonic:
                    harmonic_blocks += 1
                    value, jacobian = population_harmonic_moments_on_host(
                        concentration, geometry, lmax=int(ell[-1]),
                        derivatives=args.derivatives, nthreads=args.threads,
                    )
                else:
                    value = np.asarray(jax.jit(evaluate)(parameter_vector))
                if not np.all(np.isfinite(value)) or not np.allclose(value[:, :, 0], 1., rtol=0, atol=1.e-9):
                    raise ValueError("painting moments are non-finite or violate global mass conservation")
                response[:, first:last], self_pair[:, first:last] = value[0][:, ell].T, value[1][:, ell].T
                if args.derivatives:
                    if not harmonic:
                        jacobian = np.asarray(jax.jit(jax.jacfwd(evaluate))(parameter_vector))
                    if (not np.all(np.isfinite(jacobian))
                            or not np.allclose(jacobian[:, :, 0], 0., rtol=0, atol=1.e-9)):
                        raise ValueError("painting derivatives are non-finite or violate global mass conservation")
                    response_jacobian[:, :, first:last] = jacobian[0][:, ell].transpose(2, 1, 0)
                    self_jacobian[:, :, first:last] = jacobian[1][:, ell].transpose(2, 1, 0)
                del geometry, value
                # Shapes vary with halo footprints. Do not accumulate executables
                # for every mass block throughout a many-shell calculation.
                jax.clear_caches()
                print(f"[painting model] segment={segment}, node={ordinal+1}: masses {last}/{len(mass)}", flush=True)
            if np.any(self_pair < response**2-1.e-7):
                raise ValueError("assignment quadrature violates D >= A^2; refine its angular grid")
            population = resolved_painting_population(
                mass, number, np.asarray(halo_bias_at_redshift(redshift, bias))[chosen], rho, particle_mass,
            )
            particle_self = float(population.uncollapsed_self_power)
            total_self = particle_self+backbone.halo_self_power
            constraint = (backbone.same_protohalo_power+particle_self-backbone.lattice_correction)/total_self
            if np.any(constraint < 0) or np.any(constraint > 1):
                raise ValueError("the pair-derived covariance constraint is outside [0,1]; no clipping")
            projected = []
            continuation = {}
            for name, table in (("coherent", backbone.coherent_power), ("constraint", constraint),
                                ("linear", backbone.linear_power)):
                values = average((ell+.5)/chi, backbone.k_h_mpc, table, hi-lo, los,
                                 low_k_value=table[0], high_k_value=table[-1])
                projected.append(values.value)
                continuation[name] = dict(low=float(jnp.max(jnp.abs(values.low_k_continuation))),
                                          high=float(jnp.max(jnp.abs(values.high_k_continuation))))
            coherent, constraint, linear_average = projected
            radial_weight = .5*(hi-lo)*quadrature_weight*chi**2/((hi**3-lo**3)/3)**2
            fixed = (population, coherent, constraint, radial_weight, jnp.asarray(pixel_window))
            moments = (jnp.asarray(response), jnp.asarray(self_pair))
            node = project(*fixed, *moments)
            components += np.asarray(node)
            stationary_linear += radial_weight*pixel_window**2*np.asarray(linear_average)
            if args.derivatives:
                for parameter in range(3):
                    component_jacobian[parameter] += np.asarray(jax.jvp(
                        lambda a, d, fixed=fixed: project(*fixed, a, d), moments,
                        (jnp.asarray(response_jacobian[parameter]), jnp.asarray(self_jacobian[parameter])),
                    )[1])
            node_records.append(dict(segment=segment, redshift=float(redshift), chi_mpc_h=float(chi),
                                     resolved_mass_fraction=backbone.resolved_mass_fraction,
                                     halo_self_power=backbone.halo_self_power,
                                     same_protohalo_zero_mode=backbone.same_protohalo_zero_mode,
                                     growth_ratios=growth._asdict(), endpoint_continuations=continuation))
        results.append((components, stationary_linear, component_jacobian))
        print(f"[painting model] segment={segment} complete, elapsed={perf_counter()-started:.1f}s", flush=True)
    components = np.stack([item[0] for item in results])
    stationary_linear = np.stack([item[1] for item in results])
    correction = linear_reference-stationary_linear
    total = components[:, -1]+correction
    if np.any(total <= 0) or not np.all(np.isfinite(total)):
        raise ValueError("the projected prediction is nonpositive or non-finite")
    spectrum_arrays = (dict(scale_factor=evolution.scale_factor, k=evolution.k_h_mpc, power=evolution.power_mpc_h3)
                       if evolution is not None else dict(k=linear.k_h_mpc, power=linear_matter_power(
                           linear.k_h_mpc, 0., linear)))
    report = dict(status="experimental statistical closure; not a production default",
                  backbone_params=asdict(settings), nodes=node_records, cctoolkit_revision=CCTOOLKIT_REVISION,
                  bias_fit_diagnostics=[asdict(item) for item in bias_diagnostics],
                  nside=nside, orientations=args.orientations, orientation_seed=args.seed,
                  mass_batch_size=args.mass_batch_size, angle_nodes=args.angle_nodes,
                  moment_backend=getattr(args, "moment_backend", "pairs"), harmonic_blocks=harmonic_blocks,
                  los_periods=args.los_periods, los_order=args.los_order,
                  concentration=concentration._asdict(), profiles=[profile._asdict() for profile in profiles],
                  linear_evolution=growth_source, reconstructed_sigma8=sigma8,
                  linear_array_sha256={key: hashlib.sha256(np.asarray(value, dtype="<f8").tobytes()).hexdigest()
                                       for key, value in spectrum_arrays.items()},
                  radial_geometry="cosmology_table; legacy manifest chi fields are not used",
                  input_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs},
                  elapsed_seconds=perf_counter()-started)
    output = dict(ell=ell, segment_indices=np.asarray(args.segments), pixel_window=pixel_window,
                  z_lo=np.array([float(row["z_lo"]) for row in rows]),
                  z_hi=np.array([float(row["z_hi"]) for row in rows]),
                  stationary_components=components, stationary_linear=stationary_linear,
                  linear_projection_correction=correction, shell_total=total,
                  metadata_json=np.asarray(json.dumps(report, default=str, sort_keys=True)))
    if args.derivatives:
        output["component_jacobian"] = np.stack([item[2] for item in results])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output)
    return args.output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--cosmology-table", type=Path, required=True)
    parser.add_argument("--hmf-glob", required=True)
    parser.add_argument("--linear-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--segments", type=int, nargs="+", required=True)
    parser.add_argument("--nside", type=int)
    parser.add_argument("--ell-max", type=int)
    parser.add_argument("--moment-backend", choices=("pairs", "auto", "harmonic"), default="pairs")
    parser.add_argument("--radial-order", type=int, default=3)
    parser.add_argument("--orientations", type=int, default=16)
    parser.add_argument("--seed", type=int, default=721)
    parser.add_argument("--mass-batch-size", type=int, default=8)
    parser.add_argument("--angle-nodes", type=int, default=8193)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--los-periods", type=int, default=256)
    parser.add_argument("--los-order", type=int, default=16)
    parser.add_argument("--derivatives", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    print(run(parse_args()), flush=True)
