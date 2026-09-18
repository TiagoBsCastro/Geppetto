#!/usr/bin/env python3
"""Schema-6 standard halo-model validation using normalized PINOCCHIO HMF fits.

Reuses measured spectra, not old theory. The cut-sky input is a schema-5
validation directory; full-sky inputs are the observed/component caches of
equal-volume seeds. Fit shapes and bias-weighted abundances are combined
before evaluating the ensemble two-halo power.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

import validate_pinocchio_angular_power as validation
from geppetto.cosmology import Cosmology, halo_radius_delta_comoving
from geppetto.halo_bias import CCTOOLKIT_REVISION
from geppetto.hmf import (
    HMFIntegrationParams,
    NormalizedHMFFitParams,
    average_normalized_hmf,
    fit_normalized_hmf,
    tabulate_normalized_hmf,
)
from geppetto.io import (
    pinocchio_mass_function_series_from_tables,
    read_pinocchio_cosmology_table,
    read_pinocchio_linear_power_evolution,
    read_pinocchio_mass_function,
    read_pinocchio_mass_map_fits,
    read_pinocchio_parameter_file,
)
from geppetto.profiles import nfw_halo_overdensity
from geppetto.theory import (
    LINEAR_HIGH_ELL_FINITE_WIDTH,
    NormalizedHaloPowerTable,
    TwoHaloResponseTable,
    _shell_radial_quadrature,
    comoving_distance_mpc_h,
    exact_halo_model_shell_cls,
    gauss_legendre_rule,
    hybrid_angular_power_spectra,
    normalized_halo_power_grid,
    normalized_one_halo_shell_cls,
    sigma8_from_linear_power,
)

SCHEMA_VERSION = 6
MODEL = "standard_normalized_hmf_castro_bias"
COMPONENTS = (
    "linear",
    "two_halo",
    "one_halo_standard",
    "one_halo_compensated",
    "particle_shot_noise",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--cosmology-table", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--hmf-glob",
        action="append",
        required=True,
        help="One glob per equal-volume realization (repeat for an ensemble)",
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--measurement-dir", type=Path)
    inputs.add_argument("--fullsky-ensemble-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mass-min", type=float, default=1e6)
    parser.add_argument("--mass-max", type=float, default=1e18)
    parser.add_argument("--mass-order", type=int, default=512)
    parser.add_argument("--mass-convergence-rtol", type=float, default=1e-3)
    parser.add_argument("--max-mass-refinements", type=int, default=3)
    parser.add_argument("--max-cutoff-refinements", type=int, default=12)
    parser.add_argument(
        "--match-ngp",
        action="store_true",
        help="Apply the map's angular NGP threshold instead of continuum NFW",
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--ell-exact-cap", type=int, default=512)
    parser.add_argument("--exact-workers", type=int, default=1)
    parser.add_argument("--exact-batch-size", type=int, default=64)
    parser.add_argument("--profile-order", type=int, default=64)
    parser.add_argument("--radial-order", type=int, default=64)
    parser.add_argument("--temporal-order", type=int, default=8)
    parser.add_argument("--ell-bin-width", type=int, default=20)
    parser.add_argument("--limber-match-rtol", type=float, default=0.01)
    parser.add_argument("--limber-match-width", type=int, default=20)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_measurements(args: argparse.Namespace, rows: list[dict[str, str]]) -> dict:
    """Validate spectral cache geometry; do not load or recompute halo maps."""

    zlo = np.array([float(row["z_lo"]) for row in rows])
    zhi = np.array([float(row["z_hi"]) for row in rows])
    if args.measurement_dir is not None:
        path = args.measurement_dir / "angular_power_theory.npz"
        diagnostics = _read_csv(args.measurement_dir / "angular_power_diagnostics.csv")
        for key, target in (("z_lo", zlo), ("z_hi", zhi)):
            if not np.allclose([float(row[key]) for row in diagnostics], target, rtol=0, atol=1e-8):
                raise ValueError("cached measurement shells do not match the manifest")
        with np.load(path, allow_pickle=False) as source:
            if int(source["validation_schema_version"]) not in (5, 6):
                raise ValueError("reuse requires validated schema-5/6 measurements")
            return dict(
                ell=source["ell"].copy(),
                observed_shell=source["observed_shell"].copy(),
                observed_sum=source["observed_sum"].copy(),
                diagnostics=diagnostics,
                mask_hash=str(source["mask_pixel_sha256"].item()),
                mask_iterations=int(source["mask_sht_iterations"]),
                nside=int(diagnostics[0]["nside"]),
                mean_total=np.array(
                    [float(row["mean_total_counts_per_pixel"]) for row in diagnostics]
                ),
                mean_uncollapsed=np.array(
                    [float(row["mean_uncollapsed_counts_per_pixel"]) for row in diagnostics]
                ),
                f_sky=float(diagnostics[0]["f_sky"]),
                provenance={str(path): _sha256(path)},
            )
    paths = sorted(
        args.fullsky_ensemble_root.glob("seed*/geppetto_reduced/fullsky_observed_spectra.npz")
    )
    if len(paths) < 2 or len(paths) != len(args.hmf_glob):
        raise ValueError("provide one HMF glob for each full-sky seed")
    observations, measured, shot_levels, means, uncollapsed, provenance = [], [], [], [], [], {}
    reference_ell, nside = None, None
    for path in paths:
        component_path = path.with_name("fullsky_component_spectra.npz")
        with (
            np.load(path, allow_pickle=False) as source,
            np.load(component_path, allow_pickle=False) as comp,
        ):
            for archive in (source, comp):
                if not np.allclose(archive["z_lo"], zlo, rtol=0, atol=1e-8) or not np.allclose(
                    archive["z_hi"], zhi, rtol=0, atol=1e-8
                ):
                    raise ValueError(f"ensemble cache shell mismatch: {path}")
            if reference_ell is None:
                reference_ell, nside = source["ell"].copy(), int(source["nside"])
            if int(source["nside"]) != nside or not np.array_equal(source["ell"], reference_ell):
                raise ValueError("ensemble cache multipoles/NSIDE differ")
            if not np.array_equal(comp["ell"], reference_ell) or int(comp["nside"]) != nside:
                raise ValueError("component and observed cache geometry differ")
            mean = np.asarray(source["theoretical_mean_counts_per_pixel"])
            unc = np.asarray(comp["measured_mean_uncollapsed_counts_per_pixel"])
            if not np.allclose(comp["theoretical_mean_counts_per_pixel"], mean, rtol=1e-10):
                raise ValueError("component and observed normalization differ")
            observations.append(source["shell_cl_theoretical_mean"].copy())
            measured.append(source["shell_cl_measured_mean"].copy())
            shot_levels.append(4 * np.pi / (12 * nside**2) * unc / mean**2)
            means.append(mean.copy())
            uncollapsed.append(unc.copy())
        provenance[str(path)] = _sha256(path)
        provenance[str(component_path)] = _sha256(component_path)
    return dict(
        ell=reference_ell,
        nside=nside,
        f_sky=1.0,
        mean_total=np.mean(means, axis=0),
        mean_uncollapsed=np.mean(uncollapsed, axis=0),
        observed_shell=np.mean(observations, axis=0),
        observed_realizations=np.stack(observations),
        measured_mean_realizations=np.stack(measured),
        shot_levels=np.mean(shot_levels, axis=0),
        provenance=provenance,
        seeds=[path.parents[1].name for path in paths],
    )


def bandpower_relative_error(
    coarse: np.ndarray, fine: np.ndarray, ell: np.ndarray, bin_width: int = 20
) -> float:
    """Maximum relative difference of positive mode-count bandpowers."""

    errors = []
    for start in range(0, ell.size, bin_width):
        sl = slice(start, start + bin_width)
        weights = 2 * ell[sl] + 1
        old = np.average(coarse[..., sl], axis=-1, weights=weights)
        new = np.average(fine[..., sl], axis=-1, weights=weights)
        errors.append(np.max(np.abs(new - old) / np.maximum(np.abs(new), 1e-100)))
    return float(max(errors))


def build_converged_power(
    args, fits, linear, evolution, zlo, zhi, profiles, concentration, ell, theta_resolution
):
    """Require closure, a bounded low tail, and mass-quadrature convergence."""

    temporal = gauss_legendre_rule(args.temporal_order)
    radial = gauss_legendre_rule(args.radial_order)
    profile_rule = gauss_legendre_rule(args.profile_order)
    shell_z = [
        np.asarray(_shell_radial_quadrature(lo, hi, linear, temporal)[1])[::-1]
        for lo, hi in zip(zlo, zhi, strict=True)
    ]
    all_z = np.unique(np.r_[np.concatenate(shell_z), [f.redshift for f in fits[0]]])
    k = evolution.k_h_mpc if evolution is not None else linear.k_h_mpc
    controls = HMFIntegrationParams(args.mass_min, args.mass_max, args.mass_order)
    history, previous = [], None
    cutoff_z = np.unique(np.r_[[f.redshift for f in fits[0]], (zlo + zhi) / 2])
    previous_bias = None
    cosmology = Cosmology(omega_m=linear.omega_m0, h=linear.h)
    for _cutoff in range(args.max_cutoff_refinements + 1):
        cache, bias_factors = {}, []
        for seed_fits in fits:
            _, diagnostic = tabulate_normalized_hmf(
                seed_fits, cutoff_z, linear, evolution, params=controls, peak_height_cache=cache
            )
            bias_factors.append([row.bias_renormalization for row in diagnostic])
        bias_factors = np.asarray(bias_factors)
        change = (
            None
            if previous_bias is None
            else float(np.max(np.abs(bias_factors / previous_bias - 1)))
        )
        support = max(
            float(
                halo_radius_delta_comoving(
                    jnp.asarray(controls.minimum_mass),
                    jnp.asarray(z),
                    cosmology,
                    overdensity=nfw_halo_overdensity(jnp.asarray(z), cosmology, profile),
                    reference_density=profile.reference_density,
                )
            )
            for z, profile in zip((zlo + zhi) / 2, profiles, strict=True)
        )
        point_error = (float(k[-1]) * support) ** 2 / 3
        history.append(
            dict(
                phase="tail_cutoff",
                **asdict(controls),
                bias_normalization_relative_change=change,
                point_limit_absolute_response_bound=point_error,
            )
        )
        print(
            f"[schema6] tail cutoff Mmin={controls.minimum_mass:g}: "
            f"bias change={change}, point-limit bound={point_error:.3g}",
            flush=True,
        )
        if (
            change is not None
            and change < args.mass_convergence_rtol / 8
            and point_error < args.mass_convergence_rtol / 8
        ):
            break
        previous_bias = bias_factors
        controls = replace(controls, minimum_mass=controls.minimum_mass / 10)
    else:
        raise ValueError(f"analytic HMF tail cutoff did not converge: {history[-1]}")
    for _refinement in range(args.max_mass_refinements + 1):
        print(
            f"[schema6] HMF integration: Mmin={controls.minimum_mass:g}, "
            f"order={controls.mass_order}",
            flush=True,
        )
        ensembles, diagnostics, variance_cache = [], [], {}
        for seed_fits in fits:
            table, closure = tabulate_normalized_hmf(
                seed_fits,
                all_z,
                linear,
                evolution,
                params=controls,
                peak_height_cache=variance_cache,
            )
            ensembles.append(table)
            diagnostics.append([asdict(row) for row in closure])
        halos = average_normalized_hmf(ensembles)
        powers, standard, compensated, response_test = [], [], [], []
        for index, (lo, hi, z, profile) in enumerate(zip(zlo, zhi, shell_z, profiles, strict=True)):
            start = perf_counter()
            arrays = normalized_halo_power_grid(
                k,
                jnp.asarray(z),
                linear,
                halos,
                concentration,
                profile,
                theta_resolution_rad=theta_resolution,
                profile_quadrature=profile_rule,
            )
            arrays.response.block_until_ready()
            power = NormalizedHaloPowerTable(jnp.asarray(1 / (1 + z)), k, *arrays)
            std, comp = normalized_one_halo_shell_cls(
                jnp.asarray(ell), lo, hi, linear, power, radial_quadrature=radial
            )
            powers.append(power)
            standard.append(np.asarray(std))
            compensated.append(np.asarray(comp))
            response_test.append(np.asarray(arrays.response))
            print(
                f"[schema6] profile integrals shell {index + 1}/{len(zlo)} "
                f"in {perf_counter() - start:.1f}s",
                flush=True,
            )
        standard, compensated = np.stack(standard), np.stack(compensated)
        response_test = np.stack(response_test)
        # Check the entire response grid as well as projected one-halo bands.
        # Exact radial cancellations are not bounded by a relative kernel test.
        error = None
        if previous is not None:
            response_error = float(
                np.max(
                    np.abs(response_test**2 - previous[2] ** 2)
                    / np.maximum(response_test**2, 1e-30)
                )
            )
            one_halo_error = max(
                bandpower_relative_error(previous[0], standard, ell),
                bandpower_relative_error(previous[1], compensated, ell),
            )
            error = max(response_error, one_halo_error)
        tail_bound = max(
            row["low_mass_one_halo_upper_bound_mpc_h3"] for seed in diagnostics for row in seed
        )
        smallest_white = min(float(np.min(power.one_halo_standard[:, 0])) for power in powers)
        tail_fraction_bound = tail_bound / smallest_white
        history.append(
            dict(
                **asdict(controls),
                maximum_relative_change=error,
                low_mass_white_power_fraction_bound=tail_fraction_bound,
            )
        )
        if (
            error is not None
            and error < args.mass_convergence_rtol
            and tail_fraction_bound < args.mass_convergence_rtol
        ):
            return powers, standard, compensated, halos, diagnostics, history
        previous = standard, compensated, response_test
        controls = replace(
            controls, minimum_mass=controls.minimum_mass / 10, mass_order=controls.mass_order * 2
        )
    raise ValueError(f"HMF power did not converge: {history[-1]}")


def run(args: argparse.Namespace) -> None:
    """Produce schema-6 theory with measured spectra and explicit provenance."""

    if args.mass_convergence_rtol <= 0 or args.max_mass_refinements < 1:
        raise ValueError("mass convergence needs a positive tolerance and at least one refinement")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = validation.load_manifest(args.manifest)
    concentration = validation.concentration_from_manifest(rows)
    profiles = validation.profiles_from_manifest(rows)
    zlo = np.array([float(row["z_lo"]) for row in rows])
    zhi = np.array([float(row["z_hi"]) for row in rows])
    metadata = read_pinocchio_parameter_file(args.params)
    linear = read_pinocchio_cosmology_table(args.cosmology_table)
    evolution = read_pinocchio_linear_power_evolution(metadata)
    if not np.isclose(metadata.cosmology.omega_m, linear.omega_m0, rtol=1e-6) or not np.isclose(
        metadata.cosmology.h, linear.h, rtol=1e-6
    ):
        raise ValueError("parameter-file and cosmology-table cosmologies differ")
    power_closure = (
        validation.validate_power_evolution_consistency(linear, evolution)
        if evolution is not None
        else None
    )
    sigma8 = float(sigma8_from_linear_power(linear))
    reference_sigma8 = metadata.sigma8_input if metadata.sigma8_input > 0 else sigma8
    sigma8_error = abs(sigma8 / reference_sigma8 - 1)
    if sigma8_error > 0.01:
        raise ValueError("PINOCCHIO power-spectrum sigma8 closure failed")
    inputs = load_measurements(args, rows)
    ell, nside = inputs["ell"], inputs["nside"]
    fits, native = [], []
    provenance = dict(inputs["provenance"])
    for pattern in args.hmf_glob:
        paths = [Path(path) for path in sorted(glob.glob(pattern))]
        tables = tuple(read_pinocchio_mass_function(path) for path in paths)
        native.append(tables)
        fits.append(fit_normalized_hmf(tables, linear, evolution, params=NormalizedHMFFitParams()))
        for path in paths:
            provenance[str(path)] = _sha256(path)
        print(
            f"[schema6] fitted {len(tables)} HMFs: max weighted residual="
            f"{max(f.weighted_log_residual for f in fits[-1]):.4f}",
            flush=True,
        )
    for path in (args.params, args.cosmology_table, args.manifest):
        provenance[str(path)] = _sha256(path)
    theta = validation._consistent_float(rows, "theta_resolution_rad") if args.match_ngp else None
    powers, standard, compensated, halos, closures, convergence = build_converged_power(
        args, fits, linear, evolution, zlo, zhi, profiles, concentration, ell, theta
    )
    audit = dict(
        schema=SCHEMA_VERSION,
        model=MODEL,
        cctoolkit_revision=CCTOOLKIT_REVISION,
        fit_parameters=[[asdict(f) for f in seed] for seed in fits],
        closure=closures,
        convergence=convergence,
        inputs_sha256=provenance,
        profile_convention="angular_ngp" if args.match_ngp else "continuum_nfw",
        bias_note="Castro-corrected PBS bias divided by its full mass-weighted integral",
        low_tail="analytic multiplicity and PBS tails; endpoint Castro correction; u=1",
        mass_definition_note="PINOCCHIO abundance; manifest profile mass definition; "
        "Castro calibration is virial, renormalized here",
        linear_power_evolution="scale_dependent_camb" if evolution is not None else "scalar_growth",
        power_z0_closure=power_closure,
        omega_m0=linear.omega_m0,
        h=linear.h,
        reconstructed_sigma8=sigma8,
        reference_sigma8=reference_sigma8,
        sigma8_relative_error=sigma8_error,
        sigma8_reference_source="parameters"
        if metadata.sigma8_input > 0
        else "cosmology_power_spectrum",
    )
    (args.output_dir / "normalized_hmf_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    np.savez_compressed(
        args.output_dir / "normalized_hmf_quadrature.npz",
        **{key: np.asarray(value) for key, value in halos._asdict().items()},
    )
    validation._write_csv(
        args.output_dir / "angular_power_bias_fit.csv",
        [dict(seed=i, **asdict(f)) for i, seed in enumerate(fits) for f in seed],
    )
    if args.preflight_only:
        return

    import healpy as hp

    pixel_window = np.array(hp.pixwin(nside, lmax=int(ell[-1])), dtype=np.float64)[ell]
    weights = inputs["mean_total"] / np.sum(inputs["mean_total"])
    chi_lo = np.asarray(comoving_distance_mpc_h(jnp.asarray(zlo), linear))
    chi_hi = np.asarray(comoving_distance_mpc_h(jnp.asarray(zhi), linear))
    volume_weights = (chi_hi**3 - chi_lo**3) / np.sum(chi_hi**3 - chi_lo**3)
    responses = tuple(TwoHaloResponseTable(p.scale_factor, p.k_h_mpc, p.response) for p in powers)
    fingerprint = validation.exact_checkpoint_fingerprint(
        zlo,
        zhi,
        weights,
        linear,
        evolution,
        response_tables=responses,
        radial_order=512,
        radial_tail_periods=256,
        relative_tolerance=1e-4,
    )
    fingerprint = hashlib.sha256(
        (MODEL + fingerprint + json.dumps(audit, sort_keys=True)).encode()
    ).hexdigest()

    def exact_batch(batch):
        def compute(missing):
            return exact_halo_model_shell_cls(
                missing,
                zlo,
                zhi,
                linear,
                responses,
                power_evolution=evolution,
                shell_weights=weights,
                radial_order=512,
                radial_tail_periods=256,
                workers=args.exact_workers,
            )

        result = validation.exact_batch_with_checkpoint(
            args.output_dir / "angular_power_exact_checkpoint.npz",
            fingerprint,
            batch,
            len(rows),
            compute,
        )
        print(f"[schema6] exact ell={batch[0]}-{batch[-1]}, cache_hit={result[-1]}", flush=True)
        return result[:4]

    print("[schema6] projecting linear and standard two-halo spectra", flush=True)
    theory = hybrid_angular_power_spectra(
        jnp.asarray(ell),
        zlo,
        zhi,
        linear,
        pinocchio_mass_function_series_from_tables(native[0]),
        concentration,
        profiles,
        shell_weights=jnp.asarray(weights),
        two_halo_response_tables=responses,
        one_halo_shell_cls=jnp.asarray(compensated),
        power_evolution=evolution,
        pixel_window=jnp.asarray(pixel_window),
        mean_uncollapsed_counts_per_pixel=jnp.asarray(inputs["mean_uncollapsed"]),
        mean_total_counts_per_pixel=jnp.asarray(inputs["mean_total"]),
        pixel_area_sr=4 * np.pi / (12 * nside**2),
        ell_exact_cap=args.ell_exact_cap,
        limber_match_rtol=args.limber_match_rtol,
        limber_match_width=args.limber_match_width,
        exact_batch_evaluator=exact_batch,
        exact_batch_size=args.exact_batch_size,
        exact_workers=args.exact_workers,
        radial_order=args.radial_order,
        profile_order=args.profile_order,
    )
    result = {
        key: np.asarray(value)
        for key, value in theory._asdict().items()
        if not key.endswith(("total", "clustering", "one_halo"))
    }
    result.update(
        shell_one_halo_standard=standard * pixel_window**2,
        shell_one_halo_compensated=np.asarray(theory.shell_one_halo),
        summed_one_halo_standard=np.sum(weights[:, None] ** 2 * standard, axis=0) * pixel_window**2,
        summed_one_halo_compensated=np.asarray(theory.summed_one_halo),
        observed_shell=inputs["observed_shell"],
        z_lo=zlo,
        z_hi=zhi,
        validation_schema_version=np.asarray(6),
        two_halo_model=np.asarray(MODEL),
        one_halo_compensation=np.asarray("lagrangian_top_hat_difference"),
        cctoolkit_revision=np.asarray(CCTOOLKIT_REVISION),
        linear_power_evolution=np.asarray(audit["linear_power_evolution"]),
        reconstructed_sigma8=np.asarray(sigma8),
        reference_sigma8=np.asarray(reference_sigma8),
        sigma8_relative_error=np.asarray(sigma8_error),
        model_audit_sha256=np.asarray(_sha256(args.output_dir / "normalized_hmf_audit.json")),
    )
    result["shell_linear_high_ell_mode"] = np.where(
        np.asarray(theory.shell_linear_high_ell_mode) == LINEAR_HIGH_ELL_FINITE_WIDTH,
        "finite_width_flat_sky",
        "limber",
    )
    coupling = None
    if args.measurement_dir is not None:
        first = read_pinocchio_mass_map_fits(
            validation._resolve_input_path(rows[0]["mass_map_path"], args.manifest)
        )
        pixel_hash = hashlib.sha256(
            np.ascontiguousarray(first.pixel, dtype=np.int64).view(np.uint8)
        ).hexdigest()
        if (
            first.nside != nside
            or pixel_hash != inputs["mask_hash"]
            or first.ordering.upper() != "RING"
        ):
            raise ValueError("cached measurement mask/ordering differs from the input map")
        coupling = validation.build_mask_coupling(
            first.pixel,
            nside,
            int(ell[-1]),
            bin_width=args.ell_bin_width,
            n_iter=inputs["mask_iterations"],
        )
        result.update(
            observed_sum=inputs["observed_sum"],
            mask_pixel_sha256=np.asarray(pixel_hash),
            mask_sht_iterations=np.asarray(inputs["mask_iterations"]),
            mask_reference_template_count=np.asarray(coupling.reference_field.n_temp),
        )
        for scope in ("shell", "summed"):
            for component in COMPONENTS:
                name = f"{scope}_{component}"
                result[f"{name}_pseudo_over_fsky"] = validation.couple_theory_component(
                    result[name], ell, coupling
                )
    else:
        result["shell_particle_shot_noise"] = np.broadcast_to(
            inputs["shot_levels"][:, None], standard.shape
        )
        result.update(
            observed_realizations=inputs["observed_realizations"],
            measured_mean_realizations=inputs["measured_mean_realizations"],
            seeds=np.asarray(inputs["seeds"]),
        )
    np.savez_compressed(args.output_dir / "angular_power_theory.npz", **result)
    diagnostics, binned = [], []
    for i, row in enumerate(rows):
        diagnostics.append(
            dict(
                segment_index=int(row["segment_index"]),
                z_lo=zlo[i],
                z_hi=zhi[i],
                f_sky=inputs["f_sky"],
                nside=nside,
                mean_total_counts_per_pixel=inputs["mean_total"][i],
                mean_uncollapsed_counts_per_pixel=inputs["mean_uncollapsed"][i],
                measured_shell_weight=weights[i],
                volume_shell_weight=volume_weights[i],
                normalized_hmf_mass_integral=float(
                    np.interp(
                        1 / (1 + (zlo[i] + zhi[i]) / 2),
                        halos.scale_factor,
                        np.sum(halos.mass_fraction_weight, axis=-1) + halos.tail_mass_fraction,
                    )
                ),
                ell_limber_start=int(theory.ell_limber_start),
                shell_ell_high_ell_start=int(theory.shell_ell_high_ell_start[i]),
                summed_ell_limber_start=int(theory.summed_ell_limber_start),
                high_ell_match_shell_relative_error=float(
                    theory.high_ell_match_shell_relative_error[i]
                ),
                limber_match_summed_relative_error=float(theory.limber_match_summed_relative_error),
                shell_linear_high_ell_mode=(
                    "finite_width_flat_sky"
                    if int(theory.shell_linear_high_ell_mode[i]) == LINEAR_HIGH_ELL_FINITE_WIDTH
                    else "limber"
                ),
                two_halo_model=MODEL,
                one_halo_compensation="lagrangian_top_hat_difference",
                reconstructed_sigma8=sigma8,
                reference_sigma8=reference_sigma8,
                sigma8_relative_error=sigma8_error,
                linear_power_evolution=audit["linear_power_evolution"],
                theory_convention="constant_deprojected_pseudo_cl_over_f_sky"
                if coupling
                else "fullsky",
            )
        )
    if coupling is not None:
        for i, row in enumerate(rows + [{"segment_index": "summed"}]):
            scope = "shell" if i < len(rows) else "summed"
            suffix = "_pseudo_over_fsky"

            def values(name, scope=scope, i=i, suffix=suffix):
                array = result[f"{scope}_{name}{suffix}"]
                return array[i] if scope == "shell" else array

            observed = inputs["observed_shell"][i] if scope == "shell" else inputs["observed_sum"]
            binned.extend(
                validation._binned_rows(
                    f"segment_{row['segment_index']}" if scope == "shell" else "summed",
                    ell,
                    observed,
                    values("linear"),
                    values("two_halo"),
                    values("one_halo_compensated"),
                    values("particle_shot_noise"),
                    ell_min=20,
                    bin_width=args.ell_bin_width,
                    f_sky=inputs["f_sky"],
                    shell_weight=weights[i] if scope == "shell" else 1.0,
                )
            )
        validation._write_csv(args.output_dir / "angular_power_binned.csv", binned)
    validation._write_csv(args.output_dir / "angular_power_diagnostics.csv", diagnostics)
    print(f"[schema6] wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    run(parse_args())
