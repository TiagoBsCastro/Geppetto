#!/usr/bin/env python3
"""Check analytical same-halo power against all three production paint branches.

This is a numerical assignment validation, not a cosmological shell prediction.
Distances are chosen to exercise each angular branch at fixed NFW mass/redshift.
The production stencil builder and JAX painter supply the actual pixel weights.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import healpy as hp
import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from geppetto.catalog import AngularAssignmentParams, LightconeHaloCatalog
from geppetto.concentration import ConcentrationParams
from geppetto.cosmology import Cosmology, halo_radius_delta_comoving
from geppetto.io import PinocchioMassMap, build_angular_histogram_geometry
from geppetto.painters import paint_lightcone_particle_count_map_sparse
from geppetto.painting_theory import (
    ConstrainedBackboneAngularModel,
    angular_assignment_moments,
    assemble_constrained_painted_angular_power,
    histogram_angular_assignment_moments,
)
from paint_halo_particles_for_pinocchio_segment import build_adaptive_lightcone_stencil_for_mass_map


def full_sky_domain(nside: int) -> PinocchioMassMap:
    """Small synthetic RING domain; no original PINOCCHIO data are fabricated."""

    pixels = np.arange(hp.nside2npix(nside))
    return PinocchioMassMap(
        pixel=pixels, temperature=np.zeros(pixels.size), source=Path("synthetic"),
        header={}, nside=nside, ordering="RING", index_scheme="EXPLICIT",
        first_pixel=None, last_pixel=None, aperture_deg=180., selection_type=None,
        axis_vector=None, filter_name=None, filter_considered=None, filter_excluded=None,
        filter_included=None, filter_excluded_fraction=None,
    )


def run(output_dir: Path, nside: int = 64, lmax: int = 128) -> dict:
    if not hp.isnsideok(nside, nest=True) or nside > 256:
        raise ValueError("this small full-sky validation requires a power-of-two NSIDE <= 256")
    if not 1 <= lmax <= 3 * nside - 1:
        raise ValueError("require 1 <= lmax <= 3*nside-1")
    output_dir.mkdir(parents=True, exist_ok=True)
    mass_map = full_sky_domain(nside)
    area = hp.nside2pixarea(nside)
    theta_map = np.sqrt(area)
    threshold = 0.5 * hp.max_pixrad(nside)
    axis = np.asarray(hp.ang2vec(1.13, .73))
    host = int(hp.vec2pix(nside, *axis))
    host_vector = np.asarray(hp.pix2vec(nside, host))
    mass, redshift = 1e14, .3
    cosmology, concentration = Cosmology(), ConcentrationParams()
    radius = float(halo_radius_delta_comoving(mass, redshift, cosmology))
    apertures = [0.5 * threshold, 1.5 * theta_map, 5 * theta_map]
    names = ["NGP", "Supersampled", "Native"]
    summary, arrays = {}, {"ell": np.arange(lmax + 1)}
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.8), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1]})
    for index, (name, aperture) in enumerate(zip(names, apertures, strict=True)):
        chi = radius / (2 * np.sin(aperture / 2))
        catalog = LightconeHaloCatalog(
            jnp.asarray(axis[None, :]), jnp.array([chi]), jnp.array([mass]), jnp.array([redshift])
        )
        stencil, diagnostics = build_adaptive_lightcone_stencil_for_mass_map(
            mass_map, catalog, np.array([radius]), AngularAssignmentParams(threshold, 4),
            collect_diagnostics=True,
        )
        painted = np.asarray(paint_lightcone_particle_count_map_sparse(
            stencil, catalog, particle_mass_msun_h=mass, cosmology=cosmology,
            concentration_params=concentration, sample_chunk_size=256,
        ))
        # Select global native rows from fixed geometry, never from profile weights.
        active = np.unique(np.r_[
            np.asarray(stencil.sample_compact_row)[
                np.asarray(stencil.sample_valid) & np.asarray(stencil.sample_in_compact)
            ],
            np.asarray(stencil.ngp_compact_row)[
                np.asarray(stencil.ngp_active) & np.asarray(stencil.ngp_in_compact)
            ],
        ])
        vectors = np.array(hp.pix2vec(nside, active)).T
        moments = angular_assignment_moments(
            vectors, painted[active], host_vector, lmax=lmax, pair_chunk_size=16
        )
        pair_cosine = np.clip(vectors @ vectors.T, -1., 1.).ravel()
        reference_cosine = np.clip(vectors @ host_vector, -1., 1.)
        angle_grid = np.unique(np.r_[0., np.arccos(pair_cosine),
                                    np.arccos(reference_cosine), np.pi])
        geometry = build_angular_histogram_geometry(
            vectors, np.array([0, len(active)]), host_vector[None, :], np.array([0]),
            np.ones(1), np.zeros(1), angle_grid, pair_chunk_size=16,
        )
        histogram_moments = histogram_angular_assignment_moments(
            jnp.asarray(painted[active]), geometry, lmax=lmax, pair_chunk_size=128,
        )
        histogram_error = float(np.max(np.abs(
            np.asarray(histogram_moments)[:, 0] - np.asarray(moments)
        )))
        if histogram_error > 1e-10:
            raise ValueError(f"{name}: histogram moment contraction failed")
        # An algebraic covariance fixture, not a cosmological prediction.
        covariance = ConstrainedBackboneAngularModel(
            jnp.ones(lmax + 1), jnp.array([.2]), jnp.asarray(.3), jnp.array([.4]),
            jnp.ones(lmax + 1),
        )
        components = assemble_constrained_painted_angular_power(
            covariance, histogram_moments.response.T, histogram_moments.self_pair.T,
        )
        component_error = float(np.max(np.abs(
            np.sum(np.asarray(components)[:4], axis=0) - np.asarray(components.total)
        )))
        if component_error > 1e-10 or abs(float(components.total[0]) - 1.) > 1e-10:
            raise ValueError(f"{name}: constrained component covariance failed")

        def complete_prediction(parameters, fixed_inputs=(stencil, catalog, active, geometry, covariance)):
            stencil, catalog, active, geometry, covariance = fixed_inputs
            counts = paint_lightcone_particle_count_map_sparse(
                stencil, catalog, particle_mass_msun_h=mass, cosmology=cosmology,
                concentration_params=ConcentrationParams(*parameters, concentration.mass_pivot),
                sample_chunk_size=256,
            )
            assignment = histogram_angular_assignment_moments(
                counts[active], geometry, lmax=lmax, pair_chunk_size=128,
            )
            return jnp.stack(assemble_constrained_painted_angular_power(
                covariance, assignment.response.T, assignment.self_pair.T,
            ))

        parameters = jnp.asarray(concentration[:3])
        derivative = np.asarray(jax.jit(jax.jacfwd(complete_prediction))(parameters))
        evaluate = jax.jit(complete_prediction)
        finite_difference = np.stack([
            (np.asarray(evaluate(parameters.at[parameter].add(step)))
             - np.asarray(evaluate(parameters.at[parameter].add(-step))))/(2*step)
            for parameter, step in enumerate([1.e-4, 1.e-5, 1.e-5])
        ], axis=-1)
        np.testing.assert_allclose(derivative, finite_difference, rtol=2.e-4, atol=2.e-8)
        np.testing.assert_allclose(derivative[:, 0], 0., atol=2.e-12)
        if name == "NGP":
            np.testing.assert_array_equal(derivative, 0.)
        analytical = np.asarray(moments.self_pair) / (4 * np.pi)
        quadrature_cl = hp.anafast(painted / area, lmax=lmax, iter=0)
        iterative_cl = hp.anafast(painted / area, lmax=lmax, iter=3)
        scale = np.maximum(analytical, 1e-14 * analytical[0])
        quadrature_error = np.max(np.abs(quadrature_cl - analytical) / scale)
        iterative_error = np.max(np.abs(iterative_cl - analytical) / scale)
        if quadrature_error > 1e-8 or abs(painted.sum() - 1) > 1e-12:
            raise ValueError(f"{name}: analytical assignment closure failed")
        summary[name] = dict(
            branch_counts=dict(ngp=diagnostics.n_unresolved_ngp,
                               supersampled=diagnostics.n_supersampled,
                               native=diagnostics.n_native_resolved),
            global_mass_fraction=float(painted.sum()), native_pixels=int(active.size),
            global_samples=int(stencil.size), quadrature_max_relative_error=float(quadrature_error),
            iterative_sht_max_relative_error=float(iterative_error),
            covariance_component_max_absolute_error=component_error,
            histogram_moment_max_absolute_error=histogram_error,
            complete_derivative_max_absolute_error=float(np.max(abs(derivative-finite_difference))),
        )
        arrays[f"{name.lower()}_self_pair"] = np.asarray(moments.self_pair)
        arrays[f"{name.lower()}_response"] = np.asarray(moments.response)
        arrays[f"{name.lower()}_algebraic_component_fixture"] = np.asarray(components)
        arrays[f"{name.lower()}_component_jacobian"] = derivative
        arrays[f"{name.lower()}_component_finite_difference"] = finite_difference
        arrays[f"{name.lower()}_measured_iter0"] = quadrature_cl
        arrays[f"{name.lower()}_measured_iter3"] = iterative_cl
        ell = arrays["ell"][1:]
        top, bottom = axes[:, index]
        top.plot(ell, 4 * np.pi * analytical[1:], color="black", label="Exact same-halo factor")
        top.plot(ell[::4], 4 * np.pi * quadrature_cl[1:][::4], "o", ms=3,
                 mfc="none", color="tab:blue", label="Painted map, iter=0")
        top.plot(ell, np.asarray(moments.response)[1:] ** 2, "--", color="tab:orange",
                 label="Squared mean response")
        top.set_title(name)
        top.set_yscale("log")
        if name == "NGP":
            top.set_ylim(.8, 1.2)
        bottom.plot(ell, quadrature_cl[1:] / scale[1:] - 1, color="tab:blue", label="iter=0")
        bottom.plot(ell, iterative_cl[1:] / scale[1:] - 1, color="tab:green", label="iter=3")
        bottom.axhline(0, color="black", lw=.5)
        bottom.set_xlabel(r"$\ell$")
        print(f"{name}: {summary[name]}", flush=True)
    axes[0, 0].set_ylabel(r"$D_\ell$")
    axes[1, 0].set_ylabel("SHT / exact - 1")
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "painting_operator.png", dpi=200)
    fig.savefig(output_dir / "painting_operator.pdf")
    plt.close(fig)
    np.savez_compressed(output_dir / "painting_operator.npz", **arrays)
    report = dict(nside=nside, lmax=lmax, cases=summary,
                  convention="unmasked delta-pixel quadrature; iterative SHT checked separately")
    (output_dir / "painting_operator.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/painting_operator"))
    parser.add_argument("--nside", type=int, default=64)
    parser.add_argument("--lmax", type=int, default=128)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    run(args.output_dir, args.nside, args.lmax)
