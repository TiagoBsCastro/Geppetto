#!/usr/bin/env python3
"""Audit experimental pair-model projections against schemes 5 and 6.

This does not fit any parameter or select a physical prescription. Inputs must
already contain a projected backbone correction and population-averaged self
power computed from actual adaptive painting. All models are re-coupled through
the same verified mask/constant-deprojection response. The nonlinear correction
is experimental; these plots do not certify the backbone closure.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def candidate_components(
    linear: np.ndarray,
    correction: np.ndarray,
    particle_self: np.ndarray,
    halo_self: np.ndarray,
    pixel_window: np.ndarray,
) -> dict[str, np.ndarray]:
    """Assemble dimensionless full-sky shell C_ell without duplicate self noise.

    ``linear`` already includes its pixel window. ``correction`` is the radial
    projection of P_backbone - P_linear - P_uncollapsed_self, with no window.
    ``halo_self`` comes from native-pixel pair weights, not a continuum profile.
    Only the distinct-particle correction receives an additional window here.
    """

    linear, correction, halo_self = map(np.asarray, (linear, correction, halo_self))
    particle_self, pixel_window = map(np.asarray, (particle_self, pixel_window))
    if (
        linear.ndim != 2 or correction.shape != linear.shape or halo_self.shape != linear.shape
        or particle_self.shape != (linear.shape[0],) or pixel_window.shape != (linear.shape[1],)
    ):
        raise ValueError("shell/multipole component dimensions disagree")
    if not all(np.all(np.isfinite(x)) for x in (
        linear, correction, halo_self, particle_self, pixel_window,
    )):
        raise ValueError("candidate components must be finite")
    terms = dict(
        linear=linear,
        distinct_correction=correction * pixel_window[None, :] ** 2,
        particle_self=np.broadcast_to(particle_self[:, None], linear.shape),
        halo_self=halo_self,
    )
    terms["total"] = sum(terms.values())
    return terms


def fractional_rms(measured: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Unweighted RMS of measured/predicted - 1 across mode-count-binned C_ell.

    This is descriptive, not a covariance-weighted goodness-of-fit statistic.
    """

    if measured.shape != predicted.shape or not np.all(np.isfinite(measured)):
        raise ValueError("measured and predicted arrays must have the same finite shape")
    if not np.all(np.isfinite(predicted)) or np.any(predicted <= 0):
        raise ValueError("predicted bandpowers must be positive and finite")
    return np.sqrt(np.mean((measured / predicted - 1) ** 2, axis=-1))


def covariance_candidate_components(projection: dict) -> dict[str, np.ndarray]:
    """Read the portable covariance output, retaining its linear correction.

    A replacement of the stationary linear projection affects the total only.
    Keep it as a separate additive term: the stationary UU/UH/HH decomposition
    must not be relabelled as an independently exact-projected decomposition.
    """

    components = np.asarray(projection["stationary_components"])
    correction = np.asarray(projection["linear_projection_correction"])
    total = np.asarray(projection["shell_total"])
    if (components.ndim != 3 or components.shape[1] != 5
            or correction.shape != components[:, -1].shape or total.shape != correction.shape):
        raise ValueError("portable covariance component dimensions disagree")
    if not all(np.all(np.isfinite(value)) for value in (components, correction, total)):
        raise ValueError("portable covariance components must be finite")
    terms = {name: components[:, index] for index, name in enumerate((
        "uncollapsed", "particle_halo_cross", "distinct_halo", "halo_self",
    ))}
    if (not np.allclose(sum(terms.values()), components[:, -1], rtol=1.e-10, atol=1.e-20)
            or not np.allclose(components[:, -1]+correction, total, rtol=1.e-10, atol=1.e-20)):
        raise ValueError("portable covariance components do not sum to their total")
    terms.update(linear_projection_correction=correction, total=total)
    return terms


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {key: np.array(source[key]) for key in source.files}


def run(args: argparse.Namespace) -> dict:
    five = _load(args.scheme5 / "angular_power_theory.npz")
    six = _load(args.scheme6 / "angular_power_theory.npz")
    assignment_path = args.assignment or args.projection
    projection, assignment, mask = map(_load, (args.projection, assignment_path, args.mask_response))
    if int(mask.get("reference_template_count", 0)) != 1:
        raise ValueError("mask response must retain the constant-deprojection template")
    for source in (six, projection, assignment, mask):
        if not np.array_equal(source["ell"], five["ell"]):
            raise ValueError("input multipole grids differ")
    for source in (five, six):
        if str(source["mask_pixel_sha256"].item()) != str(mask["mask_pixel_sha256"].item()):
            raise ValueError("baseline and response masks differ")
        if int(source["mask_sht_iterations"]) != int(mask["mask_sht_iterations"]):
            raise ValueError("baseline and response SHT conventions differ")
    indices = np.asarray(projection.get("segment_indices", projection.get("index")), dtype=int)
    if len(indices) != 4 or not np.array_equal(indices, assignment["segment_indices"]):
        raise ValueError("require four matching representative segment indices")
    if "stationary_components" in projection:
        terms = covariance_candidate_components(projection)
    else:
        terms = candidate_components(
            five["shell_linear"][indices], projection["correction"], projection["particle_self"],
            assignment["shell_self_pair"], projection["pixel_window"],
        )
    baseline5 = sum(five[f"shell_{name}"][indices] for name in (
        "two_halo", "one_halo", "particle_shot_noise",
    ))
    baseline6 = sum(six[f"shell_{name}"][indices] for name in (
        "two_halo", "one_halo_compensated", "particle_shot_noise",
    ))
    measured = five["observed_shell"][indices] @ mask["binning"].T
    spectra = dict(
        scheme5=baseline5 @ mask["response"].T,
        scheme6=baseline6 @ mask["response"].T,
        candidate=terms["total"] @ mask["response"].T,
    )
    rms = {name: fractional_rms(measured, prediction) for name, prediction in spectra.items()}
    effective_ell = mask["binning"] @ five["ell"]
    with (args.scheme5 / "angular_power_diagnostics.csv").open(newline="") as stream:
        diagnostics = list(csv.DictReader(stream))
    records = []
    for row, index in enumerate(indices):
        records.append(dict(
            segment=int(index), z_lo=float(diagnostics[index]["z_lo"]),
            z_hi=float(diagnostics[index]["z_hi"]),
            **{f"{name}_rms_percent": float(100 * values[row]) for name, values in rms.items()},
            candidate_better_than_both=bool(rms["candidate"][row] < min(rms["scheme5"][row], rms["scheme6"][row])),
        ))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = dict(
        status="experimental, not an accepted replacement for production theory",
        projection=str(args.projection), assignment=str(assignment_path),
        mask_response=str(args.mask_response),
        metric="unweighted RMS(measured / prediction - 1), mode-count-weighted bands",
        ell_min=int(mask["edges"][0]), ell_stop=int(mask["edges"][-1]), shells=records,
    )
    (args.output_dir / "painting_matched_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.output_dir / "painting_matched_audit.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    np.savez_compressed(
        args.output_dir / "painting_matched_audit.npz", ell=five["ell"], band_ell=effective_ell,
        edges=mask["edges"], segment_indices=indices, measured=measured, **spectra,
        **{f"candidate_fullsky_{name}": values for name, values in terms.items()},
    )
    with plt.rc_context({"font.family": "serif", "font.size": 10, "mathtext.fontset": "stix"}):
        figure = plt.figure(figsize=(10.5, 8.1))
        layout = figure.add_gridspec(2, 2, hspace=.30, wspace=.25)
        styles = {"scheme5": ("#0072B2", "--", "Scheme 5"),
                  "scheme6": ("#009E73", "-.", "Scheme 6"),
                  "candidate": ("#B04A37", "-", "Pair-model candidate")}
        for panel, record in enumerate(records):
            grid = layout[panel // 2, panel % 2].subgridspec(2, 1, height_ratios=[2.2, 1], hspace=.04)
            top = figure.add_subplot(grid[0])
            bottom = figure.add_subplot(grid[1], sharex=top)
            factor = effective_ell * (effective_ell + 1) / (2 * np.pi)
            top.plot(effective_ell, measured[panel] * factor, "o", color="black", mfc="white",
                     ms=3, mew=.6, label="Painted map", zorder=5)
            for name, (color, style, label) in styles.items():
                top.plot(effective_ell, spectra[name][panel] * factor, style, color=color, label=label)
                bottom.plot(effective_ell, measured[panel] / spectra[name][panel], style, color=color)
            top.set(xscale="log", yscale="log", title=rf"${record['z_lo']:.3f}<z<{record['z_hi']:.3f}$")
            top.tick_params(labelbottom=False)
            top.set_ylabel(r"$\ell(\ell+1)C_\ell/(2\pi)$")
            top.text(.04, .92, f"({chr(97 + panel)})", transform=top.transAxes)
            if panel == 0:
                top.legend(fontsize=8, loc="lower right", frameon=False)
            bottom.axhline(1., color=".45", linewidth=.6)
            bottom.set(xlabel=r"$\ell$", ylabel="Map / theory", ylim=(.65, 1.3))
            for axis in (top, bottom):
                axis.set_xlim(mask["edges"][0], mask["edges"][-1])
                axis.tick_params(which="both", direction="in", top=True, right=True)
        figure.savefig(args.output_dir / "painting_matched_representative_shells.png", dpi=200, bbox_inches="tight")
        figure.savefig(args.output_dir / "painting_matched_representative_shells.pdf", bbox_inches="tight")
        plt.close(figure)
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheme5", type=Path, default=Path("examples/angular_power_validation_schema5"))
    parser.add_argument("--scheme6", type=Path, default=Path("examples/angular_power_validation_schema6"))
    parser.add_argument("--projection", type=Path, required=True)
    parser.add_argument("--assignment", type=Path, help="legacy separate assignment NPZ; otherwise use --projection")
    parser.add_argument("--mask-response", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("examples/angular_power_painting_matched_audit"))
    run(parser.parse_args())
