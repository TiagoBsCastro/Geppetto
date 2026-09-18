#!/usr/bin/env python3
"""Compare a fixed painting-matched candidate with full-sky seed realizations.

This reads existing predictions and measured spectra; it never refits theory.
The primary normalization is the theoretical total shell mean. No mask
coupling or f_sky correction is applied to this full-sky comparison.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aggregate_angular_power_ensemble import (
    THEORY_MODEL_CORRECTED_TWO_HALO,
    FullskyTheory,
    build_ensemble,
    load_aligned_theory,
    load_observation,
    write_shell_mean_csv,
    write_spectrum_csv,
)


def select_observation(observation, shells, multipoles):
    """Select explicit shell and multipole indices without changing normalization."""

    return replace(
        observation, ell=observation.ell[multipoles],
        shell_cl_theoretical_mean=observation.shell_cl_theoretical_mean[shells][:, multipoles],
        shell_cl_measured_mean=observation.shell_cl_measured_mean[shells][:, multipoles],
        theoretical_mean_counts_per_pixel=observation.theoretical_mean_counts_per_pixel[shells],
        measured_mean_counts_per_pixel=observation.measured_mean_counts_per_pixel[shells],
        segment_index=observation.segment_index[shells], z_lo=observation.z_lo[shells], z_hi=observation.z_hi[shells],
    )


def load_predictions(paths, observation, *, ell_stop=2000):
    """Require one complete prediction for every requested shell; reject mixtures."""

    by_segment, reference_ell, reference_settings = {}, None, None
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            ell = np.asarray(data["ell"], dtype=int)
            metadata = json.loads(str(data["metadata_json"]))
            if int(metadata["nside"]) != observation.nside:
                raise ValueError("prediction and observation NSIDE differ")
            settings = (metadata["concentration"], metadata["linear_evolution"],
                        metadata["input_sha256"], metadata.get("orientation_seed"))
            if reference_settings is not None and settings != reference_settings:
                raise ValueError("prediction files mix physical inputs or orientation seeds")
            reference_settings = settings
            if reference_ell is not None and not np.array_equal(ell, reference_ell):
                raise ValueError("prediction multipole grids differ")
            reference_ell = ell
            for index, segment in enumerate(data["segment_indices"]):
                segment = int(segment)
                if segment in by_segment:
                    raise ValueError(f"duplicate prediction for segment {segment}")
                components, correction, total = (np.asarray(data[key][index], dtype=float) for key in
                                                 ("stationary_components", "linear_projection_correction", "shell_total"))
                if (components.shape != (5, ell.size) or correction.shape != ell.shape or total.shape != ell.shape
                        or not np.all(np.isfinite(components)) or not np.all(np.isfinite(correction))
                        or not np.all(np.isfinite(total)) or np.any(total <= 0)
                        or not np.allclose(components[:4].sum(axis=0), components[-1], rtol=1.e-10, atol=1.e-20)
                        or not np.allclose(components[-1]+correction, total, rtol=1.e-10, atol=1.e-20)):
                    raise ValueError("prediction components do not close to a positive finite total")
                by_segment[segment] = (float(data["z_lo"][index]), float(data["z_hi"][index]),
                                       components, correction, total)
    if any(int(segment) not in by_segment for segment in observation.segment_index):
        raise ValueError("predictions are incomplete for the requested shells")
    if reference_ell is None or np.any(np.diff(reference_ell) <= 0):
        raise ValueError("no increasing prediction multipole grid")
    selected = (reference_ell >= 2) & (reference_ell < ell_stop)
    ell = reference_ell[selected]
    positions = np.searchsorted(observation.ell, ell)
    if (not ell.size or np.any(positions >= observation.ell.size)
            or not np.array_equal(observation.ell[positions], ell)):
        raise ValueError("prediction multipoles are absent from observations")
    entries = []
    for segment, lo, hi in zip(observation.segment_index, observation.z_lo, observation.z_hi, strict=True):
        item = by_segment[int(segment)]
        if not np.allclose(item[:2], [lo, hi], rtol=0., atol=1.e-8):
            raise ValueError("prediction redshift bounds differ from observed shell")
        entries.append(item)
    components = np.stack([item[2][:, selected] for item in entries])
    correction = np.stack([item[3][selected] for item in entries])
    total = np.stack([item[4][selected] for item in entries])
    return (FullskyTheory(ell, total, "painting-pair-model", reference_settings[1]),
            components, correction, positions)


def bin_values(values, ell, data):
    """Apply exactly the same mode-count weights as the ensemble plot."""

    return np.stack([np.average(values[..., (ell >= lo) & (ell <= hi)], axis=-1,
                                weights=2*ell[(ell >= lo) & (ell <= hi)]+1)
                     for lo, hi in zip(data.ell_min, data.ell_max, strict=True)], axis=-1)


def write_csv(path, records):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def render(data, baselines, output):
    """All requested shells, including the nearest shell, without a fixed y clip."""

    with plt.rc_context({"font.family": "serif", "font.size": 9, "mathtext.fontset": "stix"}):
        ncolumns = min(4, data.segment_index.size)
        nrows = int(np.ceil(data.segment_index.size/ncolumns))
        figure, axes = plt.subplots(nrows, ncolumns, figsize=(3.*ncolumns, 2.65*nrows), squeeze=False,
                                    constrained_layout=True)
        for index, axis in enumerate(axes.flat):
            if index >= data.segment_index.size:
                axis.set_visible(False)
                continue
            prediction = data.theory_shell[index]
            ratio, error = data.mean_theoretical_mean[index]/prediction, data.sem_theoretical_mean[index]/prediction
            axis.fill_between(data.ell_effective, ratio-error, ratio+error, color="#B04A37", alpha=.25, linewidth=0)
            axis.plot(data.ell_effective, ratio, color="#B04A37", label="Pair candidate")
            for (name, values), color, style in zip(baselines.items(), ("#0072B2", "#009E73"), ("--", "-."), strict=False):
                axis.plot(data.ell_effective, data.mean_theoretical_mean[index]/values[index],
                          color=color, linestyle=style, linewidth=1., label=name)
            axis.axhline(1., color=".35", linewidth=.7)
            axis.set(xscale="log", title=rf"${data.z_lo[index]:.3f}<z<{data.z_hi[index]:.3f}$",
                     xlabel=r"$\ell$", ylabel="Ensemble / theory")
            axis.tick_params(which="both", direction="in", top=True, right=True)
            if index == 0:
                axis.legend(frameon=False, fontsize=7)
        for suffix in ("png", "pdf"):
            figure.savefig(output / f"fullsky_pair_model_all_shells.{suffix}", dpi=180)
        plt.close(figure)


def run(args):
    caches = sorted(args.ensemble_root.glob("seed*/geppetto_reduced/fullsky_observed_spectra.npz"))
    if len(caches) < 2:
        raise ValueError("at least two seed observation caches are required")
    observations = [load_observation(path) for path in caches]
    seeds = [int(path.parents[1].name.removeprefix("seed")) for path in caches]
    reference = observations[0]
    shells = np.arange(reference.segment_index.size)
    if args.segments is not None:
        if len(set(args.segments)) != len(args.segments):
            raise ValueError("requested segments must be unique")
        lookup = {int(value): index for index, value in enumerate(reference.segment_index)}
        if any(segment not in lookup for segment in args.segments):
            raise ValueError("requested segment absent from full-sky observations")
        shells = np.array([lookup[segment] for segment in args.segments])
    observations = [select_observation(item, shells, slice(None)) for item in observations]
    reference = observations[0]
    paths = [Path(path) for path in sorted(glob.glob(args.prediction_glob))]
    theory, components, correction, multipoles = load_predictions(paths, reference, ell_stop=args.ell_stop)
    baselines = {}
    for name, directory, model in (("Scheme 5", args.scheme5, THEORY_MODEL_CORRECTED_TWO_HALO),
                                   ("Scheme 6", args.scheme6, None)):
        if directory is not None:
            baseline = load_aligned_theory(directory, reference, theory_model=model, require_scale_dependent_camb=True)
            baselines[name] = baseline.shell_total[:, multipoles]
    observations = [select_observation(item, np.arange(len(shells)), multipoles) for item in observations]
    data = build_ensemble(observations, theory, seeds, ell_min=args.ell_min, bin_width=args.bin_width)
    baselines = {name: bin_values(value, theory.ell, data) for name, value in baselines.items()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_spectrum_csv(data, args.output_dir / "fullsky_pair_model_bandpowers.csv")
    write_shell_mean_csv(data, args.output_dir / "fullsky_pair_model_shell_means.csv")
    records = []
    for index, segment in enumerate(data.segment_index):
        for lower, upper in ((20, 200), (200, args.ell_stop), (20, args.ell_stop)):
            mask = (data.ell_min >= lower) & (data.ell_max < upper)
            if not np.any(mask):
                continue
            record = dict(segment=int(segment), z_lo=data.z_lo[index], z_hi=data.z_hi[index],
                          ell_lo=lower, ell_stop=upper, n_bins=int(mask.sum()))
            for label, value in dict(pair=data.theory_shell, **baselines).items():
                residual = data.mean_theoretical_mean[index, mask]/value[index, mask]-1
                record[f"{label}_rms_percent"] = 100*np.sqrt(np.mean(residual**2))
                record[f"{label}_mean_bias_percent"] = 100*np.mean(residual)
            heldout = np.asarray(seeds) != 1386
            if np.count_nonzero(heldout) >= 2:
                residual = data.realization_theoretical_mean[heldout, index][:, mask].mean(axis=0)/data.theory_shell[index, mask]-1
                record["excluding_reference_seed_rms_percent"] = 100*np.sqrt(np.mean(residual**2))
            records.append(record)
    write_csv(args.output_dir / "fullsky_pair_model_summary.csv", records)
    measured_components = []
    component_paths = [path.with_name("fullsky_component_spectra.npz") for path in caches]
    if all(path.exists() for path in component_paths):
        for path, observation in zip(component_paths, observations, strict=True):
            with np.load(path, allow_pickle=False) as source:
                if (int(source["nside"]) != observation.nside
                        or not np.array_equal(source["segment_index"][shells], observation.segment_index)
                        or not np.allclose(source["z_lo"][shells], reference.z_lo, rtol=0., atol=1.e-8)
                        or not np.allclose(source["z_hi"][shells], reference.z_hi, rtol=0., atol=1.e-8)
                        or not np.array_equal(source["ell"][multipoles], theory.ell)):
                    raise ValueError("component cache shell/multipole/NSIDE mismatch")
                if not np.allclose(source["theoretical_mean_counts_per_pixel"][shells],
                                   observation.theoretical_mean_counts_per_pixel, rtol=1.e-12, atol=0.):
                    raise ValueError("component cache theoretical normalization mismatch")
                measured = np.stack([source[key][shells][:, multipoles] for key in
                                     ("shell_cl_uncollapsed", "shell_cl_cross", "shell_cl_halo")], axis=1)
                if not np.allclose(measured[:, 0]+2*measured[:, 1]+measured[:, 2],
                                   observation.shell_cl_theoretical_mean, rtol=1.e-8, atol=1.e-20):
                    raise ValueError("measured components do not close to the observed total")
                measured_components.append(measured)
        mean_components = bin_values(np.mean(measured_components, axis=0), theory.ell, data)
        mean_components[:, 1] *= 2
        prediction_components = bin_values(np.stack((components[:, 0], components[:, 1],
                                                     components[:, 2]+components[:, 3]), axis=1), theory.ell, data)
        correction_binned = bin_values(correction, theory.ell, data)
        component_records = []
        for shell, segment in enumerate(data.segment_index):
            for band, ell in enumerate(data.ell_effective):
                record = dict(segment=int(segment), ell=float(ell))
                for index, name in enumerate(("UU", "2UH", "HH")):
                    record[f"{name}_measured_cl"] = mean_components[shell, index, band]
                    record[f"{name}_candidate_cl"] = prediction_components[shell, index, band]
                record["linear_projection_correction_cl"] = correction_binned[shell, band]
                component_records.append(record)
        write_csv(args.output_dir / "fullsky_pair_model_components.csv", component_records)
    np.savez_compressed(args.output_dir / "fullsky_pair_model_comparison.npz", ell=data.ell_effective,
                        segment_indices=data.segment_index, seeds=data.seeds, z_lo=data.z_lo, z_hi=data.z_hi,
                        pair=data.theory_shell, observed=data.mean_theoretical_mean, sem=data.sem_theoretical_mean,
                        realizations=data.realization_theoretical_mean, measured_mean=data.mean_measured_mean,
                        **{name.replace(" ", "_").lower(): value for name, value in baselines.items()})
    report = dict(seeds=seeds, n_shells=len(shells), ell_min=args.ell_min, ell_stop=args.ell_stop,
                  normalization="theoretical total shell mean", mask="none; full sky",
                  uncertainties="sample standard deviation / sqrt(n_seed); no covariance inversion",
                  theory="fixed candidate and original HMF; not refit to ensemble spectra",
                  reference_seed=1386, component_note="stationary components; exact-linear replacement is separate",
                  prediction_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
                  observation_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in caches})
    (args.output_dir / "fullsky_pair_model_audit.json").write_text(json.dumps(report, indent=2)+"\n")
    render(data, baselines, args.output_dir)
    print(json.dumps(report, indent=2), flush=True)
    return data


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-glob", required=True)
    parser.add_argument("--ensemble-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scheme5", type=Path)
    parser.add_argument("--scheme6", type=Path)
    parser.add_argument("--ell-min", type=int, default=20)
    parser.add_argument("--ell-stop", type=int, default=2000)
    parser.add_argument("--bin-width", type=int, default=20)
    parser.add_argument("--segments", type=int, nargs="+")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
