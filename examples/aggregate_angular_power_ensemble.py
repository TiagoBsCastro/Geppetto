#!/usr/bin/env python3
"""Compare an ensemble of full-sky PINOCCHIO maps with shell theory."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

THEORY_MODEL_LINEAR_BASELINE = "linear-baseline"
THEORY_MODEL_CORRECTED_TWO_HALO = "corrected-two-halo"
THEORY_MODEL_STANDARD = "standard-halo-model"
THEORY_MODEL_STANDARD_WHITE = "standard-halo-model-uncompensated"
THEORY_MODELS = (
    THEORY_MODEL_LINEAR_BASELINE,
    THEORY_MODEL_CORRECTED_TWO_HALO,
    THEORY_MODEL_STANDARD,
    THEORY_MODEL_STANDARD_WHITE,
)
THEORY_MODEL_LABELS = {
    THEORY_MODEL_LINEAR_BASELINE: "linear baseline theory",
    THEORY_MODEL_CORRECTED_TWO_HALO: "corrected two-halo theory",
    THEORY_MODEL_STANDARD: "standard two halo + compensated one halo",
    THEORY_MODEL_STANDARD_WHITE: "standard two halo + standard one halo",
}


@dataclass(frozen=True)
class FullskyObservation:
    """One full-sky realization under both density-normalization conventions."""

    ell: np.ndarray
    shell_cl_theoretical_mean: np.ndarray
    shell_cl_measured_mean: np.ndarray
    theoretical_mean_counts_per_pixel: np.ndarray
    measured_mean_counts_per_pixel: np.ndarray
    segment_index: np.ndarray
    z_lo: np.ndarray
    z_hi: np.ndarray
    nside: int


@dataclass(frozen=True)
class FullskyTheory:
    """Full-sky shell theory aligned to the ensemble shell boundaries."""

    ell: np.ndarray
    shell_total: np.ndarray
    model: str = THEORY_MODEL_LINEAR_BASELINE
    power_evolution: str = "unknown"


@dataclass(frozen=True)
class BinnedFullskyEnsemble:
    """Mode-count-binned spectra and ensemble statistics."""

    seeds: np.ndarray
    ell_min: np.ndarray
    ell_max: np.ndarray
    ell_effective: np.ndarray
    theory_shell: np.ndarray
    realization_theoretical_mean: np.ndarray
    realization_measured_mean: np.ndarray
    mean_theoretical_mean: np.ndarray
    mean_measured_mean: np.ndarray
    std_theoretical_mean: np.ndarray
    sem_theoretical_mean: np.ndarray
    theoretical_mean_counts_per_pixel: np.ndarray
    measured_mean_counts_per_pixel: np.ndarray
    segment_index: np.ndarray
    z_lo: np.ndarray
    z_hi: np.ndarray
    theory_model: str = THEORY_MODEL_LINEAR_BASELINE
    power_evolution: str = "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theory-dir", type=Path, required=True)
    parser.add_argument("--observed-cache", type=Path, action="append", required=True)
    parser.add_argument("--seed", type=int, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ell-min", type=int, default=20)
    parser.add_argument("--bin-width", type=int, default=20)
    parser.add_argument("--png-dpi", type=int, default=300)
    parser.add_argument(
        "--theory-model",
        choices=THEORY_MODELS,
        default=None,
        help=(
            "auto-selects standard-halo-model for schema 6, linear-baseline for older archives; "
            "corrected-two-halo is the legacy schema-5 response"
        ),
    )
    parser.add_argument(
        "--require-scale-dependent-camb",
        action="store_true",
        help="reject theory archives that do not use the tabulated CAMB P(k,z) series",
    )
    return parser.parse_args()


def load_observation(path: Path) -> FullskyObservation:
    """Load one schema-v1 full-sky observed-spectrum cache."""

    required = {
        "schema_version",
        "ell",
        "shell_cl_theoretical_mean",
        "shell_cl_measured_mean",
        "theoretical_mean_counts_per_pixel",
        "measured_mean_counts_per_pixel",
        "segment_index",
        "z_lo",
        "z_hi",
        "nside",
    }
    try:
        with np.load(path, allow_pickle=False) as source:
            missing = required - set(source.files)
            if missing:
                raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
            if int(np.asarray(source["schema_version"])) != 1:
                raise ValueError(f"{path} uses an unsupported cache schema")
            observation = FullskyObservation(
                ell=np.array(source["ell"], dtype=np.int64, copy=True),
                shell_cl_theoretical_mean=np.array(
                    source["shell_cl_theoretical_mean"], dtype=np.float64, copy=True
                ),
                shell_cl_measured_mean=np.array(
                    source["shell_cl_measured_mean"], dtype=np.float64, copy=True
                ),
                theoretical_mean_counts_per_pixel=np.array(
                    source["theoretical_mean_counts_per_pixel"],
                    dtype=np.float64,
                    copy=True,
                ),
                measured_mean_counts_per_pixel=np.array(
                    source["measured_mean_counts_per_pixel"],
                    dtype=np.float64,
                    copy=True,
                ),
                segment_index=np.array(source["segment_index"], dtype=np.int64, copy=True),
                z_lo=np.array(source["z_lo"], dtype=np.float64, copy=True),
                z_hi=np.array(source["z_hi"], dtype=np.float64, copy=True),
                nside=int(np.asarray(source["nside"])),
            )
    except OSError as exc:
        raise ValueError(f"cannot read full-sky observed cache: {path}") from exc
    n_shell = observation.segment_index.size
    expected_shape = (n_shell, observation.ell.size)
    if observation.shell_cl_theoretical_mean.shape != expected_shape or (
        observation.shell_cl_measured_mean.shape != expected_shape
    ):
        raise ValueError(f"{path} contains inconsistent shell spectrum shapes")
    means = (
        observation.theoretical_mean_counts_per_pixel,
        observation.measured_mean_counts_per_pixel,
    )
    if any(values.shape != (n_shell,) for values in means):
        raise ValueError(f"{path} contains inconsistent shell mean shapes")
    arrays = (*means, observation.shell_cl_theoretical_mean, observation.shell_cl_measured_mean)
    if any(not np.all(np.isfinite(values)) for values in arrays):
        raise ValueError(f"{path} contains non-finite values")
    return observation


def _read_diagnostics(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise ValueError(f"cannot read angular-power diagnostics: {path}") from exc
    if not rows:
        raise ValueError(f"angular-power diagnostics are empty: {path}")
    return rows


def load_aligned_theory(
    path: Path,
    observation: FullskyObservation,
    *,
    theory_model: str | None = None,
    require_scale_dependent_camb: bool = False,
) -> FullskyTheory:
    """Load existing full-sky shell theory and align it by redshift bounds."""

    if theory_model is not None and theory_model not in THEORY_MODELS:
        raise ValueError(f"unsupported full-sky theory model: {theory_model}")
    archive_path = path / "angular_power_theory.npz"
    try:
        with np.load(archive_path, allow_pickle=False) as source:
            if "validation_schema_version" not in source.files:
                raise ValueError(f"{archive_path} has no validation schema metadata")
            schema_version = int(np.asarray(source["validation_schema_version"]))
            if theory_model is None:
                theory_model = THEORY_MODEL_STANDARD if schema_version == 6 else THEORY_MODEL_LINEAR_BASELINE
            if schema_version not in {4, 5, 6}:
                raise ValueError(
                    f"{archive_path} uses obsolete validation schema {schema_version}; "
                    "use a schema-4 or schema-5 theory archive"
                )
            if "linear_power_evolution" not in source.files:
                raise ValueError(f"{archive_path} has no linear-power evolution metadata")
            power_evolution = str(np.asarray(source["linear_power_evolution"]).item())
            if power_evolution not in {"scalar_growth", "scale_dependent_camb"}:
                raise ValueError(
                    f"{archive_path} declares unsupported power evolution {power_evolution!r}"
                )
            if require_scale_dependent_camb and power_evolution != "scale_dependent_camb":
                raise ValueError(
                    f"{archive_path} uses {power_evolution!r}; "
                    "scale-dependent CAMB P(k,z) is required"
                )
            ell = np.array(source["ell"], dtype=np.int64, copy=True)
            deterministic_key = "shell_linear"
            one_halo_key = "shell_one_halo"
            if theory_model in (THEORY_MODEL_STANDARD, THEORY_MODEL_STANDARD_WHITE):
                if schema_version != 6 or str(source["two_halo_model"].item()) != "standard_normalized_hmf_castro_bias":
                    raise ValueError("standard-halo-model requires a schema-6 normalized HMF archive")
                deterministic_key = "shell_two_halo"
                one_halo_key = ("shell_one_halo_standard" if theory_model == THEORY_MODEL_STANDARD_WHITE
                                else "shell_one_halo_compensated")
            elif schema_version == 6:
                raise ValueError("schema-6 archives require a standard-halo-model choice")
            if theory_model == THEORY_MODEL_CORRECTED_TWO_HALO:
                if schema_version != 5 or "shell_two_halo" not in source.files:
                    raise ValueError(
                        "corrected-two-halo theory requires a schema-5 archive with "
                        "shell_two_halo"
                    )
                deterministic_key = "shell_two_halo"
            shell_total = sum(
                np.array(source[key], dtype=np.float64, copy=True)
                for key in (
                    deterministic_key,
                    one_halo_key,
                    "shell_particle_shot_noise",
                )
            )
    except (OSError, KeyError) as exc:
        raise ValueError(f"cannot read full-sky shell theory: {archive_path}") from exc
    rows = _read_diagnostics(path / "angular_power_diagnostics.csv")
    theory_z_lo = np.asarray([float(row["z_lo"]) for row in rows])
    theory_z_hi = np.asarray([float(row["z_hi"]) for row in rows])
    indices: list[int] = []
    for z_lo, z_hi in zip(observation.z_lo, observation.z_hi, strict=True):
        matches = np.flatnonzero(
            np.isclose(theory_z_lo, z_lo, rtol=0.0, atol=1.0e-8)
            & np.isclose(theory_z_hi, z_hi, rtol=0.0, atol=1.0e-8)
        )
        if matches.size != 1:
            raise ValueError(f"no unique theory shell matches {z_lo} < z < {z_hi}")
        indices.append(int(matches[0]))
    if not np.array_equal(ell, observation.ell):
        raise ValueError("theory and observed multipoles differ")
    return FullskyTheory(
        ell=ell,
        shell_total=shell_total[np.asarray(indices)],
        model=theory_model,
        power_evolution=power_evolution,
    )


def _validate_observations(observations: list[FullskyObservation]) -> None:
    reference = observations[0]
    for observation in observations[1:]:
        if observation.nside != reference.nside or not np.array_equal(
            observation.ell, reference.ell
        ):
            raise ValueError("ensemble map resolutions or multipoles differ")
        if not np.allclose(observation.z_lo, reference.z_lo) or not np.allclose(
            observation.z_hi, reference.z_hi
        ):
            raise ValueError("ensemble shell boundaries differ")
        if not np.allclose(
            observation.theoretical_mean_counts_per_pixel,
            reference.theoretical_mean_counts_per_pixel,
            rtol=1.0e-12,
            atol=0.0,
        ):
            raise ValueError("ensemble theoretical shell means differ")


def build_ensemble(
    observations: list[FullskyObservation],
    theory: FullskyTheory,
    seeds: list[int],
    *,
    ell_min: int,
    bin_width: int,
) -> BinnedFullskyEnsemble:
    """Mode-count bin and aggregate independent full-sky realizations."""

    if len(observations) < 2 or len(observations) != len(seeds):
        raise ValueError("provide at least two observations and one seed per observation")
    if len(set(seeds)) != len(seeds):
        raise ValueError("ensemble seeds must be unique")
    if ell_min < 0 or bin_width < 1:
        raise ValueError("ell_min must be non-negative and bin_width must be positive")
    _validate_observations(observations)
    reference = observations[0]
    if not np.array_equal(reference.ell, theory.ell):
        raise ValueError("theory and observation multipoles differ")
    lower_edges: list[int] = []
    upper_edges: list[int] = []
    effective: list[float] = []
    theory_bins: list[np.ndarray] = []
    theoretical_bins: list[np.ndarray] = []
    measured_bins: list[np.ndarray] = []
    first = max(ell_min, int(reference.ell[0]))
    for lower in range(first, int(reference.ell[-1]) + 1, bin_width):
        upper = min(lower + bin_width, int(reference.ell[-1]) + 1)
        selected = (reference.ell >= lower) & (reference.ell < upper)
        if not np.any(selected):
            continue
        weights = 2.0 * reference.ell[selected] + 1.0
        lower_edges.append(lower)
        upper_edges.append(upper - 1)
        effective.append(float(np.average(reference.ell[selected], weights=weights)))
        theory_bins.append(np.average(theory.shell_total[:, selected], axis=1, weights=weights))
        theoretical_bins.append(
            np.stack(
                [
                    np.average(
                        observation.shell_cl_theoretical_mean[:, selected],
                        axis=1,
                        weights=weights,
                    )
                    for observation in observations
                ]
            )
        )
        measured_bins.append(
            np.stack(
                [
                    np.average(
                        observation.shell_cl_measured_mean[:, selected],
                        axis=1,
                        weights=weights,
                    )
                    for observation in observations
                ]
            )
        )
    if not lower_edges:
        raise ValueError("no multipoles remain after applying ell_min")
    theory_shell = np.stack(theory_bins, axis=1)
    realization_theoretical = np.stack(theoretical_bins, axis=2)
    realization_measured = np.stack(measured_bins, axis=2)
    std = np.std(realization_theoretical, axis=0, ddof=1)
    return BinnedFullskyEnsemble(
        seeds=np.asarray(seeds, dtype=np.int64),
        ell_min=np.asarray(lower_edges, dtype=np.int64),
        ell_max=np.asarray(upper_edges, dtype=np.int64),
        ell_effective=np.asarray(effective),
        theory_shell=theory_shell,
        realization_theoretical_mean=realization_theoretical,
        realization_measured_mean=realization_measured,
        mean_theoretical_mean=np.mean(realization_theoretical, axis=0),
        mean_measured_mean=np.mean(realization_measured, axis=0),
        std_theoretical_mean=std,
        sem_theoretical_mean=std / np.sqrt(len(observations)),
        theoretical_mean_counts_per_pixel=reference.theoretical_mean_counts_per_pixel,
        measured_mean_counts_per_pixel=np.stack(
            [observation.measured_mean_counts_per_pixel for observation in observations]
        ),
        segment_index=reference.segment_index,
        z_lo=reference.z_lo,
        z_hi=reference.z_hi,
        theory_model=theory.model,
        power_evolution=theory.power_evolution,
    )


def write_spectrum_csv(data: BinnedFullskyEnsemble, path: Path) -> Path:
    """Write ensemble shell spectra and normalization dependence."""

    rows: list[dict[str, object]] = []
    for shell, segment in enumerate(data.segment_index):
        for index, ell in enumerate(data.ell_effective):
            theory = data.theory_shell[shell, index]
            mean = data.mean_theoretical_mean[shell, index]
            measured = data.mean_measured_mean[shell, index]
            sem = data.sem_theoretical_mean[shell, index]
            rows.append(
                {
                    "segment_index": segment,
                    "z_lo": data.z_lo[shell],
                    "z_hi": data.z_hi[shell],
                    "ell_min": data.ell_min[index],
                    "ell_max": data.ell_max[index],
                    "ell_effective": ell,
                    "theory_cl": theory,
                    "ensemble_mean_theoretical_normalization_cl": mean,
                    "ensemble_mean_measured_normalization_cl": measured,
                    "realization_std_cl": data.std_theoretical_mean[shell, index],
                    "ensemble_sem_cl": sem,
                    "theoretical_normalization_mean_over_theory": mean / theory,
                    "measured_normalization_mean_over_theory": measured / theory,
                    "mean_minus_theory_over_sem": (mean - theory) / sem,
                    "theory_model": data.theory_model,
                    "linear_power_evolution": data.power_evolution,
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_shell_mean_csv(data: BinnedFullskyEnsemble, path: Path) -> Path:
    """Write shell-wide density fluctuations for every seed."""

    rows: list[dict[str, object]] = []
    for realization, seed in enumerate(data.seeds):
        for shell, segment in enumerate(data.segment_index):
            measured = data.measured_mean_counts_per_pixel[realization, shell]
            theoretical = data.theoretical_mean_counts_per_pixel[shell]
            rows.append(
                {
                    "seed": seed,
                    "segment_index": segment,
                    "z_lo": data.z_lo[shell],
                    "z_hi": data.z_hi[shell],
                    "measured_mean_counts_per_pixel": measured,
                    "theoretical_mean_counts_per_pixel": theoretical,
                    "shell_delta_mean": measured / theoretical - 1.0,
                    "measured_mean_power_rescaling": (theoretical / measured) ** 2,
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _save_figure(figure: object, base: Path, png_dpi: int) -> list[Path]:
    outputs = [base.with_suffix(".pdf"), base.with_suffix(".png")]
    figure.savefig(outputs[0], bbox_inches="tight")
    figure.savefig(outputs[1], dpi=png_dpi, bbox_inches="tight")
    return outputs


def render_all_shells(
    data: BinnedFullskyEnsemble,
    output_dir: Path,
    *,
    png_dpi: int,
) -> list[Path]:
    """Render ensemble shell ratios under both mean conventions."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional install
        raise ImportError("ensemble plotting requires matplotlib") from exc
    n_columns = 4
    n_rows = int(np.ceil(data.segment_index.size / n_columns))
    figure, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(9.0, 2.2 * n_rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    ell = data.ell_effective
    for shell, axis in enumerate(axes.flat):
        if shell >= data.segment_index.size:
            axis.set_visible(False)
            continue
        theory = data.theory_shell[shell]
        for realization in data.realization_theoretical_mean[:, shell]:
            axis.semilogx(ell, realization / theory, color="0.8", linewidth=0.4)
        primary = data.mean_theoretical_mean[shell] / theory
        sem = data.sem_theoretical_mean[shell] / theory
        measured = data.mean_measured_mean[shell] / theory
        axis.fill_between(
            ell,
            primary - sem,
            primary + sem,
            color="#D55E00",
            alpha=0.25,
            linewidth=0.0,
        )
        axis.semilogx(
            ell,
            primary,
            color="#D55E00",
            linewidth=1.0,
            label="Global-mean normalization",
        )
        axis.semilogx(
            ell,
            measured,
            color="#0072B2",
            linewidth=0.9,
            linestyle="--",
            label="Shell-mean normalization",
        )
        axis.axhline(1.0, color="0.25", linewidth=0.7)
        axis.set_title(
            f"{data.z_lo[shell]:.3f} < z < {data.z_hi[shell]:.3f}",
            fontsize=8,
        )
        axis.tick_params(labelsize=7)
        if shell + n_columns >= data.segment_index.size:
            axis.tick_params(axis="x", labelbottom=True)
    axes[0, 0].legend(frameon=False, fontsize=7, loc="lower left")
    figure.supxlabel(r"Multipole $\ell$", fontsize=10)
    figure.supylabel(
        f"{data.seeds.size}-seed mean / {THEORY_MODEL_LABELS[data.theory_model]}",
        fontsize=10,
    )
    figure.subplots_adjust(left=0.075, right=0.995, bottom=0.07, top=0.96, hspace=0.3)
    outputs = _save_figure(
        figure,
        output_dir / "angular_power_fullsky_ensemble_all_shells",
        png_dpi,
    )
    plt.close(figure)
    return outputs


def render_shell_means(
    data: BinnedFullskyEnsemble,
    output_dir: Path,
    *,
    png_dpi: int,
) -> list[Path]:
    """Render shell mean-density fluctuations and their ensemble mean."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional install
        raise ImportError("ensemble plotting requires matplotlib") from exc
    midpoint = 0.5 * (data.z_lo + data.z_hi)
    delta = (
        data.measured_mean_counts_per_pixel
        / data.theoretical_mean_counts_per_pixel[None, :]
        - 1.0
    )
    mean = np.mean(delta, axis=0)
    sem = np.std(delta, axis=0, ddof=1) / np.sqrt(data.seeds.size)
    figure, axis = plt.subplots(figsize=(6.4, 3.5))
    for realization in delta:
        axis.plot(midpoint, realization, color="0.75", linewidth=0.6)
    axis.fill_between(midpoint, mean - sem, mean + sem, color="#D55E00", alpha=0.25)
    axis.plot(midpoint, mean, color="#D55E00", label=f"{data.seeds.size}-seed mean")
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel("Shell midpoint redshift")
    axis.set_ylabel(r"$\bar{\rho}_{\rm shell}/\bar{\rho}_{\rm theory}-1$")
    axis.legend(frameon=False)
    outputs = _save_figure(
        figure,
        output_dir / "angular_power_fullsky_ensemble_shell_means",
        png_dpi,
    )
    plt.close(figure)
    return outputs


def main() -> None:
    args = parse_args()
    if len(args.observed_cache) != len(args.seed):
        raise ValueError("provide one --seed for every --observed-cache")
    observations = [load_observation(path) for path in args.observed_cache]
    theory = load_aligned_theory(
        args.theory_dir,
        observations[0],
        theory_model=args.theory_model,
        require_scale_dependent_camb=args.require_scale_dependent_camb,
    )
    ensemble = build_ensemble(
        observations,
        theory,
        args.seed,
        ell_min=args.ell_min,
        bin_width=args.bin_width,
    )
    outputs: list[Path] = [
        write_spectrum_csv(
            ensemble,
            args.output_dir / "angular_power_fullsky_ensemble.csv",
        ),
        write_shell_mean_csv(
            ensemble,
            args.output_dir / "angular_power_fullsky_ensemble_shell_means.csv",
        ),
    ]
    outputs.extend(render_all_shells(ensemble, args.output_dir, png_dpi=args.png_dpi))
    outputs.extend(render_shell_means(ensemble, args.output_dir, png_dpi=args.png_dpi))
    for path in outputs:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
