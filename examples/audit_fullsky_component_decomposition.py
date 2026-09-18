#!/usr/bin/env python3
"""Audit the full-sky uncollapsed/halo angular-power decomposition."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ComponentSpectrum:
    """One realization of full-sky component spectra."""

    ell: np.ndarray
    uncollapsed: np.ndarray
    halo: np.ndarray
    cross: np.ndarray
    total: np.ndarray
    theoretical_mean: np.ndarray
    measured_uncollapsed_mean: np.ndarray
    measured_halo_mean: np.ndarray
    segment_index: np.ndarray
    z_lo: np.ndarray
    z_hi: np.ndarray
    nside: int


@dataclass(frozen=True)
class AlignedTheory:
    """Schema-5 theory components aligned with the full-sky shells."""

    ell: np.ndarray
    linear: np.ndarray
    two_halo: np.ndarray
    one_halo: np.ndarray
    particle_shot_noise: np.ndarray
    power_evolution: str
    two_halo_model: str


@dataclass(frozen=True)
class BinnedComponentAudit:
    """Mode-count-binned component spectra and derived diagnostics."""

    seeds: np.ndarray
    ell_min: np.ndarray
    ell_max: np.ndarray
    ell_effective: np.ndarray
    mode_count: np.ndarray
    realization_uncollapsed: np.ndarray
    realization_halo: np.ndarray
    realization_cross: np.ndarray
    realization_total: np.ndarray
    realization_uncollapsed_shot: np.ndarray
    linear: np.ndarray
    two_halo: np.ndarray
    one_halo: np.ndarray
    theory_particle_shot_noise: np.ndarray
    uncollapsed_clustering: np.ndarray
    halo_parallel: np.ndarray
    halo_residual: np.ndarray
    coherent_total: np.ndarray
    orthogonal_total: np.ndarray
    cross_correlation: np.ndarray
    coherent_total_sem: np.ndarray
    cross_correlation_sem: np.ndarray
    theoretical_mean: np.ndarray
    measured_uncollapsed_mean: np.ndarray
    measured_halo_mean: np.ndarray
    segment_index: np.ndarray
    z_lo: np.ndarray
    z_hi: np.ndarray
    power_evolution: str
    two_halo_model: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theory-dir", type=Path, required=True)
    parser.add_argument("--component-cache", type=Path, action="append", required=True)
    parser.add_argument("--seed", type=int, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ell-min", type=int, default=20)
    parser.add_argument("--ell-max", type=int, default=512)
    parser.add_argument("--bin-width", type=int, default=20)
    parser.add_argument("--summary-ell-max", type=int, default=199)
    parser.add_argument("--png-dpi", type=int, default=300)
    return parser.parse_args()


def _scalar_string(value: np.ndarray) -> str:
    return str(np.asarray(value).item())


def load_component_spectrum(path: Path) -> ComponentSpectrum:
    """Load and validate one full-sky component-spectrum cache."""

    required = {
        "ell",
        "shell_cl_uncollapsed",
        "shell_cl_halo",
        "shell_cl_cross",
        "shell_cl_total_closure",
        "theoretical_mean_counts_per_pixel",
        "measured_mean_uncollapsed_counts_per_pixel",
        "measured_mean_halo_counts_per_pixel",
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
            spectrum = ComponentSpectrum(
                ell=np.array(source["ell"], dtype=np.int64, copy=True),
                uncollapsed=np.array(
                    source["shell_cl_uncollapsed"], dtype=np.float64, copy=True
                ),
                halo=np.array(source["shell_cl_halo"], dtype=np.float64, copy=True),
                cross=np.array(source["shell_cl_cross"], dtype=np.float64, copy=True),
                total=np.array(
                    source["shell_cl_total_closure"], dtype=np.float64, copy=True
                ),
                theoretical_mean=np.array(
                    source["theoretical_mean_counts_per_pixel"],
                    dtype=np.float64,
                    copy=True,
                ),
                measured_uncollapsed_mean=np.array(
                    source["measured_mean_uncollapsed_counts_per_pixel"],
                    dtype=np.float64,
                    copy=True,
                ),
                measured_halo_mean=np.array(
                    source["measured_mean_halo_counts_per_pixel"],
                    dtype=np.float64,
                    copy=True,
                ),
                segment_index=np.array(source["segment_index"], dtype=np.int64, copy=True),
                z_lo=np.array(source["z_lo"], dtype=np.float64, copy=True),
                z_hi=np.array(source["z_hi"], dtype=np.float64, copy=True),
                nside=int(np.asarray(source["nside"])),
            )
    except OSError as exc:
        raise ValueError(f"cannot read component cache: {path}") from exc

    n_shell = spectrum.segment_index.size
    expected_shape = (n_shell, spectrum.ell.size)
    spectra = (spectrum.uncollapsed, spectrum.halo, spectrum.cross, spectrum.total)
    means = (
        spectrum.theoretical_mean,
        spectrum.measured_uncollapsed_mean,
        spectrum.measured_halo_mean,
    )
    if spectrum.ell.ndim != 1 or not np.array_equal(
        spectrum.ell, np.arange(spectrum.ell[0], spectrum.ell[-1] + 1)
    ):
        raise ValueError(f"{path} must contain contiguous multipoles")
    if any(values.shape != expected_shape for values in spectra):
        raise ValueError(f"{path} contains inconsistent spectrum shapes")
    if any(values.shape != (n_shell,) for values in means):
        raise ValueError(f"{path} contains inconsistent mean-count shapes")
    if spectrum.nside < 1 or any(not np.all(np.isfinite(values)) for values in (*spectra, *means)):
        raise ValueError(f"{path} contains invalid component values")
    reconstructed = spectrum.uncollapsed + spectrum.halo + 2.0 * spectrum.cross
    scale = np.maximum(np.abs(spectrum.total), np.finfo(np.float64).tiny)
    closure_error = np.max(np.abs(reconstructed - spectrum.total) / scale)
    if closure_error > 1.0e-8:
        raise ValueError(f"{path} fails component closure: {closure_error:.6g}")
    return spectrum


def _read_diagnostics(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise ValueError(f"cannot read angular-power diagnostics: {path}") from exc
    if not rows:
        raise ValueError(f"angular-power diagnostics are empty: {path}")
    return rows


def load_aligned_theory(path: Path, reference: ComponentSpectrum) -> AlignedTheory:
    """Load schema-5 theory and align it to the component shell boundaries."""

    archive = path / "angular_power_theory.npz"
    try:
        with np.load(archive, allow_pickle=False) as source:
            if int(np.asarray(source["validation_schema_version"])) != 5:
                raise ValueError("component audit requires a schema-5 theory archive")
            power_evolution = _scalar_string(source["linear_power_evolution"])
            if power_evolution != "scale_dependent_camb":
                raise ValueError(
                    "component audit requires scale-dependent CAMB power evolution"
                )
            two_halo_model = _scalar_string(source["two_halo_model"])
            ell = np.array(source["ell"], dtype=np.int64, copy=True)
            components = {
                key: np.array(source[key], dtype=np.float64, copy=True)
                for key in (
                    "shell_linear",
                    "shell_two_halo",
                    "shell_one_halo",
                    "shell_particle_shot_noise",
                )
            }
    except (OSError, KeyError) as exc:
        raise ValueError(f"cannot read schema-5 theory archive: {archive}") from exc
    if not np.array_equal(ell, reference.ell):
        raise ValueError("theory and component-cache multipoles differ")

    rows = _read_diagnostics(path / "angular_power_diagnostics.csv")
    theory_z_lo = np.asarray([float(row["z_lo"]) for row in rows])
    theory_z_hi = np.asarray([float(row["z_hi"]) for row in rows])
    indices: list[int] = []
    for z_lo, z_hi in zip(reference.z_lo, reference.z_hi, strict=True):
        matches = np.flatnonzero(
            np.isclose(theory_z_lo, z_lo, rtol=0.0, atol=1.0e-8)
            & np.isclose(theory_z_hi, z_hi, rtol=0.0, atol=1.0e-8)
        )
        if matches.size != 1:
            raise ValueError(f"no unique theory shell matches {z_lo} < z < {z_hi}")
        indices.append(int(matches[0]))
    aligned = np.asarray(indices)
    expected_shape = (len(rows), ell.size)
    if any(values.shape != expected_shape for values in components.values()):
        raise ValueError("theory archive contains inconsistent shell-component shapes")
    return AlignedTheory(
        ell=ell,
        linear=components["shell_linear"][aligned],
        two_halo=components["shell_two_halo"][aligned],
        one_halo=components["shell_one_halo"][aligned],
        particle_shot_noise=components["shell_particle_shot_noise"][aligned],
        power_evolution=power_evolution,
        two_halo_model=two_halo_model,
    )


def _validate_ensemble(spectra: list[ComponentSpectrum]) -> None:
    reference = spectra[0]
    for spectrum in spectra[1:]:
        if spectrum.nside != reference.nside or not np.array_equal(
            spectrum.ell, reference.ell
        ):
            raise ValueError("component caches use different map resolutions")
        integer_arrays = (spectrum.segment_index,)
        reference_integer_arrays = (reference.segment_index,)
        if any(
            not np.array_equal(values, expected)
            for values, expected in zip(
                integer_arrays, reference_integer_arrays, strict=True
            )
        ):
            raise ValueError("component caches use different segment indices")
        if not (
            np.allclose(spectrum.z_lo, reference.z_lo, rtol=0.0, atol=1.0e-10)
            and np.allclose(spectrum.z_hi, reference.z_hi, rtol=0.0, atol=1.0e-10)
            and np.allclose(
                spectrum.theoretical_mean,
                reference.theoretical_mean,
                rtol=1.0e-12,
                atol=0.0,
            )
        ):
            raise ValueError("component caches use different shell definitions")


def _bin_values(values: np.ndarray, selected: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.average(values[..., selected], axis=-1, weights=weights)


def _decompose_component_power(
    uncollapsed: np.ndarray,
    halo: np.ndarray,
    cross: np.ndarray,
    uncollapsed_shot: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split ``HH`` into parts parallel and orthogonal to clustered ``U``.

    The uncollapsed Poisson level is removed before projecting ``H`` onto
    ``U``. Invalid bins, including shot-dominated bins with non-positive
    clustered ``UU``, are returned as NaN.
    """

    uu_clustering = np.asarray(uncollapsed) - np.asarray(uncollapsed_shot)
    hh = np.asarray(halo)
    uh = np.asarray(cross)
    valid = (
        np.isfinite(uu_clustering)
        & np.isfinite(hh)
        & np.isfinite(uh)
        & (uu_clustering > 0.0)
        & (hh > 0.0)
    )
    parallel = np.full(
        np.broadcast_shapes(uu_clustering.shape, hh.shape, uh.shape), np.nan
    )
    correlation = np.full_like(parallel, np.nan)
    np.divide(uh**2, uu_clustering, out=parallel, where=valid)
    correlation_denominator = np.zeros_like(parallel)
    np.sqrt(
        uu_clustering * hh,
        out=correlation_denominator,
        where=valid,
    )
    np.divide(uh, correlation_denominator, out=correlation, where=valid)
    residual = hh - parallel
    tolerance = 128.0 * np.finfo(np.float64).eps * np.maximum(
        np.abs(hh), np.finfo(np.float64).tiny
    )
    valid &= residual >= -tolerance
    parallel = np.where(valid, parallel, np.nan)
    residual = np.where(valid, np.maximum(residual, 0.0), np.nan)
    correlation = np.where(valid, correlation, np.nan)
    coherent = uu_clustering + 2.0 * uh + parallel
    orthogonal = np.asarray(uncollapsed_shot) + residual
    coherent = np.where(valid, coherent, np.nan)
    orthogonal = np.where(valid, orthogonal, np.nan)
    return uu_clustering, parallel, residual, coherent, orthogonal, correlation


def _jackknife_sem(values: np.ndarray) -> np.ndarray:
    valid = np.all(np.isfinite(values), axis=0)
    result = np.full(values.shape[1:], np.nan)
    if np.any(valid):
        center = np.mean(values[:, valid], axis=0)
        result[valid] = np.sqrt(
            (values.shape[0] - 1.0)
            / values.shape[0]
            * np.sum((values[:, valid] - center) ** 2, axis=0)
        )
    return result


def _derived_jackknife(
    uncollapsed: np.ndarray,
    halo: np.ndarray,
    cross: np.ndarray,
    shot: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    coherent: list[np.ndarray] = []
    correlation: list[np.ndarray] = []
    for omitted in range(uncollapsed.shape[0]):
        retained = np.arange(uncollapsed.shape[0]) != omitted
        result = _decompose_component_power(
            np.mean(uncollapsed[retained], axis=0),
            np.mean(halo[retained], axis=0),
            np.mean(cross[retained], axis=0),
            np.mean(shot[retained], axis=0),
        )
        coherent.append(result[3])
        correlation.append(result[5])
    return _jackknife_sem(np.stack(coherent)), _jackknife_sem(np.stack(correlation))


def build_component_audit(
    spectra: list[ComponentSpectrum],
    theory: AlignedTheory,
    seeds: list[int],
    *,
    ell_min: int,
    ell_max: int,
    bin_width: int,
) -> BinnedComponentAudit:
    """Bin an ensemble and construct a shot-corrected component audit."""

    if len(spectra) < 2 or len(spectra) != len(seeds):
        raise ValueError("provide at least two caches and one seed per cache")
    if len(set(seeds)) != len(seeds):
        raise ValueError("component-cache seeds must be unique")
    if ell_min < 2 or ell_max < ell_min or bin_width < 1:
        raise ValueError("invalid multipole range or bin width")
    _validate_ensemble(spectra)
    reference = spectra[0]
    if not np.array_equal(reference.ell, theory.ell):
        raise ValueError("theory and component-cache multipoles differ")
    ell_max = min(ell_max, int(reference.ell[-1]))

    raw = {
        "uncollapsed": np.stack([spectrum.uncollapsed for spectrum in spectra]),
        "halo": np.stack([spectrum.halo for spectrum in spectra]),
        "cross": np.stack([spectrum.cross for spectrum in spectra]),
        "total": np.stack([spectrum.total for spectrum in spectra]),
    }
    lower_edges: list[int] = []
    upper_edges: list[int] = []
    effective: list[float] = []
    mode_count: list[float] = []
    binned_raw = {key: [] for key in raw}
    binned_theory: dict[str, list[np.ndarray]] = {
        "linear": [],
        "two_halo": [],
        "one_halo": [],
        "particle_shot_noise": [],
    }
    for lower in range(ell_min, ell_max + 1, bin_width):
        upper = min(lower + bin_width, ell_max + 1)
        selected = (reference.ell >= lower) & (reference.ell < upper)
        if not np.any(selected):
            continue
        weights = 2.0 * reference.ell[selected] + 1.0
        lower_edges.append(lower)
        upper_edges.append(upper - 1)
        effective.append(float(np.average(reference.ell[selected], weights=weights)))
        mode_count.append(float(np.sum(weights)))
        for key, values in raw.items():
            binned_raw[key].append(_bin_values(values, selected, weights))
        for key in binned_theory:
            binned_theory[key].append(
                _bin_values(getattr(theory, key), selected, weights)
            )
    if not lower_edges:
        raise ValueError("no multipoles remain in the requested audit range")
    realization = {key: np.stack(values, axis=-1) for key, values in binned_raw.items()}
    theory_bins = {
        key: np.stack(values, axis=-1) for key, values in binned_theory.items()
    }

    pixel_area = 4.0 * np.pi / (12 * reference.nside**2)
    shot = np.stack(
        [
            pixel_area
            * spectrum.measured_uncollapsed_mean
            / spectrum.theoretical_mean**2
            for spectrum in spectra
        ]
    )[..., None]
    mean_uncollapsed = np.mean(realization["uncollapsed"], axis=0)
    mean_halo = np.mean(realization["halo"], axis=0)
    mean_cross = np.mean(realization["cross"], axis=0)
    mean_shot = np.mean(shot, axis=0)
    decomposition = _decompose_component_power(
        mean_uncollapsed,
        mean_halo,
        mean_cross,
        mean_shot,
    )
    coherent_sem, correlation_sem = _derived_jackknife(
        realization["uncollapsed"],
        realization["halo"],
        realization["cross"],
        shot,
    )
    reconstructed = decomposition[3] + decomposition[4]
    total_mean = np.mean(realization["total"], axis=0)
    valid = np.isfinite(reconstructed)
    if not np.allclose(reconstructed[valid], total_mean[valid], rtol=1.0e-11, atol=0.0):
        raise ValueError("derived coherent/orthogonal decomposition does not close")

    return BinnedComponentAudit(
        seeds=np.asarray(seeds, dtype=np.int64),
        ell_min=np.asarray(lower_edges, dtype=np.int64),
        ell_max=np.asarray(upper_edges, dtype=np.int64),
        ell_effective=np.asarray(effective),
        mode_count=np.asarray(mode_count),
        realization_uncollapsed=realization["uncollapsed"],
        realization_halo=realization["halo"],
        realization_cross=realization["cross"],
        realization_total=realization["total"],
        realization_uncollapsed_shot=np.broadcast_to(
            shot, realization["uncollapsed"].shape
        ),
        linear=theory_bins["linear"],
        two_halo=theory_bins["two_halo"],
        one_halo=theory_bins["one_halo"],
        theory_particle_shot_noise=theory_bins["particle_shot_noise"],
        uncollapsed_clustering=decomposition[0],
        halo_parallel=decomposition[1],
        halo_residual=decomposition[2],
        coherent_total=decomposition[3],
        orthogonal_total=decomposition[4],
        cross_correlation=decomposition[5],
        coherent_total_sem=coherent_sem,
        cross_correlation_sem=correlation_sem,
        theoretical_mean=reference.theoretical_mean,
        measured_uncollapsed_mean=np.stack(
            [spectrum.measured_uncollapsed_mean for spectrum in spectra]
        ),
        measured_halo_mean=np.stack(
            [spectrum.measured_halo_mean for spectrum in spectra]
        ),
        segment_index=reference.segment_index,
        z_lo=reference.z_lo,
        z_hi=reference.z_hi,
        power_evolution=theory.power_evolution,
        two_halo_model=theory.two_halo_model,
    )


def _safe_ratio(numerator: np.ndarray | float, denominator: np.ndarray | float) -> np.ndarray:
    numerator_array = np.asarray(numerator, dtype=np.float64)
    denominator_array = np.asarray(denominator, dtype=np.float64)
    return np.divide(
        numerator_array,
        denominator_array,
        out=np.full(np.broadcast_shapes(numerator_array.shape, denominator_array.shape), np.nan),
        where=np.isfinite(denominator_array) & (denominator_array != 0.0),
    )


def write_detailed_csv(data: BinnedComponentAudit, path: Path) -> Path:
    """Write every binned measured and theoretical decomposition term."""

    uu = np.mean(data.realization_uncollapsed, axis=0)
    hh = np.mean(data.realization_halo, axis=0)
    uh = np.mean(data.realization_cross, axis=0)
    total = np.mean(data.realization_total, axis=0)
    shot = np.mean(data.realization_uncollapsed_shot, axis=0)
    rows: list[dict[str, object]] = []
    for shell, segment in enumerate(data.segment_index):
        for index, ell in enumerate(data.ell_effective):
            linear = data.linear[shell, index]
            baseline_total = (
                linear
                + data.one_halo[shell, index]
                + data.theory_particle_shot_noise[shell, index]
            )
            corrected_total = (
                data.two_halo[shell, index]
                + data.one_halo[shell, index]
                + data.theory_particle_shot_noise[shell, index]
            )
            rows.append(
                {
                    "n_realizations": data.seeds.size,
                    "seeds": ";".join(str(seed) for seed in data.seeds),
                    "segment_index": segment,
                    "z_lo": data.z_lo[shell],
                    "z_hi": data.z_hi[shell],
                    "ell_min": data.ell_min[index],
                    "ell_max": data.ell_max[index],
                    "ell_effective": ell,
                    "measured_uu": uu[shell, index],
                    "measured_uu_particle_shot": shot[shell, index],
                    "measured_uu_clustering": data.uncollapsed_clustering[shell, index],
                    "measured_2uh": 2.0 * uh[shell, index],
                    "measured_hh": hh[shell, index],
                    "measured_hh_parallel": data.halo_parallel[shell, index],
                    "measured_hh_residual": data.halo_residual[shell, index],
                    "measured_coherent_total": data.coherent_total[shell, index],
                    "measured_orthogonal_total": data.orthogonal_total[shell, index],
                    "measured_total": total[shell, index],
                    "cross_correlation": data.cross_correlation[shell, index],
                    "cross_correlation_jackknife_sem": data.cross_correlation_sem[
                        shell, index
                    ],
                    "linear_theory": linear,
                    "corrected_two_halo_theory": data.two_halo[shell, index],
                    "one_halo_theory": data.one_halo[shell, index],
                    "particle_shot_theory": data.theory_particle_shot_noise[
                        shell, index
                    ],
                    "coherent_over_linear": _safe_ratio(
                        data.coherent_total[shell, index], linear
                    ).item(),
                    "coherent_over_linear_jackknife_sem": _safe_ratio(
                        data.coherent_total_sem[shell, index], linear
                    ).item(),
                    "corrected_two_halo_over_linear": _safe_ratio(
                        data.two_halo[shell, index], linear
                    ).item(),
                    "measured_total_over_baseline_total": _safe_ratio(
                        total[shell, index], baseline_total
                    ).item(),
                    "measured_total_over_corrected_total": _safe_ratio(
                        total[shell, index], corrected_total
                    ).item(),
                    "decomposition_valid": bool(
                        np.isfinite(data.coherent_total[shell, index])
                    ),
                    "linear_power_evolution": data.power_evolution,
                    "two_halo_model": data.two_halo_model,
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def summarize_band(
    data: BinnedComponentAudit,
    *,
    ell_min: int,
    ell_max: int,
) -> list[dict[str, object]]:
    """Summarize one contiguous two-halo band for every redshift shell."""

    selected = (data.ell_min >= ell_min) & (data.ell_max <= ell_max)
    if not np.any(selected):
        raise ValueError("no complete bins remain in the requested summary band")
    weights = data.mode_count[selected]

    def average(values: np.ndarray) -> np.ndarray:
        return np.average(values[..., selected], axis=-1, weights=weights)

    uu = average(data.realization_uncollapsed)
    hh = average(data.realization_halo)
    uh = average(data.realization_cross)
    total = average(data.realization_total)
    shot = average(data.realization_uncollapsed_shot)
    decomposition = _decompose_component_power(
        np.mean(uu, axis=0),
        np.mean(hh, axis=0),
        np.mean(uh, axis=0),
        np.mean(shot, axis=0),
    )
    coherent_sem, correlation_sem = _derived_jackknife(uu, hh, uh, shot)
    linear = average(data.linear)
    corrected = average(data.two_halo)
    one_halo = average(data.one_halo)
    theory_shot = average(data.theory_particle_shot_noise)
    measured_total = np.mean(total, axis=0)
    mean_uncollapsed = np.mean(data.measured_uncollapsed_mean, axis=0)
    mean_halo = np.mean(data.measured_halo_mean, axis=0)
    mean_total = mean_uncollapsed + mean_halo
    rows: list[dict[str, object]] = []
    for shell, segment in enumerate(data.segment_index):
        coherent_ratio = _safe_ratio(decomposition[3][shell], linear[shell]).item()
        coherent_sem_ratio = _safe_ratio(coherent_sem[shell], linear[shell]).item()
        corrected_ratio = _safe_ratio(corrected[shell], linear[shell]).item()
        rows.append(
            {
                "n_realizations": data.seeds.size,
                "seeds": ";".join(str(seed) for seed in data.seeds),
                "segment_index": segment,
                "z_lo": data.z_lo[shell],
                "z_hi": data.z_hi[shell],
                "ell_min": int(data.ell_min[selected][0]),
                "ell_max": int(data.ell_max[selected][-1]),
                "measured_uu_clustering_over_linear": _safe_ratio(
                    decomposition[0][shell], linear[shell]
                ).item(),
                "measured_2uh_over_linear": _safe_ratio(
                    2.0 * np.mean(uh, axis=0)[shell], linear[shell]
                ).item(),
                "measured_hh_parallel_over_linear": _safe_ratio(
                    decomposition[1][shell], linear[shell]
                ).item(),
                "measured_coherent_over_linear": coherent_ratio,
                "measured_coherent_over_linear_jackknife_sem": coherent_sem_ratio,
                "corrected_two_halo_over_linear": corrected_ratio,
                "corrected_excess_over_measured_coherent": _safe_ratio(
                    corrected[shell], decomposition[3][shell]
                ).item()
                - 1.0,
                "corrected_minus_coherent_jackknife_sigma": _safe_ratio(
                    corrected[shell] - decomposition[3][shell], coherent_sem[shell]
                ).item(),
                "cross_correlation": decomposition[5][shell],
                "cross_correlation_jackknife_sem": correlation_sem[shell],
                "measured_total_over_baseline_total": _safe_ratio(
                    measured_total[shell],
                    linear[shell] + one_halo[shell] + theory_shot[shell],
                ).item(),
                "measured_total_over_corrected_total": _safe_ratio(
                    measured_total[shell],
                    corrected[shell] + one_halo[shell] + theory_shot[shell],
                ).item(),
                "measured_total_mean_over_theoretical_mean": _safe_ratio(
                    mean_total[shell], data.theoretical_mean[shell]
                ).item(),
                "realized_uncollapsed_shot_over_theory": _safe_ratio(
                    np.mean(shot, axis=0)[shell], theory_shot[shell]
                ).item(),
                "measured_halo_mass_fraction": _safe_ratio(
                    mean_halo[shell], mean_total[shell]
                ).item(),
                "linear_power_evolution": data.power_evolution,
                "two_halo_model": data.two_halo_model,
            }
        )
    return rows


def write_summary_csv(rows: list[dict[str, object]], path: Path) -> Path:
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


def _format_log_ell_axis(axis: object, ell_max: float) -> None:
    from matplotlib.ticker import NullFormatter, ScalarFormatter

    ticks = [value for value in (20, 50, 100, 200, 500, 1000) if value <= ell_max]
    axis.set_xlim(20, ell_max)
    axis.set_xticks(ticks)
    axis.xaxis.set_major_formatter(ScalarFormatter())
    axis.xaxis.set_minor_formatter(NullFormatter())


def render_all_shells(
    data: BinnedComponentAudit,
    output_dir: Path,
    *,
    png_dpi: int,
) -> list[Path]:
    """Plot measured and modeled deterministic power for every shell."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional install
        raise ImportError("component-audit plotting requires matplotlib") from exc
    figure, axes = plt.subplots(
        int(np.ceil(data.segment_index.size / 4)),
        4,
        figsize=(9.2, 8.0),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    ell = data.ell_effective
    total = np.mean(data.realization_total, axis=0)
    baseline_total = data.linear + data.one_halo + data.theory_particle_shot_noise
    for shell, axis in enumerate(axes.flat):
        if shell >= data.segment_index.size:
            axis.set_visible(False)
            continue
        empirical = _safe_ratio(data.coherent_total[shell], data.linear[shell])
        empirical_sem = _safe_ratio(data.coherent_total_sem[shell], data.linear[shell])
        corrected = _safe_ratio(data.two_halo[shell], data.linear[shell])
        total_ratio = _safe_ratio(total[shell], baseline_total[shell])
        valid = np.isfinite(empirical) & np.isfinite(empirical_sem)
        axis.fill_between(
            ell[valid],
            empirical[valid] - empirical_sem[valid],
            empirical[valid] + empirical_sem[valid],
            color="#D55E00",
            alpha=0.22,
            linewidth=0.0,
        )
        axis.semilogx(
            ell[valid],
            empirical[valid],
            color="#D55E00",
            linewidth=1.1,
            label=r"Measured coherent $U+H$",
        )
        axis.semilogx(
            ell,
            corrected,
            color="#0072B2",
            linewidth=1.0,
            linestyle="--",
            label="Corrected two-halo",
        )
        axis.semilogx(
            ell,
            total_ratio,
            color="0.45",
            linewidth=0.8,
            linestyle=":",
            label="Measured total / baseline total",
        )
        axis.axhline(1.0, color="black", linewidth=0.7)
        axis.set_ylim(0.0, 1.8)
        axis.set_title(
            f"{data.z_lo[shell]:.3f} < z < {data.z_hi[shell]:.3f}", fontsize=8
        )
        _format_log_ell_axis(axis, float(data.ell_max[-1]))
        axis.tick_params(labelsize=7)
    axes[0, 0].legend(frameon=False, fontsize=6.4, loc="upper left")
    figure.supxlabel(r"Multipole $\ell$", fontsize=10)
    figure.supylabel("Power relative to linear shell theory", fontsize=10)
    figure.subplots_adjust(left=0.075, right=0.995, bottom=0.065, top=0.965, hspace=0.3)
    outputs = _save_figure(
        figure,
        output_dir / "fullsky_component_two_halo_audit_all_shells",
        png_dpi,
    )
    plt.close(figure)
    return outputs


def render_band_summary(
    rows: list[dict[str, object]],
    output_dir: Path,
    *,
    png_dpi: int,
) -> list[Path]:
    """Plot the large-scale response and component correlation versus redshift."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional install
        raise ImportError("component-audit plotting requires matplotlib") from exc
    order = np.argsort([0.5 * (float(row["z_lo"]) + float(row["z_hi"])) for row in rows])

    def values(key: str) -> np.ndarray:
        return np.asarray([float(rows[index][key]) for index in order])

    redshift = np.asarray(
        [0.5 * (float(rows[index]["z_lo"]) + float(rows[index]["z_hi"])) for index in order]
    )
    figure, axes = plt.subplots(2, 1, figsize=(6.4, 5.5), sharex=True)
    empirical = values("measured_coherent_over_linear")
    empirical_sem = values("measured_coherent_over_linear_jackknife_sem")
    corrected = values("corrected_two_halo_over_linear")
    axes[0].fill_between(
        redshift,
        empirical - empirical_sem,
        empirical + empirical_sem,
        color="#D55E00",
        alpha=0.22,
        linewidth=0.0,
    )
    axes[0].plot(redshift, empirical, color="#D55E00", marker="o", label="Measured coherent")
    axes[0].plot(
        redshift,
        corrected,
        color="#0072B2",
        marker="s",
        linestyle="--",
        label="Corrected two-halo",
    )
    axes[0].axhline(1.0, color="black", linewidth=0.8, label="Linear baseline")
    axes[0].set_ylabel(r"$C_\ell^{\rm coherent}/C_\ell^{\rm linear}$")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_ylim(0.5, 1.6)
    axes[1].errorbar(
        redshift,
        values("cross_correlation"),
        yerr=values("cross_correlation_jackknife_sem"),
        color="#009E73",
        marker="o",
        capsize=2,
    )
    axes[1].axhline(1.0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Shell midpoint redshift")
    axes[1].set_ylabel(r"$r_{UH}$ after $U$ shot subtraction")
    axes[1].set_ylim(0.0, 1.02)
    ell_min = int(rows[0]["ell_min"])
    ell_max = int(rows[0]["ell_max"])
    axes[0].set_title(
        rf"Mode-count weighted ${ell_min} \leq \ell \leq {ell_max}$", fontsize=10
    )
    figure.subplots_adjust(left=0.13, right=0.98, bottom=0.1, top=0.94, hspace=0.08)
    outputs = _save_figure(
        figure,
        output_dir / "fullsky_component_two_halo_band_summary",
        png_dpi,
    )
    plt.close(figure)
    return outputs


def render_component_budget(
    data: BinnedComponentAudit,
    output_dir: Path,
    *,
    png_dpi: int,
    ell_max: int,
) -> list[Path]:
    """Plot the coherent power budget for representative redshift shells."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional install
        raise ImportError("component-audit plotting requires matplotlib") from exc
    midpoint = 0.5 * (data.z_lo + data.z_hi)
    targets = np.asarray([0.47, 0.23, 0.085, 0.017])
    selected = []
    for target in targets:
        candidate = int(np.argmin(np.abs(midpoint - target)))
        if candidate not in selected:
            selected.append(candidate)
    figure, axes = plt.subplots(2, 2, figsize=(8.0, 6.0), sharex=True, sharey=True)
    mean_cross = np.mean(data.realization_cross, axis=0)
    for shell, axis in zip(selected, axes.flat, strict=True):
        linear = data.linear[shell]
        terms = (
            (data.uncollapsed_clustering[shell], r"$UU-N_U$", "#0072B2", "-"),
            (2.0 * mean_cross[shell], r"$2UH$", "#009E73", "-"),
            (data.halo_parallel[shell], r"$HH_\parallel$", "#CC79A7", "-"),
            (data.coherent_total[shell], "Measured coherent sum", "#D55E00", "-"),
            (data.two_halo[shell], "Corrected two-halo", "black", "--"),
        )
        for term, label, color, linestyle in terms:
            ratio = _safe_ratio(term, linear)
            valid = np.isfinite(ratio) & (data.ell_effective <= ell_max)
            axis.semilogx(
                data.ell_effective[valid],
                ratio[valid],
                color=color,
                linestyle=linestyle,
                linewidth=1.0,
                label=label,
            )
        axis.axhline(1.0, color="0.55", linewidth=0.7)
        axis.set_title(f"{data.z_lo[shell]:.3f} < z < {data.z_hi[shell]:.3f}", fontsize=9)
        axis.set_ylim(0.0, 1.8)
        _format_log_ell_axis(axis, float(ell_max))
    axes[0, 0].legend(frameon=False, fontsize=7, ncol=2)
    figure.supxlabel(r"Multipole $\ell$")
    figure.supylabel("Contribution relative to linear shell theory")
    figure.subplots_adjust(left=0.1, right=0.99, bottom=0.09, top=0.96, hspace=0.25)
    outputs = _save_figure(
        figure,
        output_dir / "fullsky_component_power_budget_representative_shells",
        png_dpi,
    )
    plt.close(figure)
    return outputs


def main() -> None:
    args = parse_args()
    if len(args.component_cache) != len(args.seed):
        raise ValueError("provide one --seed for every --component-cache")
    spectra = [load_component_spectrum(path) for path in args.component_cache]
    theory = load_aligned_theory(args.theory_dir, spectra[0])
    audit = build_component_audit(
        spectra,
        theory,
        args.seed,
        ell_min=args.ell_min,
        ell_max=args.ell_max,
        bin_width=args.bin_width,
    )
    summary_rows = summarize_band(
        audit,
        ell_min=args.ell_min,
        ell_max=min(args.summary_ell_max, args.ell_max),
    )
    outputs = [
        write_detailed_csv(
            audit,
            args.output_dir / "fullsky_component_two_halo_audit.csv",
        ),
        write_summary_csv(
            summary_rows,
            args.output_dir / "fullsky_component_two_halo_band_summary.csv",
        ),
    ]
    outputs.extend(render_all_shells(audit, args.output_dir, png_dpi=args.png_dpi))
    outputs.extend(render_band_summary(summary_rows, args.output_dir, png_dpi=args.png_dpi))
    outputs.extend(
        render_component_budget(
            audit,
            args.output_dir,
            png_dpi=args.png_dpi,
            ell_max=min(args.summary_ell_max, args.ell_max),
        )
    )
    for path in outputs:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
