"""Host-side calibration of halo bias from numerical PINOCCHIO HMFs.

The fit and optional CCToolkit call are deliberately outside the JAX theory
kernels. The resulting :class:`geppetto.theory.HaloBiasTable` contains plain
JAX arrays and can be passed through differentiable two-halo calculations.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from geppetto.cosmology import RHO_CRIT0_MSUNH_PER_MPCH3
from geppetto.io import PinocchioMassFunction
from geppetto.theory import (
    HaloBiasTable,
    HaloMassFunctionTable,
    LinearTheoryTable,
    sigma8_from_linear_power,
)

CCTOOLKIT_REVISION = "ac16ab613eb93f795f562f928ad145597d981b2f"


@dataclass(frozen=True)
class NumericalPBSFitParams:
    """Controls for fitting a smooth dimensionless multiplicity.

    ``collapse_threshold`` is dimensionless. The residual limit is the
    count-weighted RMS difference in log multiplicity; it measures profile
    fidelity without treating neighboring numerical-HMF bins as independent
    Poisson observations.
    """

    collapse_threshold: float = 1.686
    minimum_populated_bins: int = 8
    maximum_weighted_log_residual: float = 0.05


@dataclass(frozen=True)
class NumericalPBSFitDiagnostic:
    """Dimensionless fit diagnostics for one native PINOCCHIO HMF snapshot."""

    source: Path
    redshift: float
    populated_bins: int
    log_amplitude: float
    a: float
    p: float
    q: float
    weighted_log_residual: float
    reduced_weighted_residual: float
    minimum_pbs_bias: float
    maximum_pbs_bias: float
    minimum_correction: float
    maximum_correction: float
    minimum_linear_bias: float
    maximum_linear_bias: float


def _load_cctoolkit_correction() -> Callable[..., np.ndarray]:
    try:
        from cctoolkit.bias import bias_correction_PBS
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise ImportError(
            "Castro-corrected halo bias requires CCToolkit; install "
            "geppetto[theory] or geppetto[validation]"
        ) from exc
    return bias_correction_PBS


def _validate_fit_params(params: NumericalPBSFitParams) -> None:
    if not np.isfinite(params.collapse_threshold) or params.collapse_threshold <= 0.0:
        raise ValueError("PBS collapse threshold must be finite and positive")
    if params.minimum_populated_bins < 8:
        raise ValueError("PBS fits require at least eight populated bins")
    if (
        not np.isfinite(params.maximum_weighted_log_residual)
        or params.maximum_weighted_log_residual <= 0.0
    ):
        raise ValueError("maximum weighted PBS log residual must be finite and positive")


def _fit_one_numerical_hmf(
    table: PinocchioMassFunction,
    common_log_mass: np.ndarray,
    mean_density: float,
    omega_m_z: float,
    s8: float,
    params: NumericalPBSFitParams,
    correction_function: Callable[..., np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, NumericalPBSFitDiagnostic]:
    try:
        from scipy.interpolate import PchipInterpolator
        from scipy.optimize import least_squares
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise ImportError("numerical PBS fitting requires scipy; install geppetto[theory]") from exc

    if table.redshift is None:
        raise ValueError(f"PINOCCHIO HMF has no redshift: {table.source}")
    mass = np.asarray(table.mass_msun_h, dtype=np.float64)
    number_density = np.asarray(table.number_density, dtype=np.float64)
    counts = np.asarray(table.halo_counts, dtype=np.float64)
    peak_height = np.asarray(table.peak_height_nu, dtype=np.float64)
    if not (
        mass.ndim == 1
        and number_density.shape == mass.shape
        and counts.shape == mass.shape
        and peak_height.shape == mass.shape
        and mass.size >= params.minimum_populated_bins
        and np.all(np.isfinite(mass))
        and np.all(np.isfinite(number_density))
        and np.all(np.isfinite(counts))
        and np.all(np.isfinite(peak_height))
        and np.all(mass > 0.0)
        and np.all(number_density >= 0.0)
        and np.all(counts >= 0.0)
        and np.all(peak_height > 0.0)
    ):
        raise ValueError(f"invalid numerical HMF columns for PBS fit: {table.source}")

    order = np.argsort(mass)
    mass = mass[order]
    number_density = number_density[order]
    counts = counts[order]
    peak_height = peak_height[order]
    log_mass = np.log(mass)
    log_peak_height = np.log(peak_height)
    if np.any(np.diff(log_mass) <= 0.0) or np.any(np.diff(log_peak_height) <= 0.0):
        raise ValueError(
            f"PBS fitting requires increasing mass and peak height: {table.source}"
        )

    peak_height_interpolator = PchipInterpolator(log_mass, log_peak_height, extrapolate=False)
    dlnnu_dlnm = peak_height_interpolator.derivative()(log_mass)
    multiplicity = mass**2 * number_density / (mean_density * dlnnu_dlnm)
    populated = (
        (counts > 0.0)
        & (number_density > 0.0)
        & (dlnnu_dlnm > 0.0)
        & np.isfinite(multiplicity)
        & (multiplicity > 0.0)
    )
    populated_bins = int(np.count_nonzero(populated))
    if populated_bins < params.minimum_populated_bins:
        raise ValueError(
            "numerical PBS fit has too few populated bins: "
            f"source={table.source}, redshift={table.redshift:g}, "
            f"populated_bins={populated_bins}, required={params.minimum_populated_bins}"
        )

    fit_nu = peak_height[populated]
    fit_log_multiplicity = np.log(multiplicity[populated])
    fit_counts = counts[populated]

    def log_shape(log_a: float, p: float, q: float) -> np.ndarray:
        a_nu_squared = np.exp(log_a) * fit_nu**2
        return (
            q * np.log(fit_nu)
            - 0.5 * a_nu_squared
            + np.logaddexp(0.0, -p * np.log(a_nu_squared))
        )

    def residual(parameters: np.ndarray) -> np.ndarray:
        log_amplitude, log_a, p, q = parameters
        model = log_amplitude + log_shape(log_a, p, q)
        return np.sqrt(fit_counts) * (model - fit_log_multiplicity)

    lower = np.asarray([-20.0, np.log(0.05), -2.0, -1.0])
    upper = np.asarray([10.0, np.log(5.0), 2.0, 5.0])
    best = None
    for initial_a in (0.5, 1.0):
        for initial_p in (-0.5, 0.3):
            for initial_q in (0.5, 1.5):
                initial_shape = log_shape(np.log(initial_a), initial_p, initial_q)
                initial_amplitude = np.average(
                    fit_log_multiplicity - initial_shape,
                    weights=fit_counts,
                )
                initial = np.asarray(
                    [
                        np.clip(initial_amplitude, lower[0], upper[0]),
                        np.log(initial_a),
                        initial_p,
                        initial_q,
                    ]
                )
                result = least_squares(
                    residual,
                    initial,
                    bounds=(lower, upper),
                    loss="soft_l1",
                    f_scale=1.0,
                    max_nfev=5000,
                )
                if best is None or result.cost < best.cost:
                    best = result
    assert best is not None
    if not best.success or not np.all(np.isfinite(best.x)):
        raise ValueError(
            "numerical PBS fit failed: "
            f"source={table.source}, redshift={table.redshift:g}, "
            f"status={best.status}, message={best.message!s}"
        )

    degrees_of_freedom = populated_bins - 4
    reduced_residual = float(np.sum(residual(best.x) ** 2) / degrees_of_freedom)
    weighted_log_residual = float(
        np.sqrt(np.sum(residual(best.x) ** 2) / np.sum(fit_counts))
    )
    log_amplitude, log_a, fitted_p, fitted_q = (float(value) for value in best.x)
    fitted_a = float(np.exp(log_a))
    if (
        not np.isfinite(weighted_log_residual)
        or weighted_log_residual > params.maximum_weighted_log_residual
    ):
        raise ValueError(
            "numerical PBS fit exceeds its residual threshold: "
            f"source={table.source}, redshift={table.redshift:g}, "
            f"weighted_log_residual={weighted_log_residual:.6g}, "
            f"maximum={params.maximum_weighted_log_residual:.6g}, "
            f"populated_bins={populated_bins}, reduced_weighted_residual={reduced_residual:.6g}, "
            f"log_amplitude={log_amplitude:.6g}, a={fitted_a:.6g}, "
            f"p={fitted_p:.6g}, q={fitted_q:.6g}"
        )

    clipped_log_mass = np.clip(common_log_mass, log_mass[0], log_mass[-1])
    common_log_nu = peak_height_interpolator(clipped_log_mass)
    common_nu = np.exp(common_log_nu)
    common_slope = peak_height_interpolator.derivative()(clipped_log_mass)
    a_nu_squared = fitted_a * common_nu**2
    logarithmic_multiplicity_slope = (
        fitted_q
        - a_nu_squared
        - 2.0 * fitted_p / (1.0 + a_nu_squared**fitted_p)
    )
    pbs_bias = 1.0 - logarithmic_multiplicity_slope / params.collapse_threshold
    dlnsigma_dlnr = -3.0 * common_slope
    correction = np.asarray(
        correction_function(omega_m_z, dlnsigma_dlnr, s8),
        dtype=np.float64,
    )
    try:
        correction = np.broadcast_to(correction, common_log_mass.shape).copy()
    except ValueError as exc:
        raise ValueError(
            "CCToolkit bias correction returned an incompatible shape: "
            f"source={table.source}, correction_shape={correction.shape}, "
            f"expected={common_log_mass.shape}"
        ) from exc
    linear_bias = pbs_bias * correction
    valid = np.isfinite(pbs_bias) & np.isfinite(correction) & np.isfinite(linear_bias)
    if not np.all(valid) or np.any(pbs_bias <= 0.0) or np.any(correction <= 0.0):
        raise ValueError(
            "numerical PBS fit produced invalid corrected bias: "
            f"source={table.source}, redshift={table.redshift:g}"
        )

    diagnostic = NumericalPBSFitDiagnostic(
        source=table.source,
        redshift=float(table.redshift),
        populated_bins=populated_bins,
        log_amplitude=log_amplitude,
        a=fitted_a,
        p=fitted_p,
        q=fitted_q,
        weighted_log_residual=weighted_log_residual,
        reduced_weighted_residual=reduced_residual,
        minimum_pbs_bias=float(np.min(pbs_bias)),
        maximum_pbs_bias=float(np.max(pbs_bias)),
        minimum_correction=float(np.min(correction)),
        maximum_correction=float(np.max(correction)),
        minimum_linear_bias=float(np.min(linear_bias)),
        maximum_linear_bias=float(np.max(linear_bias)),
    )
    return pbs_bias, correction, linear_bias, diagnostic


def fit_pinocchio_numerical_halo_bias(
    tables: Sequence[PinocchioMassFunction],
    mass_function: HaloMassFunctionTable,
    linear_theory: LinearTheoryTable,
    *,
    params: NumericalPBSFitParams | None = None,
    correction_function: Callable[..., np.ndarray] | None = None,
) -> tuple[HaloBiasTable, tuple[NumericalPBSFitDiagnostic, ...]]:
    """Fit numerical PBS bias and apply the Castro et al. correction.

    The numerical HMF and PINOCCHIO peak heights determine the PBS term in
    their native halo-mass convention. Only the multiplicative correction is
    obtained from CCToolkit. This host-side fit is not differentiable; the
    returned dimensionless bias is concentration-independent input to JAX
    theory kernels. Input masses are ``Msun/h`` and number densities follow
    :class:`geppetto.io.PinocchioMassFunction`.
    """

    params = NumericalPBSFitParams() if params is None else params
    _validate_fit_params(params)
    if not tables:
        raise ValueError("at least one PINOCCHIO HMF is required for PBS fitting")
    missing_redshift = [str(table.source) for table in tables if table.redshift is None]
    if missing_redshift:
        raise ValueError(
            "PINOCCHIO HMF has no redshift for PBS fitting: " + ", ".join(missing_redshift)
        )
    correction = correction_function or _load_cctoolkit_correction()
    common_log_mass = np.asarray(mass_function.log_mass_msun_h, dtype=np.float64)
    if common_log_mass.ndim != 1 or common_log_mass.size < 2:
        raise ValueError("mass-function common mass grid must contain at least two bins")
    mean_density = RHO_CRIT0_MSUNH_PER_MPCH3 * linear_theory.omega_m0
    sigma8 = float(np.asarray(sigma8_from_linear_power(linear_theory)))
    s8 = sigma8 * np.sqrt(linear_theory.omega_m0 / 0.3)
    scale_grid = np.asarray(linear_theory.scale_factor, dtype=np.float64)
    omega_grid = np.asarray(linear_theory.omega_m, dtype=np.float64)

    ordered_tables = sorted(tables, key=lambda table: 1.0 / (1.0 + float(table.redshift)))
    scale_factor = np.asarray(
        [1.0 / (1.0 + float(table.redshift)) for table in ordered_tables],
        dtype=np.float64,
    )
    expected_scale = np.asarray(mass_function.scale_factor, dtype=np.float64)
    source_dtype = np.asarray(mass_function.scale_factor).dtype
    scale_tolerance = max(1.0e-10, 8.0 * np.finfo(source_dtype).eps)
    if scale_factor.shape != expected_scale.shape or not np.allclose(
        scale_factor,
        expected_scale,
        rtol=scale_tolerance,
        atol=scale_tolerance,
    ):
        raise ValueError("native HMF redshifts do not match the mass-function series")

    pbs_rows: list[np.ndarray] = []
    correction_rows: list[np.ndarray] = []
    bias_rows: list[np.ndarray] = []
    diagnostics: list[NumericalPBSFitDiagnostic] = []
    for table, scale in zip(ordered_tables, scale_factor, strict=True):
        omega_m_z = float(np.interp(scale, scale_grid, omega_grid))
        pbs, correction_values, bias, diagnostic = _fit_one_numerical_hmf(
            table,
            common_log_mass,
            mean_density,
            omega_m_z,
            s8,
            params,
            correction,
        )
        pbs_rows.append(pbs)
        correction_rows.append(correction_values)
        bias_rows.append(bias)
        diagnostics.append(diagnostic)

    return (
        HaloBiasTable(
            scale_factor=jnp.asarray(scale_factor),
            log_mass_msun_h=jnp.asarray(common_log_mass),
            pbs_bias=jnp.asarray(np.stack(pbs_rows)),
            correction=jnp.asarray(np.stack(correction_rows)),
            linear_bias=jnp.asarray(np.stack(bias_rows)),
        ),
        tuple(diagnostics),
    )
