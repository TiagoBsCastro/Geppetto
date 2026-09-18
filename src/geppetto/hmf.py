"""Host-side, mass-normalized PINOCCHIO HMF fits for standard halo theory.

The fit is to abundance, never to measured angular power. Integration and
Castro-bias calibration are fixed inputs to the differentiable JAX kernels
in :mod:`geppetto.theory`. All masses are Msun/h and volumes (Mpc/h)^3.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from geppetto.cosmology import RHO_CRIT0_MSUNH_PER_MPCH3
from geppetto.halo_bias import _load_cctoolkit_correction
from geppetto.io import PinocchioMassFunction
from geppetto.theory import (
    LinearPowerEvolutionTable,
    LinearTheoryTable,
    NormalizedHaloTable,
    linear_matter_power,
    sigma8_from_linear_power,
)


@dataclass(frozen=True)
class NormalizedHMFFitParams:
    """Dimensionless fit tolerances for F(nu) per dln(nu)."""

    collapse_threshold: float = 1.686
    minimum_populated_bins: int = 8
    maximum_weighted_log_residual: float = 0.05
    peak_height_rtol: float = 0.01


@dataclass(frozen=True)
class NormalizedHMFFit:
    """One native snapshot fit; amplitude is analytic, not a free parameter."""

    source: str
    redshift: float
    log_a: float
    p: float
    s: float
    populated_bins: int
    weighted_log_residual: float
    peak_height_max_relative_error: float
    collapse_threshold: float = 1.686


@dataclass(frozen=True)
class HMFIntegrationParams:
    """Log-mass quadrature bounds in Msun/h and dimensionless tolerances.

    Gauss--Legendre nodes in log mass avoid endpoint trapezoidal closure
    errors. Low-tail mass and bias integrals are analytic; that population
    has u=1. A second, lower-cutoff integration must test its power error.
    """

    minimum_mass: float = 1.0e6
    maximum_mass: float = 1.0e18
    mass_order: int = 512
    closure_rtol: float = 1.0e-5
    maximum_high_tail: float = 1.0e-12


@dataclass(frozen=True)
class HMFClosureDiagnostic:
    """Closure before any numerical rescaling of the fitted abundance."""

    redshift: float
    mass_integral: float
    bias_integral: float
    bias_renormalization: float
    low_mass_fraction: float
    high_mass_fraction: float
    low_mass_one_halo_upper_bound_mpc_h3: float


DEFAULT_HMF_FIT_PARAMS = NormalizedHMFFitParams()
DEFAULT_HMF_INTEGRATION_PARAMS = HMFIntegrationParams()


def multiplicity_log_amplitude(log_a: float, p: float, s: float) -> float:
    """Analytically normalize nu^(2p+s) [1+(a nu^2)^-p] exp(-a nu^2/2)."""

    from scipy.special import gammaln

    q = 2.0 * p + s
    return float(
        -(
            -np.log(2.0)
            + 0.5 * q * (np.log(2.0) - log_a)
            + np.logaddexp(gammaln(q / 2.0), -p * np.log(2.0) + gammaln(s / 2.0))
        )
    )


def normalized_multiplicity(nu: np.ndarray, log_a: float, p: float, s: float) -> np.ndarray:
    """Return the mass fraction F(nu) per dln(nu), integrating to one."""

    log_nu = np.log(nu)
    log_x = log_a + 2.0 * log_nu
    return np.exp(
        multiplicity_log_amplitude(log_a, p, s)
        + (2.0 * p + s) * log_nu
        + np.logaddexp(0.0, -p * log_x)
        - 0.5 * np.exp(log_x)
    )


def multiplicity_tail_integrals(
    nu: float,
    log_a: float,
    p: float,
    s: float,
    *,
    upper: bool = False,
    collapse_threshold: float = 1.686,
) -> tuple[float, float]:
    """Return analytic mass and PBS-bias integrals below (or above) nu."""

    from scipy.special import gammainc, gammaincc, gammaln

    q = 2.0 * p + s
    log_terms = np.array([gammaln(q / 2.0), -p * np.log(2.0) + gammaln(s / 2.0)])
    mixture = np.exp(log_terms - np.logaddexp(*log_terms))
    gamma = gammaincc if upper else gammainc
    mass = float(mixture @ gamma(np.array([q / 2.0, s / 2.0]), np.exp(log_a) * nu**2 / 2.0))
    boundary = float(normalized_multiplicity(np.asarray(nu), log_a, p, s))
    return mass, mass + (1.0 if upper else -1.0) * boundary / collapse_threshold


def peak_height_from_power(
    mass_msun_h: np.ndarray,
    redshift: float,
    linear_theory: LinearTheoryTable,
    power_evolution: LinearPowerEvolutionTable | None = None,
    *,
    collapse_threshold: float = 1.686,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute nu and dln(nu)/dln(M) using the supplied top-hat variance.

    This host precomputation integrates k in h/Mpc and P in (Mpc/h)^3.
    It never extrapolates the input power or assumes separable growth when
    the PINOCCHIO CAMB series is available. Mass blocks bound memory use.
    """

    mass = np.asarray(mass_msun_h, dtype=np.float64)
    if mass.ndim != 1 or not np.all(np.isfinite(mass) & (mass > 0.0)):
        raise ValueError("peak-height masses must be a finite positive vector")
    source = linear_theory if power_evolution is None else power_evolution
    scale = 1.0 / (1.0 + redshift)
    if not float(source.scale_factor[0]) <= scale <= float(source.scale_factor[-1]):
        raise ValueError(f"HMF redshift {redshift:g} exceeds the linear-power table")
    k = np.asarray(source.k_h_mpc, dtype=np.float64)
    power = np.asarray(
        linear_matter_power(jnp.asarray(k), jnp.asarray(redshift), linear_theory, power_evolution),
        dtype=np.float64,
    )
    density = RHO_CRIT0_MSUNH_PER_MPCH3 * linear_theory.omega_m0
    radius = (3.0 * mass / (4.0 * np.pi * density)) ** (1.0 / 3.0)
    nu_blocks, slope_blocks = [], []
    from scipy.integrate import trapezoid

    for start in range(0, mass.size, 128):
        x = radius[start : start + 128, None] * k[None, :]
        x2 = x**2
        small = x < 0.03
        window = np.where(
            small, 1 - x2 / 10 + x2**2 / 280 - x2**3 / 15120, 3 * (np.sin(x) - x * np.cos(x)) / x**3
        )
        derivative = np.where(
            small, -x2 / 5 + x2**2 / 70 - x2**3 / 2520, 3 * np.sin(x) / x - 3 * window
        )
        measure = k**3 * power / (2.0 * np.pi**2)
        variance = trapezoid(measure * window**2, x=np.log(k), axis=-1)
        slope = -trapezoid(measure * window * derivative, x=np.log(k), axis=-1) / (3.0 * variance)
        nu_blocks.append(collapse_threshold / np.sqrt(variance))
        slope_blocks.append(slope)
    nu, slope = np.concatenate(nu_blocks), np.concatenate(slope_blocks)
    if not np.all(np.isfinite(nu) & np.isfinite(slope) & (slope > 0.0)):
        raise ValueError(f"non-positive/non-finite HMF variance or slope at z={redshift:g}")
    return nu, slope


def fit_normalized_hmf(
    tables: Sequence[PinocchioMassFunction],
    linear_theory: LinearTheoryTable,
    power_evolution: LinearPowerEvolutionTable | None = None,
    *,
    params: NormalizedHMFFitParams = DEFAULT_HMF_FIT_PARAMS,
) -> tuple[NormalizedHMFFit, ...]:
    """Fit each snapshot without rescaling its measured abundance.

    A, a, p, q follow F(nu) per dln(nu), q=2p+s. Fits use positive s below
    delta_c, guaranteeing an integrable tail and positive low-nu PBS bias.
    Tabulation checks positivity across the complete numerical bias grid.
    Count-weighted robust residuals constrain shape, not Poisson likelihood.
    """

    from scipy.optimize import least_squares

    if (
        params.minimum_populated_bins < 8
        or params.collapse_threshold <= 0.011
        or not np.all(
            np.isfinite(
                [
                    params.collapse_threshold,
                    params.maximum_weighted_log_residual,
                    params.peak_height_rtol,
                ]
            )
        )
        or params.maximum_weighted_log_residual <= 0
        or params.peak_height_rtol <= 0
    ):
        raise ValueError("invalid normalized-HMF fit controls")
    if not tables or any(table.redshift is None for table in tables):
        raise ValueError("normalized HMF requires at least one snapshot with a redshift")
    ordered = sorted(tables, key=lambda table: -float(table.redshift))
    if len({table.redshift for table in ordered}) != len(ordered):
        raise ValueError("fit each realization separately: duplicate HMF redshifts")
    density = RHO_CRIT0_MSUNH_PER_MPCH3 * linear_theory.omega_m0
    fits = []
    for table in ordered:
        mass = np.asarray(table.mass_msun_h, dtype=np.float64)
        counts = np.asarray(table.halo_counts, dtype=np.float64)
        dndm = np.asarray(table.number_density, dtype=np.float64)
        native_nu = np.asarray(table.peak_height_nu, dtype=np.float64)
        if (
            mass.ndim != 1
            or any(x.shape != mass.shape for x in (counts, dndm, native_nu))
            or not np.all(np.isfinite([mass, counts, dndm, native_nu]))
            or np.any(mass <= 0)
            or np.any(np.diff(mass) <= 0)
            or np.any(counts < 0)
            or np.any(dndm < 0)
            or np.any(native_nu <= 0)
        ):
            raise ValueError(f"invalid numerical HMF columns: {table.source}")
        populated = (counts > 0) & (dndm > 0)
        if np.count_nonzero(populated) < params.minimum_populated_bins:
            raise ValueError(f"too few populated normalized-HMF bins: {table.source}")
        mass, counts, dndm = mass[populated], counts[populated], dndm[populated]
        nu, slope = peak_height_from_power(
            mass,
            float(table.redshift),
            linear_theory,
            power_evolution,
            collapse_threshold=params.collapse_threshold,
        )
        peak_error = float(np.max(np.abs(nu / native_nu[populated] - 1.0)))
        if peak_error > params.peak_height_rtol:
            raise ValueError(f"peak-height closure failed: {table.source}, error={peak_error:.6g}")
        observed = np.log(mass**2 * dndm / (density * slope))

        def residual(shape: np.ndarray, nu=nu, counts=counts, observed=observed) -> np.ndarray:
            log_a, p, s = shape
            log_x = log_a + 2 * np.log(nu)
            prediction = (
                multiplicity_log_amplitude(log_a, p, s)
                + (2 * p + s) * np.log(nu)
                + np.logaddexp(0.0, -p * log_x)
                - 0.5 * np.exp(log_x)
            )
            return np.sqrt(counts) * (prediction - observed)

        s_upper = params.collapse_threshold - 0.001
        results = [
            least_squares(
                residual,
                [np.log(a), p, s if s < s_upper else (0.01 + s_upper) / 2],
                bounds=(
                    [np.log(0.05), 0.0, 0.01],
                    [np.log(5.0), 2.0, s_upper],
                ),
                loss="soft_l1",
                max_nfev=5000,
            )
            for a in (0.5, 1.0)
            for p in (0.1, 0.5)
            for s in (0.2, 1.0)
        ]
        successful = [
            result for result in results if result.success and np.all(np.isfinite(result.x))
        ]
        if not successful:
            raise ValueError(f"normalized-HMF optimizer failed: {table.source}")
        best = min(successful, key=lambda result: result.cost)
        rms = float(np.sqrt(np.sum(residual(best.x) ** 2) / np.sum(counts)))
        if rms > params.maximum_weighted_log_residual:
            raise ValueError(
                f"normalized-HMF fit residual exceeded: {table.source}, "
                f"weighted_log_residual={rms:.6g}"
            )
        fits.append(
            NormalizedHMFFit(
                str(table.source),
                float(table.redshift),
                *map(float, best.x),
                int(mass.size),
                rms,
                peak_error,
                params.collapse_threshold,
            )
        )
    return tuple(fits)


def tabulate_normalized_hmf(
    fits: Sequence[NormalizedHMFFit],
    redshifts: np.ndarray,
    linear_theory: LinearTheoryTable,
    power_evolution: LinearPowerEvolutionTable | None = None,
    *,
    params: HMFIntegrationParams = DEFAULT_HMF_INTEGRATION_PARAMS,
    correction_function: Callable[..., np.ndarray] | None = None,
    peak_height_cache: dict | None = None,
) -> tuple[NormalizedHaloTable, tuple[HMFClosureDiagnostic, ...]]:
    """Evaluate PCHIP-interpolated fit shapes at the projection redshifts.

    Recompute analytic mass normalization and the Castro-corrected bias
    integral at each redshift. Divide bias, not abundance, by its complete
    mass-weighted integral. Outside the mass grid the correction is held at
    its boundary value and analytic PBS tails are used. This is an adapted,
    normalized Castro bias, not the unmodified CCToolkit calibration.
    An optional caller-owned variance cache avoids repeating identical power
    integrals across ensemble fits. Keys include the complete power inputs.
    """

    from scipy.interpolate import PchipInterpolator

    if (
        not 0 < params.minimum_mass < params.maximum_mass
        or params.mass_order < 16
        or params.closure_rtol <= 0
        or params.maximum_high_tail <= 0
        or not np.all(
            np.isfinite(
                [
                    params.minimum_mass,
                    params.maximum_mass,
                    params.closure_rtol,
                    params.maximum_high_tail,
                ]
            )
        )
    ):
        raise ValueError("invalid HMF integration controls")
    if len(fits) < 2:
        raise ValueError("HMF interpolation requires at least two fits")
    ordered = sorted(fits, key=lambda fit: -fit.redshift)
    fit_a = np.array([1 / (1 + fit.redshift) for fit in ordered])
    scale = np.sort(np.unique(1 / (1 + np.asarray(redshifts, dtype=np.float64))))
    if (
        scale.size < 2
        or not np.all(np.isfinite(scale))
        or (scale[0] < fit_a[0] - 1e-10 or scale[-1] > fit_a[-1] + 1e-10)
    ):
        raise ValueError("projection redshifts are outside fitted HMF coverage")
    shape = PchipInterpolator(fit_a, [[f.log_a, f.p, f.s] for f in ordered], axis=0)(
        np.clip(scale, fit_a[0], fit_a[-1])
    )
    delta = ordered[0].collapse_threshold
    if any(f.collapse_threshold != delta for f in fits):
        raise ValueError("HMF fits have inconsistent collapse thresholds")
    nodes, weights = np.polynomial.legendre.leggauss(params.mass_order)
    lower, upper = np.log([params.minimum_mass, params.maximum_mass])
    mass = np.exp((lower + upper) / 2 + (upper - lower) * nodes / 2)
    weights = weights * (upper - lower) / 2
    mass_with_edges = np.r_[params.minimum_mass, mass, params.maximum_mass]
    digest = hashlib.sha256()
    for value in (*linear_theory, *(power_evolution or ()), mass_with_edges, delta):
        array = np.ascontiguousarray(value)
        digest.update(array.dtype.str.encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    variance_key = digest.hexdigest()
    correction = (
        _load_cctoolkit_correction() if correction_function is None else correction_function
    )
    s8 = float(sigma8_from_linear_power(linear_theory)) * np.sqrt(linear_theory.omega_m0 / 0.3)
    density = RHO_CRIT0_MSUNH_PER_MPCH3 * linear_theory.omega_m0
    mass_rows, bias_rows, tail_rows, tail_bias_rows, diagnostics = [], [], [], [], []
    for a, (log_a, p, s) in zip(scale, shape, strict=True):
        z = float(1 / a - 1)
        key = (variance_key, z)
        cached = None if peak_height_cache is None else peak_height_cache.get(key)
        if cached is None:
            cached = peak_height_from_power(
                mass_with_edges, z, linear_theory, power_evolution, collapse_threshold=delta
            )
            if peak_height_cache is not None:
                peak_height_cache[key] = cached
        nu, slope = cached
        multiplicity = normalized_multiplicity(nu, log_a, p, s)
        pbs = (
            1
            - (2 * p + s - np.exp(log_a) * nu**2 - 2 * p / (1 + (np.exp(log_a) * nu**2) ** p))
            / delta
        )
        omega_m = float(np.interp(a, linear_theory.scale_factor, linear_theory.omega_m))
        corrected = np.broadcast_to(np.asarray(correction(omega_m, -3 * slope, s8)), nu.shape)
        if not np.all(np.isfinite(corrected) & (corrected > 0) & (pbs > 0)):
            raise ValueError(f"invalid Castro/PBS bias at z={z:g}")
        low_mass, low_bias = multiplicity_tail_integrals(
            nu[0], log_a, p, s, collapse_threshold=delta
        )
        high_mass, high_bias = multiplicity_tail_integrals(
            nu[-1], log_a, p, s, upper=True, collapse_threshold=delta
        )
        if high_mass > params.maximum_high_tail:
            raise ValueError(f"upper HMF tail is not negligible at z={z:g}: {high_mass:.6g}")
        mass_weight = multiplicity[1:-1] * slope[1:-1] * weights
        biased_weight = mass_weight * pbs[1:-1] * corrected[1:-1]
        tail_bias = low_bias * corrected[0] + high_bias * corrected[-1]
        bias_norm = float(np.sum(biased_weight) + tail_bias)
        mass_norm = float(np.sum(mass_weight) + low_mass + high_mass)
        if not np.isfinite(bias_norm) or bias_norm <= 0 or abs(mass_norm - 1) > params.closure_rtol:
            raise ValueError(
                f"HMF quadrature closure failed at z={z:g}: mass={mass_norm:.12g}, "
                f"bias={bias_norm:.12g}; increase mass_order"
            )
        mass_rows.append(mass_weight)
        bias_rows.append(biased_weight / bias_norm)
        tail_rows.append(low_mass + high_mass)
        tail_bias_rows.append(tail_bias / bias_norm)
        diagnostics.append(
            HMFClosureDiagnostic(
                z,
                mass_norm,
                float(np.sum(biased_weight / bias_norm) + tail_bias / bias_norm),
                1 / bias_norm,
                low_mass,
                high_mass,
                low_mass * params.minimum_mass / density,
            )
        )
    return NormalizedHaloTable(
        *map(
            jnp.asarray,
            (scale, mass, np.stack(mass_rows), np.stack(bias_rows), tail_rows, tail_bias_rows),
        )
    ), tuple(diagnostics)


def average_normalized_hmf(tables: Sequence[NormalizedHaloTable]) -> NormalizedHaloTable:
    """Average abundances and bias-weighted abundances, not bias or P2h.

    Equal-volume realizations must share the same mass and time quadratures.
    Averaging the response before squaring avoids the finite-ensemble variance
    term introduced by averaging per-realization two-halo powers.
    """

    if not tables:
        raise ValueError("cannot average an empty HMF ensemble")
    first = tables[0]
    for table in tables[1:]:
        if not np.array_equal(table.mass_msun_h, first.mass_msun_h) or not np.array_equal(
            table.scale_factor, first.scale_factor
        ):
            raise ValueError("ensemble HMF quadratures must match")
    return NormalizedHaloTable(
        first.scale_factor,
        first.mass_msun_h,
        *(
            jnp.mean(jnp.stack([table[index] for table in tables]), axis=0)
            for index in range(2, len(first))
        ),
    )
