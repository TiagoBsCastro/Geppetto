"""Native-mass HMF measurement and independent-LF HOD calibration on the host."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from geppetto.galaxies.adapters import (
    HODPY_REVISION,
    IndependentLuminosityFunction,
    NativeHaloInput,
    PinocchioDistances,
    file_sha256,
    sky_coordinates,
)
from geppetto.galaxies.models import HODShapeParams, ThresholdParams, occupation


@dataclass(frozen=True)
class CalibrationConfig:
    """Explicit numerical/domain choices; magnitudes are ^0.1 M_r - 5log10(h)."""

    z_min: float = .04
    z_max: float = .33
    redshift_step: float = .01
    aperture_deg: float = 65.
    min_particles: int = 32
    magnitude_bright: float = -25.
    magnitude_faint: float = -21.5
    magnitude_step: float = .01
    mass_bin_width_dex: float = .01
    nesting_tolerance: float = 1.e-10
    shape: HODShapeParams = HODShapeParams()


@dataclass(frozen=True)
class CalibratedHOD:
    """Piecewise-redshift threshold model plus native-mass quadrature diagnostics."""

    magnitudes: np.ndarray
    redshift_edges: np.ndarray
    thresholds: np.ndarray
    log_shifts: np.ndarray
    target_cumulative: np.ndarray
    prediction: np.ndarray
    below_cut_prediction: np.ndarray
    support_min_mass: np.ndarray
    cache_key: str
    metadata: dict

    def cell(self, redshift):
        index = np.searchsorted(self.redshift_edges, redshift, side="right")-1
        if np.any(index < 0) or np.any(index >= len(self.redshift_edges)-1):
            raise ValueError("Halo redshift outside calibrated HOD cells")
        return index


def threshold_shapes(reference_magnitude, shape: HODShapeParams) -> ThresholdParams:
    """Translate hodpy's luminosity-dependent HOD shapes, without MXXL shifts.

    Inverse M(L) interpolation uses its Eq.11-type functional form, and the
    remaining shape functions are identical to hodpy's HOD_BGS. Result mass
    scales are subsequently calibrated as scales of M_PIN, not M200m.
    """

    def inverse_mass(ls, mt, am):
        grid = np.linspace(9., 20., 22001)
        log_lum = np.log10(ls)+am*(grid-np.log10(mt))+(1-mt/10.**grid)/np.log(10)
        mag = 4.76-2.5*log_lum
        if np.any(reference_magnitude < mag[-1]) or np.any(reference_magnitude > mag[0]):
            raise ValueError("HOD mass-luminosity inversion outside mass grid")
        return np.interp(reference_magnitude, mag[::-1], grid[::-1])

    lum = (4.76-reference_magnitude)/2.5
    return ThresholdParams(
        inverse_mass(shape.mmin_ls, shape.mmin_mt, shape.mmin_am),
        shape.m0_a*lum+shape.m0_b,
        inverse_mass(shape.m1_ls, shape.m1_mt, shape.m1_am),
        shape.sigma_faint+(shape.sigma_bright-shape.sigma_faint)/(1+np.exp((reference_magnitude+shape.sigma_step)*shape.sigma_width)),
        np.log10(shape.alpha_c+(shape.alpha_a*lum)**shape.alpha_b),
    )


@jax.jit
def _calibrate(log_mass, weight, target, threshold):
    """Solve every threshold's common mass shift with a monotone bracket."""
    params = ThresholdParams(*(value[:, None] for value in threshold))

    def predict(shift):
        central, satellite = occupation(log_mass[None, :], params, shift[:, None])
        return (central+satellite) @ weight

    low, high = jnp.full(target.shape, -6.), jnp.full(target.shape, 6.)
    initial_low, initial_high = predict(low), predict(high)

    def step(_, bounds):
        lo, hi = bounds
        mid = .5*(lo+hi)
        too_many = predict(mid) > target
        return jnp.where(too_many, mid, lo), jnp.where(too_many, hi, mid)

    low, high = jax.lax.fori_loop(0, 64, step, (low, high))
    shift = .5*(low+high)
    return shift, predict(shift), initial_low, initial_high


@jax.jit
def evaluate_occupation(log_mass, threshold_array, shifts):
    """Host-callable compiled cumulative counts, output shape (halo, threshold)."""
    params = ThresholdParams(*(threshold_array[:, i][None, :] for i in range(5)))
    return occupation(log_mass[:, None], params, shifts[None, :])


def native_hmf_quadrature(mass, selection, log_edges, volume):
    """Count/volume weights for a measured dN/dlog10(M_PIN) quadrature.

    Each occupied log-mass bin is represented by its mean log mass. Empty bins
    have zero weight. No analytic MXXL HMF or change of halo mass definition
    is applied. The small bin width is an explicit convergence parameter.
    """
    log_edges = np.asarray(log_edges, dtype=float)
    if volume <= 0 or np.any(np.diff(log_edges) <= 0):
        raise ValueError("Native HMF needs positive volume and increasing mass-bin edges")
    logs = np.log10(mass[selection])
    if np.any(~np.isfinite(logs)) or np.any((logs < log_edges[0]) | (logs > log_edges[-1])):
        raise ValueError("Native HMF mass edges must include every selected halo")
    counts = np.histogram(logs, bins=log_edges)[0]
    total_logs = np.histogram(logs, bins=log_edges, weights=logs)[0]
    centers = .5*(log_edges[1:]+log_edges[:-1])
    centers = np.divide(total_logs, counts, out=centers, where=counts > 0)
    return centers, counts.astype(float)/volume


def calibrate_hod(
    halos: NativeHaloInput, distances: PinocchioDistances, target: IndependentLuminosityFunction,
    config: CalibrationConfig, *, input_path: str | Path, distance_path: str | Path,
    cache_dir: str | Path,
) -> CalibratedHOD:
    """Fit common native-mass shifts to a fixed, volume-averaged target LF.

    Calibration is conditional on the measured PLC HMF in thin redshift cells.
    This enforces abundance, not an independent prediction of cosmic variance.
    No measured galaxy LF enters the fit. The cache is content addressed.
    """
    if not jax.config.x64_enabled:
        raise ValueError("Native HOD calibration requires JAX_ENABLE_X64=true")
    if not (0 <= config.z_min < config.z_max and 0 < config.aperture_deg <= 180
            and config.min_particles >= int(halos.metadata["parameters"]["MinHaloMass"][0])
            and config.magnitude_bright < config.magnitude_faint and config.magnitude_step > 0
            and config.redshift_step > 0 and config.mass_bin_width_dex > 0):
        raise ValueError("Invalid HOD calibration domain")
    metadata = dict(schema_version=1, config=asdict(config), hodpy_revision=HODPY_REVISION,
                    native_input_sha256=file_sha256(input_path), distance_sha256=file_sha256(distance_path),
                    upstream_hashes=target.input_hashes, target_parameters=target.parameters,
                    code_hashes={name: file_sha256(Path(__file__).parent / name)
                                 for name in ("adapters.py", "models.py", "calibration.py")})
    cache_key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    cache = Path(cache_dir)/f"hod_{cache_key}.npz"
    if cache.exists():
        with np.load(cache) as saved:
            if str(saved["cache_key"]) != cache_key:
                raise ValueError("HOD cache content key mismatch")
            return CalibratedHOD(*(saved[name].copy() for name in (
                "magnitudes", "redshift_edges", "thresholds", "log_shifts", "target_cumulative",
                "prediction", "below_cut_prediction", "support_min_mass")), cache_key,
                json.loads(str(saved["metadata_json"])))
    mags = np.linspace(config.magnitude_bright, config.magnitude_faint,
                       round((config.magnitude_faint-config.magnitude_bright)/config.magnitude_step)+1)
    zedges = np.linspace(config.z_min, config.z_max, round((config.z_max-config.z_min)/config.redshift_step)+1)
    values = halos.values
    _, latitude = sky_coordinates(values["position"], halos.basis)
    angular = latitude >= 90-config.aperture_deg
    resolved = values["particle_count"] >= config.min_particles
    log_cut = np.log10(config.min_particles*halos.particle_mass)
    if not np.any(resolved):
        raise ValueError("No native halos above the analysis particle cut")
    # Native float32 masses can lie just below the ideal integer-particle mass.
    log_lower = np.nextafter(np.log10(values["M_PIN"][resolved].min()), -np.inf)
    log_edges = np.arange(log_lower, np.log10(values["M_PIN"].max())+2*config.mass_bin_width_dex, config.mass_bin_width_dex)
    below_lower = min(np.log10(values["M_PIN"].min())-1.e-7, log_cut-config.mass_bin_width_dex)
    below_edges = np.linspace(below_lower, log_cut, 100)
    cells = []
    min_central_difference, min_satellite_difference = 0., 0.
    for index, (lo, hi) in enumerate(zip(zedges[:-1], zedges[1:], strict=True)):
        cell = angular & (values["z_cos"] >= lo) & (values["z_cos"] < hi)
        volume = distances.volume(lo, hi, config.aperture_deg)
        mass_grid, weight = native_hmf_quadrature(values["M_PIN"], cell & resolved, log_edges, volume)
        if np.sum(weight > 0) < 10:
            raise ValueError(f"Too few native mass bins in cell {lo}-{hi}")
        shapes = threshold_shapes(target.reference_magnitude(mags, .5*(lo+hi)), config.shape)
        target_average = distances.volume_average(lambda z: target.cumulative(mags[:, None], z), lo, hi)
        shift, predicted, low, high = [np.asarray(x) for x in _calibrate(
            jnp.asarray(mass_grid), jnp.asarray(weight), jnp.asarray(target_average),
            ThresholdParams(*(jnp.asarray(value) for value in shapes)),
        )]
        if np.any(target_average > low) or np.any(target_average < high):
            bad = np.flatnonzero((target_average > low) | (target_average < high))[0]
            raise ValueError(f"Could not bracket target LF in redshift cell {index}: M_r={mags[bad]}, target={target_average[bad]}, bracket=({low[bad]}, {high[bad]})")
        thresholds = np.stack(shapes, axis=-1)
        # Cover the full input mass range, including halos below the analysis cut.
        audit_masses = np.linspace(np.log10(values["M_PIN"].min()), np.log10(values["M_PIN"].max())+.01, 1401)
        central, satellite = [np.asarray(x) for x in evaluate_occupation(jnp.asarray(audit_masses), jnp.asarray(thresholds), jnp.asarray(shift))]
        dc, ds = np.diff(central, axis=1), np.diff(satellite, axis=1)
        min_central_difference = min(min_central_difference, float(dc.min()))
        min_satellite_difference = min(min_satellite_difference, float(ds.min()))
        if min(dc.min(), ds.min()) < -config.nesting_tolerance:
            where = np.unravel_index(np.argmin(np.minimum(dc, ds)), dc.shape)
            raise ValueError(f"Non-nested luminosity thresholds at z={lo}-{hi}, logM={audit_masses[where[0]]:.4f}, M_r={mags[where[1]]:.3f}: dNcen={dc.min():.4g}, dNsat={ds.min():.4g}")
        lower_masses, lower_weight = native_hmf_quadrature(values["M_PIN"], cell & ~resolved, below_edges, volume)
        lower_c, lower_s = evaluate_occupation(jnp.asarray(lower_masses), jnp.asarray(thresholds), jnp.asarray(shift))
        below = lower_weight @ np.asarray(lower_c+lower_s)
        support = 10**(shapes.log_mmin+shift-np.sqrt(6)*shapes.sigma)
        cells.append((thresholds, shift, target_average, predicted, below, support))
        print(f"[galaxy HOD] cell {index+1}/{len(zedges)-1} z={lo:.3f}-{hi:.3f}, halos={np.sum(cell & resolved)}", flush=True)
    metadata["nesting_minimum_differences"] = [min_central_difference, min_satellite_difference]
    table = CalibratedHOD(mags, zedges, *(np.stack([row[i] for row in cells]) for i in range(6)), cache_key, metadata)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **{key: getattr(table, key) for key in (
        "magnitudes", "redshift_edges", "thresholds", "log_shifts", "target_cumulative",
        "prediction", "below_cut_prediction", "support_min_mass", "cache_key")},
        metadata_json=json.dumps(metadata, sort_keys=True))
    return table
