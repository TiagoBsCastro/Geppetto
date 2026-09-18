#!/usr/bin/env python
"""Read-only diagnostics of an existing, frozen PINOCCHIO-hodpy catalogue.

No fit, generation, cache update, or model-parameter change is performed.
The --plot-only path regenerates figures from portable CSV/JSON products,
without the halo input, galaxy catalogue, JAX or optional upstream checkout.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import shutil
import sys
import textwrap
from pathlib import Path

import numpy as np
from scipy.special import ndtr
from scipy.stats import chi2, kstest, norm

ROOT = Path(__file__).resolve().parents[1]
COLORS = ["#2166ac", "#b35806", "#188977"]
MAG_LABEL = r"$^{0.1}M_r-5\log_{10}h$ [AB mag]"
MASS_LABEL = r"$\log_{10}(M_{\rm PIN}/[M_\odot/h])$"


def digest(path):
    with Path(path).open("rb") as handle:
        result = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def write_rows(path, rows):
    if not rows:
        raise ValueError(f"No diagnostic rows for {path}")
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path):
    def convert(value):
        if value == "":
            return None
        if value in ("True", "False"):
            return value == "True"
        try:
            return float(value)
        except ValueError:
            return value
    with Path(path).open() as handle:
        return [{key: convert(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def bin_index(values, edges):
    """Half-open bins; unlike clipping, out-of-domain points stay excluded."""
    result = np.searchsorted(edges, values, side="right")-1
    return np.where((values >= edges[0]) & (values < edges[-1]), result, -1)


def selected_bins(values, edges, weights=None):
    index = bin_index(values, edges)
    valid = index >= 0
    return np.bincount(index[valid], weights=None if weights is None else weights[valid],
                       minlength=len(edges)-1)


def mixture_components(model, magnitude, redshift, central):
    """Independent evaluation of the upstream conditional two-Gaussian model."""
    red_sat = model.probability_red_satellite(magnitude, redshift)
    blue_sat = 1-red_sat
    blue_central = blue_sat+(model.fraction_blue(magnitude, redshift)-blue_sat)/model.fraction_central(magnitude, redshift)
    probability = np.clip(np.where(central, 1-blue_central, red_sat), 0., 1.)
    return probability, model.red_mean(magnitude, redshift), model.red_rms(magnitude, redshift), model.blue_mean(magnitude, redshift), model.blue_rms(magnitude, redshift)


def mixture_cdf(values, components):
    p, mr, sr, mb, sb = components
    return p*ndtr((values-mr)/sr)+(1-p)*ndtr((values-mb)/sb)


def probability_histogram(edges, cdf, count, chunk=4096):
    """Conditional histogram expectation and Poisson-binomial variance.

    cdf(edges[:,None], slice) returns one CDF per independent object, never
    an observed histogram used as its own reference. The total count is fixed.
    """
    expected = np.zeros(len(edges)-1)
    variance = np.zeros_like(expected)
    for first in range(0, count, chunk):
        probability = np.diff(cdf(np.asarray(edges)[:, None], slice(first, first+chunk)), axis=0)
        if np.any(probability < -1.e-10) or np.any(probability > 1+1.e-10):
            raise ValueError("Invalid conditional histogram probabilities")
        probability = np.clip(probability, 0., 1.)
        expected += np.sum(probability, axis=1)
        variance += np.sum(probability*(1-probability), axis=1)
    return expected, variance


def piecewise_colour_cdf(edges, colour_knots, apparent_knots, components):
    """Exact CDF of a clipped, piecewise-linear colour-to-magnitude map.

    The frozen hodpy k-correction is linear between its colour knots at each
    z<0.5 and constant beyond its end knots. Integrate each Gaussian mixture
    over the corresponding colour intervals, including both clipped tails.
    """
    edges = np.asarray(edges)[:, None]
    result = mixture_cdf(colour_knots[0], components)[None, :]*(edges > apparent_knots[0])
    result += (1-mixture_cdf(colour_knots[-1], components))[None, :]*(edges > apparent_knots[-1])
    for i, (lo, hi) in enumerate(zip(colour_knots[:-1], colour_knots[1:], strict=True)):
        left, right = apparent_knots[i], apparent_knots[i+1]
        delta = right-left
        constant = abs(delta) < 1.e-13
        colour_edge = lo+(edges-left)*(hi-lo)/np.where(constant, 1., delta)
        clipped = np.clip(colour_edge, lo, hi)
        low_cdf, high_cdf = mixture_cdf(lo, components), mixture_cdf(hi, components)
        partial = mixture_cdf(clipped, components)
        integral = np.where(delta > 0, partial-low_cdf, high_cdf-partial)
        integral = np.where(constant, (high_cdf-low_cdf)*(edges > left), integral)
        result += integral
    return np.clip(result, 0., 1.)


def histogram_rows(edges, values, expected, variance, **labels):
    measured = selected_bins(values, edges)
    rows = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        error = np.sqrt(variance[i])
        rows.append(dict(**labels, lower=lo, upper=hi, count=int(measured[i]),
                         expected=float(expected[i]), sigma=float(error),
                         ratio=float(measured[i]/expected[i]) if expected[i] > 0 else None,
                         pull=float((measured[i]-expected[i])/error) if error > 0 else None))
    return rows


def read_frozen(settings):
    """Load existing products and reject source/input drift; never recalibrate."""
    os.environ.setdefault("JAX_ENABLE_X64", "true")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import h5py
    import jax

    from geppetto.galaxies.adapters import (
        IndependentLuminosityFunction,
        PinocchioDistances,
        load_native_halos,
    )
    from geppetto.galaxies.calibration import CalibratedHOD

    jax.config.update("jax_enable_x64", True)
    path = Path(settings["catalogue"])
    with h5py.File(path) as handle:
        metadata = json.loads(handle.attrs["metadata_json"])
        galaxies = {key: value[:] for key, value in handle["galaxies"].items()}
    frozen = metadata["config"]
    cache = Path(frozen["output_dir"])/"cache"/f"hod_{metadata['calibration_key']}.npz"
    with np.load(cache) as handle:
        table_meta = json.loads(str(handle["metadata_json"]))
        if str(handle["cache_key"]) != metadata["calibration_key"]:
            raise ValueError("Frozen calibration key mismatch")
        table = CalibratedHOD(*(handle[key].copy() for key in (
            "magnitudes", "redshift_edges", "thresholds", "log_shifts", "target_cumulative",
            "prediction", "below_cut_prediction", "support_min_mass")), metadata["calibration_key"], table_meta)
    hashes = {str(path): digest(path), str(cache): digest(cache)}
    for field, key in (("halo_input", "native_input_sha256"), ("cosmology_table", "distance_sha256")):
        hashes[frozen[field]] = digest(frozen[field])
        if hashes[frozen[field]] != table_meta[key]:
            raise ValueError(f"Frozen {field} changed")
    for name, expected in table_meta["code_hashes"].items():
        source = ROOT/"src/geppetto/galaxies"/name
        if digest(source) != expected:
            raise ValueError(f"Frozen model source changed: {source}")
        hashes[str(source)] = expected
    target = IndependentLuminosityFunction(frozen["hodpy_root"])
    if target.input_hashes != metadata["target_input_hashes"]:
        raise ValueError("Frozen upstream LF/colour inputs changed")
    for name, expected in target.input_hashes.items():
        hashes[str(Path(frozen["hodpy_root"])/name)] = expected
    pipeline = ROOT/"src/geppetto/galaxies/pipeline.py"
    if digest(pipeline) != metadata["pipeline_sha256"]:
        raise ValueError("Frozen catalogue-generation source changed")
    hashes[str(pipeline)] = metadata["pipeline_sha256"]
    halos = load_native_halos(frozen["halo_input"])
    h = float(halos.metadata["parameters"]["Hubble100"][0])
    distances = PinocchioDistances(frozen["cosmology_table"], h)
    return galaxies, halos, distances, target, table, metadata, hashes


def predict_halos(halos, table, settings):
    import jax.numpy as jnp

    from geppetto.galaxies.adapters import sky_coordinates
    from geppetto.galaxies.calibration import evaluate_occupation

    lon, lat = sky_coordinates(halos.values["position"], halos.basis)
    z = halos.values["z_cos"]
    mask = (lat >= 90-settings["aperture_deg"]) & (z >= settings["redshift_edges"][0]) & (z < settings["redshift_edges"][-1])
    mask &= halos.values["particle_count"] >= 32
    index = np.flatnonzero(mask)
    mags = np.asarray(settings["luminosity_edges"])
    edge_index = np.array([np.argmin(abs(table.magnitudes-mag)) for mag in mags])
    if not np.allclose(mags, table.magnitudes[edge_index], rtol=0, atol=1.e-8):
        raise ValueError("Audit magnitude bins must align with frozen threshold grid")
    central, satellite = np.zeros((len(index), len(mags))), np.zeros((len(index), len(mags)))
    cell = table.cell(z[index])
    for icell in np.unique(cell):
        selected = np.flatnonzero(cell == icell)
        for first in range(0, len(selected), 8192):
            rows = selected[first:first+8192]
            mass = halos.values["M_PIN"][index[rows]]
            count = len(mass)
            log_mass = np.pad(np.log10(mass), (0, 8192-count), mode="edge")
            c, s = evaluate_occupation(jnp.asarray(log_mass), jnp.asarray(table.thresholds[icell, edge_index]),
                                       jnp.asarray(table.log_shifts[icell, edge_index]))
            central[rows], satellite[rows] = np.asarray(c)[:count], np.asarray(s)[:count]
    return dict(index=index, central=central, satellite=satellite, lon=lon, lat=lat)


def completeness_rows(table, halos, settings):
    rows = []
    for iz, (zlo, zhi) in enumerate(zip(settings["redshift_edges"][:-1], settings["redshift_edges"][1:], strict=True)):
        cell = (table.redshift_edges[:-1] >= zlo-1.e-9) & (table.redshift_edges[1:] <= zhi+1.e-9)
        if not np.any(cell):
            raise ValueError("No complete calibration cells in audit redshift bin")
        for lo, hi in zip(settings["luminosity_edges"][:-1], settings["luminosity_edges"][1:], strict=True):
            a, b = [np.argmin(abs(table.magnitudes-m)) for m in (lo, hi)]
            kept = table.prediction[cell, b]-table.prediction[cell, a]
            missing = table.below_cut_prediction[cell, b]-table.below_cut_prediction[cell, a]
            fraction = float(np.max(missing/(missing+kept)))
            support = float(np.min(table.support_min_mass[cell, b]/halos.particle_mass))
            rows.append(dict(zbin=iz, lower=lo, upper=hi, missing_fraction=fraction,
                             minimum_support_particles=support,
                             complete=bool(fraction < .01 and support >= 10 and hi <= settings["complete_magnitude_limit"]+1.e-8)))
    return rows


def count_products(data, halos, distances, target, table, prediction, settings):
    """HMF, LF, occupation, redshift and sky expectations on the actual halos."""
    from geppetto.galaxies.validation import lf_validation

    p, s = prediction["central"], prediction["satellite"]
    hi = np.flatnonzero(np.isclose(settings["luminosity_edges"], settings["complete_magnitude_limit"]))[0]
    pc, rate = p[:, hi], s[:, hi]
    indices = prediction["index"]
    z = halos.values["z_cos"][indices]
    logmass = np.log10(halos.values["M_PIN"][indices])
    n_halos = len(halos.values["M_PIN"])
    intrinsic = data["absolute_magnitude_r"] < settings["complete_magnitude_limit"]
    central_count = np.bincount(data["host_index"][intrinsic & data["is_central"]], minlength=n_halos)[indices]
    satellite_count = np.bincount(data["host_index"][intrinsic & ~data["is_central"]], minlength=n_halos)[indices]
    mass_edges = np.linspace(11.85, 15., 22)
    hmf, occupation = [], []
    for iz, (zlo, zhi) in enumerate(zip(settings["redshift_edges"][:-1], settings["redshift_edges"][1:], strict=True)):
        chosen = (z >= zlo) & (z < zhi)
        native = (halos.values["z_cos"] >= zlo) & (halos.values["z_cos"] < zhi)
        native &= prediction["lat"] >= 90-settings["aperture_deg"]
        native_counts = selected_bins(np.log10(halos.values["M_PIN"][native]), mass_edges)
        resolved_count = selected_bins(logmass[chosen], mass_edges)
        occupied = selected_bins(logmass[chosen], mass_edges, (central_count[chosen]+satellite_count[chosen] > 0))
        prob_occupied = 1-(1-pc[chosen])*np.exp(-rate[chosen])
        exp_occupied = selected_bins(logmass[chosen], mass_edges, prob_occupied)
        var_occupied = selected_bins(logmass[chosen], mass_edges, prob_occupied*(1-prob_occupied))
        volume = distances.volume(zlo, zhi, settings["aperture_deg"])
        for j, (lo, high) in enumerate(zip(mass_edges[:-1], mass_edges[1:], strict=True)):
            hmf.append(dict(zbin=iz, lower=lo, upper=high, n_native=int(native_counts[j]),
                            n_resolved=int(resolved_count[j]), n_occupied=int(occupied[j]),
                            expected_occupied=exp_occupied[j], sigma_occupied=np.sqrt(var_occupied[j]), volume=volume))
        for name, counts, expectation, variance in (
            ("central", central_count, pc, pc*(1-pc)), ("satellite", satellite_count, rate, rate)):
            observed = selected_bins(logmass[chosen], mass_edges, counts[chosen])
            expected = selected_bins(logmass[chosen], mass_edges, expectation[chosen])
            var = selected_bins(logmass[chosen], mass_edges, variance[chosen])
            for j, (lo, high) in enumerate(zip(mass_edges[:-1], mass_edges[1:], strict=True)):
                occupation.append(dict(zbin=iz, status=name, lower=lo, upper=high,
                                       n_halos=int(resolved_count[j]), count=int(observed[j]), expected=expected[j], sigma=np.sqrt(var[j])))
    audit_config = dict(table.metadata["config"])
    validation_config = dict(calibration=audit_config, validation=dict(
        magnitude_edges=settings["luminosity_edges"], redshift_slices=list(zip(settings["redshift_edges"][:-1], settings["redshift_edges"][1:], strict=True)),
        jackknife_azimuth_regions=8, jackknife_radial_regions=4, cumulative_prediction_rtol=.02,
        mock_rtol_floor=.05, mock_sigma_multiplier=3., minimum_bin_count=settings["minimum_lf_count"]),
    )
    lf, cumulative = lf_validation(data, halos, distances, target, table, validation_config)
    # Native .01 redshift cells align with the frozen model and audit boundaries.
    z_edges = table.redshift_edges[(table.redshift_edges >= settings["redshift_edges"][0]-1.e-9)
                                  & (table.redshift_edges <= settings["redshift_edges"][-1]+1.e-9)]
    observed_selection = intrinsic & (data["dec_deg"] >= 90-settings["aperture_deg"]) & data["selected_r_limit"]
    measured = selected_bins(data["z_cos"][observed_selection], z_edges)
    predicted = selected_bins(z, z_edges, pc+rate)
    variance = selected_bins(z, z_edges, pc*(1-pc)+rate)
    nz = []
    for j, (lo, high) in enumerate(zip(z_edges[:-1], z_edges[1:], strict=True)):
        volume = distances.volume(lo, high, settings["aperture_deg"])
        truth = float(distances.volume_average(lambda zs: target.cumulative(settings["complete_magnitude_limit"], zs), lo, high))*volume
        nz.append(dict(lower=lo, upper=high, count=int(measured[j]), expected=predicted[j], sigma=np.sqrt(variance[j]),
                       independent_lf_count=truth, volume=volume))
    # Equal-solid-angle pixels: azimuth by q=(1-cos angular distance)/(1-cos aperture).
    phi_edges, area_edges = np.linspace(0., 360., 25), np.linspace(0., 1., 13)
    denominator = 1-np.cos(np.deg2rad(settings["aperture_deg"]))
    harea = (1-np.sin(np.deg2rad(prediction["lat"][indices])))/denominator
    garea = (1-np.sin(np.deg2rad(data["dec_deg"])))/denominator
    sky = []
    for iz, (lo, high) in enumerate(zip(settings["redshift_edges"][:-1], settings["redshift_edges"][1:], strict=True)):
        hmask = (z >= lo) & (z < high)
        gmask = observed_selection & (data["z_cos"] >= lo) & (data["z_cos"] < high)
        observed = np.histogram2d(data["ra_deg"][gmask], garea[gmask], bins=(phi_edges, area_edges))[0]
        expected = np.histogram2d(prediction["lon"][indices][hmask], harea[hmask], bins=(phi_edges, area_edges), weights=(pc+rate)[hmask])[0]
        var = np.histogram2d(prediction["lon"][indices][hmask], harea[hmask], bins=(phi_edges, area_edges), weights=(pc*(1-pc)+rate)[hmask])[0]
        for i in range(len(phi_edges)-1):
            for j in range(len(area_edges)-1):
                sky.append(dict(zbin=iz, phi_lower=phi_edges[i], phi_upper=phi_edges[i+1], area_lower=area_edges[j], area_upper=area_edges[j+1],
                                count=int(observed[i, j]), expected=expected[i, j], sigma=np.sqrt(var[i, j]),
                                pull=float((observed[i, j]-expected[i, j])/np.sqrt(var[i, j])) if var[i, j] >= 10 else None))
    return dict(hmf=hmf, occupation=occupation, luminosity_function=lf, cumulative_lf=cumulative, redshift_counts=nz, sky=sky)


def photometry_products(data, distances, metadata, settings):
    from geppetto.galaxies.adapters import hodpy_modules

    upstream = hodpy_modules(metadata["config"]["hodpy_root"])
    model = upstream["colour"].Colour()
    kcorr = upstream["k_correction"].GAMA_KCorrection(distances)
    redshift_edges = settings["redshift_edges"]
    angular = data["dec_deg"] >= 90-settings["aperture_deg"]
    components = mixture_components(model, data["absolute_magnitude_r"], data["z_cos"], data["is_central"])
    p, mr, sr, mb, sb = components
    expectation = p*mr+(1-p)*mb
    variance = p*(sr**2+mr**2)+(1-p)*(sb**2+mb**2)-expectation**2
    app_edges, colour_edges = np.linspace(12., 21., 31), np.linspace(.2, 1.4, 31)
    mag_edges = np.round(np.arange(-23.4, -21.5+.05, .1), 4)
    app_rows, cmr, distribution, moments = [], [], [], []
    rng = np.random.default_rng(settings["diagnostic_seed"]+19)
    for iz, (zlo, zhi) in enumerate(zip(redshift_edges[:-1], redshift_edges[1:], strict=True)):
        base = angular & (data["z_cos"] >= zlo) & (data["z_cos"] < zhi)
        index = np.flatnonzero(base)
        expected, var = np.zeros(len(app_edges)-1), np.zeros(len(app_edges)-1)
        for first in range(0, len(index), 4096):
            part = index[first:first+4096]
            knots = kcorr.colour_med
            apparent = np.stack([kcorr.apparent_magnitude(data["absolute_magnitude_r"][part], data["z_cos"][part], np.full(len(part), c)) for c in knots])
            cdf = piecewise_colour_cdf(app_edges, knots, apparent, tuple(value[part] for value in components))
            probability = np.clip(np.diff(cdf, axis=0), 0., 1.)
            expected += probability.sum(axis=1)
            var += (probability*(1-probability)).sum(axis=1)
        app_rows.extend(histogram_rows(app_edges, data["apparent_magnitude_r"][base], expected, var, zbin=iz, n_sample=len(index)))
        # Colour diagnostics precede flux selection, avoiding colour truncation.
        selected = base
        histogram = np.histogram2d(data["absolute_magnitude_r"][selected], data["g_r_rest_0p1"][selected], bins=(mag_edges, colour_edges))[0]
        for im, (lo, high) in enumerate(zip(mag_edges[:-1], mag_edges[1:], strict=True)):
            group = selected & (data["absolute_magnitude_r"] >= lo) & (data["absolute_magnitude_r"] < high)
            n = int(group.sum())
            for ic, (clo, chi) in enumerate(zip(colour_edges[:-1], colour_edges[1:], strict=True)):
                cmr.append(dict(zbin=iz, magnitude_lower=lo, magnitude_upper=high, colour_lower=clo, colour_upper=chi,
                                count=int(histogram[im, ic]), n_magnitude_bin=n,
                                mean_measured=float(np.mean(data["g_r_rest_0p1"][group])) if n else None,
                                mean_expected=float(np.mean(expectation[group])) if n else None,
                                sigma_mean=float(np.sqrt(np.sum(variance[group]))/n) if n else None))
        complete = selected & (data["absolute_magnitude_r"] < settings["complete_magnitude_limit"])
        for ib, (mlo, mhi) in enumerate(zip(settings["log10_host_mass_edges"][:-1], settings["log10_host_mass_edges"][1:], strict=True)):
            group = complete & (np.log10(data["M_PIN"]) >= mlo) & (np.log10(data["M_PIN"]) < mhi)
            index = np.flatnonzero(group)
            comp = tuple(value[index] for value in components)
            def cdf(edges, part, comp=comp):
                return mixture_cdf(edges, tuple(value[part] for value in comp))
            expected, var = probability_histogram(colour_edges, cdf, len(index))
            distribution.extend(histogram_rows(colour_edges, data["g_r_rest_0p1"][group], expected, var,
                                               zbin=iz, massbin=ib, n_sample=len(index)))
            for central in (True, False):
                status = group & (data["is_central"] == central)
                n = int(status.sum())
                if n == 0:
                    continue
                measured_red, expected_red = float(np.mean(data["is_red_draw"][status])), float(np.mean(p[status]))
                error = float(np.sqrt(np.sum(p[status]*(1-p[status])))/n)
                count_variance = (error*n)**2
                deviation = abs((measured_red-expected_red)*n)
                if count_variance == 0:
                    pvalue = float(deviation < 1.e-10)
                elif count_variance >= 10:
                    pvalue = float(2*norm.sf(deviation/np.sqrt(count_variance)))
                else:
                    probability = p[status]
                    fixed = np.count_nonzero(probability == 1)
                    probability = probability[(probability > 0) & (probability < 1)]
                    exceed = 0
                    for _ in range(200):
                        trial = fixed+np.sum(rng.random((250, len(probability))) < probability, axis=1)
                        exceed += np.count_nonzero(abs(trial-expected_red*n) >= deviation-1.e-10)
                    pvalue = (exceed+1)/50001
                moments.append(dict(zbin=iz, massbin=ib, status="central" if central else "satellite", component="mixture",
                                    statistic="red_fraction", n=n, measured=measured_red, expected=expected_red, sigma=error,
                                    p_value=pvalue))
                for red in (True, False):
                    use = status & (data["is_red_draw"] == red)
                    n = int(use.sum())
                    if n < 2:
                        continue
                    standardized = ((data["g_r_rest_0p1"][use]-(mr if red else mb)[use])/(sr if red else sb)[use])
                    mean, sample_var = float(np.mean(standardized)), float(np.var(standardized, ddof=1))
                    q = (n-1)*sample_var
                    for statistic, measured, truth, sigma, pv in (
                        ("standardized_mean", mean, 0., 1/np.sqrt(n), 2*norm.sf(abs(mean)*np.sqrt(n))),
                        ("standardized_variance", sample_var, 1., np.sqrt(2/(n-1)), min(1., 2*min(chi2.cdf(q, n-1), chi2.sf(q, n-1))))):
                        moments.append(dict(zbin=iz, massbin=ib, status="central" if central else "satellite", component="red" if red else "blue",
                                            statistic=statistic, n=n, measured=measured, expected=truth, sigma=sigma, p_value=float(pv)))
    return dict(apparent_magnitude=app_rows, colour_magnitude=cmr, colour_distribution=distribution, colour_moments=moments)


def placement_products(data, halos, prediction, metadata, settings):
    host = data["host_index"]
    z = data["host_z_cos"]
    log_mass = np.log10(data["M_PIN"])
    selected = (prediction["lat"][host] >= 90-settings["aperture_deg"]) & (z >= settings["redshift_edges"][0]) & (z < settings["redshift_edges"][-1])
    selected &= (~data["is_central"]) & (data["absolute_magnitude_r"] < settings["complete_magnitude_limit"])
    offset = data["position"]-halos.values["position"][host]
    distance = np.linalg.norm(offset, axis=1)
    params = metadata["satellite_parameters"]
    expected_radius = params["radius_factor"]*(3*data["M_PIN"]/(4*np.pi*params["overdensity_mean"]*halos.mean_density))**(1/3)
    expected_c = params["concentration_amplitude"]*(data["M_PIN"]/params["concentration_mass_pivot"])**params["concentration_mass_slope"]*(1+z)**params["concentration_redshift_slope"]
    expected_sigma = params["velocity_factor"]*np.sqrt(params["gravitational_constant"]*data["M_PIN"]*(1+z)/(2*expected_radius))
    fraction = distance/expected_radius
    line_of_sight = data["position"]/np.linalg.norm(data["position"], axis=1)[:, None]
    velocity_offset = np.sum((data["velocity"]-halos.values["velocity"][host])*line_of_sight, axis=1)
    standardized = velocity_offset/expected_sigma
    velocity_edges = np.linspace(-4., 4., 25)
    radius_rows, velocity_rows, tests = [], [], []
    for dimension, values, edges in (("redshift", z, settings["redshift_edges"]), ("mass", log_mass, settings["log10_host_mass_edges"])):
        for i, (lo, high) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
            group = selected & (values >= lo) & (values < high)
            c, x, v = expected_c[group], fraction[group], standardized[group]
            radial_edges = np.linspace(0., 1., 7 if len(c) < 100 else 16)
            def radial_cdf(r, part, c=c):
                cc = c[part]
                argument = r*cc
                return (np.log1p(argument)-argument/(1+argument))/(np.log1p(cc)-cc/(1+cc))
            expected, variance = probability_histogram(radial_edges, radial_cdf, len(c))
            radius_rows.extend(histogram_rows(radial_edges, x, expected, variance, dimension=dimension, group=i, n_sample=len(c)))
            probability = np.diff(ndtr(velocity_edges))
            velocity_rows.extend(histogram_rows(velocity_edges, v, len(c)*probability, len(c)*probability*(1-probability),
                                                dimension=dimension, group=i, n_sample=len(c)))
            uniform = (np.log1p(c*x)-c*x/(1+c*x))/(np.log1p(c)-c/(1+c))
            for name, result in (("radial", kstest(uniform, "uniform")), ("velocity", kstest(v, "norm"))):
                tests.append(dict(test=name, dimension=dimension, group=i, n=len(c), statistic=float(result.statistic), p_value=float(result.pvalue)))
    total_vlos = np.sum(data["velocity"]*line_of_sight, axis=1)
    zclosure = data["z_obs"]-data["z_cos"]-(1+data["z_cos"])*total_vlos/299792.458
    diagnostics = dict(max_radius_relative_error=float(np.max(abs(data["R_sat_comoving"]/expected_radius-1))),
                       max_concentration_relative_error=float(np.max(abs(data["c_sat"]/expected_c-1))),
                       max_velocity_scale_relative_error=float(np.max(abs(data["sigma_sat_km_s"]/expected_sigma-1))),
                       max_observed_redshift_closure=float(np.max(abs(zclosure))),
                       max_satellite_radius_fraction=float(np.max(fraction)),
                       satellite_count=int(selected.sum()), placement_tests=tests)
    return dict(satellite_radii=radius_rows, satellite_velocity=velocity_rows), diagnostics


def angular_vectors(lon, lat):
    phi, theta = np.deg2rad(lon), np.deg2rad(lat)
    return np.column_stack((np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), np.sin(theta)))


def landy_szalay(points, randoms, chord_edges):
    from scipy.spatial import cKDTree
    tree, rtree = cKDTree(points), cKDTree(randoms)
    n, nr = len(points), len(randoms)
    if min(n, nr) < 2:
        raise ValueError("Insufficient clustering sample")
    dd = np.diff(tree.count_neighbors(tree, chord_edges))/(n*(n-1))
    dr = np.diff(tree.count_neighbors(rtree, chord_edges))/(n*nr)
    rr = np.diff(rtree.count_neighbors(rtree, chord_edges))/(nr*(nr-1))
    result = np.divide(dd-2*dr+rr, rr, out=np.full_like(rr, np.nan), where=rr > 0)
    return result, dd, dr, rr


def clustering_products(data, settings):
    rng = np.random.default_rng(settings["diagnostic_seed"])
    angle_edges = np.geomspace(.08, 8., 13)
    chord_edges = 2*np.sin(np.deg2rad(angle_edges)/2)
    aperture_mu = np.cos(np.deg2rad(settings["aperture_deg"]))
    rows = []
    for iz, (zlo, zhi) in enumerate(zip(settings["redshift_edges"][:-1], settings["redshift_edges"][1:], strict=True)):
        for im, (mlo, mhi) in enumerate(zip(settings["log10_host_mass_edges"][:-1], settings["log10_host_mass_edges"][1:], strict=True)):
            mask = ((data["z_cos"] >= zlo) & (data["z_cos"] < zhi) & (data["dec_deg"] >= 90-settings["aperture_deg"])
                    & (data["absolute_magnitude_r"] < settings["complete_magnitude_limit"]) & data["selected_r_limit"]
                    & (np.log10(data["M_PIN"]) >= mlo) & (np.log10(data["M_PIN"]) < mhi))
            index = np.flatnonzero(mask)
            n_available = len(index)
            if len(index) > settings["clustering_max_data"]:
                index = rng.choice(index, settings["clustering_max_data"], replace=False)
            lon, lat = data["ra_deg"][index], data["dec_deg"][index]
            points = angular_vectors(lon, lat)
            nrandom = settings["clustering_random_multiplier"]*len(points)
            rlon = rng.uniform(0, 360., nrandom)
            rlat = np.rad2deg(np.arcsin(rng.uniform(aperture_mu, 1., nrandom)))
            randoms = angular_vectors(rlon, rlat)
            value, dd, dr, rr = landy_szalay(points, randoms, chord_edges)
            nregions = settings["clustering_jackknife_regions"]
            if nregions != 8:
                raise ValueError("This audit uses eight equal-area patches (4 azimuth by 2 polar bands)")
            region = np.floor(lon/90).astype(int)+4*((1-np.sin(np.deg2rad(lat)))/(1-aperture_mu) >= .5)
            rregion = np.floor(rlon/90).astype(int)+4*((1-np.sin(np.deg2rad(rlat)))/(1-aperture_mu) >= .5)
            jackknife = np.array([landy_szalay(points[region != patch], randoms[rregion != patch], chord_edges)[0] for patch in range(nregions)])
            error = np.sqrt((nregions-1)/nregions*np.sum((jackknife-np.mean(jackknife, axis=0))**2, axis=0))
            for j, (lo, high) in enumerate(zip(angle_edges[:-1], angle_edges[1:], strict=True)):
                finite = np.isfinite(value[j]+error[j])
                rows.append(dict(zbin=iz, massbin=im, lower=lo, upper=high, n_available=n_available, n_data=len(points),
                                 n_random=nrandom, w_theta=float(value[j]) if finite else None,
                                 jackknife_sigma=float(error[j]) if finite else None,
                                 dd=dd[j], dr=dr[j], rr=rr[j], null_reference=0.))
            print(f"[diagnostics] clustering z-bin {iz+1}, mass-bin {im+1}: N={len(points)}", flush=True)
    return rows


def compute(settings, output):
    from geppetto.galaxies.validation import _integrity

    print("[diagnostics] verifying frozen catalogue, native input and calibration", flush=True)
    data, halos, distances, target, table, metadata, hashes = read_frozen(settings)
    if metadata["config"]["calibration"]["min_particles"] != 32:
        raise ValueError("Diagnostic conventions require the frozen 32-particle model")
    print("[diagnostics] evaluating frozen occupations (no fitting)", flush=True)
    prediction = predict_halos(halos, table, settings)
    products = count_products(data, halos, distances, target, table, prediction, settings)
    products["completeness"] = completeness_rows(table, halos, settings)
    print("[diagnostics] conditional colour and apparent-magnitude predictions", flush=True)
    products.update(photometry_products(data, distances, metadata, settings))
    extra, placement = placement_products(data, halos, prediction, metadata, settings)
    products.update(extra)
    print("[diagnostics] unfitted angular clustering with spatial jackknife", flush=True)
    products["clustering"] = clustering_products(data, settings)
    integrity = _integrity(data, halos, metadata["config"], distances)
    chi = np.linalg.norm(halos.values["position"], axis=1)
    geometry_error = chi/distances.comoving_distance(halos.values["z_cos"])-1
    identity = dict(extra_native_group_occurrences=len(chi)-len(np.unique(halos.values["group_id"])),
                    extra_central_group_occurrences=int(data["is_central"].sum())-len(np.unique(data["host_group_id"][data["is_central"]])),
                    max_native_distance_relative_error=float(np.max(abs(geometry_error))))
    for name, rows in products.items():
        write_rows(output/f"{name}.csv", rows)
    for path, expected in hashes.items():
        if digest(path) != expected:
            raise ValueError(f"Frozen product changed during audit: {path}")
    cache = Path(metadata["config"]["output_dir"])/"cache"/f"hod_{metadata['calibration_key']}.npz"
    shutil.copy2(cache, output/"frozen_hod.npz")
    manifest = dict(settings=settings, frozen_model=metadata, input_hashes=hashes, integrity=integrity,
                    placement=placement, identity=identity, n_galaxies=len(data["galaxy_id"]), n_halos=len(chi),
                    source_sha256=digest(__file__), command=" ".join(sys.argv), python=sys.executable,
                    products=list(products), model_parameters_changed=False,
                    versions={name: importlib.metadata.version(name) for name in ("numpy", "scipy", "matplotlib", "jax", "h5py")},
                    groups={"01": "Native HMF and completeness", "02": "Luminosity functions",
                            "03": "Redshift and apparent-magnitude selection", "04": "Central and satellite occupations",
                            "05": "Colour-magnitude and colour distributions", "06": "Satellite radii",
                            "07": "Satellite velocities and observed-redshift closure", "08": "Sky geometry and identity checks",
                            "09": "Unfitted angular clustering"})
    write_json(output/"manifest.json", manifest)
    write_json(output/"frozen_model.json", metadata)
    return manifest


def paired_panels(columns=3, rows=1, right=.98):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(13.2, 4.1*rows+2.2))
    grid = fig.add_gridspec(rows, columns, left=.065, right=right,
                           bottom=1.7/(4.1*rows+2.2),
                           top=.85 if rows == 1 else .93, hspace=.55, wspace=.32)
    axes = []
    for row in range(rows):
        for col in range(columns):
            pair = grid[row, col].subgridspec(2, 1, height_ratios=[3., 1.], hspace=.06)
            top = fig.add_subplot(pair[0])
            bottom = fig.add_subplot(pair[1], sharex=top)
            top.tick_params(labelbottom=False)
            axes.append((top, bottom))
    return fig, axes


def decorate_ratio(axis, ylabel="Mock / model"):
    axis.axhline(1., color=".35", lw=.9)
    axis.set_ylabel(ylabel, fontsize=9)
    axis.grid(alpha=.15)


def save_figure(fig, output, name, title, manifest, selection, uncertainty, reference, sample):
    import matplotlib.pyplot as plt
    fig.suptitle(title, x=.065, ha="left", fontsize=15, y=.985)
    lines = [
        "L3870N4096/000: native fragmentation-group M_PIN [Msun/h], NOT M200m; native floor 10 particles; painting cut 32 particles = 2.931e12 Msun/h.",
        "Magnitudes: 0.1 M_r - 5 log10(h), AB; colour: rest-frame 0.1(g-r). Distances: comoving Mpc/h; peculiar velocities: proper km/s. Cone frame, not ICRS.",
        f"Selection: {selection} | Sample: {sample}",
        f"Uncertainty: {uncertainty}",
        f"Reference: {reference}. Completeness claims: M_r <= -22, inner 65 deg, 0.05 <= z_cos < 0.32 only.",
    ]
    wrapped = "\n".join(textwrap.fill(line, 202, subsequent_indent="  ") for line in lines)
    footer = fig.text(.065, .018, wrapped, fontsize=7.2, va="bottom", linespacing=1.4)
    fig.canvas.draw()
    bounds = footer.get_window_extent(fig.canvas.get_renderer())
    if bounds.x1 > fig.bbox.x1-5 or bounds.y1 > min(ax.bbox.y0 for ax in fig.axes)-30:
        raise ValueError(f"Figure footer exceeds its reserved area: {name}")
    fig.savefig(output/f"{name}.png", dpi=190)
    plt.close(fig)
    manifest.setdefault("figures", []).append(dict(file=f"{name}.png", title=title, group=name[:2],
                                                   selection=selection, uncertainty=uncertainty,
                                                   reference=reference, sample=sample, caption="\n".join(lines)))


def plot_histogram(axis, residual, rows, xlabel, color=COLORS[0]):
    lo = np.array([r["lower"] for r in rows])
    hi = np.array([r["upper"] for r in rows])
    x, width = (lo+hi)/2, hi-lo
    count, expected, sigma = [np.array([r[key] for r in rows]) for key in ("count", "expected", "sigma")]
    n = rows[0]["n_sample"]
    axis.stairs(expected/(n*width), np.r_[lo, hi[-1]], color="black", label="Frozen-model prediction", lw=1.4)
    axis.errorbar(x, count/(n*width), sigma/(n*width), fmt="o", ms=3, color=color, capsize=2, label="Mock")
    keep = expected >= 5
    residual.errorbar(x[keep], count[keep]/expected[keep], sigma[keep]/expected[keep], fmt="o", ms=3, color=color, capsize=2)
    decorate_ratio(residual)
    residual.set_xlabel(xlabel)
    axis.set_ylabel("Probability density")
    axis.text(.97, .93, f"N={n:,.0f}", ha="right", va="top", transform=axis.transAxes, fontsize=9)


def plot_all(output, manifest):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/geppetto-galaxy-diagnostics-mpl")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize

    plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "axes.titlesize": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "legend.fontsize": 8})
    manifest["figures"] = []
    settings = manifest["settings"]
    zbins = list(zip(settings["redshift_edges"][:-1], settings["redshift_edges"][1:], strict=True))
    mbins = list(zip(settings["log10_host_mass_edges"][:-1], settings["log10_host_mass_edges"][1:], strict=True))
    data = {key: read_rows(output/f"{key}.csv") for key in manifest["products"]}
    mass_cut = np.log10(manifest["frozen_model"]["particle_mass_msun_h"]*32)
    native_cut = np.log10(manifest["frozen_model"]["particle_mass_msun_h"]*10)
    basic = "inner 65 deg, z_host bins; HMF: all native halos; occupied hosts: M_r<-22, before r<20"
    count_text = "; ".join(f"z={lo:.2f}-{hi:.2f}" for lo, hi in zbins)
    fig, axes = paired_panels()
    for iz, ((lo, high), (ax, res)) in enumerate(zip(zbins, axes, strict=True)):
        rows = [r for r in data["hmf"] if r["zbin"] == iz]
        x = np.array([.5*(r["lower"]+r["upper"]) for r in rows])
        width = np.array([r["upper"]-r["lower"] for r in rows])
        scale = np.array([r["volume"] for r in rows])*width
        native, kept, occupied, expected, sigma = [np.array([r[key] for r in rows]) for key in ("n_native", "n_resolved", "n_occupied", "expected_occupied", "sigma_occupied")]
        ax.plot(x, np.where(native > 0, native/scale, np.nan), color=".55", label="Native PLC (measured)")
        ax.plot(x, np.where(kept > 0, kept/scale, np.nan), color="black", ls="--", label="32-particle selection")
        ax.plot(x, np.where(expected > 0, expected/scale, np.nan), color=COLORS[2], label="Occupied hosts: HOD")
        mask = occupied > 0
        ax.errorbar(x[mask], occupied[mask]/scale[mask], sigma[mask]/scale[mask], fmt="o", ms=3, color=COLORS[1], label="Occupied hosts: mock")
        use = expected >= 5
        res.errorbar(x[use], occupied[use]/expected[use], sigma[use]/expected[use], fmt="o", ms=3, color=COLORS[1])
        for axis in (ax, res):
            axis.axvspan(x.min()-.1, mass_cut, color=".92", zorder=-10)
            axis.axvline(mass_cut, color=".3", ls=":")
            axis.axvline(native_cut, color=".5", ls="--", lw=.8)
        ax.set(yscale="log", ylim=(2.e-9, .04), title=f"{lo:.2f} <= z_host < {high:.2f}")
        ax.text(.97, .95, f"N_halo={int(native.sum()):,}\nN_occupied={int(occupied.sum()):,}", transform=ax.transAxes, va="top", ha="right", fontsize=8)
        ax.set_ylabel(r"$dn/d\log_{10}M\ [(h/{\rm Mpc})^3]$")
        decorate_ratio(res, "Occupied / HOD")
        res.set_xlabel(MASS_LABEL)
    axes[0][0].legend(loc="lower left")
    save_figure(fig, output, "01_native_hmf", "01  Native HMF, halo selection and occupied hosts", manifest, basic,
                "occupied-host Bernoulli 1 sigma; ratios shown only for expected counts >=5; grey = below painting cut",
                "native HMF is measured input, NOT an independent analytic HMF; occupied-host curve is the tested HOD", count_text)

    fig, axes = paired_panels()
    for (lo, high), (ax, res) in zip(zbins, axes, strict=True):
        rows = [r for r in data["luminosity_function"] if abs(r["z_min"]-lo) < 1.e-8]
        x = np.array([.5*(r["magnitude_bright"]+r["magnitude_faint"]) for r in rows])
        observed, target, prediction, sigma = [np.array([r[key] for r in rows]) for key in ("mock_lf", "target_lf", "deterministic_lf", "acceptance_sigma")]
        reliable = np.array([r["count"] >= 200 and r["magnitude_faint"] <= -22+1.e-8 for r in rows])
        ax.plot(x, target, "k-", label="Independent SDSS/GAMA target")
        ax.plot(x, prediction, "s", mfc="none", color=COLORS[2], label="Frozen HOD prediction")
        for subset, fill in ((reliable, COLORS[1]), (~reliable, "white")):
            ax.errorbar(x[subset], observed[subset], sigma[subset], fmt="o", ms=4, mfc=fill, color=COLORS[1], capsize=2,
                        label="Mock" if fill != "white" else None)
            res.errorbar(x[subset], observed[subset]/target[subset], sigma[subset]/target[subset], fmt="o", ms=4, mfc=fill, color=COLORS[1], capsize=2)
        res.plot(x, prediction/target, "s", ms=4, mfc="none", color=COLORS[2])
        decorate_ratio(res, "LF / target")
        res.axhspan(.95, 1.05, color=".90", zorder=-10)
        for axis in (ax, res):
            axis.axvspan(-22., -21.5, color="#eadede", alpha=.55, zorder=-10)
            axis.axvline(-22., color=".4", ls=":")
        ax.set(yscale="log", title=f"{lo:.2f} <= z_cos < {high:.2f}", ylabel=r"$\phi\ [(h/{\rm Mpc})^3\,mag^{-1}]$")
        ax.text(.03, .04, "N bins: "+", ".join(str(int(r["count"])) for r in rows), transform=ax.transAxes, fontsize=7.5)
        res.set_xlabel(MAG_LABEL)
    axes[0][0].legend(loc="upper left")
    save_figure(fig, output, "02_luminosity_function", "02  Luminosity-function closure to an independent target", manifest,
                "inner 65 deg; r<20; z_cos and M_r bins shown; red shading = no completeness claim",
                "1 sigma = max(HOD Bernoulli+Poisson, 32-region jackknife); open symbols: N<200 or M_r>-22",
                "SDSS/GAMA LF is independently specified; HOD is calibrated to that target", "counts per bin printed; 3 required complete bins per z slice")

    fig, axes = paired_panels(columns=2, rows=2)
    ax, res = axes[0]
    rows = data["redshift_counts"]
    x = np.array([.5*(r["lower"]+r["upper"]) for r in rows])
    observed, expected, sigma, truth = [np.array([r[key] for r in rows]) for key in ("count", "expected", "sigma", "independent_lf_count")]
    ax.plot(x, truth, "k-", label="Independent LF x native volume")
    ax.plot(x, expected, "--", color=COLORS[2], label="Frozen HOD halo sum")
    ax.errorbar(x, observed, sigma, fmt="o", ms=3, color=COLORS[1], label="Mock")
    ax.set(title=f"M_r < -22; N={int(observed.sum()):,}", ylabel="Galaxies per dz=0.01")
    res.errorbar(x, observed/expected, sigma/expected, fmt="o", ms=3, color=COLORS[1])
    decorate_ratio(res)
    res.set_xlabel("Cosmological redshift")
    ax.legend()
    for iz, ((lo, high), (ax, res)) in enumerate(zip(zbins, axes[1:], strict=True)):
        rows = [r for r in data["apparent_magnitude"] if r["zbin"] == iz]
        plot_histogram(ax, res, rows, "Apparent r [AB mag]", COLORS[iz])
        ax.set_title(f"{lo:.2f} <= z_cos < {high:.2f}; underlying sample")
        for axis in (ax, res):
            axis.axvline(20., color=".35", ls=":")
            axis.axvspan(20., 21., color=".92")
    axes[1][0].legend(loc="upper left")
    save_figure(fig, output, "03_selection", "03  Redshift counts and the apparent-magnitude selection", manifest,
                "inner 65 deg; upper-left M_r<-22, r<20; other panels underlying M_r<-21.5 BEFORE r<20",
                "upper-left: conditional HOD 1 sigma; apparent-r: conditional colour-draw 1 sigma (N and M_r,z held fixed)",
                "upper-left independent LF and frozen HOD; other panels analytic colour+k-correction of the same model", "N printed; r=20 line is NOT a claim of faint flux completeness")

    fig, axes = paired_panels()
    for iz, ((lo, high), (ax, res)) in enumerate(zip(zbins, axes, strict=True)):
        for status, color in (("central", COLORS[0]), ("satellite", COLORS[1])):
            rows = [r for r in data["occupation"] if r["zbin"] == iz and r["status"] == status and r["n_halos"] > 0]
            x = np.array([.5*(r["lower"]+r["upper"]) for r in rows])
            count, expected, sigma, n = [np.array([r[key] for r in rows]) for key in ("count", "expected", "sigma", "n_halos")]
            ax.plot(x, np.where(expected > 0, expected/n, np.nan), color=color, label=status+": HOD")
            use = count > 0
            ax.errorbar(x[use], count[use]/n[use], sigma[use]/n[use], fmt="o", ms=3, color=color)
            valid = expected >= 5
            res.errorbar(x[valid], count[valid]/expected[valid], sigma[valid]/expected[valid], fmt="o", ms=3, color=color)
        ax.set(title=f"{lo:.2f} <= z_host < {high:.2f}", yscale="log", ylabel="Mean galaxies per halo", ylim=(1.e-5, 30.))
        ax.text(.97, .95, f"N_halo={int(n.sum()):,}", ha="right", va="top", transform=ax.transAxes, fontsize=9)
        for axis in (ax, res):
            axis.axvline(mass_cut, color=".4", ls=":")
            axis.axvspan(12.3, mass_cut, color=".92")
            axis.set_xlim(12.3, 15.)
        decorate_ratio(res)
        res.set_xlabel(MASS_LABEL)
    axes[0][0].legend(loc="upper left")
    save_figure(fig, output, "04_occupation", "04  Luminosity-threshold occupations on native halo mass", manifest,
                "host-selected inner 65 deg and z_host; M_r<-22, before r<20; ALL resolved halos in denominators",
                "central Bernoulli and satellite Poisson 1 sigma; lines=HOD, markers=mock; ratios require expectation >=5",
                "frozen native-mass HOD averaged over actual masses and redshifts, NOT observed richness", count_text)

    fig, axes = paired_panels(right=.915)
    colour_maximum = max(r["count"] for r in data["colour_magnitude"])
    for iz, ((lo, high), (ax, res)) in enumerate(zip(zbins, axes, strict=True)):
        rows = [r for r in data["colour_magnitude"] if r["zbin"] == iz]
        mlo = sorted({r["magnitude_lower"] for r in rows})
        clo = sorted({r["colour_lower"] for r in rows})
        matrix = np.array([r["count"] for r in rows]).reshape(len(mlo), len(clo))
        medges = np.r_[mlo, max(r["magnitude_upper"] for r in rows)]
        cedges = np.r_[clo, max(r["colour_upper"] for r in rows)]
        mesh = ax.pcolormesh(medges, cedges, np.ma.masked_less(matrix.T, 1), norm=LogNorm(vmin=1, vmax=max(2, colour_maximum)), cmap="viridis", rasterized=True)
        representative = [next(r for r in rows if r["magnitude_lower"] == m) for m in mlo]
        good = [r for r in representative if r["n_magnitude_bin"] >= 20]
        x = [.5*(r["magnitude_lower"]+r["magnitude_upper"]) for r in good]
        ax.plot(x, [r["mean_expected"] for r in good], color="white", lw=1.8, label="Conditional colour mean")
        res.errorbar(x, [r["mean_measured"]-r["mean_expected"] for r in good], [r["sigma_mean"] for r in good], fmt="o", ms=3, color=COLORS[iz])
        res.axhline(0., color=".35", lw=.8)
        res.set(xlabel=MAG_LABEL, ylabel="Mean residual [mag]")
        ax.set(ylabel=r"Rest-frame $^{0.1}(g-r)$ [mag]", title=f"{lo:.2f} <= z_cos < {high:.2f}\nN={int(matrix.sum()):,}")
        for axis in (ax, res):
            axis.axvline(-22., color="#b35806", ls="--", lw=1)
    fig.colorbar(mesh, cax=fig.add_axes([.935, .38, .013, .4]), label="Galaxies / 2D bin")
    axes[0][0].legend(loc="lower left", facecolor=".2", labelcolor="white")
    save_figure(fig, output, "05a_colour_magnitude", "05a  Rest-frame colour versus luminosity", manifest,
                "inner 65 deg; z_cos bins shown; underlying M_r<-21.5 BEFORE flux cut; right of dashed line not LF-complete",
                "mean residual: exact conditional-mixture 1 sigma; count image uses log colour scale, not an uncertainty",
                "white curve is hodpy's conditional mixture mean, NOT an independently observed colour target", "N inside displayed magnitude/colour range printed; residual bins N>=20")

    fig, axes = paired_panels(columns=3, rows=3)
    for iz, (lo, high) in enumerate(zbins):
        for im, (mlo, mhi) in enumerate(mbins):
            ax, res = axes[iz*3+im]
            rows = [r for r in data["colour_distribution"] if r["zbin"] == iz and r["massbin"] == im]
            plot_histogram(ax, res, rows, r"$^{0.1}(g-r)$ [mag]", COLORS[im])
            ax.set_title(f"z={lo:.2f}-{high:.2f}; log M={mlo:.1f}-{mhi:.1f}", fontsize=10)
    axes[0][0].legend(loc="upper left")
    save_figure(fig, output, "05b_colour_distributions", "05b  Colour mixtures across redshift and host mass", manifest,
                "inner 65 deg; M_r<-22; pre-flux-cut intrinsic sample; z_cos and log10 native host mass bins labelled",
                "conditional Poisson-binomial 1 sigma from Gaussian-mixture bin probabilities; ratios only E[N]>=5",
                "integrated upstream hodpy colour mixture at each galaxy's M_r,z and central status; model test only", "N in each panel; 3 redshift x 3 host-mass bins")

    for key, name, title, xlabel, reference in (
        ("satellite_radii", "06_satellite_radii", "06  Truncated-NFW satellite radii", r"$r/R_{\rm sat}$ [dimensionless]", "exact truncated 3D NFW CDF, averaged over each halo's predicted concentration"),
        ("satellite_velocity", "07_satellite_velocities", "07  Satellite velocities and redshift-space prescription", r"$\Delta v_{\rm los}/\sigma_{\rm sat}$ [dimensionless]", "unit Gaussian for satellite velocity relative to its host; sigma from the frozen effective radius")):
        fig, axes = paired_panels(columns=3, rows=2)
        for row, (dimension, bins) in enumerate((("redshift", zbins), ("mass", mbins))):
            for i, (lo, high) in enumerate(bins):
                ax, res = axes[row*3+i]
                rows = [r for r in data[key] if r["dimension"] == dimension and r["group"] == i]
                plot_histogram(ax, res, rows, xlabel, COLORS[i])
                ax.set_title((f"z_host={lo:.2f}-{high:.2f}; all host masses" if dimension == "redshift" else f"log M={lo:.1f}-{high:.1f}; pooled redshifts"), fontsize=10)
        axes[0][0].legend(loc="upper left")
        special = "hard support at r/R_sat=1; effective radius is NOT measured R200m" if key == "satellite_radii" else f"max Doppler-redshift closure error={manifest['placement']['max_observed_redshift_closure']:.2e}"
        save_figure(fig, output, name, title, manifest,
                    "SATELLITES; host-selected cone65, 0.05<=z_host<0.32; M_r<-22; pre-flux cut (avoids position/velocity selection)",
                    "conditional binomial 1 sigma; ratios require expected bin count>=5; low-z satellite sample is small",
                    reference+"; internal sampler test", f"N per panel; {special}")

    fig, axes = plt.subplots(3, 3, figsize=(13.2, 11.5))
    fig.subplots_adjust(left=.065, right=.965, top=.92, bottom=.17, hspace=.48, wspace=.4)
    for iz, (lo, high) in enumerate(zbins):
        rows = [r for r in data["sky"] if r["zbin"] == iz]
        phi = sorted({r["phi_lower"] for r in rows})
        area = sorted({r["area_lower"] for r in rows})
        xedges, yedges = np.r_[phi, 360.], np.r_[area, 1.]
        upper = max(max(r["count"], r["expected"]) for r in rows)
        for col, (field, label) in enumerate((("count", "Mock counts"), ("expected", "Frozen HOD at host centres"), ("pull", "(Mock - HOD) / conditional sigma"))):
            values = np.array([np.nan if r[field] is None else r[field] for r in rows]).reshape(len(phi), len(area)).T
            norm_value, cmap = (Normalize(-4, 4), "RdBu_r") if field == "pull" else (LogNorm(1, max(2, upper)), "viridis")
            mesh = axes[iz, col].pcolormesh(xedges, yedges, np.ma.masked_invalid(values) if field == "pull" else np.ma.masked_less(values, 1), norm=norm_value, cmap=cmap, rasterized=True)
            axes[iz, col].set(title=f"{label}\nz={lo:.2f}-{high:.2f}; N={sum(r['count'] for r in rows):,.0f}", xlabel="Cone longitude [deg]", ylabel="Equal-area coordinate q")
            fig.colorbar(mesh, ax=axes[iz, col], label="sigma" if field == "pull" else "Galaxies per cell", fraction=.05, pad=.02)
    save_figure(fig, output, "08_sky_geometry", "08  Sky distribution, footprint and identity audit", manifest,
                "inner 65 deg; M_r<-22, r<20; z_cos bins shown; q=(1-cos opening angle)/(1-cos65deg)",
                "conditional HOD 1 sigma; pull cells need variance>=10; no cosmic-variance error (same fixed halo realization)",
                "HOD at host centres, a model diagnostic; satellite boundary migrations are not integrated into the reference",
                f"counts per row; 374 extra native group occurrences preserved; max native distance residual={manifest['identity']['max_native_distance_relative_error']:.2e}")

    fig, axes = paired_panels()
    for im, ((lo, high), (ax, res)) in enumerate(zip(mbins, axes, strict=True)):
        for iz, (zlo, zhi) in enumerate(zbins):
            rows = [r for r in data["clustering"] if r["massbin"] == im and r["zbin"] == iz and r["w_theta"] is not None]
            x = np.array([np.sqrt(r["lower"]*r["upper"]) for r in rows])
            value, error = [np.array([r[key] for r in rows]) for key in ("w_theta", "jackknife_sigma")]
            ax.errorbar(x, value, error, fmt="o-", ms=3, color=COLORS[iz], label=f"z={zlo:.2f}-{zhi:.2f}, N={rows[0]['n_data']:,.0f}")
            valid = error > 0
            res.plot(x[valid], value[valid]/error[valid], "o-", ms=3, color=COLORS[iz])
        ax.axhline(0., color=".35", ls="--", label="Unclustered null")
        res.axhline(0., color=".35", ls="--")
        ax.set(xscale="log", ylabel=r"$w(\theta)$", title=f"log M_PIN={lo:.1f}-{high:.1f}")
        ax.set_yscale("symlog", linthresh=.03)
        res.set(xscale="log", xlabel=r"$\theta$ [deg]", ylabel=r"$w/\sigma_w$")
        ax.legend(loc="lower right")
    save_figure(fig, output, "09_clustering", "09  Unfitted angular clustering: a diagnostic, not a calibration", manifest,
                "inner65 deg; M_r<-22, r<20; z_cos and host-mass bins labelled; deterministic random subsampling",
                "8 equal-area-region delete-one jackknife, 1 sigma; 3 randoms per data object; finite-random and covariance limitations apply",
                "zero is an UNCLUSTERED NULL, NOT a predicted galaxy clustering model; no goodness-of-fit threshold",
                "N per curve; <=10,000 data galaxies each; plotted null significance is diagnostic only")
    manifest["plot_source_sha256"] = digest(__file__)
    write_json(output/"manifest.json", manifest)


def write_report(output, manifest):
    settings = manifest["settings"]
    lf = read_rows(output/"luminosity_function.csv")
    cumulative = read_rows(output/"cumulative_lf.csv")
    complete = read_rows(output/"completeness.csv")
    colours = read_rows(output/"colour_moments.csv")
    required = [r for r in lf if r["magnitude_bright"] >= -22.8-1.e-8
                and r["magnitude_faint"] <= settings["complete_magnitude_limit"]+1.e-8]
    required_complete = [r for r in complete if r["lower"] >= -22.8-1.e-8
                         and r["upper"] <= settings["complete_magnitude_limit"]+1.e-8]
    if len(required) != 9 or len(required_complete) != 9:
        raise ValueError("The preselected audit requires three LF bins in each of three redshift slices")
    colour_cutoff = .01/len(colours)
    colour_failures = [r for r in colours if r["p_value"] < colour_cutoff]
    tests = manifest["placement"]["placement_tests"]
    sampling_cutoff = .01/len(tests)
    placement_failures = [r for r in tests if r["p_value"] < sampling_cutoff]
    occupancy = read_rows(output/"occupation.csv")
    occupancy_pulls = [(r["count"]-r["expected"])/r["sigma"] for r in occupancy if r["expected"] >= 20 and r["sigma"]**2 >= 10]
    histogram_checks = []
    for name in ("apparent_magnitude", "redshift_counts", "colour_distribution", "satellite_radii", "satellite_velocity", "sky"):
        rows = read_rows(output/f"{name}.csv")
        pulls = [(r["count"]-r["expected"])/r["sigma"] for r in rows if r["expected"] >= 20 and r["sigma"]**2 >= 10]
        histogram_checks.append(dict(product=name, populated_bins=len(pulls), maximum_absolute_pull=max(map(abs, pulls))))
    write_rows(output/"histogram_checks.csv", histogram_checks)
    max_lf = max(abs(r["mock_residual"]) for r in required)
    max_cumulative = max(abs(r["residual"]) for r in cumulative if -22.8-1.e-8 <= r["magnitude_threshold"] <= -22+1.e-8)
    max_missing = max(r["missing_fraction"] for r in complete if r["complete"])
    result = dict(model_parameters_changed=False, required_lf_bins=len(required), minimum_required_count=int(min(r["count"] for r in required)),
                  required_lf_passed=bool(all(r["passed"] for r in required)
                                         and all(r["complete"] for r in required_complete)
                                         and max_cumulative <= .02), maximum_mock_lf_residual=max_lf,
                  maximum_cumulative_lf_residual=max_cumulative, maximum_complete_bin_missing_fraction=max_missing,
                  colour_tests=len(colours), colour_p_value_cutoff=colour_cutoff, colour_failures=colour_failures,
                  sampling_tests=len(tests), sampling_p_value_cutoff=sampling_cutoff, sampling_failures=placement_failures,
                  occupation_chi2=float(np.sum(np.square(occupancy_pulls))), occupation_bins=len(occupancy_pulls),
                  occupation_chi2_p_value=float(chi2.sf(np.sum(np.square(occupancy_pulls)), len(occupancy_pulls))),
                  integrity_passed=manifest["integrity"]["passed"], clustering_fitted=False,
                  occupation_p_value_cutoff=.01, histogram_checks=histogram_checks)
    checks = [
        dict(check="Frozen inputs and model", status="PASS", evidence="Catalogue, input, calibration and model-source hashes verified before/after; no fitter called", cause="Read-only audit"),
        dict(check="Independent LF target", status="PASS" if result["required_lf_passed"] else "REVIEW", evidence=f"{len(required)} complete bins; min N={result['minimum_required_count']}; max mock residual={100*max_lf:.3f}%; cumulative={100*max_cumulative:.5f}%", cause="Fixed target is fitted by the HOD; remaining mock deviations are stochastic and include boundary migrations"),
        dict(check="Native-mass completeness", status="LIMIT", evidence=f"10-particle native floor; 32-particle painting cut; max missing fraction in claimed bins={100*max_missing:.4f}%", cause="Fainter M_r>-22 and sparse bright bins are diagnostic only; not a physical halo-finder convergence test"),
        dict(check="HOD occupation", status="PASS" if result["occupation_chi2_p_value"] > result["occupation_p_value_cutoff"] else "REVIEW", evidence=f"Conditional chi2={result['occupation_chi2']:.1f}/{len(occupancy_pulls)} bins; p={result['occupation_chi2_p_value']:.3g}, alpha=.01", cause="Mild excess scatter; E[N]>=20 and variance>=10 for Gaussian chi2. Compatible at the stated threshold, not a precision richness validation"),
        dict(check="Conditional colours", status="PASS" if not colour_failures else "REVIEW", evidence=f"{len(colours)} mixture/mean/width tests; {len(colour_failures)} failures at familywise alpha=.01", cause="Same hodpy prescription tested, not independent observed colour data; red probability saturates in bright satellite bins"),
        dict(check="NFW radial / Gaussian velocity draws", status="PASS" if not placement_failures else "REVIEW", evidence=f"{len(tests)} mass/redshift-stratified KS tests; {len(placement_failures)} failures; minimum p={min(r['p_value'] for r in tests):.3g}", cause="Model tests only; low-z satellites have limited statistics"),
        dict(check="Units, support and redshift closure", status="PASS" if result["integrity_passed"] else "REVIEW", evidence=f"Integrity passed={result['integrity_passed']}; Doppler closure={manifest['placement']['max_observed_redshift_closure']:.2e}", cause="R_sat is an effective model radius, not measured R200m"),
        dict(check="Native PLC identities", status="CAVEAT", evidence=f"{manifest['identity']['extra_native_group_occurrences']} extra group-ID occurrences; {manifest['identity']['extra_central_group_occurrences']} extra centrals across persistent IDs", cause="Nearby distinct native crossing records retained; one central per unique PLC occurrence, not persistent ID"),
        dict(check="Sky / distance closure", status="CAVEAT", evidence=f"Native distance max relative mismatch={manifest['identity']['max_native_distance_relative_error']:.2e}; sky residuals shown", cause="Finite-precision native table/PLC; host-centre sky prediction omits satellite boundary migrations"),
        dict(check="Angular clustering", status="DIAGNOSTIC", evidence="Measured across 3 z and 3 host-mass bins, with spatial jackknife; no fitting", cause="Nonzero clustering relative to the unclustered null is expected; no observed clustering or richness claim"),
    ]
    result["checks"] = checks
    write_json(output/"audit_summary.json", result)
    write_rows(output/"checks.csv", checks)
    lines = ["# Frozen PINOCCHIO-hodpy diagnostic report", "", "## Scope and reproducibility", "",
             "This audit reads the completed **660,873-galaxy** catalogue from real L3870N4096/000 data. No model parameter, calibration table or galaxy value is changed. The eight groups below were confirmed for this audit. Group 09 is an additional unfitted diagnostic.", "",
             "The only independent observational target here is the fixed SDSS/GAMA r-band LF supplied by hodpy. HOD, colour, NFW and velocity curves are **predictions of the same frozen model being tested**. The native HMF is measured input. Clustering is compared only with an unclustered null, not an observed galaxy-clustering target.", "",
             "## Conventions and selections", "",
             "- Simulation: L=3870 Mpc/h, 4096^3 particles, seed 1386. Omega_m=0.3913, Omega_b=0.0419, Omega_DE=0.6087, h=0.7276, n_s=0.9312, sigma8=0.60850325, w0=-1.168, wa=-0.6605.",
             "- M_PIN is native fragmentation-group particle-count mass [Msun/h], not M200m or M200c. Particle mass=9.160181752e10 Msun/h; native minimum=10 particles; painting minimum=32 particles (2.931258161e12 Msun/h).",
             "- Positions and radii: comoving Mpc/h. Velocities: proper peculiar km/s. Sky coordinates: cone-aligned longitude/latitude, not ICRS. Cosmological and Doppler-observed redshifts remain distinct.",
             "- Magnitudes: AB, ^0.1 M_r-5log10(h). Colours: rest-frame ^0.1(g-r), not observed multiband photometry. Underlying sample M_r<-21.5; survey flag r<20 is applied only where labelled.",
             "- Audit redshift bins: [0.05,0.14), [0.14,0.23), [0.23,0.32). Host log10(M_PIN/[Msun/h]) bins: [12.4,13.5), [13.5,14.2), [14.2,15.0), with the 32-particle lower cut inside the first bin. Inner cone half-angle=65 deg; the original 70 deg cone and z=0.04-0.33 input provide satellite buffers.",
             "- LF claims require M_r<=-22, model-estimated omitted native low-mass fraction<1%, support above the native 10-particle floor, and N>=200 in each differential bin. Sparse bright bins and the displayed faint extension are labelled diagnostics, not completeness claims.",
             "- HOD and satellite diagnostics select on **host** coordinates/redshift before flux selection, avoiding radial/velocity selection bias. LF, redshift counts, sky and clustering use final galaxy coordinates. Colour distributions are before flux selection; apparent-r references explicitly integrate the colour-dependent selection.", "",
             "## Frozen prescriptions", "",
             "The luminosity-dependent hodpy central Bernoulli and satellite Poisson occupations use M_PIN. The existing calibration shifts all three HOD mass scales together at each luminosity threshold and redshift cell, against the measured native PLC HMF. The numerical thresholds and shifts are archived unchanged in frozen_hod.npz. This audit does not rerun the calibration or update those tables.", "",
             "Satellite radius: R_sat,com=[3 M_PIN/(4 pi 200 rho_mean,com)]^(1/3). Concentration: c_sat=5 (M_PIN/1e14 Msun/h)^(-0.1) (1+z_host)^(-0.5). Draws follow the NFW enclosed-mass CDF truncated at R_sat; there is no measured spherical-overdensity radius. Velocities add independent Gaussian components with sigma_1D=[G M_PIN (1+z_host)/(2 R_sat,com)]^(1/2). Centrals inherit host positions and velocities. Observed redshift is z_cos+(1+z_cos) v_los/c.", "",
             "The independent LF combines the fixed SDSS cumulative table and hodpy's evolving GAMA Schechter prescription (phi_star=0.0094 h^3 Mpc^-3, M_star=-20.7, alpha=-1.23, P=1.8, Q=0.7). The exact transition and source hashes are frozen in the input metadata and upstream code. Colours use hodpy's two-Gaussian mixture at each M_r, z_cos and central/satellite status; apparent r uses its colour-dependent GAMA k-correction. This is not an independently tested colour observation or complete multiband photometry.", "",
             "## Uncertainties and references", "",
             "LF errors use max(conditional Bernoulli+Poisson HOD sigma, 32 equal-area-region jackknife sigma). The required tolerance remains max(5%,3 sigma). Other count/colour/radius histograms condition on the listed sample: their analytic bin probabilities give sum p(1-p), not an ensemble cosmic-variance covariance. Gaussian-mixture means and standardized sequence widths are tested explicitly in colour_moments.csv. NFW/velocity KS tests use a 1% Bonferroni familywise threshold across the reported strata. These tests are correlated across overlapping redshift and mass projections, so the correction is conservative.", "",
             "The apparent-r prediction is integrated analytically over each clipped, piecewise-linear colour-to-k-correction interval at the actual M_r,z and central/satellite status. It is not an independent survey number-count prediction. It includes both Gaussian tails; the sample is not a complete faint flux-limited survey.", "",
             "Clustering uses angular Landy-Szalay, deterministic subsampling at <=10,000 galaxies per stratum, three uniform randoms per data object, and eight equal-area delete-one regions. Jackknife limitations and finite random-pair noise remain, particularly in small-angle sparse bins. No covariance inversion, clustering fit or richness calibration is attempted.", "",
             "The aggregate occupation chi-square is reported only for bins with expected counts>=20 and variance>=10, with alpha=0.01. Its p-value indicates mild excess scatter, not exact agreement. Sparse saturated-central bins are excluded from that Gaussian statistic but remain in the binned data. Histogram residual summaries require the same count/variance cuts; they are descriptive, not independent multiple-bin significance tests.", "",
             "Visible sparse-bin excursions are not hidden: the lowest-z brightest LF bin has only seven galaxies; the highest-z log10(M_PIN)=14.85-15.0 occupation bin has five hosts and 38 bright satellites versus 23.56 expected (2.98 conditional sigma). The lowest-z radial/velocity sample has only 56 satellites, so its radial histogram uses six bins instead of fifteen. The unbinned KS tests do not depend on this plotting choice. These are plausible sampling fluctuations, not evidence for observed richness agreement.", "",
             "## Rerun", "", "```bash", "/home/tcastro/miniforge3/envs/geppetto-dev/bin/python examples/diagnose_pinocchio_galaxies.py --config examples/pinocchio_galaxy_diagnostics_config.json",
             "# Replot from the saved CSV/JSON data without the large input/catalogue:", "/home/tcastro/miniforge3/envs/geppetto-dev/bin/python examples/diagnose_pinocchio_galaxies.py --config examples/pinocchio_galaxy_diagnostics_config.json --plot-only",
             "/usr/bin/python3 examples/render_pinocchio_galaxy_report.py --input-dir examples/pinocchio_galaxy_diagnostics", "```", "",
             "The complete frozen configuration, input/code hashes, package versions and command are in manifest.json and frozen_model.json; frozen_hod.npz is a byte-for-byte copy of the existing calibration. Plotting and rendering source snapshots accompany the figures in code/. The PDF renderer uses reportlab; the plotting code uses the existing galaxy extras. No calibration function is called. Plot-only needs NumPy, SciPy and Matplotlib, but not the simulation or hodpy.", "",
             "## LF results", "", "| z range | M_r bin | N | Target LF | HOD LF | Mock LF | HOD residual | Mock residual | Allowed | Status |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for row in required:
        lines.append(f"| {row['z_min']:.2f}-{row['z_max']:.2f} | [{row['magnitude_bright']:.1f},{row['magnitude_faint']:.1f}) | {int(row['count']):,} | {row['target_lf']:.5e} | {row['deterministic_lf']:.5e} | {row['mock_lf']:.5e} | {100*row['deterministic_residual']:+.4f}% | {100*row['mock_residual']:+.3f}% | {100*row['allowed_relative_error']:.2f}% | {'PASS' if row['passed'] else 'REVIEW'} |")
    lines.extend(["", "LF units are h^3 Mpc^-3 mag^-1; residual=(value/target-1). Exact volumes, Poisson/HOD/jackknife errors and all additional bins are in luminosity_function.csv; cumulative checks and completeness estimates are in separate CSVs.", "", "## Figure groups", ""])
    for figure in manifest["figures"]:
        lines += [f"### {figure['title']}", "", f"![{figure['title']}]({figure['file']})", "", figure["caption"], ""]
    lines += ["## Passed checks, discrepancies and likely causes", "", "| Check | Result | Evidence / visible discrepancy | Interpretation or likely cause |", "|---|---|---|---|"]
    for row in checks:
        lines.append(f"| {row['check']} | {row['status']} | {row['evidence']} | {row['cause']} |")
    lines += ["", "References: [hodpy](https://github.com/amjsmith/hodpy), pinned d303bef896fe6a92593d815f640f02865f11df60; [Smith et al. (2017)](https://arxiv.org/abs/1701.06581). These establish the implemented prescriptions, not independent agreement with observed clustering or cluster richness.", ""]
    (output/"report.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="examples/pinocchio_galaxy_diagnostics_config.json")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    settings = json.loads(Path(args.config).read_text())
    output = Path(settings["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        manifest = json.loads((output/"manifest.json").read_text())
        if settings != manifest["settings"]:
            raise ValueError("Plot-only config differs from the binned-data config")
    else:
        manifest = compute(settings, output)
    plot_all(output, manifest)
    write_report(output, manifest)
    code = output/"code"
    code.mkdir(exist_ok=True)
    shutil.copy2(__file__, code/Path(__file__).name)
    shutil.copy2(args.config, code/Path(args.config).name)
    print(f"[diagnostics] report: {output/'report.md'}", flush=True)


if __name__ == "__main__":
    main()
