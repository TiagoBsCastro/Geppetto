"""Numerical real-catalogue LF/colour/integrity validation and diagnostic plots."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import h5py
import jax.numpy as jnp
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import chi2, norm

from geppetto.galaxies.adapters import (
    file_sha256,
    hodpy_modules,
    observed_redshift,
    sky_coordinates,
)
from geppetto.galaxies.calibration import evaluate_occupation, native_hmf_quadrature


def _csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _integrity(data, halos, config, distances):
    host = data["host_index"]
    if np.any(host < 0) or np.any(host >= len(halos.values["M_PIN"])):
        raise ValueError("Galaxy has invalid native host index")
    cent = data["is_central"]
    radial_offset = np.linalg.norm(data["position"]-halos.values["position"][host], axis=1)
    extra_velocity = data["velocity"]-halos.values["velocity"][host]
    sat = ~cent
    normal_velocity = extra_velocity[sat]/data["sigma_sat_km_s"][sat, None]
    chi = np.linalg.norm(data["position"], axis=1)
    source_aperture = float(halos.metadata["parameters"]["PLCAperture"][0])
    buffer_angle = np.deg2rad(source_aperture-config["calibration"]["aperture_deg"])
    redshift_padding = min(config["validation"]["redshift_slices"][0][0]-config["calibration"]["z_min"],
                           config["calibration"]["z_max"]-config["validation"]["redshift_slices"][-1][1])
    checks = dict(
        finite_values=all(np.all(np.isfinite(v)) for v in data.values()),
        unique_galaxy_ids=len(np.unique(data["galaxy_id"])) == len(host),
        valid_host_ids=np.array_equal(data["host_halo_id"], halos.values["occurrence_id"][host]),
        native_group_ids_preserved=np.array_equal(data["host_group_id"], halos.values["group_id"][host]),
        native_host_mass_preserved=np.array_equal(data["M_PIN"], halos.values["M_PIN"][host]),
        at_most_one_central=len(np.unique(host[cent])) == int(np.sum(cent)),
        central_position_exact=np.array_equal(data["position"][cent], halos.values["position"][host[cent]]),
        central_velocity_exact=np.array_equal(data["velocity"][cent], halos.values["velocity"][host[cent]]),
        central_redshift_exact=np.array_equal(data["z_cos"][cent], halos.values["z_cos"][host[cent]]),
        satellite_support=np.all(radial_offset <= data["R_sat_comoving"]+1.e-10),
        satellite_offsets_consistent=np.allclose(radial_offset, data["satellite_radius_comoving"], atol=1.e-10, rtol=1.e-9),
        angular_buffer_sufficient=bool(np.max(np.arcsin(np.minimum(1., data["R_sat_comoving"]/chi))) < buffer_angle),
        redshift_buffer_sufficient=bool(np.max(np.abs(data["z_cos"]-data["host_z_cos"]), initial=0) < redshift_padding),
        observed_redshift_consistent=np.allclose(data["z_obs"], observed_redshift(data["position"], data["velocity"], data["z_cos"]), atol=1.e-14, rtol=0),
        physical_ranges=bool(np.all((data["z_cos"] > 0) & (data["c_sat"] > 0) & (data["R_sat_comoving"] > 0)
                                      & (data["ra_deg"] >= 0) & (data["ra_deg"] < 360)
                                      & (data["dec_deg"] >= -90) & (data["dec_deg"] <= 90)
                                      & (data["g_r_rest_0p1"] > -.5) & (data["g_r_rest_0p1"] < 2.))),
        magnitude_domain=bool(np.all((data["absolute_magnitude_r"] >= config["calibration"]["magnitude_bright"])
                                     & (data["absolute_magnitude_r"] < config["calibration"]["magnitude_faint"]))),
    )
    return dict(checks={key: bool(value) for key, value in checks.items()},
                passed=all(checks.values()), n_galaxies=len(host), n_centrals=int(np.sum(cent)),
                n_satellites=int(np.sum(sat)), satellite_velocity_standardized_mean=np.mean(normal_velocity, axis=0).tolist(),
                satellite_velocity_standardized_std=np.std(normal_velocity, axis=0).tolist(),
                max_satellite_radius_fraction=float(np.max(radial_offset/data["R_sat_comoving"])),
                distance_units="Mpc/h comoving; sigma_v uses physical radius R_com/(1+z)",
                min_colour=float(data["g_r_rest_0p1"].min()), max_colour=float(data["g_r_rest_0p1"].max()))


def conditional_lf_moments(halos, table, config):
    """Exact unbinned halo-sum expectations and HOD sampling variances.

    In each magnitude bin, independent central Bernoulli and satellite Poisson
    sampling has variance sum[p_bin(1-p_bin)+lambda_bin]. This is conditional
    on the actual halo catalogue, not an estimate of cosmological variance.
    """
    edges = np.asarray(config["validation"]["magnitude_edges"])
    edge_index = np.array([np.argmin(abs(table.magnitudes-m)) for m in edges])
    result = []
    _, latitude = sky_coordinates(halos.values["position"], halos.basis)
    resolved = (halos.values["particle_count"] >= config["calibration"]["min_particles"])
    resolved &= latitude >= 90-config["calibration"]["aperture_deg"]
    for zlo, zhi in config["validation"]["redshift_slices"]:
        selection = resolved & (halos.values["z_cos"] >= zlo) & (halos.values["z_cos"] < zhi)
        indices = np.flatnonzero(selection)
        cells = table.cell(halos.values["z_cos"][indices])
        cumulative = np.zeros(len(edges))
        variance = np.zeros(len(edges)-1)
        for cell in np.unique(cells):
            chosen = indices[cells == cell]
            for first in range(0, len(chosen), 8192):
                mass = halos.values["M_PIN"][chosen[first:first+8192]]
                central, satellite = [np.asarray(x) for x in evaluate_occupation(jnp.asarray(np.log10(mass)),
                    jnp.asarray(table.thresholds[cell, edge_index]), jnp.asarray(table.log_shifts[cell, edge_index]))]
                cumulative += np.sum(central+satellite, axis=0)
                probability = np.diff(central, axis=1)
                rate = np.diff(satellite, axis=1)
                if np.any(probability < -1.e-9) or np.any(rate < -1.e-9):
                    raise ValueError("Non-nested HOD in unbinned validation")
                variance += np.sum(probability*(1-probability)+rate, axis=0)
        result.append((cumulative, variance))
    return result


def lf_validation(data, halos, distances, target, table, config):
    """Compare fixed LF, unbinned deterministic prediction and final mock counts."""
    settings = config["validation"]
    edges = np.asarray(settings["magnitude_edges"])
    widths = np.diff(edges)
    aperture = config["calibration"]["aperture_deg"]
    angular = data["dec_deg"] >= 90-aperture
    expected = conditional_lf_moments(halos, table, config)
    n_az, n_rad = settings["jackknife_azimuth_regions"], settings["jackknife_radial_regions"]
    regions = n_az*n_rad
    area_coordinate = (1-np.sin(np.deg2rad(data["dec_deg"])))/(1-np.cos(np.deg2rad(aperture)))
    area_coordinate = np.clip(area_coordinate, 0, np.nextafter(1., 0.))
    region = np.floor(data["ra_deg"]*n_az/360).astype(int)+n_az*np.floor(area_coordinate*n_rad).astype(int)
    rows, cumulative_rows = [], []
    for index, (zlo, zhi) in enumerate(settings["redshift_slices"]):
        volume = distances.volume(zlo, zhi, aperture)
        target_cum = distances.volume_average(lambda z: target.cumulative(edges[:, None], z), zlo, zhi)
        prediction_cum, variance = expected[index]
        prediction_cum /= volume
        target_bin = np.diff(target_cum)/widths
        predicted_bin = np.diff(prediction_cum)/widths
        selection = angular & (data["z_cos"] >= zlo) & (data["z_cos"] < zhi)
        apparent_complete = bool(np.all(data["selected_r_limit"][selection & (data["absolute_magnitude_r"] >= edges[0]) & (data["absolute_magnitude_r"] < edges[-1])]))
        for m, truth, prediction in zip(edges, target_cum, prediction_cum, strict=True):
            residual = prediction/truth-1
            cumulative_rows.append(dict(z_min=zlo, z_max=zhi, magnitude_threshold=m,
                                        target_cumulative=truth, deterministic_cumulative=prediction,
                                        residual=residual, passed=bool(abs(residual) <= settings["cumulative_prediction_rtol"])))
        for j, (bright, faint) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
            keep = selection & (data["absolute_magnitude_r"] >= bright) & (data["absolute_magnitude_r"] < faint)
            keep &= data["selected_r_limit"]
            count = int(np.sum(keep))
            region_counts = np.bincount(region[keep], minlength=regions)
            jackknife = (count-region_counts)/(volume*(regions-1)/regions*widths[j])
            jackknife_sigma = np.sqrt((regions-1)/regions*np.sum((jackknife-np.mean(jackknife))**2))
            hod_sigma = np.sqrt(variance[j])/(volume*widths[j])
            sigma = max(hod_sigma, jackknife_sigma)
            observed = count/(volume*widths[j])
            residual = observed/target_bin[j]-1
            tolerance = max(settings["mock_rtol_floor"], settings["mock_sigma_multiplier"]*sigma/target_bin[j])
            rows.append(dict(z_min=zlo, z_max=zhi, magnitude_bright=bright, magnitude_faint=faint,
                             volume_mpc_h3=volume, count=count, target_lf=target_bin[j],
                             deterministic_lf=predicted_bin[j], mock_lf=observed,
                             deterministic_residual=predicted_bin[j]/target_bin[j]-1,
                             mock_residual=residual, poisson_sigma=np.sqrt(count)/(volume*widths[j]),
                             hod_sigma=hod_sigma, jackknife_sigma=jackknife_sigma,
                             acceptance_sigma=sigma, allowed_relative_error=tolerance,
                             apparent_complete=apparent_complete,
                             passed=bool(count >= settings["minimum_bin_count"] and apparent_complete and abs(residual) <= tolerance)))
    return rows, cumulative_rows


def colour_validation(data, config):
    """Conditional mixture tests with a predeclared familywise error control.

    The red-count uncertainty is sum[p(1-p)]; standardized Gaussian means and
    variances use their exact Normal and chi-square sampling distributions.
    A small-variance red count uses a fixed-seed Bernoulli Monte Carlo tail.
    """
    model = hodpy_modules(config["hodpy_root"])["colour"].Colour()
    magnitude, redshift = data["absolute_magnitude_r"], data["z_cos"]
    red_sat = model.probability_red_satellite(magnitude, redshift)
    blue_sat = 1-red_sat
    blue_central = blue_sat + (model.fraction_blue(magnitude, redshift)-blue_sat)/model.fraction_central(magnitude, redshift)
    red_probability = np.clip(np.where(data["is_central"], 1-blue_central, red_sat), 0., 1.)
    sequence_mean = np.where(data["is_red_draw"], model.red_mean(magnitude, redshift), model.blue_mean(magnitude, redshift))
    sequence_sigma = np.where(data["is_red_draw"], model.red_rms(magnitude, redshift), model.blue_rms(magnitude, redshift))
    # Recompute from upstream, never validate a draw against only its own labels.
    for key, expected in (("colour_red_probability", red_probability),
                          ("colour_sequence_mean", sequence_mean),
                          ("colour_sequence_sigma", sequence_sigma)):
        if not np.allclose(data[key], expected, atol=1.e-14, rtol=1.e-13):
            raise ValueError(f"Stored {key} disagrees with the independent upstream colour model")
    settings = config["validation"]
    rows = []
    angular = data["dec_deg"] >= 90-config["calibration"]["aperture_deg"]
    rng = np.random.default_rng(config["random_seed"]+941)
    for zlo, zhi in settings["redshift_slices"]:
        shell = angular & (data["z_cos"] >= zlo) & (data["z_cos"] < zhi) & data["selected_r_limit"]
        for bright, faint in zip(settings["magnitude_edges"][:-1], settings["magnitude_edges"][1:], strict=True):
            magnitude_selection = (data["absolute_magnitude_r"] >= bright) & (data["absolute_magnitude_r"] < faint)
            for central in (True, False):
                selected = shell & magnitude_selection & (data["is_central"] == central)
                n = int(np.sum(selected))
                base = dict(z_min=zlo, z_max=zhi, magnitude_bright=bright, magnitude_faint=faint,
                            status="central" if central else "satellite")
                if n == 0:
                    rows.append(dict(**base, component="mixture", statistic="empty_status_bin", n=0,
                                     measured=None, expected=None, standard_error=None, p_value=None))
                    continue
                probability = red_probability[selected]
                observed_red = int(np.sum(data["is_red_draw"][selected]))
                mean_red, variance = float(np.sum(probability)), float(np.sum(probability*(1-probability)))
                deviation = abs(observed_red-mean_red)
                if variance == 0:
                    pvalue = 1. if deviation < 1.e-12 else 0.
                elif variance >= 10:
                    pvalue = float(2*norm.sf(deviation/np.sqrt(variance)))
                else:
                    trials = 50000
                    exceed = 0
                    for first in range(0, trials, 250):
                        simulation = np.sum(rng.random((min(250, trials-first), n)) < probability, axis=1)
                        exceed += np.count_nonzero(abs(simulation-mean_red) >= deviation-1.e-12)
                    pvalue = (exceed+1)/(trials+1)
                rows.append(dict(**base, component="mixture", statistic="red_fraction", n=n,
                                 measured=observed_red/n, expected=mean_red/n,
                                 standard_error=np.sqrt(variance)/n, p_value=pvalue))
                for red in (True, False):
                    sample = selected & (data["is_red_draw"] == red)
                    count = int(np.sum(sample))
                    if count < 2:
                        continue
                    value = (data["g_r_rest_0p1"][sample]-sequence_mean[sample])/sequence_sigma[sample]
                    average, sample_variance = float(np.mean(value)), float(np.var(value, ddof=1))
                    rows.append(dict(**base, component="red" if red else "blue", statistic="standardized_mean",
                                     n=count, measured=average, expected=0., standard_error=1/np.sqrt(count),
                                     p_value=float(2*norm.sf(abs(average)*np.sqrt(count)))))
                    q = (count-1)*sample_variance
                    rows.append(dict(**base, component="red" if red else "blue", statistic="standardized_variance",
                                     n=count, measured=sample_variance, expected=1., standard_error=np.sqrt(2/(count-1)),
                                     p_value=float(min(1., 2*min(chi2.cdf(q, count-1), chi2.sf(q, count-1))))))
    tested = sum(row["n"] > 0 for row in rows)
    if tested == 0:
        raise ValueError("No populated colour-validation bins")
    cutoff = settings["colour_familywise_alpha"]/tested
    for row in rows:
        row["p_value_cutoff"] = cutoff
        row["passed"] = row["p_value"] >= cutoff if row["n"] > 0 else None
    return rows


def angular_clustering_diagnostic(data, config):
    """Subsampled angular Landy-Szalay diagnostic; never used for calibration."""
    rng = np.random.default_rng(config["random_seed"]+712)
    maximum = config["validation"]["clustering_max_galaxies"]
    aperture = np.deg2rad(config["calibration"]["aperture_deg"])
    angle_edges = np.geomspace(.05, 10., 17)
    chord_edges = 2*np.sin(np.deg2rad(angle_edges)/2)
    rows = []
    for zlo, zhi in config["validation"]["redshift_slices"]:
        selected = ((data["z_cos"] >= zlo) & (data["z_cos"] < zhi)
                    & (data["dec_deg"] >= 90-np.rad2deg(aperture)) & data["selected_r_limit"]
                    & (data["absolute_magnitude_r"] < config["validation"]["magnitude_edges"][-1]))
        indices = np.flatnonzero(selected)
        if len(indices) > maximum:
            indices = rng.choice(indices, maximum, replace=False)
        lon, lat = np.deg2rad(data["ra_deg"][indices]), np.deg2rad(data["dec_deg"][indices])
        points = np.column_stack((np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)))
        count = len(points)
        random_count = 2*count
        mu = rng.uniform(np.cos(aperture), 1., random_count)
        phi = rng.uniform(0, 2*np.pi, random_count)
        random = np.column_stack((np.sqrt(1-mu**2)*np.cos(phi), np.sqrt(1-mu**2)*np.sin(phi), mu))
        tree, rtree = cKDTree(points), cKDTree(random)
        dd = np.diff(tree.count_neighbors(tree, chord_edges))/(count*(count-1))
        rr = np.diff(rtree.count_neighbors(rtree, chord_edges))/(random_count*(random_count-1))
        dr = np.diff(tree.count_neighbors(rtree, chord_edges))/(count*random_count)
        correlation = (dd-2*dr+rr)/rr
        for i, value in enumerate(correlation):
            rows.append(dict(z_min=zlo, z_max=zhi, theta_min_deg=angle_edges[i], theta_max_deg=angle_edges[i+1],
                             w_theta=float(value), n_data=count, n_random=random_count))
    return rows


def native_input_diagnostics(halos, distances, table, config, output):
    """Publish native HMF and fixed-HOD mass-cut/identity sensitivity.

    Raising the cut without recalibration measures the omitted contribution;
    it is not a resolution-convergence test of the native halo finder.
    """
    values = halos.values
    settings = config["calibration"]
    _, latitude = sky_coordinates(values["position"], halos.basis)
    angular = latitude >= 90-settings["aperture_deg"]
    log_edges = np.arange(np.log10(values["M_PIN"].min())-1.e-7,
                          np.log10(values["M_PIN"].max())+2*settings["mass_bin_width_dex"],
                          settings["mass_bin_width_dex"])
    edges = np.asarray(config["validation"]["magnitude_edges"])
    edge_index = np.array([np.argmin(abs(table.magnitudes-m)) for m in edges])
    hmf_rows, sensitivity = [], []
    for cell, (zlo, zhi) in enumerate(zip(table.redshift_edges[:-1], table.redshift_edges[1:], strict=True)):
        selection = angular & (values["z_cos"] >= zlo) & (values["z_cos"] < zhi)
        volume = distances.volume(zlo, zhi, settings["aperture_deg"])
        mass, weight = native_hmf_quadrature(values["M_PIN"], selection, log_edges, volume)
        for j in np.flatnonzero(weight):
            hmf_rows.append(dict(z_min=zlo, z_max=zhi, volume_mpc_h3=volume,
                                 log10_mass_min=log_edges[j], log10_mass_max=log_edges[j+1],
                                 mean_log10_mass=mass[j], count=int(round(weight[j]*volume)),
                                 number_density=weight[j], dn_dlog10_mass=weight[j]/np.diff(log_edges)[j]))
        omitted = selection & (values["particle_count"] >= settings["min_particles"]) & (values["particle_count"] < 64)
        mass, weight = native_hmf_quadrature(values["M_PIN"], omitted, log_edges, volume)
        central, satellite = evaluate_occupation(jnp.asarray(mass), jnp.asarray(table.thresholds[cell, edge_index]),
                                                  jnp.asarray(table.log_shifts[cell, edge_index]))
        lost = np.diff(weight @ np.asarray(central+satellite))
        reference = np.diff(table.prediction[cell, edge_index])
        for j, fraction in enumerate(lost/reference):
            sensitivity.append(dict(z_min=zlo, z_max=zhi, magnitude_bright=edges[j], magnitude_faint=edges[j+1],
                                    baseline_min_particles=settings["min_particles"], raised_min_particles=64,
                                    fraction_removed_without_refit=float(fraction)))
    _csv(output/"native_hmf.csv", hmf_rows)
    _csv(output/"mass_cut_sensitivity.csv", sensitivity)
    ids, inverse, counts = np.unique(values["group_id"], return_inverse=True, return_counts=True)
    repeated = np.flatnonzero(counts[inverse] > 1)
    repeated = repeated[np.argsort(values["group_id"][repeated], kind="stable")]
    same = np.diff(values["group_id"][repeated]) == 0
    separation = np.linalg.norm(np.diff(values["position"][repeated], axis=0), axis=1)[same]
    diagnostics = dict(n_native_records=len(inverse), n_distinct_group_ids=len(ids),
                       repeated_group_ids=int(np.sum(counts > 1)), extra_group_occurrences=int(np.sum(counts-1)),
                       repeated_position_separation_mpc_h_quantiles=(np.quantile(separation, [0, .5, 1]).tolist() if len(separation) else []),
                       identity_policy="Preserve native PLC occurrences; host_halo_id=(part<<48)+row, host_group_id is the persistent native identifier. No mass- or redshift-based deduplication.",
                       mass_cut_interpretation="64-particle diagnostic holds the calibrated HOD fixed; not a physical halo-finder convergence test")
    (output/"native_input_diagnostics.json").write_text(json.dumps(diagnostics, indent=2)+"\n")
    return diagnostics


def diagnostic_plots(data, lf_rows, colour_rows, clustering, config, output):
    """Render reproducible scientific PNG figures; no observed-clustering fit."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "figure.dpi": 130,
                         "savefig.dpi": 180, "axes.spines.top": False, "axes.spines.right": False})
    slices = config["validation"]["redshift_slices"]
    fig, axes = plt.subplots(2, len(slices), figsize=(10.4, 6.6), sharex="col",
                             gridspec_kw={"height_ratios": [3, 1]}, layout="constrained", squeeze=False)
    for i, (zlo, zhi) in enumerate(slices):
        rows = [r for r in lf_rows if r["z_min"] == zlo]
        m = np.array([.5*(r["magnitude_bright"]+r["magnitude_faint"]) for r in rows])
        truth, prediction, measured, sigma = [np.array([r[key] for r in rows]) for key in (
            "target_lf", "deterministic_lf", "mock_lf", "acceptance_sigma")]
        ax = axes[0, i]
        ax.plot(m, truth, "k-", lw=1.8, label="Independent SDSS/GAMA target")
        ax.plot(m, prediction, "s", ms=7, mfc="none", color="#16877b", label="Native-mass HOD prediction")
        ax.errorbar(m, measured, sigma, fmt="o", ms=4, color="#c04c38", capsize=3, label="PINOCCHIO galaxy mock")
        ax.set(yscale="log", title=f"{zlo:.2f} < z < {zhi:.2f}")
        ax.grid(alpha=.2)
        axes[1, i].axhspan(-.05, .05, color=".92")
        axes[1, i].axhline(0, color="black", lw=.8)
        axes[1, i].errorbar(m, measured/truth-1, sigma/truth, fmt="o", color="#c04c38", capsize=3)
        axes[1, i].plot(m, prediction/truth-1, "s", mfc="none", color="#16877b")
        axes[1, i].set_xlabel(r"$^{0.1}M_r-5\log_{10}h$")
    axes[0, 0].set_ylabel(r"$\phi\ [(h/{\rm Mpc})^3\,{\rm mag}^{-1}]$")
    axes[1, 0].set_ylabel("Fractional residual")
    axes[0, 0].legend(fontsize=8, loc="upper left")
    fig.savefig(output/"luminosity_function.png")
    plt.close(fig)
    fig, axes = plt.subplots(1, len(slices), figsize=(10.4, 4.5), sharey=True, layout="constrained", squeeze=False)
    for i, (zlo, zhi) in enumerate(slices):
        keep = ((data["z_cos"] >= zlo) & (data["z_cos"] < zhi)
                & (data["dec_deg"] >= 90-config["calibration"]["aperture_deg"])
                & data["selected_r_limit"])
        image = axes[0, i].hist2d(data["absolute_magnitude_r"][keep], data["g_r_rest_0p1"][keep],
                                 bins=(55, 55), range=((-23.2, -21.5), (.35, 1.25)), norm=LogNorm(), cmap="viridis")
        axes[0, i].axvline(config["validation"]["magnitude_edges"][-1], color="white", ls="--", lw=1)
        axes[0, i].set(title=f"{zlo:.2f} < z < {zhi:.2f}", xlabel=r"$^{0.1}M_r-5\log_{10}h$")
        axes[0, i].set_xticks([-23., -22.5, -22., -21.5])
        fig.colorbar(image[3], ax=axes[0, i], label="Galaxies per bin")
    axes[0, 0].set_ylabel(r"Rest-frame $^{0.1}(g-r)$")
    fig.savefig(output/"colour_magnitude.png")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    for zlo, zhi in slices:
        rows = [r for r in clustering if r["z_min"] == zlo]
        x = [np.sqrt(r["theta_min_deg"]*r["theta_max_deg"]) for r in rows]
        ax.plot(x, [r["w_theta"] for r in rows], "o-", ms=4, label=f"{zlo:.2f} < z < {zhi:.2f}")
    ax.axhline(0, color=".5", lw=.8)
    ax.set(xscale="log", xlabel=r"$\theta$ [degrees]", ylabel=r"$w(\theta)$", title="Angular clustering diagnostic (no fit)")
    ax.legend()
    fig.savefig(output/"angular_clustering.png")
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), layout="constrained")
    for ax, statistic, expected in zip(axes, ("red_fraction", "standardized_mean", "standardized_variance"), (None, 0., 1.), strict=True):
        rows = [r for r in colour_rows if r["statistic"] == statistic]
        y = np.array([r["measured"]-(r["expected"] if expected is None else 0.) for r in rows])
        error = np.array([r["standard_error"] for r in rows])
        ax.errorbar(np.arange(len(rows)), y, error, fmt="o", ms=3, capsize=2, color="#16877b")
        ax.axhline(0. if expected is None else expected, color="black", lw=.9)
        ax.set(title=statistic.replace("_", " "), xlabel="Validation group")
    fig.savefig(output/"colour_validation.png")
    plt.close(fig)


def validate_catalogue(config, path, halos, distances, target, table):
    """Write all results before failing any unmet completion condition."""
    output = Path(config["output_dir"])
    (output/"validation_summary.json").write_text(json.dumps(dict(catalogue=str(path), passed=False,
                                                                 status="validation in progress; an interrupted run is not a pass"), indent=2)+"\n")
    with h5py.File(path) as handle:
        data = {key: np.asarray(value) for key, value in handle["galaxies"].items()}
    print("[galaxy validation] integrity, independent LF and conditional colour tests", flush=True)
    integrity = _integrity(data, halos, config, distances)
    lf_rows, cumulative = lf_validation(data, halos, distances, target, table, config)
    colours = colour_validation(data, config)
    native_diagnostics = native_input_diagnostics(halos, distances, table, config, output)
    print("[galaxy validation] angular clustering diagnostic", flush=True)
    clustering = angular_clustering_diagnostic(data, config)
    _csv(output/"lf_validation.csv", lf_rows)
    _csv(output/"cumulative_lf_validation.csv", cumulative)
    _csv(output/"colour_validation.csv", colours)
    _csv(output/"angular_clustering.csv", clustering)
    diagnostic_plots(data, lf_rows, colours, clustering, config, output)
    summary = dict(catalogue=str(path), catalogue_sha256=file_sha256(path), integrity=integrity,
                   native_input=native_diagnostics, colour_expectations_recomputed_from_upstream=True,
                   lf_passed=all(r["passed"] for r in lf_rows),
                   cumulative_lf_passed=all(r["passed"] for r in cumulative),
                   colour_passed=all(r["passed"] for r in colours if r["n"] > 0),
                   n_colour_tests=sum(r["n"] > 0 for r in colours),
                   empty_colour_status_bins=[r for r in colours if r["n"] == 0],
                   uncertainty="max(conditional Bernoulli+Poisson HOD sigma, 32-region equal-solid-angle jackknife sigma); 3 sigma or 5%, whichever is larger",
                   interpretation="LF is calibrated conditional on the native PLC HMF; clustering is diagnostic only, not fitted",
                   figures=["luminosity_function.png", "colour_magnitude.png", "colour_validation.png", "angular_clustering.png"])
    summary["passed"] = all((summary["lf_passed"], summary["cumulative_lf_passed"], summary["colour_passed"], integrity["passed"]))
    (output/"validation_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2), flush=True)
    if not summary["passed"]:
        raise ValueError("Galaxy validation has unmet checks; inspect the written numerical results")
    return summary
