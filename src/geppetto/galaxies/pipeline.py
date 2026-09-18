"""Chunked stochastic native-mass galaxy painting, entirely on the host."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np

from geppetto.galaxies.adapters import (
    file_sha256,
    hodpy_modules,
    observed_redshift,
    sample_colours,
    sky_coordinates,
)
from geppetto.galaxies.calibration import evaluate_occupation
from geppetto.galaxies.models import SatelliteParams, nfw_inverse_cdf, satellite_scales


def inverse_cumulative_rows(magnitudes, cumulative, quantile):
    """Invert monotone per-object cumulative luminosity counts by interpolation.

    Counts increase toward fainter (more positive) magnitudes. Quantiles must
    lie strictly between zero and the faint-end cumulative count. No MXXL
    luminosity lookup, extrapolation, or post-hoc magnitude remapping is used.
    """
    if cumulative.shape != (len(quantile), len(magnitudes)):
        raise ValueError("Cumulative luminosity table shape mismatch")
    if (np.any(~np.isfinite(cumulative)) or np.any(np.diff(cumulative, axis=1) < -1.e-9)
            or np.any(quantile <= 0) or np.any(quantile >= cumulative[:, -1])):
        raise ValueError("Invalid or non-nested luminosity CDF/quantile")
    if len(quantile) == 0:
        return np.empty(0)
    index = np.sum(cumulative < quantile[:, None], axis=1)
    index = np.clip(index, 1, len(magnitudes)-1)
    row = np.arange(len(quantile))
    low, high = cumulative[row, index-1], cumulative[row, index]
    if np.any(high <= low):
        raise ValueError("Cannot invert a flat luminosity CDF interval")
    fraction = (quantile-low)/(high-low)
    return magnitudes[index-1]+fraction*(magnitudes[index]-magnitudes[index-1])


def sample_luminosities(magnitudes, central, satellite, rng):
    """Draw at most one central and Poisson satellites, preserving hodpy means.

    Central and satellite occupations are independent draws as in hodpy.
    A satellite may have a central fainter than the underlying magnitude cut.
    This does not imply a second central or an orphan halo ID. The far-bright
    endpoint is a zero-CDF sentinel whose omitted expected count is audited.
    """
    central, satellite = central.copy(), satellite.copy()
    if (np.min(central, initial=0) < -1.e-10 or np.max(central, initial=1) > 1+1.e-10
            or np.min(satellite, initial=0) < -1.e-10):
        raise ValueError("Invalid occupation probability/mean")
    central[:, 0], satellite[:, 0] = 0., 0.
    central = np.clip(central, 0., 1.)
    satellite = np.maximum(satellite, 0.)
    u = rng.random(len(central))
    selected = (u > 0) & (u < central[:, -1])
    central_hosts = np.flatnonzero(selected)
    central_mags = inverse_cumulative_rows(magnitudes, central[selected], u[selected])
    number = rng.poisson(satellite[:, -1])
    satellite_hosts = np.repeat(np.arange(len(number)), number)
    # nextafter excludes the zero endpoint without changing the continuous law.
    uniform = np.maximum(rng.random(len(satellite_hosts)), np.nextafter(0., 1.))
    quantile = uniform*satellite[satellite_hosts, -1]
    satellite_mags = inverse_cumulative_rows(magnitudes, satellite[satellite_hosts], quantile)
    return (np.concatenate((central_hosts, satellite_hosts)),
            np.concatenate((central_mags, satellite_mags)),
            np.arange(len(central_hosts)+len(satellite_hosts)) < len(central_hosts))


_inverse_radius = jax.jit(nfw_inverse_cdf)
_satellite_scales = jax.jit(satellite_scales)


def _sample_radii(uniform, concentration, chunk_size=4096):
    """Use a fixed-size JAX bucket, avoiding one compilation per random count."""
    result = np.empty(len(uniform))
    for first in range(0, len(uniform), chunk_size):
        count = min(chunk_size, len(uniform)-first)
        u = np.full(chunk_size, .5)
        c = np.full(chunk_size, 5.)
        u[:count], c[:count] = uniform[first:first+count], concentration[first:first+count]
        result[first:first+count] = np.asarray(_inverse_radius(jnp.asarray(u), jnp.asarray(c)))[:count]
    return result


def populate_halo_chunk(indices, halos, distances, table, cell, rng, satellite_params, hodpy_root, r_limit,
                        *, chunk_size=None, kcorr=None):
    """Return one galaxy chunk, with positions comoving Mpc/h and proper km/s."""
    values = halos.values
    chunk_size = len(indices) if chunk_size is None else chunk_size
    mass = np.pad(values["M_PIN"][indices], (0, chunk_size-len(indices)), mode="edge")
    redshift = np.pad(values["z_cos"][indices], (0, chunk_size-len(indices)), mode="edge")
    log_mass = np.log10(mass)
    central, satellite = [np.asarray(value) for value in evaluate_occupation(
        jnp.asarray(log_mass), jnp.asarray(table.thresholds[cell]), jnp.asarray(table.log_shifts[cell]),
    )]
    central, satellite = central[:len(indices)], satellite[:len(indices)]
    hosts_local, magnitude, is_central = sample_luminosities(table.magnitudes, central, satellite, rng)
    hosts = indices[hosts_local]
    n = len(hosts)
    position, velocity = values["position"][hosts].copy(), values["velocity"][hosts].copy()
    halo_z = values["z_cos"][hosts]
    z_cos = halo_z.copy()
    radius, concentration, sigma = [np.asarray(value)[hosts_local] for value in _satellite_scales(
        jnp.asarray(mass), jnp.asarray(redshift), halos.mean_density, satellite_params,
    )]
    satellite_radius = np.zeros(n)
    satellite_index = ~is_central
    count = int(np.sum(satellite_index))
    if count:
        radius_fraction = _sample_radii(rng.random(count), concentration[satellite_index])
        direction = rng.standard_normal((count, 3))
        direction /= np.linalg.norm(direction, axis=1)[:, None]
        satellite_radius[satellite_index] = radius_fraction*radius[satellite_index]
        old_chi = np.linalg.norm(position[satellite_index], axis=1)
        position[satellite_index] += direction*satellite_radius[satellite_index, None]
        new_chi = np.linalg.norm(position[satellite_index], axis=1)
        # Keep native central redshifts exactly and use the table for displacement.
        z_cos[satellite_index] += distances.redshift(new_chi)-distances.redshift(old_chi)
        velocity[satellite_index] += sigma[satellite_index, None]*rng.standard_normal((count, 3))
    z_obs = observed_redshift(position, velocity, z_cos)
    ra, dec = sky_coordinates(position, halos.basis)
    colour, is_red, p_red, colour_mean, colour_width = sample_colours(magnitude, z_cos, is_central, rng, hodpy_root)
    if kcorr is None:
        kcorr = hodpy_modules(hodpy_root)["k_correction"].GAMA_KCorrection(distances)
    apparent = kcorr.apparent_magnitude(magnitude, z_cos, colour)
    return dict(
        host_index=hosts, host_halo_id=values["occurrence_id"][hosts], host_group_id=values["group_id"][hosts],
        M_PIN=values["M_PIN"][hosts], is_central=is_central, position=position, velocity=velocity,
        ra_deg=ra, dec_deg=dec, z_cos=z_cos, z_obs=z_obs, host_z_cos=halo_z,
        absolute_magnitude_r=magnitude, apparent_magnitude_r=apparent, g_r_rest_0p1=colour,
        is_red_draw=is_red, colour_red_probability=p_red, colour_sequence_mean=colour_mean,
        colour_sequence_sigma=colour_width, satellite_radius_comoving=satellite_radius,
        R_sat_comoving=radius, c_sat=concentration, sigma_sat_km_s=sigma,
        selected_r_limit=apparent < r_limit,
    ), float(np.sum(central[:, 0]+satellite[:, 0]))


def generate_catalogue(config, halos, distances, target, table) -> Path:
    """Write a complete underlying bright catalogue and separate survey selection.

    No object is dropped on apparent magnitude before luminosity/colour sampling.
    Bounded halo chunks share one explicit RNG. Reproducibility includes the
    seed, chunk size, configuration, upstream hashes and calibration inputs.
    """
    satellite_params = SatelliteParams(**config.get("satellites", {}))
    if any(value <= 0 for value in (satellite_params.overdensity_mean, satellite_params.radius_factor,
                                    satellite_params.concentration_amplitude, satellite_params.concentration_mass_pivot,
                                    satellite_params.gravitational_constant)) or satellite_params.velocity_factor < 0:
        raise ValueError("Satellite scales and concentration must be positive")
    size = int(config["halo_chunk_size"])
    if size < 1 or not np.isfinite(config["survey_r_limit"]):
        raise ValueError("Invalid chunk size or survey limit")
    recipe = dict(config=config, calibration_key=table.cache_key, pipeline_sha256=file_sha256(__file__),
                  numpy_version=np.__version__, jax_version=jax.__version__)
    run_key = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
    directory = Path(config["output_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/f"galaxies_{run_key[:16]}.hdf5"
    if path.exists():
        with h5py.File(path) as handle:
            saved_metadata = json.loads(handle.attrs["metadata_json"])
            if saved_metadata["run_key"] != run_key:
                raise ValueError("Galaxy catalogue run-key mismatch")
        (directory/"catalogue.json").write_text(json.dumps(dict(path=str(path), run_key=run_key,
                                                                n_galaxies=saved_metadata["n_galaxies"]), indent=2)+"\n")
        print(f"[galaxies] existing reproducible catalogue: {path}", flush=True)
        return path
    rng = np.random.default_rng(config["random_seed"])
    kcorr = hodpy_modules(config["hodpy_root"])["k_correction"].GAMA_KCorrection(distances)
    mask = (halos.values["particle_count"] >= config["calibration"]["min_particles"])
    mask &= (halos.values["z_cos"] >= table.redshift_edges[0]) & (halos.values["z_cos"] < table.redshift_edges[-1])
    selected = np.flatnonzero(mask)
    cells = table.cell(halos.values["z_cos"][selected])
    total, bright_tail = 0, 0.
    temporary = tempfile.NamedTemporaryFile(prefix="galaxies_", suffix=".partial.hdf5", dir=directory, delete=False)
    temporary.close()
    with h5py.File(temporary.name, "w") as handle:
        group = handle.create_group("galaxies")
        for cell in range(len(table.redshift_edges)-1):
            indices = selected[cells == cell]
            for first in range(0, len(indices), size):
                block, tail = populate_halo_chunk(indices[first:first+size], halos, distances, table, cell,
                                                  rng, satellite_params, config["hodpy_root"], config["survey_r_limit"],
                                                  chunk_size=size, kcorr=kcorr)
                bright_tail += tail
                n = len(block["M_PIN"])
                block["galaxy_id"] = np.arange(total, total+n, dtype=np.uint64)
                for name, value in block.items():
                    if name not in group:
                        group.create_dataset(name, shape=(0, *value.shape[1:]), maxshape=(None, *value.shape[1:]),
                                             dtype=value.dtype, chunks=True, compression="gzip", shuffle=True)
                    dataset = group[name]
                    dataset.resize(total+n, axis=0)
                    dataset[total:] = value
                total += n
            print(f"[galaxies] generated cell {cell+1}/{len(table.redshift_edges)-1}, cumulative galaxies={total}", flush=True)
        if bright_tail > .1:
            raise ValueError(f"Bright luminosity endpoint omits {bright_tail} expected objects; extend the domain")
        metadata = dict(
            **recipe, run_key=run_key, n_galaxies=total, expected_omitted_bright_tail=bright_tail,
            native_input=halos.metadata, mass_definition=halos.metadata["mass_definition"],
            particle_mass_msun_h=halos.particle_mass, mean_density_comoving=halos.mean_density,
            satellite_parameters=satellite_params._asdict(),
            target_parameters=target.parameters, target_input_hashes=target.input_hashes,
            colour_reference_redshift=.1, magnitude_reference_redshift=.1,
            absolute_magnitude_convention="^0.1 M_r - 5 log10(h); AB, hodpy SDSS/GAMA r band",
            position_units="observer-relative comoving Mpc/h; native simulation Cartesian basis",
            velocity_units="proper peculiar km/s; native simulation Cartesian basis",
            sky_coordinates="cone-aligned longitude/latitude in degrees; not ICRS",
            sky_basis_rows=halos.basis.tolist(), observed_redshift="z_cos + (1+z_cos) dot(v,n)/299792.458",
            apparent_magnitude="M_r_h1 + 5 log10((1+z_cos) chi_mpc_h) +25 + K_r(z_cos, ^0.1(g-r))",
            selection="Underlying bright absolute-magnitude sample; selected_r_limit is applied only after painting. Not a complete faint flux-limited survey.",
            occupation_sampling="Independent central Bernoulli and satellite Poisson, as hodpy; unobserved centrals may be fainter than the underlying cut",
        )
        handle.attrs["metadata_json"] = json.dumps(metadata, sort_keys=True)
        for name, unit in {
            "M_PIN": "Msun/h", "position": "comoving Mpc/h", "velocity": "proper km/s",
            "R_sat_comoving": "comoving Mpc/h (effective model radius)",
            "satellite_radius_comoving": "comoving Mpc/h", "sigma_sat_km_s": "proper km/s",
            "absolute_magnitude_r": "^0.1 M_r - 5log10(h)", "g_r_rest_0p1": "rest frame z_ref=0.1",
        }.items():
            group[name].attrs["units"] = unit
    os.replace(temporary.name, path)
    (directory/"catalogue.json").write_text(json.dumps(dict(path=str(path), run_key=run_key, n_galaxies=total), indent=2)+"\n")
    return path
