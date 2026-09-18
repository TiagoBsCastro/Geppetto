#!/usr/bin/env python
"""Calibrate and generate a native-mass PINOCCHIO galaxy lightcone.

Run from the repository root with the supplied JSON configuration. The optional
--calibrate-only mode performs the real-input completeness assessment before
any stochastic galaxy generation.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np  # noqa: E402

from geppetto.galaxies.adapters import (  # noqa: E402
    IndependentLuminosityFunction,
    PinocchioDistances,
    file_sha256,
    load_native_halos,
)
from geppetto.galaxies.calibration import CalibrationConfig, calibrate_hod  # noqa: E402
from geppetto.galaxies.models import HODShapeParams  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="examples/galaxy_lightcone_config.json")
    parser.add_argument("--calibrate-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    params = dict(config["calibration"])
    if "shape" in params:
        params["shape"] = HODShapeParams(**params["shape"])
    calibration = CalibrationConfig(**params)
    print("[galaxies] reading and auditing native halos", flush=True)
    halos = load_native_halos(config["halo_input"])
    h = float(halos.metadata["parameters"]["Hubble100"][0])
    distances = PinocchioDistances(config["cosmology_table"], h)
    target = IndependentLuminosityFunction(config["hodpy_root"])
    chi = np.linalg.norm(halos.values["position"], axis=1)
    chi_residual = chi/distances.comoving_distance(halos.values["z_cos"])-1
    audit = dict(config=config, config_sha256=file_sha256(args.config),
                 n_halos=len(chi), particle_mass_msun_h=halos.particle_mass,
                 mass_cut_msun_h=halos.particle_mass*calibration.min_particles,
                 native_min_particles=int(np.min(halos.values["particle_count"])),
                 group_id_repeated_occurrences=len(chi)-len(np.unique(halos.values["group_id"])),
                 distance_relative_error_quantiles=np.quantile(chi_residual, [0, .01, .5, .99, 1]).tolist(),
                 target_input_hashes=target.input_hashes)
    (output/"input_audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    if np.max(np.abs(chi_residual)) > 2.e-3:
        raise ValueError("Native PLC radii disagree with the PINOCCHIO distance table by >0.2%")
    table = calibrate_hod(halos, distances, target, calibration, input_path=config["halo_input"],
                          distance_path=config["cosmology_table"], cache_dir=output/"cache")
    completeness = []
    for lo, hi in config["validation"]["redshift_slices"]:
        cells = (table.redshift_edges[:-1] >= lo-1.e-10) & (table.redshift_edges[1:] <= hi+1.e-10)
        if not np.any(cells):
            raise ValueError("Validation slice contains no full calibration cells")
        if not (np.isclose(table.redshift_edges[:-1][cells][0], lo)
                and np.isclose(table.redshift_edges[1:][cells][-1], hi)):
            raise ValueError("Validation slice boundaries must align with calibration cells")
        edge_indices = []
        for magnitude in config["validation"]["magnitude_edges"]:
            idx = int(np.argmin(np.abs(table.magnitudes-magnitude)))
            if not np.isclose(table.magnitudes[idx], magnitude):
                raise ValueError("Validation magnitude edge must align with the HOD threshold grid")
            edge_indices.append(idx)
            below = table.below_cut_prediction[cells, idx]
            kept = table.prediction[cells, idx]
            missing_fraction = float(np.max(below/(below+kept)))
            support_particles = float(np.min(table.support_min_mass[cells, idx]/halos.particle_mass))
            passed = (missing_fraction < config["validation"]["max_unresolved_fraction"]
                      and support_particles >= audit["native_min_particles"])
            completeness.append(dict(kind="cumulative", z_min=lo, z_max=hi, magnitude=magnitude,
                                     max_below_cut_fraction=missing_fraction,
                                     min_support_particles=support_particles, passed=passed))
        below = np.diff(table.below_cut_prediction[cells][:, edge_indices], axis=1)
        kept = np.diff(table.prediction[cells][:, edge_indices], axis=1)
        fractions = np.max(below/(below+kept), axis=0)
        for j, fraction in enumerate(fractions):
            completeness.append(dict(kind="differential", z_min=lo, z_max=hi,
                                     magnitude_bright=config["validation"]["magnitude_edges"][j],
                                     magnitude_faint=config["validation"]["magnitude_edges"][j+1],
                                     max_below_cut_fraction=float(fraction),
                                     passed=bool(fraction < config["validation"]["max_unresolved_fraction"])))
    report = dict(calibration_key=table.cache_key, calibration_config=asdict(calibration),
                  completeness=completeness, all_passed=all(row["passed"] for row in completeness))
    (output/"completeness.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2), flush=True)
    if not report["all_passed"]:
        raise ValueError("Preregistered luminosity bins fail the native halo-completeness assessment")
    if not args.calibrate_only:
        from geppetto.galaxies.pipeline import generate_catalogue
        from geppetto.galaxies.validation import validate_catalogue

        path = generate_catalogue(config, halos, distances, target, table)
        validate_catalogue(config, path, halos, distances, target, table)


if __name__ == "__main__":
    main()
