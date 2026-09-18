#!/usr/bin/env python
"""Independent radial/velocity checks and optional real-run replay comparison."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import kstest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalogue", required=True)
    parser.add_argument("--replicate", help="Separate same-seed run; compare every galaxy column exactly")
    parser.add_argument("--output", default="outputs/galaxy_lightcone/validation/sampling_audit.json")
    args = parser.parse_args()
    with h5py.File(args.catalogue) as handle:
        metadata = json.loads(handle.attrs["metadata_json"])
        data = {key: value[:] for key, value in handle["galaxies"].items()}
    satellite = ~data["is_central"]
    hosts = data["host_index"][satellite]
    with h5py.File(metadata["config"]["halo_input"]) as handle:
        native_velocity = handle["halos/velocity"][:][hosts]
        native_position = handle["halos/position"][:][hosts]
    c = data["c_sat"][satellite]
    fraction = data["satellite_radius_comoving"][satellite]/data["R_sat_comoving"][satellite]
    x = c*fraction
    uniform = (np.log1p(x)-x/(1+x))/(np.log1p(c)-c/(1+c))
    delta = data["position"][satellite]-native_position
    polar_uniform = .5*(1+delta[:, 2]/np.linalg.norm(delta, axis=1))
    velocity = (data["velocity"][satellite]-native_velocity)/data["sigma_sat_km_s"][satellite, None]
    tests = {"radial_cdf_uniform": kstest(uniform, "uniform"),
             "isotropic_polar_angle": kstest(polar_uniform, "uniform")}
    tests.update({f"standardized_velocity_{axis}": kstest(velocity[:, i], "norm") for i, axis in enumerate("xyz")})
    cutoff = .01/len(tests)
    results = {key: dict(statistic=float(value.statistic), p_value=float(value.pvalue),
                         p_value_cutoff=cutoff, passed=bool(value.pvalue >= cutoff)) for key, value in tests.items()}
    replay = None
    if args.replicate:
        with h5py.File(args.replicate) as other:
            other_metadata = json.loads(other.attrs["metadata_json"])
            if metadata["config"] != other_metadata["config"]:
                raise ValueError("Replay comparison requires the same configuration and seed")
            assert set(data) == set(other["galaxies"])
            for name, values in data.items():
                np.testing.assert_array_equal(values, other["galaxies"][name][:], err_msg=f"Replay mismatch: {name}")
        replay = dict(passed=True, replicate=args.replicate, n_columns=len(data), n_galaxies=len(data["galaxy_id"]),
                      comparison="bit-for-bit equality of all galaxy columns; metadata/cache provenance may differ")
    central_groups = data["host_group_id"][data["is_central"]]
    report = dict(catalogue=args.catalogue, n_satellites=int(np.sum(satellite)), tests=results, replay=replay,
                  extra_centrals_across_repeated_native_group_ids=len(central_groups)-len(np.unique(central_groups)),
                  identity_note="One central per unique native PLC occurrence, not per persistent group across different native crossings",
                  passed=all(result["passed"] for result in results.values()))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise ValueError("Independent galaxy sampling audit failed")


if __name__ == "__main__":
    main()
