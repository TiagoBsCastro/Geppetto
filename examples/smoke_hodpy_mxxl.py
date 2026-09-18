#!/usr/bin/env python
"""Exercise hodpy's untouched supplied MXXL input, placement, colour and writer.

This is an API smoke test, not an MXXL abundance/clustering validation. A small
fixed occupation fixture avoids upstream MXXL table generation. If legacy
nbodykit is unavailable, only the cosmology dependency is bridged to Astropy
in this isolated process; no upstream source file is modified.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import types
from pathlib import Path

import h5py
import numpy as np

from geppetto.galaxies.adapters import file_sha256, hodpy_modules


def astropy_cosmology_bridge():
    """Supply the original MXXL cosmology methods without nbodykit/CLASS."""
    from astropy.cosmology import FlatLambdaCDM
    from scipy.interpolate import CubicSpline

    class CosmologyMXXL:
        def __init__(self):
            self.background = FlatLambdaCDM(H0=73., Om0=.25, Ob0=.045, Tcmb0=0.)
            z = np.linspace(0., 1., 10001)
            chi = self.background.comoving_distance(z).value*.73
            self._chi = CubicSpline(z, chi)
            self._z = CubicSpline(chi, z)

        def mean_density(self, z):
            return 2.77536627e11*.25*(1+z)**3

        def comoving_distance(self, z):
            return self._chi(z)

        def redshift(self, chi):
            return self._z(chi)

    module = types.ModuleType("hodpy.cosmology")
    module.CosmologyMXXL = CosmologyMXXL
    sys.modules["hodpy.cosmology"] = module


class SmokeOccupation:
    """One central and one satellite per halo, for testing the upstream API."""

    def get_magnitude_centrals(self, log_mass, redshift):
        return np.full(len(log_mass), -22.5)

    def get_number_satellites(self, log_mass, redshift):
        return np.ones(len(log_mass), dtype=int)

    def get_magnitude_satellites(self, log_mass, number, redshift):
        hosts = np.repeat(np.arange(len(log_mass)), number)
        return hosts, np.full(len(hosts), -22.)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hodpy-root", default="outputs/galaxy_lightcone/vendor/hodpy")
    parser.add_argument("--output", default="outputs/galaxy_lightcone/validation/mxxl_smoke.json")
    args = parser.parse_args()
    modules = hodpy_modules(args.hodpy_root)
    backend = "upstream nbodykit"
    if importlib.util.find_spec("nbodykit") is None:
        astropy_cosmology_bridge()
        backend = "Astropy bridge for absent legacy nbodykit; original reader/placement/colour/writer unchanged"
    from hodpy.galaxy_catalogue import BGSGalaxyCatalogue
    from hodpy.halo_catalogue import MXXLCatalogue

    path = Path(args.hodpy_root)/"input/halo_catalogue_small.hdf5"
    halos = MXXLCatalogue(str(path))
    total = halos.size
    with h5py.File(path) as handle:
        np.testing.assert_array_equal(halos.get("mass"), handle["Data/M200m"][:]*1.e10)
    selection = np.flatnonzero((halos.get("zcos") > .05) & (halos.get("zcos") < .32)
                              & (halos.get("mass") > 1.e12) & (halos.get("mass") < 1.e15)
                              & (halos.get("rvmax") > 0))[:64]
    mask = np.zeros(total, dtype=bool)
    mask[selection] = True
    halos.cut(mask)
    assert halos.size == 64
    np.testing.assert_allclose(halos.get_r200()/halos.get_r200(comoving=False), 1+halos.get("zcos"), rtol=1.e-6)
    assert np.all(np.isfinite(halos.get("conc")))
    np.random.seed(20260917)
    galaxies = BGSGalaxyCatalogue(halos)
    galaxies.add_galaxies(SmokeOccupation())
    galaxies.position_galaxies()
    galaxies.add_colours(modules["colour"].Colour())
    galaxies.add_apparent_magnitude(modules["k_correction"].GAMA_KCorrection(halos.cosmology))
    assert galaxies.size == 128 and np.count_nonzero(galaxies.get("is_cen")) == 64
    assert all(np.all(np.isfinite(value)) for value in galaxies._quantities.values())
    with tempfile.TemporaryDirectory() as temporary:
        saved = Path(temporary)/"mxxl_smoke.hdf5"
        galaxies.save_to_file(str(saved), "hdf5", halo_properties=["mass"])
        with h5py.File(saved) as handle:
            assert len(handle["abs_mag"]) == 128
            np.testing.assert_array_equal(handle["halo_mass"][:], galaxies.get_halo("mass"))
    report = dict(passed=True, source=str(path), source_sha256=file_sha256(path),
                  n_example_halos=total, n_test_halos=64, n_test_galaxies=128,
                  cosmology_backend=backend, occupation="fixed API fixture, not scientific MXXL HOD calibration",
                  checks=["original MXXL mass conversion", "physical/comoving R200m", "original satellite placement",
                          "original colours", "original apparent magnitudes", "original HDF5 writer"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
