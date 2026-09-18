"""Small native-data, upstream-compatibility and cache regression checks."""

import importlib.util
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
pytest.importorskip("scipy")

from geppetto.galaxies.adapters import (  # noqa: E402
    PINOCCHIO_RHO_CRIT,
    NativeHaloInput,
    PinocchioDistances,
    load_native_halos,
)
from geppetto.galaxies.calibration import (  # noqa: E402
    CalibrationConfig,
    calibrate_hod,
    native_hmf_quadrature,
)
from geppetto.galaxies.models import HODShapeParams  # noqa: E402


def test_distances_use_native_units_and_ignore_zero_placeholder(tmp_path):
    a = np.array([.5, .6, .7, .8, .9, .95, 1., 1.02])
    chi = -9000*np.log10(a)-1.
    chi[a >= 1] = 0.
    path = tmp_path/"cosmology.out"
    np.savetxt(path, np.column_stack((a, np.zeros(len(a)), chi)))
    distance = PinocchioDistances(path, .7)
    z = np.array([.04, .08, .2, .5])
    expected = .7*(9000*np.log10(1+z)-1.)
    np.testing.assert_allclose(distance.comoving_distance(z), expected, rtol=1.e-12)
    np.testing.assert_allclose(distance.redshift(expected), z, atol=1.e-10)
    assert distance.comoving_distance(0.) == 0.
    volume = 2*np.pi/3*(1-np.cos(np.deg2rad(65)))*(expected[2]**3-expected[0]**3)
    np.testing.assert_allclose(distance.volume(.04, .2, 65.), volume)
    np.testing.assert_allclose(distance.volume_average(lambda z: np.ones_like(z), .04, .2), 1.)
    with pytest.raises(ValueError, match="outside"):
        distance.comoving_distance(-.1)


def test_staged_native_mass_particle_closure_and_occurrence_identity(tmp_path):
    path = tmp_path/"native.hdf5"
    mass = PINOCCHIO_RHO_CRIT*.3*(100/100)**3
    metadata = dict(mass_unit="Msun/h", parameters=dict(Hubble100=[".7"], Omega0=[".3"],
                    BoxSize=["100"], GridSize=["100"], BoxInH100=[], MinHaloMass=["10"], PLCAxis=["0", "0", "1"]))
    with h5py.File(path, "w") as handle:
        handle.attrs["metadata_json"] = json.dumps(metadata)
        data = handle.create_group("halos")
        for key, value in dict(M_PIN=[mass*10, mass*32], occurrence_id=[0, 2**48], group_id=[7, 7],
                               position=[[10., 0., 0.], [0., 20., 0.]], velocity=np.zeros((2, 3)),
                               z_cos=[.1, .2], z_obs_native=[.1, .2],
                               phi_native_deg=[0., 90.], theta_native_deg=[0., 0.]).items():
            data[key] = value
    halos = load_native_halos(path)
    np.testing.assert_allclose(halos.particle_mass, mass)
    np.testing.assert_array_equal(halos.values["particle_count"], [10, 32])
    with h5py.File(path, "a") as handle:
        handle["halos/occurrence_id"][1] = 0
    with pytest.raises(ValueError, match="Duplicate lightcone"):
        load_native_halos(path)
    with h5py.File(path, "a") as handle:
        handle["halos/occurrence_id"][1] = 1
        handle["halos/M_PIN"][1] *= 1.01
    with pytest.raises(ValueError, match="integer particle"):
        load_native_halos(path)


def test_input_staging_converts_physical_h_units_and_reads_only_one_part(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("prepare_galaxies_test", Path(__file__).parents[1]/"examples/prepare_pinocchio_galaxy_input.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    params = tmp_path/"params.txt"
    params.write_text("fixture\n")
    table = tmp_path/"cosmology.out"
    table.write_text("fixture\n")
    base = tmp_path/"plc.out"
    for part in range(2):
        Path(f"{base}.{part}").write_bytes(bytes([part]))
    calls = []

    class Catalogue(SimpleNamespace):
        def __len__(self):
            return 2

    catalogue = Catalogue(true_redshift=np.array([.1, .8]), group_ids=np.array([5, 9]),
                          masses_msun_h=np.array([1.e13, 2.e13]), positions_mpc_h=np.array([[100., 0., 0.], [200., 0., 0.]]),
                          velocities_km_s=np.array([[300., 0., 0.], [400., 0., 0.]]), observed_redshift=np.array([.101, .802]),
                          theta_deg=np.zeros(2), phi_deg=np.zeros(2), los_velocity_km_s=np.array([300., 400.]))

    def reader(path, **kwargs):
        calls.append(path)
        return catalogue

    monkeypatch.setattr(module, "read_pinocchio_parameter_file", lambda _: SimpleNamespace(parameters={}, cosmology=SimpleNamespace(h=.7)))
    monkeypatch.setattr(module, "read_pinocchio_lightcone_catalog", reader)
    output = tmp_path/"staged.hdf5"
    module.prepare(SimpleNamespace(params=str(params), cosmology_table=str(table), plc_catalog=str(base),
                                   z_min=.04, z_max=.33, output=str(output), format="binary"))
    assert calls == [Path(f"{base}.0"), Path(f"{base}.1")]
    with h5py.File(output) as handle:
        np.testing.assert_allclose(handle["halos/M_PIN"][:], [7.e12, 7.e12])
        np.testing.assert_allclose(handle["halos/position"][:, 0], [70., 70.])
        np.testing.assert_allclose(handle["halos/velocity"][:, 0], [300., 300.])
        np.testing.assert_array_equal(handle["halos/occurrence_id"][:], [0, 2**48])


def test_hmf_never_silently_drops_float32_threshold_mass():
    mass_cut = 2.93125816065136e12
    mass = np.array([float(np.float32(mass_cut)), mass_cut*2])
    with pytest.raises(ValueError, match="include every"):
        native_hmf_quadrature(mass, np.ones(2, bool), np.log10([mass_cut, mass_cut*4]), 100.)
    lower = np.nextafter(np.log10(mass.min()), -np.inf)
    _, weights = native_hmf_quadrature(mass, np.ones(2, bool), [lower, 14.], 100.)
    np.testing.assert_allclose(weights.sum(), .02)


def test_calibration_cache_changes_with_input_target_and_configuration(tmp_path, monkeypatch):
    import geppetto.galaxies.calibration as module

    mass = np.geomspace(1.e12, 1.e15, 400)
    halos = NativeHaloInput(dict(M_PIN=mass, particle_count=np.rint(mass/1.e10).astype(int),
                                z_cos=np.full(400, .105), position=np.tile([0., 0., 100.], (400, 1))),
                            dict(parameters={"MinHaloMass": ["10"]}), 1.e10, 1.e11, np.eye(3))

    class Target:
        input_hashes = {"observational_lf": "fixed"}
        parameters = {"P": 1.8}

        def cumulative(self, magnitude, redshift):
            return 1.e-4*10**(.5*(magnitude+22))+np.zeros_like(redshift)

        def reference_magnitude(self, magnitude, redshift):
            return magnitude

    target = Target()
    distances = SimpleNamespace(volume=lambda *args: 1.e6, volume_average=lambda function, *args: np.ravel(function(np.array([.105]))))
    input_path, distance_path = tmp_path/"native", tmp_path/"distance"
    input_path.write_bytes(b"native1")
    distance_path.write_bytes(b"distance1")
    shape = HODShapeParams(m1_ls=HODShapeParams().mmin_ls, m1_mt=10*HODShapeParams().mmin_mt,
                           m1_am=HODShapeParams().mmin_am, m0_a=0., m0_b=9.,
                           alpha_a=0., alpha_b=1., sigma_faint=.3, sigma_bright=.3)
    config = CalibrationConfig(z_min=.10, z_max=.11, magnitude_bright=-23., magnitude_faint=-22., magnitude_step=.5, shape=shape)

    def calibrate(settings=config):
        return calibrate_hod(halos, distances, target, settings, input_path=input_path,
                             distance_path=distance_path, cache_dir=tmp_path/"cache")

    first = calibrate()
    np.testing.assert_allclose(first.prediction, first.target_cumulative, rtol=1.e-10)
    with monkeypatch.context() as patch:
        patch.setattr(module, "_calibrate", lambda *args: pytest.fail("Cache miss with unchanged inputs"))
        cached = calibrate()
        np.testing.assert_array_equal(cached.log_shifts, first.log_shifts)
    input_path.write_bytes(b"native2")
    second = calibrate()
    assert second.cache_key != first.cache_key
    target.input_hashes = {"observational_lf": "changed"}
    third = calibrate()
    assert third.cache_key != second.cache_key
    fourth = calibrate(replace(config, mass_bin_width_dex=.02))
    assert fourth.cache_key != third.cache_key
    central, satellite = module.evaluate_occupation(jnp.array([12., 13., 14.]), jnp.asarray(first.thresholds[0]), jnp.asarray(first.log_shifts[0]))
    assert np.all(np.diff(central+satellite, axis=1) >= -1.e-10)


def test_written_galaxy_shapes_and_fixed_seed_reproducibility(tmp_path, monkeypatch):
    import geppetto.galaxies.pipeline as pipeline

    n = 64
    halos = NativeHaloInput(dict(M_PIN=np.full(n, 1.e14), particle_count=np.full(n, 1000),
                                position=np.tile([0., 0., 300.], (n, 1)), velocity=np.zeros((n, 3)),
                                z_cos=np.full(n, .1), occurrence_id=np.arange(n), group_id=np.arange(n)),
                            dict(mass_definition="native PINOCCHIO group mass"), 1.e11, 8.e10, np.eye(3))
    table = SimpleNamespace(magnitudes=np.array([-25., -22., -21.5]),
                            thresholds=np.array([[[20., 19., 21., .3, 1.], [13.5, 12., 14., .3, 1.], [13.2, 11.7, 13.7, .3, 1.]]]),
                            log_shifts=np.zeros((1, 3)), redshift_edges=np.array([.04, .33]), cache_key="fixture",
                            cell=lambda z: np.zeros(len(z), dtype=int))
    distances = SimpleNamespace(redshift=lambda chi: chi/3000.)
    kcorr = SimpleNamespace(apparent_magnitude=lambda m, z, c: m+40)
    monkeypatch.setattr(pipeline, "hodpy_modules", lambda _: {"k_correction": SimpleNamespace(GAMA_KCorrection=lambda _: kcorr)})
    monkeypatch.setattr(pipeline, "sample_colours", lambda m, z, central, rng, root:
                        (rng.normal(.9, .05, len(m)), np.ones(len(m), bool), np.ones(len(m)),
                         np.full(len(m), .9), np.full(len(m), .05)))
    config = dict(random_seed=1386, halo_chunk_size=32, survey_r_limit=20., hodpy_root="fixture",
                  calibration={"min_particles": 32}, output_dir=str(tmp_path/"first"))
    target = SimpleNamespace(parameters={}, input_hashes={})
    first = pipeline.generate_catalogue(config, halos, distances, target, table)
    assert pipeline.generate_catalogue(config, halos, distances, target, table) == first
    second = pipeline.generate_catalogue(dict(config, output_dir=str(tmp_path/"second")), halos, distances, target, table)
    with h5py.File(first) as a, h5py.File(second) as b:
        for key in a["galaxies"]:
            np.testing.assert_array_equal(a["galaxies"][key][:], b["galaxies"][key][:])
        data = a["galaxies"]
        count = len(data["galaxy_id"])
        assert count >= n and data["position"].shape == data["velocity"].shape == (count, 3)
        assert all(value.shape[0] == count for value in data.values())
        assert np.all(data["satellite_radius_comoving"][:] <= data["R_sat_comoving"][:])
        assert len(np.unique(data["host_halo_id"][:][data["is_central"][:]])) == np.sum(data["is_central"][:])
        assert "native PINOCCHIO" in json.loads(a.attrs["metadata_json"])["mass_definition"]
