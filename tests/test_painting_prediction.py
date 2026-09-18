import argparse
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from geppetto.cosmology import Cosmology
from geppetto.lpt_backbone import LPTGrowthRatios


@pytest.fixture(autouse=True)
def double_precision():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


@pytest.fixture
def example(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    return importlib.import_module("predict_painting_matched_power")


def test_linear_reference_reads_no_measurements_and_respects_explicit_order(tmp_path, example):
    path = tmp_path / "linear.npz"
    np.savez(path, ell=np.arange(2, 5, dtype=">i8"), segment_indices=[8, 3],
             shell_linear=np.array([[1., 2., 3.], [4., 5., 6.]], dtype=">f8"),
             observed_shell=np.array([object()], dtype=object), reconstructed_sigma8=.8,
             linear_power_evolution="scalar_growth")
    ell, linear = example.load_linear_reference(path, [], [3, 8], sigma8=.8, evolution="scalar_growth")
    np.testing.assert_array_equal(ell, [2, 3, 4])
    np.testing.assert_array_equal(linear, [[4, 5, 6], [1, 2, 3]])
    assert linear.dtype.isnative
    with pytest.raises(ValueError, match="sigma8"):
        example.load_linear_reference(path, [], [3], sigma8=.9)
    with pytest.raises(ValueError, match="growth"):
        example.load_linear_reference(path, [], [3], evolution="scale_dependent_camb")


@pytest.mark.parametrize("invalid", ["ell", "shape", "duplicate", "missing", "negative"])
def test_linear_reference_rejects_invalid_grids(tmp_path, example, invalid):
    values = dict(ell=np.array([2, 3]), shell_linear=np.ones((2, 2)), segment_indices=np.array([3, 8]))
    if invalid == "ell":
        values["ell"] = [3, 2]
    elif invalid == "shape":
        values["shell_linear"] = np.ones((1, 2))
    elif invalid == "duplicate":
        values["segment_indices"] = [3, 3]
    elif invalid == "missing":
        values["segment_indices"] = [2, 8]
    else:
        values["shell_linear"][0, 0] = -1
    path = tmp_path / "linear.npz"
    np.savez(path, **values)
    with pytest.raises(ValueError):
        example.load_linear_reference(path, [], [3])


def test_legacy_reference_requires_all_manifest_rows_in_sorted_order(tmp_path, example):
    path = tmp_path / "linear.npz"
    np.savez(path, ell=[2, 3], shell_linear=[[1., 2.], [3., 4.]])
    rows = [dict(segment_index=2), dict(segment_index=6)]
    _, values = example.load_linear_reference(path, rows, [6, 2])
    np.testing.assert_array_equal(values, [[3, 4], [1, 2]])
    with pytest.raises(ValueError, match="indexing"):
        example.load_linear_reference(path, rows[:1], [2])


def test_manifest_resolution_and_selection_guards(example):
    row = dict(theta_map_rad=np.sqrt(hp.nside2pixarea(16)), bounds_mode="z", redshift_mode="true",
               nfw_mass_conversion="none_catalog_mass_interpreted_as_profile_mass")
    assert example.manifest_nside([row], None) == 16
    example.validate_painting_selection([row])
    with pytest.raises(ValueError, match="NSIDE"):
        example.manifest_nside([row], 32)
    for key, invalid in (("bounds_mode", "chi"), ("redshift_mode", "observed"), ("nfw_mass_conversion", "unknown")):
        with pytest.raises(ValueError, match=key):
            example.validate_painting_selection([dict(row, **{key: invalid})])


def _fixture(tmp_path, example, monkeypatch):
    paths = {key: tmp_path / name for key, name in dict(
        params="params.txt", manifest="manifest.csv", cosmology_table="cosmology.out",
        linear_reference="linear.npz", hmf="native.mf.out",
    ).items()}
    for path in paths.values():
        path.write_text("synthetic input\n")
    np.savez(paths["linear_reference"], ell=np.arange(2, 6), shell_linear=np.full((1, 4), .001),
             reconstructed_sigma8=.8, linear_power_evolution="scalar_growth")
    row = dict(segment_index="4", theta_map_rad=str(np.sqrt(hp.nside2pixarea(16))),
               z_lo=".1", z_hi=".2", chi_lo_mpc_h="9999", chi_hi_mpc_h="99999",
               bounds_mode="z", redshift_mode="true", particle_mass_msun_h=".1",
               nfw_mass_conversion="none_catalog_mass_interpreted_as_profile_mass",
               nfw_concentration_amplitude="5.71", nfw_concentration_mass_slope="-.084",
               nfw_concentration_redshift_slope="-.47", nfw_concentration_mass_pivot_msun_h="2.e12",
               nfw_overdensity_mode="constant", nfw_overdensity="200", nfw_reference_density="critical",
               theta_resolution_rad=".001", n_resolution="4")
    cosmology = Cosmology()
    metadata = SimpleNamespace(cosmology=cosmology, particle_mass_msun_h=.1, box_size_mpc_h=10., grid_size=1)
    wave = jnp.array([1.e-5, 100.])
    linear = SimpleNamespace(h=cosmology.h, omega_m0=cosmology.omega_m, k_h_mpc=wave)
    counts = SimpleNamespace(log_mass_msun_h=jnp.log(jnp.array([2., 8.])), scale_factor=np.array([.1, 1.]))
    monkeypatch.setattr(example, "read_pinocchio_parameter_file", lambda *a: metadata)
    monkeypatch.setattr(example, "read_pinocchio_cosmology_table", lambda *a: linear)
    monkeypatch.setattr(example, "read_pinocchio_linear_power_evolution", lambda *a: None)
    monkeypatch.setattr(example, "read_pinocchio_mass_function", lambda *a: None)
    monkeypatch.setattr(example, "pinocchio_mass_function_series_from_tables", lambda *a: counts)
    monkeypatch.setattr(example, "read_pinocchio_halo_count_quadrature", lambda *a, **k: counts)
    monkeypatch.setattr(example, "load_manifest", lambda *a: [row])
    monkeypatch.setattr(example, "fit_pinocchio_numerical_halo_bias", lambda *a: (None, []))
    monkeypatch.setattr(example, "sigma8_from_linear_power", lambda *a: .8)
    monkeypatch.setattr(example, "rho_mean_comoving", lambda *a: 1.)
    monkeypatch.setattr(example, "comoving_distance_mpc_h", lambda z, *a: 100+100*z)
    monkeypatch.setattr(example, "redshift_at_comoving_distance", lambda chi, *a: (chi-100)/100)
    monkeypatch.setattr(example, "halo_count_weights", lambda *a: jnp.array([.02, .005]))
    monkeypatch.setattr(example, "halo_bias_at_redshift", lambda *a: jnp.array([1.5, 3.]))
    monkeypatch.setattr(example, "linear_matter_power", lambda wave, *a: jnp.full_like(wave, 8.))
    monkeypatch.setattr(example, "read_pinocchio_lpt_growth_ratios", lambda *a: LPTGrowthRatios(1.01, 1.02))
    monkeypatch.setattr(example.hp, "pixwin", lambda nside, lmax: np.ones(lmax+1))
    seen = []
    def backbone(wave, power, mass, number, **kwargs):
        seen.append(kwargs)
        return SimpleNamespace(k_h_mpc=wave, coherent_power=np.full(2, 10.), linear_power=power,
                               same_protohalo_power=np.full(2, .2), lattice_correction=np.zeros(2),
                               halo_self_power=.4, same_protohalo_zero_mode=.4, resolved_mass_fraction=.08)
    monkeypatch.setattr(example, "build_lpt_particle_backbone", backbone)
    monkeypatch.setattr(example, "build_population_geometry", lambda mass, *a, **k: jnp.asarray(mass))
    def moments(parameters, mass, *, lmax):
        t = jnp.arange(lmax+1)/(lmax+1)
        alpha = parameters.amplitude+.01*parameters.mass_slope*mass+.1*parameters.redshift_slope
        response = 1-.02*alpha[:, None]*t[None, :]
        return response, response**2+.05*t[None, :]
    monkeypatch.setattr(example, "population_assignment_moments", moments)
    args = argparse.Namespace(**{key: value for key, value in paths.items() if key != "hmf"},
                              hmf_glob=str(paths["hmf"]), segments=[4], output=tmp_path / "result.npz",
                              nside=None, radial_order=2, orientations=2, seed=42, mass_batch_size=1,
                              angle_nodes=64, threads=1, los_periods=4, los_order=16, derivatives=True)
    return args, row, seen


def test_raw_input_orchestration_conserves_components_and_propagates_jacobians(tmp_path, example, monkeypatch):
    args, _, seen = _fixture(tmp_path, example, monkeypatch)
    assert example.run(args) == args.output
    assert len(seen) == args.radial_order
    assert seen[0]["growth"] == LPTGrowthRatios(1.01, 1.02)
    with np.load(args.output, allow_pickle=False) as data:
        components = data["stationary_components"]
        assert components.shape == (1, 5, 4)
        np.testing.assert_allclose(components[:, :4].sum(axis=1), components[:, -1], atol=1.e-18)
        np.testing.assert_allclose(data["shell_total"], components[:, -1]+data["linear_projection_correction"])
        jacobian = data["component_jacobian"]
        assert jacobian.shape == (1, 3, 5, 4)
        assert np.isfinite(jacobian).all()
        assert (np.max(np.abs(jacobian), axis=(0, 2, 3)) > 0).all()
        np.testing.assert_allclose(jacobian[:, :, :4].sum(axis=2), jacobian[:, :, -1], atol=1.e-18)
        report = json.loads(str(data["metadata_json"]))
        assert len(report["input_sha256"]) == 5
        assert set(report["linear_array_sha256"]) == {"k", "power"}
        assert all(110 < node["chi_mpc_h"] < 120 for node in report["nodes"])


@pytest.mark.parametrize("invalid", ["particle_mass", "overwrite", "duplicate_segment", "selection"])
def test_raw_input_workflow_rejects_inconsistent_inputs_before_backbone(tmp_path, example, monkeypatch, invalid):
    args, row, seen = _fixture(tmp_path, example, monkeypatch)
    if invalid == "particle_mass":
        row["particle_mass_msun_h"] = "1."
    elif invalid == "overwrite":
        args.output = args.linear_reference
    elif invalid == "duplicate_segment":
        args.segments = [4, 4]
    else:
        row["bounds_mode"] = "chi"
    with pytest.raises(ValueError):
        example.run(args)
    assert not seen
