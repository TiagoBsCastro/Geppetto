import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    return importlib.import_module("compare_fullsky_pair_model")


def fixture(tmp_path, module):
    ell = np.arange(2, 62)
    signal = np.array([1., 2.])[:, None]/ell
    for seed, factor in ((1386, .9), (2281, 1.1), (3253, 1.)):
        directory = tmp_path / f"seed{seed:06d}" / "geppetto_reduced"
        directory.mkdir(parents=True)
        np.savez(directory / "fullsky_observed_spectra.npz", schema_version=1, ell=ell,
                 shell_cl_theoretical_mean=factor*signal, shell_cl_measured_mean=.8*factor*signal,
                 theoretical_mean_counts_per_pixel=np.ones(2), measured_mean_counts_per_pixel=np.full(2, 1.1),
                 segment_index=[0, 1], z_lo=[.1, .2], z_hi=[.2, .3], nside=32)
        np.savez(directory / "fullsky_component_spectra.npz", ell=ell, z_lo=[.1, .2], z_hi=[.2, .3],
                 nside=32, segment_index=[0, 1], theoretical_mean_counts_per_pixel=np.ones(2),
                 shell_cl_uncollapsed=.5*factor*signal, shell_cl_cross=.1*factor*signal,
                 shell_cl_halo=.3*factor*signal)
    paths = []
    for index in range(2):
        path = tmp_path / f"prediction{index}.npz"
        components = np.stack([factor*signal[index] for factor in (.5, .2, .1, .1, .9)])[None]
        np.savez(path, ell=ell, z_lo=[.1+.1*index], z_hi=[.2+.1*index], segment_indices=[index],
                 stationary_components=components, linear_projection_correction=.1*signal[index:index+1],
                 shell_total=signal[index:index+1], metadata_json=json.dumps(dict(
                     nside=32, concentration=dict(amplitude=5.71), linear_evolution="scale_dependent_camb",
                     input_sha256={"params": "hash"}, orientation_seed=737)))
        paths.append(path)
    observation = module.load_observation(tmp_path / "seed001386/geppetto_reduced/fullsky_observed_spectra.npz")
    return paths, observation


def test_complete_fullsky_prediction_alignment_and_component_closure(tmp_path, module):
    paths, observation = fixture(tmp_path, module)
    theory, components, correction, indices = module.load_predictions(paths[::-1], observation, ell_stop=60)
    assert theory.shell_total.shape == (2, 58)
    np.testing.assert_allclose(theory.shell_total, components[:, -1]+correction)
    np.testing.assert_array_equal(theory.ell, observation.ell[indices])
    with pytest.raises(ValueError, match="incomplete"):
        module.load_predictions(paths[:1], observation)
    with pytest.raises(ValueError, match="duplicate"):
        module.load_predictions(paths+paths[:1], observation)


def test_fullsky_comparison_keeps_global_mean_and_ensemble_sem(tmp_path, module):
    fixture(tmp_path, module)
    args = argparse.Namespace(ensemble_root=tmp_path, prediction_glob=str(tmp_path / "prediction*.npz"),
                              output_dir=tmp_path / "plots", ell_min=20, ell_stop=60, bin_width=20,
                              segments=None, scheme5=None, scheme6=None)
    data = module.run(args)
    np.testing.assert_allclose(data.mean_theoretical_mean/data.theory_shell, 1.)
    np.testing.assert_allclose(data.mean_measured_mean/data.theory_shell, .8)
    np.testing.assert_allclose(data.sem_theoretical_mean/data.theory_shell, .1/np.sqrt(3))
    assert (args.output_dir / "fullsky_pair_model_all_shells.png").exists()
    assert (args.output_dir / "fullsky_pair_model_components.csv").exists()
    report = json.loads((args.output_dir / "fullsky_pair_model_audit.json").read_text())
    assert report["mask"] == "none; full sky"
    assert report["n_shells"] == 2
    assert len(report["observation_sha256"]) == 3


def test_prediction_rejects_wrong_shell_bounds_and_resolution(tmp_path, module):
    paths, observation = fixture(tmp_path, module)
    with pytest.raises(ValueError, match="bounds"):
        module.load_predictions(paths, module.replace(observation, z_lo=np.array([0., .2])))
    with pytest.raises(ValueError, match="NSIDE"):
        module.load_predictions(paths, module.replace(observation, nside=16))


@pytest.mark.parametrize("key,value,error", [
    ("nside", 16, "NSIDE"),
    ("theoretical_mean_counts_per_pixel", np.full(2, 2.), "normalization"),
    ("shell_cl_halo", np.zeros((2, 60)), "close"),
])
def test_component_cache_must_match_total_measurement(tmp_path, module, key, value, error):
    fixture(tmp_path, module)
    path = tmp_path / "seed001386/geppetto_reduced/fullsky_component_spectra.npz"
    with np.load(path, allow_pickle=False) as source:
        data = dict(source)
    data[key] = value
    np.savez(path, **data)
    args = argparse.Namespace(ensemble_root=tmp_path, prediction_glob=str(tmp_path / "prediction*.npz"),
                              output_dir=tmp_path / "plots", ell_min=20, ell_stop=60, bin_width=20,
                              segments=None, scheme5=None, scheme6=None)
    with pytest.raises(ValueError, match=error):
        module.run(args)
