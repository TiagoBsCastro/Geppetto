import csv
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _load_example(name):
    path = Path(__file__).parents[1] / "examples" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _observation(module, scale, measured_scale):
    ell = np.arange(2, 42)
    shell = np.vstack((1.0 / ell, 2.0 / ell))
    return module.FullskyObservation(
        ell=ell,
        shell_cl_theoretical_mean=scale * shell,
        shell_cl_measured_mean=measured_scale * shell,
        theoretical_mean_counts_per_pixel=np.asarray([1.0, 2.0]),
        measured_mean_counts_per_pixel=np.asarray([1.01, 1.98]),
        segment_index=np.asarray([0, 1]),
        z_lo=np.asarray([0.0, 0.1]),
        z_hi=np.asarray([0.1, 0.2]),
        nside=8,
    )


@pytest.mark.parametrize("uncompensated", [False, True])
def test_schema6_fullsky_loader_uses_explicit_one_halo_choice(tmp_path, uncompensated):
    module = _load_example("aggregate_angular_power_ensemble")
    observation = _observation(module, 1., 1.)
    ones = np.ones((2, observation.ell.size))
    np.savez_compressed(
        tmp_path / "angular_power_theory.npz", validation_schema_version=6,
        two_halo_model="standard_normalized_hmf_castro_bias",
        linear_power_evolution="scale_dependent_camb", ell=observation.ell,
        shell_two_halo=2*ones, shell_one_halo_standard=3*ones,
        shell_one_halo_compensated=ones, shell_particle_shot_noise=.1*ones,
    )
    with (tmp_path / "angular_power_diagnostics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["z_lo", "z_hi"])
        writer.writeheader()
        writer.writerows([dict(z_lo=0., z_hi=.1), dict(z_lo=.1, z_hi=.2)])
    model = module.THEORY_MODEL_STANDARD_WHITE if uncompensated else module.THEORY_MODEL_STANDARD
    result = module.load_aligned_theory(tmp_path, observation, theory_model=model)
    np.testing.assert_allclose(result.shell_total, 5.1 if uncompensated else 3.1)


def test_theoretical_shell_mean_counts(monkeypatch):
    module = _load_example("measure_fullsky_angular_power")
    monkeypatch.setattr(
        module,
        "read_pinocchio_parameter_file",
        lambda _: SimpleNamespace(
            grid_size=10,
            box_size_mpc_h=5.0,
            cosmology=SimpleNamespace(h=1.0),
        ),
    )
    monkeypatch.setattr(
        module,
        "read_pinocchio_cosmology_table",
        lambda _: SimpleNamespace(
            h=1.0,
            scale_factor=np.asarray([2.0 / 3.0, 1.0]),
            chi_mpc_h=np.asarray([100.0, 0.0]),
        ),
    )

    result = module.theoretical_shell_mean_counts(
        np.asarray([0.0]),
        np.asarray([0.5]),
        8,
        params=Path("params"),
        cosmology_table=Path("cosmology"),
    )

    pixel_area = 4.0 * np.pi / (12 * 8**2)
    np.testing.assert_allclose(result, [8.0 * pixel_area * 100.0**3 / 3.0])


def test_fullsky_mean_normalization_rescaling(monkeypatch):
    module = _load_example("measure_fullsky_angular_power")
    counts = np.asarray([1.0, 2.0, 3.0])
    calls = []

    class FakeHealpy:
        @staticmethod
        def anafast(values, lmax, iter):
            calls.append((values.copy(), lmax, iter))
            return np.arange(lmax + 1, dtype=np.float64)

    monkeypatch.setitem(sys.modules, "healpy", FakeHealpy)
    result = module._fullsky_cl(counts, 4.0, lmax=4, iterations=3)

    np.testing.assert_array_equal(result, np.arange(5))
    np.testing.assert_allclose(calls[0][0], [-0.25, 0.0, 0.25])


def test_fullsky_component_spectra_close_to_total(monkeypatch):
    module = _load_example("measure_fullsky_component_power")
    calls = []

    class FakeHealpy:
        @staticmethod
        def map2alm(values, lmax, iter):
            calls.append((values.copy(), lmax, iter))
            return np.asarray(values)

        @staticmethod
        def alm2cl(first, second=None):
            other = first if second is None else second
            value = float(np.dot(first, other))
            return np.full(5, value)

    monkeypatch.setitem(sys.modules, "healpy", FakeHealpy)
    uu, hh, uh, total = module.component_cls(
        np.asarray([1.0, 2.0, 3.0]),
        np.asarray([2.0, 4.0, 3.0]),
        2.0,
        lmax=4,
        iterations=3,
    )

    np.testing.assert_allclose(total, uu + hh + 2.0 * uh)
    np.testing.assert_allclose(calls[0][0] + calls[1][0], [-1.0, 0.5, 0.5])
    assert calls[0][1:] == (4, 3)


def test_build_fullsky_ensemble():
    module = _load_example("aggregate_angular_power_ensemble")
    observations = [
        _observation(module, 0.9, 0.88),
        _observation(module, 1.1, 1.06),
    ]
    ell = observations[0].ell
    theory = module.FullskyTheory(
        ell=ell,
        shell_total=np.vstack((1.0 / ell, 2.0 / ell)),
    )

    ensemble = module.build_ensemble(
        observations,
        theory,
        [1, 2],
        ell_min=2,
        bin_width=20,
    )

    np.testing.assert_allclose(ensemble.mean_theoretical_mean / ensemble.theory_shell, 1.0)
    np.testing.assert_allclose(ensemble.mean_measured_mean / ensemble.theory_shell, 0.97)
    assert ensemble.realization_theoretical_mean.shape == (2, 2, 2)


def _write_theory_archive(path, observation, *, schema_version=5, power_evolution="scale_dependent_camb"):
    shell_linear = np.vstack((1.0 / observation.ell, 2.0 / observation.ell))
    np.savez(
        path / "angular_power_theory.npz",
        validation_schema_version=np.asarray(schema_version),
        linear_power_evolution=np.asarray(power_evolution),
        ell=observation.ell,
        shell_linear=shell_linear,
        shell_two_halo=1.1 * shell_linear,
        shell_one_halo=0.2 * shell_linear,
        shell_particle_shot_noise=0.1 * shell_linear,
    )
    with (path / "angular_power_diagnostics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=("segment_index", "z_lo", "z_hi"))
        writer.writeheader()
        for segment, (z_lo, z_hi) in enumerate(
            zip(observation.z_lo, observation.z_hi, strict=True)
        ):
            writer.writerow({"segment_index": segment, "z_lo": z_lo, "z_hi": z_hi})


def test_load_aligned_theory_selects_explicit_schema5_model(tmp_path):
    module = _load_example("aggregate_angular_power_ensemble")
    observation = _observation(module, 1.0, 1.0)
    _write_theory_archive(tmp_path, observation)

    baseline = module.load_aligned_theory(
        tmp_path,
        observation,
        theory_model=module.THEORY_MODEL_LINEAR_BASELINE,
        require_scale_dependent_camb=True,
    )
    corrected = module.load_aligned_theory(
        tmp_path,
        observation,
        theory_model=module.THEORY_MODEL_CORRECTED_TWO_HALO,
        require_scale_dependent_camb=True,
    )

    shell_linear = np.vstack((1.0 / observation.ell, 2.0 / observation.ell))
    np.testing.assert_allclose(baseline.shell_total, 1.3 * shell_linear)
    np.testing.assert_allclose(corrected.shell_total, 1.4 * shell_linear)
    assert baseline.power_evolution == "scale_dependent_camb"
    assert corrected.model == module.THEORY_MODEL_CORRECTED_TWO_HALO


def test_load_aligned_theory_rejects_obsolete_or_scalar_growth(tmp_path):
    module = _load_example("aggregate_angular_power_ensemble")
    observation = _observation(module, 1.0, 1.0)
    _write_theory_archive(tmp_path, observation, schema_version=3)
    with pytest.raises(ValueError, match="obsolete validation schema 3"):
        module.load_aligned_theory(tmp_path, observation)

    _write_theory_archive(tmp_path, observation, power_evolution="scalar_growth")
    with pytest.raises(ValueError, match="scale-dependent CAMB"):
        module.load_aligned_theory(
            tmp_path,
            observation,
            require_scale_dependent_camb=True,
        )


def test_render_fullsky_ensemble(tmp_path):
    pytest.importorskip("matplotlib")
    module = _load_example("aggregate_angular_power_ensemble")
    observations = [
        _observation(module, 0.9, 0.88),
        _observation(module, 1.1, 1.06),
    ]
    ell = observations[0].ell
    theory = module.FullskyTheory(
        ell=ell,
        shell_total=np.vstack((1.0 / ell, 2.0 / ell)),
    )
    ensemble = module.build_ensemble(
        observations,
        theory,
        [1, 2],
        ell_min=2,
        bin_width=20,
    )

    outputs = module.render_all_shells(ensemble, tmp_path, png_dpi=72)
    outputs.extend(module.render_shell_means(ensemble, tmp_path, png_dpi=72))

    assert len(outputs) == 4
    assert all(path.stat().st_size > 0 for path in outputs)


def test_component_decomposition_subtracts_particle_shot_and_closes():
    module = _load_example("audit_fullsky_component_decomposition")
    uncollapsed = np.asarray([5.0, 10.0])
    halo = np.asarray([2.0, 4.0])
    cross = np.asarray([2.0, 4.0])
    shot = np.asarray([1.0, 2.0])

    uu_clustering, parallel, residual, coherent, incoherent, correlation = (
        module._decompose_component_power(uncollapsed, halo, cross, shot)
    )

    np.testing.assert_allclose(uu_clustering, [4.0, 8.0])
    np.testing.assert_allclose(parallel, [1.0, 2.0])
    np.testing.assert_allclose(residual, [1.0, 2.0])
    np.testing.assert_allclose(coherent, [9.0, 18.0])
    np.testing.assert_allclose(incoherent, [2.0, 4.0])
    np.testing.assert_allclose(coherent + incoherent, uncollapsed + halo + 2.0 * cross)
    np.testing.assert_allclose(correlation, np.sqrt(0.5))


def test_ensemble_plot_does_not_clip_realization_ratios(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    module = _load_example("aggregate_angular_power_ensemble")
    observations = [_observation(module, 0.6, 0.5), _observation(module, 1.4, 1.5)]
    ell = observations[0].ell
    theory = module.FullskyTheory(ell=ell, shell_total=np.vstack((1.0 / ell, 2.0 / ell)))
    ensemble = module.build_ensemble(observations, theory, [1, 2], ell_min=2, bin_width=20)
    limits = []

    def capture(figure, base, dpi):
        limits.append(figure.axes[0].get_ylim())
        return []

    monkeypatch.setattr(module, "_save_figure", capture)
    module.render_all_shells(ensemble, tmp_path, png_dpi=72)
    assert limits[0][0] < 0.6
    assert limits[0][1] > 1.4


def test_component_decomposition_masks_shot_dominated_bins():
    module = _load_example("audit_fullsky_component_decomposition")

    result = module._decompose_component_power(
        np.asarray([1.0]),
        np.asarray([2.0]),
        np.asarray([0.5]),
        np.asarray([1.0]),
    )

    assert np.isnan(result[1:]).all()
