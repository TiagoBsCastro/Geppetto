import csv
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _load_example_module():
    path = (
        Path(__file__).parents[1]
        / "examples"
        / "validate_pinocchio_angular_power.py"
    )
    spec = importlib.util.spec_from_file_location("validate_pinocchio_angular_power", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _cosmology_row(a, chi, omega_m, growth, k, power):
    values = np.ones(20, dtype=np.float64)
    values[0] = a
    values[2] = chi
    values[3] = chi * a
    values[4] = omega_m
    values[5] = -1.0
    values[6] = growth
    values[18] = k
    values[19] = power
    return " ".join(str(value) for value in values)


def _write_hmf(path, redshift):
    path.write_text(
        "\n".join(
            [
                f"# Mass function for redshift {redshift:.6f}",
                "1e12 1e-15 1e-15 1e-15 100 1e-15 1",
                "1e13 2e-17 2e-17 2e-17 20 2e-17 1",
                "1e14 1e-19 1e-19 1e-19 1 1e-19 1",
            ]
        ),
        encoding="utf-8",
    )


def test_sigma8_reference_uses_parameter_or_consistent_headers():
    module = _load_example_module()
    headers = [{"COS_S8": 0.81} for _ in range(2)]

    assert module.sigma8_reference(
        0.81, headers, reconstructed_sigma8=0.81
    ) == (0.81, "parameter_file")
    assert module.sigma8_reference(
        0.0, headers, reconstructed_sigma8=0.81
    ) == (0.81, "mass_map_COS_S8")


def test_sigma8_reference_falls_back_to_cosmology_power_spectrum():
    module = _load_example_module()

    assert module.sigma8_reference(
        0.0,
        [{}, {}],
        reconstructed_sigma8=0.805,
    ) == (0.805, "cosmology_power_spectrum")


def test_sigma8_reference_rejects_partial_or_inconsistent_headers():
    module = _load_example_module()
    with pytest.raises(ValueError, match="either every mass-map header or none"):
        module.sigma8_reference(
            0.0,
            [{"COS_S8": 0.8}, {}],
            reconstructed_sigma8=0.8,
        )
    with pytest.raises(ValueError, match="disagree across shells"):
        module.sigma8_reference(
            0.0,
            [{"COS_S8": 0.8}, {"COS_S8": 0.81}],
            reconstructed_sigma8=0.8,
        )


def test_exact_projection_checkpoint_computes_only_missing_multipoles(tmp_path):
    module = _load_example_module()
    checkpoint = tmp_path / "exact_checkpoint.npz"
    computed: list[np.ndarray] = []

    def compute(ell):
        ell_values = np.asarray(ell, dtype=np.int64)
        computed.append(ell_values.copy())
        shell = np.stack((ell_values, 2 * ell_values)).astype(np.float64)
        return shell, 3.0 * ell_values

    first = module.exact_batch_with_checkpoint(
        checkpoint,
        "fingerprint",
        np.asarray([2, 3]),
        2,
        compute,
    )
    second = module.exact_batch_with_checkpoint(
        checkpoint,
        "fingerprint",
        np.asarray([3, 4]),
        2,
        compute,
    )
    third = module.exact_batch_with_checkpoint(
        checkpoint,
        "fingerprint",
        np.asarray([4, 2]),
        2,
        compute,
    )

    assert checkpoint.exists()
    assert [values.tolist() for values in computed] == [[2, 3], [4]]
    np.testing.assert_allclose(first[0], [[2, 3], [4, 6]])
    np.testing.assert_allclose(second[0], [[3, 4], [6, 8]])
    np.testing.assert_allclose(third[0], [[4, 2], [8, 4]])
    assert first[2] is False
    assert second[2] is False
    assert third[2] is True


def test_theory_component_coupling_includes_deprojection_and_fsky():
    module = _load_example_module()

    class Workspace:
        @staticmethod
        def couple_cell(component):
            return 0.5 * component

    fake_module = SimpleNamespace(
        deprojection_bias=lambda *args, **kwargs: 0.1 * args[2],
    )
    coupling = module.MaskCoupling(
        module=fake_module,
        mask=np.ones(12),
        template=np.ones((1, 1, 12)),
        reference_field=object(),
        workspace=Workspace(),
        ell_full=np.arange(6),
        f_sky=0.5,
        nside=1,
        n_iter=0,
    )

    result = module.couple_theory_component(
        np.asarray([[2.0, 4.0], [1.0, 3.0]]),
        np.asarray([2, 3]),
        coupling,
    )

    np.testing.assert_allclose(result, [[2.4, 4.8], [1.2, 3.6]])


def test_memory_reduced_namaster_estimator_matches_standard_field():
    nmt = pytest.importorskip("pymaster")
    module = _load_example_module()
    nside = 4
    npix = 12 * nside**2
    lmax = 7
    pixels = np.arange(npix // 2, dtype=np.int64)
    counts = 2.0 + np.sin(0.1 * pixels)
    coupling = module.build_mask_coupling(
        pixels,
        nside,
        lmax,
        bin_width=2,
        n_iter=0,
    )

    reduced = module.estimate_pseudo_cls(
        counts,
        pixels,
        coupling,
        full_sky_buffer=np.zeros(npix, dtype=np.float64),
    )
    standard_map = np.zeros(npix, dtype=np.float64)
    standard_map[pixels] = counts / np.mean(counts)
    standard_field = nmt.NmtField(
        coupling.mask,
        standard_map[None, :],
        spin=0,
        templates=np.ones((1, 1, npix), dtype=np.float64),
        n_iter=0,
        n_iter_mask=0,
        lmax=lmax,
        lmax_mask=2 * lmax,
    )
    standard = nmt.compute_coupled_cell(standard_field, standard_field)[0] / coupling.f_sky

    np.testing.assert_allclose(reduced, standard, rtol=1.0e-12, atol=1.0e-14)


def test_angular_power_validation_end_to_end(tmp_path, monkeypatch):
    hp = pytest.importorskip("healpy")
    pytest.importorskip("pymaster")
    fits = pytest.importorskip("astropy.io.fits")
    monkeypatch.setattr(
        hp,
        "pixwin",
        lambda nside, lmax: np.ones(lmax + 1, dtype=">f8"),
    )
    module = _load_example_module()
    nside = 4
    pixels = np.arange(hp.nside2npix(nside), dtype=np.int64)
    manifest_rows = []
    for index, (z_lo, z_hi) in enumerate(((0.0, 0.1), (0.1, 0.2))):
        mass_map_path = tmp_path / f"pinocchio.test.massmap.seg{index:03d}.fits"
        temperature = 20.0 + 0.5 * np.sin(0.1 * pixels + index)
        columns = [
            fits.Column(name="PIXEL", format="K", array=pixels),
            fits.Column(name="TEMPERATURE", format="D", array=temperature),
        ]
        hdu = fits.BinTableHDU.from_columns(columns, name="HEALPIX")
        hdu.header["NSIDE"] = nside
        hdu.header["ORDERING"] = "RING"
        hdu.header["INDXSCHM"] = "EXPLICIT"
        fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(mass_map_path)
        output_npz = tmp_path / f"painted_nfw_seg{index:03d}.npz"
        np.savez_compressed(
            output_npz,
            nfw_particle_counts=1.0 + 0.1 * np.cos(0.2 * pixels + index),
        )
        manifest_rows.append(
            {
                "segment_index": index,
                "mass_map_path": mass_map_path,
                "output_npz": output_npz,
                "z_lo": z_lo,
                "z_hi": z_hi,
                "nfw_concentration_amplitude": 5.71,
                "nfw_concentration_mass_slope": -0.084,
                "nfw_concentration_redshift_slope": -0.47,
                "nfw_concentration_mass_pivot_msun_h": 2.0e12,
                "nfw_overdensity_mode": "constant",
                "nfw_overdensity": 200.0,
                "nfw_reference_density": "critical",
                "theta_resolution_rad": 0.01,
            }
        )

    manifest = tmp_path / "painted_nfw_manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    cosmology = tmp_path / "pinocchio.test.cosmology.out"
    cosmology.write_text(
        "\n".join(
            [
                "# Cosmological quantities used in PINOCCHIO (h=0.700000)",
                _cosmology_row(0.5, 2000.0, 0.75, 0.5, 0.001, 1000.0),
                _cosmology_row(1.0 / 1.2, 550.0, 0.4, 0.82, 0.01, 5000.0),
                _cosmology_row(1.0 / 1.1, 280.0, 0.35, 0.91, 0.1, 1000.0),
                _cosmology_row(1.0, 0.0, 0.3, 1.0, 10.0, 1.0),
            ]
        ),
        encoding="utf-8",
    )
    params = tmp_path / "params.txt"
    params.write_text(
        "\n".join(
            [
                "BoxSize 100",
                "BoxInH100",
                "GridSize 10",
                "Omega0 0.3",
                "Hubble100 0.7",
                "Sigma8 0",
            ]
        ),
        encoding="utf-8",
    )
    for redshift in (0.0, 0.1, 0.2):
        _write_hmf(tmp_path / f"pinocchio.{redshift:.4f}.test.mf.out", redshift)

    output_dir = tmp_path / "validation"
    outputs = module.run_validation(
        SimpleNamespace(
            manifest=manifest,
            params=params,
            cosmology_table=cosmology,
            hmf_glob=str(tmp_path / "*.mf.out"),
            output_dir=output_dir,
            ell_max=7,
            ell_min_compare=2,
            ell_bin_width=3,
            ell_exact_cap=0,
            limber_match_rtol=0.01,
            limber_match_width=2,
            exact_batch_size=4,
            exact_workers=1,
            radial_order=4,
            profile_order=6,
            exact_relative_tolerance=1.0e-3,
            sigma8_rtol=0.01,
            mask_sht_iterations=0,
            jax_precision="float64",
        )
    )

    assert all(path.exists() for path in outputs)
    with np.load(outputs[0], allow_pickle=False) as result:
        assert result["ell"].shape == (6,)
        assert result["observed_shell"].shape == (2, 6)
        assert result["shell_linear_pseudo_over_fsky"].shape == (2, 6)
        assert result["summed_linear_pseudo_over_fsky"].shape == (6,)
        assert int(result["validation_schema_version"]) == 2
        assert result["sigma8_reference_source"].item() == "cosmology_power_spectrum"
        assert float(result["sigma8_relative_error"]) == pytest.approx(0.0)
    assert len(outputs[1].read_text(encoding="utf-8").splitlines()) > 2
    assert len(outputs[2].read_text(encoding="utf-8").splitlines()) == 3
