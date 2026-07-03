from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from geppetto.catalog import LightconeHaloCatalog, LightconeSparseStencil
from geppetto.cosmology import Cosmology
from geppetto.io import PinocchioMassMap, PinocchioMassSheetTable

EXAMPLE_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "paint_halo_particles_for_pinocchio_segment.py"
)


def _load_example_module():
    spec = importlib.util.spec_from_file_location("paint_halo_particles_for_pinocchio_segment", EXAMPLE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _mass_map(
    pixels: np.ndarray,
    temperature: np.ndarray | None = None,
    *,
    nside: int = 1,
    ordering: str = "RING",
) -> PinocchioMassMap:
    if temperature is None:
        temperature = np.zeros_like(pixels, dtype=np.float64)
    return PinocchioMassMap(
        pixel=np.asarray(pixels, dtype=np.int64),
        temperature=np.asarray(temperature, dtype=np.float64),
        source=Path("massmap.fits"),
        header={
            "PIXTYPE": "HEALPIX",
            "ORDERING": ordering,
            "NSIDE": nside,
            "INDXSCHM": "EXPLICIT",
        },
        nside=nside,
        ordering=ordering,
        index_scheme="EXPLICIT",
        first_pixel=None,
        last_pixel=None,
        aperture_deg=None,
        selection_type=None,
        axis_vector=None,
        filter_name=None,
        filter_considered=None,
        filter_excluded=None,
        filter_included=None,
        filter_excluded_fraction=None,
    )


def _catalog(
    unit_vector: np.ndarray | None = None,
    *,
    redshift: np.ndarray | None = None,
    chi: np.ndarray | None = None,
    mass: np.ndarray | None = None,
) -> LightconeHaloCatalog:
    if unit_vector is None:
        unit_vector = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ]
        )
    n_halo = unit_vector.shape[0]
    if redshift is None:
        redshift = np.linspace(0.1, 0.4, n_halo)
    if chi is None:
        chi = np.linspace(100.0, 400.0, n_halo)
    if mass is None:
        mass = np.arange(1, n_halo + 1, dtype=np.float64) * 10.0
    return LightconeHaloCatalog(
        unit_vector=unit_vector,
        chi=chi,
        mass=mass,
        redshift=redshift,
    )


def _sheets() -> PinocchioMassSheetTable:
    return PinocchioMassSheetTable(
        sheet_ids=np.array([0, 1]),
        z_hi=np.array([0.5, 0.1]),
        z_lo=np.array([0.2, 0.3]),
        delta_z=np.array([0.3, 0.2]),
        chi_hi_mpc_h=np.array([1500.0, 900.0]),
        chi_lo_mpc_h=np.array([700.0, 1200.0]),
        delta_chi_mpc_h=np.array([800.0, 300.0]),
        inv_delta_chi_h_mpc=np.array([1.0 / 800.0, 1.0 / 300.0]),
        da_hi_mpc_h=np.array([1000.0, 800.0]),
        da_lo_mpc_h=np.array([500.0, 600.0]),
        chi3_diff_mpc_h3=np.array([1.0, 2.0]),
        source=Path("sheets.out"),
    )


def test_example_script_help_runs():
    result = subprocess.run(
        [sys.executable, str(EXAMPLE_PATH), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--mass-map" in result.stdout
    assert "--sheet-index" in result.stdout
    assert "--nfw-gradient-demo" in result.stdout
    assert "--nfw-paint" in result.stdout
    assert "--nfw-dense-demo" in result.stdout
    assert "--nfw-taper-radius-factor" in result.stdout
    assert "--nfw-validate-sum-only" in result.stdout
    assert "--nfw-map-derivatives" in result.stdout
    assert "--nfw-concentration-mass-slope" in result.stdout
    assert "--nfw-concentration-redshift-slope" in result.stdout
    assert "--nfw-concentration-mass-pivot" in result.stdout
    assert "--profile" in result.stdout
    assert "--profile-jax-repeat" in result.stdout


def test_timed_stage_disabled_does_not_print(capsys):
    module = _load_example_module()

    with module.timed_stage("quiet stage", enabled=False):
        pass

    captured = capsys.readouterr()
    assert captured.out == ""


def test_timed_stage_enabled_prints_profile_line(capsys):
    module = _load_example_module()

    with module.timed_stage("visible stage", enabled=True):
        pass

    captured = capsys.readouterr()
    assert "[profile]" in captured.out
    assert "visible stage" in captured.out


def test_segment_bounds_sort_redshift_chi_and_compute_scale_factor():
    module = _load_example_module()
    bounds = module.segment_bounds(_sheets(), 1)

    assert bounds["sheet_index"] == 1
    assert bounds["z_lo"] == 0.1
    assert bounds["z_hi"] == 0.3
    assert bounds["chi_lo_mpc_h"] == 900.0
    assert bounds["chi_hi_mpc_h"] == 1200.0
    assert np.isclose(bounds["a_lo"], 1.0 / 1.3)
    assert np.isclose(bounds["a_hi"], 1.0 / 1.1)


def test_segment_bounds_reject_out_of_range_index():
    module = _load_example_module()
    with pytest.raises(ValueError, match="sheet_index"):
        module.segment_bounds(_sheets(), 2)


def test_select_segment_mask_supports_half_open_and_inclusive_bounds():
    module = _load_example_module()
    catalog = _catalog(
        redshift=np.array([0.1, 0.2, 0.3, 0.4]),
        chi=np.array([100.0, 200.0, 300.0, 400.0]),
    )
    bounds = {
        "z_lo": 0.1,
        "z_hi": 0.3,
        "chi_lo_mpc_h": 100.0,
        "chi_hi_mpc_h": 300.0,
    }

    np.testing.assert_array_equal(
        module.select_segment_mask(catalog, bounds, "z", inclusive_upper=False),
        [True, True, False, False],
    )
    np.testing.assert_array_equal(
        module.select_segment_mask(catalog, bounds, "z", inclusive_upper=True),
        [True, True, True, False],
    )
    np.testing.assert_array_equal(
        module.select_segment_mask(catalog, bounds, "chi", inclusive_upper=False),
        [True, True, False, False],
    )
    np.testing.assert_array_equal(
        module.select_segment_mask(catalog, bounds, "chi", inclusive_upper=True),
        [True, True, True, False],
    )


def test_catalog_and_mass_map_validation_errors_are_clear():
    module = _load_example_module()
    with pytest.raises(ValueError, match="redshift"):
        module.validate_catalog_for_binning(
            LightconeHaloCatalog(
                unit_vector=np.ones((2, 3)),
                chi=np.ones(2),
                mass=np.ones(2),
                redshift=np.ones(1),
            )
        )
    with pytest.raises(ValueError, match="RING"):
        module.validate_mass_map(_mass_map(np.array([0]), ordering="NESTED"))
    with pytest.raises(ValueError, match="same length"):
        module.validate_mass_map(
            _mass_map(
                np.array([0]),
                temperature=np.array([1.0, 2.0]),
            )
        )


def test_light_plc_requires_hubble_table():
    module = _load_example_module()
    args = SimpleNamespace(
        light_plc=True,
        hubble_table=None,
        plc_catalog=Path("missing.plc.out"),
        catalog_format="auto",
        redshift_mode="true",
    )
    with pytest.raises(ValueError, match="--hubble-table"):
        module.load_lightcone_catalog(args)


def test_halo_particle_count_map_uses_compact_pixel_order_and_ignores_outside_pixels():
    hp = pytest.importorskip("healpy")
    module = _load_example_module()

    halo_pixels = np.array([0, 5, 8], dtype=np.int64)
    unit_vector = np.stack(hp.pix2vec(1, halo_pixels), axis=-1)
    catalog = _catalog(
        unit_vector=unit_vector,
        mass=np.array([10.0, 20.0, 30.0]),
        redshift=np.array([0.2, 0.2, 0.2]),
        chi=np.array([100.0, 100.0, 100.0]),
    )
    mass_map = _mass_map(
        np.array([5, 0], dtype=np.int64),
        temperature=np.array([100.0, 200.0]),
        nside=1,
    )
    mask = np.array([True, True, True])

    rows, inside = module.halo_rows_in_mass_map(catalog, mask, mass_map)
    out = module.build_halo_particle_count_map(
        catalog,
        mask,
        mass_map,
        particle_mass_msun_h=5.0,
    )

    np.testing.assert_array_equal(rows, [1, 0, -1])
    np.testing.assert_array_equal(inside, [True, True, False])
    np.testing.assert_allclose(out, [4.0, 2.0])
    assert out.shape == mass_map.temperature.shape


def test_save_npz_preserves_pixel_array_and_diagnostics(tmp_path):
    module = _load_example_module()
    mass_map = _mass_map(np.array([5, 0]), temperature=np.array([100.0, 200.0]))
    args = SimpleNamespace(output=tmp_path / "halo_particles.seg000.npz")
    metadata = SimpleNamespace(particle_mass_msun_h=5.0)
    bounds = {
        "sheet_index": 0,
        "z_lo": 0.1,
        "z_hi": 0.2,
        "a_lo": 1.0 / 1.2,
        "a_hi": 1.0 / 1.1,
        "chi_lo_mpc_h": 100.0,
        "chi_hi_mpc_h": 200.0,
    }
    diagnostics = {
        "n_halos_total": 3,
        "n_halos_in_segment": 2,
        "n_halos_in_segment_and_pixels": 2,
        "sum_halo_particle_counts": 6.0,
        "sum_pinocchio_mass_map_values": 300.0,
    }

    module.save_npz(
        args,
        np.array([4.0, 2.0]),
        mass_map,
        bounds,
        metadata,
        diagnostics,
    )

    with np.load(args.output) as data:
        np.testing.assert_allclose(data["halo_particle_counts"], [4.0, 2.0])
        np.testing.assert_array_equal(data["pixel"], [5, 0])
        np.testing.assert_allclose(data["pinocchio_mass_map_values"], [100.0, 200.0])
        assert int(data["n_halos_in_segment_and_pixels"]) == 2
        assert float(data["sum_halo_particle_counts"]) == 6.0


def test_save_npz_can_include_nfw_gradient_diagnostics(tmp_path):
    module = _load_example_module()
    mass_map = _mass_map(np.array([5, 0]), temperature=np.array([100.0, 200.0]))
    args = SimpleNamespace(output=tmp_path / "halo_particles.seg000.npz")
    metadata = SimpleNamespace(particle_mass_msun_h=5.0)
    bounds = {
        "sheet_index": 0,
        "z_lo": 0.1,
        "z_hi": 0.2,
        "a_lo": 1.0 / 1.2,
        "a_hi": 1.0 / 1.1,
        "chi_lo_mpc_h": 100.0,
        "chi_hi_mpc_h": 200.0,
    }
    diagnostics = {
        "n_halos_total": 3,
        "n_halos_in_segment": 2,
        "n_halos_in_segment_and_pixels": 2,
        "sum_halo_particle_counts": 6.0,
        "sum_pinocchio_mass_map_values": 300.0,
    }
    nfw_diagnostics = {
        "nfw_particle_counts": np.array([0.75, 0.5]),
        "nfw_map_derivatives": "concentration",
        "d_nfw_particle_counts_d_concentration_amplitude": np.array([0.1, 0.2]),
        "d_nfw_particle_counts_d_concentration_mass_slope": np.array([0.3, 0.4]),
        "d_nfw_particle_counts_d_concentration_redshift_slope": np.array([0.5, 0.6]),
        "sum_d_nfw_particle_counts_d_concentration_amplitude": 0.3,
        "sum_d_nfw_particle_counts_d_concentration_mass_slope": 0.7,
        "sum_d_nfw_particle_counts_d_concentration_redshift_slope": 1.1,
        "nfw_derivatives_computed": True,
        "nfw_operation_mode": "paint_plus_derivatives",
        "nfw_paint_mode": "sparse",
        "nfw_gradient_mode": "sparse",
        "nfw_objective_mode": "sum_only_sparse",
        "nfw_gradient_demo_n_halos": 2,
        "nfw_compact_pixel_count": 2,
        "nfw_sparse_pair_count": 2,
        "nfw_dense_pair_count": 4,
        "nfw_sparse_compression_factor": 2.0,
        "nfw_sum_particle_counts": 1.25,
        "nfw_concentration_amplitude": 5.71,
        "nfw_d_sum_d_concentration_amplitude": 0.5,
        "nfw_truncation_width_fraction": 0.05,
        "nfw_d_sum_d_truncation_width_fraction": -0.25,
    }

    module.save_npz(
        args,
        np.array([4.0, 2.0]),
        mass_map,
        bounds,
        metadata,
        diagnostics,
        nfw_diagnostics,
    )

    with np.load(args.output) as data:
        np.testing.assert_allclose(data["nfw_particle_counts"], [0.75, 0.5])
        assert str(data["nfw_map_derivatives"]) == "concentration"
        np.testing.assert_allclose(
            data["d_nfw_particle_counts_d_concentration_amplitude"], [0.1, 0.2]
        )
        np.testing.assert_allclose(
            data["d_nfw_particle_counts_d_concentration_mass_slope"], [0.3, 0.4]
        )
        np.testing.assert_allclose(
            data["d_nfw_particle_counts_d_concentration_redshift_slope"], [0.5, 0.6]
        )
        assert float(data["sum_d_nfw_particle_counts_d_concentration_amplitude"]) == 0.3
        assert float(data["sum_d_nfw_particle_counts_d_concentration_mass_slope"]) == 0.7
        assert float(data["sum_d_nfw_particle_counts_d_concentration_redshift_slope"]) == 1.1
        assert bool(data["nfw_derivatives_computed"])
        assert str(data["nfw_operation_mode"]) == "paint_plus_derivatives"
        assert str(data["nfw_paint_mode"]) == "sparse"
        assert str(data["nfw_gradient_mode"]) == "sparse"
        assert str(data["nfw_objective_mode"]) == "sum_only_sparse"
        assert int(data["nfw_gradient_demo_n_halos"]) == 2
        assert int(data["nfw_compact_pixel_count"]) == 2
        assert int(data["nfw_sparse_pair_count"]) == 2
        assert int(data["nfw_dense_pair_count"]) == 4
        assert float(data["nfw_sparse_compression_factor"]) == 2.0
        assert float(data["nfw_sum_particle_counts"]) == 1.25
        assert float(data["nfw_d_sum_d_concentration_amplitude"]) == 0.5
        assert float(data["nfw_d_sum_d_truncation_width_fraction"]) == -0.25


def test_nfw_sparse_total_particle_count_matches_sparse_map_sum():
    module = _load_example_module()
    catalog = LightconeHaloCatalog(
        unit_vector=jnp.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        chi=jnp.asarray([1000.0, 1100.0]),
        mass=jnp.asarray([1.0e13, 2.0e13]),
        redshift=jnp.asarray([0.2, 0.25]),
    )
    stencil = LightconeSparseStencil(
        pix_id=jnp.asarray([0, 1, 0], dtype=jnp.int32),
        halo_id=jnp.asarray([0, 0, 1], dtype=jnp.int32),
        r_perp=jnp.asarray([0.05, 0.10, 0.15]),
        n_pix=2,
    )
    concentration_params = module.ConcentrationParams(amplitude=5.71)
    profile_params = module.NFWProfileParams(truncation_width_fraction=0.05)

    total = module.nfw_sparse_total_particle_count(
        stencil,
        catalog,
        particle_mass_msun_h=1.0e10,
        pixel_area_sr=0.01,
        cosmology=Cosmology(),
        concentration_params=concentration_params,
        profile_params=profile_params,
    )
    sparse_map = module.paint_lightcone_particle_count_map_sparse(
        stencil,
        catalog,
        particle_mass_msun_h=1.0e10,
        pixel_area_sr=0.01,
        cosmology=Cosmology(),
        concentration_params=concentration_params,
        profile_params=profile_params,
    )

    np.testing.assert_allclose(
        np.asarray(total),
        np.asarray(jnp.sum(sparse_map)),
        rtol=1.0e-5,
    )


def test_nfw_gradient_demo_default_sparse_does_not_call_bruteforce(monkeypatch):
    hp = pytest.importorskip("healpy")
    module = _load_example_module()

    def fail_if_called(*args, **kwargs):
        raise AssertionError("The sparse tutorial must not allocate an N_pix x N_halo matrix")

    monkeypatch.setattr(module, "build_lightcone_sparse_stencil_bruteforce", fail_if_called)
    monkeypatch.setattr(module, "healpix_pixel_unit_vectors", fail_if_called)

    halo_pixels = np.array([0, 5], dtype=np.int64)
    unit_vector = np.stack(hp.pix2vec(1, halo_pixels), axis=-1)
    catalog = _catalog(
        unit_vector=unit_vector,
        mass=np.array([1.0e13, 2.0e13]),
        redshift=np.array([0.2, 0.25]),
        chi=np.array([1000.0, 1100.0]),
    )
    mass_map = _mass_map(
        np.array([5, 0], dtype=np.int64),
        temperature=np.array([100.0, 200.0]),
        nside=1,
    )
    metadata = SimpleNamespace(cosmology=Cosmology())

    diagnostics = module.nfw_gradient_demo(
        catalog,
        np.array([True, True]),
        mass_map,
        metadata,
        particle_mass_msun_h=1.0e10,
        concentration_amplitude=5.71,
        truncation_width_fraction=0.05,
        chunk_size=1,
    )

    assert diagnostics["nfw_gradient_mode"] == "sparse"
    assert diagnostics["nfw_derivatives_computed"] is True
    assert diagnostics["nfw_operation_mode"] == "paint_plus_derivatives"
    assert diagnostics["nfw_paint_mode"] == "sparse"
    assert diagnostics["nfw_objective_mode"] == "sum_only_sparse"
    assert diagnostics["nfw_particle_counts"].shape == (2,)
    assert np.all(np.isfinite(diagnostics["nfw_particle_counts"]))
    assert diagnostics["nfw_gradient_demo_n_halos"] == 2
    assert diagnostics["nfw_compact_pixel_count"] == 2
    assert diagnostics["nfw_sparse_pair_count"] == 2
    assert diagnostics["nfw_dense_pair_count"] == 4
    assert diagnostics["nfw_sparse_compression_factor"] == 2.0
    assert np.isfinite(diagnostics["nfw_sum_particle_counts"])
    assert np.isfinite(diagnostics["nfw_d_sum_d_concentration_amplitude"])
    assert np.isfinite(diagnostics["nfw_d_sum_d_truncation_width_fraction"])
    np.testing.assert_allclose(
        np.sum(diagnostics["nfw_particle_counts"]),
        diagnostics["nfw_sum_particle_counts"],
        rtol=1.0e-5,
    )


def test_nfw_gradient_demo_sparse_sum_only_validation_matches_sparse_map_sum():
    hp = pytest.importorskip("healpy")
    module = _load_example_module()

    halo_pixels = np.array([0], dtype=np.int64)
    unit_vector = np.stack(hp.pix2vec(1, halo_pixels), axis=-1)
    catalog = _catalog(
        unit_vector=unit_vector,
        mass=np.array([1.0e13]),
        redshift=np.array([0.2]),
        chi=np.array([1000.0]),
    )
    mass_map = _mass_map(
        np.array([0], dtype=np.int64),
        temperature=np.array([100.0]),
        nside=1,
    )
    metadata = SimpleNamespace(cosmology=Cosmology())

    diagnostics = module.nfw_gradient_demo(
        catalog,
        np.array([True]),
        mass_map,
        metadata,
        particle_mass_msun_h=1.0e10,
        concentration_amplitude=5.71,
        truncation_width_fraction=0.05,
        chunk_size=1,
        validate_sum_only=True,
    )

    assert diagnostics["nfw_gradient_mode"] == "sparse"
    assert diagnostics["nfw_objective_mode"] == "sum_only_sparse"
    assert diagnostics["nfw_particle_counts"].shape == (1,)
    np.testing.assert_allclose(
        np.sum(diagnostics["nfw_particle_counts"]),
        diagnostics["nfw_sum_particle_counts"],
        rtol=1.0e-5,
    )


def test_nfw_map_concentration_derivatives_have_map_shape_and_match_scalar_gradient():
    hp = pytest.importorskip("healpy")
    module = _load_example_module()

    halo_pixels = np.array([0, 5], dtype=np.int64)
    unit_vector = np.stack(hp.pix2vec(1, halo_pixels), axis=-1)
    catalog = _catalog(
        unit_vector=unit_vector,
        mass=np.array([1.0e13, 2.0e13]),
        redshift=np.array([0.2, 0.25]),
        chi=np.array([1000.0, 1100.0]),
    )
    mass_map = _mass_map(
        np.array([5, 0], dtype=np.int64),
        temperature=np.array([100.0, 200.0]),
        nside=1,
    )
    metadata = SimpleNamespace(cosmology=Cosmology())

    diagnostics = module.nfw_gradient_demo(
        catalog,
        np.array([True, True]),
        mass_map,
        metadata,
        particle_mass_msun_h=1.0e10,
        concentration_amplitude=5.71,
        concentration_mass_slope=-0.084,
        concentration_redshift_slope=-0.47,
        concentration_mass_pivot=2.0e12,
        truncation_width_fraction=0.05,
        chunk_size=1,
        map_derivatives="concentration",
    )

    assert diagnostics["nfw_map_derivatives"] == "concentration"
    for key in (
        "d_nfw_particle_counts_d_concentration_amplitude",
        "d_nfw_particle_counts_d_concentration_mass_slope",
        "d_nfw_particle_counts_d_concentration_redshift_slope",
    ):
        assert diagnostics[key].shape == diagnostics["nfw_particle_counts"].shape
        assert np.all(np.isfinite(diagnostics[key]))

    scalar_amp_grad = diagnostics["nfw_d_sum_d_concentration_amplitude"]
    np.testing.assert_allclose(
        np.sum(diagnostics["d_nfw_particle_counts_d_concentration_amplitude"]),
        scalar_amp_grad,
        rtol=1.0e-5,
        atol=1.0e-5 * max(1.0, abs(scalar_amp_grad)),
    )
    np.testing.assert_allclose(
        np.sum(diagnostics["d_nfw_particle_counts_d_concentration_mass_slope"]),
        diagnostics["sum_d_nfw_particle_counts_d_concentration_mass_slope"],
        rtol=1.0e-5,
    )
    np.testing.assert_allclose(
        np.sum(diagnostics["d_nfw_particle_counts_d_concentration_redshift_slope"]),
        diagnostics["sum_d_nfw_particle_counts_d_concentration_redshift_slope"],
        rtol=1.0e-5,
    )


def test_nfw_map_concentration_derivatives_are_sparse_only():
    module = _load_example_module()
    catalog = _catalog(
        unit_vector=np.array([[1.0, 0.0, 0.0]]),
        mass=np.array([1.0e13]),
        redshift=np.array([0.2]),
        chi=np.array([1000.0]),
    )
    mass_map = _mass_map(
        np.array([0], dtype=np.int64),
        temperature=np.array([100.0]),
        nside=1,
    )
    metadata = SimpleNamespace(cosmology=Cosmology())

    with pytest.raises(ValueError, match="only supported for sparse"):
        module.nfw_gradient_demo(
            catalog,
            np.array([True]),
            mass_map,
            metadata,
            particle_mass_msun_h=1.0e10,
            dense_demo=True,
            map_derivatives="concentration",
        )


def test_nfw_map_concentration_derivative_profile_labels(capsys):
    hp = pytest.importorskip("healpy")
    module = _load_example_module()

    halo_pixels = np.array([0], dtype=np.int64)
    unit_vector = np.stack(hp.pix2vec(1, halo_pixels), axis=-1)
    catalog = _catalog(
        unit_vector=unit_vector,
        mass=np.array([1.0e13]),
        redshift=np.array([0.2]),
        chi=np.array([1000.0]),
    )
    mass_map = _mass_map(
        np.array([0], dtype=np.int64),
        temperature=np.array([100.0]),
        nside=1,
    )
    metadata = SimpleNamespace(cosmology=Cosmology())

    module.nfw_gradient_demo(
        catalog,
        np.array([True]),
        mass_map,
        metadata,
        particle_mass_msun_h=1.0e10,
        map_derivatives="concentration",
        profile=True,
    )

    captured = capsys.readouterr()
    assert "NFW map concentration JVPs" in captured.out
    assert "NFW map concentration derivatives to numpy" in captured.out


def test_nfw_paint_only_outputs_map_without_gradients():
    hp = pytest.importorskip("healpy")
    module = _load_example_module()

    halo_pixels = np.array([0], dtype=np.int64)
    unit_vector = np.stack(hp.pix2vec(1, halo_pixels), axis=-1)
    catalog = _catalog(
        unit_vector=unit_vector,
        mass=np.array([1.0e13]),
        redshift=np.array([0.2]),
        chi=np.array([1000.0]),
    )
    mass_map = _mass_map(
        np.array([0], dtype=np.int64),
        temperature=np.array([100.0]),
        nside=1,
    )
    metadata = SimpleNamespace(cosmology=Cosmology())

    diagnostics = module.nfw_gradient_demo(
        catalog,
        np.array([True]),
        mass_map,
        metadata,
        particle_mass_msun_h=1.0e10,
        concentration_amplitude=5.71,
        truncation_width_fraction=0.05,
        chunk_size=1,
        compute_gradients=False,
    )

    assert diagnostics["nfw_paint_mode"] == "sparse"
    assert diagnostics["nfw_derivatives_computed"] is False
    assert diagnostics["nfw_operation_mode"] == "paint_only"
    assert diagnostics["nfw_gradient_mode"] == "not_computed"
    assert diagnostics["nfw_objective_mode"] == "not_computed"
    assert diagnostics["nfw_particle_counts"].shape == (1,)
    assert "nfw_d_sum_d_concentration_amplitude" not in diagnostics
    assert "nfw_d_sum_d_truncation_width_fraction" not in diagnostics
    np.testing.assert_allclose(
        np.sum(diagnostics["nfw_particle_counts"]),
        diagnostics["nfw_sum_particle_counts"],
        rtol=1.0e-5,
    )


def test_nfw_paint_only_rejects_sum_only_validation():
    module = _load_example_module()
    catalog = _catalog(
        unit_vector=np.array([[1.0, 0.0, 0.0]]),
        mass=np.array([1.0e13]),
        redshift=np.array([0.2]),
        chi=np.array([1000.0]),
    )
    mass_map = _mass_map(
        np.array([0], dtype=np.int64),
        temperature=np.array([100.0]),
        nside=1,
    )
    metadata = SimpleNamespace(cosmology=Cosmology())

    with pytest.raises(ValueError, match="requires --nfw-gradient-demo"):
        module.nfw_gradient_demo(
            catalog,
            np.array([True]),
            mass_map,
            metadata,
            particle_mass_msun_h=1.0e10,
            compute_gradients=False,
            validate_sum_only=True,
        )


def test_nfw_paint_only_profile_prints_map_labels_without_derivatives(capsys):
    hp = pytest.importorskip("healpy")
    module = _load_example_module()

    halo_pixels = np.array([0], dtype=np.int64)
    unit_vector = np.stack(hp.pix2vec(1, halo_pixels), axis=-1)
    catalog = _catalog(
        unit_vector=unit_vector,
        mass=np.array([1.0e13]),
        redshift=np.array([0.2]),
        chi=np.array([1000.0]),
    )
    mass_map = _mass_map(
        np.array([0], dtype=np.int64),
        temperature=np.array([100.0]),
        nside=1,
    )
    metadata = SimpleNamespace(cosmology=Cosmology())

    module.nfw_gradient_demo(
        catalog,
        np.array([True]),
        mass_map,
        metadata,
        particle_mass_msun_h=1.0e10,
        concentration_amplitude=5.71,
        truncation_width_fraction=0.05,
        chunk_size=1,
        compute_gradients=False,
        profile=True,
    )

    captured = capsys.readouterr()
    assert "NFW particle map" in captured.out
    assert "NFW particle map to numpy" in captured.out
    assert "NFW derivatives value_and_grad" not in captured.out


def test_nfw_gradient_demo_profile_prints_derivative_and_map_labels(capsys):
    hp = pytest.importorskip("healpy")
    module = _load_example_module()

    halo_pixels = np.array([0], dtype=np.int64)
    unit_vector = np.stack(hp.pix2vec(1, halo_pixels), axis=-1)
    catalog = _catalog(
        unit_vector=unit_vector,
        mass=np.array([1.0e13]),
        redshift=np.array([0.2]),
        chi=np.array([1000.0]),
    )
    mass_map = _mass_map(
        np.array([0], dtype=np.int64),
        temperature=np.array([100.0]),
        nside=1,
    )
    metadata = SimpleNamespace(cosmology=Cosmology())

    module.nfw_gradient_demo(
        catalog,
        np.array([True]),
        mass_map,
        metadata,
        particle_mass_msun_h=1.0e10,
        concentration_amplitude=5.71,
        truncation_width_fraction=0.05,
        chunk_size=1,
        profile=True,
        profile_jax_repeat=True,
    )

    captured = capsys.readouterr()
    assert "NFW derivatives value_and_grad" in captured.out
    assert "NFW derivatives cached value_and_grad" in captured.out
    assert "NFW particle map" in captured.out
    assert "NFW particle map to numpy" in captured.out


def test_nfw_stage_label_distinguishes_scalar_and_map_derivatives():
    module = _load_example_module()

    assert module.nfw_stage_label(False, "none") == "NFW paint only"
    assert (
        module.nfw_stage_label(False, "concentration")
        == "NFW paint + map derivatives"
    )
    assert module.nfw_stage_label(True, "none") == "NFW paint + scalar derivatives"
    assert (
        module.nfw_stage_label(True, "concentration")
        == "NFW paint + scalar and map derivatives"
    )


def test_print_nfw_summary_distinguishes_operation_modes(capsys):
    module = _load_example_module()
    common = {
        "nfw_paint_mode": "sparse",
        "nfw_gradient_demo_n_halos": 1,
        "nfw_compact_pixel_count": 1,
        "nfw_sparse_pair_count": 1,
        "nfw_dense_pair_count": 1,
        "nfw_sparse_compression_factor": 1.0,
        "nfw_sum_particle_counts": 1.25,
    }

    module.print_nfw_gradient_summary(
        {
            **common,
            "nfw_derivatives_computed": False,
            "nfw_operation_mode": "paint_only",
            "nfw_gradient_mode": "not_computed",
            "nfw_objective_mode": "not_computed",
        }
    )
    paint_only = capsys.readouterr().out
    assert "NFW one-halo map:" in paint_only
    assert "Operation mode: paint_only" in paint_only
    assert "Scalar gradients: not computed" in paint_only

    module.print_nfw_gradient_summary(
        {
            **common,
            "nfw_derivatives_computed": False,
            "nfw_operation_mode": "paint_only",
            "nfw_gradient_mode": "not_computed",
            "nfw_objective_mode": "not_computed",
            "nfw_map_derivatives": "concentration",
            "sum_d_nfw_particle_counts_d_concentration_amplitude": 0.5,
            "sum_d_nfw_particle_counts_d_concentration_mass_slope": 1.5,
            "sum_d_nfw_particle_counts_d_concentration_redshift_slope": -2.5,
        }
    )
    map_only = capsys.readouterr().out
    assert "NFW one-halo map + derivatives:" in map_only
    assert "Operation mode: paint_only" in map_only
    assert "Scalar gradients: not computed" in map_only
    assert "Map derivatives: concentration" in map_only

    module.print_nfw_gradient_summary(
        {
            **common,
            "nfw_derivatives_computed": True,
            "nfw_operation_mode": "paint_plus_derivatives",
            "nfw_gradient_mode": "sparse",
            "nfw_objective_mode": "sum_only_sparse",
            "nfw_d_sum_d_concentration_amplitude": 0.5,
            "nfw_d_sum_d_truncation_width_fraction": -0.25,
            "nfw_map_derivatives": "concentration",
            "sum_d_nfw_particle_counts_d_concentration_amplitude": 0.5,
            "sum_d_nfw_particle_counts_d_concentration_mass_slope": 1.5,
            "sum_d_nfw_particle_counts_d_concentration_redshift_slope": -2.5,
        }
    )
    with_derivatives = capsys.readouterr().out
    assert "NFW one-halo map + derivatives:" in with_derivatives
    assert "Operation mode: paint_plus_derivatives" in with_derivatives
    assert "Objective mode: sum_only_sparse" in with_derivatives
    assert "Map derivatives: concentration" in with_derivatives
    assert "Sum d(map)/d concentration amplitude" in with_derivatives
    assert "Sum d(map)/d concentration mass slope" in with_derivatives
    assert "Sum d(map)/d concentration redshift slope" in with_derivatives


def test_local_sparse_stencil_uses_compact_rows_not_global_pixels():
    hp = pytest.importorskip("healpy")
    module = _load_example_module()

    halo_pixels = np.array([0, 5, 8], dtype=np.int64)
    unit_vector = np.stack(hp.pix2vec(1, halo_pixels), axis=-1)
    catalog = _catalog(
        unit_vector=unit_vector,
        mass=np.array([1.0e13, 2.0e13, 3.0e13]),
        redshift=np.array([0.2, 0.25, 0.3]),
        chi=np.array([1000.0, 1100.0, 1200.0]),
    )
    mass_map = _mass_map(
        np.array([5, 0], dtype=np.int64),
        temperature=np.array([100.0, 200.0]),
        nside=1,
    )

    stencil = module.build_lightcone_sparse_stencil_for_mass_map_local(
        mass_map,
        catalog,
        rmax_mpc_h=np.array([1.0, 1.0, 1.0]),
    )

    assert stencil.n_pix == 2
    np.testing.assert_array_equal(np.asarray(stencil.pix_id), [1, 0])
    np.testing.assert_array_equal(np.asarray(stencil.halo_id), [0, 1])
    assert np.all(np.asarray(stencil.r_perp) <= 1.0)


def test_nfw_gradient_demo_dense_reports_dense_map_sum_objective():
    hp = pytest.importorskip("healpy")
    module = _load_example_module()

    pixels = np.array([0], dtype=np.int64)
    unit_vector = np.stack(hp.pix2vec(1, pixels), axis=-1)
    catalog = _catalog(
        unit_vector=unit_vector,
        mass=np.array([1.0e13]),
        redshift=np.array([0.2]),
        chi=np.array([1000.0]),
    )
    mass_map = _mass_map(
        pixels,
        temperature=np.array([100.0]),
        nside=1,
    )
    metadata = SimpleNamespace(cosmology=Cosmology())

    diagnostics = module.nfw_gradient_demo(
        catalog,
        np.array([True]),
        mass_map,
        metadata,
        particle_mass_msun_h=1.0e10,
        concentration_amplitude=5.71,
        truncation_width_fraction=0.05,
        chunk_size=1,
        dense_demo=True,
    )

    assert diagnostics["nfw_gradient_mode"] == "dense"
    assert diagnostics["nfw_objective_mode"] == "dense_map_sum"
    assert diagnostics["nfw_particle_counts"].shape == (1,)
    assert np.isfinite(diagnostics["nfw_sum_particle_counts"])
    assert np.isfinite(diagnostics["nfw_d_sum_d_concentration_amplitude"])
    assert np.isfinite(diagnostics["nfw_d_sum_d_truncation_width_fraction"])
    np.testing.assert_allclose(
        np.sum(diagnostics["nfw_particle_counts"]),
        diagnostics["nfw_sum_particle_counts"],
        rtol=1.0e-5,
    )


def test_nfw_gradient_demo_sparse_matches_dense_when_stencil_contains_all_pairs():
    hp = pytest.importorskip("healpy")
    module = _load_example_module()

    pixels = np.array([5, 0], dtype=np.int64)
    pixel_unit_vectors = np.stack(hp.pix2vec(1, pixels), axis=-1)
    catalog = _catalog(
        unit_vector=pixel_unit_vectors,
        mass=np.array([1.0e13, 2.0e13]),
        redshift=np.array([0.2, 0.25]),
        chi=np.array([1000.0, 1100.0]),
    )
    mass_map = _mass_map(
        pixels,
        temperature=np.array([100.0, 200.0]),
        nside=1,
    )
    metadata = SimpleNamespace(cosmology=Cosmology())

    sparse = module.nfw_gradient_demo(
        catalog,
        np.array([True, True]),
        mass_map,
        metadata,
        particle_mass_msun_h=1.0e10,
        concentration_amplitude=5.71,
        truncation_width_fraction=0.05,
        chunk_size=1,
        taper_radius_factor=1.0e6,
    )
    dense = module.nfw_gradient_demo(
        catalog,
        np.array([True, True]),
        mass_map,
        metadata,
        particle_mass_msun_h=1.0e10,
        concentration_amplitude=5.71,
        truncation_width_fraction=0.05,
        chunk_size=1,
        taper_radius_factor=1.0e6,
        dense_demo=True,
    )

    assert sparse["nfw_gradient_mode"] == "sparse"
    assert dense["nfw_gradient_mode"] == "dense"
    assert sparse["nfw_objective_mode"] == "sum_only_sparse"
    assert dense["nfw_objective_mode"] == "dense_map_sum"
    np.testing.assert_allclose(
        sparse["nfw_particle_counts"],
        dense["nfw_particle_counts"],
        rtol=1.0e-5,
    )
    assert sparse["nfw_sparse_pair_count"] == sparse["nfw_dense_pair_count"]
    np.testing.assert_allclose(
        sparse["nfw_sum_particle_counts"],
        dense["nfw_sum_particle_counts"],
        rtol=1.0e-5,
    )
    np.testing.assert_allclose(
        sparse["nfw_d_sum_d_concentration_amplitude"],
        dense["nfw_d_sum_d_concentration_amplitude"],
        rtol=1.0e-5,
    )
    np.testing.assert_allclose(
        sparse["nfw_d_sum_d_truncation_width_fraction"],
        dense["nfw_d_sum_d_truncation_width_fraction"],
        rtol=1.0e-5,
    )


def test_nfw_gradient_demo_sparse_uses_fewer_pairs_for_local_stencil():
    hp = pytest.importorskip("healpy")
    module = _load_example_module()

    pixels = np.array([0, 5], dtype=np.int64)
    unit_vector = np.stack(hp.pix2vec(1, np.array([0], dtype=np.int64)), axis=-1)
    catalog = _catalog(
        unit_vector=unit_vector,
        mass=np.array([1.0e13]),
        redshift=np.array([0.2]),
        chi=np.array([1000.0]),
    )
    mass_map = _mass_map(
        pixels,
        temperature=np.array([100.0, 200.0]),
        nside=1,
    )
    metadata = SimpleNamespace(cosmology=Cosmology())

    diagnostics = module.nfw_gradient_demo(
        catalog,
        np.array([True]),
        mass_map,
        metadata,
        particle_mass_msun_h=1.0e10,
        concentration_amplitude=5.71,
        truncation_width_fraction=0.05,
        chunk_size=1,
    )

    assert diagnostics["nfw_sparse_pair_count"] < diagnostics["nfw_dense_pair_count"]
    assert diagnostics["nfw_sparse_compression_factor"] > 1.0
    assert diagnostics["nfw_particle_counts"].shape == (2,)


def test_write_output_fits_preserves_compact_pixel_table(tmp_path):
    fits = pytest.importorskip("astropy.io.fits")
    module = _load_example_module()
    mass_map = _mass_map(np.array([5, 0]), temperature=np.array([100.0, 200.0]))
    bounds = {
        "sheet_index": 0,
        "z_lo": 0.1,
        "z_hi": 0.2,
        "a_lo": 1.0 / 1.2,
        "a_hi": 1.0 / 1.1,
        "chi_lo_mpc_h": 100.0,
        "chi_hi_mpc_h": 200.0,
    }
    diagnostics = {
        "particle_mass_msun_h": 5.0,
        "n_halos_in_segment": 2,
        "n_halos_in_segment_and_pixels": 2,
    }
    path = tmp_path / "halo_particles.seg000.fits"

    module.write_output_fits(
        path,
        np.array([4.0, 2.0]),
        mass_map,
        bounds,
        diagnostics,
    )

    with fits.open(path) as hdul:
        table = hdul["HEALPIX"]
        np.testing.assert_array_equal(table.data["PIXEL"], [5, 0])
        np.testing.assert_allclose(table.data["TEMPERATURE"], [4.0, 2.0])
        assert table.header["ORDERING"] == "RING"
        assert table.header["MAPTYPE"] == "HALO_PARTICLE_COUNT"
        assert table.header["SHEETIDX"] == 0
