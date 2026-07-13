from __future__ import annotations

import csv
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
from geppetto.io import (
    PinocchioMassMap,
    PinocchioMassSheetTable,
    pinocchio_plc_angle_unit_vectors,
)

EXAMPLE_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "paint_halo_particles_for_pinocchio_segment.py"
)


def _load_example_module():
    spec = importlib.util.spec_from_file_location("paint_halo_particles_for_pinocchio_segment", EXAMPLE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _probe_example_precision(*args: str) -> list[str]:
    argv = [str(EXAMPLE_PATH), *args]
    code = f"""
import importlib.util
import sys

sys.argv = {argv!r}
spec = importlib.util.spec_from_file_location("precision_probe", {str(EXAMPLE_PATH)!r})
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = module
spec.loader.exec_module(module)
print(module._CONFIGURED_JAX_PRECISION)
print(module.jnp.asarray([1.0]).dtype)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip().splitlines()


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


def _workflow_args(**overrides) -> SimpleNamespace:
    values = {
        "mass_map": Path("pinocchio.example.massmap.seg000.fits"),
        "sheet_index": 0,
        "output": Path("painted.seg000.npz"),
        "mass_map_glob": None,
        "output_dir": None,
        "output_fits": None,
        "bounds": "z",
        "last_segment_inclusive": False,
        "mode": "paint",
        "concentration_amplitude": 5.71,
        "concentration_mass_slope": -0.084,
        "concentration_redshift_slope": -0.47,
        "concentration_mass_pivot": 2.0e12,
        "truncation_width_fraction": 0.05,
        "nfw_chunk_size": 1,
        "nfw_taper_radius_factor": 10.0,
        "nfw_dense_demo": False,
        "stencil_query_mode": "inclusive",
        "stencil_diagnostics": False,
        "stencil_compare_query_modes": False,
        "mpi_plc_parts": False,
        "segment_workers": 1,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _fake_segment_result(
    module,
    *,
    segment_index: int,
    mass_map_path: Path,
    output_npz: Path,
    output_fits: Path | None,
    inclusive_upper: bool,
):
    return module.CalibrationSegmentResult(
        segment_index=segment_index,
        mass_map_path=mass_map_path,
        output_npz=output_npz,
        output_fits=output_fits,
        bounds={
            "sheet_index": segment_index,
            "z_lo": 0.1,
            "z_hi": 0.2,
            "a_lo": 1.0 / 1.2,
            "a_hi": 1.0 / 1.1,
            "chi_lo_mpc_h": 100.0,
            "chi_hi_mpc_h": 200.0,
        },
        inclusive_upper=inclusive_upper,
        mass_map=_mass_map(np.array([0]), temperature=np.array([1.0])),
        halo_particle_counts=np.array([1.0]),
        diagnostics={
            "particle_mass_msun_h": 1.0,
            "n_halos_total": 1,
            "n_halos_in_segment": 1,
            "n_halos_in_segment_and_pixels": 1,
            "sum_halo_particle_counts": 1.0,
            "sum_pinocchio_mass_map_values": 1.0,
        },
        nfw_diagnostics={
            "pipeline_mode": "paint",
            "particle_mass_msun_h": 1.0,
            "nfw_particle_counts": np.array([1.0]),
            "nfw_map_derivatives": "none",
            "nfw_paint_mode": "sparse",
            "nfw_selected_halo_count": 1,
            "nfw_compact_pixel_count": 1,
            "nfw_sparse_pair_count": 1,
            "nfw_dense_pair_count": 1,
            "nfw_sparse_compression_factor": 1.0,
            "nfw_sum_particle_counts": 1.0,
            "nfw_concentration_amplitude": 5.71,
            "nfw_concentration_mass_slope": -0.084,
            "nfw_concentration_redshift_slope": -0.47,
            "nfw_concentration_mass_pivot": 2.0e12,
            "nfw_truncation_width_fraction": 0.05,
        },
    )


def test_example_script_help_runs():
    result = subprocess.run(
        [sys.executable, str(EXAMPLE_PATH), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    normalized_help = " ".join(result.stdout.split())
    assert "--mass-map" in result.stdout
    assert "--mass-map-glob" in result.stdout
    assert "--output-dir" in result.stdout
    assert "--sheet-index" in result.stdout
    assert "--output-fits" in result.stdout
    assert "--mode" in result.stdout
    assert "--concentration-amplitude" in result.stdout
    assert "--concentration-mass-slope" in result.stdout
    assert "--concentration-redshift-slope" in result.stdout
    assert "--concentration-mass-pivot" in result.stdout
    assert "--truncation-width-fraction" in result.stdout
    assert "In single-segment mode, use an inclusive upper segment bound." in normalized_help
    assert "the final discovered segment is inclusive automatically." in normalized_help
    assert "--nfw-paint" not in result.stdout
    assert "--nfw-gradient-demo" not in result.stdout
    assert "--nfw-map-derivatives" not in result.stdout
    assert "--profile" not in result.stdout
    assert "--profile-jax-repeat" not in result.stdout
    assert "--nfw-dense-demo" not in result.stdout
    assert "--nfw-validate-sum-only" not in result.stdout
    assert "--nfw-chunk-size" not in result.stdout
    assert "--nfw-taper-radius-factor" not in result.stdout
    assert "--stencil-query-mode" not in result.stdout
    assert "--stencil-diagnostics" not in result.stdout
    assert "--stencil-compare-query-modes" not in result.stdout
    assert "--mpi-plc-parts" not in result.stdout
    assert "--mpi-output-mode" not in result.stdout
    assert "--segment-workers" not in result.stdout
    assert "--jax-precision" not in result.stdout


def test_example_script_rejects_removed_mpi_output_mode():
    result = subprocess.run(
        [
            sys.executable,
            str(EXAMPLE_PATH),
            "--params",
            "params.txt",
            "--sheets",
            "sheets.out",
            "--plc-catalog",
            "plc.out",
            "--mass-map",
            "massmap.seg000.fits",
            "--sheet-index",
            "0",
            "--output",
            "painted.npz",
            "--mpi-output-mode",
            "reduce",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "unrecognized arguments: --mpi-output-mode" in result.stderr


def test_example_module_defaults_to_float64_jax_precision():
    assert _probe_example_precision() == ["float64", "float64"]


def test_example_module_allows_float32_jax_precision():
    assert _probe_example_precision("--jax-precision", "float32") == ["float32", "float32"]


def test_example_script_rejects_invalid_jax_precision():
    result = subprocess.run(
        [sys.executable, str(EXAMPLE_PATH), "--jax-precision", "bad"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "invalid choice" in result.stderr


def test_float32_precision_warning_is_root_only(capsys):
    module = _load_example_module()
    args = SimpleNamespace(jax_precision="float32")

    module.warn_if_float32_precision(args, module.MpiContext(enabled=True, rank=1, size=2))
    assert capsys.readouterr().out == ""

    module.warn_if_float32_precision(args, module.MpiContext(enabled=True, rank=0, size=2))
    assert "float32 saves memory" in capsys.readouterr().out


def test_parse_segment_index_from_mass_map_path():
    module = _load_example_module()

    assert module.parse_segment_index_from_mass_map_path(
        Path("pinocchio.massmap.seg000.fits")
    ) == 0
    assert module.parse_segment_index_from_mass_map_path(
        Path("pinocchio.massmap.seg012.fits")
    ) == 12
    with pytest.raises(ValueError, match="Cannot parse segment index"):
        module.parse_segment_index_from_mass_map_path(Path("pinocchio.massmap.fits"))


def test_discover_mass_map_segments_sorts_by_segment_index(tmp_path):
    module = _load_example_module()
    seg002 = tmp_path / "run.massmap.seg002.fits"
    seg000 = tmp_path / "run.massmap.seg000.fits"
    seg001 = tmp_path / "run.massmap.seg001.fits"
    for path in (seg002, seg000, seg001):
        path.touch()

    discovered = module.discover_mass_map_segments(str(tmp_path / "*.fits"))

    assert discovered == [(0, seg000), (1, seg001), (2, seg002)]


def test_discover_mass_map_segments_rejects_duplicates_and_empty(tmp_path):
    module = _load_example_module()
    (tmp_path / "run.massmap.seg001.fits").touch()
    (tmp_path / "other.massmap.seg001.fits").touch()

    with pytest.raises(ValueError, match="Duplicate mass-map segment index"):
        module.discover_mass_map_segments(str(tmp_path / "*.fits"))
    with pytest.raises(ValueError, match="No mass-map segments match glob"):
        module.discover_mass_map_segments(str(tmp_path / "missing*.fits"))


def test_discover_plc_catalog_parts_sorts_and_validates_mpi_size(tmp_path):
    module = _load_example_module()
    base = tmp_path / "pinocchio.demo.plc.out"
    part1 = Path(f"{base}.1")
    part0 = Path(f"{base}.0")
    part1.touch()
    part0.touch()

    parts = module.discover_plc_catalog_parts(base)

    assert parts == [part0, part1]
    module.validate_mpi_plc_part_count(parts, mpi_size=2)
    with pytest.raises(ValueError, match="one rank per PLC part"):
        module.validate_mpi_plc_part_count(parts, mpi_size=3)


def test_discover_plc_catalog_parts_rejects_empty_and_noncontiguous(tmp_path):
    module = _load_example_module()
    base = tmp_path / "pinocchio.demo.plc.out"

    with pytest.raises(ValueError, match="No split PLC part files"):
        module.discover_plc_catalog_parts(base)

    Path(f"{base}.0").touch()
    Path(f"{base}.2").touch()
    with pytest.raises(ValueError, match="contiguous"):
        module.discover_plc_catalog_parts(base)


def test_validate_segment_workflow_args_accepts_single_and_all_modes(tmp_path):
    module = _load_example_module()

    assert module.validate_segment_workflow_args(_workflow_args()) == "single"
    assert (
        module.validate_segment_workflow_args(
            _workflow_args(
                mass_map=None,
                sheet_index=None,
                output=None,
                mass_map_glob=str(tmp_path / "*.fits"),
                output_dir=tmp_path / "painted",
            )
        )
        == "all"
    )


def test_validate_segment_workflow_args_rejects_mixed_and_incomplete(tmp_path):
    module = _load_example_module()

    with pytest.raises(ValueError, match="not both"):
        module.validate_segment_workflow_args(
            _workflow_args(
                mass_map_glob=str(tmp_path / "*.fits"),
                output_dir=tmp_path / "painted",
            )
        )
    with pytest.raises(ValueError, match="Provide either"):
        module.validate_segment_workflow_args(
            _workflow_args(sheet_index=None)
        )
    with pytest.raises(ValueError, match="only supported in single-segment"):
        module.validate_segment_workflow_args(
            _workflow_args(
                mass_map=None,
                sheet_index=None,
                output=None,
                mass_map_glob=str(tmp_path / "*.fits"),
                output_dir=tmp_path / "painted",
                output_fits=tmp_path / "single.fits",
            )
        )
    with pytest.raises(ValueError, match="only in single-segment"):
        module.validate_segment_workflow_args(
            _workflow_args(
                mass_map=None,
                sheet_index=None,
                output=None,
                mass_map_glob=str(tmp_path / "*.fits"),
                output_dir=tmp_path / "painted",
                stencil_compare_query_modes=True,
            )
        )


def test_validate_mpi_workflow_args_rejects_missing_flag_and_query_comparison(tmp_path):
    module = _load_example_module()
    base = tmp_path / "pinocchio.demo.plc.out"
    args = _workflow_args(plc_catalog=base, segment_workers=1, mpi_plc_parts=False)

    with pytest.raises(ValueError, match="requires --mpi-plc-parts"):
        module.validate_mpi_workflow_args(
            args,
            workflow="single",
            mpi_context=module.MpiContext(enabled=False, size=2),
        )

    args = _workflow_args(
        mass_map=None,
        sheet_index=None,
        output=None,
        mass_map_glob=str(tmp_path / "*.fits"),
        output_dir=tmp_path / "painted",
        plc_catalog=base,
        segment_workers=1,
        mpi_plc_parts=True,
        stencil_compare_query_modes=True,
    )
    with pytest.raises(ValueError, match="not supported with --mpi-plc-parts"):
        module.validate_mpi_workflow_args(
            args,
            workflow="all",
            mpi_context=module.MpiContext(enabled=True, size=2),
        )


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


def test_mass_map_pixel_index_dense_backend_preserves_compact_order():
    module = _load_example_module()
    index = module.MassMapPixelIndex.from_pixels(
        np.array([5, 0, 8], dtype=np.int64),
        max_dense_bytes=1024,
    )

    assert index.backend == "dense"
    np.testing.assert_array_equal(
        index.lookup(np.array([0, 5, 7, 8, -1], dtype=np.int64)),
        [1, 0, -1, 2, -1],
    )


def test_mass_map_pixel_index_sorted_backend_preserves_compact_order():
    module = _load_example_module()
    index = module.MassMapPixelIndex.from_pixels(
        np.array([5, 0, 8], dtype=np.int64),
        max_dense_bytes=0,
    )

    assert index.backend == "sorted"
    np.testing.assert_array_equal(
        index.lookup(np.array([8, 1, 0, 5], dtype=np.int64)),
        [2, -1, 1, 0],
    )


def test_mass_map_pixel_index_rejects_negative_pixels():
    module = _load_example_module()

    with pytest.raises(ValueError, match="non-negative"):
        module.MassMapPixelIndex.from_pixels(np.array([0, -1], dtype=np.int64))


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


def test_load_rank_local_lightcone_catalog_reads_this_rank_part(tmp_path, monkeypatch):
    module = _load_example_module()
    base = tmp_path / "pinocchio.demo.plc.out"
    Path(f"{base}.0").touch()
    Path(f"{base}.1").touch()
    calls = []

    class RawCatalog:
        def to_lightcone_catalog(self, *, redshift):
            assert redshift == "true"
            return _catalog()

    def fake_reader(path, *, format):
        calls.append((Path(path), format))
        return RawCatalog()

    monkeypatch.setattr(module, "read_pinocchio_lightcone_catalog", fake_reader)
    args = SimpleNamespace(
        light_plc=False,
        hubble_table=None,
        plc_catalog=base,
        catalog_format="auto",
        redshift_mode="true",
    )

    catalog = module.load_rank_local_lightcone_catalog(
        args,
        module.MpiContext(enabled=True, rank=1, size=2),
    )

    assert catalog.mass.shape[0] == 4
    assert calls == [(Path(f"{base}.1"), "auto")]


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


def test_pinocchio_plc_angles_bin_to_mass_map_internal_basis():
    module = _load_example_module()

    unit_vector = pinocchio_plc_angle_unit_vectors(
        np.array([90.0]),
        np.array([0.0]),
    )
    catalog = _catalog(
        unit_vector=unit_vector,
        mass=np.array([10.0]),
        redshift=np.array([0.2]),
        chi=np.array([100.0]),
    )
    mass_map = _mass_map(
        np.array([5, 0], dtype=np.int64),
        temperature=np.array([100.0, 200.0]),
        nside=1,
    )
    mask = np.array([True])

    rows, inside = module.halo_rows_in_mass_map(catalog, mask, mass_map)
    stencil = module.build_lightcone_sparse_stencil_for_mass_map_local(
        mass_map,
        catalog,
        rmax_mpc_h=np.array([100.0]),
    )

    np.testing.assert_array_equal(rows, [1])
    np.testing.assert_array_equal(inside, [True])
    np.testing.assert_array_equal(np.asarray(stencil.pix_id), [1])
    np.testing.assert_array_equal(np.asarray(stencil.halo_id), [0])


def test_local_sparse_stencil_preserves_float64_geometry_dtype():
    hp = pytest.importorskip("healpy")
    module = _load_example_module()
    pixel = np.array([0], dtype=np.int64)
    unit_vector = np.stack(hp.pix2vec(1, pixel), axis=-1)
    catalog = _catalog(
        unit_vector=unit_vector,
        mass=np.array([1.0e13], dtype=np.float64),
        redshift=np.array([0.2], dtype=np.float64),
        chi=np.array([1000.0], dtype=np.float64),
    )
    mass_map = _mass_map(pixel, temperature=np.array([100.0]), nside=1)

    stencil = module.build_lightcone_sparse_stencil_for_mass_map_local(
        mass_map,
        catalog,
        rmax_mpc_h=np.array([10.0], dtype=np.float64),
    )

    assert stencil.r_perp.dtype == jnp.float64


def test_save_npz_omits_pinocchio_input_arrays_and_keeps_diagnostics(tmp_path):
    module = _load_example_module()
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
        bounds,
        metadata,
        diagnostics,
    )

    with np.load(args.output) as data:
        np.testing.assert_allclose(data["halo_particle_counts"], [4.0, 2.0])
        assert "pixel" not in data
        assert "pinocchio_mass_map_values" not in data
        assert "nside" not in data
        assert "ordering" not in data
        assert "sum_pinocchio_mass_map_values" not in data
        assert int(data["n_halos_in_segment_and_pixels"]) == 2
        assert float(data["sum_halo_particle_counts"]) == 6.0


def test_save_npz_can_include_nfw_calibration_diagnostics(tmp_path):
    module = _load_example_module()
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
        "pipeline_mode": "derivatives",
        "nfw_particle_counts": np.array([0.75, 0.5]),
        "nfw_map_derivatives": "concentration",
        "d_nfw_particle_counts_d_concentration_amplitude": np.array([0.1, 0.2]),
        "d_nfw_particle_counts_d_concentration_mass_slope": np.array([0.3, 0.4]),
        "d_nfw_particle_counts_d_concentration_redshift_slope": np.array([0.5, 0.6]),
        "nfw_paint_mode": "sparse",
        "nfw_selected_halo_count": 2,
        "nfw_compact_pixel_count": 2,
        "nfw_sparse_pair_count": 2,
        "nfw_dense_pair_count": 4,
        "nfw_sparse_compression_factor": 2.0,
        "nfw_sum_particle_counts": 1.25,
        "nfw_concentration_amplitude": 5.71,
        "nfw_concentration_mass_slope": -0.084,
        "nfw_concentration_redshift_slope": -0.47,
        "nfw_concentration_mass_pivot": 2.0e12,
        "nfw_truncation_width_fraction": 0.05,
    }

    module.save_npz(
        args,
        np.array([4.0, 2.0]),
        bounds,
        metadata,
        diagnostics,
        nfw_diagnostics,
    )

    with np.load(args.output) as data:
        assert str(data["pipeline_mode"]) == "derivatives"
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
        assert "sum_d_nfw_particle_counts_d_concentration_amplitude" not in data
        assert "sum_d_nfw_particle_counts_d_concentration_mass_slope" not in data
        assert "sum_d_nfw_particle_counts_d_concentration_redshift_slope" not in data
        assert str(data["nfw_paint_mode"]) == "sparse"
        assert int(data["nfw_selected_halo_count"]) == 2
        assert int(data["nfw_compact_pixel_count"]) == 2
        assert int(data["nfw_sparse_pair_count"]) == 2
        assert int(data["nfw_dense_pair_count"]) == 4
        assert float(data["nfw_sparse_compression_factor"]) == 2.0
        assert float(data["nfw_sum_particle_counts"]) == 1.25
        assert float(data["nfw_concentration_amplitude"]) == 5.71
        assert float(data["nfw_concentration_mass_slope"]) == -0.084
        assert float(data["nfw_concentration_redshift_slope"]) == -0.47
        assert float(data["nfw_concentration_mass_pivot"]) == 2.0e12
        assert float(data["nfw_truncation_width_fraction"]) == 0.05


def test_run_calibration_for_segment_writes_npz_with_derivative_arrays(tmp_path, monkeypatch):
    pytest.importorskip("healpy")
    module = _load_example_module()
    catalog, _, mass_map, _ = _single_pixel_pipeline_case()
    metadata = SimpleNamespace(particle_mass_msun_h=1.0e10, cosmology=Cosmology())
    output_npz = tmp_path / "painted_nfw.seg000.npz"
    monkeypatch.setattr(
        module,
        "read_pinocchio_mass_map_fits",
        lambda path: mass_map,
    )

    row = module.run_calibration_for_segment(
        segment_index=0,
        mass_map_path=tmp_path / "pinocchio.example.massmap.seg000.fits",
        output_npz=output_npz,
        output_fits=None,
        catalog=catalog,
        sheets=_sheets(),
        metadata=metadata,
        particle_mass=metadata.particle_mass_msun_h,
        args=_workflow_args(mode="derivatives"),
        profile=False,
        compute_map_derivatives=True,
        inclusive_upper=False,
    )

    assert row["segment_index"] == 0
    assert row["inclusive_upper"] is False
    assert row["output_npz"] == str(output_npz)
    with np.load(output_npz) as data:
        assert "halo_particle_counts" in data
        assert "nfw_particle_counts" in data
        assert "pinocchio_mass_map_values" not in data
        assert "pixel" not in data
        assert "nside" not in data
        assert "ordering" not in data
        assert "sheet_index" in data
        assert "nfw_sum_particle_counts" in data
        assert "pipeline_mode" in data
        assert str(data["pipeline_mode"]) == "derivatives"
        assert data["nfw_particle_counts"].shape == data["halo_particle_counts"].shape
        assert "d_nfw_particle_counts_d_concentration_amplitude" in data
        assert "d_nfw_particle_counts_d_concentration_mass_slope" in data
        assert "d_nfw_particle_counts_d_concentration_redshift_slope" in data
        assert (
            data["d_nfw_particle_counts_d_concentration_amplitude"].shape
            == data["halo_particle_counts"].shape
        )


def test_run_calibration_for_segment_saves_stencil_diagnostics(tmp_path, monkeypatch):
    pytest.importorskip("healpy")
    module = _load_example_module()
    catalog, _, mass_map, _ = _single_pixel_pipeline_case()
    metadata = SimpleNamespace(particle_mass_msun_h=1.0e10, cosmology=Cosmology())
    output_npz = tmp_path / "painted_nfw.seg000.npz"
    monkeypatch.setattr(
        module,
        "read_pinocchio_mass_map_fits",
        lambda path: mass_map,
    )

    module.run_calibration_for_segment(
        segment_index=0,
        mass_map_path=tmp_path / "pinocchio.example.massmap.seg000.fits",
        output_npz=output_npz,
        output_fits=None,
        catalog=catalog,
        sheets=_sheets(),
        metadata=metadata,
        particle_mass=metadata.particle_mass_msun_h,
        args=_workflow_args(stencil_diagnostics=True),
        profile=False,
        compute_map_derivatives=False,
        inclusive_upper=False,
    )

    with np.load(output_npz) as data:
        assert str(data["stencil_query_mode"]) == "inclusive"
        assert "stencil_query_pixels_total" in data
        assert "stencil_inside_domain_total" in data
        assert "stencil_kept_pairs_total" in data
        assert "stencil_inside_over_query" in data
        assert "stencil_kept_over_query" in data
        assert "stencil_kept_over_inside" in data
        assert "stencil_build_seconds" in data
        assert int(data["stencil_query_pixels_total"]) >= int(data["stencil_inside_domain_total"])
        assert int(data["stencil_inside_domain_total"]) >= int(data["stencil_kept_pairs_total"])


def test_run_calibration_for_segment_saves_query_mode_comparison(tmp_path, monkeypatch):
    pytest.importorskip("healpy")
    module = _load_example_module()
    catalog, _, mass_map, _ = _single_pixel_pipeline_case()
    metadata = SimpleNamespace(particle_mass_msun_h=1.0e10, cosmology=Cosmology())
    output_npz = tmp_path / "painted_nfw.seg000.npz"
    monkeypatch.setattr(
        module,
        "read_pinocchio_mass_map_fits",
        lambda path: mass_map,
    )

    module.run_calibration_for_segment(
        segment_index=0,
        mass_map_path=tmp_path / "pinocchio.example.massmap.seg000.fits",
        output_npz=output_npz,
        output_fits=None,
        catalog=catalog,
        sheets=_sheets(),
        metadata=metadata,
        particle_mass=metadata.particle_mass_msun_h,
        args=_workflow_args(stencil_compare_query_modes=True),
        profile=False,
        compute_map_derivatives=False,
        inclusive_upper=False,
    )

    with np.load(output_npz) as data:
        for key in (
            "inclusive_stencil_seconds",
            "center_stencil_seconds",
            "inclusive_query_pixels_total",
            "center_query_pixels_total",
            "inclusive_kept_pairs_total",
            "center_kept_pairs_total",
            "max_abs_map_difference",
            "sum_abs_map_difference",
            "relative_sum_difference",
            "differing_pixels",
            "maps_allclose",
        ):
            assert key in data
        assert int(data["center_query_pixels_total"]) <= int(data["inclusive_query_pixels_total"])


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


def _single_pixel_pipeline_case():
    hp = pytest.importorskip("healpy")
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
    return catalog, np.array([True]), mass_map, metadata


def test_run_nfw_calibration_pipeline_default_sparse_does_not_call_bruteforce(monkeypatch):
    hp = pytest.importorskip("healpy")
    module = _load_example_module()

    def fail_if_called(*args, **kwargs):
        raise AssertionError("The sparse pipeline must not allocate an N_pix x N_halo matrix")

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

    diagnostics = module.run_nfw_calibration_pipeline(
        catalog,
        np.array([True, True]),
        mass_map,
        metadata,
        particle_mass_msun_h=1.0e10,
        pipeline_mode="paint",
        concentration_amplitude=5.71,
        truncation_width_fraction=0.05,
        chunk_size=1,
    )

    assert diagnostics["pipeline_mode"] == "paint"
    assert diagnostics["nfw_paint_mode"] == "sparse"
    assert diagnostics["nfw_map_derivatives"] == "none"
    assert diagnostics["nfw_particle_counts"].shape == (2,)
    assert np.all(np.isfinite(diagnostics["nfw_particle_counts"]))
    assert diagnostics["nfw_selected_halo_count"] == 2
    assert diagnostics["nfw_compact_pixel_count"] == 2
    assert diagnostics["nfw_sparse_pair_count"] == 2
    assert diagnostics["nfw_dense_pair_count"] == 4
    assert diagnostics["nfw_sparse_compression_factor"] == 2.0
    assert np.isfinite(diagnostics["nfw_sum_particle_counts"])
    assert "d_nfw_particle_counts_d_concentration_amplitude" not in diagnostics
    np.testing.assert_allclose(
        np.sum(diagnostics["nfw_particle_counts"]),
        diagnostics["nfw_sum_particle_counts"],
        rtol=1.0e-5,
    )


def test_run_nfw_calibration_pipeline_derivative_mode_saves_map_derivatives():
    module = _load_example_module()
    catalog, mask, mass_map, metadata = _single_pixel_pipeline_case()

    diagnostics = module.run_nfw_calibration_pipeline(
        catalog,
        mask,
        mass_map,
        metadata,
        particle_mass_msun_h=1.0e10,
        pipeline_mode="derivatives",
        chunk_size=1,
        compute_map_derivatives=True,
    )

    assert diagnostics["pipeline_mode"] == "derivatives"
    assert diagnostics["nfw_map_derivatives"] == "concentration"
    for key in (
        "d_nfw_particle_counts_d_concentration_amplitude",
        "d_nfw_particle_counts_d_concentration_mass_slope",
        "d_nfw_particle_counts_d_concentration_redshift_slope",
    ):
        assert diagnostics[key].shape == diagnostics["nfw_particle_counts"].shape
        assert np.all(np.isfinite(diagnostics[key]))
    assert "sum_d_nfw_particle_counts_d_concentration_amplitude" not in diagnostics
    assert "sum_d_nfw_particle_counts_d_concentration_mass_slope" not in diagnostics
    assert "sum_d_nfw_particle_counts_d_concentration_redshift_slope" not in diagnostics


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
        module.run_nfw_calibration_pipeline(
            catalog,
            np.array([True]),
            mass_map,
            metadata,
            particle_mass_msun_h=1.0e10,
            dense_demo=True,
            compute_map_derivatives=True,
        )


def test_nfw_map_concentration_derivative_profile_labels(capsys):
    module = _load_example_module()
    catalog, mask, mass_map, metadata = _single_pixel_pipeline_case()

    module.run_nfw_calibration_pipeline(
        catalog,
        mask,
        mass_map,
        metadata,
        particle_mass_msun_h=1.0e10,
        pipeline_mode="derivatives-profile",
        compute_map_derivatives=True,
        profile=True,
    )

    captured = capsys.readouterr()
    assert "NFW map concentration JVPs" in captured.out
    assert "NFW map concentration derivatives to numpy" in captured.out


def test_run_nfw_calibration_pipeline_paint_mode_outputs_map_without_derivatives():
    module = _load_example_module()
    catalog, mask, mass_map, metadata = _single_pixel_pipeline_case()

    diagnostics = module.run_nfw_calibration_pipeline(
        catalog,
        mask,
        mass_map,
        metadata,
        particle_mass_msun_h=1.0e10,
        pipeline_mode="paint",
        concentration_amplitude=5.71,
        truncation_width_fraction=0.05,
        chunk_size=1,
    )

    assert diagnostics["pipeline_mode"] == "paint"
    assert diagnostics["nfw_paint_mode"] == "sparse"
    assert diagnostics["nfw_map_derivatives"] == "none"
    assert diagnostics["nfw_particle_counts"].shape == (1,)
    assert "d_nfw_particle_counts_d_concentration_amplitude" not in diagnostics
    np.testing.assert_allclose(
        np.sum(diagnostics["nfw_particle_counts"]),
        diagnostics["nfw_sum_particle_counts"],
        rtol=1.0e-5,
    )


def test_run_nfw_calibration_pipeline_profile_mode_prints_timing(capsys):
    module = _load_example_module()
    catalog, mask, mass_map, metadata = _single_pixel_pipeline_case()

    with module.timed_stage(module.nfw_stage_label("profile"), True):
        diagnostics = module.run_nfw_calibration_pipeline(
            catalog,
            mask,
            mass_map,
            metadata,
            particle_mass_msun_h=1.0e10,
            pipeline_mode="profile",
            concentration_amplitude=5.71,
            truncation_width_fraction=0.05,
            chunk_size=1,
            profile=True,
        )

    captured = capsys.readouterr()
    assert diagnostics["pipeline_mode"] == "profile"
    assert module.nfw_stage_label("profile") in captured.out
    assert "NFW particle map" in captured.out
    assert "NFW particle map to numpy" in captured.out
    assert "NFW map concentration JVPs" not in captured.out


def test_run_nfw_calibration_pipeline_derivatives_profile_mode_does_both(capsys):
    module = _load_example_module()
    catalog, mask, mass_map, metadata = _single_pixel_pipeline_case()

    with module.timed_stage(module.nfw_stage_label("derivatives-profile"), True):
        diagnostics = module.run_nfw_calibration_pipeline(
            catalog,
            mask,
            mass_map,
            metadata,
            particle_mass_msun_h=1.0e10,
            pipeline_mode="derivatives-profile",
            concentration_amplitude=5.71,
            truncation_width_fraction=0.05,
            chunk_size=1,
            compute_map_derivatives=True,
            profile=True,
        )

    captured = capsys.readouterr()
    assert diagnostics["pipeline_mode"] == "derivatives-profile"
    assert diagnostics["nfw_map_derivatives"] == "concentration"
    assert module.nfw_stage_label("derivatives-profile") in captured.out
    assert "NFW map concentration JVPs" in captured.out
    assert "NFW particle map" in captured.out
    assert "NFW particle map to numpy" in captured.out


def test_nfw_stage_label_uses_pipeline_mode():
    module = _load_example_module()

    assert module.nfw_stage_label("paint") == "NFW calibration pipeline: paint"
    assert (
        module.nfw_stage_label("derivatives")
        == "NFW calibration pipeline: derivatives"
    )
    assert module.nfw_stage_label("profile") == "NFW calibration pipeline: profile"
    assert (
        module.nfw_stage_label("derivatives-profile")
        == "NFW calibration pipeline: derivatives-profile"
    )


def test_print_nfw_calibration_summary_reports_map_derivatives(capsys):
    module = _load_example_module()
    common = {
        "pipeline_mode": "paint",
        "nfw_paint_mode": "sparse",
        "nfw_selected_halo_count": 1,
        "nfw_compact_pixel_count": 1,
        "nfw_sparse_pair_count": 1,
        "nfw_dense_pair_count": 1,
        "nfw_sparse_compression_factor": 1.0,
        "nfw_sum_particle_counts": 1.25,
    }

    module.print_nfw_calibration_summary(
        {
            **common,
            "nfw_map_derivatives": "none",
        }
    )
    paint_only = capsys.readouterr().out
    assert "NFW calibration map:" in paint_only
    assert "Pipeline mode: paint" in paint_only
    assert "Map derivatives: concentration" not in paint_only

    module.print_nfw_calibration_summary(
        {
            **common,
            "pipeline_mode": "derivatives",
            "nfw_map_derivatives": "concentration",
        }
    )
    map_only = capsys.readouterr().out
    assert "NFW calibration map + derivatives:" in map_only
    assert "Pipeline mode: derivatives" in map_only
    assert "Map derivatives: concentration" in map_only
    assert "Sum d(map)/d concentration amplitude" not in map_only
    assert "Sum d(map)/d concentration mass slope" not in map_only
    assert "Sum d(map)/d concentration redshift slope" not in map_only


def test_reduce_calibration_segment_result_sums_additive_payloads():
    module = _load_example_module()
    mass_map = _mass_map(np.array([0, 1]), temperature=np.array([10.0, 20.0]))
    bounds = {
        "sheet_index": 0,
        "z_lo": 0.1,
        "z_hi": 0.2,
        "a_lo": 1.0 / 1.2,
        "a_hi": 1.0 / 1.1,
        "chi_lo_mpc_h": 100.0,
        "chi_hi_mpc_h": 200.0,
    }
    local = module.CalibrationSegmentResult(
        segment_index=0,
        mass_map_path=Path("massmap.seg000.fits"),
        output_npz=Path("painted.seg000.npz"),
        output_fits=None,
        bounds=bounds,
        inclusive_upper=False,
        mass_map=mass_map,
        halo_particle_counts=np.array([1.0, 2.0]),
        diagnostics={
            "particle_mass_msun_h": 1.0,
            "n_halos_total": 2,
            "n_halos_in_segment": 1,
            "n_halos_in_segment_and_pixels": 1,
            "sum_halo_particle_counts": 3.0,
            "sum_pinocchio_mass_map_values": 30.0,
        },
        nfw_diagnostics={
            "pipeline_mode": "derivatives",
            "particle_mass_msun_h": 1.0,
            "nfw_particle_counts": np.array([0.5, 1.5]),
            "nfw_map_derivatives": "concentration",
            "d_nfw_particle_counts_d_concentration_amplitude": np.array([0.1, 0.2]),
            "d_nfw_particle_counts_d_concentration_mass_slope": np.array([0.3, 0.4]),
            "d_nfw_particle_counts_d_concentration_redshift_slope": np.array([0.5, 0.6]),
            "nfw_paint_mode": "sparse",
            "nfw_selected_halo_count": 1,
            "nfw_compact_pixel_count": 2,
            "nfw_sparse_pair_count": 2,
            "nfw_dense_pair_count": 4,
            "nfw_sparse_compression_factor": 2.0,
            "nfw_sum_particle_counts": 2.0,
            "nfw_concentration_amplitude": 5.71,
            "nfw_concentration_mass_slope": -0.084,
            "nfw_concentration_redshift_slope": -0.47,
            "nfw_concentration_mass_pivot": 2.0e12,
            "nfw_truncation_width_fraction": 0.05,
            "stencil_query_pixels_total": 10,
            "stencil_inside_domain_total": 8,
            "stencil_kept_pairs_total": 6,
            "stencil_inside_over_query": 0.8,
            "stencil_kept_over_query": 0.6,
            "stencil_kept_over_inside": 0.75,
        },
    )

    remote_values = [
        np.array([3.0, 4.0]),
        np.array([2.0, 3.0]),
        np.array([0.4, 0.5]),
        np.array([0.6, 0.7]),
        np.array([0.8, 0.9]),
        np.array([3, 2, 2, 2, 3, 6, 20, 16, 12], dtype=np.int64),
    ]

    class PairwiseSumComm:
        def __init__(self, values):
            self.values = list(values)
            self.send_buffers = []

        def Reduce(self, send_buffer, receive_buffer, op=None, root=0):
            del op
            assert root == 0
            assert receive_buffer is not None
            other = self.values.pop(0)
            self.send_buffers.append(np.asarray(send_buffer).copy())
            receive_buffer[...] = np.asarray(send_buffer) + np.asarray(other)

    comm = PairwiseSumComm(remote_values)

    context = module.MpiContext(
        enabled=True,
        comm=comm,
        rank=0,
        size=2,
    )

    reduced = module.reduce_calibration_segment_result(local, context)

    assert reduced is not None
    np.testing.assert_allclose(reduced.halo_particle_counts, [4.0, 6.0])
    np.testing.assert_allclose(reduced.nfw_diagnostics["nfw_particle_counts"], [2.5, 4.5])
    np.testing.assert_allclose(
        reduced.nfw_diagnostics["d_nfw_particle_counts_d_concentration_amplitude"],
        [0.5, 0.7],
    )
    assert reduced.diagnostics["n_halos_total"] == 5
    assert reduced.diagnostics["n_halos_in_segment"] == 3
    assert reduced.diagnostics["n_halos_in_segment_and_pixels"] == 3
    assert reduced.diagnostics["sum_pinocchio_mass_map_values"] == 30.0
    assert reduced.diagnostics["sum_halo_particle_counts"] == 10.0
    assert reduced.nfw_diagnostics["nfw_selected_halo_count"] == 3
    assert reduced.nfw_diagnostics["nfw_sparse_pair_count"] == 5
    assert reduced.nfw_diagnostics["nfw_dense_pair_count"] == 10
    assert reduced.nfw_diagnostics["nfw_compact_pixel_count"] == 2
    assert reduced.nfw_diagnostics["nfw_sparse_compression_factor"] == 2.0
    assert reduced.nfw_diagnostics["nfw_sum_particle_counts"] == 7.0
    assert reduced.nfw_diagnostics["stencil_query_pixels_total"] == 30
    assert reduced.nfw_diagnostics["stencil_inside_domain_total"] == 24
    assert reduced.nfw_diagnostics["stencil_kept_pairs_total"] == 18
    assert reduced.nfw_diagnostics["stencil_inside_over_query"] == 0.8
    assert reduced.nfw_diagnostics["stencil_kept_over_query"] == 0.6
    assert reduced.nfw_diagnostics["stencil_kept_over_inside"] == 0.75
    assert "sum_d_nfw_particle_counts_d_concentration_amplitude" not in reduced.nfw_diagnostics
    assert "sum_d_nfw_particle_counts_d_concentration_mass_slope" not in reduced.nfw_diagnostics
    assert "sum_d_nfw_particle_counts_d_concentration_redshift_slope" not in reduced.nfw_diagnostics
    assert len(comm.send_buffers) == 6
    assert comm.send_buffers[-1].dtype == np.dtype(np.int64)
    np.testing.assert_array_equal(
        comm.send_buffers[-1],
        [2, 1, 1, 1, 2, 4, 10, 8, 6],
    )
    assert not comm.values


def test_mpi_reduce_array_preserves_float32_and_uses_receive_buffer():
    module = _load_example_module()

    class DoublingComm:
        def Reduce(self, send_buffer, receive_buffer, op=None, root=0):
            del op
            assert root == 0
            assert receive_buffer is not None
            assert receive_buffer.dtype == np.dtype(np.float32)
            receive_buffer[...] = 2 * send_buffer

    reduced = module._mpi_reduce_array(
        np.array([1.0, 2.0], dtype=np.float32),
        module.MpiContext(
            enabled=True,
            comm=DoublingComm(),
            rank=0,
            size=2,
        ),
    )

    assert reduced is not None
    assert reduced.dtype == np.dtype(np.float32)
    np.testing.assert_allclose(reduced, [2.0, 4.0])


def test_reduce_calibration_segment_result_non_root_uses_no_receive_buffers():
    module = _load_example_module()
    local = _fake_segment_result(
        module,
        segment_index=0,
        mass_map_path=Path("massmap.seg000.fits"),
        output_npz=Path("painted.seg000.npz"),
        output_fits=None,
        inclusive_upper=False,
    )
    local.halo_particle_counts = local.halo_particle_counts.astype(np.float32)
    local.nfw_diagnostics["nfw_particle_counts"] = np.asarray(
        local.nfw_diagnostics["nfw_particle_counts"],
        dtype=np.float32,
    )

    class NonRootComm:
        def __init__(self):
            self.send_dtypes = []

        def Reduce(self, send_buffer, receive_buffer, op=None, root=0):
            del op
            assert root == 0
            assert receive_buffer is None
            self.send_dtypes.append(np.asarray(send_buffer).dtype)

    comm = NonRootComm()
    reduced = module.reduce_calibration_segment_result(
        local,
        module.MpiContext(
            enabled=True,
            comm=comm,
            rank=1,
            size=2,
        ),
    )

    assert reduced is None
    assert comm.send_dtypes == [
        np.dtype(np.float32),
        np.dtype(np.float32),
        np.dtype(np.int64),
    ]


def test_real_mpi_buffer_reduction():
    module = _load_example_module()
    if module._mpi_environment_size_hint() < 2:
        pytest.skip("requires execution under mpiexec with at least two ranks")

    mpi = pytest.importorskip("mpi4py.MPI")
    comm = mpi.COMM_WORLD
    if comm.Get_size() < 2:
        pytest.skip("requires execution under mpiexec with at least two ranks")

    rank_value = comm.Get_rank() + 1
    local = _fake_segment_result(
        module,
        segment_index=0,
        mass_map_path=Path("massmap.seg000.fits"),
        output_npz=Path("painted.seg000.npz"),
        output_fits=None,
        inclusive_upper=False,
    )
    local.halo_particle_counts = np.array([rank_value], dtype=np.float64)
    local.nfw_diagnostics["nfw_particle_counts"] = np.array(
        [2 * rank_value],
        dtype=np.float64,
    )
    for key in (
        "n_halos_total",
        "n_halos_in_segment",
        "n_halos_in_segment_and_pixels",
    ):
        local.diagnostics[key] = rank_value
    for key in (
        "nfw_selected_halo_count",
        "nfw_sparse_pair_count",
        "nfw_dense_pair_count",
    ):
        local.nfw_diagnostics[key] = rank_value

    reduced = module.reduce_calibration_segment_result(
        local,
        module.MpiContext(
            enabled=True,
            comm=comm,
            rank=comm.Get_rank(),
            size=comm.Get_size(),
            sum_op=mpi.SUM,
        ),
    )

    if comm.Get_rank() != 0:
        assert reduced is None
        return

    assert reduced is not None
    expected = comm.Get_size() * (comm.Get_size() + 1) // 2
    np.testing.assert_allclose(reduced.halo_particle_counts, [expected])
    np.testing.assert_allclose(
        reduced.nfw_diagnostics["nfw_particle_counts"],
        [2 * expected],
    )
    assert reduced.diagnostics["n_halos_total"] == expected
    assert reduced.nfw_diagnostics["nfw_selected_halo_count"] == expected


def test_run_segment_workflow_all_segments_names_outputs_and_sets_inclusive_bounds(
    tmp_path,
    monkeypatch,
):
    module = _load_example_module()
    seg001 = tmp_path / "run.massmap.seg001.fits"
    seg000 = tmp_path / "run.massmap.seg000.fits"
    seg001.touch()
    seg000.touch()
    output_dir = tmp_path / "painted"
    args = _workflow_args(
        mass_map=None,
        sheet_index=None,
        output=None,
        mass_map_glob=str(tmp_path / "*.fits"),
        output_dir=output_dir,
        mode="derivatives",
    )
    calls = []

    def fake_segment_runner(**kwargs):
        calls.append(kwargs)
        return {
            "segment_index": kwargs["segment_index"],
            "mass_map_path": str(kwargs["mass_map_path"]),
            "output_npz": str(kwargs["output_npz"]),
            "output_fits": str(kwargs["output_fits"]),
            "inclusive_upper": kwargs["inclusive_upper"],
        }

    manifest_calls = []
    monkeypatch.setattr(module, "run_calibration_for_segment", fake_segment_runner)
    monkeypatch.setattr(
        module,
        "write_manifest",
        lambda path, rows: manifest_calls.append((path, rows)),
    )

    rows = module.run_segment_workflow(
        args,
        workflow="all",
        catalog=_catalog(),
        sheets=_sheets(),
        metadata=SimpleNamespace(particle_mass_msun_h=1.0, cosmology=Cosmology()),
        particle_mass=1.0,
        profile=False,
        compute_map_derivatives=True,
    )

    assert [call["segment_index"] for call in calls] == [0, 1]
    assert [call["inclusive_upper"] for call in calls] == [False, True]
    assert calls[0]["output_npz"] == output_dir / "painted_nfw.seg000.npz"
    assert calls[0]["output_fits"] == output_dir / "painted_nfw.seg000.fits"
    assert calls[1]["output_npz"] == output_dir / "painted_nfw.seg001.npz"
    assert calls[1]["output_fits"] == output_dir / "painted_nfw.seg001.fits"
    assert manifest_calls[0][0] == output_dir / "painted_nfw_manifest.csv"
    assert rows == manifest_calls[0][1]


def test_run_segment_workflow_single_segment_uses_last_segment_flag(monkeypatch):
    module = _load_example_module()
    args = _workflow_args(
        output_fits=Path("painted.seg000.fits"),
        last_segment_inclusive=True,
    )
    calls = []

    def fake_segment_runner(**kwargs):
        calls.append(kwargs)
        return {"segment_index": kwargs["segment_index"]}

    monkeypatch.setattr(module, "run_calibration_for_segment", fake_segment_runner)

    rows = module.run_segment_workflow(
        args,
        workflow="single",
        catalog=_catalog(),
        sheets=_sheets(),
        metadata=SimpleNamespace(particle_mass_msun_h=1.0, cosmology=Cosmology()),
        particle_mass=1.0,
        profile=False,
        compute_map_derivatives=False,
    )

    assert rows == [{"segment_index": 0}]
    assert calls[0]["inclusive_upper"] is True
    assert calls[0]["output_npz"] == Path("painted.seg000.npz")
    assert calls[0]["output_fits"] == Path("painted.seg000.fits")


def test_compute_calibration_for_segment_reuses_one_mass_map_pixel_index(
    tmp_path,
    monkeypatch,
):
    module = _load_example_module()
    mass_map = _mass_map(np.array([0]), temperature=np.array([100.0]), nside=1)
    catalog = _catalog(
        unit_vector=np.array([[1.0, 0.0, 0.0]]),
        mass=np.array([10.0]),
        redshift=np.array([0.3]),
        chi=np.array([1000.0]),
    )
    seen = {"builds": 0, "halo_index_id": None, "nfw_index_id": None}
    original_from_mass_map = module.MassMapPixelIndex.from_mass_map

    def fake_from_mass_map(mass_map_arg, *, max_dense_bytes=module._PIXEL_INDEX_DENSE_MAX_BYTES):
        seen["builds"] += 1
        return original_from_mass_map(mass_map_arg, max_dense_bytes=max_dense_bytes)

    def fake_halo_rows(catalog_arg, mask, mass_map_arg, pixel_index=None):
        del catalog_arg, mask, mass_map_arg
        seen["halo_index_id"] = id(pixel_index)
        return np.array([0], dtype=np.int64), np.array([True])

    def fake_nfw_pipeline(*args, **kwargs):
        del args
        seen["nfw_index_id"] = id(kwargs["pixel_index"])
        return {
            "pipeline_mode": "paint",
            "particle_mass_msun_h": 1.0,
            "nfw_particle_counts": np.array([0.5]),
            "nfw_map_derivatives": "none",
            "nfw_paint_mode": "sparse",
            "nfw_selected_halo_count": 1,
            "nfw_compact_pixel_count": 1,
            "nfw_sparse_pair_count": 1,
            "nfw_dense_pair_count": 1,
            "nfw_sparse_compression_factor": 1.0,
            "nfw_sum_particle_counts": 0.5,
            "nfw_concentration_amplitude": 5.71,
            "nfw_concentration_mass_slope": -0.084,
            "nfw_concentration_redshift_slope": -0.47,
            "nfw_concentration_mass_pivot": 2.0e12,
            "nfw_truncation_width_fraction": 0.05,
        }

    monkeypatch.setattr(
        module.MassMapPixelIndex,
        "from_mass_map",
        staticmethod(fake_from_mass_map),
    )
    monkeypatch.setattr(module, "read_pinocchio_mass_map_fits", lambda path: mass_map)
    monkeypatch.setattr(module, "halo_rows_in_mass_map", fake_halo_rows)
    monkeypatch.setattr(module, "run_nfw_calibration_pipeline", fake_nfw_pipeline)

    result = module.compute_calibration_for_segment(
        segment_index=0,
        mass_map_path=tmp_path / "massmap.seg000.fits",
        output_npz=tmp_path / "painted.seg000.npz",
        output_fits=None,
        catalog=catalog,
        sheets=_sheets(),
        metadata=SimpleNamespace(particle_mass_msun_h=1.0, cosmology=Cosmology()),
        particle_mass=1.0,
        args=_workflow_args(),
        profile=False,
        compute_map_derivatives=False,
        inclusive_upper=False,
        verbose=False,
    )

    assert result.segment_index == 0
    assert seen["builds"] == 1
    assert seen["halo_index_id"] == seen["nfw_index_id"]
    assert seen["halo_index_id"] != id(None)


def test_run_segment_workflow_all_segments_uses_segment_workers(tmp_path, monkeypatch):
    module = _load_example_module()
    seg001 = tmp_path / "run.massmap.seg001.fits"
    seg000 = tmp_path / "run.massmap.seg000.fits"
    seg001.touch()
    seg000.touch()
    output_dir = tmp_path / "painted"
    args = _workflow_args(
        mass_map=None,
        sheet_index=None,
        output=None,
        mass_map_glob=str(tmp_path / "*.fits"),
        output_dir=output_dir,
        segment_workers=2,
    )
    compute_calls = []
    write_calls = []

    def fake_compute(**kwargs):
        compute_calls.append(kwargs)
        return module.CalibrationSegmentResult(
            segment_index=kwargs["segment_index"],
            mass_map_path=kwargs["mass_map_path"],
            output_npz=kwargs["output_npz"],
            output_fits=kwargs["output_fits"],
            bounds={
                "sheet_index": kwargs["segment_index"],
                "z_lo": 0.1,
                "z_hi": 0.2,
                "a_lo": 1.0 / 1.2,
                "a_hi": 1.0 / 1.1,
                "chi_lo_mpc_h": 100.0,
                "chi_hi_mpc_h": 200.0,
            },
            inclusive_upper=kwargs["inclusive_upper"],
            mass_map=_mass_map(np.array([0]), temperature=np.array([1.0])),
            halo_particle_counts=np.array([1.0]),
            diagnostics={
                "particle_mass_msun_h": 1.0,
                "n_halos_total": 1,
                "n_halos_in_segment": 1,
                "n_halos_in_segment_and_pixels": 1,
                "sum_halo_particle_counts": 1.0,
                "sum_pinocchio_mass_map_values": 1.0,
            },
            nfw_diagnostics={
                "pipeline_mode": "paint",
                "particle_mass_msun_h": 1.0,
                "nfw_particle_counts": np.array([1.0]),
                "nfw_map_derivatives": "none",
                "nfw_paint_mode": "sparse",
                "nfw_selected_halo_count": 1,
                "nfw_compact_pixel_count": 1,
                "nfw_sparse_pair_count": 1,
                "nfw_dense_pair_count": 1,
                "nfw_sparse_compression_factor": 1.0,
                "nfw_sum_particle_counts": 1.0,
                "nfw_concentration_amplitude": 5.71,
                "nfw_concentration_mass_slope": -0.084,
                "nfw_concentration_redshift_slope": -0.47,
                "nfw_concentration_mass_pivot": 2.0e12,
                "nfw_truncation_width_fraction": 0.05,
            },
        )

    def fake_write(result, metadata, *, profile, verbose):
        del metadata, profile, verbose
        write_calls.append(result.segment_index)
        return {"segment_index": result.segment_index}

    manifest_calls = []
    monkeypatch.setattr(module, "compute_calibration_for_segment", fake_compute)
    monkeypatch.setattr(module, "write_calibration_segment_outputs", fake_write)
    monkeypatch.setattr(
        module,
        "write_manifest",
        lambda path, rows: manifest_calls.append((path, rows)),
    )

    rows = module.run_segment_workflow(
        args,
        workflow="all",
        catalog=_catalog(),
        sheets=_sheets(),
        metadata=SimpleNamespace(particle_mass_msun_h=1.0, cosmology=Cosmology()),
        particle_mass=1.0,
        profile=False,
        compute_map_derivatives=False,
    )

    assert sorted(call["segment_index"] for call in compute_calls) == [0, 1]
    assert all(call["verbose"] is False for call in compute_calls)
    assert write_calls == [0, 1]
    assert rows == [{"segment_index": 0}, {"segment_index": 1}]
    assert manifest_calls[0][1] == rows


def test_run_segment_workflow_mpi_reduce_mode_reduces_before_root_write(
    tmp_path,
    monkeypatch,
):
    module = _load_example_module()
    seg001 = tmp_path / "run.massmap.seg001.fits"
    seg000 = tmp_path / "run.massmap.seg000.fits"
    seg001.touch()
    seg000.touch()
    output_dir = tmp_path / "painted"
    args = _workflow_args(
        mass_map=None,
        sheet_index=None,
        output=None,
        mass_map_glob=str(tmp_path / "*.fits"),
        output_dir=output_dir,
        mpi_plc_parts=True,
    )
    reduce_calls = []
    write_calls = []
    manifest_calls = []
    events = []

    def fake_compute(**kwargs):
        events.append(("compute", kwargs["segment_index"]))
        return _fake_segment_result(
            module,
            segment_index=kwargs["segment_index"],
            mass_map_path=kwargs["mass_map_path"],
            output_npz=kwargs["output_npz"],
            output_fits=kwargs["output_fits"],
            inclusive_upper=kwargs["inclusive_upper"],
        )

    def fake_reduce(result, mpi_context):
        events.append(("reduce", result.segment_index))
        reduce_calls.append((result.segment_index, mpi_context.rank))
        return result

    def fake_write(result, metadata, *, profile, verbose):
        del metadata, profile, verbose
        events.append(("write", result.segment_index))
        write_calls.append((result.output_npz, result.output_fits))
        return {"segment_index": result.segment_index}

    monkeypatch.setattr(module, "compute_calibration_for_segment", fake_compute)
    monkeypatch.setattr(module, "reduce_calibration_segment_result", fake_reduce)
    monkeypatch.setattr(module, "write_calibration_segment_outputs", fake_write)
    monkeypatch.setattr(
        module,
        "write_manifest",
        lambda path, rows: manifest_calls.append((path, rows)),
    )

    rows = module.run_segment_workflow(
        args,
        workflow="all",
        catalog=_catalog(),
        sheets=_sheets(),
        metadata=SimpleNamespace(particle_mass_msun_h=1.0, cosmology=Cosmology()),
        particle_mass=1.0,
        profile=False,
        compute_map_derivatives=False,
        mpi_context=module.MpiContext(enabled=True, rank=0, size=2),
    )

    assert reduce_calls == [(0, 0), (1, 0)]
    assert events == [
        ("compute", 0),
        ("reduce", 0),
        ("write", 0),
        ("compute", 1),
        ("reduce", 1),
        ("write", 1),
    ]
    assert write_calls == [
        (output_dir / "painted_nfw.seg000.npz", output_dir / "painted_nfw.seg000.fits"),
        (output_dir / "painted_nfw.seg001.npz", output_dir / "painted_nfw.seg001.fits"),
    ]
    assert manifest_calls[0][0] == output_dir / "painted_nfw_manifest.csv"
    assert rows == [{"segment_index": 0}, {"segment_index": 1}]


def test_run_segment_workflow_mpi_reduce_mode_uses_bounded_segment_pipeline(
    tmp_path,
    monkeypatch,
):
    module = _load_example_module()
    for segment_index in range(3):
        (tmp_path / f"run.massmap.seg{segment_index:03d}.fits").touch()
    output_dir = tmp_path / "painted"
    args = _workflow_args(
        mass_map=None,
        sheet_index=None,
        output=None,
        mass_map_glob=str(tmp_path / "*.fits"),
        output_dir=output_dir,
        mpi_plc_parts=True,
        segment_workers=2,
    )
    manifest_calls = []
    events = []

    class FakeFuture:
        def __init__(self, result):
            self._result = result

        def result(self):
            return self._result

    class FakeExecutor:
        def __init__(self, max_workers):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def submit(self, fn, spec):
            (segment_index, _mass_map_path), _output_spec, _inclusive_upper = spec
            events.append(("submit", segment_index, self.max_workers))
            return FakeFuture(fn(spec))

    def fake_compute(**kwargs):
        events.append(("compute", kwargs["segment_index"]))
        return _fake_segment_result(
            module,
            segment_index=kwargs["segment_index"],
            mass_map_path=kwargs["mass_map_path"],
            output_npz=kwargs["output_npz"],
            output_fits=kwargs["output_fits"],
            inclusive_upper=kwargs["inclusive_upper"],
        )

    def fake_reduce(result, mpi_context):
        events.append(("reduce", result.segment_index, mpi_context.rank))
        return result

    def fake_write(result, metadata, *, profile, verbose):
        del metadata, profile, verbose
        events.append(("write", result.segment_index))
        return {"segment_index": result.segment_index}

    monkeypatch.setattr(module, "ThreadPoolExecutor", FakeExecutor)
    monkeypatch.setattr(module, "compute_calibration_for_segment", fake_compute)
    monkeypatch.setattr(module, "reduce_calibration_segment_result", fake_reduce)
    monkeypatch.setattr(module, "write_calibration_segment_outputs", fake_write)
    monkeypatch.setattr(
        module,
        "write_manifest",
        lambda path, rows: manifest_calls.append((path, rows)),
    )

    rows = module.run_segment_workflow(
        args,
        workflow="all",
        catalog=_catalog(),
        sheets=_sheets(),
        metadata=SimpleNamespace(particle_mass_msun_h=1.0, cosmology=Cosmology()),
        particle_mass=1.0,
        profile=False,
        compute_map_derivatives=False,
        mpi_context=module.MpiContext(enabled=True, rank=0, size=2),
    )

    assert events == [
        ("submit", 0, 2),
        ("compute", 0),
        ("submit", 1, 2),
        ("compute", 1),
        ("reduce", 0, 0),
        ("write", 0),
        ("submit", 2, 2),
        ("compute", 2),
        ("reduce", 1, 0),
        ("write", 1),
        ("reduce", 2, 0),
        ("write", 2),
    ]
    assert rows == [{"segment_index": 0}, {"segment_index": 1}, {"segment_index": 2}]
    assert manifest_calls[0][0] == output_dir / "painted_nfw_manifest.csv"


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


def test_local_sparse_stencil_query_modes_and_validation():
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
    mass_map = _mass_map(pixels, temperature=np.array([100.0]), nside=1)

    inclusive = module.build_lightcone_sparse_stencil_for_mass_map_local(
        mass_map,
        catalog,
        rmax_mpc_h=np.array([1.0]),
        query_mode="inclusive",
    )
    center = module.build_lightcone_sparse_stencil_for_mass_map_local(
        mass_map,
        catalog,
        rmax_mpc_h=np.array([1.0]),
        query_mode="center",
    )

    assert inclusive.n_pix == 1
    assert center.n_pix == 1
    assert np.asarray(inclusive.pix_id).shape[0] >= 1
    assert np.asarray(center.pix_id).shape[0] >= 1
    with pytest.raises(ValueError, match="query_mode"):
        module.build_lightcone_sparse_stencil_for_mass_map_local(
            mass_map,
            catalog,
            rmax_mpc_h=np.array([1.0]),
            query_mode="bad",
        )


@pytest.mark.parametrize("query_mode", ["inclusive", "center"])
def test_local_sparse_stencil_diagnostics_counters(query_mode):
    hp = pytest.importorskip("healpy")
    module = _load_example_module()

    pixels = np.array([0, 5], dtype=np.int64)
    unit_vector = np.stack(hp.pix2vec(1, np.array([0, 5], dtype=np.int64)), axis=-1)
    catalog = _catalog(
        unit_vector=unit_vector,
        mass=np.array([1.0e13, 2.0e13]),
        redshift=np.array([0.2, 0.25]),
        chi=np.array([1000.0, 1100.0]),
    )
    mass_map = _mass_map(pixels, temperature=np.array([100.0, 200.0]), nside=1)

    stencil, diag = module.build_lightcone_sparse_stencil_for_mass_map_local(
        mass_map,
        catalog,
        rmax_mpc_h=np.array([1.0, 1.0]),
        query_mode=query_mode,
        collect_diagnostics=True,
    )

    assert diag.n_halos == len(catalog.mass)
    assert diag.n_query_pixels_total >= diag.n_inside_domain_total
    assert diag.n_inside_domain_total >= diag.n_kept_pairs_total
    assert diag.n_kept_pairs_total == len(np.asarray(stencil.pix_id))
    assert diag.query_mode == query_mode
    assert diag.elapsed_seconds >= 0.0


def test_center_query_mode_has_no_more_query_pixels_than_inclusive():
    hp = pytest.importorskip("healpy")
    module = _load_example_module()

    pixels = np.array([0, 5], dtype=np.int64)
    unit_vector = np.stack(hp.pix2vec(1, pixels), axis=-1)
    catalog = _catalog(
        unit_vector=unit_vector,
        mass=np.array([1.0e13, 2.0e13]),
        redshift=np.array([0.2, 0.25]),
        chi=np.array([1000.0, 1100.0]),
    )
    mass_map = _mass_map(pixels, temperature=np.array([100.0, 200.0]), nside=1)

    _, inclusive_diag = module.build_lightcone_sparse_stencil_for_mass_map_local(
        mass_map,
        catalog,
        rmax_mpc_h=np.array([1.0, 1.0]),
        query_mode="inclusive",
        collect_diagnostics=True,
    )
    _, center_diag = module.build_lightcone_sparse_stencil_for_mass_map_local(
        mass_map,
        catalog,
        rmax_mpc_h=np.array([1.0, 1.0]),
        query_mode="center",
        collect_diagnostics=True,
    )

    assert center_diag.n_query_pixels_total <= inclusive_diag.n_query_pixels_total


def test_inclusive_and_center_maps_are_finite_and_match_when_pairs_match():
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
    mass_map = _mass_map(pixels, temperature=np.array([100.0]), nside=1)
    metadata = SimpleNamespace(cosmology=Cosmology())
    concentration_params = module.ConcentrationParams()
    profile_params = module.NFWProfileParams()
    rmax = np.array([1.0])
    pixel_area_sr = module.healpix_pixel_area_sr(mass_map.nside)

    maps = {}
    pair_sets = {}
    for query_mode in ("inclusive", "center"):
        stencil = module.build_lightcone_sparse_stencil_for_mass_map_local(
            mass_map,
            catalog,
            rmax_mpc_h=rmax,
            query_mode=query_mode,
        )
        counts = module.paint_lightcone_particle_count_map_sparse(
            stencil,
            catalog,
            particle_mass_msun_h=1.0e10,
            pixel_area_sr=pixel_area_sr,
            cosmology=metadata.cosmology,
            concentration_params=concentration_params,
            profile_params=profile_params,
        )
        maps[query_mode] = np.asarray(counts)
        pair_sets[query_mode] = {
            (
                int(pixel),
                int(halo),
                float(r_perp),
            )
            for pixel, halo, r_perp in zip(
                np.asarray(stencil.pix_id),
                np.asarray(stencil.halo_id),
                np.asarray(stencil.r_perp),
                strict=True,
            )
        }

    assert maps["inclusive"].shape == maps["center"].shape
    assert np.all(np.isfinite(maps["inclusive"]))
    assert np.all(np.isfinite(maps["center"]))
    if pair_sets["inclusive"] == pair_sets["center"]:
        np.testing.assert_allclose(maps["inclusive"], maps["center"])


def test_stencil_query_mode_comparison_helper_returns_audit_keys():
    module = _load_example_module()
    catalog, mask, mass_map, metadata = _single_pixel_pipeline_case()
    selected_catalog = module.selected_lightcone_catalog(catalog, mask)
    concentration_params = module.ConcentrationParams()
    profile_params = module.NFWProfileParams()
    rmax = module.nfw_stencil_rmax_mpc_h(
        selected_catalog,
        metadata,
        concentration_params,
        profile_params,
        taper_radius_factor=10.0,
    )

    counts, stencil, comparison = module.compare_stencil_query_modes(
        mass_map,
        selected_catalog,
        rmax,
        metadata,
        particle_mass_msun_h=1.0e10,
        pixel_area_sr=module.healpix_pixel_area_sr(mass_map.nside),
        concentration_params=concentration_params,
        profile_params=profile_params,
    )

    assert counts.shape == mass_map.temperature.shape
    assert stencil.n_pix == len(mass_map.pixel)
    for key in (
        "inclusive_stencil_seconds",
        "center_stencil_seconds",
        "inclusive_query_pixels_total",
        "center_query_pixels_total",
        "inclusive_kept_pairs_total",
        "center_kept_pairs_total",
        "max_abs_map_difference",
        "sum_abs_map_difference",
        "relative_sum_difference",
        "differing_pixels",
        "maps_allclose",
    ):
        assert key in comparison
    assert comparison["center_query_pixels_total"] <= comparison["inclusive_query_pixels_total"]
    assert isinstance(comparison["maps_allclose"], bool)


def test_run_nfw_calibration_pipeline_dense_debug_path_paints_map():
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

    diagnostics = module.run_nfw_calibration_pipeline(
        catalog,
        np.array([True]),
        mass_map,
        metadata,
        particle_mass_msun_h=1.0e10,
        pipeline_mode="paint",
        concentration_amplitude=5.71,
        truncation_width_fraction=0.05,
        chunk_size=1,
        dense_demo=True,
    )

    assert diagnostics["pipeline_mode"] == "paint"
    assert diagnostics["nfw_paint_mode"] == "dense"
    assert diagnostics["nfw_map_derivatives"] == "none"
    assert diagnostics["nfw_particle_counts"].shape == (1,)
    assert np.isfinite(diagnostics["nfw_sum_particle_counts"])
    np.testing.assert_allclose(
        np.sum(diagnostics["nfw_particle_counts"]),
        diagnostics["nfw_sum_particle_counts"],
        rtol=1.0e-5,
    )


def test_run_nfw_calibration_pipeline_sparse_matches_dense_when_stencil_contains_all_pairs():
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

    sparse = module.run_nfw_calibration_pipeline(
        catalog,
        np.array([True, True]),
        mass_map,
        metadata,
        particle_mass_msun_h=1.0e10,
        pipeline_mode="paint",
        concentration_amplitude=5.71,
        truncation_width_fraction=0.05,
        chunk_size=1,
        taper_radius_factor=1.0e6,
    )
    dense = module.run_nfw_calibration_pipeline(
        catalog,
        np.array([True, True]),
        mass_map,
        metadata,
        particle_mass_msun_h=1.0e10,
        pipeline_mode="paint",
        concentration_amplitude=5.71,
        truncation_width_fraction=0.05,
        chunk_size=1,
        taper_radius_factor=1.0e6,
        dense_demo=True,
    )

    assert sparse["nfw_paint_mode"] == "sparse"
    assert dense["nfw_paint_mode"] == "dense"
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


def test_run_nfw_calibration_pipeline_sparse_uses_fewer_pairs_for_local_stencil():
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

    diagnostics = module.run_nfw_calibration_pipeline(
        catalog,
        np.array([True]),
        mass_map,
        metadata,
        particle_mass_msun_h=1.0e10,
        pipeline_mode="paint",
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


def test_write_nfw_painted_fits_preserves_compact_pixel_table(tmp_path):
    fits = pytest.importorskip("astropy.io.fits")
    module = _load_example_module()
    mass_map = _mass_map(np.array([5, 0]), temperature=np.array([100.0, 200.0]))
    bounds = {
        "sheet_index": 1,
        "z_lo": 0.1,
        "z_hi": 0.2,
        "a_lo": 1.0 / 1.2,
        "a_hi": 1.0 / 1.1,
        "chi_lo_mpc_h": 100.0,
        "chi_hi_mpc_h": 200.0,
    }
    nfw_diagnostics = {
        "particle_mass_msun_h": 5.0,
        "nfw_concentration_amplitude": 5.71,
        "nfw_concentration_mass_slope": -0.084,
        "nfw_concentration_redshift_slope": -0.47,
        "nfw_concentration_mass_pivot": 2.0e12,
        "nfw_truncation_width_fraction": 0.05,
    }
    path = tmp_path / "painted_nfw.seg001.fits"

    module.write_nfw_painted_fits(
        path,
        np.array([0.75, 0.5]),
        mass_map,
        bounds,
        nfw_diagnostics,
    )

    with fits.open(path) as hdul:
        table = hdul["HEALPIX"]
        np.testing.assert_array_equal(table.data["PIXEL"], [5, 0])
        np.testing.assert_allclose(table.data["TEMPERATURE"], [0.75, 0.5])
        assert table.header["NSIDE"] == 1
        assert table.header["ORDERING"] == "RING"
        assert table.header["MAPTYPE"] == "NFW_PARTICLE_COUNT"
        assert table.header["SHEETIDX"] == 1
        assert np.isclose(table.header["CONCAMP"], 5.71)


def test_write_nfw_painted_fits_rejects_shape_mismatch(tmp_path):
    module = _load_example_module()
    pytest.importorskip("astropy.io.fits")
    mass_map = _mass_map(np.array([5, 0]), temperature=np.array([100.0, 200.0]))
    bounds = {
        "sheet_index": 1,
        "z_lo": 0.1,
        "z_hi": 0.2,
        "a_lo": 1.0 / 1.2,
        "a_hi": 1.0 / 1.1,
        "chi_lo_mpc_h": 100.0,
        "chi_hi_mpc_h": 200.0,
    }
    nfw_diagnostics = {
        "particle_mass_msun_h": 5.0,
        "nfw_concentration_amplitude": 5.71,
        "nfw_concentration_mass_slope": -0.084,
        "nfw_concentration_redshift_slope": -0.47,
        "nfw_concentration_mass_pivot": 2.0e12,
        "nfw_truncation_width_fraction": 0.05,
    }

    with pytest.raises(RuntimeError, match="NFW map shape"):
        module.write_nfw_painted_fits(
            tmp_path / "painted_nfw.seg001.fits",
            np.array([0.75]),
            mass_map,
            bounds,
            nfw_diagnostics,
        )


def test_write_manifest_omits_derivative_sum_columns(tmp_path):
    module = _load_example_module()
    path = tmp_path / "painted_nfw_manifest.csv"

    module.write_manifest(
        path,
        [
            {
                "segment_index": 0,
                "mass_map_path": "massmap.seg000.fits",
                "output_npz": "painted_nfw.seg000.npz",
                "output_fits": "painted_nfw.seg000.fits",
                "z_lo": 0.1,
                "z_hi": 0.2,
                "chi_lo_mpc_h": 100.0,
                "chi_hi_mpc_h": 200.0,
                "inclusive_upper": False,
                "n_halos_in_segment": 2,
                "n_halos_in_segment_and_pixels": 1,
                "nfw_selected_halo_count": 2,
                "nfw_compact_pixel_count": 12,
                "nfw_sparse_pair_count": 5,
                "nfw_sum_particle_counts": 1.25,
                "nfw_map_derivatives": "concentration",
                "sum_d_nfw_particle_counts_d_concentration_amplitude": 0.3,
                "sum_d_nfw_particle_counts_d_concentration_mass_slope": 0.7,
                "sum_d_nfw_particle_counts_d_concentration_redshift_slope": 1.1,
            }
        ],
    )

    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    assert rows[0]["segment_index"] == "0"
    assert rows[0]["output_npz"] == "painted_nfw.seg000.npz"
    assert rows[0]["nfw_map_derivatives"] == "concentration"
    assert "sum_d_nfw_particle_counts_d_concentration_amplitude" not in rows[0]
    assert "sum_d_nfw_particle_counts_d_concentration_mass_slope" not in rows[0]
    assert "sum_d_nfw_particle_counts_d_concentration_redshift_slope" not in rows[0]
