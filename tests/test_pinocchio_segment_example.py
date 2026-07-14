from __future__ import annotations

import csv
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZIP_DEFLATED, ZipFile

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from geppetto import AngularAssignmentParams, paint_lightcone_particle_count_map_sparse
from geppetto.catalog import AdaptiveLightconeStencil, LightconeHaloCatalog
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
        "params": Path("params.txt"),
        "sheets": Path("sheets.out"),
        "plc_catalog": Path("plc.out"),
        "mass_map": Path("pinocchio.example.massmap.seg000.fits"),
        "sheet_index": 0,
        "output": Path("painted.seg000.npz"),
        "mass_map_glob": None,
        "output_dir": None,
        "catalog_format": "auto",
        "redshift_mode": "true",
        "bounds": "z",
        "hubble_table": None,
        "light_plc": False,
        "last_segment_inclusive": False,
        "mode": "paint",
        "concentration_amplitude": 5.71,
        "concentration_mass_slope": -0.084,
        "concentration_redshift_slope": -0.47,
        "concentration_mass_pivot": 2.0e12,
        "nfw_overdensity": 200.0,
        "nfw_virial_overdensity": False,
        "nfw_overdensity_by_segment": None,
        "nfw_reference_density": "critical",
        "theta_resolution_rad": None,
        "n_resolution": 4,
        "nfw_sample_chunk_size": 65536,
        "jax_precision": "float64",
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
    inclusive_upper: bool,
):
    return module.CalibrationSegmentResult(
        segment_index=segment_index,
        mass_map_path=mass_map_path,
        output_npz=output_npz,
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
        nfw_diagnostics={
            "pipeline_mode": "paint",
            "particle_mass_msun_h": 1.0,
            "nfw_particle_counts": np.array([1.0]),
            "nfw_map_derivatives": "none",
            "nfw_paint_mode": "adaptive_global_support",
            "nfw_selected_halo_count": 1,
            "nfw_selected_halo_mass_msun_h": 1.0,
            "nfw_expected_global_particle_count": 1.0,
            "nfw_assigned_global_particle_count": 1.0,
            "nfw_assigned_to_expected_ratio": 1.0,
            "nfw_retained_compact_particle_count": 1.0,
            "nfw_outside_compact_particle_count": 0.0,
            "nfw_unresolved_ngp_count": 1,
            "nfw_native_resolved_count": 0,
            "nfw_supersampled_count": 0,
            "nfw_zero_sample_ngp_fallback_count": 0,
            "nfw_invalid_normalizations": 0,
            "nfw_compact_pixel_count": 1,
            "nfw_global_profile_sample_count": 0,
            "nfw_retained_profile_sample_count": 0,
            "nfw_max_requested_supersampling_level": 0,
            "nfw_max_used_supersampling_level": 0,
            "nfw_theta_resolution_rad": 0.1,
            "nfw_theta_resolution_source": "automatic",
            "nfw_n_resolution": 4,
            "nfw_theta_map_rad": 0.2,
            "nfw_sample_chunk_size": 65536,
            "nfw_sum_particle_counts": 1.0,
            "nfw_concentration_amplitude": 5.71,
            "nfw_concentration_mass_slope": -0.084,
            "nfw_concentration_redshift_slope": -0.47,
            "nfw_concentration_mass_pivot": 2.0e12,
            "nfw_profile_support": "hard_3d_r_delta_los_projection",
            "nfw_mass_definition": "200c",
            "nfw_overdensity_mode": "constant",
            "nfw_overdensity": 200.0,
            "nfw_reference_density": "critical",
            "nfw_overdensity_file": "",
            "nfw_mass_conversion": "none_catalog_mass_interpreted_as_profile_mass",
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
    assert "--output-fits" not in result.stdout
    assert "--mode" in result.stdout
    assert "--concentration-amplitude" in result.stdout
    assert "--concentration-mass-slope" in result.stdout
    assert "--concentration-redshift-slope" in result.stdout
    assert "--concentration-mass-pivot" in result.stdout
    assert "--nfw-overdensity" in result.stdout
    assert "--nfw-virial-overdensity" in result.stdout
    assert "--nfw-overdensity-by-segment" in result.stdout
    assert "--nfw-reference-density" in result.stdout
    assert "--theta-resolution-rad" in result.stdout
    assert "--n-resolution" in result.stdout
    assert "In single-segment mode, use an inclusive upper segment bound." in normalized_help
    assert "physical final sheet" in normalized_help
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


@pytest.mark.parametrize(
    ("removed_args", "removed_option"),
    [
        (("--mpi-output-mode", "reduce"), "--mpi-output-mode"),
        (("--output-fits", "painted.fits"), "--output-fits"),
        (("--stencil-query-mode", "inclusive"), "--stencil-query-mode"),
        (("--stencil-compare-query-modes",), "--stencil-compare-query-modes"),
        (("--truncation-width-fraction", "0.05"), "--truncation-width-fraction"),
        (("--nfw-dense-demo",), "--nfw-dense-demo"),
    ],
)
def test_example_script_rejects_removed_options(removed_args, removed_option):
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
            "--nfw-overdensity",
            "200",
            "--nfw-reference-density",
            "critical",
            *removed_args,
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert f"unrecognized arguments: {removed_option}" in result.stderr


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


def test_validate_mass_map_segment_batch_accepts_partial_contiguous_ranges():
    module = _load_example_module()

    module.validate_mass_map_segment_batch(
        [(0, Path("massmap.seg000.fits"))],
        _sheets(),
    )
    module.validate_mass_map_segment_batch(
        [(0, Path("massmap.seg000.fits")), (1, Path("massmap.seg001.fits"))],
        _sheets(),
    )


def test_validate_mass_map_segment_batch_rejects_gaps_and_invalid_indices():
    module = _load_example_module()

    with pytest.raises(ValueError, match="contiguous batch"):
        module.validate_mass_map_segment_batch(
            [(0, Path("massmap.seg000.fits")), (2, Path("massmap.seg002.fits"))],
            _sheets(),
        )
    with pytest.raises(ValueError, match="outside the sheet table"):
        module.validate_mass_map_segment_batch(
            [(2, Path("massmap.seg002.fits"))],
            _sheets(),
        )


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

    with pytest.raises(ValueError, match="only valid in single-segment mode"):
        module.validate_segment_workflow_args(
            _workflow_args(
                mass_map=None,
                sheet_index=None,
                output=None,
                mass_map_glob=str(tmp_path / "*.fits"),
                output_dir=tmp_path / "painted",
                last_segment_inclusive=True,
            )
        )


def test_load_halo_mass_definition_supports_constant_and_bryan_norman():
    module = _load_example_module()

    constant = module.load_halo_mass_definition(_workflow_args(), _sheets())
    resolved_constant = constant.resolve(1)
    assert resolved_constant.label == "200c"
    assert resolved_constant.profile_mode == "constant"
    assert resolved_constant.profile_overdensity == 200.0

    virial = module.load_halo_mass_definition(
        _workflow_args(
            nfw_overdensity=None,
            nfw_virial_overdensity=True,
            nfw_reference_density="mean",
        ),
        _sheets(),
    )
    resolved_virial = virial.resolve(0)
    assert resolved_virial.label == "virial_bn98m"
    assert resolved_virial.profile_mode == "bryan_norman"
    assert resolved_virial.reference_density == "mean"


def test_load_halo_mass_definition_reads_strict_per_segment_array(tmp_path):
    module = _load_example_module()
    source = tmp_path / "overdensity.npy"
    np.save(source, np.array([200.0, 500.0]))

    definition = module.load_halo_mass_definition(
        _workflow_args(
            nfw_overdensity=None,
            nfw_overdensity_by_segment=source,
            nfw_reference_density="mean",
        ),
        _sheets(),
    )

    assert definition.resolve(0).label == "200m"
    assert definition.resolve(1).label == "500m"
    assert definition.resolve(1).source == source
    assert definition.per_segment_overdensity.flags.writeable is False

    np.save(source, np.array([200.0]))
    with pytest.raises(ValueError, match="length must match"):
        module.load_halo_mass_definition(
            _workflow_args(
                nfw_overdensity=None,
                nfw_overdensity_by_segment=source,
            ),
            _sheets(),
        )

    np.save(source, np.array([200.0, -1.0]))
    with pytest.raises(ValueError, match="finite and positive"):
        module.load_halo_mass_definition(
            _workflow_args(
                nfw_overdensity=None,
                nfw_overdensity_by_segment=source,
            ),
            _sheets(),
        )


def test_validate_mpi_workflow_args_rejects_missing_flag(tmp_path):
    module = _load_example_module()
    base = tmp_path / "pinocchio.demo.plc.out"
    args = _workflow_args(plc_catalog=base, segment_workers=1, mpi_plc_parts=False)

    with pytest.raises(ValueError, match="requires --mpi-plc-parts"):
        module.validate_mpi_workflow_args(
            args,
            workflow="single",
            mpi_context=module.MpiContext(enabled=False, size=2),
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


def test_gather_segment_execution_timings_uses_root_receive_buffer(capsys):
    module = _load_example_module()

    class RootGatherComm:
        def Gather(self, send_buffer, receive_buffer, root=0):
            assert root == 0
            assert receive_buffer is not None
            receive_buffer[0] = send_buffer
            receive_buffer[1] = 2.0 * send_buffer

    stencil_diagnostics = module.StencilBuildDiagnostics(
        n_halos=10,
        n_unresolved_ngp=2,
        n_native_resolved=3,
        n_supersampled=5,
        n_zero_sample_ngp_fallbacks=1,
        n_query_pixels_total=20,
        n_global_profile_samples=15,
        n_retained_profile_samples=12,
        max_requested_supersampling_level=3,
        max_used_supersampling_level=3,
        elapsed_seconds=10.0,
        query_disc_seconds=2.0,
        compact_lookup_seconds=1.0,
        pix2vec_filter_seconds=3.0,
        concatenate_seconds=0.5,
        jax_transfer_seconds=1.0,
    )

    timing = module.SegmentExecutionTiming(
        compute_seconds=2.0,
        result_wait_seconds=0.5,
        reduction_seconds=0.25,
        stencil_diagnostics=stencil_diagnostics,
    )
    gathered = module.gather_segment_execution_timings(
        timing,
        module.MpiContext(
            enabled=True,
            comm=RootGatherComm(),
            rank=0,
            size=2,
        ),
    )

    assert gathered is not None
    np.testing.assert_allclose(gathered, [timing.as_array(), 2.0 * timing.as_array()])
    module.print_rank_timing_summary("segment 004", gathered)
    captured = capsys.readouterr().out
    assert "[profile] segment 004 rank min/mean/max (s)" in captured
    assert "compute 2.000/3.000/4.000" in captured
    assert "result wait 0.500/0.750/1.000" in captured
    assert "MPI reduce 0.250/0.375/0.500" in captured
    assert "segment 004 stencil rank min/mean/max (s)" in captured
    assert "query_disc 2.000/3.000/4.000" in captured
    assert "stencil counts (rank sum): halos 30" in captured
    assert "NGP 6" in captured


def test_gather_segment_execution_timings_non_root_has_no_receive_buffer():
    module = _load_example_module()

    class NonRootGatherComm:
        def __init__(self):
            self.send_buffer = None

        def Gather(self, send_buffer, receive_buffer, root=0):
            assert root == 0
            assert receive_buffer is None
            self.send_buffer = np.asarray(send_buffer).copy()

    comm = NonRootGatherComm()
    gathered = module.gather_segment_execution_timings(
        module.SegmentExecutionTiming(1.0, 0.2, 0.1),
        module.MpiContext(
            enabled=True,
            comm=comm,
            rank=1,
            size=2,
        ),
    )

    assert gathered is None
    expected = module.SegmentExecutionTiming(1.0, 0.2, 0.1).as_array()
    np.testing.assert_allclose(comm.send_buffer, expected)
    assert comm.send_buffer.dtype == np.dtype(np.float64)


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
    stencil = module.build_adaptive_lightcone_stencil_for_mass_map(
        mass_map,
        catalog,
        r_delta_mpc_h=np.array([100.0]),
        assignment_params=AngularAssignmentParams(
            theta_resolution_rad=1.1,
            n_resolution=2,
        ),
    )

    np.testing.assert_array_equal(np.asarray(stencil.ngp_compact_row), [1])
    np.testing.assert_array_equal(np.asarray(stencil.ngp_active), [True])


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

    stencil = module.build_adaptive_lightcone_stencil_for_mass_map(
        mass_map,
        catalog,
        r_delta_mpc_h=np.array([900.0], dtype=np.float64),
        assignment_params=AngularAssignmentParams(
            theta_resolution_rad=0.1,
            n_resolution=1,
        ),
    )

    assert stencil.sample_r_perp.dtype == jnp.float64


def test_save_npz_paint_mode_writes_only_compressed_nfw_map(tmp_path):
    module = _load_example_module()
    output = tmp_path / "painted.seg000.npz"
    values = np.array([0.75, 0.0], dtype=np.float32)

    module.save_npz(
        output,
        {
            "nfw_particle_counts": values,
            "nfw_map_derivatives": "none",
            "nfw_sum_particle_counts": 0.75,
        },
    )

    with np.load(output) as data:
        assert data.files == ["nfw_particle_counts"]
        np.testing.assert_array_equal(data["nfw_particle_counts"], values)
        assert data["nfw_particle_counts"].dtype == np.dtype(np.float32)
    with ZipFile(output) as archive:
        assert {item.filename for item in archive.infolist()} == {
            "nfw_particle_counts.npy"
        }
        assert all(item.compress_type == ZIP_DEFLATED for item in archive.infolist())


def test_save_npz_derivative_mode_writes_exact_training_arrays(tmp_path):
    module = _load_example_module()
    output = tmp_path / "painted.seg000.npz"
    nfw_diagnostics = {
        "pipeline_mode": "derivatives",
        "nfw_particle_counts": np.array([0.75, 0.5]),
        "nfw_map_derivatives": "concentration",
        "d_nfw_particle_counts_d_concentration_amplitude": np.array([0.1, 0.2]),
        "d_nfw_particle_counts_d_concentration_mass_slope": np.array([0.3, 0.4]),
        "d_nfw_particle_counts_d_concentration_redshift_slope": np.array([0.5, 0.6]),
        "nfw_paint_mode": "adaptive_global_support",
        "nfw_selected_halo_count": 2,
        "nfw_compact_pixel_count": 2,
        "nfw_global_profile_sample_count": 2,
        "nfw_retained_profile_sample_count": 2,
        "nfw_sum_particle_counts": 1.25,
        "nfw_concentration_amplitude": 5.71,
        "nfw_concentration_mass_slope": -0.084,
        "nfw_concentration_redshift_slope": -0.47,
        "nfw_concentration_mass_pivot": 2.0e12,
        "nfw_profile_support": "hard_3d_r_delta_los_projection",
    }

    module.save_npz(output, nfw_diagnostics)

    expected_keys = {
        "nfw_particle_counts",
        "d_nfw_particle_counts_d_concentration_amplitude",
        "d_nfw_particle_counts_d_concentration_mass_slope",
        "d_nfw_particle_counts_d_concentration_redshift_slope",
    }
    with np.load(output) as data:
        assert set(data.files) == expected_keys
        np.testing.assert_allclose(data["nfw_particle_counts"], [0.75, 0.5])
        np.testing.assert_allclose(
            data["d_nfw_particle_counts_d_concentration_amplitude"], [0.1, 0.2]
        )
        np.testing.assert_allclose(
            data["d_nfw_particle_counts_d_concentration_mass_slope"], [0.3, 0.4]
        )
        np.testing.assert_allclose(
            data["d_nfw_particle_counts_d_concentration_redshift_slope"], [0.5, 0.6]
        )
    with ZipFile(output) as archive:
        assert {item.filename.removesuffix(".npy") for item in archive.infolist()} == expected_keys
        assert all(item.compress_type == ZIP_DEFLATED for item in archive.infolist())


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
        halo_pixels,
        temperature=np.array([100.0]),
        nside=1,
    )
    return catalog, np.array([True]), mass_map, SimpleNamespace(cosmology=Cosmology())


def _adaptive_test_stencil(dtype=jnp.float64):
    return AdaptiveLightconeStencil(
        sample_compact_row=jnp.array([0, 1, 0], dtype=jnp.int32),
        sample_halo_id=jnp.array([0, 0, 1], dtype=jnp.int32),
        sample_r_perp=jnp.array([0.05, 0.15, 0.1], dtype=dtype),
        sample_solid_angle_sr=jnp.array([1.0e-8, 1.0e-8, 1.0e-8], dtype=dtype),
        sample_valid=jnp.array([True, True, True]),
        sample_in_compact=jnp.array([True, True, True]),
        ngp_compact_row=jnp.array([0, 0], dtype=jnp.int32),
        ngp_active=jnp.array([False, False]),
        ngp_in_compact=jnp.array([False, False]),
        resolved_halo_mask=jnp.array([True, True]),
        n_pix=2,
    )


def test_run_calibration_for_segment_writes_npz_with_derivative_arrays(
    tmp_path,
    monkeypatch,
):
    module = _load_example_module()
    catalog, _, mass_map, _ = _single_pixel_pipeline_case()
    metadata = SimpleNamespace(particle_mass_msun_h=1.0e10, cosmology=Cosmology())
    output_npz = tmp_path / "painted_nfw.seg000.npz"
    monkeypatch.setattr(module, "read_pinocchio_mass_map_fits", lambda path: mass_map)

    row = module.run_calibration_for_segment(
        segment_index=0,
        mass_map_path=tmp_path / "pinocchio.example.massmap.seg000.fits",
        output_npz=output_npz,
        catalog=catalog,
        sheets=_sheets(),
        metadata=metadata,
        particle_mass=metadata.particle_mass_msun_h,
        args=_workflow_args(mode="derivatives"),
        profile=False,
        compute_map_derivatives=True,
        inclusive_upper=False,
    )

    assert output_npz.exists()
    with np.load(output_npz) as data:
        assert set(data.files) == {
            "nfw_particle_counts",
            "d_nfw_particle_counts_d_concentration_amplitude",
            "d_nfw_particle_counts_d_concentration_mass_slope",
            "d_nfw_particle_counts_d_concentration_redshift_slope",
        }
    assert row["nfw_paint_mode"] == "adaptive_global_support"
    assert row["assigned_to_expected_ratio"] == pytest.approx(1.0)


def test_adaptive_pair_bucket_size_and_compilation_shape_reuse():
    module = _load_example_module()
    assert module.sparse_pair_bucket_size(0) == 0
    assert module.sparse_pair_bucket_size(5) == 8

    stencil = _adaptive_test_stencil()
    selected_mask = np.array([False, True, True])
    bucketed = module.bucket_adaptive_stencil_for_rank_catalog(
        stencil,
        selected_mask,
        3,
    )

    assert bucketed.sample_r_perp.shape == (4,)
    np.testing.assert_array_equal(np.asarray(bucketed.sample_halo_id[:3]), [1, 1, 2])
    np.testing.assert_array_equal(np.asarray(bucketed.resolved_halo_mask), [False, True, True])
    assert jax.jit(lambda value: jnp.sum(value.sample_r_perp))(bucketed).shape == ()


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_bucketed_adaptive_jit_matches_public_painter_and_conserves_mass(dtype):
    module = _load_example_module()
    selected_catalog = LightconeHaloCatalog(
        unit_vector=jnp.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=dtype),
        chi=jnp.asarray([1000.0, 1200.0], dtype=dtype),
        mass=jnp.asarray([1.0e13, 2.0e13], dtype=dtype),
        redshift=jnp.asarray([0.2, 0.3], dtype=dtype),
    )
    stencil = _adaptive_test_stencil(dtype)
    public_map = paint_lightcone_particle_count_map_sparse(
        stencil,
        selected_catalog,
        particle_mass_msun_h=1.0e10,
        sample_chunk_size=2,
    )
    bucketed = module.bucket_adaptive_stencil_for_rank_catalog(
        stencil,
        np.array([True, True]),
        2,
    )
    diagnostics = module.paint_bucketed_nfw_sparse_map(
        bucketed,
        selected_catalog,
        SimpleNamespace(cosmology=Cosmology()),
        1.0e10,
        5.71,
        -0.084,
        -0.47,
        2.0e12,
        2,
        np.array([1.0, 1.0]),
        8,
        compute_map_derivatives=True,
        profile=False,
    )
    compiled_map, derivative_diagnostics = diagnostics

    tolerance = 5.0e-5 if dtype == jnp.float32 else 1.0e-10
    np.testing.assert_allclose(compiled_map, public_map, rtol=tolerance)
    assert derivative_diagnostics["nfw_assigned_global_particle_count"] == pytest.approx(
        3000.0,
        rel=tolerance,
    )
    np.testing.assert_allclose(
        derivative_diagnostics["nfw_global_derivative_sums"],
        0.0,
        atol=1.0e-4 if dtype == jnp.float32 else 1.0e-9,
    )


def test_bucketed_adaptive_jit_handles_empty_catalog():
    module = _load_example_module()
    catalog = LightconeHaloCatalog(
        unit_vector=jnp.empty((0, 3)),
        chi=jnp.empty((0,)),
        mass=jnp.empty((0,)),
        redshift=jnp.empty((0,)),
    )
    empty = AdaptiveLightconeStencil(
        sample_compact_row=jnp.empty((0,), dtype=jnp.int32),
        sample_halo_id=jnp.empty((0,), dtype=jnp.int32),
        sample_r_perp=jnp.empty((0,)),
        sample_solid_angle_sr=jnp.empty((0,)),
        sample_valid=jnp.empty((0,), dtype=bool),
        sample_in_compact=jnp.empty((0,), dtype=bool),
        ngp_compact_row=jnp.empty((0,), dtype=jnp.int32),
        ngp_active=jnp.empty((0,), dtype=bool),
        ngp_in_compact=jnp.empty((0,), dtype=bool),
        resolved_halo_mask=jnp.empty((0,), dtype=bool),
        n_pix=3,
    )
    bucketed = module.bucket_adaptive_stencil_for_rank_catalog(
        empty,
        np.zeros(0, dtype=bool),
        0,
    )
    painted = paint_lightcone_particle_count_map_sparse(
        bucketed,
        catalog,
        particle_mass_msun_h=1.0e10,
    )

    np.testing.assert_array_equal(np.asarray(painted), np.zeros(3))


def test_run_nfw_calibration_pipeline_uses_adaptive_builder_and_conserves_global_mass(
    monkeypatch,
):
    module = _load_example_module()
    catalog, mask, mass_map, metadata = _single_pixel_pipeline_case()

    def fail_if_called(*args, **kwargs):
        raise AssertionError("the dense all-pairs builder must not be called")

    monkeypatch.setattr(module, "build_lightcone_sparse_stencil_bruteforce", fail_if_called)
    diagnostics = module.run_nfw_calibration_pipeline(
        catalog,
        mask,
        mass_map,
        metadata,
        particle_mass_msun_h=1.0e10,
        theta_resolution_rad=None,
        n_resolution=4,
    )

    assert diagnostics["nfw_paint_mode"] == "adaptive_global_support"
    assert diagnostics["nfw_assigned_to_expected_ratio"] == pytest.approx(1.0)
    assert diagnostics["nfw_unresolved_ngp_count"] == 1
    assert diagnostics["nfw_global_profile_sample_count"] == 0


def test_run_nfw_calibration_pipeline_derivatives_include_global_normalization():
    module = _load_example_module()
    catalog, mask, mass_map, metadata = _single_pixel_pipeline_case()
    diagnostics = module.run_nfw_calibration_pipeline(
        catalog,
        mask,
        mass_map,
        metadata,
        particle_mass_msun_h=1.0e10,
        compute_map_derivatives=True,
    )

    assert diagnostics["nfw_map_derivatives"] == "concentration"
    np.testing.assert_allclose(diagnostics["nfw_global_derivative_sums"], 0.0)
    for key in (
        "d_nfw_particle_counts_d_concentration_amplitude",
        "d_nfw_particle_counts_d_concentration_mass_slope",
        "d_nfw_particle_counts_d_concentration_redshift_slope",
    ):
        assert np.asarray(diagnostics[key]).shape == (1,)


def test_nfw_stage_label_and_summary_use_adaptive_diagnostics(capsys):
    module = _load_example_module()
    assert module.nfw_stage_label("paint") == "NFW calibration pipeline: paint"
    diagnostics = _fake_segment_result(
        module,
        segment_index=0,
        mass_map_path=Path("massmap.seg000.fits"),
        output_npz=Path("painted.seg000.npz"),
        inclusive_upper=False,
    ).nfw_diagnostics
    module.print_nfw_calibration_summary(diagnostics)
    output = capsys.readouterr().out

    assert "Painter mode: adaptive_global_support" in output
    assert "Assignment branches" in output
    assert "Assigned global / expected particle count" in output


def test_reduce_calibration_segment_result_sums_adaptive_payloads():
    module = _load_example_module()
    local = _fake_segment_result(
        module,
        segment_index=0,
        mass_map_path=Path("massmap.seg000.fits"),
        output_npz=Path("painted.seg000.npz"),
        inclusive_upper=False,
    )
    local.nfw_diagnostics.update(
        {
            "pipeline_mode": "derivatives",
            "nfw_particle_counts": np.array([0.5, 1.5]),
            "nfw_map_derivatives": "concentration",
            "d_nfw_particle_counts_d_concentration_amplitude": np.array([0.1, 0.2]),
            "d_nfw_particle_counts_d_concentration_mass_slope": np.array([0.3, 0.4]),
            "d_nfw_particle_counts_d_concentration_redshift_slope": np.array([0.5, 0.6]),
            "nfw_global_derivative_sums": np.zeros(3),
            "nfw_compact_derivative_sums": np.array([0.3, 0.7, 1.1]),
            "nfw_selected_halo_mass_msun_h": 4.0,
            "nfw_expected_global_particle_count": 4.0,
            "nfw_assigned_global_particle_count": 4.0,
            "nfw_retained_compact_particle_count": 2.0,
            "nfw_outside_compact_particle_count": 2.0,
            "nfw_global_profile_sample_count": 2,
            "nfw_retained_profile_sample_count": 1,
            "nfw_unresolved_ngp_count": 0,
            "nfw_native_resolved_count": 1,
            "nfw_supersampled_count": 0,
            "nfw_max_requested_supersampling_level": 0,
            "nfw_max_used_supersampling_level": 0,
        }
    )
    remote_values = [
        np.array([2.0, 3.0]),
        np.array([0.4, 0.5]),
        np.array([0.6, 0.7]),
        np.array([0.8, 0.9]),
        np.zeros(3),
        np.array([0.9, 1.3, 1.7]),
        np.array([2, 4, 3, 1, 1, 1, 0, 0], dtype=np.int64),
        np.array([5.0, 5.0, 5.0, 5.0, 0.0], dtype=np.float64),
        np.array([2, 2], dtype=np.int64),
    ]
    max_marker = object()

    class PairwiseReduceComm:
        def __init__(self, values):
            self.values = list(values)

        def Reduce(self, send_buffer, receive_buffer, op=None, root=0):
            assert root == 0
            other = self.values.pop(0)
            operation = np.maximum if op is max_marker else np.add
            receive_buffer[...] = operation(np.asarray(send_buffer), np.asarray(other))

    comm = PairwiseReduceComm(remote_values)
    reduced = module.reduce_calibration_segment_result(
        local,
        module.MpiContext(
            enabled=True,
            comm=comm,
            rank=0,
            size=2,
            max_op=max_marker,
        ),
    )

    assert reduced is not None
    diagnostics = reduced.nfw_diagnostics
    np.testing.assert_allclose(diagnostics["nfw_particle_counts"], [2.5, 4.5])
    assert diagnostics["nfw_selected_halo_count"] == 3
    assert diagnostics["nfw_global_profile_sample_count"] == 6
    assert diagnostics["nfw_native_resolved_count"] == 2
    assert diagnostics["nfw_supersampled_count"] == 1
    assert diagnostics["nfw_selected_halo_mass_msun_h"] == 9.0
    assert diagnostics["nfw_expected_global_particle_count"] == 9.0
    assert diagnostics["nfw_assigned_global_particle_count"] == 9.0
    assert diagnostics["nfw_retained_compact_particle_count"] == 7.0
    assert diagnostics["nfw_outside_compact_particle_count"] == 2.0
    assert diagnostics["nfw_assigned_to_expected_ratio"] == pytest.approx(1.0)
    assert diagnostics["nfw_max_used_supersampling_level"] == 2
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
        inclusive_upper=False,
    )
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
        np.dtype(np.int64),
        np.dtype(np.float64),
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
        inclusive_upper=False,
    )
    local.nfw_diagnostics["nfw_particle_counts"] = np.array(
        [2 * rank_value],
        dtype=np.float64,
    )
    for key in (
        "nfw_selected_halo_count",
        "nfw_global_profile_sample_count",
        "nfw_retained_profile_sample_count",
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
            max_op=mpi.MAX,
        ),
    )

    if comm.Get_rank() != 0:
        assert reduced is None
        return

    assert reduced is not None
    expected = comm.Get_size() * (comm.Get_size() + 1) // 2
    np.testing.assert_allclose(
        reduced.nfw_diagnostics["nfw_particle_counts"],
        [2 * expected],
    )
    assert reduced.nfw_diagnostics["nfw_selected_halo_count"] == expected


def test_real_mpi_workflow_matches_serial_for_worker_counts_and_derivatives(
    tmp_path,
    monkeypatch,
):
    module = _load_example_module()
    if module._mpi_environment_size_hint() < 2:
        pytest.skip("requires execution under mpiexec with at least two ranks")

    mpi = pytest.importorskip("mpi4py.MPI")
    hp = pytest.importorskip("healpy")
    comm = mpi.COMM_WORLD
    if comm.Get_size() < 2:
        pytest.skip("requires execution under mpiexec with at least two ranks")

    rank_path = tmp_path / f"rank_{comm.Get_rank()}"
    rank_path.mkdir(parents=True, exist_ok=True)
    for segment_index in range(2):
        path = rank_path / f"run.massmap.seg{segment_index:03d}.fits"
        path.touch()
    sheets = PinocchioMassSheetTable(
        sheet_ids=np.array([0, 1]),
        z_hi=np.array([0.3, 0.5]),
        z_lo=np.array([0.1, 0.3]),
        delta_z=np.array([0.2, 0.2]),
        chi_hi_mpc_h=np.array([300.0, 500.0]),
        chi_lo_mpc_h=np.array([100.0, 300.0]),
        delta_chi_mpc_h=np.array([200.0, 200.0]),
        inv_delta_chi_h_mpc=np.array([0.005, 0.005]),
        da_hi_mpc_h=np.array([230.0, 330.0]),
        da_lo_mpc_h=np.array([90.0, 230.0]),
        chi3_diff_mpc_h3=np.array([1.0, 1.0]),
        source=rank_path / "sheets.out",
    )
    halo_pixels = np.array([0, 5, 8, 11], dtype=np.int64)
    full_catalog = _catalog(
        unit_vector=np.stack(hp.pix2vec(1, halo_pixels), axis=-1),
        mass=np.array([1.0e13, 2.0e13, 3.0e13, 4.0e13]),
        redshift=np.array([0.15, 0.25, 0.35, 0.5]),
        chi=np.array([150.0, 250.0, 350.0, 500.0]),
    )
    rank_indices = np.array_split(
        np.arange(full_catalog.mass.shape[0]),
        comm.Get_size(),
    )[comm.Get_rank()]
    rank_catalog = LightconeHaloCatalog(
        unit_vector=full_catalog.unit_vector[rank_indices],
        chi=full_catalog.chi[rank_indices],
        mass=full_catalog.mass[rank_indices],
        redshift=full_catalog.redshift[rank_indices],
    )
    mass_maps = {
        0: _mass_map(np.arange(12, dtype=np.int64), nside=1),
        1: _mass_map(np.arange(11, -1, -1, dtype=np.int64), nside=1),
    }

    def fake_read_mass_map(path):
        return mass_maps[module.parse_segment_index_from_mass_map_path(Path(path))]

    monkeypatch.setattr(module, "read_pinocchio_mass_map_fits", fake_read_mass_map)
    metadata = SimpleNamespace(particle_mass_msun_h=1.0e10, cosmology=Cosmology())

    def workflow_args(output_dir, workers, *, mpi_enabled):
        return _workflow_args(
            mass_map=None,
            sheet_index=None,
            output=None,
            mass_map_glob=str(rank_path / "run.massmap.seg*.fits"),
            output_dir=output_dir,
            mode="derivatives",
            segment_workers=workers,
            mpi_plc_parts=mpi_enabled,
        )

    serial_dir = rank_path / "serial"
    serial_rows = module.run_segment_workflow(
        workflow_args(serial_dir, 1, mpi_enabled=False),
        workflow="all",
        catalog=full_catalog,
        sheets=sheets,
        metadata=metadata,
        particle_mass=metadata.particle_mass_msun_h,
        profile=False,
        compute_map_derivatives=True,
    )
    map_keys = (
        "nfw_particle_counts",
        "d_nfw_particle_counts_d_concentration_amplitude",
        "d_nfw_particle_counts_d_concentration_mass_slope",
        "d_nfw_particle_counts_d_concentration_redshift_slope",
    )
    serial_arrays = {}
    for segment_index in range(2):
        with np.load(module.segment_output_path(serial_dir, segment_index)) as data:
            serial_arrays[segment_index] = {
                key: np.asarray(data[key]) for key in map_keys
            }

    assert [row["segment_index"] for row in serial_rows] == [0, 1]
    for workers in (1, 2):
        mpi_dir = rank_path / f"mpi_workers_{workers}"
        mpi_rows = module.run_segment_workflow(
            workflow_args(mpi_dir, workers, mpi_enabled=True),
            workflow="all",
            catalog=rank_catalog,
            sheets=sheets,
            metadata=metadata,
            particle_mass=metadata.particle_mass_msun_h,
            profile=False,
            compute_map_derivatives=True,
            mpi_context=module.MpiContext(
                enabled=True,
                comm=comm,
                rank=comm.Get_rank(),
                size=comm.Get_size(),
                sum_op=mpi.SUM,
                max_op=mpi.MAX,
            ),
        )
        comm.Barrier()
        if comm.Get_rank() != 0:
            assert mpi_rows == []
            continue

        assert [row["segment_index"] for row in mpi_rows] == [0, 1]
        assert [row["selected_halo_count"] for row in mpi_rows] == [
            row["selected_halo_count"] for row in serial_rows
        ]
        assert [row["global_profile_sample_count"] for row in mpi_rows] == [
            row["global_profile_sample_count"] for row in serial_rows
        ]
        for segment_index in range(2):
            with np.load(module.segment_output_path(mpi_dir, segment_index)) as data:
                for key in map_keys:
                    np.testing.assert_allclose(
                        data[key],
                        serial_arrays[segment_index][key],
                        rtol=1.0e-12,
                        atol=1.0e-12,
                    )


def test_real_mpi_segment_timing_gather():
    module = _load_example_module()
    if module._mpi_environment_size_hint() < 2:
        pytest.skip("requires execution under mpiexec with at least two ranks")

    mpi = pytest.importorskip("mpi4py.MPI")
    comm = mpi.COMM_WORLD
    if comm.Get_size() < 2:
        pytest.skip("requires execution under mpiexec with at least two ranks")

    rank_value = float(comm.Get_rank() + 1)
    gathered = module.gather_segment_execution_timings(
        module.SegmentExecutionTiming(
            compute_seconds=rank_value,
            result_wait_seconds=2.0 * rank_value,
            reduction_seconds=3.0 * rank_value,
        ),
        module.MpiContext(
            enabled=True,
            comm=comm,
            rank=comm.Get_rank(),
            size=comm.Get_size(),
            sum_op=mpi.SUM,
            max_op=mpi.MAX,
        ),
    )

    if comm.Get_rank() != 0:
        assert gathered is None
        return

    assert gathered is not None
    expected_rank_values = np.arange(1, comm.Get_size() + 1, dtype=np.float64)
    np.testing.assert_allclose(gathered[:, 0], expected_rank_values)
    np.testing.assert_allclose(gathered[:, 1], 2.0 * expected_rank_values)
    np.testing.assert_allclose(gathered[:, 2], 3.0 * expected_rank_values)


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
    assert calls[1]["output_npz"] == output_dir / "painted_nfw.seg001.npz"
    assert all("output_fits" not in call for call in calls)
    assert manifest_calls[0][0] == output_dir / "painted_nfw_manifest.csv"
    assert rows == manifest_calls[0][1]


def test_run_segment_workflow_single_segment_uses_last_segment_flag(monkeypatch):
    module = _load_example_module()
    args = _workflow_args(last_segment_inclusive=True)
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
    assert "output_fits" not in calls[0]


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
    seen = {"builds": 0, "nfw_index_id": None}
    original_from_mass_map = module.MassMapPixelIndex.from_mass_map

    def fake_from_mass_map(mass_map_arg, *, max_dense_bytes=module._PIXEL_INDEX_DENSE_MAX_BYTES):
        seen["builds"] += 1
        return original_from_mass_map(mass_map_arg, max_dense_bytes=max_dense_bytes)

    def fake_nfw_pipeline(*args, **kwargs):
        del args
        seen["nfw_index_id"] = id(kwargs["pixel_index"])
        return {
            "pipeline_mode": "paint",
            "particle_mass_msun_h": 1.0,
            "nfw_particle_counts": np.array([0.5]),
            "nfw_map_derivatives": "none",
            "nfw_paint_mode": "adaptive_global_support",
            "nfw_selected_halo_count": 1,
            "nfw_compact_pixel_count": 1,
            "nfw_global_profile_sample_count": 1,
            "nfw_retained_profile_sample_count": 1,
            "nfw_sum_particle_counts": 0.5,
            "nfw_concentration_amplitude": 5.71,
            "nfw_concentration_mass_slope": -0.084,
            "nfw_concentration_redshift_slope": -0.47,
            "nfw_concentration_mass_pivot": 2.0e12,
            "nfw_profile_support": "hard_3d_r_delta_los_projection",
        }

    monkeypatch.setattr(
        module.MassMapPixelIndex,
        "from_mass_map",
        staticmethod(fake_from_mass_map),
    )
    monkeypatch.setattr(module, "read_pinocchio_mass_map_fits", lambda path: mass_map)
    monkeypatch.setattr(module, "run_nfw_calibration_pipeline", fake_nfw_pipeline)

    result = module.compute_calibration_for_segment(
        segment_index=0,
        mass_map_path=tmp_path / "massmap.seg000.fits",
        output_npz=tmp_path / "painted.seg000.npz",
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
    assert seen["nfw_index_id"] != id(None)


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
        return _fake_segment_result(
            module,
            segment_index=kwargs["segment_index"],
            mass_map_path=kwargs["mass_map_path"],
            output_npz=kwargs["output_npz"],
            inclusive_upper=kwargs["inclusive_upper"],
        )

    def fake_write(result, args, metadata, *, profile, verbose, provenance):
        del args, metadata, profile, verbose
        assert provenance.segment_worker_count == 2
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
    assert [call["inclusive_upper"] for call in compute_calls] == [False, True]
    assert write_calls == [0, 1]
    assert rows == [{"segment_index": 0}, {"segment_index": 1}]
    assert manifest_calls[0][1] == rows


def test_run_segment_workflow_partial_batch_keeps_highest_segment_half_open(
    tmp_path,
    monkeypatch,
):
    module = _load_example_module()
    segment_path = tmp_path / "run.massmap.seg000.fits"
    segment_path.touch()
    args = _workflow_args(
        mass_map=None,
        sheet_index=None,
        output=None,
        mass_map_glob=str(tmp_path / "*.fits"),
        output_dir=tmp_path / "painted",
        segment_workers=2,
    )
    inclusive_values = []

    def fake_compute(**kwargs):
        inclusive_values.append(kwargs["inclusive_upper"])
        return _fake_segment_result(
            module,
            segment_index=kwargs["segment_index"],
            mass_map_path=kwargs["mass_map_path"],
            output_npz=kwargs["output_npz"],
            inclusive_upper=kwargs["inclusive_upper"],
        )

    monkeypatch.setattr(module, "compute_calibration_for_segment", fake_compute)
    monkeypatch.setattr(
        module,
        "write_calibration_segment_outputs",
        lambda result, args, metadata, **kwargs: {
            "segment_index": result.segment_index
        },
    )
    monkeypatch.setattr(module, "write_manifest", lambda path, rows: None)

    module.run_segment_workflow(
        args,
        workflow="all",
        catalog=_catalog(),
        sheets=_sheets(),
        metadata=SimpleNamespace(particle_mass_msun_h=1.0, cosmology=Cosmology()),
        particle_mass=1.0,
        profile=False,
        compute_map_derivatives=False,
    )

    assert inclusive_values == [False]


def test_run_segment_workflow_aborts_mpi_job_on_rank_local_compute_error(
    monkeypatch,
    capsys,
):
    module = _load_example_module()

    class AbortComm:
        def __init__(self):
            self.error_codes = []

        def Abort(self, error_code):
            self.error_codes.append(error_code)

    comm = AbortComm()

    def fail_compute(**kwargs):
        raise OSError(f"cannot read {kwargs['mass_map_path']}")

    monkeypatch.setattr(module, "compute_calibration_for_segment", fail_compute)

    with pytest.raises(RuntimeError, match="returned after Abort"):
        module.run_segment_workflow(
            _workflow_args(mpi_plc_parts=True),
            workflow="single",
            catalog=_catalog(),
            sheets=_sheets(),
            metadata=SimpleNamespace(particle_mass_msun_h=1.0, cosmology=Cosmology()),
            particle_mass=1.0,
            profile=False,
            compute_map_derivatives=False,
            mpi_context=module.MpiContext(
                enabled=True,
                comm=comm,
                rank=1,
                size=2,
            ),
        )

    assert comm.error_codes == [1]
    error_output = capsys.readouterr().err
    assert "rank 1/2" in error_output
    assert "segment 0" in error_output


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
            inclusive_upper=kwargs["inclusive_upper"],
        )

    def fake_reduce(result, mpi_context):
        events.append(("reduce", result.segment_index))
        reduce_calls.append((result.segment_index, mpi_context.rank))
        return result

    def fake_write(result, args, metadata, *, profile, verbose, provenance):
        del args, metadata, profile, verbose
        assert provenance.mpi_rank_count == 2
        events.append(("write", result.segment_index))
        write_calls.append(result.output_npz)
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
        output_dir / "painted_nfw.seg000.npz",
        output_dir / "painted_nfw.seg001.npz",
    ]
    assert manifest_calls[0][0] == output_dir / "painted_nfw_manifest.csv"
    assert rows == [{"segment_index": 0}, {"segment_index": 1}]


def test_run_segment_workflow_mpi_profile_gathers_root_timing_summaries(
    tmp_path,
    monkeypatch,
    capsys,
):
    module = _load_example_module()
    for segment_index in range(2):
        (tmp_path / f"run.massmap.seg{segment_index:03d}.fits").touch()
    output_dir = tmp_path / "painted"
    args = _workflow_args(
        mass_map=None,
        sheet_index=None,
        output=None,
        mass_map_glob=str(tmp_path / "*.fits"),
        output_dir=output_dir,
        mpi_plc_parts=True,
        segment_workers=1,
    )

    class ProfileGatherComm:
        def __init__(self):
            self.send_buffers = []

        def Gather(self, send_buffer, receive_buffer, root=0):
            assert root == 0
            assert receive_buffer is not None
            self.send_buffers.append(np.asarray(send_buffer).copy())
            receive_buffer[0] = send_buffer
            receive_buffer[1] = send_buffer + 1.0

    comm = ProfileGatherComm()

    def fake_compute(**kwargs):
        return _fake_segment_result(
            module,
            segment_index=kwargs["segment_index"],
            mass_map_path=kwargs["mass_map_path"],
            output_npz=kwargs["output_npz"],
            inclusive_upper=kwargs["inclusive_upper"],
        )

    def fake_write(result, args, metadata, *, profile, verbose, provenance):
        del args, metadata
        assert profile is True
        assert verbose is True
        assert provenance.mpi_rank_count == 2
        assert "compute_seconds" not in result.nfw_diagnostics
        return {"segment_index": result.segment_index}

    monkeypatch.setattr(module, "compute_calibration_for_segment", fake_compute)
    monkeypatch.setattr(
        module,
        "reduce_calibration_segment_result",
        lambda result, mpi_context: result,
    )
    monkeypatch.setattr(module, "write_calibration_segment_outputs", fake_write)
    monkeypatch.setattr(module, "write_manifest", lambda path, rows: None)

    rows = module.run_segment_workflow(
        args,
        workflow="all",
        catalog=_catalog(),
        sheets=_sheets(),
        metadata=SimpleNamespace(particle_mass_msun_h=1.0, cosmology=Cosmology()),
        particle_mass=1.0,
        profile=True,
        compute_map_derivatives=False,
        mpi_context=module.MpiContext(
            enabled=True,
            comm=comm,
            rank=0,
            size=2,
        ),
    )

    assert rows == [{"segment_index": 0}, {"segment_index": 1}]
    assert len(comm.send_buffers) == 2
    assert all(buffer.shape == (19,) for buffer in comm.send_buffers)
    captured = capsys.readouterr().out
    assert "[profile] segment 000 rank min/mean/max (s)" in captured
    assert "[profile] segment 001 rank min/mean/max (s)" in captured
    assert "[profile] all-segment totals rank min/mean/max (s)" in captured


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
    outstanding = {"current": 0, "maximum": 0}

    class FakeFuture:
        def __init__(self, result):
            self._result = result

        def result(self):
            outstanding["current"] -= 1
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
            outstanding["current"] += 1
            outstanding["maximum"] = max(
                outstanding["maximum"],
                outstanding["current"],
            )
            return FakeFuture(fn(spec))

    def fake_compute(**kwargs):
        events.append(("compute", kwargs["segment_index"]))
        return _fake_segment_result(
            module,
            segment_index=kwargs["segment_index"],
            mass_map_path=kwargs["mass_map_path"],
            output_npz=kwargs["output_npz"],
            inclusive_upper=kwargs["inclusive_upper"],
        )

    def fake_reduce(result, mpi_context):
        events.append(("reduce", result.segment_index, mpi_context.rank))
        return result

    def fake_write(result, args, metadata, *, profile, verbose, provenance):
        del args, metadata, profile, verbose
        assert provenance.segment_worker_count == 2
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
        sheets=[None, None, None],
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
    assert outstanding == {"current": 0, "maximum": 2}
    assert rows == [{"segment_index": 0}, {"segment_index": 1}, {"segment_index": 2}]
    assert manifest_calls[0][0] == output_dir / "painted_nfw_manifest.csv"


def _r_delta_for_theta(theta, chi):
    return 2.0 * chi * np.sin(0.5 * theta)


def test_adaptive_unresolved_branch_uses_ngp_without_query(monkeypatch):
    hp = pytest.importorskip("healpy")
    module = _load_example_module()
    nside = 8
    halo_ring = 100
    halo_vector = np.asarray(hp.pix2vec(nside, halo_ring))[None, :]
    catalog = _catalog(
        unit_vector=halo_vector,
        mass=np.array([1.0e13]),
        redshift=np.array([0.2]),
        chi=np.array([1000.0]),
    )
    mass_map = _mass_map(np.array([halo_ring]), nside=nside)
    theta_resolution = 0.01

    monkeypatch.setattr(
        hp,
        "query_disc",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("unresolved halos must not call query_disc")
        ),
    )
    stencil, diagnostics = module.build_adaptive_lightcone_stencil_for_mass_map(
        mass_map,
        catalog,
        np.array([_r_delta_for_theta(0.5 * theta_resolution, 1000.0)]),
        AngularAssignmentParams(theta_resolution, 4),
        collect_diagnostics=True,
    )

    assert diagnostics.n_unresolved_ngp == 1
    assert diagnostics.n_global_profile_samples == 0
    np.testing.assert_array_equal(np.asarray(stencil.ngp_active), [True])
    np.testing.assert_array_equal(np.asarray(stencil.ngp_compact_row), [0])


def test_adaptive_supersampling_queries_children_and_maps_nested_parents_to_ring():
    hp = pytest.importorskip("healpy")
    module = _load_example_module()
    nside = 8
    pixels = np.arange(hp.nside2npix(nside), dtype=np.int64)
    halo_ring = 100
    halo_vector = np.asarray(hp.pix2vec(nside, halo_ring))[None, :]
    chi = 1000.0
    theta_map = np.sqrt(hp.nside2pixarea(nside))
    theta_h = 3.0 * theta_map
    catalog = _catalog(
        unit_vector=halo_vector,
        mass=np.array([1.0e13]),
        redshift=np.array([0.2]),
        chi=np.array([chi]),
    )
    catalog = module.selected_lightcone_catalog(catalog, np.array([True]))
    stencil, diagnostics = module.build_adaptive_lightcone_stencil_for_mass_map(
        _mass_map(pixels, nside=nside),
        catalog,
        np.array([_r_delta_for_theta(theta_h, chi)]),
        AngularAssignmentParams(theta_map / 10.0, 4),
        collect_diagnostics=True,
    )

    assert diagnostics.n_supersampled == 1
    assert diagnostics.max_requested_supersampling_level == 1
    assert diagnostics.n_native_resolved == 0
    assert stencil.size > 0
    child_nside = 2 * nside
    children = hp.query_disc(
        child_nside,
        halo_vector[0],
        np.nextafter(theta_h, np.inf),
        inclusive=False,
        nest=True,
    )
    child_vectors = np.stack(hp.pix2vec(child_nside, children, nest=True), axis=-1)
    r_perp = chi * np.sqrt(
        np.maximum(2.0 * (1.0 - child_vectors @ halo_vector[0]), 0.0)
    )
    children = children[r_perp <= _r_delta_for_theta(theta_h, chi)]
    expected_parent_ring = hp.nest2ring(nside, children // 4)

    np.testing.assert_array_equal(
        np.sort(np.asarray(stencil.sample_compact_row)),
        np.sort(expected_parent_ring),
    )
    assert np.unique(np.asarray(stencil.sample_compact_row)).size < stencil.size


def test_adaptive_native_branch_and_selection_are_concentration_independent():
    hp = pytest.importorskip("healpy")
    module = _load_example_module()
    nside = 8
    pixels = np.arange(hp.nside2npix(nside), dtype=np.int64)
    halo_vector = np.asarray(hp.pix2vec(nside, 100))[None, :]
    chi = 1000.0
    theta_map = np.sqrt(hp.nside2pixarea(nside))
    theta_h = 4.5 * theta_map
    catalog = _catalog(
        unit_vector=halo_vector,
        mass=np.array([1.0e13]),
        redshift=np.array([0.2]),
        chi=np.array([chi]),
    )
    arguments = (
        _mass_map(pixels, nside=nside),
        catalog,
        np.array([_r_delta_for_theta(theta_h, chi)]),
        AngularAssignmentParams(theta_map / 10.0, 4),
    )
    first, first_diagnostics = module.build_adaptive_lightcone_stencil_for_mass_map(
        *arguments,
        collect_diagnostics=True,
    )
    second, second_diagnostics = module.build_adaptive_lightcone_stencil_for_mass_map(
        *arguments,
        collect_diagnostics=True,
    )

    assert first_diagnostics.n_native_resolved == 1
    assert first_diagnostics.n_supersampled == 0
    assert second_diagnostics.n_native_resolved == 1
    np.testing.assert_array_equal(first.sample_halo_id, second.sample_halo_id)
    np.testing.assert_allclose(first.sample_r_perp, second.sample_r_perp)


def test_adaptive_global_support_is_kept_before_compact_filtering():
    hp = pytest.importorskip("healpy")
    module = _load_example_module()
    nside = 8
    halo_ring = 100
    halo_vector = np.asarray(hp.pix2vec(nside, halo_ring))[None, :]
    chi = 1.0
    theta_map = np.sqrt(hp.nside2pixarea(nside))
    catalog = _catalog(
        unit_vector=halo_vector,
        mass=np.array([1.0e13]),
        redshift=np.array([0.2]),
        chi=np.array([chi]),
    )
    catalog = module.selected_lightcone_catalog(catalog, np.array([True]))
    r_delta = module.nfw_support_rdelta_mpc_h(
        catalog,
        SimpleNamespace(cosmology=Cosmology()),
        module.NFWProfileParams(),
    )
    stencil, diagnostics = module.build_adaptive_lightcone_stencil_for_mass_map(
        _mass_map(np.array([halo_ring]), nside=nside),
        catalog,
        r_delta,
        AngularAssignmentParams(theta_map / 10.0, 4),
        collect_diagnostics=True,
    )
    painted = paint_lightcone_particle_count_map_sparse(
        stencil,
        catalog,
        particle_mass_msun_h=1.0e10,
    )

    assert diagnostics.n_global_profile_samples > diagnostics.n_retained_profile_samples
    assert np.any(~np.asarray(stencil.sample_in_compact))
    assert 0.0 < float(jnp.sum(painted)) < 1000.0


def test_adaptive_zero_sample_query_falls_back_to_ngp(monkeypatch):
    hp = pytest.importorskip("healpy")
    module = _load_example_module()
    nside = 8
    halo_ring = 100
    halo_vector = np.asarray(hp.pix2vec(nside, halo_ring))[None, :]
    catalog = _catalog(
        unit_vector=halo_vector,
        mass=np.array([1.0e13]),
        redshift=np.array([0.2]),
        chi=np.array([1000.0]),
    )
    monkeypatch.setattr(hp, "query_disc", lambda *args, **kwargs: np.empty(0, dtype=np.int64))
    stencil, diagnostics = module.build_adaptive_lightcone_stencil_for_mass_map(
        _mass_map(np.array([halo_ring]), nside=nside),
        catalog,
        np.array([500.0]),
        AngularAssignmentParams(0.01, 1),
        collect_diagnostics=True,
    )

    assert diagnostics.n_zero_sample_ngp_fallbacks == 1
    np.testing.assert_array_equal(np.asarray(stencil.ngp_active), [True])
    np.testing.assert_array_equal(np.asarray(stencil.resolved_halo_mask), [False])


def test_adaptive_controls_default_and_invalid_child_nside():
    hp = pytest.importorskip("healpy")
    module = _load_example_module()
    resolved = module.resolve_angular_assignment(8, AngularAssignmentParams())

    assert resolved.theta_resolution_rad == pytest.approx(0.5 * hp.max_pixrad(8))
    assert resolved.n_resolution == 4
    with pytest.raises(ValueError, match="invalid child NSIDE"):
        module.resolve_angular_assignment(
            2**29,
            AngularAssignmentParams(theta_resolution_rad=1.0e-30, n_resolution=4),
        )


def test_unresolved_halo_outside_compact_map_deposits_zero_without_redirection():
    hp = pytest.importorskip("healpy")
    module = _load_example_module()
    nside = 8
    host_ring = 100
    compact_ring = 300
    catalog = LightconeHaloCatalog(
        unit_vector=jnp.asarray(np.asarray(hp.pix2vec(nside, host_ring))[None, :]),
        chi=jnp.asarray([1000.0]),
        mass=jnp.asarray([1.0e13]),
        redshift=jnp.asarray([0.2]),
    )
    stencil = module.build_adaptive_lightcone_stencil_for_mass_map(
        _mass_map(np.array([compact_ring]), nside=nside),
        catalog,
        np.array([0.1]),
        AngularAssignmentParams(theta_resolution_rad=0.01, n_resolution=4),
    )
    painted = paint_lightcone_particle_count_map_sparse(
        stencil,
        catalog,
        particle_mass_msun_h=1.0e10,
    )

    np.testing.assert_array_equal(np.asarray(stencil.ngp_in_compact), [False])
    np.testing.assert_array_equal(np.asarray(painted), [0.0])


def test_invalid_resolved_normalization_raises_detailed_error():
    module = _load_example_module()
    catalog = LightconeHaloCatalog(
        unit_vector=jnp.asarray([[1.0, 0.0, 0.0]]),
        chi=jnp.asarray([1000.0]),
        mass=jnp.asarray([1.0e13]),
        redshift=jnp.asarray([0.2]),
    )
    stencil = AdaptiveLightconeStencil(
        sample_compact_row=jnp.asarray([0], dtype=jnp.int32),
        sample_halo_id=jnp.asarray([0], dtype=jnp.int32),
        sample_r_perp=jnp.asarray([100.0]),
        sample_solid_angle_sr=jnp.asarray([1.0e-8]),
        sample_valid=jnp.asarray([True]),
        sample_in_compact=jnp.asarray([True]),
        ngp_compact_row=jnp.asarray([0], dtype=jnp.int32),
        ngp_active=jnp.asarray([False]),
        ngp_in_compact=jnp.asarray([False]),
        resolved_halo_mask=jnp.asarray([True]),
        n_pix=1,
    )

    with pytest.raises(ValueError, match="invalid adaptive NFW normalization"):
        module.paint_bucketed_nfw_sparse_map(
            stencil,
            catalog,
            SimpleNamespace(cosmology=Cosmology()),
            1.0e10,
            5.71,
            -0.084,
            -0.47,
            2.0e12,
            16,
            np.array([1.0]),
            8,
            compute_map_derivatives=False,
            profile=False,
        )


def test_supersampling_converges_to_finer_child_grid_reference():
    hp = pytest.importorskip("healpy")
    module = _load_example_module()
    nside = 8
    pixels = np.arange(hp.nside2npix(nside), dtype=np.int64)
    halo_vector = np.asarray(hp.pix2vec(nside, 100))[None, :]
    mass = jnp.asarray([1.0e14])
    redshift = jnp.asarray([0.3])
    provisional = LightconeHaloCatalog(
        unit_vector=jnp.asarray(halo_vector),
        chi=jnp.asarray([1.0]),
        mass=mass,
        redshift=redshift,
    )
    r_delta = module.nfw_support_rdelta_mpc_h(
        provisional,
        SimpleNamespace(cosmology=Cosmology()),
        module.NFWProfileParams(),
    )
    theta_map = np.sqrt(hp.nside2pixarea(nside))
    theta_h = 2.5 * theta_map
    chi = r_delta[0] / (2.0 * np.sin(0.5 * theta_h))
    catalog = provisional._replace(chi=jnp.asarray([chi]))
    mass_map = _mass_map(pixels, nside=nside)

    maps = []
    for n_resolution in (4, 8, 16, 32):
        stencil = module.build_adaptive_lightcone_stencil_for_mass_map(
            mass_map,
            catalog,
            r_delta,
            AngularAssignmentParams(theta_map / 20.0, n_resolution),
        )
        maps.append(
            np.asarray(
                paint_lightcone_particle_count_map_sparse(
                    stencil,
                    catalog,
                    particle_mass_msun_h=1.0e10,
                    sample_chunk_size=1024,
                )
            )
        )

    reference = maps[-1]
    errors = [np.sum(np.abs(value - reference)) for value in maps[:-1]]
    assert errors[2] < errors[1] < errors[0]
    for value in maps:
        assert np.sum(value) == pytest.approx(1.0e4, rel=1.0e-10)


def test_supersampled_profile_is_stable_under_random_subpixel_shifts():
    hp = pytest.importorskip("healpy")
    module = _load_example_module()
    nside = 8
    pixels = np.arange(hp.nside2npix(nside), dtype=np.int64)
    pixel_vectors = np.stack(hp.pix2vec(nside, pixels), axis=-1)
    theta_map = np.sqrt(hp.nside2pixarea(nside))
    theta_h = 2.5 * theta_map
    theta0, phi0 = hp.pix2ang(nside, 100)
    provisional = LightconeHaloCatalog(
        unit_vector=jnp.asarray([[1.0, 0.0, 0.0]]),
        chi=jnp.asarray([1.0]),
        mass=jnp.asarray([1.0e14]),
        redshift=jnp.asarray([0.3]),
    )
    r_delta = module.nfw_support_rdelta_mpc_h(
        provisional,
        SimpleNamespace(cosmology=Cosmology()),
        module.NFWProfileParams(),
    )
    chi = r_delta[0] / (2.0 * np.sin(0.5 * theta_h))
    rng = np.random.default_rng(20260714)
    radial_moments = []

    for theta_shift, phi_shift in rng.uniform(-0.2, 0.2, size=(5, 2)):
        halo_vector = np.asarray(
            hp.ang2vec(
                theta0 + theta_shift * theta_map,
                phi0 + phi_shift * theta_map,
            )
        )
        catalog = provisional._replace(
            unit_vector=jnp.asarray(halo_vector[None, :]),
            chi=jnp.asarray([chi]),
        )
        stencil = module.build_adaptive_lightcone_stencil_for_mass_map(
            _mass_map(pixels, nside=nside),
            catalog,
            r_delta,
            AngularAssignmentParams(theta_map / 20.0, 16),
        )
        painted = np.asarray(
            paint_lightcone_particle_count_map_sparse(
                stencil,
                catalog,
                particle_mass_msun_h=1.0e10,
                sample_chunk_size=1024,
            )
        )
        chord = np.sqrt(
            np.maximum(2.0 * (1.0 - pixel_vectors @ halo_vector), 0.0)
        )
        radial_moments.append(float(np.sum(painted * chord) / np.sum(painted)))
        assert np.sum(painted) == pytest.approx(1.0e4, rel=1.0e-10)

    radial_moments = np.asarray(radial_moments)
    assert np.ptp(radial_moments) / np.mean(radial_moments) < 0.06


def test_write_manifest_contains_adaptive_scientific_diagnostics_and_provenance(tmp_path):
    module = _load_example_module()
    path = tmp_path / "painted_nfw_manifest.csv"
    result = _fake_segment_result(
        module,
        segment_index=0,
        mass_map_path=Path("massmap.seg000.fits"),
        output_npz=Path("painted_nfw.seg000.npz"),
        inclusive_upper=False,
    )
    args = _workflow_args()
    row = module.calibration_manifest_row(
        result,
        args,
        SimpleNamespace(particle_mass_msun_h=1.0),
        module.ExecutionProvenance(2, 3, "abc123"),
    )
    module.write_manifest(path, [row])

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert len(rows) == 1
    assert "nfw_profile_support" in reader.fieldnames
    assert "theta_resolution_rad" in reader.fieldnames
    assert "assigned_global_particle_count" in reader.fieldnames
    assert "retained_compact_particle_count" in reader.fieldnames
    assert "outside_compact_particle_count" in reader.fieldnames
    assert "nfw_truncation_width_fraction" not in reader.fieldnames
    assert rows[0]["mpi_rank_count"] == "2"
    assert rows[0]["git_commit"] == "abc123"
