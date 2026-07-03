from pathlib import Path

import numpy as np
import pytest

from geppetto.catalog import HaloCatalog, LightconeHaloCatalog
from geppetto.cosmology import rho_mean_comoving
from geppetto.io import (
    PinocchioCatalogError,
    healpix_pixel_area_sr,
    healpix_pixel_unit_vectors,
    pinocchio_plc_angle_unit_vectors,
    read_pinocchio_binary_lightcone_catalog,
    read_pinocchio_binary_lightcone_light_catalog,
    read_pinocchio_binary_snapshot_catalog,
    read_pinocchio_hubble_table,
    read_pinocchio_lightcone_catalog,
    read_pinocchio_lightcone_light_catalog,
    read_pinocchio_mass_function,
    read_pinocchio_mass_map_fits,
    read_pinocchio_mass_sheets,
    read_pinocchio_nz,
    read_pinocchio_parameter_file,
    read_pinocchio_snapshot_catalog,
    validate_tabulated_projected_profile_params,
)
from geppetto.profiles import TabulatedProjectedProfileParams


@pytest.mark.parametrize(
    ("theta_deg", "phi_deg", "expected"),
    [
        (0.0, 0.0, [1.0, 0.0, 0.0]),
        (0.0, 90.0, [0.0, 1.0, 0.0]),
        (90.0, 0.0, [0.0, 0.0, 1.0]),
        (-90.0, 0.0, [0.0, 0.0, -1.0]),
    ],
)
def test_pinocchio_plc_angle_unit_vectors_use_latitude_convention(
    theta_deg,
    phi_deg,
    expected,
):
    vectors = pinocchio_plc_angle_unit_vectors(
        np.array([theta_deg]),
        np.array([phi_deg]),
    )

    np.testing.assert_allclose(vectors[0], expected, atol=1.0e-12)


def test_validate_tabulated_projected_profile_params_accepts_valid_params():
    validate_tabulated_projected_profile_params(
        TabulatedProjectedProfileParams(
            x=np.array([0.0, 0.5, 1.0]),
            log_shape=np.array([0.0, -0.1, -0.2]),
        )
    )


@pytest.mark.parametrize(
    ("profile_params", "match"),
    [
        (
            TabulatedProjectedProfileParams(
                x=np.array([[0.0, 1.0]]),
                log_shape=np.array([0.0, 0.0]),
            ),
            "x must be one-dimensional",
        ),
        (
            TabulatedProjectedProfileParams(
                x=np.array([0.0, 1.0]),
                log_shape=np.array([[0.0, 0.0]]),
            ),
            "log_shape must be one-dimensional",
        ),
        (
            TabulatedProjectedProfileParams(
                x=np.array([0.0, 0.5, 1.0]),
                log_shape=np.array([0.0, 0.0]),
            ),
            "matching shapes",
        ),
        (
            TabulatedProjectedProfileParams(
                x=np.array([0.0]),
                log_shape=np.array([0.0]),
            ),
            "at least two",
        ),
        (
            TabulatedProjectedProfileParams(
                x=np.array([0.0, np.nan, 1.0]),
                log_shape=np.array([0.0, 0.0, 0.0]),
            ),
            "x values must be finite",
        ),
        (
            TabulatedProjectedProfileParams(
                x=np.array([0.0, 0.5, 1.0]),
                log_shape=np.array([0.0, np.inf, 0.0]),
            ),
            "log_shape values must be finite",
        ),
        (
            TabulatedProjectedProfileParams(
                x=np.array([0.0, 0.5, 0.5, 1.0]),
                log_shape=np.array([0.0, 0.0, 0.0, 0.0]),
            ),
            "strictly increasing",
        ),
        (
            TabulatedProjectedProfileParams(
                x=np.array([0.0, 0.8, 0.7, 1.0]),
                log_shape=np.array([0.0, 0.0, 0.0, 0.0]),
            ),
            "strictly increasing",
        ),
        (
            TabulatedProjectedProfileParams(
                x=np.array([0.1, 0.5, 1.0]),
                log_shape=np.array([0.0, 0.0, 0.0]),
            ),
            r"\[0, 1\]",
        ),
        (
            TabulatedProjectedProfileParams(
                x=np.array([0.0, 0.5, 0.9]),
                log_shape=np.array([0.0, 0.0, 0.0]),
            ),
            r"\[0, 1\]",
        ),
    ],
)
def test_validate_tabulated_projected_profile_params_rejects_invalid_params(
    profile_params, match
):
    with pytest.raises(PinocchioCatalogError, match=match):
        validate_tabulated_projected_profile_params(profile_params)


def test_read_pinocchio_snapshot_catalog_and_convert(tmp_path):
    path = tmp_path / "pinocchio.0.0000.demo.catalog.out"
    path.write_text(
        "\n".join(
            [
                "# Group catalog for redshift 0.000000 and minimal mass of 10 particles",
                "# columns omitted in this fixture",
                "101 1.0e13 1 2 3 4 5 6 10 20 30 20",
                "102 2.5e13 7 8 9 260 -1 257 -10 -20 -30 50",
            ]
        ),
        encoding="utf-8",
    )

    catalog = read_pinocchio_snapshot_catalog(path)

    assert len(catalog) == 2
    assert catalog.redshift == 0.0
    assert catalog.group_ids.tolist() == [101, 102]
    np.testing.assert_allclose(catalog.masses_msun_h, [1.0e13, 2.5e13])
    np.testing.assert_allclose(catalog.velocities_km_s[1], [-10.0, -20.0, -30.0])
    assert catalog.n_particles.tolist() == [20, 50]

    halo_catalog = catalog.to_halo_catalog(wrap_box_size_mpc_h=256.0)

    assert isinstance(halo_catalog, HaloCatalog)
    np.testing.assert_allclose(np.asarray(halo_catalog.position[1]), [4.0, 255.0, 1.0])
    np.testing.assert_allclose(np.asarray(halo_catalog.redshift), [0.0, 0.0])


def test_snapshot_catalog_can_use_initial_positions_and_override_redshift(tmp_path):
    path = tmp_path / "snapshot.out"
    path.write_text("101 1.0e13 260 -1 257 4 5 6 10 20 30 20\n", encoding="utf-8")

    catalog = read_pinocchio_snapshot_catalog(path)
    halo_catalog = catalog.to_halo_catalog(
        position="initial",
        redshift=0.5,
        wrap_box_size_mpc_h=256.0,
    )

    np.testing.assert_allclose(np.asarray(halo_catalog.position[0]), [4.0, 255.0, 1.0])
    np.testing.assert_allclose(np.asarray(halo_catalog.redshift), [0.5])


def test_snapshot_conversion_requires_redshift_without_header(tmp_path):
    path = tmp_path / "snapshot.out"
    path.write_text("101 1.0e13 1 2 3 4 5 6 10 20 30 20\n", encoding="utf-8")

    catalog = read_pinocchio_snapshot_catalog(path)

    with pytest.raises(PinocchioCatalogError, match="redshift"):
        catalog.to_halo_catalog()


def test_snapshot_reader_rejects_wrong_column_count(tmp_path):
    path = tmp_path / "invalid.catalog.out"
    path.write_text("1 2 3\n", encoding="utf-8")

    with pytest.raises(PinocchioCatalogError, match="12 columns"):
        read_pinocchio_snapshot_catalog(path)


def test_read_pinocchio_binary_snapshot_catalog_and_auto_detect(tmp_path):
    path = tmp_path / "pinocchio.0.5000.demo.catalog.out"
    dtype = np.dtype(
        [
            ("name", np.uint64),
            ("Mass", np.float32),
            ("pos", np.float32, 3),
            ("vel", np.float32, 3),
            ("posin", np.float32, 3),
            ("npart", np.int32),
            ("pad", np.int32),
        ]
    )
    data = np.array(
        [
            (
                101,
                1.0e13,
                [4.0, 5.0, 6.0],
                [10.0, 20.0, 30.0],
                [1.0, 2.0, 3.0],
                20,
                0,
            ),
            (
                102,
                2.5e13,
                [260.0, -1.0, 257.0],
                [-10.0, -20.0, -30.0],
                [7.0, 8.0, 9.0],
                50,
                0,
            ),
        ],
        dtype=dtype,
    )
    _write_new_binary_catalog_file(path, data)

    catalog = read_pinocchio_snapshot_catalog(path)
    explicit = read_pinocchio_binary_snapshot_catalog(path)

    assert catalog.redshift == 0.5
    assert explicit.redshift == 0.5
    assert catalog.group_ids.tolist() == [101, 102]
    assert catalog.n_particles.tolist() == [20, 50]
    np.testing.assert_allclose(catalog.initial_positions_mpc_h[0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(catalog.final_positions_mpc_h[1], [260.0, -1.0, 257.0])
    np.testing.assert_allclose(catalog.velocities_km_s[0], [10.0, 20.0, 30.0])


def test_read_pinocchio_binary_snapshot_catalog_split_files(tmp_path):
    base = tmp_path / "pinocchio.0.0000.demo.catalog.out"
    dtype = np.dtype(
        [
            ("name", np.uint64),
            ("Mass", np.float32),
            ("pos", np.float32, 3),
            ("vel", np.float32, 3),
            ("posin", np.float32, 3),
            ("npart", np.int32),
            ("pad", np.int32),
        ]
    )
    first = np.array(
        [(101, 1.0e13, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [4.0, 5.0, 6.0], 20, 0)],
        dtype=dtype,
    )
    second = np.array(
        [(102, 2.0e13, [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [7.0, 8.0, 9.0], 30, 0)],
        dtype=dtype,
    )
    _write_new_binary_catalog_file(Path(f"{base}.0"), first)
    _write_new_binary_catalog_file(Path(f"{base}.1"), second)

    catalog = read_pinocchio_binary_snapshot_catalog(base)

    assert len(catalog) == 2
    assert catalog.group_ids.tolist() == [101, 102]
    np.testing.assert_allclose(catalog.final_positions_mpc_h[:, 0], [1.0, 2.0])


def test_read_pinocchio_lightcone_catalog_uses_plc_angles_for_unit_vectors(tmp_path):
    path = tmp_path / "pinocchio.demo.plc.out"
    path.write_text(
        "\n".join(
            [
                "# Group catalog on the Past Light Cone",
                "11 0.10 3 4 0 10 20 30 1.0e13 0.0 90.0 100 0.101",
                "12 0.20 0 0 5 -1 -2 -3 2.0e13 90.00 0.0 -50 0.199",
            ]
        ),
        encoding="utf-8",
    )

    catalog = read_pinocchio_lightcone_catalog(path)

    assert len(catalog) == 2
    assert catalog.group_ids.tolist() == [11, 12]
    np.testing.assert_allclose(catalog.chi_mpc_h, [5.0, 5.0])
    np.testing.assert_allclose(catalog.unit_vectors[0], [0.0, 1.0, 0.0], atol=1.0e-12)
    np.testing.assert_allclose(catalog.cartesian_unit_vectors[0], [0.6, 0.8, 0.0])

    lightcone = catalog.to_lightcone_catalog()
    observed = catalog.to_lightcone_catalog(redshift="observed")

    assert isinstance(lightcone, LightconeHaloCatalog)
    np.testing.assert_allclose(
        np.asarray(lightcone.unit_vector[1]),
        [0.0, 0.0, 1.0],
        atol=1.0e-12,
    )
    np.testing.assert_allclose(np.asarray(lightcone.redshift), [0.10, 0.20])
    np.testing.assert_allclose(np.asarray(observed.redshift), [0.101, 0.199])


def test_lightcone_reader_rejects_unknown_redshift_mode(tmp_path):
    path = tmp_path / "pinocchio.demo.plc.out"
    path.write_text("11 0.10 3 4 0 10 20 30 1.0e13 0.0 90.0 100 0.101\n")

    catalog = read_pinocchio_lightcone_catalog(path)

    with pytest.raises(PinocchioCatalogError, match="redshift"):
        catalog.to_lightcone_catalog(redshift="cosmological")


def test_read_pinocchio_binary_lightcone_catalog_and_auto_detect(tmp_path):
    path = tmp_path / "pinocchio.demo.plc.out"
    dtype = np.dtype(
        [
            ("name", np.uint64),
            ("truez", np.float32),
            ("pos", np.float32, 3),
            ("vel", np.float32, 3),
            ("Mass", np.float32),
            ("theta", np.float32),
            ("phi", np.float32),
            ("vlos", np.float32),
            ("obsz", np.float32),
        ]
    )
    data = np.array(
        [
            (
                11,
                0.10,
                [3.0, 4.0, 0.0],
                [10.0, 20.0, 30.0],
                1.0e13,
                0.0,
                90.0,
                100.0,
                0.101,
            ),
            (
                12,
                0.20,
                [0.0, 0.0, 5.0],
                [-1.0, -2.0, -3.0],
                2.0e13,
                90.0,
                0.0,
                -50.0,
                0.199,
            ),
        ],
        dtype=dtype,
    )
    _write_new_binary_plc_file(path, data)

    catalog = read_pinocchio_lightcone_catalog(path)
    explicit = read_pinocchio_binary_lightcone_catalog(path)

    assert len(catalog) == 2
    assert explicit.group_ids.tolist() == [11, 12]
    np.testing.assert_allclose(catalog.chi_mpc_h, [5.0, 5.0])
    np.testing.assert_allclose(catalog.unit_vectors[0], [0.0, 1.0, 0.0], atol=1.0e-12)
    np.testing.assert_allclose(catalog.cartesian_unit_vectors[0], [0.6, 0.8, 0.0])
    np.testing.assert_allclose(catalog.observed_redshift, [0.101, 0.199], rtol=1.0e-6)


def test_binary_lightcone_reader_rejects_light_output_without_positions(tmp_path):
    path = tmp_path / "pinocchio.demo.plc.out"
    path.write_bytes(np.array([4, 32, 4], dtype=np.int32).tobytes())

    with pytest.raises(PinocchioCatalogError, match="lightcone_light_catalog"):
        read_pinocchio_binary_lightcone_catalog(path)


def test_read_pinocchio_lightcone_light_ascii_and_convert(tmp_path):
    path = tmp_path / "pinocchio.demo.plc.out"
    path.write_text(
        "\n".join(
            [
                "# Light PLC catalog",
                "11 0.10 1.0e13 0.0 0.0 0.101",
                "12 0.20 2.0e13 0.0 90.0 0.199",
            ]
        ),
        encoding="utf-8",
    )
    hubble_path = tmp_path / "hubble.dat"
    hubble_path.write_text("0.0 1.0\n0.5 1.0\n", encoding="utf-8")

    distance = read_pinocchio_hubble_table(hubble_path, n_grid=16)
    catalog = read_pinocchio_lightcone_light_catalog(path)

    assert len(catalog) == 2
    assert catalog.group_ids.tolist() == [11, 12]
    np.testing.assert_allclose(catalog.masses_msun_h, [1.0e13, 2.0e13])
    np.testing.assert_allclose(catalog.unit_vectors[0], [1.0, 0.0, 0.0], atol=1.0e-12)
    np.testing.assert_allclose(catalog.unit_vectors[1], [0.0, 1.0, 0.0], atol=1.0e-12)

    lightcone = catalog.to_lightcone_catalog(distance)
    observed = catalog.to_lightcone_catalog(distance, redshift="observed")

    assert isinstance(lightcone, LightconeHaloCatalog)
    np.testing.assert_allclose(np.asarray(lightcone.chi), [299.792458, 599.584916], rtol=1.0e-6)
    np.testing.assert_allclose(np.asarray(lightcone.redshift), [0.10, 0.20])
    np.testing.assert_allclose(
        np.asarray(observed.chi),
        [302.79038258, 596.58699142],
        rtol=1.0e-6,
    )
    np.testing.assert_allclose(np.asarray(observed.redshift), [0.101, 0.199])


def test_read_pinocchio_binary_lightcone_light_catalog_and_auto_detect(tmp_path):
    path = tmp_path / "pinocchio.demo.plc.out"
    dtype = np.dtype(
        [
            ("name", np.uint64),
            ("truez", np.float32),
            ("Mass", np.float32),
            ("theta", np.float32),
            ("phi", np.float32),
            ("obsz", np.float32),
            ("pad", np.float32),
        ]
    )
    data = np.array(
        [
            (11, 0.10, 1.0e13, 0.0, 0.0, 0.101, 0.0),
            (12, 0.20, 2.0e13, 0.0, 90.0, 0.199, 0.0),
        ],
        dtype=dtype,
    )
    _write_new_binary_plc_file(path, data)

    catalog = read_pinocchio_lightcone_light_catalog(path)
    explicit = read_pinocchio_binary_lightcone_light_catalog(path)

    assert len(catalog) == 2
    assert explicit.group_ids.tolist() == [11, 12]
    np.testing.assert_allclose(catalog.true_redshift, [0.10, 0.20], rtol=1.0e-6)
    np.testing.assert_allclose(catalog.theta_deg, [0.0, 0.0])
    np.testing.assert_allclose(catalog.phi_deg, [0.0, 90.0])
    np.testing.assert_allclose(catalog.observed_redshift, [0.101, 0.199], rtol=1.0e-6)


def test_lightcone_light_conversion_rejects_unknown_redshift_mode(tmp_path):
    path = tmp_path / "pinocchio.demo.plc.out"
    path.write_text("11 0.10 1.0e13 90.0 0.0 0.101\n", encoding="utf-8")
    hubble_path = tmp_path / "hubble.dat"
    hubble_path.write_text("0.0 1.0\n0.5 1.0\n", encoding="utf-8")

    distance = read_pinocchio_hubble_table(hubble_path, n_grid=16)
    catalog = read_pinocchio_lightcone_light_catalog(path)

    with pytest.raises(PinocchioCatalogError, match="redshift"):
        catalog.to_lightcone_catalog(distance, redshift="cosmological")


def test_read_pinocchio_hubble_table_validation(tmp_path):
    path = tmp_path / "hubble_without_z0.dat"
    path.write_text("0.5 2.0\n1.0 3.0\n", encoding="utf-8")

    distance = read_pinocchio_hubble_table(path, n_grid=8)

    np.testing.assert_allclose(distance.redshift[0], 0.0)
    np.testing.assert_allclose(distance.e_z[0], 1.0)
    np.testing.assert_allclose(distance.chi_mpc_h([0.0]), [0.0])

    bad_z0 = tmp_path / "bad_z0.dat"
    bad_z0.write_text("0.0 2.0\n1.0 3.0\n", encoding="utf-8")
    with pytest.raises(PinocchioCatalogError, match=r"E\(0\) = 1"):
        read_pinocchio_hubble_table(bad_z0)

    bad_e = tmp_path / "bad_e.dat"
    bad_e.write_text("0.0 1.0\n1.0 -3.0\n", encoding="utf-8")
    with pytest.raises(PinocchioCatalogError, match="positive"):
        read_pinocchio_hubble_table(bad_e)

    with pytest.raises(PinocchioCatalogError, match="range"):
        distance.chi_mpc_h([2.0])


def test_read_pinocchio_parameter_file_particle_mass_with_h100_box(tmp_path):
    path = tmp_path / "parameter_file"
    path.write_text(
        "\n".join(
            [
                "# run properties",
                "RunFlag                demo",
                "BoxSize                256          % Mpc/h because BoxInH100 is present",
                "BoxInH100",
                "GridSize               128",
                "Omega0                 0.3110",
                "Hubble100              0.6766",
            ]
        ),
        encoding="utf-8",
    )

    metadata = read_pinocchio_parameter_file(path)

    expected_particle_mass = rho_mean_comoving(metadata.cosmology) * 256.0**3 / 128.0**3
    assert metadata.run_flag == "demo"
    assert metadata.box_in_h100
    assert metadata.grid_size == 128
    np.testing.assert_allclose(metadata.box_size_mpc_h, 256.0)
    np.testing.assert_allclose(metadata.cosmology.omega_m, 0.3110)
    np.testing.assert_allclose(metadata.cosmology.h, 0.6766)
    np.testing.assert_allclose(metadata.particle_mass_msun_h, expected_particle_mass)


def test_read_pinocchio_parameter_file_converts_physical_box_to_mpc_h(tmp_path):
    path = tmp_path / "parameter_file"
    path.write_text(
        "\n".join(
            [
                "BoxSize                100",
                "GridSize               10",
                "Omega0                 0.3",
                "Hubble100              0.7",
            ]
        ),
        encoding="utf-8",
    )

    metadata = read_pinocchio_parameter_file(path)

    expected_box_size = 70.0
    expected_particle_mass = (
        rho_mean_comoving(metadata.cosmology) * expected_box_size**3 / 10.0**3
    )
    assert not metadata.box_in_h100
    np.testing.assert_allclose(metadata.box_size_mpc_h, expected_box_size)
    np.testing.assert_allclose(metadata.particle_mass_msun_h, expected_particle_mass)


def test_read_pinocchio_parameter_file_validation(tmp_path):
    missing = tmp_path / "missing_parameter_file"
    missing.write_text("BoxSize 100\nGridSize 10\nOmega0 0.3\n", encoding="utf-8")
    with pytest.raises(PinocchioCatalogError, match="Hubble100"):
        read_pinocchio_parameter_file(missing)

    invalid = tmp_path / "invalid_parameter_file"
    invalid.write_text(
        "BoxSize 100\nGridSize 10.5\nOmega0 0.3\nHubble100 0.7\n",
        encoding="utf-8",
    )
    with pytest.raises(PinocchioCatalogError, match="GridSize"):
        read_pinocchio_parameter_file(invalid)

    nonpositive = tmp_path / "nonpositive_parameter_file"
    nonpositive.write_text(
        "BoxSize -100\nGridSize 10\nOmega0 0.3\nHubble100 0.7\n",
        encoding="utf-8",
    )
    with pytest.raises(PinocchioCatalogError, match="BoxSize"):
        read_pinocchio_parameter_file(nonpositive)


def test_healpix_pixel_area_sr():
    np.testing.assert_allclose(healpix_pixel_area_sr(2), 4.0 * np.pi / 48.0)

    with pytest.raises(PinocchioCatalogError, match="power of two"):
        healpix_pixel_area_sr(3)


def test_healpix_pixel_unit_vectors_preserve_order():
    hp = pytest.importorskip("healpy")
    pixels = np.array([3, 1, 7], dtype=np.int64)

    vectors = healpix_pixel_unit_vectors(1, pixels)
    expected = np.stack(hp.pix2vec(1, pixels, nest=False), axis=-1)

    assert vectors.shape == (3, 3)
    np.testing.assert_allclose(vectors, expected)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), np.ones(3))


def test_healpix_pixel_unit_vectors_full_sky_and_validation():
    pytest.importorskip("healpy")

    vectors = healpix_pixel_unit_vectors(1)

    assert vectors.shape == (12, 3)
    with pytest.raises(PinocchioCatalogError, match="pixels"):
        healpix_pixel_unit_vectors(1, np.array([12]))


def test_read_pinocchio_auxiliary_ascii_tables(tmp_path):
    sheets_path = tmp_path / "pinocchio.demo.sheets.out"
    sheets_path.write_text(
        "# id z_hi z_lo ...\n"
        "0 0.10 0.05 0.05 432.5 219.0 213.5 0.00468 393.3 208.3 7.0e7\n"
        "1 0.05 0.00 0.05 219.0 0.0 219.0 0.00456 208.3 0.0 1.0e7\n",
        encoding="utf-8",
    )
    nz_path = tmp_path / "pinocchio.demo.nz.out"
    nz_path.write_text("0.000 0.050 8109 0.393 4814.04\n", encoding="utf-8")
    mf_path = tmp_path / "pinocchio.0.0000.demo.mf.out"
    mf_path.write_text(
        "# Mass function for redshift 0.000000\n"
        "7.2e12 6.0e-17 6.2e-17 5.9e-17 1409 8.5e-17 0.736\n",
        encoding="utf-8",
    )

    sheets = read_pinocchio_mass_sheets(sheets_path)
    nz = read_pinocchio_nz(nz_path)
    mass_function = read_pinocchio_mass_function(mf_path)

    assert sheets.sheet_ids.tolist() == [0, 1]
    np.testing.assert_allclose(sheets.delta_chi_mpc_h, [213.5, 219.0])
    assert nz.counts.tolist() == [8109]
    np.testing.assert_allclose(nz.predicted_counts, [4814.04])
    assert mass_function.redshift == 0.0
    assert mass_function.halo_counts.tolist() == [1409]
    np.testing.assert_allclose(mass_function.peak_height_nu, [0.736])


def test_read_pinocchio_mass_map_fits(tmp_path):
    fits = pytest.importorskip("astropy.io.fits")
    path = tmp_path / "pinocchio.demo.massmap.seg000.fits"
    table = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="PIXEL", format="1J", array=np.array([0, 5], dtype=np.int32)),
            fits.Column(
                name="TEMPERATURE",
                format="1D",
                array=np.array([12.5, 3.25], dtype=np.float64),
            ),
        ],
        name="HEALPIX",
    )
    table.header["PIXTYPE"] = "HEALPIX"
    table.header["ORDERING"] = "RING"
    table.header["NSIDE"] = 8
    table.header["INDXSCHM"] = "EXPLICIT"
    table.header["FIRSTPIX"] = 0
    table.header["LASTPIX"] = 767
    table.header["APERTURE"] = 180.0
    table.header["SELTYPE"] = "PIXEL_CAP_PREFIX"
    table.header["AXISV1"] = 0.0
    table.header["AXISV2"] = 0.0
    table.header["AXISV3"] = 1.0
    table.header["FILTER"] = "ZPLC>ZACC"
    table.header["ZF_CONS"] = 20
    table.header["ZF_EXCL"] = 5
    table.header["ZF_INCL"] = 15
    table.header["ZF_FEXCL"] = 0.25
    fits.HDUList([fits.PrimaryHDU(), table]).writeto(path)

    mass_map = read_pinocchio_mass_map_fits(path)

    assert len(mass_map) == 2
    assert mass_map.nside == 8
    assert mass_map.ordering == "RING"
    assert mass_map.index_scheme == "EXPLICIT"
    assert mass_map.selection_type == "PIXEL_CAP_PREFIX"
    np.testing.assert_array_equal(mass_map.pixel, [0, 5])
    np.testing.assert_allclose(mass_map.temperature, [12.5, 3.25])
    np.testing.assert_allclose(mass_map.axis_vector, [0.0, 0.0, 1.0])
    assert mass_map.filter_name == "ZPLC>ZACC"
    assert mass_map.filter_considered == 20
    assert mass_map.filter_excluded == 5
    assert mass_map.filter_included == 15
    assert mass_map.filter_excluded_fraction == 0.25


def test_mass_map_reader_rejects_missing_columns(tmp_path):
    fits = pytest.importorskip("astropy.io.fits")
    path = tmp_path / "invalid.fits"
    table = fits.BinTableHDU.from_columns(
        [fits.Column(name="PIXEL", format="1J", array=np.array([0], dtype=np.int32))],
        name="HEALPIX",
    )
    table.header["ORDERING"] = "RING"
    table.header["NSIDE"] = 8
    fits.HDUList([fits.PrimaryHDU(), table]).writeto(path)

    with pytest.raises(PinocchioCatalogError, match="PIXEL and TEMPERATURE"):
        read_pinocchio_mass_map_fits(path)


def _write_new_binary_catalog_file(path, data):
    record_length = data.dtype.itemsize
    payload = b"".join(
        [
            np.array([8, 1, record_length, 8], dtype=np.int32).tobytes(),
            np.array([4, len(data), 4], dtype=np.int32).tobytes(),
            np.array([record_length * len(data)], dtype=np.int32).tobytes(),
            data.tobytes(),
            np.array([record_length * len(data)], dtype=np.int32).tobytes(),
        ]
    )
    path.write_bytes(payload)


def _write_new_binary_plc_file(path, data):
    record_length = data.dtype.itemsize
    payload = b"".join(
        [
            np.array([4, record_length, 4], dtype=np.int32).tobytes(),
            np.array([4, len(data), 4], dtype=np.int32).tobytes(),
            np.array([record_length * len(data)], dtype=np.int32).tobytes(),
            data.tobytes(),
            np.array([record_length * len(data)], dtype=np.int32).tobytes(),
        ]
    )
    path.write_bytes(payload)
