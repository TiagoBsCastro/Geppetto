"""I/O adapters for PINOCCHIO catalogues.

The differentiable core should not depend on FITS/HDF5/ASCII readers. Keep all
file parsing here, convert columns to JAX arrays at the boundary, and pass the
resulting catalogue objects to ``geppetto.painters``.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any, Literal

import jax.numpy as jnp
import numpy as np

from geppetto.catalog import (
    HaloCatalog,
    LightconeHaloCatalog,
    LightconeSparseStencil,
    unit_vectors_from_angles,
)
from geppetto.cosmology import Cosmology, rho_mean_comoving
from geppetto.profiles import TabulatedProjectedProfileParams


class PinocchioCatalogError(ValueError):
    """Raised when a PINOCCHIO output file cannot be parsed or validated."""


PathLike = str | Path
CatalogFormat = Literal["auto", "ascii", "binary"]
PositionMode = Literal["initial", "final"]
LightconeRedshiftMode = Literal["true", "observed"]
_C_LIGHT_KM_S = 299_792.458


def pinocchio_plc_angle_unit_vectors(theta_deg: np.ndarray, phi_deg: np.ndarray) -> np.ndarray:
    """Return unit vectors from PINOCCHIO PLC angular columns.

    PINOCCHIO PLC ``theta`` is latitude-like in degrees, measured from the PLC
    equator toward the PLC axis. It is not the HEALPix colatitude. ``phi`` is
    the longitude in the same internal PLC basis used by PINOCCHIO mass-map
    HEALPix pixels.
    """

    theta = np.deg2rad(np.asarray(theta_deg, dtype=np.float64))
    phi = np.deg2rad(np.asarray(phi_deg, dtype=np.float64))
    if theta.shape != phi.shape:
        raise PinocchioCatalogError("theta_deg and phi_deg must have matching shapes")
    if not np.all(np.isfinite(theta)) or not np.all(np.isfinite(phi)):
        raise PinocchioCatalogError("PINOCCHIO PLC angles must be finite")

    cos_theta = np.cos(theta)
    return np.stack(
        [
            cos_theta * np.cos(phi),
            cos_theta * np.sin(phi),
            np.sin(theta),
        ],
        axis=-1,
    )


@dataclass(frozen=True)
class PinocchioSnapshotCatalog:
    """Raw PINOCCHIO snapshot halo catalogue.

    Masses are ``Msun/h``; positions are comoving ``Mpc/h``; velocities are
    ``km/s``. ``initial_positions_mpc_h`` and ``n_particles`` can be ``None``
    for legacy/light binary outputs that did not store those fields.
    """

    group_ids: np.ndarray
    masses_msun_h: np.ndarray
    initial_positions_mpc_h: np.ndarray | None
    final_positions_mpc_h: np.ndarray
    velocities_km_s: np.ndarray
    n_particles: np.ndarray | None
    source: Path
    redshift: float | None = None

    def __len__(self) -> int:
        return int(self.masses_msun_h.shape[0])

    def to_halo_catalog(
        self,
        *,
        position: PositionMode = "final",
        redshift: float | None = None,
        wrap_box_size_mpc_h: float | None = None,
    ) -> HaloCatalog:
        """Convert to a GEPPETTO box catalogue.

        Parameters
        ----------
        position:
            ``"final"`` for Eulerian positions or ``"initial"`` for Lagrangian
            positions, both in comoving ``Mpc/h``.
        redshift:
            Snapshot redshift. If omitted, the value parsed from the file header
            is used. A value is required if the header does not contain one.
        wrap_box_size_mpc_h:
            Optional periodic box size in comoving ``Mpc/h``. When supplied,
            selected positions are wrapped into ``[0, box_size)``.
        """

        if position == "final":
            positions = self.final_positions_mpc_h
        elif position == "initial":
            if self.initial_positions_mpc_h is None:
                raise PinocchioCatalogError(
                    "initial positions are not available in this PINOCCHIO catalog"
                )
            positions = self.initial_positions_mpc_h
        else:
            raise PinocchioCatalogError("position must be 'initial' or 'final'")

        redshift_value = self.redshift if redshift is None else redshift
        if redshift_value is None:
            raise PinocchioCatalogError(
                "redshift must be provided when it cannot be parsed from the catalog header"
            )

        if wrap_box_size_mpc_h is not None:
            positions = _wrap_positions(positions, wrap_box_size_mpc_h)

        return HaloCatalog(
            position=jnp.asarray(positions),
            mass=jnp.asarray(self.masses_msun_h),
            redshift=jnp.full((len(self),), float(redshift_value)),
        )


@dataclass(frozen=True)
class PinocchioLightconeCatalog:
    """Raw PINOCCHIO past-light-cone halo catalogue.

    Positions are comoving ``Mpc/h``; masses are ``Msun/h``; velocities are
    ``km/s``. PINOCCHIO angle columns are latitude-like ``theta`` and longitude
    ``phi`` in degrees, expressed in the internal PLC basis used by PINOCCHIO
    mass-map HEALPix pixels.
    """

    group_ids: np.ndarray
    true_redshift: np.ndarray
    positions_mpc_h: np.ndarray
    velocities_km_s: np.ndarray
    masses_msun_h: np.ndarray
    theta_deg: np.ndarray
    phi_deg: np.ndarray
    los_velocity_km_s: np.ndarray
    observed_redshift: np.ndarray
    source: Path

    def __len__(self) -> int:
        return int(self.masses_msun_h.shape[0])

    @property
    def chi_mpc_h(self) -> np.ndarray:
        """Comoving radial distances in ``Mpc/h``."""

        return np.linalg.norm(self.positions_mpc_h, axis=1)

    @property
    def unit_vectors(self) -> np.ndarray:
        """Unit vectors from PINOCCHIO PLC angular columns."""

        return pinocchio_plc_angle_unit_vectors(self.theta_deg, self.phi_deg)

    @property
    def cartesian_unit_vectors(self) -> np.ndarray:
        """Unit vectors inferred from Cartesian comoving positions."""

        chi = self.chi_mpc_h
        if np.any(chi <= 0.0):
            raise PinocchioCatalogError("PLC catalog contains zero-distance halo positions")
        return self.positions_mpc_h / chi[:, None]

    def to_lightcone_catalog(
        self, *, redshift: LightconeRedshiftMode = "true"
    ) -> LightconeHaloCatalog:
        """Convert to a GEPPETTO lightcone catalogue.

        Parameters
        ----------
        redshift:
            ``"true"`` uses PINOCCHIO true redshift; ``"observed"`` uses the
            observed redshift including the line-of-sight peculiar velocity.
        """

        if redshift == "true":
            redshift_values = self.true_redshift
        elif redshift == "observed":
            redshift_values = self.observed_redshift
        else:
            raise PinocchioCatalogError("redshift must be 'true' or 'observed'")

        return LightconeHaloCatalog(
            unit_vector=jnp.asarray(self.unit_vectors),
            chi=jnp.asarray(self.chi_mpc_h),
            mass=jnp.asarray(self.masses_msun_h),
            redshift=jnp.asarray(redshift_values),
        )


@dataclass(frozen=True)
class PinocchioDistanceInterpolator:
    """Comoving-distance convention derived from a PINOCCHIO Hubble table.

    ``redshift`` and ``e_z`` preserve the sorted table values, where ``e_z`` is
    PINOCCHIO's dimensionless ``H(z) / H0``. ``chi_grid_mpc_h`` stores
    ``c / 100 * integral dz / E(z)`` in comoving ``Mpc/h``. This is an I/O
    adapter convention, not part of the differentiable core.
    """

    redshift: np.ndarray
    e_z: np.ndarray
    integration_redshift: np.ndarray
    chi_grid_mpc_h: np.ndarray
    source: Path

    def chi_mpc_h(self, redshift: np.ndarray | float) -> np.ndarray:
        """Interpolate comoving radial distance in ``Mpc/h`` at redshift."""

        values = np.asarray(redshift, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise PinocchioCatalogError("redshift values must be finite")
        if np.any(values < 0.0):
            raise PinocchioCatalogError("redshift values must be non-negative")
        if values.size:
            z_max = float(self.integration_redshift[-1])
            tolerance = 1.0e-10 * max(1.0, z_max)
            if float(np.max(values)) > z_max + tolerance:
                raise PinocchioCatalogError(
                    f"redshift exceeds Hubble table range {z_max:g}: {self.source}"
                )
        return np.interp(values, self.integration_redshift, self.chi_grid_mpc_h)


@dataclass(frozen=True)
class PinocchioLightconeLightCatalog:
    """Raw PINOCCHIO light past-light-cone halo catalogue.

    Light PLC output stores group ID, true redshift, mass in ``Msun/h``,
    latitude-like ``theta`` and longitude ``phi`` in degrees, and observed
    redshift. It does not store Cartesian positions or radial distances.
    Conversion to ``LightconeHaloCatalog`` therefore requires an explicit
    ``PinocchioDistanceInterpolator`` built from the PINOCCHIO cosmology/Hubble
    table, returning distances in comoving ``Mpc/h``.
    """

    group_ids: np.ndarray
    true_redshift: np.ndarray
    masses_msun_h: np.ndarray
    theta_deg: np.ndarray
    phi_deg: np.ndarray
    observed_redshift: np.ndarray
    source: Path

    def __len__(self) -> int:
        return int(self.masses_msun_h.shape[0])

    @property
    def unit_vectors(self) -> np.ndarray:
        """Unit vectors from PINOCCHIO angular columns."""

        return pinocchio_plc_angle_unit_vectors(self.theta_deg, self.phi_deg)

    def to_lightcone_catalog(
        self,
        distance_interpolator: PinocchioDistanceInterpolator,
        *,
        redshift: LightconeRedshiftMode = "true",
    ) -> LightconeHaloCatalog:
        """Convert to a GEPPETTO lightcone catalogue.

        Parameters
        ----------
        distance_interpolator:
            Distance convention from ``read_pinocchio_hubble_table``. Distances
            are interpolated in comoving ``Mpc/h``.
        redshift:
            ``"true"`` uses PINOCCHIO true redshift for both distance and output
            redshift; ``"observed"`` uses the observed redshift including the
            line-of-sight peculiar velocity.
        """

        if redshift == "true":
            redshift_values = self.true_redshift
        elif redshift == "observed":
            redshift_values = self.observed_redshift
        else:
            raise PinocchioCatalogError("redshift must be 'true' or 'observed'")

        return LightconeHaloCatalog(
            unit_vector=jnp.asarray(self.unit_vectors),
            chi=jnp.asarray(distance_interpolator.chi_mpc_h(redshift_values)),
            mass=jnp.asarray(self.masses_msun_h),
            redshift=jnp.asarray(redshift_values),
        )


@dataclass(frozen=True)
class PinocchioMassSheetTable:
    """PINOCCHIO mass-sheet table from ``*.sheets.out``."""

    sheet_ids: np.ndarray
    z_hi: np.ndarray
    z_lo: np.ndarray
    delta_z: np.ndarray
    chi_hi_mpc_h: np.ndarray
    chi_lo_mpc_h: np.ndarray
    delta_chi_mpc_h: np.ndarray
    inv_delta_chi_h_mpc: np.ndarray
    da_hi_mpc_h: np.ndarray
    da_lo_mpc_h: np.ndarray
    chi3_diff_mpc_h3: np.ndarray
    source: Path

    def __len__(self) -> int:
        return int(self.sheet_ids.shape[0])


@dataclass(frozen=True)
class PinocchioNzTable:
    """PINOCCHIO PLC number-count table from ``*.nz.out``."""

    z_low: np.ndarray
    z_high: np.ndarray
    counts: np.ndarray
    number_per_square_degree: np.ndarray
    predicted_counts: np.ndarray
    source: Path

    def __len__(self) -> int:
        return int(self.counts.shape[0])


@dataclass(frozen=True)
class PinocchioMassFunction:
    """PINOCCHIO mass-function table from ``*.mf.out``."""

    mass_msun_h: np.ndarray
    number_density: np.ndarray
    number_density_plus_1sigma: np.ndarray
    number_density_minus_1sigma: np.ndarray
    halo_counts: np.ndarray
    analytic_number_density: np.ndarray
    peak_height_nu: np.ndarray
    source: Path
    redshift: float | None = None

    def __len__(self) -> int:
        return int(self.mass_msun_h.shape[0])


@dataclass(frozen=True)
class PinocchioMassMap:
    """PINOCCHIO HEALPix mass map from ``*.massmap.seg*.fits``.

    The FITS binary table is preserved in explicit form. ``pixel`` contains
    HEALPix pixel indices in the ordering declared by the header, and
    ``temperature`` contains the map values as written by PINOCCHIO.
    """

    pixel: np.ndarray
    temperature: np.ndarray
    source: Path
    header: dict[str, Any]
    nside: int
    ordering: str
    index_scheme: str | None
    first_pixel: int | None
    last_pixel: int | None
    aperture_deg: float | None
    selection_type: str | None
    axis_vector: np.ndarray | None
    filter_name: str | None
    filter_considered: int | None
    filter_excluded: int | None
    filter_included: int | None
    filter_excluded_fraction: float | None

    def __len__(self) -> int:
        return int(self.pixel.shape[0])


@dataclass(frozen=True)
class PinocchioRunMetadata:
    """PINOCCHIO run metadata needed for GEPPETTO map normalization.

    ``box_size_mpc_h`` is comoving ``Mpc/h``. ``particle_mass_msun_h`` is the
    mass represented by one PINOCCHIO grid element in ``Msun/h``:
    ``rho_mean_comoving(cosmology) * box_size_mpc_h**3 / grid_size**3``.
    """

    source: Path
    run_flag: str | None
    box_size_mpc_h: float
    grid_size: int
    cosmology: Cosmology
    particle_mass_msun_h: float
    box_in_h100: bool
    parameters: Mapping[str, tuple[str, ...]]


def halo_catalog_from_columns(
    columns: Mapping[str, object],
    *,
    position_keys=("x", "y", "z"),
    mass_key="mass",
    redshift_key="redshift",
) -> HaloCatalog:
    """Build a box catalogue from a mapping of column arrays."""

    position = jnp.stack([jnp.asarray(columns[key]) for key in position_keys], axis=-1)
    return HaloCatalog(
        position=position,
        mass=jnp.asarray(columns[mass_key]),
        redshift=jnp.asarray(columns[redshift_key]),
    )


def lightcone_catalog_from_columns(
    columns: Mapping[str, object],
    *,
    theta_key="theta",
    phi_key="phi",
    chi_key="chi",
    mass_key="mass",
    redshift_key="redshift",
) -> LightconeHaloCatalog:
    """Build a PLC catalogue from a mapping of HEALPix-style spherical columns."""

    theta = jnp.asarray(columns[theta_key])
    phi = jnp.asarray(columns[phi_key])
    return LightconeHaloCatalog(
        unit_vector=unit_vectors_from_angles(theta, phi),
        chi=jnp.asarray(columns[chi_key]),
        mass=jnp.asarray(columns[mass_key]),
        redshift=jnp.asarray(columns[redshift_key]),
    )


def read_pinocchio_snapshot_catalog(
    path: PathLike, *, format: CatalogFormat = "auto"
) -> PinocchioSnapshotCatalog:
    """Read a PINOCCHIO snapshot halo catalogue from ``*.catalog.out``.

    ``format="auto"`` detects PINOCCHIO ASCII tables and binary C-struct output.
    ASCII files use 12 columns: group ID, mass, initial position, final
    position, velocity, and particle count. Binary files are the native
    PINOCCHIO ``catalog_data`` layout, including split files named
    ``*.catalog.out.0``, ``*.catalog.out.1``, ...
    """

    source = Path(path)
    if format == "auto":
        format = _detect_catalog_format(source)
    if format == "binary":
        return read_pinocchio_binary_snapshot_catalog(source)
    if format != "ascii":
        raise PinocchioCatalogError("format must be 'auto', 'ascii', or 'binary'")

    data = _load_numeric_table(source, expected_columns=12, label="snapshot catalog")
    masses = data[:, 1]
    _require_positive(masses, source, "snapshot masses")
    return PinocchioSnapshotCatalog(
        group_ids=_integer_column(data[:, 0], source, "group IDs"),
        masses_msun_h=masses,
        initial_positions_mpc_h=data[:, 2:5],
        final_positions_mpc_h=data[:, 5:8],
        velocities_km_s=data[:, 8:11],
        n_particles=_integer_column(data[:, 11], source, "particle counts"),
        source=source,
        redshift=_parse_snapshot_redshift(source),
    )


def read_pinocchio_binary_snapshot_catalog(path: PathLike) -> PinocchioSnapshotCatalog:
    """Read a binary PINOCCHIO snapshot halo catalogue.

    Supports the native PINOCCHIO ``catalog_data`` binary layout written when
    ``CatalogInAscii`` is disabled, including split files named
    ``*.catalog.out.0``, ``*.catalog.out.1``, ... Masses are ``Msun/h``;
    positions are comoving ``Mpc/h``; velocities are ``km/s``.
    """

    source = Path(path)
    files = _pinocchio_output_files(source, label="snapshot catalog")
    chunks = [_read_binary_snapshot_catalog_file(file) for file in files]
    data = _concatenate_structured_chunks(chunks)
    if data is None:
        data = np.empty(0, dtype=_binary_snapshot_catalog_dtype(56, new_run=True)[0])

    masses = np.asarray(data["Mass"], dtype=np.float64)
    _require_positive(masses, source, "snapshot masses")
    n_particles = (
        np.asarray(data["npart"], dtype=np.int64) if "npart" in data.dtype.names else None
    )
    initial_positions = (
        np.asarray(data["posin"], dtype=np.float64) if "posin" in data.dtype.names else None
    )
    return PinocchioSnapshotCatalog(
        group_ids=np.asarray(data["name"], dtype=np.uint64),
        masses_msun_h=masses,
        initial_positions_mpc_h=initial_positions,
        final_positions_mpc_h=np.asarray(data["pos"], dtype=np.float64),
        velocities_km_s=np.asarray(data["vel"], dtype=np.float64),
        n_particles=n_particles,
        source=source,
        redshift=_parse_snapshot_redshift(source),
    )


def read_pinocchio_lightcone_catalog(
    path: PathLike, *, format: CatalogFormat = "auto"
) -> PinocchioLightconeCatalog:
    """Read a PINOCCHIO past-light-cone halo catalogue from ``*.plc.out``.

    ``format="auto"`` detects PINOCCHIO ASCII tables and binary C-struct output.
    Full ASCII files use 13 columns: group ID, true redshift, Cartesian
    comoving position, velocity, mass, latitude-like theta, longitude phi,
    line-of-sight velocity, and observed redshift. Binary files are the native
    PINOCCHIO ``plc_write_data`` layout, including split files named
    ``*.plc.out.0``, ``*.plc.out.1``, ...
    """

    source = Path(path)
    if format == "auto":
        format = _detect_catalog_format(source)
    if format == "binary":
        return read_pinocchio_binary_lightcone_catalog(source)
    if format != "ascii":
        raise PinocchioCatalogError("format must be 'auto', 'ascii', or 'binary'")

    data = _load_numeric_table(source, expected_columns=13, label="PLC catalog")
    masses = data[:, 8]
    _require_positive(masses, source, "PLC masses")
    return PinocchioLightconeCatalog(
        group_ids=_integer_column(data[:, 0], source, "group IDs"),
        true_redshift=data[:, 1],
        positions_mpc_h=data[:, 2:5],
        velocities_km_s=data[:, 5:8],
        masses_msun_h=masses,
        theta_deg=data[:, 9],
        phi_deg=data[:, 10],
        los_velocity_km_s=data[:, 11],
        observed_redshift=data[:, 12],
        source=source,
    )


def read_pinocchio_binary_lightcone_catalog(path: PathLike) -> PinocchioLightconeCatalog:
    """Read a binary PINOCCHIO past-light-cone halo catalogue.

    Supports the full native PINOCCHIO ``plc_write_data`` binary layout written
    when ``CatalogInAscii`` is disabled, including split files named
    ``*.plc.out.0``, ``*.plc.out.1``, ... Positions are comoving ``Mpc/h``;
    masses are ``Msun/h``; velocities are ``km/s``.

    PINOCCHIO light binary PLC output is intentionally rejected because it does
    not contain Cartesian positions or radial distances, so it cannot be
    converted into GEPPETTO's lightcone catalogue convention.
    """

    source = Path(path)
    files = _pinocchio_output_files(source, label="PLC catalog")
    chunks = [_read_binary_lightcone_catalog_file(file) for file in files]
    data = _concatenate_structured_chunks(chunks)
    if data is None:
        data = np.empty(0, dtype=_binary_lightcone_catalog_dtype(56, new_run=True)[0])

    masses = np.asarray(data["Mass"], dtype=np.float64)
    _require_positive(masses, source, "PLC masses")
    return PinocchioLightconeCatalog(
        group_ids=np.asarray(data["name"], dtype=np.uint64),
        true_redshift=np.asarray(data["truez"], dtype=np.float64),
        positions_mpc_h=np.asarray(data["pos"], dtype=np.float64),
        velocities_km_s=np.asarray(data["vel"], dtype=np.float64),
        masses_msun_h=masses,
        theta_deg=np.asarray(data["theta"], dtype=np.float64),
        phi_deg=np.asarray(data["phi"], dtype=np.float64),
        los_velocity_km_s=np.asarray(data["vlos"], dtype=np.float64),
        observed_redshift=np.asarray(data["obsz"], dtype=np.float64),
        source=source,
    )


def read_pinocchio_lightcone_light_catalog(
    path: PathLike, *, format: CatalogFormat = "auto"
) -> PinocchioLightconeLightCatalog:
    """Read a PINOCCHIO light PLC catalogue from ``*.plc.out``.

    Light ASCII PLC files use 6 columns: group ID, true redshift, mass in
    ``Msun/h``, latitude-like ``theta`` and longitude ``phi`` in degrees, and
    observed redshift. Light binary files are the 32-byte native PINOCCHIO
    ``plc_write_data`` layout. Because light PLC output does not contain
    Cartesian positions or radial distances, use
    ``PinocchioLightconeLightCatalog.to_lightcone_catalog`` with a
    ``PinocchioDistanceInterpolator`` from ``read_pinocchio_hubble_table``
    before passing it to GEPPETTO painters.
    """

    source = Path(path)
    if format == "auto":
        format = _detect_catalog_format(source)
    if format == "binary":
        return read_pinocchio_binary_lightcone_light_catalog(source)
    if format != "ascii":
        raise PinocchioCatalogError("format must be 'auto', 'ascii', or 'binary'")

    data = _load_numeric_table(source, expected_columns=6, label="light PLC catalog")
    masses = data[:, 2]
    _require_positive(masses, source, "light PLC masses")
    return PinocchioLightconeLightCatalog(
        group_ids=_integer_column(data[:, 0], source, "group IDs"),
        true_redshift=data[:, 1],
        masses_msun_h=masses,
        theta_deg=data[:, 3],
        phi_deg=data[:, 4],
        observed_redshift=data[:, 5],
        source=source,
    )


def read_pinocchio_binary_lightcone_light_catalog(
    path: PathLike,
) -> PinocchioLightconeLightCatalog:
    """Read a binary PINOCCHIO light PLC catalogue.

    Supports the native 32-byte PINOCCHIO ``plc_write_data`` layout written when
    ``LIGHT_OUTPUT`` is enabled and ``CatalogInAscii`` is disabled, including
    split files named ``*.plc.out.0``, ``*.plc.out.1``, ... Masses are
    ``Msun/h``; angles are latitude-like ``theta`` and longitude ``phi`` in
    degrees; redshifts are dimensionless.
    """

    source = Path(path)
    files = _pinocchio_output_files(source, label="light PLC catalog")
    chunks = [_read_binary_lightcone_light_catalog_file(file) for file in files]
    data = _concatenate_structured_chunks(chunks)
    if data is None:
        data = np.empty(0, dtype=_binary_lightcone_light_catalog_dtype(32, new_run=True)[0])

    masses = np.asarray(data["Mass"], dtype=np.float64)
    _require_positive(masses, source, "light PLC masses")
    return PinocchioLightconeLightCatalog(
        group_ids=np.asarray(data["name"], dtype=np.uint64),
        true_redshift=np.asarray(data["truez"], dtype=np.float64),
        masses_msun_h=masses,
        theta_deg=np.asarray(data["theta"], dtype=np.float64),
        phi_deg=np.asarray(data["phi"], dtype=np.float64),
        observed_redshift=np.asarray(data["obsz"], dtype=np.float64),
        source=source,
    )


def read_pinocchio_hubble_table(
    path: PathLike, *, n_grid: int = 16_384
) -> PinocchioDistanceInterpolator:
    """Read a PINOCCHIO ``HubbleTableFile`` and build distance interpolation.

    The table must contain two columns: redshift and dimensionless
    ``E(z) = H(z) / H0``. The returned interpolator converts redshift to
    comoving distance in ``Mpc/h`` using ``c / 100 * integral dz / E(z)``. This
    mirrors PINOCCHIO's tabulated-Hubble convention while keeping distance
    construction outside GEPPETTO's differentiable kernels.
    """

    if n_grid < 2:
        raise PinocchioCatalogError("n_grid must be at least 2")

    source = Path(path)
    data = _load_numeric_table(source, expected_columns=2, label="Hubble table")
    if data.shape[0] == 0:
        raise PinocchioCatalogError(f"PINOCCHIO Hubble table is empty: {source}")

    redshift = np.asarray(data[:, 0], dtype=np.float64)
    e_z = np.asarray(data[:, 1], dtype=np.float64)
    if np.any(redshift < 0.0):
        raise PinocchioCatalogError(f"PINOCCHIO Hubble table redshifts must be non-negative: {source}")
    _require_positive(e_z, source, "Hubble table E(z)")

    z0 = np.isclose(redshift, 0.0, rtol=0.0, atol=1.0e-12)
    if np.any(z0):
        if not np.allclose(e_z[z0], 1.0, rtol=1.0e-4, atol=1.0e-8):
            raise PinocchioCatalogError(f"PINOCCHIO Hubble table must have E(0) = 1: {source}")
    else:
        redshift = np.concatenate([np.array([0.0]), redshift])
        e_z = np.concatenate([np.array([1.0]), e_z])

    order = np.argsort(redshift)
    redshift = redshift[order]
    e_z = e_z[order]
    if np.any(np.diff(redshift) <= 0.0):
        raise PinocchioCatalogError(
            f"PINOCCHIO Hubble table redshifts must be unique and increasing: {source}"
        )

    integration_redshift = np.unique(
        np.concatenate([redshift, np.linspace(0.0, float(redshift[-1]), int(n_grid))])
    )
    loga_table = -np.log10(1.0 + redshift)
    loge_table = np.log10(e_z)
    loga_order = np.argsort(loga_table)
    loga_grid = -np.log10(1.0 + integration_redshift)
    loge_grid = np.interp(loga_grid, loga_table[loga_order], loge_table[loga_order])
    e_grid = np.power(10.0, loge_grid)
    integrand = 1.0 / e_grid
    dz = np.diff(integration_redshift)
    chi_grid_mpc_h = np.concatenate(
        [
            np.array([0.0]),
            (_C_LIGHT_KM_S / 100.0)
            * np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * dz),
        ]
    )

    return PinocchioDistanceInterpolator(
        redshift=redshift,
        e_z=e_z,
        integration_redshift=integration_redshift,
        chi_grid_mpc_h=chi_grid_mpc_h,
        source=source,
    )


def read_pinocchio_parameter_file(path: PathLike) -> PinocchioRunMetadata:
    """Read PINOCCHIO run metadata needed for particle-count map painting.

    The parser extracts ``BoxSize``, ``BoxInH100``, ``GridSize``, ``Omega0`` and
    ``Hubble100`` from a PINOCCHIO parameter file. ``BoxSize`` is interpreted as
    ``Mpc/h`` when ``BoxInH100`` is present, otherwise as physical ``Mpc`` and
    converted to ``Mpc/h`` by multiplying by ``Hubble100``. The returned particle
    mass is in ``Msun/h``.
    """

    source = Path(path)
    parameters = _parse_pinocchio_parameter_file(source)
    box_size = _required_parameter_float(parameters, "BoxSize", source)
    grid_size = _required_parameter_int(parameters, "GridSize", source)
    omega_m = _required_parameter_float(parameters, "Omega0", source)
    h = _required_parameter_float(parameters, "Hubble100", source)

    if box_size <= 0.0:
        raise PinocchioCatalogError(f"PINOCCHIO BoxSize must be positive: {source}")
    if grid_size <= 0:
        raise PinocchioCatalogError(f"PINOCCHIO GridSize must be positive: {source}")
    if omega_m <= 0.0:
        raise PinocchioCatalogError(f"PINOCCHIO Omega0 must be positive: {source}")
    if h <= 0.0:
        raise PinocchioCatalogError(f"PINOCCHIO Hubble100 must be positive: {source}")

    box_in_h100 = "BoxInH100" in parameters
    box_size_mpc_h = box_size if box_in_h100 else box_size * h
    cosmology = Cosmology(omega_m=omega_m, h=h)
    particle_mass_msun_h = (
        rho_mean_comoving(cosmology) * box_size_mpc_h**3 / float(grid_size) ** 3
    )
    run_flag_values = parameters.get("RunFlag", ())
    run_flag = run_flag_values[0] if run_flag_values else None

    return PinocchioRunMetadata(
        source=source,
        run_flag=run_flag,
        box_size_mpc_h=box_size_mpc_h,
        grid_size=grid_size,
        cosmology=cosmology,
        particle_mass_msun_h=particle_mass_msun_h,
        box_in_h100=box_in_h100,
        parameters=parameters,
    )


def healpix_pixel_area_sr(nside: int) -> float:
    """Return the HEALPix pixel solid angle in steradians."""

    nside = _validate_healpix_nside(nside)
    return 4.0 * math.pi / (12 * nside * nside)


def healpix_pixel_unit_vectors(
    nside: int,
    pixels: np.ndarray | None = None,
    *,
    nest: bool = False,
) -> np.ndarray:
    """Return HEALPix pixel-centre unit vectors for fixed map geometry.

    The returned array has shape ``(n_pix, 3)`` and preserves the supplied pixel
    order. This helper imports ``healpy`` lazily and belongs outside the
    differentiable core because HEALPix index arithmetic is discrete.
    """

    try:
        import healpy as hp
    except ImportError as exc:  # pragma: no cover - exercised only without healpy
        raise PinocchioCatalogError(
            "healpix_pixel_unit_vectors requires healpy; install geppetto[io]"
        ) from exc

    nside = _validate_healpix_nside(nside)
    npix = 12 * nside * nside
    if pixels is None:
        pixel_values = np.arange(npix, dtype=np.int64)
    else:
        pixel_values = np.asarray(pixels, dtype=np.int64)
        if pixel_values.ndim != 1:
            raise PinocchioCatalogError("pixels must be a one-dimensional array")
        if np.any(pixel_values < 0) or np.any(pixel_values >= npix):
            raise PinocchioCatalogError(f"HEALPix pixels must be in [0, {npix})")

    x, y, z = hp.pix2vec(nside, pixel_values, nest=nest)
    return np.stack([x, y, z], axis=-1).astype(np.float64, copy=False)


def validate_lightcone_sparse_stencil(
    stencil: LightconeSparseStencil,
    catalog: LightconeHaloCatalog | None = None,
) -> None:
    """Validate a sparse lightcone stencil outside JAX-transformed paths.

    This helper is intended for I/O adapters and manually constructed stencils.
    It performs host-side NumPy checks and must not be called from differentiable
    kernels.
    """

    pix_id = np.asarray(stencil.pix_id)
    halo_id = np.asarray(stencil.halo_id)
    r_perp = np.asarray(stencil.r_perp)

    if pix_id.ndim != 1:
        raise PinocchioCatalogError("stencil.pix_id must be one-dimensional")
    if halo_id.ndim != 1:
        raise PinocchioCatalogError("stencil.halo_id must be one-dimensional")
    if r_perp.ndim != 1:
        raise PinocchioCatalogError("stencil.r_perp must be one-dimensional")
    if pix_id.shape[0] != halo_id.shape[0] or pix_id.shape[0] != r_perp.shape[0]:
        raise PinocchioCatalogError("stencil fields must have matching lengths")
    if not np.issubdtype(pix_id.dtype, np.integer):
        raise PinocchioCatalogError("stencil.pix_id must contain integer indices")
    if not np.issubdtype(halo_id.dtype, np.integer):
        raise PinocchioCatalogError("stencil.halo_id must contain integer indices")

    try:
        n_pix = int(stencil.n_pix)
    except TypeError as exc:
        raise PinocchioCatalogError("stencil.n_pix must be an integer") from exc
    if n_pix < 0:
        raise PinocchioCatalogError("stencil.n_pix must be non-negative")
    if pix_id.size and (np.any(pix_id < 0) or np.any(pix_id >= n_pix)):
        raise PinocchioCatalogError("stencil.pix_id contains out-of-range pixel indices")
    if halo_id.size and np.any(halo_id < 0):
        raise PinocchioCatalogError("stencil.halo_id contains negative halo indices")
    if not np.all(np.isfinite(r_perp)) or np.any(r_perp < 0.0):
        raise PinocchioCatalogError("stencil.r_perp values must be finite and non-negative")

    if catalog is not None:
        mass = np.asarray(catalog.mass)
        if mass.ndim != 1:
            raise PinocchioCatalogError("catalog.mass must have shape (n_halo,)")
        n_halo = int(mass.shape[0])
        if halo_id.size and np.any(halo_id >= n_halo):
            raise PinocchioCatalogError("stencil.halo_id contains out-of-range halo indices")


def validate_tabulated_projected_profile_params(
    profile_params: TabulatedProjectedProfileParams,
) -> None:
    """Validate tabulated projected-profile parameters outside JAX paths.

    This helper checks host-side profile-grid contracts before callers enter
    JIT-compiled kernels. The differentiable profile kernel assumes validated
    one-dimensional arrays with fixed support covering ``0 <= x <= 1``.
    """

    x = np.asarray(profile_params.x)
    log_shape = np.asarray(profile_params.log_shape)

    if x.ndim != 1:
        raise PinocchioCatalogError("profile_params.x must be one-dimensional")
    if log_shape.ndim != 1:
        raise PinocchioCatalogError("profile_params.log_shape must be one-dimensional")
    if x.shape != log_shape.shape:
        raise PinocchioCatalogError("profile_params.x and log_shape must have matching shapes")
    if x.shape[0] < 2:
        raise PinocchioCatalogError("profile_params.x must contain at least two grid points")
    if not np.all(np.isfinite(x)):
        raise PinocchioCatalogError("profile_params.x values must be finite")
    if not np.all(np.isfinite(log_shape)):
        raise PinocchioCatalogError("profile_params.log_shape values must be finite")
    if not np.all(np.diff(x) > 0.0):
        raise PinocchioCatalogError("profile_params.x must be strictly increasing")
    if x[0] > 0.0 or x[-1] < 1.0:
        raise PinocchioCatalogError("profile_params.x must cover the interval [0, 1]")


def build_lightcone_sparse_stencil_bruteforce(
    pixel_unit_vectors: np.ndarray,
    catalog: LightconeHaloCatalog,
    rmax_mpc_h: np.ndarray | float,
) -> LightconeSparseStencil:
    """Build a brute-force sparse lightcone halo-pixel stencil.

    This non-core helper uses NumPy and fixed geometry. It accepts target pixel
    unit vectors, a lightcone catalogue with halo unit vectors and comoving
    distances in ``Mpc/h``, and a scalar or per-halo ``Rmax`` in ``Mpc/h``.
    Returned pairs satisfy ``R_perp <= Rmax_halo`` using the same chord
    transverse-distance convention as GEPPETTO's dense lightcone painter.

    This implementation materializes an ``n_pix * n_halo`` separation matrix
    before filtering. It is useful for validation, examples, and small maps, but
    is not the scalable production HEALPix-local stencil builder.
    """

    pixels = np.asarray(pixel_unit_vectors, dtype=np.float64)
    if pixels.ndim != 2 or pixels.shape[1] != 3:
        raise PinocchioCatalogError("pixel_unit_vectors must have shape (n_pix, 3)")
    n_pix = int(pixels.shape[0])

    halo_unit_vectors = np.asarray(catalog.unit_vector, dtype=np.float64)
    halo_chi = np.asarray(catalog.chi, dtype=np.float64)
    if halo_unit_vectors.ndim != 2 or halo_unit_vectors.shape[1] != 3:
        raise PinocchioCatalogError("catalog.unit_vector must have shape (n_halo, 3)")
    if halo_chi.ndim != 1 or halo_chi.shape[0] != halo_unit_vectors.shape[0]:
        raise PinocchioCatalogError("catalog.chi must have shape (n_halo,)")
    if not np.all(np.isfinite(pixels)) or not np.all(np.isfinite(halo_unit_vectors)):
        raise PinocchioCatalogError("stencil unit vectors must be finite")
    if not np.all(np.isfinite(halo_chi)):
        raise PinocchioCatalogError("catalog.chi must be finite")

    n_halo = int(halo_chi.shape[0])
    rmax = np.asarray(rmax_mpc_h, dtype=np.float64)
    if rmax.ndim == 0:
        rmax_value = float(rmax)
        if not math.isfinite(rmax_value) or rmax_value < 0.0:
            raise PinocchioCatalogError("rmax_mpc_h values must be finite and non-negative")
        rmax = np.full((n_halo,), rmax_value, dtype=np.float64)
    elif rmax.shape != (n_halo,):
        raise PinocchioCatalogError("rmax_mpc_h must be scalar or have shape (n_halo,)")
    if not np.all(np.isfinite(rmax)) or np.any(rmax < 0.0):
        raise PinocchioCatalogError("rmax_mpc_h values must be finite and non-negative")

    if n_pix == 0 or n_halo == 0:
        stencil = LightconeSparseStencil(
            pix_id=jnp.empty((0,), dtype=jnp.int32),
            halo_id=jnp.empty((0,), dtype=jnp.int32),
            r_perp=jnp.empty((0,), dtype=jnp.float64),
            n_pix=n_pix,
        )
        validate_lightcone_sparse_stencil(stencil, catalog)
        return stencil

    cosang = np.clip(pixels @ halo_unit_vectors.T, -1.0, 1.0)
    chord = np.sqrt(np.maximum(2.0 * (1.0 - cosang), 0.0))
    r_perp = chord * halo_chi[None, :]
    pix_id, halo_id = np.nonzero(r_perp <= rmax[None, :])
    stencil = LightconeSparseStencil(
        pix_id=jnp.asarray(pix_id, dtype=jnp.int32),
        halo_id=jnp.asarray(halo_id, dtype=jnp.int32),
        r_perp=jnp.asarray(r_perp[pix_id, halo_id]),
        n_pix=n_pix,
    )
    validate_lightcone_sparse_stencil(stencil, catalog)
    return stencil


def build_lightcone_sparse_stencil(
    pixel_unit_vectors: np.ndarray,
    catalog: LightconeHaloCatalog,
    rmax_mpc_h: np.ndarray | float,
) -> LightconeSparseStencil:
    """Backward-compatible alias for the brute-force sparse stencil builder.

    Prefer :func:`build_lightcone_sparse_stencil_bruteforce` when the
    construction cost matters. This alias retains the original prototype name
    but still materializes an ``n_pix * n_halo`` separation matrix.
    """

    return build_lightcone_sparse_stencil_bruteforce(
        pixel_unit_vectors,
        catalog,
        rmax_mpc_h,
    )


def read_pinocchio_mass_sheets(path: PathLike) -> PinocchioMassSheetTable:
    """Read a PINOCCHIO mass-sheet table from ``*.sheets.out``."""

    source = Path(path)
    data = _load_numeric_table(source, expected_columns=11, label="mass-sheet table")
    return PinocchioMassSheetTable(
        sheet_ids=_integer_column(data[:, 0], source, "sheet IDs"),
        z_hi=data[:, 1],
        z_lo=data[:, 2],
        delta_z=data[:, 3],
        chi_hi_mpc_h=data[:, 4],
        chi_lo_mpc_h=data[:, 5],
        delta_chi_mpc_h=data[:, 6],
        inv_delta_chi_h_mpc=data[:, 7],
        da_hi_mpc_h=data[:, 8],
        da_lo_mpc_h=data[:, 9],
        chi3_diff_mpc_h3=data[:, 10],
        source=source,
    )


def read_pinocchio_nz(path: PathLike) -> PinocchioNzTable:
    """Read a PINOCCHIO PLC number-count table from ``*.nz.out``."""

    source = Path(path)
    data = _load_numeric_table(source, expected_columns=5, label="n(z) table")
    return PinocchioNzTable(
        z_low=data[:, 0],
        z_high=data[:, 1],
        counts=_integer_column(data[:, 2], source, "n(z) counts"),
        number_per_square_degree=data[:, 3],
        predicted_counts=data[:, 4],
        source=source,
    )


def read_pinocchio_mass_function(path: PathLike) -> PinocchioMassFunction:
    """Read a PINOCCHIO mass-function table from ``*.mf.out``."""

    source = Path(path)
    data = _load_numeric_table(source, expected_columns=7, label="mass-function table")
    return PinocchioMassFunction(
        mass_msun_h=data[:, 0],
        number_density=data[:, 1],
        number_density_plus_1sigma=data[:, 2],
        number_density_minus_1sigma=data[:, 3],
        halo_counts=_integer_column(data[:, 4], source, "mass-function halo counts"),
        analytic_number_density=data[:, 5],
        peak_height_nu=data[:, 6],
        source=source,
        redshift=_parse_redshift_from_header(source),
    )


def read_pinocchio_mass_map_fits(path: PathLike) -> PinocchioMassMap:
    """Read a PINOCCHIO HEALPix mass-map FITS binary table.

    Astropy is imported lazily so installing GEPPETTO without the ``io`` extra
    does not import FITS dependencies. The current PINOCCHIO output is expected
    to contain a ``HEALPIX`` binary table with ``PIXEL`` and ``TEMPERATURE``
    columns.
    """

    try:
        from astropy.io import fits
    except ImportError as exc:  # pragma: no cover - exercised only without io extra
        raise PinocchioCatalogError(
            "read_pinocchio_mass_map_fits requires astropy; install geppetto[io]"
        ) from exc

    source = Path(path)
    try:
        with fits.open(source) as hdul:
            hdu = hdul["HEALPIX"] if "HEALPIX" in hdul else hdul[1]
            if hdu.data is None:
                raise PinocchioCatalogError(f"Mass-map FITS table has no data: {source}")
            names = {name.upper(): name for name in hdu.data.names or ()}
            pixel_name = names.get("PIXEL")
            temperature_name = names.get("TEMPERATURE")
            if pixel_name is None or temperature_name is None:
                raise PinocchioCatalogError(
                    f"Mass-map FITS table must contain PIXEL and TEMPERATURE columns: {source}"
                )
            header = {key: hdu.header[key] for key in hdu.header if key}
            pixel = np.asarray(hdu.data[pixel_name], dtype=np.int64)
            temperature = np.asarray(hdu.data[temperature_name], dtype=np.float64)
    except OSError as exc:
        raise PinocchioCatalogError(f"Cannot read PINOCCHIO mass-map FITS file: {source}") from exc

    if pixel.shape != temperature.shape:
        raise PinocchioCatalogError(f"PIXEL and TEMPERATURE shapes differ: {source}")
    _require_finite(temperature, source, "mass-map values")

    nside = _required_header_int(header, "NSIDE", source)
    ordering = _required_header_str(header, "ORDERING", source)
    axis_vector = _optional_axis_vector(header)
    return PinocchioMassMap(
        pixel=pixel,
        temperature=temperature,
        source=source,
        header=header,
        nside=nside,
        ordering=ordering,
        index_scheme=_optional_header_str(header, "INDXSCHM"),
        first_pixel=_optional_header_int(header, "FIRSTPIX"),
        last_pixel=_optional_header_int(header, "LASTPIX"),
        aperture_deg=_optional_header_float(header, "APERTURE"),
        selection_type=_optional_header_str(header, "SELTYPE"),
        axis_vector=axis_vector,
        filter_name=_optional_header_str(header, "FILTER"),
        filter_considered=_optional_header_int(header, "ZF_CONS"),
        filter_excluded=_optional_header_int(header, "ZF_EXCL"),
        filter_included=_optional_header_int(header, "ZF_INCL"),
        filter_excluded_fraction=_optional_header_float(header, "ZF_FEXCL"),
    )


def _detect_catalog_format(path: Path) -> CatalogFormat:
    first_file = _pinocchio_output_files(path, label="catalog")[0]
    try:
        sample = first_file.read_bytes()[:4096]
    except OSError as exc:
        raise PinocchioCatalogError(f"Cannot read PINOCCHIO catalog: {first_file}") from exc
    if b"\x00" in sample:
        return "binary"
    try:
        sample.decode("utf-8")
    except UnicodeDecodeError:
        return "binary"
    return "ascii"


def _pinocchio_output_files(path: Path, *, label: str) -> list[Path]:
    if path.exists():
        return [path]

    files: list[Path] = []
    index = 0
    while True:
        candidate = Path(f"{path}.{index}")
        if not candidate.exists():
            break
        files.append(candidate)
        index += 1

    if files:
        return files
    raise PinocchioCatalogError(f"Cannot find PINOCCHIO {label}: {path} or {path}.0")


def _read_binary_snapshot_catalog_file(path: Path) -> np.ndarray:
    try:
        bindata = path.read_bytes()
    except OSError as exc:
        raise PinocchioCatalogError(f"Cannot read binary PINOCCHIO snapshot catalog: {path}") from exc
    if len(bindata) < 16:
        raise PinocchioCatalogError(f"Binary PINOCCHIO snapshot catalog is truncated: {path}")

    header = np.frombuffer(bindata, dtype=np.int32, count=min(10, len(bindata) // 4))
    if header.size < 4:
        raise PinocchioCatalogError(f"Binary PINOCCHIO snapshot catalog header is truncated: {path}")

    new_run = bool(header[2] > 10)
    if new_run:
        if int(header[0]) != 2 * np.dtype(np.int32).itemsize or int(header[3]) != int(header[0]):
            raise PinocchioCatalogError(f"Invalid binary PINOCCHIO snapshot catalog header: {path}")
        record_length = int(header[2])
    else:
        if header.size < 8:
            raise PinocchioCatalogError(
                f"Classic binary PINOCCHIO snapshot catalog header is truncated: {path}"
            )
        record_length = int(header[7])

    cat_dtype, stored_dtype = _binary_snapshot_catalog_dtype(record_length, new_run=new_run)
    expected_record_size = record_length if new_run else record_length + 2 * np.dtype(np.int32).itemsize
    if stored_dtype.itemsize != expected_record_size:
        raise PinocchioCatalogError(
            f"Unsupported binary PINOCCHIO snapshot record layout "
            f"for record length {record_length}: {path}"
        )

    offset = 4 * np.dtype(np.int32).itemsize
    chunks: list[np.ndarray] = []
    while offset < len(bindata):
        count_record, offset = _read_int32_triplet(bindata, offset, path, "snapshot block count")
        if count_record[0] != np.dtype(np.int32).itemsize or count_record[2] != count_record[0]:
            raise PinocchioCatalogError(f"Invalid snapshot block count record marker: {path}")
        n_halos = int(count_record[1])
        if n_halos < 0:
            raise PinocchioCatalogError(f"Negative snapshot block halo count in {path}")
        if n_halos == 0:
            continue

        if new_run:
            block_bytes, offset = _read_int32(bindata, offset, path, "snapshot data record")
            expected_bytes = n_halos * record_length
            if int(block_bytes) != expected_bytes:
                raise PinocchioCatalogError(f"Invalid snapshot data block size in {path}")
            raw = _read_bytes(bindata, offset, expected_bytes, path, "snapshot records")
            offset += expected_bytes
            closing_bytes, offset = _read_int32(bindata, offset, path, "snapshot data record")
            if int(closing_bytes) != expected_bytes:
                raise PinocchioCatalogError(f"Invalid closing snapshot data block size in {path}")
        else:
            expected_bytes = n_halos * stored_dtype.itemsize
            raw = _read_bytes(bindata, offset, expected_bytes, path, "snapshot records")
            offset += expected_bytes

        stored = np.frombuffer(raw, dtype=stored_dtype, count=n_halos)
        chunks.append(_copy_structured_fields(stored, cat_dtype))

    if offset != len(bindata):
        raise PinocchioCatalogError(f"Unexpected trailing bytes in binary snapshot catalog: {path}")
    catalog = _concatenate_structured_chunks(chunks)
    return catalog if catalog is not None else np.empty(0, dtype=cat_dtype)


def _read_binary_lightcone_catalog_file(path: Path) -> np.ndarray:
    return _read_binary_lightcone_file(path, dtype_factory=_binary_lightcone_catalog_dtype)


def _read_binary_lightcone_light_catalog_file(path: Path) -> np.ndarray:
    return _read_binary_lightcone_file(path, dtype_factory=_binary_lightcone_light_catalog_dtype)


def _read_binary_lightcone_file(path: Path, *, dtype_factory) -> np.ndarray:
    try:
        bindata = path.read_bytes()
    except OSError as exc:
        raise PinocchioCatalogError(f"Cannot read binary PINOCCHIO PLC catalog: {path}") from exc
    if not bindata:
        raise PinocchioCatalogError(f"Binary PINOCCHIO PLC catalog is empty: {path}")

    header = np.frombuffer(bindata, dtype=np.int32, count=min(3, len(bindata) // 4))
    if header.size < 1:
        raise PinocchioCatalogError(f"Binary PINOCCHIO PLC catalog header is truncated: {path}")

    new_run = bool(header[0] == np.dtype(np.int32).itemsize)
    if new_run:
        if header.size < 3 or int(header[2]) != int(header[0]):
            raise PinocchioCatalogError(f"Invalid binary PINOCCHIO PLC header: {path}")
        record_length = int(header[1])
        offset = 3 * np.dtype(np.int32).itemsize
    else:
        record_length = int(header[0])
        offset = 0

    cat_dtype, stored_dtype = dtype_factory(record_length, new_run=new_run)
    expected_record_size = record_length if new_run else record_length + 2 * np.dtype(np.int32).itemsize
    if stored_dtype.itemsize != expected_record_size:
        raise PinocchioCatalogError(
            f"Unsupported binary PINOCCHIO PLC record layout "
            f"for record length {record_length}: {path}"
        )

    if not new_run:
        if len(bindata) % stored_dtype.itemsize != 0:
            raise PinocchioCatalogError(f"Classic binary PINOCCHIO PLC size is inconsistent: {path}")
        stored = np.frombuffer(bindata, dtype=stored_dtype)
        return _copy_structured_fields(stored, cat_dtype)

    chunks: list[np.ndarray] = []
    while offset < len(bindata):
        count_record, offset = _read_int32_triplet(bindata, offset, path, "PLC block count")
        if count_record[0] != np.dtype(np.int32).itemsize or count_record[2] != count_record[0]:
            raise PinocchioCatalogError(f"Invalid PLC block count record marker: {path}")
        n_halos = int(count_record[1])
        if n_halos < 0:
            raise PinocchioCatalogError(f"Negative PLC block halo count in {path}")

        block_bytes, offset = _read_int32(bindata, offset, path, "PLC data record")
        expected_bytes = n_halos * record_length
        if int(block_bytes) != expected_bytes:
            raise PinocchioCatalogError(f"Invalid PLC data block size in {path}")
        raw = _read_bytes(bindata, offset, expected_bytes, path, "PLC records")
        offset += expected_bytes
        closing_bytes, offset = _read_int32(bindata, offset, path, "PLC data record")
        if int(closing_bytes) != expected_bytes:
            raise PinocchioCatalogError(f"Invalid closing PLC data block size in {path}")
        if n_halos:
            stored = np.frombuffer(raw, dtype=stored_dtype, count=n_halos)
            chunks.append(_copy_structured_fields(stored, cat_dtype))

    if offset != len(bindata):
        raise PinocchioCatalogError(f"Unexpected trailing bytes in binary PLC catalog: {path}")
    catalog = _concatenate_structured_chunks(chunks)
    return catalog if catalog is not None else np.empty(0, dtype=cat_dtype)


def _binary_snapshot_catalog_dtype(
    record_length: int, *, new_run: bool
) -> tuple[np.dtype, np.dtype]:
    if record_length == 96:
        scalar_dtype = np.float64
        if new_run:
            fields = [
                ("name", np.uint64),
                ("Mass", scalar_dtype),
                ("pos", scalar_dtype, 3),
                ("vel", scalar_dtype, 3),
                ("posin", scalar_dtype, 3),
                ("npart", np.int32),
            ]
            stored_fields = [*fields, ("pad", np.int32)]
        else:
            fields = [
                ("name", np.uint64),
                ("Mass", scalar_dtype),
                ("posin", scalar_dtype, 3),
                ("pos", scalar_dtype, 3),
                ("vel", scalar_dtype, 3),
                ("npart", np.int32),
            ]
            stored_fields = [("fort", np.int32), *fields, ("pad", np.int32), ("trof", np.int32)]
    elif record_length == 56:
        scalar_dtype = np.float32
        if new_run:
            fields = [
                ("name", np.uint64),
                ("Mass", scalar_dtype),
                ("pos", scalar_dtype, 3),
                ("vel", scalar_dtype, 3),
                ("posin", scalar_dtype, 3),
                ("npart", np.int32),
            ]
            stored_fields = [*fields, ("pad", np.int32)]
        else:
            fields = [
                ("name", np.uint64),
                ("Mass", scalar_dtype),
                ("posin", scalar_dtype, 3),
                ("pos", scalar_dtype, 3),
                ("vel", scalar_dtype, 3),
                ("npart", np.int32),
            ]
            stored_fields = [("fort", np.int32), *fields, ("pad", np.int32), ("trof", np.int32)]
    elif record_length == 48 and new_run:
        scalar_dtype = np.float32
        fields = [
            ("name", np.uint64),
            ("Mass", scalar_dtype),
            ("pos", scalar_dtype, 3),
            ("vel", scalar_dtype, 3),
            ("posin", scalar_dtype, 3),
        ]
        stored_fields = fields
    elif record_length == 40 and not new_run:
        scalar_dtype = np.float32
        fields = [
            ("name", np.uint64),
            ("Mass", scalar_dtype),
            ("pos", scalar_dtype, 3),
            ("vel", scalar_dtype, 3),
            ("npart", np.int32),
        ]
        stored_fields = [("fort", np.int32), *fields, ("trof", np.int32)]
    else:
        raise PinocchioCatalogError(
            f"Unsupported PINOCCHIO binary snapshot record length: {record_length}"
        )
    return np.dtype(fields), np.dtype(stored_fields)


def _binary_lightcone_catalog_dtype(
    record_length: int, *, new_run: bool
) -> tuple[np.dtype, np.dtype]:
    if record_length == 104:
        scalar_dtype = np.float64
        fields = [
            ("name", np.uint64),
            ("truez", scalar_dtype),
            ("pos", scalar_dtype, 3),
            ("vel", scalar_dtype, 3),
            ("Mass", scalar_dtype),
            ("theta", scalar_dtype),
            ("phi", scalar_dtype),
            ("vlos", scalar_dtype),
            ("obsz", scalar_dtype),
        ]
        stored_fields = fields if new_run else [("fort", np.int32), *fields, ("trof", np.int32)]
    elif record_length == 56:
        scalar_dtype = np.float32
        fields = [
            ("name", np.uint64),
            ("truez", scalar_dtype),
            ("pos", scalar_dtype, 3),
            ("vel", scalar_dtype, 3),
            ("Mass", scalar_dtype),
            ("theta", scalar_dtype),
            ("phi", scalar_dtype),
            ("vlos", scalar_dtype),
            ("obsz", scalar_dtype),
        ]
        stored_fields = fields if new_run else [("fort", np.int32), *fields, ("trof", np.int32)]
    elif record_length == 32:
        raise PinocchioCatalogError(
            "PINOCCHIO light binary PLC output lacks Cartesian positions and "
            "requires read_pinocchio_binary_lightcone_light_catalog plus a "
            "PinocchioDistanceInterpolator"
        )
    else:
        raise PinocchioCatalogError(f"Unsupported PINOCCHIO binary PLC record length: {record_length}")
    return np.dtype(fields), np.dtype(stored_fields)


def _binary_lightcone_light_catalog_dtype(
    record_length: int, *, new_run: bool
) -> tuple[np.dtype, np.dtype]:
    if record_length != 32:
        raise PinocchioCatalogError(
            f"Unsupported PINOCCHIO light binary PLC record length: {record_length}"
        )

    scalar_dtype = np.float32
    fields = [
        ("name", np.uint64),
        ("truez", scalar_dtype),
        ("Mass", scalar_dtype),
        ("theta", scalar_dtype),
        ("phi", scalar_dtype),
        ("obsz", scalar_dtype),
    ]
    stored_fields = (
        [*fields, ("pad", scalar_dtype)]
        if new_run
        else [("fort", np.int32), *fields, ("pad", scalar_dtype), ("trof", np.int32)]
    )
    return np.dtype(fields), np.dtype(stored_fields)


def _copy_structured_fields(data: np.ndarray, dtype: np.dtype) -> np.ndarray:
    copied = np.empty(data.shape[0], dtype=dtype)
    for name in copied.dtype.names or ():
        copied[name] = data[name]
    return copied


def _concatenate_structured_chunks(chunks: list[np.ndarray]) -> np.ndarray | None:
    if not chunks:
        return None
    nonempty = [chunk for chunk in chunks if chunk.size]
    if nonempty:
        return np.concatenate(nonempty)
    return np.empty(0, dtype=chunks[0].dtype)


def _read_int32(data: bytes, offset: int, path: Path, label: str) -> tuple[int, int]:
    raw = _read_bytes(data, offset, np.dtype(np.int32).itemsize, path, label)
    return int(np.frombuffer(raw, dtype=np.int32, count=1)[0]), offset + np.dtype(np.int32).itemsize


def _read_int32_triplet(
    data: bytes, offset: int, path: Path, label: str
) -> tuple[np.ndarray, int]:
    nbytes = 3 * np.dtype(np.int32).itemsize
    raw = _read_bytes(data, offset, nbytes, path, label)
    return np.frombuffer(raw, dtype=np.int32, count=3), offset + nbytes


def _read_bytes(data: bytes, offset: int, nbytes: int, path: Path, label: str) -> bytes:
    end = offset + nbytes
    if offset < 0 or nbytes < 0 or end > len(data):
        raise PinocchioCatalogError(f"Truncated binary PINOCCHIO {label}: {path}")
    return data[offset:end]


def _load_numeric_table(path: Path, *, expected_columns: int, label: str) -> np.ndarray:
    try:
        files = _pinocchio_output_files(path, label=label)
        text = "\n".join(file.read_text(encoding="utf-8") for file in files)
    except (OSError, UnicodeDecodeError) as exc:
        raise PinocchioCatalogError(f"Cannot read PINOCCHIO {label}: {path}") from exc

    data_lines = [
        line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")
    ]
    if not data_lines:
        return np.empty((0, expected_columns), dtype=np.float64)

    try:
        data = np.loadtxt(StringIO("\n".join(data_lines)), dtype=np.float64, ndmin=2)
    except ValueError as exc:
        raise PinocchioCatalogError(f"Invalid numeric rows in PINOCCHIO {label}: {path}") from exc

    if data.ndim != 2 or data.shape[1] != expected_columns:
        raise PinocchioCatalogError(
            f"PINOCCHIO {label} must have {expected_columns} columns; "
            f"found shape {data.shape}: {path}"
        )
    _require_finite(data, path, label)
    return data


def _parse_pinocchio_parameter_file(path: Path) -> dict[str, tuple[str, ...]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise PinocchioCatalogError(f"Cannot read PINOCCHIO parameter file: {path}") from exc

    parameters: dict[str, tuple[str, ...]] = {}
    for line in lines:
        stripped = line.split("%", 1)[0].split("#", 1)[0].strip()
        if not stripped:
            continue
        tokens = tuple(stripped.split())
        parameters[tokens[0]] = tokens[1:]
    return parameters


def _required_parameter_values(
    parameters: Mapping[str, tuple[str, ...]], key: str, path: Path
) -> tuple[str, ...]:
    values = parameters.get(key)
    if not values:
        raise PinocchioCatalogError(f"PINOCCHIO parameter file is missing {key}: {path}")
    return values


def _required_parameter_float(
    parameters: Mapping[str, tuple[str, ...]], key: str, path: Path
) -> float:
    values = _required_parameter_values(parameters, key, path)
    try:
        value = float(values[0])
    except ValueError as exc:
        raise PinocchioCatalogError(f"PINOCCHIO {key} must be numeric: {path}") from exc
    if not math.isfinite(value):
        raise PinocchioCatalogError(f"PINOCCHIO {key} must be finite: {path}")
    return value


def _required_parameter_int(
    parameters: Mapping[str, tuple[str, ...]], key: str, path: Path
) -> int:
    value = _required_parameter_float(parameters, key, path)
    rounded = round(value)
    if not math.isclose(value, rounded, rel_tol=0.0, abs_tol=1.0e-8):
        raise PinocchioCatalogError(f"PINOCCHIO {key} must be an integer: {path}")
    return int(rounded)


def _validate_healpix_nside(nside: int) -> int:
    try:
        value = int(nside)
    except (TypeError, ValueError) as exc:
        raise PinocchioCatalogError("HEALPix nside must be an integer") from exc
    if value <= 0:
        raise PinocchioCatalogError("HEALPix nside must be positive")
    if value & (value - 1):
        raise PinocchioCatalogError("HEALPix nside must be a power of two")
    return value


def _integer_column(column: np.ndarray, path: Path, label: str) -> np.ndarray:
    rounded = np.rint(column)
    if not np.allclose(column, rounded, rtol=0.0, atol=1.0e-8):
        raise PinocchioCatalogError(f"PINOCCHIO {label} must contain integer values: {path}")
    return rounded.astype(np.int64)


def _require_finite(values: np.ndarray, path: Path, label: str) -> None:
    if not np.all(np.isfinite(values)):
        raise PinocchioCatalogError(f"PINOCCHIO {label} contains non-finite values: {path}")


def _require_positive(values: np.ndarray, path: Path, label: str) -> None:
    if np.any(values <= 0.0):
        raise PinocchioCatalogError(f"PINOCCHIO {label} must be positive: {path}")


def _wrap_positions(positions: np.ndarray, box_size_mpc_h: float) -> np.ndarray:
    if box_size_mpc_h <= 0.0:
        raise PinocchioCatalogError("wrap_box_size_mpc_h must be positive")
    return np.mod(positions, box_size_mpc_h)


def _parse_redshift_from_header(path: Path) -> float | None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return None
    for line in lines:
        if not line.lstrip().startswith("#"):
            break
        match = re.search(r"redshift\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)", line)
        if match:
            return float(match.group(1))
    return None


def _parse_snapshot_redshift(path: Path) -> float | None:
    try:
        files = _pinocchio_output_files(path, label="snapshot catalog")
    except PinocchioCatalogError:
        files = [path]
    for file in files:
        redshift = _parse_redshift_from_header(file)
        if redshift is not None:
            return redshift
    return _parse_snapshot_redshift_from_filename(path)


def _parse_snapshot_redshift_from_filename(path: Path) -> float | None:
    match = re.search(
        r"pinocchio\.([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\..*\.catalog\.out(?:\.\d+)?$",
        path.name,
    )
    if match:
        return float(match.group(1))
    return None


def _required_header_int(header: Mapping[str, Any], key: str, path: Path) -> int:
    value = _optional_header_int(header, key)
    if value is None:
        raise PinocchioCatalogError(f"Mass-map FITS header is missing {key}: {path}")
    return value


def _required_header_str(header: Mapping[str, Any], key: str, path: Path) -> str:
    value = _optional_header_str(header, key)
    if value is None:
        raise PinocchioCatalogError(f"Mass-map FITS header is missing {key}: {path}")
    return value


def _optional_header_int(header: Mapping[str, Any], key: str) -> int | None:
    return int(header[key]) if key in header else None


def _optional_header_float(header: Mapping[str, Any], key: str) -> float | None:
    return float(header[key]) if key in header else None


def _optional_header_str(header: Mapping[str, Any], key: str) -> str | None:
    return str(header[key]) if key in header else None


def _optional_axis_vector(header: Mapping[str, Any]) -> np.ndarray | None:
    keys = ("AXISV1", "AXISV2", "AXISV3")
    if not all(key in header for key in keys):
        return None
    return np.asarray([float(header[key]) for key in keys], dtype=np.float64)
