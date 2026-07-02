"""Bin resolved PLC halo mass into a PINOCCHIO mass-map segment pixel domain.

This diagnostic script reads a PINOCCHIO parameter file, mass-sheet table,
on-the-fly HEALPix mass-map FITS file, and PLC halo catalogue. It writes a map
with the same compact pixel domain and row ordering as the selected
``*.massmap.segXXX.fits`` file, but containing only resolved halo mass in
particle-count-equivalent units:

    halo contribution = halo_mass_msun_h / particle_mass_msun_h

The mandatory ``halo_particle_counts`` output does not paint NFW or tabulated
profiles. For tutorial use, ``--nfw-gradient-demo`` separately paints an NFW
one-halo map on the same selected segment and compact pixel domain, then reports
gradients of the total NFW particle-count map with respect to selected NFW
parameters.

Coordinate-basis warning
------------------------
The script assumes that the PLC halo catalogue directions and the PINOCCHIO
mass-map pixels are expressed in the same internal PLC basis. This should be
true when both files come from the same PINOCCHIO PLC run. If the halo catalogue
has been converted to an external sky basis, rotate it back into the PINOCCHIO
PLC basis before binning.

Example
-------
python examples/paint_halo_particles_for_pinocchio_segment.py \\
  --params path/to/parameter_file.params \\
  --sheets path/to/pinocchio.RUN.sheets.out \\
  --mass-map path/to/pinocchio.RUN.massmap.seg000.fits \\
  --plc-catalog path/to/pinocchio.RUN.plc.out \\
  --sheet-index 0 \\
  --output path/to/halo_particles.seg000.npz

Add ``--nfw-gradient-demo`` to also print and save a differentiability tutorial
for an NFW one-halo map evaluated on the same segment and pixel domain.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from geppetto import (
    ConcentrationParams,
    NFWProfileParams,
    paint_lightcone_particle_count_map,
)
from geppetto.catalog import LightconeHaloCatalog
from geppetto.io import (
    PinocchioMassMap,
    PinocchioRunMetadata,
    healpix_pixel_area_sr,
    healpix_pixel_unit_vectors,
    read_pinocchio_hubble_table,
    read_pinocchio_lightcone_catalog,
    read_pinocchio_lightcone_light_catalog,
    read_pinocchio_mass_map_fits,
    read_pinocchio_mass_sheets,
    read_pinocchio_parameter_file,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Bin resolved PINOCCHIO PLC halo masses into the pixel domain of one "
            "on-the-fly mass-map segment."
        )
    )
    parser.add_argument("--params", type=Path, required=True, help="PINOCCHIO parameter file")
    parser.add_argument("--sheets", type=Path, required=True, help="PINOCCHIO *.sheets.out file")
    parser.add_argument(
        "--mass-map",
        type=Path,
        required=True,
        help="PINOCCHIO *.massmap.segXXX.fits file",
    )
    parser.add_argument("--plc-catalog", type=Path, required=True, help="PINOCCHIO PLC catalogue")
    parser.add_argument("--sheet-index", type=int, required=True, help="Mass-sheet row index")
    parser.add_argument("--output", type=Path, required=True, help="Output .npz file")
    parser.add_argument(
        "--catalog-format",
        choices=("auto", "ascii", "binary"),
        default="auto",
        help="PLC catalogue format",
    )
    parser.add_argument(
        "--redshift-mode",
        choices=("true", "observed"),
        default="true",
        help="Halo redshift column to use",
    )
    parser.add_argument(
        "--bounds",
        choices=("z", "chi"),
        default="z",
        help="Segment-bound coordinate used to select halos",
    )
    parser.add_argument(
        "--hubble-table",
        type=Path,
        help="PINOCCHIO Hubble table required for --light-plc",
    )
    parser.add_argument(
        "--light-plc",
        action="store_true",
        help="Read a light PLC catalogue without Cartesian positions",
    )
    parser.add_argument("--output-fits", type=Path, help="Optional output HEALPix FITS table")
    parser.add_argument(
        "--last-segment-inclusive",
        action="store_true",
        help="Use an inclusive upper segment bound",
    )
    parser.add_argument(
        "--nfw-gradient-demo",
        action="store_true",
        help=(
            "Also paint an NFW one-halo map on the same selected halos and pixel "
            "domain, then print/save gradients of its total count with respect "
            "to NFW parameters."
        ),
    )
    parser.add_argument(
        "--nfw-concentration-amplitude",
        type=float,
        default=5.71,
        help="Concentration amplitude used for --nfw-gradient-demo",
    )
    parser.add_argument(
        "--nfw-truncation-width-fraction",
        type=float,
        default=0.05,
        help="NFW smooth truncation-width fraction used for --nfw-gradient-demo",
    )
    parser.add_argument(
        "--nfw-chunk-size",
        type=int,
        default=1024,
        help="Static halo chunk size for --nfw-gradient-demo; use 0 to disable chunking",
    )
    return parser.parse_args()


def load_lightcone_catalog(args: argparse.Namespace) -> LightconeHaloCatalog:
    """Load a full or light PINOCCHIO PLC catalogue as a GEPPETTO catalogue."""

    if args.light_plc:
        if args.hubble_table is None:
            raise ValueError("--hubble-table is required when --light-plc is used")
        raw = read_pinocchio_lightcone_light_catalog(
            args.plc_catalog,
            format=args.catalog_format,
        )
        distance_interpolator = read_pinocchio_hubble_table(args.hubble_table)
        return raw.to_lightcone_catalog(
            distance_interpolator,
            redshift=args.redshift_mode,
        )

    raw = read_pinocchio_lightcone_catalog(args.plc_catalog, format=args.catalog_format)
    return raw.to_lightcone_catalog(redshift=args.redshift_mode)


def segment_bounds(sheets: Any, sheet_index: int) -> dict[str, float]:
    """Return sorted redshift, scale-factor, and distance bounds for one sheet."""

    n_sheet = len(sheets)
    if sheet_index < 0 or sheet_index >= n_sheet:
        raise ValueError(f"sheet_index {sheet_index} is outside [0, {n_sheet})")

    z_lo = min(float(sheets.z_lo[sheet_index]), float(sheets.z_hi[sheet_index]))
    z_hi = max(float(sheets.z_lo[sheet_index]), float(sheets.z_hi[sheet_index]))
    chi_lo = min(
        float(sheets.chi_lo_mpc_h[sheet_index]),
        float(sheets.chi_hi_mpc_h[sheet_index]),
    )
    chi_hi = max(
        float(sheets.chi_lo_mpc_h[sheet_index]),
        float(sheets.chi_hi_mpc_h[sheet_index]),
    )
    return {
        "sheet_index": int(sheet_index),
        "z_lo": z_lo,
        "z_hi": z_hi,
        "a_lo": 1.0 / (1.0 + z_hi),
        "a_hi": 1.0 / (1.0 + z_lo),
        "chi_lo_mpc_h": chi_lo,
        "chi_hi_mpc_h": chi_hi,
    }


def validate_mass_map(mass_map: PinocchioMassMap) -> None:
    """Check the mass-map fields needed for compact RING pixel binning."""

    if mass_map.ordering.upper() != "RING":
        raise ValueError(f"Only RING mass maps are supported, got ORDERING={mass_map.ordering!r}")

    pixels = np.asarray(mass_map.pixel)
    temperature = np.asarray(mass_map.temperature)
    if pixels.ndim != 1:
        raise ValueError("mass_map.pixel must be one-dimensional")
    if temperature.ndim != 1:
        raise ValueError("mass_map.temperature must be one-dimensional")
    if pixels.shape != temperature.shape:
        raise ValueError("mass_map.pixel and mass_map.temperature must have the same length")


def validate_catalog_for_binning(catalog: LightconeHaloCatalog) -> None:
    """Check catalogue array shapes before segment selection and binning."""

    mass = np.asarray(catalog.mass)
    redshift = np.asarray(catalog.redshift)
    chi = np.asarray(catalog.chi)
    unit_vector = np.asarray(catalog.unit_vector)

    if mass.ndim != 1:
        raise ValueError("catalog.mass must have shape (n_halo,)")
    n_halo = int(mass.shape[0])
    if redshift.ndim != 1 or redshift.shape[0] != n_halo:
        raise ValueError("catalog.redshift must have shape matching catalog.mass")
    if chi.ndim != 1 or chi.shape[0] != n_halo:
        raise ValueError("catalog.chi must have shape matching catalog.mass")
    if unit_vector.ndim != 2 or unit_vector.shape != (n_halo, 3):
        raise ValueError("catalog.unit_vector must have shape (n_halo, 3)")
    if not np.all(np.isfinite(mass)):
        raise ValueError("catalog.mass values must be finite")
    if not np.all(np.isfinite(redshift)):
        raise ValueError("catalog.redshift values must be finite")
    if not np.all(np.isfinite(chi)):
        raise ValueError("catalog.chi values must be finite")
    if not np.all(np.isfinite(unit_vector)):
        raise ValueError("catalog.unit_vector values must be finite")


def select_segment_mask(
    catalog: LightconeHaloCatalog,
    bounds: dict[str, float],
    mode: str,
    inclusive_upper: bool,
) -> np.ndarray:
    """Select halos inside the requested segment bounds."""

    if mode == "z":
        values = np.asarray(catalog.redshift)
        lo = bounds["z_lo"]
        hi = bounds["z_hi"]
    elif mode == "chi":
        values = np.asarray(catalog.chi)
        lo = bounds["chi_lo_mpc_h"]
        hi = bounds["chi_hi_mpc_h"]
    else:
        raise ValueError("mode must be 'z' or 'chi'")

    if inclusive_upper:
        return (values >= lo) & (values <= hi)
    return (values >= lo) & (values < hi)


def halo_rows_in_mass_map(
    catalog: LightconeHaloCatalog,
    mask: np.ndarray,
    mass_map: PinocchioMassMap,
) -> tuple[np.ndarray, np.ndarray]:
    """Map selected halo directions to compact mass-map rows."""

    try:
        import healpy as hp
    except ImportError as exc:  # pragma: no cover - exercised only without io extra
        raise RuntimeError("halo pixel binning requires healpy; install geppetto[io]") from exc

    validate_mass_map(mass_map)
    validate_catalog_for_binning(catalog)

    mass = np.asarray(catalog.mass)
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 1 or mask.shape[0] != mass.shape[0]:
        raise ValueError("mask must have shape (n_halo,)")

    uv = np.asarray(catalog.unit_vector)[mask]
    if uv.shape[0] == 0:
        empty_rows = np.empty((0,), dtype=np.int64)
        empty_inside = np.empty((0,), dtype=bool)
        return empty_rows, empty_inside

    halo_pix = hp.vec2pix(mass_map.nside, uv[:, 0], uv[:, 1], uv[:, 2], nest=False)
    pixel_to_row = {int(pixel): row for row, pixel in enumerate(np.asarray(mass_map.pixel))}
    rows = np.array([pixel_to_row.get(int(pixel), -1) for pixel in halo_pix], dtype=np.int64)
    inside_pixel_domain = rows >= 0
    return rows, inside_pixel_domain


def _accumulate_halo_particle_counts(
    catalog: LightconeHaloCatalog,
    mask: np.ndarray,
    mass_map: PinocchioMassMap,
    particle_mass_msun_h: float,
    rows: np.ndarray,
    inside_pixel_domain: np.ndarray,
) -> np.ndarray:
    if particle_mass_msun_h <= 0.0:
        raise ValueError("particle_mass_msun_h must be positive")

    out = np.zeros(np.asarray(mass_map.temperature).shape, dtype=np.float64)
    selected_masses = np.asarray(catalog.mass)[mask][inside_pixel_domain]
    selected_rows = rows[inside_pixel_domain]
    np.add.at(out, selected_rows, selected_masses / particle_mass_msun_h)
    return out


def build_halo_particle_count_map(
    catalog: LightconeHaloCatalog,
    mask: np.ndarray,
    mass_map: PinocchioMassMap,
    particle_mass_msun_h: float,
) -> np.ndarray:
    """Build the compact halo-particle-count map for selected halos."""

    rows, inside_pixel_domain = halo_rows_in_mass_map(catalog, mask, mass_map)
    return _accumulate_halo_particle_counts(
        catalog,
        np.asarray(mask, dtype=bool),
        mass_map,
        particle_mass_msun_h,
        rows,
        inside_pixel_domain,
    )


def selected_lightcone_catalog(catalog: LightconeHaloCatalog, mask: np.ndarray) -> LightconeHaloCatalog:
    """Return the segment-selected catalogue as JAX arrays for differentiable painters."""

    mask = np.asarray(mask, dtype=bool)
    return LightconeHaloCatalog(
        unit_vector=jnp.asarray(np.asarray(catalog.unit_vector)[mask]),
        chi=jnp.asarray(np.asarray(catalog.chi)[mask]),
        mass=jnp.asarray(np.asarray(catalog.mass)[mask]),
        redshift=jnp.asarray(np.asarray(catalog.redshift)[mask]),
    )


def nfw_gradient_demo(
    catalog: LightconeHaloCatalog,
    mask: np.ndarray,
    mass_map: PinocchioMassMap,
    metadata: PinocchioRunMetadata,
    particle_mass_msun_h: float,
    *,
    concentration_amplitude: float = 5.71,
    truncation_width_fraction: float = 0.05,
    chunk_size: int | None = 1024,
) -> dict[str, float | int]:
    """Paint an NFW map and differentiate its total count with respect to parameters.

    This tutorial helper uses the same selected segment and compact
    ``mass_map.pixel`` domain as the point-count diagnostic, but it is separate
    from the mandatory ``halo_particle_counts`` output.
    """

    if particle_mass_msun_h <= 0.0:
        raise ValueError("particle_mass_msun_h must be positive")
    if chunk_size is not None and chunk_size <= 0:
        chunk_size = None

    selected_catalog = selected_lightcone_catalog(catalog, mask)
    pixel_unit_vectors = jnp.asarray(
        healpix_pixel_unit_vectors(mass_map.nside, np.asarray(mass_map.pixel), nest=False)
    )
    pixel_area_sr = healpix_pixel_area_sr(mass_map.nside)

    def total_for_concentration_amplitude(amplitude: float):
        counts = paint_lightcone_particle_count_map(
            pixel_unit_vectors,
            selected_catalog,
            particle_mass_msun_h=particle_mass_msun_h,
            pixel_area_sr=pixel_area_sr,
            cosmology=metadata.cosmology,
            concentration_params=ConcentrationParams(amplitude=amplitude),
            profile_params=NFWProfileParams(
                truncation_width_fraction=truncation_width_fraction
            ),
            chunk_size=chunk_size,
        )
        return jnp.sum(counts)

    def total_for_truncation_width_fraction(width_fraction: float):
        counts = paint_lightcone_particle_count_map(
            pixel_unit_vectors,
            selected_catalog,
            particle_mass_msun_h=particle_mass_msun_h,
            pixel_area_sr=pixel_area_sr,
            cosmology=metadata.cosmology,
            concentration_params=ConcentrationParams(amplitude=concentration_amplitude),
            profile_params=NFWProfileParams(
                truncation_width_fraction=width_fraction
            ),
            chunk_size=chunk_size,
        )
        return jnp.sum(counts)

    total_counts = total_for_concentration_amplitude(concentration_amplitude)
    grad_concentration = jax.grad(total_for_concentration_amplitude)(concentration_amplitude)
    grad_truncation_width = jax.grad(total_for_truncation_width_fraction)(
        truncation_width_fraction
    )
    return {
        "nfw_gradient_demo_n_halos": int(selected_catalog.mass.shape[0]),
        "nfw_sum_particle_counts": float(total_counts),
        "nfw_concentration_amplitude": float(concentration_amplitude),
        "nfw_d_sum_d_concentration_amplitude": float(grad_concentration),
        "nfw_truncation_width_fraction": float(truncation_width_fraction),
        "nfw_d_sum_d_truncation_width_fraction": float(grad_truncation_width),
    }


def diagnostics_for_map(
    catalog: LightconeHaloCatalog,
    mask: np.ndarray,
    mass_map: PinocchioMassMap,
    out: np.ndarray,
    particle_mass_msun_h: float,
    inside_pixel_domain: np.ndarray,
) -> dict[str, float | int]:
    """Return scalar diagnostics for printed and saved summaries."""

    return {
        "particle_mass_msun_h": float(particle_mass_msun_h),
        "n_halos_total": int(np.asarray(catalog.mass).shape[0]),
        "n_halos_in_segment": int(np.count_nonzero(mask)),
        "n_halos_in_segment_and_pixels": int(np.count_nonzero(inside_pixel_domain)),
        "sum_halo_particle_counts": float(np.sum(out)),
        "sum_pinocchio_mass_map_values": float(np.sum(mass_map.temperature)),
    }


def save_npz(
    args: argparse.Namespace,
    out: np.ndarray,
    mass_map: PinocchioMassMap,
    bounds: dict[str, float],
    metadata: PinocchioRunMetadata,
    diagnostics: dict[str, float | int],
    nfw_diagnostics: dict[str, float | int] | None = None,
) -> None:
    """Save the diagnostic map and metadata to the requested ``.npz`` file."""

    payload = {
        "halo_particle_counts": out,
        "pinocchio_mass_map_values": np.asarray(mass_map.temperature),
        "pixel": np.asarray(mass_map.pixel),
        "nside": int(mass_map.nside),
        "ordering": mass_map.ordering,
        "sheet_index": int(bounds["sheet_index"]),
        "z_lo": float(bounds["z_lo"]),
        "z_hi": float(bounds["z_hi"]),
        "a_lo": float(bounds["a_lo"]),
        "a_hi": float(bounds["a_hi"]),
        "chi_lo_mpc_h": float(bounds["chi_lo_mpc_h"]),
        "chi_hi_mpc_h": float(bounds["chi_hi_mpc_h"]),
        "particle_mass_msun_h": float(metadata.particle_mass_msun_h),
        "n_halos_total": int(diagnostics["n_halos_total"]),
        "n_halos_in_segment": int(diagnostics["n_halos_in_segment"]),
        "n_halos_in_segment_and_pixels": int(diagnostics["n_halos_in_segment_and_pixels"]),
        "sum_halo_particle_counts": float(diagnostics["sum_halo_particle_counts"]),
        "sum_pinocchio_mass_map_values": float(diagnostics["sum_pinocchio_mass_map_values"]),
    }
    if nfw_diagnostics is not None:
        payload.update(nfw_diagnostics)

    np.savez(args.output, **payload)


def _pixel_column_format(pixels: np.ndarray) -> tuple[str, np.ndarray]:
    if pixels.size and (
        np.min(pixels) < np.iinfo(np.int32).min or np.max(pixels) > np.iinfo(np.int32).max
    ):
        return "1K", pixels.astype(np.int64)
    return "1J", pixels.astype(np.int32)


def write_output_fits(
    path: Path,
    out: np.ndarray,
    mass_map: PinocchioMassMap,
    bounds: dict[str, float],
    diagnostics: dict[str, float | int],
) -> None:
    """Write an optional HEALPix FITS binary table matching the compact domain."""

    try:
        from astropy.io import fits
    except ImportError as exc:  # pragma: no cover - exercised only without io extra
        raise RuntimeError("FITS output requires astropy; install geppetto[io]") from exc

    pixels = np.asarray(mass_map.pixel, dtype=np.int64)
    pixel_format, pixel_values = _pixel_column_format(pixels)
    table = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="PIXEL", format=pixel_format, array=pixel_values),
            fits.Column(name="TEMPERATURE", format="1D", array=out.astype(np.float64)),
        ],
        name="HEALPIX",
    )

    reserved_keys = {
        "XTENSION",
        "BITPIX",
        "NAXIS",
        "NAXIS1",
        "NAXIS2",
        "PCOUNT",
        "GCOUNT",
        "TFIELDS",
        "EXTNAME",
        "CHECKSUM",
        "DATASUM",
    }
    reserved_prefixes = ("TTYPE", "TFORM", "TUNIT", "TDIM", "TSCAL", "TZERO", "TNULL", "TDISP")
    for key, value in mass_map.header.items():
        key = str(key).upper()
        if key in reserved_keys or any(key.startswith(prefix) for prefix in reserved_prefixes):
            continue
        try:
            table.header[key] = value
        except (TypeError, ValueError):
            continue

    table.header["PIXTYPE"] = "HEALPIX"
    table.header["ORDERING"] = mass_map.ordering
    table.header["NSIDE"] = int(mass_map.nside)
    table.header["INDXSCHM"] = mass_map.index_scheme or "EXPLICIT"
    table.header["SHEETIDX"] = int(bounds["sheet_index"])
    table.header["ZLO"] = float(bounds["z_lo"])
    table.header["ZHI"] = float(bounds["z_hi"])
    table.header["ALO"] = float(bounds["a_lo"])
    table.header["AHI"] = float(bounds["a_hi"])
    table.header["CHILO"] = float(bounds["chi_lo_mpc_h"])
    table.header["CHIHI"] = float(bounds["chi_hi_mpc_h"])
    table.header["PMASS"] = float(diagnostics["particle_mass_msun_h"])
    table.header["NHALSEG"] = int(diagnostics["n_halos_in_segment"])
    table.header["NHALPIX"] = int(diagnostics["n_halos_in_segment_and_pixels"])
    table.header["MAPTYPE"] = "HALO_PARTICLE_COUNT"
    table.header["COMMENT"] = "Diagnostic halo-catalog particle-count map."
    table.header["COMMENT"] = "TEMPERATURE is halo mass / PINOCCHIO particle mass."
    table.header["COMMENT"] = "This is not the original PINOCCHIO on-the-fly mass map."
    fits.HDUList([fits.PrimaryHDU(), table]).writeto(path, overwrite=True)


def _range_text(lo: float, hi: float, inclusive_upper: bool) -> str:
    right = "]" if inclusive_upper else ")"
    return f"[{lo:g}, {hi:g}{right}"


def print_segment_summary(bounds: dict[str, float], inclusive_upper: bool) -> None:
    """Print selected segment bounds."""

    print("Selected segment:")
    print(f"  sheet_index = {int(bounds['sheet_index'])}")
    print(f"  z range     = {_range_text(bounds['z_lo'], bounds['z_hi'], inclusive_upper)}")
    print(f"  a range     = {_range_text(bounds['a_lo'], bounds['a_hi'], inclusive_upper)}")
    print(
        "  chi range   = "
        f"{_range_text(bounds['chi_lo_mpc_h'], bounds['chi_hi_mpc_h'], inclusive_upper)} Mpc/h"
    )


def print_output_summary(
    diagnostics: dict[str, float | int],
    out: np.ndarray,
    mass_map: PinocchioMassMap,
) -> None:
    """Print the final comparison summary."""

    print("Halo particle-count map summary:")
    print(f"  Read {diagnostics['n_halos_total']} total halos")
    print(f"  Selected {diagnostics['n_halos_in_segment']} halos in segment bounds")
    print(f"  Kept {diagnostics['n_halos_in_segment_and_pixels']} halos inside map pixel domain")
    print(f"  Output pixels: {len(out)}")
    print(f"  PINOCCHIO map pixels: {len(mass_map.temperature)}")
    print(f"  Sum halo particle counts: {diagnostics['sum_halo_particle_counts']:.12g}")
    print(
        "  Sum PINOCCHIO on-the-fly map values: "
        f"{diagnostics['sum_pinocchio_mass_map_values']:.12g}"
    )


def print_nfw_gradient_summary(nfw_diagnostics: dict[str, float | int]) -> None:
    """Print the optional NFW differentiability tutorial summary."""

    print("NFW differentiability demo:")
    print("  Objective: sum of NFW one-halo particle-count map on the same compact pixels")
    print(f"  Selected halos painted: {nfw_diagnostics['nfw_gradient_demo_n_halos']}")
    print(f"  NFW sum particle counts: {nfw_diagnostics['nfw_sum_particle_counts']:.12g}")
    print(
        "  d(sum) / d concentration amplitude: "
        f"{nfw_diagnostics['nfw_d_sum_d_concentration_amplitude']:.12g}"
    )
    print(
        "  d(sum) / d truncation width fraction: "
        f"{nfw_diagnostics['nfw_d_sum_d_truncation_width_fraction']:.12g}"
    )


def main() -> None:
    """Run the diagnostic command-line workflow."""

    args = parse_args()
    metadata = read_pinocchio_parameter_file(args.params)
    particle_mass = float(metadata.particle_mass_msun_h)
    if particle_mass <= 0.0:
        raise ValueError("particle_mass_msun_h must be positive")

    sheets = read_pinocchio_mass_sheets(args.sheets)
    bounds = segment_bounds(sheets, args.sheet_index)

    mass_map = read_pinocchio_mass_map_fits(args.mass_map)
    validate_mass_map(mass_map)

    catalog = load_lightcone_catalog(args)
    validate_catalog_for_binning(catalog)
    mask = select_segment_mask(
        catalog,
        bounds,
        mode=args.bounds,
        inclusive_upper=args.last_segment_inclusive,
    )

    print_segment_summary(bounds, args.last_segment_inclusive)
    rows, inside_pixel_domain = halo_rows_in_mass_map(catalog, mask, mass_map)
    out = _accumulate_halo_particle_counts(
        catalog,
        mask,
        mass_map,
        particle_mass,
        rows,
        inside_pixel_domain,
    )

    if out.shape != np.asarray(mass_map.temperature).shape:
        raise RuntimeError("output map shape does not match mass_map.temperature")
    if len(out) != len(mass_map.pixel):
        raise RuntimeError("output map length does not match mass_map.pixel")

    diagnostics = diagnostics_for_map(
        catalog,
        mask,
        mass_map,
        out,
        particle_mass,
        inside_pixel_domain,
    )
    nfw_diagnostics = None
    if args.nfw_gradient_demo:
        nfw_diagnostics = nfw_gradient_demo(
            catalog,
            mask,
            mass_map,
            metadata,
            particle_mass,
            concentration_amplitude=args.nfw_concentration_amplitude,
            truncation_width_fraction=args.nfw_truncation_width_fraction,
            chunk_size=args.nfw_chunk_size,
        )

    save_npz(args, out, mass_map, bounds, metadata, diagnostics, nfw_diagnostics)
    if args.output_fits is not None:
        write_output_fits(args.output_fits, out, mass_map, bounds, diagnostics)

    print_output_summary(diagnostics, out, mass_map)
    if nfw_diagnostics is not None:
        print_nfw_gradient_summary(nfw_diagnostics)
    print(f"Wrote NPZ: {args.output}")
    if args.output_fits is not None:
        print(f"Wrote FITS: {args.output_fits}")


if __name__ == "__main__":
    main()
