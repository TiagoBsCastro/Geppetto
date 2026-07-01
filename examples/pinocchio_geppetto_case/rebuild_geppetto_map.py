"""Rebuild the stored GEPPETTO one-halo HEALPix map for this example case."""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from astropy.io import fits

from geppetto import paint_lightcone_particle_count_map
from geppetto.io import (
    healpix_pixel_area_sr,
    read_pinocchio_lightcone_catalog,
    read_pinocchio_parameter_file,
)

CASE_DIR = Path(__file__).resolve().parent
RUN_FLAG = "example"
NSIDE = 256
PLC_SLICE = CASE_DIR / "pinocchio.example.plc.slice32.out"
PARAMETER_FILE = CASE_DIR / "parameter_file"
OUTPUT_FITS = CASE_DIR / "geppetto.example.one_halo_counts.nside256.seg000.fits"
SUMMARY_JSON = CASE_DIR / "geppetto.example.one_halo_counts.nside256.seg000.summary.json"


def healpix_ring_pixel_unit_vectors(nside: int, pixels: np.ndarray) -> np.ndarray:
    """Return RING-order HEALPix pixel-centre vectors without requiring healpy.

    This local fallback is used only to make the checked-in example runnable in
    minimal environments. Library users should prefer
    ``geppetto.io.healpix_pixel_unit_vectors`` when ``healpy`` is installed.
    """

    pixel_values = np.asarray(pixels, dtype=np.int64)
    npix = 12 * nside * nside
    ncap = 2 * nside * (nside - 1)
    vectors = np.empty((pixel_values.size, 3), dtype=np.float64)

    for index, pixel in enumerate(pixel_values):
        if pixel < 0 or pixel >= npix:
            raise ValueError(f"pixel index {pixel} is outside [0, {npix})")

        if pixel < ncap:
            ring = int(0.5 * (1.0 + np.sqrt(1.0 + 2.0 * pixel)))
            phi_index = pixel - 2 * ring * (ring - 1) + 1
            z = 1.0 - ring * ring / (3.0 * nside * nside)
            phi = (phi_index - 0.5) * np.pi / (2.0 * ring)
        elif pixel < npix - ncap:
            offset = pixel - ncap
            ring = offset // (4 * nside) + nside
            phi_index = offset % (4 * nside) + 1
            half_shift = 0.5 * (1 + ((ring + nside) & 1))
            z = (2 * nside - ring) * 2.0 / (3.0 * nside)
            phi = (phi_index - half_shift) * np.pi / (2.0 * nside)
        else:
            offset = npix - pixel
            ring = int(0.5 * (1.0 + np.sqrt(2.0 * offset - 1.0)))
            phi_index = 4 * ring + 1 - (offset - 2 * ring * (ring - 1))
            z = -1.0 + ring * ring / (3.0 * nside * nside)
            phi = (phi_index - 0.5) * np.pi / (2.0 * ring)

        sin_theta = np.sqrt(max(0.0, 1.0 - z * z))
        vectors[index] = [sin_theta * np.cos(phi), sin_theta * np.sin(phi), z]

    return vectors


def write_healpix_table(path: Path, pixels: np.ndarray, values: np.ndarray) -> None:
    """Write a HEALPix-compatible FITS binary table."""

    table = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="PIXEL", format="1J", array=pixels.astype(np.int32)),
            fits.Column(name="TEMPERATURE", format="1D", array=values.astype(np.float64)),
        ],
        name="HEALPIX",
    )
    table.header["PIXTYPE"] = "HEALPIX"
    table.header["ORDERING"] = "RING"
    table.header["NSIDE"] = NSIDE
    table.header["INDXSCHM"] = "EXPLICIT"
    table.header["FIRSTPIX"] = int(pixels[0])
    table.header["LASTPIX"] = int(pixels[-1])
    table.header["RUNFLAG"] = RUN_FLAG
    table.header["MAPTYPE"] = "GEPPETTO_ONE_HALO_COUNT"
    table.header["COMMENT"] = "TEMPERATURE is projected NFW mass per pixel / particle mass"
    fits.HDUList([fits.PrimaryHDU(), table]).writeto(path, overwrite=True)


def main() -> None:
    metadata = read_pinocchio_parameter_file(PARAMETER_FILE)
    plc = read_pinocchio_lightcone_catalog(PLC_SLICE)
    catalog = plc.to_lightcone_catalog(redshift="true")

    pixels = np.arange(12 * NSIDE * NSIDE, dtype=np.int64)
    pixel_unit_vectors = healpix_ring_pixel_unit_vectors(NSIDE, pixels)
    counts = paint_lightcone_particle_count_map(
        jnp.asarray(pixel_unit_vectors),
        catalog,
        particle_mass_msun_h=metadata.particle_mass_msun_h,
        pixel_area_sr=healpix_pixel_area_sr(NSIDE),
        cosmology=metadata.cosmology,
        chunk_size=4,
    )
    count_values = np.asarray(counts, dtype=np.float64)

    write_healpix_table(OUTPUT_FITS, pixels, count_values)
    summary = {
        "run_flag": RUN_FLAG,
        "nside": NSIDE,
        "n_pixels": int(pixels.size),
        "n_halos": len(plc),
        "particle_mass_msun_h": float(metadata.particle_mass_msun_h),
        "sum_count_equivalent": float(np.sum(count_values)),
        "max_count_equivalent": float(np.max(count_values)),
        "nonzero_pixels": int(np.count_nonzero(count_values)),
        "source_plc": PLC_SLICE.name,
        "output_fits": OUTPUT_FITS.name,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
