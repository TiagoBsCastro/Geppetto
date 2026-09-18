#!/usr/bin/env python3
"""Audit catalogue-conditioned halo mass moments without reading painted C_ell.

Read one PLC part at a time, bin count/M/M^2 by the original painting shells,
and verify the catalogue selection against the manifest before saving results.
This is an abundance diagnostic, not a calibration to the painted map power.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from time import perf_counter

import numpy as np

from geppetto.io import read_pinocchio_lightcone_catalog, read_pinocchio_parameter_file
from paint_halo_particles_for_pinocchio_segment import discover_plc_catalog_parts


def halo_mass_moments(
    mass: np.ndarray, redshift: np.ndarray, z_edges: np.ndarray, log_mass_edges: np.ndarray,
    *, inclusive_upper: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return count, sum M, and sum M^2 on a redshift/log-mass grid.

    Mass units are Msun/h. Shells are lower-inclusive and upper-exclusive,
    except an explicitly included final upper boundary. No halo is assigned
    to an adjacent mass bin when the requested mass grid does not cover it.
    """

    mass, redshift = np.asarray(mass), np.asarray(redshift)
    z_edges, log_mass_edges = np.asarray(z_edges), np.asarray(log_mass_edges)
    if mass.ndim != 1 or mass.shape != redshift.shape:
        raise ValueError("mass and redshift must be matching vectors")
    if np.any(~np.isfinite(mass)) or np.any(mass <= 0) or np.any(~np.isfinite(redshift)):
        raise ValueError("catalogue masses/redshifts must be finite, with positive masses")
    for edges in (z_edges, log_mass_edges):
        if edges.ndim != 1 or len(edges) < 2 or not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
            raise ValueError("histogram edges must be finite and strictly increasing")
    selected = (redshift >= z_edges[0]) & (
        (redshift <= z_edges[-1]) if inclusive_upper else (redshift < z_edges[-1])
    )
    log_mass = np.log(mass[selected])
    if np.any(log_mass < log_mass_edges[0]) or np.any(log_mass > log_mass_edges[-1]):
        raise ValueError("mass grid does not cover every selected halo")
    points = np.column_stack((redshift[selected], log_mass))
    return tuple(np.histogramdd(points, bins=(z_edges, log_mass_edges), weights=weight)[0]
                 for weight in (None, mass[selected], mass[selected] ** 2))


def run(args: argparse.Namespace) -> Path:
    if args.redshift_bins < 1 or args.mass_bins < 1:
        raise ValueError("bin counts must be positive")
    with args.manifest.open(newline="") as stream:
        rows = sorted(csv.DictReader(stream), key=lambda row: int(row["segment_index"]))
    if not rows:
        raise ValueError("manifest is empty")
    metadata = read_pinocchio_parameter_file(args.params)
    mode = {row["redshift_mode"] for row in rows}
    if mode not in ({"true"}, {"observed"}):
        raise ValueError("manifest must use a single supported redshift mode")
    z_edges = np.array([np.linspace(float(row["z_lo"]), float(row["z_hi"]), args.redshift_bins + 1)
                        for row in rows])
    log_mass_edges = np.linspace(np.log(metadata.particle_mass_msun_h / 2), np.log(1e17), args.mass_bins + 1)
    moments = np.zeros((3, len(rows), args.redshift_bins, args.mass_bins))
    part_paths = discover_plc_catalog_parts(args.plc_catalog)
    started = perf_counter()
    for part_number, path in enumerate(part_paths, 1):
        catalogue = read_pinocchio_lightcone_catalog(path)
        redshift = catalogue.true_redshift if mode == {"true"} else catalogue.observed_redshift
        for index, row in enumerate(rows):
            upper = row.get("inclusive_upper", "False").lower() in {"true", "1"}
            moments[:, index] += np.array(halo_mass_moments(
                catalogue.masses_msun_h, redshift, z_edges[index], log_mass_edges,
                inclusive_upper=upper,
            ))
        print(f"[shell-hmf] part {part_number}/{len(part_paths)}; halos={len(catalogue)}; "
              f"elapsed={perf_counter()-started:.1f}s", flush=True)
        del catalogue
    count, mass_sum, mass_squared_sum = moments
    expected_count = np.array([int(row["selected_halo_count"]) for row in rows])
    expected_mass = np.array([float(row["selected_halo_mass_msun_h"]) for row in rows])
    if not np.array_equal(count.sum(axis=(1, 2)), expected_count):
        raise ValueError("PLC halo counts do not match the painting manifest; refusing stale/mismatched inputs")
    if not np.allclose(mass_sum.sum(axis=(1, 2)), expected_mass, rtol=1e-8, atol=0):
        raise ValueError("PLC halo masses do not match the painting manifest; refusing mismatched inputs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output, count=count, mass_sum=mass_sum, mass_squared_sum=mass_squared_sum,
        z_edges=z_edges, log_mass_edges=log_mass_edges,
        segment_indices=np.array([int(row["segment_index"]) for row in rows]),
        catalogue_parts=np.asarray([str(path) for path in part_paths]),
        particle_mass_msun_h=metadata.particle_mass_msun_h,
        manifest=np.asarray(str(args.manifest)), convention=np.asarray("catalogue_only_mass_moments"),
    )
    print(f"[shell-hmf] verified manifest selection and saved {args.output}", flush=True)
    return args.output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--plc-catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--redshift-bins", type=int, default=8)
    parser.add_argument("--mass-bins", type=int, default=200)
    run(parser.parse_args())
