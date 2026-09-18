#!/usr/bin/env python
"""Stage a hashed, unit-explicit subset of real full PINOCCHIO PLC parts.

This host-side adapter uses GEPPETTO's native reader, not a synthetic catalogue.
Positions remain observer-relative comoving Mpc/h, velocities proper km/s,
and M_PIN remains the native fragmentation-group particle-count mass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from geppetto.io import read_pinocchio_lightcone_catalog, read_pinocchio_parameter_file


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
        return digest.hexdigest()


def prepare(args: argparse.Namespace) -> dict:
    metadata = read_pinocchio_parameter_file(args.params)
    parameters = metadata.parameters
    base = Path(args.plc_catalog)
    files = [base] if base.is_file() else sorted(
        (p for p in base.parent.glob(base.name + ".*") if p.suffix[1:].isdigit()),
        key=lambda p: int(p.suffix[1:]),
    )
    if not files:
        raise ValueError(f"No native PLC parts found for {base}")
    if not base.is_file() and [int(p.suffix[1:]) for p in files] != list(range(len(files))):
        raise ValueError("Native PLC part indices must be contiguous, starting at zero")
    if not 0 <= args.z_min < args.z_max:
        raise ValueError("Require 0 <= z_min < z_max")
    factor = 1.0 if "OutputInH100" in parameters else metadata.cosmology.h
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite staged input {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    sources = []
    total = 0
    with h5py.File(output, "w") as handle:
        halos = handle.create_group("halos")
        for part, path in enumerate(files):
            catalog = read_pinocchio_lightcone_catalog(path, format=args.format)
            keep = (catalog.true_redshift >= args.z_min) & (catalog.true_redshift < args.z_max)
            rows = np.flatnonzero(keep)
            if len(catalog) >= 2**48 or part >= 2**16:
                raise ValueError("Catalogue exceeds unique occurrence-ID bit allocation")
            values = {
                "occurrence_id": np.uint64(part) * np.uint64(2**48) + rows.astype(np.uint64),
                "group_id": catalog.group_ids[keep],
                "M_PIN": factor * catalog.masses_msun_h[keep],
                "position": factor * catalog.positions_mpc_h[keep],
                "velocity": catalog.velocities_km_s[keep],
                "z_cos": catalog.true_redshift[keep],
                "z_obs_native": catalog.observed_redshift[keep],
                "theta_native_deg": catalog.theta_deg[keep],
                "phi_native_deg": catalog.phi_deg[keep],
                "v_los_native": catalog.los_velocity_km_s[keep],
            }
            for name, value in values.items():
                if name not in halos:
                    halos.create_dataset(name, shape=(0, *value.shape[1:]),
                                         maxshape=(None, *value.shape[1:]), dtype=value.dtype,
                                         chunks=True, compression="gzip", shuffle=True)
                dataset = halos[name]
                dataset.resize(total + len(rows), axis=0)
                dataset[total:] = value
            total += len(rows)
            sources.append(dict(path=str(path.resolve()), sha256=sha256(path),
                                rows_read=len(catalog), rows_retained=len(rows)))
            print(f"[galaxy input] part {part+1}/{len(files)}: {len(rows)}/{len(catalog)} retained", flush=True)
        provenance = dict(
            schema_version=1, source_format="PINOCCHIO full native PLC", sources=sources,
            parameters={key: list(value) for key, value in parameters.items()},
            parameter_sha256=sha256(Path(args.params)), cosmology_sha256=sha256(Path(args.cosmology_table)),
            z_min=args.z_min, z_max=args.z_max, n_halos=total,
            mass_definition="M_PIN = PINOCCHIO fragmentation-group particle count times particle mass; not spherical overdensity",
            mass_unit="Msun/h", position_unit="comoving Mpc/h, observer-relative simulation Cartesian basis",
            velocity_unit="proper peculiar km/s, simulation Cartesian basis",
            occurrence_id_definition="(source part index << 48) + zero-based native row index",
            input_to_h100_factor=factor,
        )
        handle.attrs["metadata_json"] = json.dumps(provenance, sort_keys=True)
    output.with_suffix(".json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", required=True)
    parser.add_argument("--cosmology-table", required=True)
    parser.add_argument("--plc-catalog", required=True)
    parser.add_argument("--format", choices=("auto", "binary", "ascii"), default="binary")
    parser.add_argument("--z-min", type=float, default=0.04)
    parser.add_argument("--z-max", type=float, default=0.33)
    parser.add_argument("--output", required=True)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
