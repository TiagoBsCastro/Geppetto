# PINOCCHIO + GEPPETTO HEALPix Output Case

This directory stores one compact, runnable example connecting native
PINOCCHIO outputs to a GEPPETTO one-halo HEALPix map.

## Contents

- `parameter_file`, `outputs`, and `run_geppetto_case.log`: PINOCCHIO run
  inputs and execution log.
- `pinocchio.0.0000.example.catalog.out`, `pinocchio.0.0500.example.catalog.out`,
  and `pinocchio.0.1000.example.catalog.out`: snapshot halo catalogues.
- `pinocchio.0.0000.example.mf.out`, `pinocchio.0.0500.example.mf.out`, and
  `pinocchio.0.1000.example.mf.out`: snapshot mass-function tables.
- `pinocchio.example.plc.out`: full PINOCCHIO PLC halo catalogue for this run.
- `pinocchio.example.massmap.seg000.fits` and
  `pinocchio.example.massmap.seg001.fits`: native PINOCCHIO HEALPix mass-map
  outputs for both mass sheets.
- `pinocchio.example.sheets.out` and `pinocchio.example.nz.out`: PINOCCHIO
  auxiliary PLC outputs for the same run.
- `pinocchio.example.FmaxPDF.out`, `pinocchio.example.cosmology.out`,
  `pinocchio.example.geometry.out`, and `pinocchio.example.histories.out`:
  remaining PINOCCHIO outputs for the same run.
- `pinocchio.example.plc.slice32.out`: first 32 rows of the full PINOCCHIO PLC
  catalogue, used only to keep the GEPPETTO regeneration script fast.
- `geppetto.example.one_halo_counts.nside256.seg000.fits`: GEPPETTO NFW
  one-halo count-equivalent HEALPix map generated from the PLC slice.
- `geppetto.example.one_halo_counts.nside256.seg000.summary.json`: summary of
  the GEPPETTO map.
- `rebuild_geppetto_map.py`: script that regenerates the GEPPETTO FITS map and
  summary from the stored PINOCCHIO slice.

## Regenerate GEPPETTO Output

From the repository root:

```bash
python examples/pinocchio_geppetto_case/rebuild_geppetto_map.py
```

The GEPPETTO map convention is:

```text
TEMPERATURE = projected NFW one-halo mass per pixel / PINOCCHIO particle mass
```

This is a count-equivalent one-halo mass collector. It is not the PINOCCHIO
two-halo/count map itself; it is the layer intended to be added after the map
normalization convention is fixed for a production workflow.

PLC halo directions are read from PINOCCHIO's `theta, phi` columns in the same
internal PLC angular basis used by the mass-map FITS pixels. The full PLC
Cartesian positions are still used to derive comoving radial distance.

## Provenance

The PINOCCHIO files come from a local rerun of the PINOCCHIO example using
`RunFlag example`, `GridSize 128`, `MassMapNSIDE 256`, and
`StartingzForPLC 0.1`. The `HubbleTableFile` line was commented in the copied
temporary parameter file because this compiled PINOCCHIO binary does not accept
that tag.
