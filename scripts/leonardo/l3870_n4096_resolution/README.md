# L3870 Resolution-Convergence Campaign

This campaign reruns the phase-matched `L=3870 Mpc/h` PINOCCHIO lightcone at
`GridSize=4096`, paints it with GEPPETTO, evaluates the standard angular-power
validation, and compares its measured spectra with the existing `2160^3` run.

## Resolution And Layout

- Low resolution: `2160^3`, particle spacing `1.7917 Mpc/h`,
  `k_Nyquist=1.7534 h/Mpc`.
- High resolution: `4096^3`, particle spacing `0.9448 Mpc/h`,
  `k_Nyquist=3.3253 h/Mpc`.
- Both runs use `L=3870 Mpc/h`, `RandomSeed=1386`, `MassMapNSIDE=2048`, the
  same CAMB tables, shell boundaries, cone aperture, observer, and cone axis.
- PINOCCHIO uses 4096 MPI ranks on 128 DCGP nodes. One radix-2 FFT slab and one
  `16 x 16 x 16` fragmentation subdomain are assigned per rank.

The common Fourier modes are phase matched. PINOCCHIO's IC generator assigns
random streams from integer Fourier coordinates, so changing `GridSize` while
retaining the seed preserves modes represented by both grids. Explicit
`PLCCenter` and `PLCAxis` parameters preserve the original lightcone geometry.

The 128-node request requires `dcgp_qos_bprod`. The original `2160^3` run used
15 nodes and 720 ranks, with a maximum rank RSS of about 10.3 GB. Cell-count
scaling predicts about 50 TB for `4096^3`; the selected layout provides about
63 TB before the explicit per-node memory reservation and leaves operational
headroom.

## Execution

Run from the GEPPETTO repository on Leonardo:

```bash
bash scripts/leonardo/l3870_n4096_resolution/stage_and_submit.sh
```

The script stages a new directory without modifying the `2160^3` run and
submits this dependency chain:

1. PINOCCHIO `4096^3` lightcone and mass maps;
2. GEPPETTO NFW painting with one segment worker per rank;
3. mask-coupled angular-power validation;
4. low/high-resolution spectra and ratio figures.

Submitted IDs are written to `pipeline_jobs.env` in the high-resolution run
directory. The stage script refuses to submit a second chain when that file
already exists.

## Outputs

The final comparison is written below the high-resolution run:

```text
geppetto_reduced/resolution_comparison/
  angular_power_resolution_comparison.csv
  angular_power_resolution_comparison.pdf
  angular_power_resolution_comparison.png
  angular_power_resolution_shells.pdf
  angular_power_resolution_shells.png
```

The comparison script verifies matching shell indices, shell boundaries,
HEALPix NSIDE, sky fraction, compact-mask hash, and multipoles before producing
ratios. A mismatch aborts the final job rather than silently comparing
different survey geometries.
