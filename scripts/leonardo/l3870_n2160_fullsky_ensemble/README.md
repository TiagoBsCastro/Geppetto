# L3870N2160 full-sky cosmic-variance ensemble

This campaign runs eight independent PINOCCHIO seeds with the production
cosmology, `GridSize=2160`, and `MassMapNSIDE=2048`. The PLC is full sky and
ends at the existing shell boundary `z=0.492562563187`.

The observer is fixed at the box centre. Since the radial support is below
half the `3870 Mpc/h` box side, the validation sphere contains no periodic
replication of structures. Every realization paints the same 13 shells with
GEPPETTO and measures full-sky spectra using both the theoretical homogeneous
mean and the realization's measured shell mean.

PINOCCHIO's current PLC `n(z)` diagnostic clamps apertures above 90 degrees to
a hemisphere when reporting number per square degree and analytic counts.
The halo selection and mass-map code correctly treat 180 degrees as full sky;
do not use the `n(z)` area normalization from these runs.

Submit from the Leonardo GEPPETTO checkout:

```bash
bash scripts/leonardo/l3870_n2160_fullsky_ensemble/stage_and_submit.sh
```

Final products are written to:

```text
/leonardo_scratch/large/userexternal/tbatalha/AB-MAH/Sims/L3870N2160/fullsky_z0493_ensemble/ensemble_results
```

The optional component decomposition is measured one realization per array
task:

```bash
sbatch --array=0-7 scripts/leonardo/l3870_n2160_fullsky_ensemble/submit_components_array.sh
```

Each task writes `geppetto_reduced/fullsky_component_spectra.npz` below its
seed directory. These compact products can be passed to
`examples/audit_fullsky_component_decomposition.py` together with a schema-5
theory directory to compare the measured `UU`, `UH`, and `HH` decomposition
with the corrected two-halo response.

## Fixed Pair-Model Candidate

The pair-model comparison reuses the eight existing observed and component
spectrum caches. It does not repaint the simulations or refit parameters.
The forecast covers all 13 shells, with comparison bands spanning
`20 <= ell < 2000`, `Delta ell = 20`, and weights `2 ell + 1`. No mask
response or `f_sky` rescaling is used. The primary statistic uses the
theoretical homogeneous shell mean; shell-mean normalization is retained
as a separate diagnostic. Shading denotes the sample standard deviation
divided by `sqrt(8)`, not a covariance-based goodness-of-fit interval.

The candidate's original `000`-run HMF, concentration relation, corrected
scale-dependent CAMB evolution, radial quadrature, and orientation samples
are held fixed. The original seed, 1386, is one of the eight realizations;
the summary also reports RMS residuals after excluding that seed. This is
not an ensemble-HMF fit or a completely independent eight-seed validation.

Full-sky segment IDs `0..12` match original IDs `16..28` by **both redshift
bounds**. The staged `linear_reference.npz` contains the original schema-5
archive's full-sky `shell_linear` rows for those bounds, not its masked
spectra. Its keys are `ell`, `shell_linear`, `segment_indices`, `z_lo`,
`z_hi`, `reconstructed_sigma8`, `linear_power_evolution`, `parent_segments`,
and `source_sha256`. The schema-6 linear projection is not substituted:
its numerical projection settings differ slightly. Orientation seed 737
preserves the original per-segment draws (`721 + original_segment`).

The current campaign uses an isolated source snapshot and optional
dependencies, without changing the shared Leonardo environment:

```text
/leonardo_scratch/large/userexternal/tbatalha/Geppetto/outputs/fullsky_pair_model/
    code/                 # example scripts and src/geppetto snapshot
    dependencies/         # velocileptors 3.1, pyfftw 0.15.1, ducc0 0.41.0
    linear_reference.npz
    predictions/          # one small spectrum archive per shell
    comparison/           # figures, bandpowers, component and residual tables
    logs/
```

After staging these inputs, submit **from that campaign directory** so the
relative Slurm log paths resolve correctly:

```bash
array_job=$(sbatch --parsable submit_pair_model.sh)
sbatch --dependency="afterok:${array_job}" submit_pair_model_comparison.sh
```

The array starts nearby shells first, permits four concurrent tasks, and
requests 16 CPUs and 128 GB per task. A failed shell prevents aggregation;
partial forecasts are not silently averaged. The comparison writes the
13-panel `fullsky_pair_model_all_shells.{png,pdf}`, an NPZ of binned spectra,
and CSVs of bandpowers, shell means, RMS residuals, and `UU`, `2UH`, `HH`.
The exact-linear replacement is reported separately from the stationary
components. Input hashes accompany the results.

For the current submission, the prediction array is **57018928** and the
dependent comparison is **57019582**. These IDs record submission, not
completion. Compact outputs can be copied locally after success:

```bash
rsync -av leonardo:/leonardo_scratch/large/userexternal/tbatalha/Geppetto/outputs/fullsky_pair_model/comparison/ \
    examples/angular_power_fullsky_pair_model/
```

### Large Nearby Halo Footprints

`--moment-backend auto` uses the existing pixel-pair path when within its
pair budget. Larger footprints use the spherical-harmonic addition theorem
on the **same normalized native-pixel weights**. This avoids allocating
every pixel pair without changing the painting prescription. The optional
`.[harmonics]` dependency supplies DUCC's
[adjoint point-synthesis transform](https://mtr.pages.mpcdf.de/ducc/sht.html).
JAX still evaluates and differentiates the profile and global normalization;
the host transform applies the linear/quadratic chain rule to those weights.
Direct-pair comparisons test the moments and all three concentration
derivatives. This backend is not a new physical approximation to the halo
profile; numerical transform tolerance and finite orientation sampling
remain distinct accuracy controls.
