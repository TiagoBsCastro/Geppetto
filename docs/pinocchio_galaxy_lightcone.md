# PINOCCHIO galaxy lightcone milestone

Status: real-input milestone validated on 17 September 2026. The final run
contains 660,873 galaxies; all six preregistered LF bins, all populated colour
tests, and catalogue-integrity checks pass. This is an abundance/colour
validation, not a clustering or cluster-richness calibration.

## Scope and frozen validation design

This first model adapts [hodpy](https://github.com/amjsmith/hodpy), revision
`d303bef896fe6a92593d815f640f02865f11df60`, and the HOD/colour prescription of
[Smith et al. (2017)](https://arxiv.org/abs/1701.06581). It calibrates galaxy
abundances, not clustering or cluster richness. Existing matter painters and
their mass-definition conventions are not changed.

The selected real input is the Leonardo simulation
`/leonardo_scratch/large/userexternal/tbatalha/AB-MAH/Sims/L3870N4096/000/`.
It contains 32 binary `pinocchio.000.plc.out.N` parts, its `params.txt`,
cosmology table, geometry file, and native snapshot mass functions. The
smaller local `GridSize=128` example and the coarser `L3870N2160` full-sky
ensemble were inspected but are less suitable for luminosity completeness.
The higher-resolution `L1000N3072` and `L1200N2160` directories have no PLC
files at their simulation-directory level in the inspected data.

The initial validation bins, fixed before generating galaxies or inspecting
LF residuals, are cosmological-redshift slices `[0.05, 0.17)` and
`[0.22, 0.32)`, and absolute-magnitude edges `[-22.8, -22.5, -22.2, -21.9]`.
Magnitudes mean rest-frame `^0.1 M_r - 5 log10(h)`. These bins must first
pass a halo-resolution and expected-count assessment. Any revision will be
recorded with its completeness/count justification, before mock generation.
The initial conservative analysis cut is 32 particles; the input itself
contains groups down to 10 particles. Compare the occupation-weighted
contribution of the 10--32-particle population and a raised mass cut before
claiming completeness. An output threshold is not evidence of convergence
of the halo finder to arbitrarily small physical masses.

Acceptance: at least 200 mock galaxies in every selected magnitude bin;
deterministic cumulative LF within 2% of the independent target; measured
differential LF within `max(5%, 3 sigma)`. Statistical uncertainty must include
the specified HOD sampling variance and a reported spatial estimate. Colour
mixture fractions and standardized sequence means/widths must pass a
predeclared Monte Carlo test. All catalog integrity checks must pass.

The validation footprint is the inner 65-degree cone; the original 70-degree
cone supplies an angular satellite buffer. The staged redshift range is
`0.04 <= z_cos < 0.33`, also buffering both validation slices. An HOD-weighted
unresolved fraction below 1% is required, and its compact central support must
not extend below the input's 10-particle threshold at validation magnitudes.
Before fitting or drawing galaxies, the fixed observational LF and native
distance table predict bin counts `(621, 2644, 8519)` and
`(7652, 28785, 80403)` in the two slices. These are feasibility estimates,
not measured LF results.

The colour test uses a 1% familywise false-rejection level with Bonferroni
correction across its reported mixture-fraction, mean and variance tests.
Check conditional red probabilities separately for central/satellite samples;
check Gaussian sequence means/widths using standardized residuals, retaining
their magnitude/redshift dependence. LF uncertainties will report Poisson,
conditional HOD sampling, and a 32-equal-solid-angle-region spatial jackknife.
The acceptance uncertainty is the larger of conditional HOD and jackknife
estimates; this avoids treating correlated satellites as independent galaxies.

The underlying bright sample extends to `M_r=-21.5`, beyond the final validation
limit `-22.0`, before the `r<20` survey flag is applied. This is not a claim
of a complete flux-limited survey down to r=20 at all redshifts: the native
halo resolution does not support its faint low-redshift galaxies. All
luminosity-completeness claims are restricted to the preregistered bright bins.

### Completeness Revision Before Galaxy Generation

The first deterministic native-HMF fit preserved nested occupations without
changing the upstream HOD shape parameters. However, the **differential**
`[-22.2,-21.9)` bin at high redshift had a maximum 1.328% contribution from
input groups below the 32-particle analysis cut. Its cumulative counterpart
was below 1%, which is not enough to qualify the differential bin.
The final magnitude edges are therefore `[-22.8, -22.5, -22.2, -22.0]`.
The final faint bin had a maximum estimated missing fraction of 0.756%
in this preregistration check (0.753% after the numerical edge fix below).
This change was made before any stochastic galaxy catalogue or mock LF
residual existed. No other bin, seed, or threshold was changed to improve
agreement. Both cumulative and differential completeness gates are enforced.

### Distance And Coordinate Audit

The native table writes zero placeholders at `a>=1`; these are not distance
spline knots. Reproduce the natural cubic spline of its positive-distance
rows in `log10(a)`, with PINOCCHIO's endpoint secant extrapolation. Adding a
synthetic `a=1, chi=0` knot instead changes the interpolation near low z.
The native construction reproduces input halo radii to a maximum relative
error of about `1.15e-4` (the printed table and PLC use finite precision).
Float32 angular columns lose some accuracy at the cone axis; the maximum
unit-vector chord discrepancy is `5.46e-5`. Cartesian positions define the
output directions. The original angular columns remain in the staged input.

## Native simulation conventions

- Run flag `000`, seed 1386, `BoxSize=3870 Mpc/h`, `GridSize=4096`.
- `Omega_m=0.3913`, `Omega_Lambda=0.6087`, `Omega_b=0.0419`, `h=0.7276`,
  `n_s=0.9312`, `w0=-1.168`, `wa=-0.6605`. `Sigma8=0` instructs PINOCCHIO
  to use its input CAMB normalization; it does not mean zero fluctuations.
  Integrating the supplied z=0 power table gives `sigma8=0.6085032467`.
  Use the actual PINOCCHIO distance table rather than hodpy's MXXL cosmology.
- `M_PIN` is PINOCCHIO fragmentation-group particle count times simulation
  particle mass. It is an algorithmic group mass, not a spherical-overdensity
  measurement and not a measured `M200m` or `M200c`. With `OutputInH100`,
  stored mass units are `Msun/h`. Derive/check the particle mass against the
  native output rather than assuming the matter painter's density constant
  exactly matches PINOCCHIO's constant.
- The native density constant is `rho_crit,0=2.775499745e11` in
  `(Msun/h)/(Mpc/h)^3`. The particle mass is `9.160181752035501e10 Msun/h`.
  The native output cut is 10 particles (`9.16018e11 Msun/h`); the analysis
  and painting cut is 32 particles (`2.93126e12 Msun/h`). No mass conversion
  to a spherical-overdensity convention is performed.
- Stored PLC positions are already observer-relative, comoving `Mpc/h` in
  the simulation Cartesian basis, including any periodic image displacement.
  Do not subtract the observer a second time. Velocities are proper peculiar
  `km/s` in that same Cartesian basis, with no additional scale-factor factor.
- The observer is at `(415.2600210416666, 3786.244397541667,
  2252.2445095416665) Mpc/h`; the cone axis is
  `(-0.822556930095597, -0.112935946744643, -0.55735587256671)`.
  The cone half-angle is 70 degrees, area `2 pi (1-cos(70 degrees)) sr`.
  The original PLC covers `0 <= z <= 2`; the first galaxy validation uses
  only the above low-redshift slices, with a surrounding buffer for satellites.
- Native `theta` is latitude, not colatitude, and `phi` is longitude in the
  internal cone-aligned basis. Output sky coordinates will explicitly identify
  their basis; no interpretation as a particular celestial frame is implied.
- Native observed redshift follows `z_obs = z_cos + (1+z_cos) v_los/c`.
  The galaxy output will use the same first-order peculiar-velocity convention.
- Native group ID is preserved alongside a unique lightcone-occurrence ID.
  The latter is `(source_part << 48) + native_row`, with both components
  checked against their allocated bit widths. Galaxy IDs are unique row IDs.

These statements were checked against the local PINOCCHIO implementation:
`build_groups.c:store_PLC` (observer-relative positions), `set_obj_vel`
(`a H` velocity conversion), and `write_halos.c:write_PLC` (mass, units,
angles, LOS velocity and observed redshift). Input hashes and numerical
consistency checks will accompany the staged subset.

## Independent luminosity target and upstream audit

Use hodpy's supplied observational `sdss_cumulative_lf.dat` and the GAMA
Schechter fit in `lf_params.dat`, not `target_lf.dat` and not
`LuminosityFunctionTargetBGS`, which can construct a target from the HOD.
The upstream GAMA values are `Phi_star=0.0094 h^3/Mpc^3`, `M_star=-20.70`,
`alpha=-1.23`, `P=1.8`, `Q=0.7`; its equations use
`M_star(z)=M_star-Q(z-0.1)` and `Phi_star(z)=Phi_star*10^(0.4 P z)`.
The SDSS table is evolved with a magnitude shift `Q(z-0.1)` and density
factor `10^(0.4 P (z-0.1))`. Retain the upstream smooth SDSS/GAMA transition
`1/(1+exp(120(z-0.15)))`, explicitly recording the different reference epochs.
No LF or clustering parameter is fitted to measured mock galaxies.

Do not load MXXL `mf_fits.dat`, `slide_factors.dat`, or its precomputed
central/satellite magnitude arrays for the PINOCCHIO path. Recalibrate native
mass scales with a native-mass HMF. New caches must be keyed by all input
data, target, HOD parameters, selection and numerical settings. Validate
threshold nesting explicitly instead of assuming that independent shifts
preserve valid cumulative luminosity distributions.

In particular, the old `MassFunctionMXXL` correction
`sigma *= (delta_c(z)/delta_c(0))**2` compensates an upstream historical
growth error and is **not** part of this native-HMF path. Neither the MXXL
cosmology/power spectrum nor its fitted HMF evolution is imported. Equal-LF
number-density mapping of magnitude to the z=0.1 HOD shape is retained; this
is based on the fixed observational LF, not on an MXXL redshift shift.

Rest-frame colours are `^0.1(g-r)` from hodpy, conditional on luminosity,
redshift and central/satellite status. They are not observed multiband
photometry. Survey selection is applied after generating an underlying
absolute-magnitude-limited catalogue.

## Provisional satellite model

Use a configurable effective overdensity radius computed from `M_PIN` and
the mean matter density, with an explicit physical-to-comoving conversion.
This is `R_sat`, not a measured `R200m`. Adopt a configurable concentration
power law, truncated NFW radial sampling, isotropic directions, and an
isotropic Gaussian satellite velocity with one-dimensional dispersion
`sqrt(G M_PIN / (2 R_sat,physical))` times a configurable velocity factor.
Central galaxies inherit the halo centre and bulk velocity. These are
provisional prescriptions, not fits to clustering or richness.

The current numerical choices are

```text
R_sat,com = [3 M_PIN / (4 pi 200 rho_mean,com)]^(1/3)
c_sat = 5 (M_PIN / 1e14 Msun/h)^(-0.1) (1+z)^(-0.5)
sigma_sat,1D = sqrt(G M_PIN / (2 R_sat,physical))
G = 4.300917270e-9 Mpc (km/s)^2 / Msun
```

All factors and slopes are configurable. Satellites sample the normalized
3D NFW cumulative law `[ln(1+cx)-cx/(1+cx)]/[ln(1+c)-c/(1+c)]`,
where `x=r/R_sat`, with isotropic directions. Three independent Gaussian
proper peculiar velocity components are added to the host velocity. Native
central redshifts are kept exactly; satellite displacements change cosmological
redshift using the difference of native inverse-distance evaluations.

## Implementation And Calibration

The implementation is confined to `geppetto.galaxies`, separate from the
matter-painting core. `HODShapeParams`, `ThresholdParams`, `SatelliteParams`,
and `CalibrationConfig` make the model and numerical choices explicit.
Mean occupations, effective satellite scales, and the NFW inverse CDF are
JAX kernels with gradient/shape tests. Catalogue I/O, calibration orchestration,
HEALPix-independent geometry, and stochastic sampling are host-side. The
discrete random catalogue is not claimed to be end-to-end differentiable.

For each luminosity threshold and 0.01-wide redshift cell, all three initial
hodpy mass scales receive the same fitted shift in `log10(M_PIN)`. The
central compact-spline probability, central scatter, satellite slope, and
luminosity dependence follow the pinned upstream prescription. The satellite
mean is `N_cen [(M_PIN - M0)/M1]^alpha` above its cutoff. These functional
shapes remain provisional MXXL-derived priors, **not** a native-mass conversion
or a clustering calibration.

The HMF is measured directly from the native PLC in the inner 65-degree cone
and each redshift cell. It is a count-per-volume quadrature in 0.01-dex mass
bins, with each bin placed at its mean log mass. Integrating this HMF times
the occupation matches the independently specified, volume-averaged target
LF. Bisection shifts are checked for bracketing and occupation nesting over
the whole native mass range. Calibration fails on a non-nested model instead
of silently repairing its cumulative luminosity distribution. Validation
then sums occupations over individual halos, independently of the binned
HMF quadrature.

The target LF is adopted as a fixed empirical prescription in its published
hodpy h convention. It is not refitted to this simulation's cosmology or mock.
Because the native PLC HMF is measured from this realization, the abundance
calibration is conditional on that realization; the residuals are not an
independent test of the cosmological HMF or galaxy cosmic variance.

The table covers `-25 <= M_r <= -21.5` in 0.01-mag steps. A compact central
CDF gives at most one central per lightcone occurrence; satellites are
independent Poisson draws, as in hodpy. Their luminosities are sampled by
inverting the nested cumulative threshold counts. A central may be fainter
than the underlying magnitude cut even when a satellite is retained. The
bright endpoint's omitted expected count must be below 0.1 galaxy.
For this run it is `4.96e-8` galaxy, and the smallest audited central and
satellite threshold differences are both exactly zero, with no negative
differences. No monotonicity correction was applied.

The catalogue uses the original hodpy red/blue Gaussian means, widths,
evolution, satellite probability and empirical central-fraction prescription.
The latter is retained even if the native HOD central fraction differs.
Validation tests these conditional probabilities, not an independently
measured observed colour distribution. Upstream's colour-dependent GAMA
r-band k-correction gives
`r = M_r,h=1 + 5 log10[(1+z_cos) chi/(Mpc/h)] + 25 + K_r`.
Cosmological redshift is used for this magnitude conversion; Doppler flux
corrections, dust, lensing, and observed multi-band photometry are not modeled.

Caches are content-addressed by native input, distance table, fixed LF and
upstream code/table hashes, HOD parameters, selection, numerical settings,
and calibration source hashes. Changing any of these rebuilds the table.
Galaxy products additionally hash the complete configuration, pipeline source,
and NumPy/JAX versions. Fixed-seed reproducibility includes halo ordering and
chunk size (2048 in this run). Old products are never silently overwritten.
`catalogue.json` identifies the current product.

During review, exactly-32-particle float32 masses were found just below the
ideal cut by `4.38e-8` fractionally. Histogram edges now enclose every selected
native mass and fail if any selected object would be silently dropped. The
final table and catalogue were regenerated after this fix, without changing
the validation bins, seed, HOD functional parameters, or acceptance limits.

## Real-Input Results

The staged input contains **5,435,239 native PLC records** at
`0.04 <= z_cos < 0.33`. Input part paths and SHA256 hashes, parameter values,
units, and observer conventions are recorded in the staged HDF5 metadata and
its JSON sidecar. The final catalogue contains **588,385 centrals and 72,488
satellites**, before applying its separate apparent-magnitude selection flag.

Validation uses cosmological redshift and the inner 65-degree footprint.
Flat-space comoving volumes are `138,069,542.708` and `546,296,291.851 (Mpc/h)^3`
in the low- and high-redshift slices. These are computed from the actual
PINOCCHIO distance table, not a default hodpy cosmology. This first adapter's
cone-volume formula is restricted to flat cosmologies such as this input.

Differential LFs below are in `h^3 Mpc^-3 mag^-1`. Residuals are
`100 (prediction/target - 1)`. All bins exceed 200 galaxies and all galaxies
in these validation bins pass `r<20`.

| z slice | Magnitude bin | N | Target LF | HOD LF | Mock LF | HOD residual | Mock residual | Allowed mock residual |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0.05--0.17 | [-22.8,-22.5) | 597 | 1.498616e-5 | 1.498706e-5 | 1.441303e-5 | +0.0060% | -3.8244% | 13.2233% |
| 0.05--0.17 | [-22.5,-22.2) | 2,652 | 6.383028e-5 | 6.383297e-5 | 6.402571e-5 | +0.0042% | +0.3062% | 10.2878% |
| 0.05--0.17 | [-22.2,-22.0) | 4,692 | 1.680459e-4 | 1.680509e-4 | 1.699144e-4 | +0.0030% | +1.1119% | 6.7003% |
| 0.22--0.32 | [-22.8,-22.5) | 7,659 | 4.668789e-5 | 4.669012e-5 | 4.673288e-5 | +0.0048% | +0.0964% | 5.0000% |
| 0.22--0.32 | [-22.5,-22.2) | 28,994 | 1.756382e-4 | 1.756433e-4 | 1.769125e-4 | +0.0029% | +0.7255% | 5.0000% |
| 0.22--0.32 | [-22.2,-22.0) | 45,228 | 4.143752e-4 | 4.143818e-4 | 4.139512e-4 | +0.0016% | -0.1023% | 5.0000% |

The maximum cumulative HOD residual over all eight validation thresholds is
**0.00876%**, below the required 2%. The maximum differential mock residual
is **3.83%**; every bin passes even the 5% floor without relying on the larger
uncertainty allowance. The LF plot shows 1-sigma errors; its grey band is the
5% floor, not the full 3-sigma acceptance envelope.

The conditional HOD variance is the sum of `p_bin(1-p_bin) + lambda_bin`
over individual halos. The spatial estimate uses 32 equal-solid-angle
delete-one regions (8 azimuth by 4 equal-cosine radial bands). The reported
acceptance sigma is the larger of these; Poisson errors are also recorded.
The spatial jackknife is a diagnostic from one realization, not a claim of
an exact ensemble cosmic-variance covariance.

At the validated magnitudes the predicted contribution of native 10--31
particle halos is below 0.7533% in every cell/bin. The minimum compact HOD
support is 14.57 particles, above the native 10-particle floor. Holding the
HOD fixed and raising the painting cut to 64 particles removes, at worst,
`(0, 0.0975%, 1.8105%)` and `(0.0203%, 2.6602%, 8.8947%)` across the three
bins in the two slices. Thus the 64-particle cut would not preserve the
current faint-bin completeness. This is a model-weighted cut sensitivity,
not a claim that 32-particle group properties are numerically converged.

**Colours:** 49 populated mixture/sequence tests pass at a 1% familywise
level (`p_cut=0.01/49`). Expectations are recomputed from upstream functions,
not just read from the labels saved by the sampler. Red counts use their
Bernoulli variance, with fixed-seed Monte Carlo tails for small variance;
standardized Gaussian means and variances use Normal/chi-square distributions.
Blue fractions are the complements of the red fractions. Sequence widths
are tested through their standardized variance. The low-z brightest bin has
no satellites; this one status-specific colour bin is explicitly reported
as empty, not assigned a passing statistical test. Its 597 centrals provide
the complete LF-bin test. Some bright satellite groups have exactly zero
blue probability in the upstream model.

**Placement and integrity:** finite values, valid IDs and native masses,
one central per occurrence, exact central position/velocity/redshift,
satellite radial support, geometry buffers, and observed-redshift conventions
all pass. On all 72,488 satellites, independent probability-integral-transform
KS tests give `p=0.532` for NFW radii, `p=0.302` for polar isotropy, and
`p=(0.290,0.210,0.803)` for standardized velocity components (familywise
cutoff 0.002). Maximum satellite radius is `0.9999754 R_sat`.
Two separate same-seed generations agree exactly in all 25 galaxy columns
for all 660,873 galaxies; their provenance keys differ because of a subsequent
empty-bin robustness fix in the calibration code, with no numerical change
for this real input.

**Native identity caveat:** 373 persistent group IDs have repeated PLC
entries (374 extra records, 0.0069% of the input). These are nearby but
non-identical mass/redshift records: separations range from 0.0024 to
1.6534 Mpc/h. They cannot be distant periodic replicas within this shallow
volume. Native PLC crossing reconstruction checks changing groups repeatedly;
the records are preserved rather than choosing a mass or epoch arbitrarily.
The model's halo identity is therefore the **unique native PLC occurrence**,
with the persistent group ID retained separately. There are 59 extra central
occurrences across these persistent IDs in the complete bright catalogue,
but never two centrals within an occurrence. These native multiple crossings
are a residual input issue, particularly for very-small-scale clustering.

**Clustering:** the output includes an angular Landy-Szalay diagnostic over
0.05--10 degrees, using at most 30,000 data galaxies and twice as many uniform
randoms per slice. It is not used to fit the HOD and has no acceptance
threshold. No observed-clustering or cluster-richness agreement is claimed.

## Products And Rerun Commands

Run from the repository root. The tested interpreter is
`/home/tcastro/miniforge3/envs/geppetto-dev/bin/python` (Python 3.14,
JAX 0.10.2, NumPy/SciPy/h5py versions are also available in product provenance).

```bash
python -m pip install -e '.[galaxies,dev]'
bash scripts/fetch_hodpy.sh
python examples/run_pinocchio_galaxies.py --config examples/galaxy_lightcone_config.json --calibrate-only
python examples/run_pinocchio_galaxies.py --config examples/galaxy_lightcone_config.json
python examples/smoke_hodpy_mxxl.py
python examples/audit_pinocchio_galaxy_output.py \
  --catalogue outputs/galaxy_lightcone/validation/galaxies_dc3fc74d63c89809.hdf5 \
  --replicate outputs/galaxy_lightcone/validation/galaxies_d2dc017ff8780071.hdf5
python -m pytest
ruff check .
```

The run, smoke, audit, and test commands above were executed with the full
interpreter path given above. Installation is a setup instruction; this
workspace already contained the required dependencies. A fresh checkout
needs the staged input below. The pinned optional upstream checkout stays
unmodified; its BSD notice is retained in `docs/licenses/hodpy.txt` and
included in built wheels.

Initial implementation checks: **414 tests passed, 7 optional tests skipped**;
`ruff check .` and `git diff --check` passed. The 14 focused galaxy tests
cover unit conversion, native distance interpolation, velocity conventions,
HMF and LF integration, float32 mass-cut edges, luminosity-threshold nesting,
cache invalidation, gradient/shape checks, NFW sampling, and reproducible
written catalogues. A wheel also builds successfully with the optional
galaxy package and its BSD licence notice.

The original MXXL smoke reads all 735,697 supplied example halos, tests a
64-halo subset through the original reader, physical/comoving radius methods,
galaxy placement, colours, magnitudes and HDF5 writer, and verifies 128 output
objects. A fixed test occupation avoids rebuilding original MXXL lookup
tables. Legacy `nbodykit` is unavailable here, so this isolated smoke uses an
explicit Astropy MXXL-cosmology bridge. This is **not** a full scientific run
of upstream MXXL HOD calibration. The upstream very-small-radius interpolator
emits a `log10` warning; all exercised output values are finite. The new
PINOCCHIO sampler does not use that interpolator.

The real input was extracted on Leonardo with
`scripts/leonardo/prepare_galaxy_lightcone.sh`, submitted from an isolated
staging directory using this exact command:

```bash
ssh leonardo 'cd /leonardo_scratch/large/userexternal/tbatalha/Geppetto/outputs/galaxy_lightcone/code && sbatch --parsable prepare_galaxy_lightcone.sh'
```

Job **58069225** completed in 2m05s (`MaxRSS=13,592,016K`, exit 0).
The extractor command in that script was:

```bash
python examples/prepare_pinocchio_galaxy_input.py \
  --params /leonardo_scratch/large/userexternal/tbatalha/AB-MAH/Sims/L3870N4096/000/params.txt \
  --cosmology-table /leonardo_scratch/large/userexternal/tbatalha/AB-MAH/Sims/L3870N4096/000/pinocchio.000.cosmology.out \
  --plc-catalog /leonardo_scratch/large/userexternal/tbatalha/AB-MAH/Sims/L3870N4096/000/pinocchio.000.plc.out \
  --z-min 0.04 --z-max 0.33 --output halos_z004_z033.hdf5
```

For a fresh local workspace, retrieve that staged HDF5 and its JSON sidecar
from the remote `.../Geppetto/outputs/galaxy_lightcone/code/` directory into
`outputs/galaxy_lightcone/input/`, together with the simulation's `params.txt`,
`pinocchio.000.cosmology.out`, and `pinocchio.000.geometry.out`.
The extractor refuses to overwrite an existing staged input. Native parts
are read individually; source part hashes permit checking that this subset
really came from the stated simulation. No synthetic catalogue substitutes
for this input in the scientific validation.

Current product links (large files remain ignored by git):

- [Galaxy catalogue](../outputs/galaxy_lightcone/validation/galaxies_dc3fc74d63c89809.hdf5),
  SHA256 `c6e6ee70a646e2b4cba34eb1593fd1e33a265d93e6072d3d03540c11f1c53c51`.
- [Configuration](../examples/galaxy_lightcone_config.json),
  [staged input metadata](../outputs/galaxy_lightcone/input/halos_z004_z033.json),
  [input audit](../outputs/galaxy_lightcone/validation/input_audit.json).
- [Validation summary](../outputs/galaxy_lightcone/validation/validation_summary.json),
  [differential LF values](../outputs/galaxy_lightcone/validation/lf_validation.csv),
  [cumulative LF values](../outputs/galaxy_lightcone/validation/cumulative_lf_validation.csv),
  [colour tests](../outputs/galaxy_lightcone/validation/colour_validation.csv).
- [LF plot](../outputs/galaxy_lightcone/validation/luminosity_function.png),
  [g-r versus magnitude](../outputs/galaxy_lightcone/validation/colour_magnitude.png),
  [colour residuals](../outputs/galaxy_lightcone/validation/colour_validation.png),
  [angular clustering](../outputs/galaxy_lightcone/validation/angular_clustering.png).
- [Native HMF](../outputs/galaxy_lightcone/validation/native_hmf.csv),
  [completeness](../outputs/galaxy_lightcone/validation/completeness.json),
  [raised-mass-cut sensitivity](../outputs/galaxy_lightcone/validation/mass_cut_sensitivity.csv),
  [native ID diagnostics](../outputs/galaxy_lightcone/validation/native_input_diagnostics.json).
- [Original MXXL smoke](../outputs/galaxy_lightcone/validation/mxxl_smoke.json),
  [independent radial/velocity and reproducibility audit](../outputs/galaxy_lightcone/validation/sampling_audit.json).

The HDF5 `galaxies` group contains absolute/apparent r magnitude,
`g_r_rest_0p1`, Cartesian position and velocity, cone-frame longitude/latitude,
cosmological and observed redshift, unique galaxy and host occurrence IDs,
persistent native host group ID, native host mass, central flag, survey flag,
effective satellite scales, and colour-draw diagnostics. Root `metadata_json`
contains the native cosmology, input provenance, seed, configuration, model
definitions, reference redshifts, and units. Rest-frame `g-r` is not a
complete set of observed photometric bands.

## Frozen-model diagnostic report

The expanded [diagnostic report](../examples/pinocchio_galaxy_diagnostics/report.md)
and [PDF](../examples/pinocchio_galaxy_diagnostics/diagnostic_report.pdf) audit
the existing catalogue without refitting or regenerating it. The eight groups
cover native HMF/completeness, LF, redshift/apparent-magnitude selection,
central/satellite occupations, colours, satellite radii, velocities/redshifts,
and sky geometry/identities. Angular clustering is an additional **unfitted**
diagnostic, with no observational-agreement threshold.

Three nonoverlapping redshift slices (0.05-0.14, 0.14-0.23, 0.23-0.32) and
three native host-mass ranges are used. Only the SDSS/GAMA LF is an
independently specified observational target; colour, occupation and
satellite curves test the frozen model's own sampling prescriptions.
The nine preselected complete LF bins pass, with a maximum mock residual
of 3.03% and maximum cumulative HOD residual of 0.00838%. All 80 colour
tests and 12 stratified radial/velocity tests pass their familywise checks.
The occupation chi-square shows mild excess scatter (p=0.0114 at a 1%
threshold). Repeated native persistent IDs and small satellite samples are
documented caveats, not silently removed.

From the repository root, in the environment described above:

```bash
python examples/diagnose_pinocchio_galaxies.py \
  --config examples/pinocchio_galaxy_diagnostics_config.json
python examples/diagnose_pinocchio_galaxies.py \
  --config examples/pinocchio_galaxy_diagnostics_config.json --plot-only
/usr/bin/python3 examples/render_pinocchio_galaxy_report.py \
  --input-dir examples/pinocchio_galaxy_diagnostics
```

The PDF renderer needs ReportLab, available in the system Python on the
test machine; any Python environment with that package can render it.
The plot-only command uses the saved CSV/JSON products, without JAX, hodpy,
or the large catalogue. Figure source snapshots, all binned values, frozen
configuration and calibration, package versions, and SHA256 provenance
are included beside the figures. Input and model-source hashes are checked
before and after the full audit; a mismatch fails explicitly.

Expanded-audit verification: **419 tests passed, 7 optional tests skipped**;
`ruff check .` and `git diff --check` passed. Replotting reproduced all
10 PNG figures and 16 CSV products byte-for-byte. The PDF was rendered and
visually checked. See the accompanying
[verification record](../examples/pinocchio_galaxy_diagnostics/VERIFICATION.md).
