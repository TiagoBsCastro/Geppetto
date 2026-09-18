# Schema-6 Results Audit

## Verdict

The normalized HMF strategy removes the previous large two-halo response boost
and passes its stored mass, bias and mass-quadrature checks. It does **not** yet
establish scientific agreement across scales. The eight-seed full-sky ensemble
has coherent excesses and deficits well above its measured standard errors.
Neither a new overall growth rescaling nor a cut-sky mask explanation is
supported by these results.

This audit checks completed products, not a new production run. It does not
modify spectra, fitted parameters, model code, or the measurement convention.

## Inputs And Method

- [Cut-sky products](../examples/angular_power_validation_schema6/README.md):
  job `56884283`, 29 shells, `0 <= z <= 2`, `f_sky=0.3290201823`.
- [Full-sky products](../examples/angular_power_fullsky_ensemble_schema6/README.md):
  job `56884299`, eight seeds, 13 shells, `0 <= z <= 0.49256256`.
- Both use `NGRID=2160`, `NSIDE=2048`, and `2 <= ell <= 4096`.
- Compared the cut-sky results with the local schema-5 archive, and reused the
  measured spectra in the [component audit](../examples/angular_power_fullsky_component_audit/README.md).

The primary total is `C_2h + C_1h_compensated + C_particle_shot`; the ordinary
one-halo alternative replaces only `C_1h_compensated`. Linear theory is not
added a second time. Thus the primary blue curve is not the entirely ordinary,
uncompensated textbook halo model. See the
[model equations and conventions](normalized_halo_model.md).

For a band B, all reported ratios are ratios of bandpowers,

```text
R_B = sum_B [(2 ell + 1) C_measured] / sum_B [(2 ell + 1) C_theory].
```

They are not unweighted averages of the plotted ratios. Full-sky uncertainties
are the sample standard deviation of the eight realization bandpowers divided
by `sqrt(8)` and by the fixed theory bandpower. Integrating each realization
first preserves correlations within a band. These are descriptive standard
errors, not a joint goodness-of-fit statistic: HMF-fit uncertainty, theoretical
error, cross-shell covariance and finite-ensemble covariance uncertainty are
not included. Cut-sky comparisons use the mask-coupled, constant-deprojected
theory arrays, not the underlying full-sky spectra.

## Remaining Discrepancies

The table gives the full-sky ensemble mean divided by the primary total. Only
the first band includes the standard error here; narrow-band examples follow.

| Redshift shell | ell 20-199 | ell 200-499 | ell 500-1499 | ell 1500-4096 |
| --- | ---: | ---: | ---: | ---: |
| 0.000-0.034 | 0.8058 +/- 0.0406 | 0.8113 | 0.8441 | 0.9706 |
| 0.034-0.068 | 0.9437 +/- 0.0176 | 0.8808 | 0.8907 | 0.9332 |
| 0.068-0.103 | 1.0261 +/- 0.0065 | 0.8783 | 0.9222 | 0.9238 |
| 0.103-0.138 | 1.0307 +/- 0.0052 | 0.8800 | 0.8958 | 0.9132 |
| 0.138-0.174 | 1.0172 +/- 0.0042 | 0.9443 | 0.8761 | 0.9112 |
| 0.174-0.211 | 1.0130 +/- 0.0042 | 1.0344 | 0.8771 | 0.9148 |
| 0.211-0.248 | 1.0094 +/- 0.0033 | 1.0878 | 0.8829 | 0.9131 |
| 0.248-0.287 | 0.9964 +/- 0.0049 | 1.1041 | 0.8878 | 0.9026 |
| 0.287-0.326 | 0.9996 +/- 0.0041 | 1.1175 | 0.9202 | 0.9012 |
| 0.326-0.366 | 0.9927 +/- 0.0033 | 1.0989 | 0.9469 | 0.8866 |
| 0.366-0.407 | 0.9820 +/- 0.0022 | 1.0836 | 0.9888 | 0.8795 |
| 0.407-0.449 | 1.0014 +/- 0.0024 | 1.0841 | 1.0420 | 0.8841 |
| 0.449-0.493 | 0.9840 +/- 0.0027 | 1.0554 | 1.0683 | 0.8740 |

For `z >= 0.068`, the first band is within approximately 3.1% of unity, but
some deviations are several measured standard errors. Broad-band agreement
also hides the alternating excess and deficit visible in the
[all-shell figure](../examples/angular_power_fullsky_ensemble_schema6/figures/angular_power_all_shells_schema6.png).

### Transition Between Terms

These are 20-multipole bins; component fractions refer to the primary theory.

| Shell | ell | Measured/theory | Two-halo fraction | One-halo fraction | Particle-shot fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.174-0.211 | 220-239 | 1.1289 +/- 0.0094 | 89.8% | 8.3% | 2.0% |
| 0.174-0.211 | 740-759 | 0.8471 +/- 0.0076 | 23.1% | 70.2% | 6.6% |
| 0.449-0.493 | 640-659 | 1.1717 +/- 0.0031 | 87.1% | 8.6% | 4.3% |
| 0.449-0.493 | 2200-2219 | 0.8470 +/- 0.0025 | 18.3% | 66.2% | 15.6% |

The excess appears while the two-halo term dominates; the trough appears when
the one-halo term dominates. An error in a single overall amplitude cannot
fix both. The ordinary one-halo alternative fills much of the excess, but
retains the trough and worsens some low-multipole comparisons. The data do
not identify a unique missing ingredient such as nonlinear bias, exclusion,
the compensation prescription, halo profiles, or particle discreteness.

Using `k ~ (ell+0.5)/chi(z_mid)` only as a scale diagnostic, the trough moves
from `ell ~ 330` at `z_mid ~ 0.085` to `ell ~ 2210` at `z_mid ~ 0.471`.
For shells above the nearest one, this corresponds to approximately
`k=1.1-1.8 h/Mpc`. The simulation's initial-grid Nyquist wavenumber is
`pi/(3870/2160)=1.7534 h/Mpc`. This coincidence motivates a resolution test;
it does not prove grid suppression. These angular spectra integrate a range
of radial wavenumbers, especially in the nearest shell. A fixed angular
`NSIDE=2048` cutoff alone does not explain the moving feature.

The existing [NGRID=4096 comparison](../examples/angular_power_resolution_quick/angular_power_low_theory_resolution_check.csv)
provides an additional constraint. In the original cut-sky `0.174-0.211`
shell, the high/low measured bandpower is 1.0014 for ell 20-199 and 1.0031
for ell 500-1499. Its low-resolution entries exactly reproduce the current
cut-sky measurements after binning. However, these are total spectra from
one phase-matched pair: particle shot noise and the resolved halo population
change with NGRID. They do not constitute a shot-matched clustering or
full-sky ensemble convergence test.

### Nearest-Shell Density

For `0 < z < 0.03362`, all eight realizations have a negative shell mean
overdensity. Their ensemble mean is `-0.0625 +/- 0.0142`. In ell 20-199:

- Theoretical-density normalization: `measured/theory = 0.8058 +/- 0.0406`.
- Measured-shell-density normalization: `0.9118 +/- 0.0210`.

The density fluctuation explains part, but not all, of this discrepancy.
The second ratio is an estimator-sensitivity diagnostic against the same
theory, not an independently corrected prediction for a locally normalized
field. It is not justified to select the normalization simply because its
ratio is closer to one. Eight seeds and a small, non-Gaussian local volume
are insufficient to turn these standard errors into a precise tail probability.
The common density deficit also deserves a radial mass-budget check.

## What Improved

In the originally problematic cut-sky `0.174 < z < 0.211` shell, the
ell 20-199 ratio changes from **0.9201 (schema 5) to 0.9810 (schema 6)**.
The measurements are bitwise unchanged. Across all cut-sky shells, the
linear bandpower changes by at most 0.054% in this band, so this improvement
is not another growth-function correction.

The component audit independently confirms that the large two-halo boost is
gone. For the full-sky `0.068-0.103` shell, `C_2h/C_linear` changes from
1.2086 to 0.9973 in ell 20-199. For `0.174-0.211`, it changes from 1.0665
to 0.9993. The new calculation uses the standard normalized response, with
no additional linear baseline in [the kernel](../src/geppetto/theory.py).

The measured component-coherent spectrum remains approximately 1.7-4.0%
below the new two-halo spectrum for `z >= 0.068`. This is not a direct
measurement of the physical two-halo term: the existing diagnostic defines
`HH_parallel = UH^2/(UU - particle_shot)` by regression onto the uncollapsed
field. It cannot uniquely distinguish nonlinear coherence, exclusion or
bias errors. Reusing its binned total measurements reproduces the current
ensemble means to floating-point precision.

For the summed cut-sky map:

| ell | Schema-5 total ratio | Schema-6 primary ratio | Schema-6 ordinary-1h ratio |
| --- | ---: | ---: | ---: |
| 20-199 | 0.9794 | 0.9818 | 0.9763 |
| 200-499 | 0.9814 | 0.9920 | 0.9776 |
| 500-1499 | 0.9970 | 1.0396 | 0.9917 |
| 1500-4096 | 0.9314 | 0.9793 | 0.9096 |

Improvement is therefore scale dependent, not universal. The summed map
can hide opposite-sign shell residuals and is not the primary acceptance test.

## Numerical Checks And Limits

Both archives have finite numeric arrays and matching audit-file SHA-256
identifiers. The cut-sky shell and summed measurements are bitwise identical
to schema 5. Stored quadratures and closure diagnostics give:

| Check | Cut sky | Full-sky ensemble |
| --- | ---: | ---: |
| Maximum individual mass-closure error | 2.40e-10 | 9.91e-11 |
| Maximum individual bias-closure error | 2.22e-16 | 3.33e-16 |
| Final mass-refinement relative change | 0.04823% | 0.02788% |
| Maximum count-weighted log HMF-fit residual | 0.03049 | 0.03097 |
| Maximum native peak-height discrepancy | 0.1964% | 0.1169% |
| Maximum shell projection matching error | 0.99992% | 0.99989% |

These checks have specific, limited meanings:

- Bias closure is imposed by normalization, not an independent validation of
  the mass-dependent bias. The ensemble Castro-corrected PBS bias is rescaled
  by factors 0.877-0.906. It is an adapted model, not the unmodified
  [Castro et al. calibration](https://arxiv.org/abs/2409.01877).
- The **averaged ensemble quadrature has 46.0-47.1% of its mass and
  38.6-40.7% of its zero-wavenumber bias response in the analytic low tail**.
  The finite input P(k) leaves a nonzero minimum peak height as mass tends
  to zero. Completing F(nu) below that point with `u=1` is a formal modeling
  assumption, not a population established by the measured HMF. The tiny
  omitted one-halo bound does not bound this physical uncertainty.
- Both runs record `profile_convention=continuum_nfw`; `--match-ngp` was not
  enabled. This is a useful continuum halo-model benchmark, but not an exact
  prediction for the adaptive painter. Applying its NGP threshold would
  still not reproduce realization-specific supersampled pixel stencils.
- Mass refinement tested the squared response grid and projected one-halo
  bands. It did not independently refine profile, temporal and radial
  quadrature orders or rerun the complete exact two-halo projection. The
  approximately 1% projection matching tolerance also precludes treating
  every sub-percent residual as physically significant.
- Growth is recorded as `scale_dependent_camb`. Its z=0 P(k) agrees with the
  cosmology table to `9.70e-6` relative error. The reconstructed sigma8 is
  0.60850325, but its reference is the **same** cosmology power spectrum,
  because the input parameter requests that normalization. Consequently,
  the reported zero sigma8 error is not an independent closure test.

## Next Checks

1. Use a few representative shells to compare continuum and `--match-ngp`
   predictions, and refine profile/temporal/radial orders independently.
   Establish numerical and painting-operator accuracy before attributing the
   residual entirely to halo-model physics. Keep both one-halo variants.
2. Extend the existing U-U, U-H and H-H audit across the excess and trough.
   Reuse the high-resolution measurement cache, but match the shot-noise and
   density estimators separately for each resolution. This can distinguish
   a change in clustering from a change in particle noise without new simulations.
3. Check the nearest shells' radial particle/mass budget and seed dependence.
   Do not erase the common density deficit by renormalizing spectra ad hoc.
4. Quantify sensitivity to admissible low-mass HMF completions and bias shapes,
   preserving both integrals and the resolved-HMF fit. Only then consider
   changes to compensation, exclusion or nonlinear two-halo modeling, guided
   by the component results rather than by fitting the total residual alone.

The present results support the new normalization strategy, but do not justify
declaring either ordinary or compensated schema-6 theory a validated model
over the full multipole range.
