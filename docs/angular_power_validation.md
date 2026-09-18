# Angular-Power Validation

GEPPETTO predicts the angular power spectra of the count-overdensity maps made
by adding each painted NFW NPZ array to its original PINOCCHIO uncollapsed-
particle FITS segment. The implementation is intended to validate the map
painting and its concentration dependence, not to replace a precision survey
likelihood.

The current standard halo-model comparison is documented in
[Schema-6 Normalized Halo Model](normalized_halo_model.md). It fits normalized
HMFs, normalizes the Castro-corrected mass-weighted bias, and reports both
standard and compensated one-halo variants. `submit_theory.sh` now selects
that workflow. The schema-4/5 workflow below remains available for historical
comparisons through `validate_pinocchio_angular_power.py`.

## Schema-5 Three-Dimensional Model

The schema-5 model is

\[
P_{\rm mm}(k,z)=P_{2h}(k,z)+P_{1h}(k,z),
\]

with the pure linear spectrum retained as a diagnostic rather than added as a
third contribution. The deterministic term is

\[
P_{2h}(k,z)=P_{\rm lin}(k,z)\,\mathcal R_{2h}^2(k,z),
\]

\[
\mathcal R_{2h}(k,z)=1+\int d\ln M\,\frac{dn}{d\ln M}
\frac{M}{\bar\rho_m}b(M,z)
\left[u_{\rm NFW}(k|M,z)-W_{\rm TH}(kR_L)\right].
\]

This is the response appropriate to a PINOCCHIO large-scale map plus the
GEPPETTO mass rearrangement. The explicit one preserves the unresolved linear
matter backbone. The compensated halo correction vanishes as `k -> 0`, so
`R_2h -> 1` exactly without an HMF completion or bias-renormalization step.

with

\[
P_{1h}(k,z)=\int d\ln M\,\frac{dn}{d\ln M}
\left(\frac{M}{\bar\rho_m}\right)^2
\left|u(k|M,z)-W_{\rm TH}(kR_L)\right|^2,
\]

where

\[
R_L(M)=\left(\frac{3M}{4\pi\bar\rho_m}\right)^{1/3}
\]

is the comoving Lagrangian radius and `W_TH` is the spherical top-hat Fourier
window. The difference represents rearranging each halo's mass from its
Lagrangian patch into the NFW profile. Both transforms equal one at zero
wavenumber, so the one-halo contribution is `O(k^4)` rather than an
unphysical constant on scales where PINOCCHIO already supplies the
large-scale field.

When `FileWithInputSpectrum CAMBTable` is present in the PINOCCHIO parameter
file, GEPPETTO resolves `CAMBMatterFile` and `CAMBRedshiftsFile` relative to
that file and reads the complete tabulated `P(k,z)` evolution. This preserves
the scale-dependent growth of the spectrum actually used by PINOCCHIO. The
reader currently requires `InputSpectrum_UnitLength_in_cm=0`, for which CAMB
tables use `h/Mpc` and `(Mpc/h)^3`. It verifies the `z=0` table against the
power stored in `*.cosmology.out`. Runs with one input spectrum retain the
fallback

\[
P_{\rm lin}(k,z)=D^2(z)P_{\rm lin}(k,0).
\]

### Numerical-HMF Bias And Castro Correction

The peak-background-split baseline is obtained from the numerical PINOCCHIO
HMF, not from a CCToolkit HMF calibration. At every native HMF redshift, the
code constructs the multiplicity from the measured `dn/dM` and PINOCCHIO
peak-height column, then fits

\[
F(\nu)=A\nu^q\exp(-a\nu^2/2)
\left[1+(a\nu^2)^{-p}\right].
\]

With PINOCCHIO's fixed `delta_c = 1.686`, the fitted PBS bias is

\[
b_{\rm PBS}(\nu)=1-\frac{1}{\delta_c}
\frac{d\ln F}{d\ln\nu}.
\]

GEPPETTO imports only `bias_correction_PBS` from CCToolkit and applies the
Castro et al. multiplicative correction,

\[
b(M,z)=b_{\rm PBS}(M,z)\,C_{\rm Castro}
\left[\Omega_m(z),d\ln\sigma/d\ln R,S_8\right].
\]

For the pinned implementation,

\[
C_{\rm Castro}=A_0(1+a_1\Omega_m)
(1+b_1s+b_2s^2)(1+c_1S_8),
\qquad s=d\ln\sigma/d\ln R,
\]

with `(A0, a1, b1, b2, c1) = (1.150, 0.0929, 0.256, 0.173,
-0.0372)`. GEPPETTO passes `s=-3 dln(nu)/dln(M)` from the PINOCCHIO
peak-height relation and reconstructs `S8=sigma8 sqrt(Omega_m0/0.3)` from the
input linear spectrum.

CCToolkit is pinned to Git revision
`ac16ab613eb93f795f562f928ad145597d981b2f` for reproducibility. A fit with
too few populated bins, non-finite or non-positive bias, or a reduced weighted
residual above the configured threshold aborts validation and writes no final
theory product. `--pbs-fit-min-populated-bins` and
`--pbs-fit-max-weighted-log-residual` control the defaults of 8 and 0.05,
respectively. The latter is a count-weighted RMS logarithmic profile error.
The Poisson-weighted reduced residual is retained diagnostically, but it is
not an acceptance statistic because neighboring numerical-HMF bins are not
independent. The fit diagnostics are retained separately.

The Castro correction was calibrated for virial halo masses. The current
painting manifest and numerical HMF can use `M_200c`; applying the correction
in that native convention is an explicit approximation. No CCToolkit HMF or
mass conversion is introduced, so the PINOCCHIO HMF, map, and NFW transform
remain internally mass-matched. Halo exclusion is not modeled.

The measured second column of every requested PINOCCHIO `*.mf.out` file is used
for `dn/dM`. A file is required at every shell-boundary redshift. Measured
zeros and the finite measured mass range are preserved; GEPPETTO does not fill
rare bins from the analytic Watson column or renormalize the resolved mass
fraction.

The halo transform is the normalized Fourier transform of the hard-truncated
3D NFW profile using the concentration and spherical-overdensity conventions
from the painting manifest. Fixed Gauss-Legendre quadrature enforces `u(0)=1`
using its own normalization and remains differentiable with respect to all
concentration parameters. Halos below the recorded angular NGP threshold use
`u=1`. Supersampling and native painting approximate the same continuum
resolved profile and therefore do not define separate theory kernels.

### Large-Scale Closure Test

The scale-dependent-growth and compensated-one-halo choices follow a
controlled paired-fixed test, not a fit to the validation plot. Four
independent phase pairs were compared over the first seven full-sky shells
(`0.23 < z < 0.49`) using only modes with
`k=(ell+0.5)/chi < 0.05 h/Mpc`. Against scalar-growth linear theory, the
pair-mean map amplitude was `0.991997 +/- 0.001584`, a 5.05-sigma deficit.
Using the tabulated scale-dependent evolution changed it to
`0.998711 +/- 0.001593`, consistent with unity. Adding the former
uncompensated one-halo floor lowered the total-theory ratio to
`0.987577 +/- 0.001572`. Thus the controlled maps do not lose large-scale
power: the two statistically resolved effects were both in the theory model.
The corrected two-halo response described above is likewise not fit to these
map spectra; its free shape is fixed by the numerical HMF, PINOCCHIO peak
heights, and the pinned Castro correction.

## Angular Projection

For a shell bounded by `chi_lo` and `chi_hi`, the count-overdensity window is

\[
W_i(\chi)=\frac{3\chi^2}{\chi_{i,\rm hi}^3-\chi_{i,\rm lo}^3}.
\]

The linear diagnostic uses the exact spherical-Bessel expression at low
multipoles:

\[
C_\ell^{\rm lin}=\frac{2}{\pi}\int dk\,k^2P_0(k)
\left|\int d\chi\,W_i(\chi)T(k,\chi)j_\ell(k\chi)\right|^2,
\]

with

\[
T(k,\chi)=\sqrt{\frac{P_{\rm lin}(k,z(\chi))}{P_0(k)}}.
\]

For a single input spectrum this reduces exactly to the scalar growth
`T=D`. For a CAMB series, eight fixed temporal interpolation nodes per shell
represent the smooth scale-dependent transfer while the configured radial
quadrature resolves the spherical Bessel kernel.

The corrected deterministic spectrum uses the same projection after replacing

\[
T(k,\chi)\longrightarrow T(k,\chi)\mathcal R_{2h}(k,z(\chi)).
\]

The response is therefore inside each shell transfer before it is squared, and
the exact summed spectrum retains corrected cross-shell terms. The code does
not multiply a projected linear `C_ell` by a response at one effective
redshift.

The integral is restricted to the tabulated PINOCCHIO k range. Exact spectra
are evaluated in batches through the configured comparison cap. A transition
is accepted only when the high-multipole approximation stays within one
percent from that multipole through the complete exact range, with at least 20
multipoles available. This prevents an accidental short agreement interval
from hiding a later divergence. The default exact search cap is `ell=512`;
failure of any spectrum to converge before the cap aborts the validation and
reports its maximum relative error over the final comparison window.
Independent exact multipoles can be evaluated concurrently with spawned worker
processes selected by `--exact-workers`. Each child is restricted to one native
thread without inheriting the parent OpenMP affinity, allowing the operating
system to distribute workers across the task's allocated cores. The Leonardo
submission example uses all 112 physical cores and checkpoints each
112-multipole batch.

Exact and Limber radial quadratures have separate controls. Limber retains the
64-node `--radial-order` default. Exact projection defaults to
`--exact-radial-order 512`, because the observer-adjacent shell requires many
more nodes to resolve high-multipole Bessel oscillations. Each exact shell has
its own wavenumber cutoff: its tail spans at least 40 radial periods, grows
with the shell's transverse wavenumber, and is capped by
`--exact-radial-tail-periods` (default 256). Samples above a shell's cutoff do
not enter either its auto-spectrum or the weighted summed transfer. This
prevents a nearby shell from forcing unresolved high-frequency evaluations
into every distant shell.

Completed exact linear and two-halo batches are atomically accumulated in
`angular_power_exact_checkpoint.npz`. A rerun with identical projection inputs
restores those multipoles and computes only missing batches. The checkpoint is
removed after all final validation products have been written; it remains
available after a timeout, node failure, or convergence error.

Standard Limber does not reach one-percent accuracy by `ell=512` for the
narrow, hard-edged radial shells in this light cone. For each individual
shell, the validator therefore also computes the finite-width flat-sky
projection

\[
C_{\ell,i}^{\rm fw} =
\frac{1}{\pi\chi_{i,\rm mid}^2}
\int_0^\infty dk_\parallel\,
\left|\widetilde{W_i\sqrt{P}\mathcal R_{2h}}
(k_\parallel,k_\perp)\right|^2,
\]

where

\[
\widetilde{W_i\sqrt{P}\mathcal R_{2h}}(k_\parallel,k_\perp)
=
\int_{\chi_{i,\rm lo}}^{\chi_{i,\rm hi}}
d\chi\,W_i(\chi)
\sqrt{P_{\rm lin}\!\left(\sqrt{k_\parallel^2+k_\perp^2},z(\chi)\right)}
\mathcal R_{2h}\!\left(\sqrt{k_\parallel^2+k_\perp^2},z(\chi)\right)
e^{ik_\parallel\chi},
\qquad
k_\perp=\frac{\ell+1/2}{\chi_{i,\rm mid}}.
\]

This keeps the Fourier width of the radial top hat, scale-dependent evolution,
and two-halo response instead of replacing the shell by a radial delta
function. The
integration defaults to 256 radial nodes, 512 line-of-sight nodes, and 40
radial Fourier periods. The same eight-node temporal interpolation avoids a
prohibitive full multipole-by-line-of-sight-by-radial allocation. Each shell
selects either this finite-width result or standard Limber according to which
has the smaller maximum error over the final exact-comparison window. This
comparison uses the corrected two-halo spectrum. The selected branch must then
satisfy the stable one-percent transition test
described above. The mode is recorded as `finite_width_flat_sky` or `limber`
in the NPZ and diagnostics.

For standard Limber, and for the one-halo term at every multipole, the code uses

\[
C_\ell=\int d\chi\,\frac{W_i(\chi)^2}{\chi^2}
P\!\left(\frac{\ell+1/2}{\chi},z(\chi)\right).
\]

Here `P` is `P_2h` for the deterministic term and `P_1h` for the stochastic
term. `P_lin` is projected and stored independently as a normalization and
model-comparison diagnostic.

The spectrum of the summed map uses measured mean-count shell weights. Its
broad radial window retains standard Limber as the high-multipole branch,
while exact low-multipole projection retains cross-shell correlations.
Disjoint one-halo shells have no Limber cross term. A HEALPix pixel window is
applied to clustering
terms.

## Power-Spectrum Normalization

The validation reconstructs the present-day normalization directly from the
tabulated spectrum,

\[
\sigma_8^2 = \int d\ln k\,\frac{k^3P_0(k)}{2\pi^2}
\left[\frac{3(\sin kR_8-kR_8\cos kR_8)}{(kR_8)^3}\right]^2,
\qquad R_8=8\,\mathrm{Mpc}/h.
\]

When `Sigma8` in the PINOCCHIO parameter file is positive, it is the reference
value. When it is zero, the effective `COS_S8` written to every mass-map FITS
header is used instead. Older mass maps may omit `COS_S8`; if every shell omits
it, the value reconstructed from the cosmology-table power spectrum is used as
a fallback and recorded with source `cosmology_power_spectrum`. That fallback
is not an independent normalization closure. Partial or inconsistent header
coverage remains an error. When an independent reference exists, the run fails
by default if it differs from the reconstruction by more than one percent.

## Units And Shot Noise

The cosmology reader performs these PINOCCHIO-to-GEPPETTO conversions:

- distance: `Mpc -> Mpc/h` by multiplying by `h`;
- wavenumber: `Mpc^-1 -> h/Mpc` by dividing by `h`;
- power: `Mpc^3 -> (Mpc/h)^3` by multiplying by `h^3`.

The optional CAMB series is already in `h/Mpc` and `(Mpc/h)^3` when
`InputSpectrum_UnitLength_in_cm=0`, so no second unit conversion is applied.

For uncollapsed mean count `n_uncollapsed`, total mean count `n_total`, and
pixel area `Omega_pix`, the particle-count shot-noise level is

\[
N_\ell=\Omega_{\rm pix}\frac{n_{\rm uncollapsed}}{n_{\rm total}^2}.
\]

Halo discreteness is already represented by the one-halo term and is not added
again as particle shot noise. The uncollapsed-particle term is reported as a
Poisson baseline; PINOCCHIO particle correlations can make the realized noise
depart from that baseline.

## Cut-Sky Estimator And Outputs

The validation command builds the actual binary RING mask from the compact
pixel list. NaMaster removes a constant template, matching normalization by the
mean within the footprint, and measures the resulting pseudo-spectrum divided
by `f_sky`. Each full-sky theory component is transformed with the same exact
MASTER coupling matrix and constant-template deprojection bias before it is
compared with the map. The default comparison excludes `ell < 20` and uses
`Delta ell = 20` bins. No noisy matrix inversion or pseudo-spectrum
deconvolution is performed.

The reference field must contain a zero-valued map as well as the template.
NaMaster treats `maps=None` as a mask-only field and returns before processing
templates. An audit found that older cut-sky theory archives therefore omitted
the constant-deprojection bias even though the measured maps removed the mean.
New archives record `mask_reference_template_count=1`. Older full-sky theory
components and measured spectra remain usable, but their cut-sky theory must
be re-coupled before making comparisons under the corrected convention.
`prepare_angular_mask_response.py` can cache a binned forward response and checks
it against direct NaMaster calls; it reports differences from legacy coupling
separately rather than requiring agreement with that incorrect convention.

Outputs are:

- `angular_power_theory.npz`: schema-v5 unbinned measured spectra, full-sky
  base components, mask-coupled comparison components, normalization closure,
  fitted bias arrays, exact-to-high-ell projection diagnostics, the
  linear-evolution source, the CCToolkit revision, and the one-halo
  compensation convention, including each shell's mode;
- `angular_power_binned.csv`: binned measured, linear diagnostic, corrected
  two-halo, one-halo, shot-noise, clustering, and total spectra;
- `angular_power_diagnostics.csv`: shell weights, map means, resolved HMF mass
  fractions, one-halo/linear and two-halo/linear ratios at the lowest tabulated
  k, the reconstructed `sigma8` closure, high-ell transition and mode, and the
  mask convention;
- `angular_power_bias_fit.csv`: one row per native HMF redshift with fitted
  multiplicity parameters, fit residual, and PBS, correction, and corrected
  bias ranges.

Install the validation and plotting extras and generate the publication figures with:

```bash
conda install -c conda-forge namaster
python -m pip install -e '.[validation,plot]'
python examples/plot_angular_power_validation.py \
  --input-dir /path/to/angular-power-validation \
  --output-dir /path/to/figures
```

The plotting command validates the archive and table schemas, shell ordering,
multipole bins, `f_sky`, and NSIDE before writing vector PDF and 300-dpi PNG
versions of the summed-spectrum decomposition and shell-residual heatmap. The
command also writes a four-panel spectrum decomposition for representative
shells nearest its configurable target redshifts. The gray residual bands are
Gaussian mode-counting guides, not covariance estimates.

At high NSIDE, map measurement streams one shell at a time and reuses one
full-sky NaMaster input buffer. The full-node submission requests the usable
DCGP node memory explicitly because the mask, harmonic coefficients, MASTER
workspace, and spawned exact-projection runtimes coexist even though the input
maps use compact pixel rows.

The compensated one-halo diagnostic ratio should approach zero and the
two-halo/linear ratio should approach one toward the lowest tabulated k. The
model is deliberately labelled `corrected two_halo + compensated one_halo`.
It includes the Castro bias correction but no halo exclusion or nonlinear
matter-power calibration.

## Full-Sky Component Audit

For a full-sky run, `measure_fullsky_component_power.py` measures the
uncollapsed auto-spectrum `UU`, painted-halo auto-spectrum `HH`, and their
cross-spectrum `UH`. Both component maps have their monopoles removed
separately and are divided by the same theoretical total-matter mean. The
measured spectra therefore obey the exact map-level identity

\[
C_\ell^{\rm total}=C_\ell^{UU}+2C_\ell^{UH}+C_\ell^{HH}.
\]

`audit_fullsky_component_decomposition.py` first subtracts the realized
uncollapsed-particle Poisson level

\[
N_\ell^U=\Omega_{\rm pix}
\frac{\bar n_U^{\rm measured}}{(\bar n_{\rm total}^{\rm theory})^2}
\]

and then projects the halo field onto the clustered uncollapsed field:

\[
C_\ell^{HH,\parallel}
=\frac{(C_\ell^{UH})^2}{C_\ell^{UU}-N_\ell^U},
\qquad
C_\ell^{HH,\perp}=C_\ell^{HH}-C_\ell^{HH,\parallel}.
\]

The corresponding component-coherent spectrum is

\[
C_\ell^{\rm coherent}
=C_\ell^{UU}-N_\ell^U+2C_\ell^{UH}+C_\ell^{HH,\parallel}.
\]

This is an exact regression decomposition relative to the measured
uncollapsed field. It is not a unique separation into deterministic and
stochastic contributions relative to the unknown initial linear mode. Bins
where the shot-subtracted `UU` is non-positive, or where the resulting
covariance is not positive semidefinite, are marked invalid. Ensemble
uncertainties on nonlinear derived quantities use a delete-one-realization
jackknife.

The audit compares `C_coherent` with both the projected linear spectrum and
the schema-5 corrected two-halo spectrum. This directly tests whether the
response assumed for the sum of uncollapsed and painted-halo fields agrees
with the map components, while keeping the orthogonal `HH` residual outside
that comparison.
