# Angular-Power Validation

GEPPETTO predicts the angular power spectra of the count-overdensity maps made
by adding each painted NFW NPZ array to its original PINOCCHIO uncollapsed-
particle FITS segment. The implementation is intended to validate the map
painting and its concentration dependence, not to replace a precision survey
likelihood.

## Three-Dimensional Model

The current model is

\[
P_{\rm mm}(k,z)=P_{\rm lin}(k,z)+P_{1h}(k,z),
\]

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

The linear term is the large-scale/two-halo approximation. This release does
not introduce a halo-bias relation, a nonlinear two-halo correction, or an
explicit halo-exclusion model; those omissions can matter in the transition
regime but do not justify a white one-halo floor at low k.

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

The change above follows a controlled paired-fixed test, not a fit to the
validation plot. Four independent phase pairs were compared over the first
seven full-sky shells (`0.23 < z < 0.49`) using only modes with
`k=(ell+0.5)/chi < 0.05 h/Mpc`. Against scalar-growth linear theory, the
pair-mean map amplitude was `0.991997 +/- 0.001584`, a 5.05-sigma deficit.
Using the tabulated scale-dependent evolution changed it to
`0.998711 +/- 0.001593`, consistent with unity. Adding the former
uncompensated one-halo floor lowered the total-theory ratio to
`0.987577 +/- 0.001572`. Thus the controlled maps do not lose large-scale
power: the two statistically resolved effects were both in the theory model.

## Angular Projection

For a shell bounded by `chi_lo` and `chi_hi`, the count-overdensity window is

\[
W_i(\chi)=\frac{3\chi^2}{\chi_{i,\rm hi}^3-\chi_{i,\rm lo}^3}.
\]

The linear term uses the exact spherical-Bessel expression at low multipoles:

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

Completed exact batches are atomically accumulated in
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
\left|\widetilde{W_i\sqrt{P}}(k_\parallel,k_\perp)\right|^2,
\]

where

\[
\widetilde{W_i\sqrt{P}}(k_\parallel,k_\perp)
=
\int_{\chi_{i,\rm lo}}^{\chi_{i,\rm hi}}
d\chi\,W_i(\chi)
\sqrt{P_{\rm lin}\!\left(\sqrt{k_\parallel^2+k_\perp^2},z(\chi)\right)}
e^{ik_\parallel\chi},
\qquad
k_\perp=\frac{\ell+1/2}{\chi_{i,\rm mid}}.
\]

This keeps both the Fourier width of the radial top hat and scale-dependent
evolution instead of replacing the shell by a radial delta function. The
integration defaults to 256 radial nodes, 512 line-of-sight nodes, and 40
radial Fourier periods. The same eight-node temporal interpolation avoids a
prohibitive full multipole-by-line-of-sight-by-radial allocation. Each shell
selects either this finite-width result or standard Limber according to which
has the smaller maximum error over the final exact-comparison window. The
selected branch must then satisfy the stable one-percent transition test
described above. The mode is recorded as `finite_width_flat_sky` or `limber`
in the NPZ and diagnostics.

For standard Limber, and for the one-halo term at every multipole, the code uses

\[
C_\ell=\int d\chi\,\frac{W_i(\chi)^2}{\chi^2}
P\!\left(\frac{\ell+1/2}{\chi},z(\chi)\right).
\]

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

Outputs are:

- `angular_power_theory.npz`: schema-v4 unbinned measured spectra, full-sky
  base components, mask-coupled comparison components, normalization closure,
  exact-to-high-ell projection diagnostics, the linear-evolution source, and
  the one-halo compensation convention, including each shell's mode;
- `angular_power_binned.csv`: binned measured, linear, one-halo, shot-noise,
  clustering, and total spectra;
- `angular_power_diagnostics.csv`: shell weights, map means, resolved HMF mass
  fractions, the one-halo/linear ratio at the lowest tabulated k, the
  reconstructed `sigma8` closure, high-ell transition and mode, and the mask
  convention.

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

The compensated one-halo diagnostic ratio should approach zero toward the
lowest tabulated k. The model remains deliberately labelled
`linear + one_halo`: compensation restores the required large-scale limit but
does not turn it into a calibrated nonlinear halo-bias, exclusion, or
two-halo-transition model.
