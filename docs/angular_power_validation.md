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
\left(\frac{M}{\bar\rho_m}\right)^2 |u(k|M,z)|^2.
\]

`P_linear` is read from the PINOCCHIO cosmology table at `z=0` and scaled by
the tabulated growth factor squared. It serves as the large-scale/two-halo
approximation. This release does not introduce a halo-bias relation, a smooth
residual component, or one-halo compensation.

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

## Angular Projection

For a shell bounded by `chi_lo` and `chi_hi`, the count-overdensity window is

\[
W_i(\chi)=\frac{3\chi^2}{\chi_{i,\rm hi}^3-\chi_{i,\rm lo}^3}.
\]

The linear term uses the exact spherical-Bessel expression at low multipoles:

\[
C_\ell^{\rm lin}=\frac{2}{\pi}\int dk\,k^2P_0(k)
\left|\int d\chi\,W_i(\chi)D(\chi)j_\ell(k\chi)\right|^2.
\]

The integral is restricted to the tabulated PINOCCHIO k range. Exact spectra
are evaluated in batches until each shell and the count-weighted sum
independently agree with Limber within one percent for 20 consecutive
multipoles. Each spectrum switches at the first multipole in its own
confirmation interval. The default exact search cap is `ell=512`; failure of
any spectrum to converge before the cap aborts the validation and reports its
maximum relative error over the final confirmation window.
Independent exact multipoles can be evaluated concurrently with spawned worker
processes selected by `--exact-workers`. Each child is restricted to one native
thread without inheriting the parent OpenMP affinity, allowing the operating
system to distribute workers across the task's allocated cores. The Leonardo
submission example uses all 112 physical cores and checks the Limber criterion
after each 112-multipole batch. The larger batch may calculate exact
multipoles beyond the eventual transition, but allows the full node to work
concurrently.

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

Above the switch, and for the one-halo term at every multipole, the code uses

\[
C_\ell=\int d\chi\,\frac{W_i(\chi)^2}{\chi^2}
P\!\left(\frac{\ell+1/2}{\chi},z(\chi)\right).
\]

The spectrum of the summed map uses measured mean-count shell weights. Exact
linear projection retains cross-shell correlations. Disjoint one-halo shells
have no Limber cross term. A HEALPix pixel window is applied to clustering
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

- `angular_power_theory.npz`: schema-v2 unbinned measured spectra, full-sky
  base components, mask-coupled comparison components, normalization closure,
  and per-shell plus summed-spectrum exact-to-Limber diagnostics;
- `angular_power_binned.csv`: binned measured, linear, one-halo, shot-noise,
  clustering, and total spectra;
- `angular_power_diagnostics.csv`: shell weights, map means, resolved HMF mass
  fractions, the one-halo/linear ratio at the lowest tabulated k, the
  reconstructed `sigma8` closure, the exact-to-Limber transition, and the mask
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

The uncompensated one-halo term approaches a constant at low k. Its diagnostic
ratio must be inspected before interpreting large-scale agreement. The model
is deliberately labelled `linear + one_halo`; it is not a formally normalized
halo-bias calculation of the full two-halo term.
