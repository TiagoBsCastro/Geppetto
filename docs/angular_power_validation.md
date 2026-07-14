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

The linear term uses the exact spherical-Bessel expression through the
configured switch, defaulting to `ell=100`:

\[
C_\ell^{\rm lin}=\frac{2}{\pi}\int dk\,k^2P_0(k)
\left|\int d\chi\,W_i(\chi)D(\chi)j_\ell(k\chi)\right|^2.
\]

The integral is restricted to the tabulated PINOCCHIO k range. Its fixed
radial quadrature retains 40 oscillations beyond the transverse shell scale;
this avoids aliasing radial cancellations and is covered by quadrature-order
and exact-to-Limber convergence tests.

Above the switch, and for the one-halo term at every multipole, the code uses

\[
C_\ell=\int d\chi\,\frac{W_i(\chi)^2}{\chi^2}
P\!\left(\frac{\ell+1/2}{\chi},z(\chi)\right).
\]

The spectrum of the summed map uses measured mean-count shell weights. Exact
linear projection retains cross-shell correlations. Disjoint one-halo shells
have no Limber cross term. A HEALPix pixel window is applied to clustering
terms.

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

The validation command builds a binary mask from the compact RING pixel list,
subtracts the mean within that footprint, and reports
`healpy.anafast(mask * delta) / f_sky`. This is only an approximate cut-sky
estimator. The default comparison excludes `ell < 20` and bins with
`Delta ell = 20` to reduce visible mode coupling. Precision work should replace
this estimator with an explicit mode-coupling calculation.

Outputs are:

- `angular_power_theory.npz`: unbinned measured spectra and all theory
  components for each shell and the summed map;
- `angular_power_binned.csv`: binned measured, linear, one-halo, shot-noise,
  clustering, and total spectra;
- `angular_power_diagnostics.csv`: shell weights, map means, resolved HMF mass
  fractions, and the one-halo/linear ratio at the lowest tabulated k.

The uncompensated one-halo term approaches a constant at low k. Its diagnostic
ratio must be inspected before interpreting large-scale agreement. The model
is deliberately labelled `linear + one_halo`; it is not a formally normalized
halo-bias calculation of the full two-halo term.
