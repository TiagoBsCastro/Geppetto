# Painting-Matched Halo Model

## Scientific Requirement

The prediction must follow the implemented map operation, not a fitted
interpolation between existing angular spectra. No calibrated multipole
transition, generalized-mean blend, or arbitrary particle-noise damping is
part of this model. An earlier exploratory transition fit is not adopted.

The target remains a comparison against schemes 5 and 6 over
`20 <= ell < 2000` in all four representative shells. The discrete assignment
operator below is validated. The experimental constrained-covariance
prediction passes the four-panel descriptive RMS comparison, but **its
PINOCCHIO backbone closure has not been independently validated**. It is not
the production default.

## Exact Assignment Identity

For one shell, let U_p be the uncollapsed particle count, Q_h=M_h/m_particle
the catalogue halo count, and w_hp the actual native-pixel assignment:

\[
C_p=U_p+\sum_h Q_h w_{hp},\qquad \sum_{p\in\mathrm{global}}w_{hp}=1.
\]

Weights come from the production painter, including its exact finite
line-of-sight NFW projection, global discrete normalization, and NGP,
supersampled, and native branches. Supersampled child contributions are
aggregated into native pixels before forming angular-power moments.
The stencil is concentration independent; the weights and their
normalization remain differentiable in the concentration parameters.

For equal-area pixels and a stated common total-shell mean count bar(C),
the unmasked harmonic quadrature at ell>0 is

\[
a_{\ell m}=\frac{\Omega_{\rm pix}}{\bar C}
\left[\sum_p (U_p-\bar C)Y^*_{\ell m}(\hat n_p)
+\sum_h Q_h\sum_p w_{hp}Y^*_{\ell m}(\hat n_p)\right].
\]

The constant term has only a monopole in continuous spherical harmonics.
It is written explicitly because raw HEALPix quadrature can leak a constant
into higher multipoles; the production constant-deprojection estimator must
not be replaced silently by an ideal continuous transform.

This is the starting point. There is no extra linear field to add after
summing the complete particle and halo contributions. The linear limit must
come from their joint statistics. Compact-footprint filtering and the actual
mask/constant-deprojection estimator are subsequent operations. Clipped halo
weights must never be renormalized inward.

## Two Different Profile Moments

For a fixed halo assignment define

\[
A_{h\ell}=\sum_p w_{hp}P_\ell(\hat n_{h,\rm ref}\cdot\hat n_p),
\qquad
D_{h\ell}=\sum_{pq}w_{hp}w_{hq}P_\ell(\hat n_p\cdot\hat n_q).
\]

The spherical-harmonic addition theorem makes D the **exact same-halo**
factor for the discrete pixel assignment. See
[DLMF 14.30.9](https://dlmf.nist.gov/14.30.E9).
A describes the orientation-averaged amplitude relative to the reference
halo position. The reference can be the true halo direction or its native
NGP pixel centre; it must match the convention of the backbone spectra.

In general **D is not A squared**. Even one fixed non-axisymmetric
assignment has contributions to its self power that its azimuthally averaged
amplitude does not retain. On an ensemble, averaging and squaring cannot
be interchanged either. Averages for the same-halo term are weighted by Q^2,
whereas field amplitudes are weighted by Q and their correlations.

For an NGP halo referenced to its host-pixel centre, A=D=1 at every ell.
Its self contribution is white:

\[
C_{\ell,h}^{\rm self}=
\frac{1}{4\pi}\left(\frac{\Omega_{\rm pix}Q_h}{\bar C}\right)^2D_{h\ell}.
\]

Do not multiply D by an additional HEALPix pixel window. It already describes
the actual discrete assignment. In particular, applying a continuous-field
pixel window to the NGP same-halo term is not the pixel-count quadrature
identity. The production theory currently windows the entire one-halo term;
that convention needs a controlled replacement, not a change to the painter.
The HEALPix iterative harmonic estimator is not identical to the direct
quadrature for an arbitrarily non-band-limited pixel map; its small numerical
response is checked separately in the example.

## Component Model

Define mass/redshift-bin reference halo maps H_i and particle map U, all
normalized by the same total-shell mean. Let S_i be the white self amplitude
of H_i, and let the input C^{H_iH_j} include this self diagonal. Then

\[
C_\ell^{\rm paint}=C_\ell^{UU}
+2\sum_i A_{i\ell}C_\ell^{UH_i}
+\sum_{ij}A_{i\ell}A_{j\ell}
  [C_\ell^{H_iH_j}-\delta_{ij}S_i]
+\sum_i D_{i\ell}S_i.
\]

The same-halo piece is exact when its Q^2-weighted D is supplied.
Factorizing distinct-halo and particle/halo terms assumes independent,
statistically isotropic profile orientations at fixed sufficiently narrow
mass/redshift bins, independent of environment. This factorization is an
approximation, not an exact identity for a particular HEALPix realization.
It requires tests under subpixel shifts and bin refinement.

This expression includes the particle/halo cross term and halo exclusion
through the input covariances. It neither assumes all these covariances are
positive nor adds independent Poisson noise to already complete spectra.
There are no haloes below the supplied catalogue/HMF selection in these sums.

## Match The Painted Population

The mass integrals must use the population that actually enters the painter.
For an ensemble prediction this is the finite, resolved PINOCCHIO HMF with
the same catalogue selection. For a catalogue-conditioned prediction it is
the actual PLC mass/redshift population. The latter includes that
realization's abundance fluctuations and must be identified as conditional,
not an independent ensemble expectation.

Neither choice permits extrapolating the HMF to unresolved halo masses and
painting those objects in the theory: the corresponding simulation matter
is still in U. In particular, imposing integral M*n(M) dM = rho_bar on the
resolved HMF would replace the uncollapsed component with fictitious haloes.
That is a different matter model from the map operation. A fully normalized,
all-mass halo model such as the schema-6 benchmark is useful for comparison
but is not an analytical description of this finite-catalogue painting.

The concentration relation changes w_hp, not the abundance or the component
mass budget. The NGP and supersampling thresholds in the painting manifest
must be used when computing A and D; a continuous NFW Fourier transform
alone does not reproduce those numerical assignments.

## Large-Scale Closure

For the resolved population, with n(M)=dn/dM,

\[
f_H=\bar\rho^{-1}\int_{\rm resolved}M n(M)\,dM,\quad f_U=1-f_H,
\]
\[
f_Ub_U+\bar\rho^{-1}\int_{\rm resolved}M n(M)b_H(M)\,dM=1.
\]

Thus the resolved halo HMF and bias are not individually renormalized to
represent all matter. A halo-plus-uncollapsed decomposition is also discussed
by [Smith & Markovic](https://arxiv.org/abs/1103.2134); their physical application
differs from this numerical resolution split.

Writing each component as its linear bias times the matter field plus a
stochastic residual gives a covariance matrix E_ij(k). At low k the complete
matter stochastic covariance must obey the appropriate conservation and
particle-sampling constraints, not independent Poisson assumptions for every
component. At high k the distinct-object correlations can become small and
the explicit same-halo term supplies the one-halo regime.

The linear input spectrum, HMF, and halo bias constrain the deterministic
large-scale limit. **They do not uniquely determine E_ij(k), exclusion,
nonlinear particle clustering, or the particle/halo cross-correlation.**
Supplying measured unpainted backbone covariances would give a conditional
painting prediction, not an independent linear-theory-only forecast. A purely
analytical forecast instead needs an explicit, independently validated model
for those quantities. These two kinds of prediction must not be mislabeled.

### Predict Directly From PINOCCHIO Inputs

The experimental `examples/predict_painting_matched_power.py` workflow now
generates the statistical tables as well as the assignment moments. Install
`.[io,theory,lpt,plot]` for this example. It does not require NaMaster unless
a new mask-coupling response also needs to be computed.

```bash
SIM=/path/to/Pinocchio/simulation
python examples/predict_painting_matched_power.py \
  --manifest "$SIM/geppetto_reduced/painted_nfw_manifest.csv" \
  --params "$SIM/params.txt" \
  --cosmology-table "$SIM/pinocchio.000.cosmology.out" \
  --hmf-glob "$SIM/pinocchio.*.000.mf.out" \
  --linear-reference examples/angular_power_validation_schema5/angular_power_theory.npz \
  --segments 23 9 3 0 \
  --derivatives \
  --output outputs/painting_prediction.npz

python examples/plot_painting_matched_audit.py \
  --projection outputs/painting_prediction.npz \
  --mask-response outputs/two_regime_research/mask_response_deprojected.npz \
  --output-dir outputs/painting_comparison
```

The linear reference and mask response must describe the same cosmology,
shells, estimator, and NSIDE as the manifest. The reference contributes only
its already calculated theoretical linear projection, never its measured
spectra. Its sigma8 and growth prescription are checked; legacy archives
without segment IDs must contain every shell in sorted manifest order.
These checks do not establish a full cosmology match for an arbitrary old
archive, so that remains an input requirement. Distances are reconstructed
from redshift using the cosmology table, not legacy manifest chi fields.
Only true-redshift selection with unconverted catalogue masses is supported.

The calculation uses native counts divided by box volume, without an extra
mass-bin width or HMF extrapolation. A/D moments come from the actual
production adaptive painter at fixed geometry. Mass blocks bound memory;
`--mass-batch-size` changes execution, not the physical assignment. Numerical
settings `--orientations`, `--angle-nodes`, `--radial-order`, `--los-periods`,
and `--los-order` should be refined for a new target. Optional derivatives
propagate all three concentration parameters through global normalization,
native child aggregation, both moments, and the covariance contraction.

`geppetto.lpt_backbone` is a host-side, concentration-independent model, not
a differentiable cosmology solver. It uses optional `velocileptors==3.1`
displacement correlators and an isotropic cubic-lattice pair calculation.
Gaussian conditioning inside uniform spherical protohaloes approximates the
particle pairs replaced by halo assignment. That selection is **not** an
exact representation of PINOCCHIO fragmentation. The rank-one component
covariance remains a separate closure assumption.

`read_pinocchio_lpt_growth_ratios` reads the actual second-order and relevant
third-order growth ratios from the cosmology table. They multiply the
appropriate displacement correlators relative to their EdS convention;
the input P(k,z) already includes linear growth and is not grown twice.
For CAMB-table runs, the supplied scale-dependent P(k,z) is retained.
The z=0 input spectrum, not a fitted normalization, supplies sigma8.

One output NPZ contains the five stationary components, their total with
the independently calculated linear-projection replacement, optional
concentration Jacobians, and provenance. The correction is kept separate
from the component decomposition; it is not added a second time to the
total. The report includes input/spectrum hashes, native mass fractions,
higher-order growth ratios, endpoint continuations, and numerical settings.
No map or per-mass temporary files are written.

The four-shell total comparison and the independent component audit answer
different questions. A small total residual does not validate each UU, UH,
and HH term. In particular, the current statistical closure still fails that
component-level check; neither consistent displacement resummation of a
Lagrangian-bias expansion nor matching the mass budget alone solved it.
This example stays experimental and does not replace the production theory
default or change the map painter.

The raw-input run with three radial nodes, 16 orientations per mass, and
PINOCCHIO's actual higher-order growth ratios gives the following comparison.
The metric is unweighted RMS(measured/theory-1) across 99 mode-count-weighted
bands of width 20 over `20 <= ell < 2000`, not pointwise superiority or a
covariance-weighted goodness of fit.

| Redshift | Scheme 5 RMS | Scheme 6 RMS | Painting-matched RMS |
| --- | ---: | ---: | ---: |
| 0.174-0.211 | 17.024% | 10.411% | 2.069% |
| 0.779-0.832 | 3.771% | 13.159% | 2.106% |
| 1.399-1.554 | 2.414% | 4.248% | 1.686% |
| 1.907-2.000 | 3.203% | 2.839% | 1.728% |

At the central radial nodes the resolved halo mass fractions are 0.13044,
0.05341, 0.01270, and 0.00328. The rest remains in the uncollapsed component;
it is not assigned to extrapolated low-mass haloes. This is essential to
describing the actual finite-resolution painting rather than a hypothetical
all-mass halo catalogue. The previous numerical screens below are retained
as the development record, not alternative production prescriptions.

The full four-shell run also succeeds with `--derivatives`: all entries of
the `(4,3,5,4095)` Jacobian are finite, and component derivatives sum to the
total derivative. Enabling them changes spectra by at most 6.3e-14
fractionally. Unit tests compare the three population moment derivatives
against Richardson-extrapolated finite differences and independently test
the JAX covariance contraction. These are derivative/implementation checks,
not validation of the statistical closure.

## Implementation And Verification

`geppetto.painting_theory` provides JAX-compatible typed containers and:

- `angular_assignment_moments`: exact A and D from actual native-pixel weights;
  a static pixel-pair chunk bounds memory.
- `catalogue_self_pair_amplitude`: the full-sky Q^2-weighted self amplitude.
- `assemble_painted_angular_power`: the explicit covariance contraction above.

It does not change `painters.py` or replace the existing production theory.

```bash
python -m pytest tests/test_painting_theory.py
python examples/validate_painting_theory_operator.py \
  --output-dir outputs/painting_operator
```

The example exercises the production stencil builder and painter in all three
branches, compares against direct harmonic quadrature, checks mass closure,
and reports iterative-SHT effects separately. Its distances are deliberately
chosen to exercise the angular branches, not to represent cosmological shells.
Tests also cover concentration JVPs against finite differences, unchanged NGP
derivatives, same-object counting, cross-term factors, zero weights, chunking,
and the distinction between squared response and self power.

At NSIDE=64, lmax=128, the direct quadrature agrees with D/(4*pi) to maximum
relative errors of 7.5e-15 (NGP), 3.6e-13 (supersampled), and 1.7e-13 (native).
Three-iteration harmonic estimates differ by at most 2.0e-4, 1.1e-4, and
2.4e-5 respectively. These are synthetic assignment tests, not an error bound
for the production masked estimator.

A separate midpoint-redshift diagnostic using the retrieved L3870N2160 native
HMFs and actual painting manifest gives the following fraction of the
**zero-wavenumber one-halo weight**, integral n(M) M^2 dM, in NGP haloes:

| Representative segment | Midpoint redshift | NGP fraction of one-halo weight |
| --- | ---: | ---: |
| 23 | 0.19244 | 0.0% |
| 9 | 0.80529 | 19.1% |
| 3 | 1.47677 | 90.6% |
| 0 | 1.95372 | 99.7% |

This uses the measured finite HMF, the manifest threshold
`theta_resolution_rad=0.0002609545367856184`, and hard R_200c support.
It is not a halo number fraction, a collapsed mass fraction, or a projected
bandpower fraction. It suggests the NGP self-power convention matters most
at high redshift; it cannot by itself explain the lowest-redshift mismatch.

Remaining work: select and validate the backbone-covariance closure, average
the numerical assignment moments over the resolved population and subpixel
geometry, apply the actual mask estimator, and compare all four shells below
ell=2000 without fitting to their painted spectra. Better agreement is an
empirical acceptance test, not something implied by the assignment identity.

## Experimental Lagrangian Pair Closure

An analytical candidate replaces same-protohalo particle pairs in a displaced
particle lattice with the painter's actual halo self pairs. It uses the native
finite HMF, with no low-mass extrapolation and no rescaling of its collapsed
fraction to one. It does not fit any multipole transition to the painted maps.
This is a **candidate backbone closure**, not an exact consequence of painting
and not a replacement for the production theory.

The continuum starting point follows the Lagrangian pair decomposition in
[Valageas & Nishimichi, equations 20--26](https://arxiv.org/abs/1009.0597).
Let R_L=(3M/(4*pi*rho_bar))^(1/3) and define the fractional overlap of two
spherical protohaloes by

\[
O(q/R_L)=1-\frac{3q}{4R_L}+\frac{q^3}{16R_L^3}
\quad (q<2R_L),
\]

with zero overlap otherwise. A mass-weighted finite-HMF sum of this overlap
gives the probability that two Lagrangian mass elements belong to the same
resolved halo. The displacement characteristic J(k,q) is approximated using
one-loop resummed LPT, currently tabulated with the MIT-licensed
[velocileptors CLEFT implementation](https://github.com/sfschen/velocileptors).
The approximation replaces collapse-conditioned distinct-pair displacement
statistics by unconditional LPT statistics. That is an important limitation;
linear bias and an HMF do not prove this closure.

For lattice spacing a=L_box/N_grid and radial shell degeneracies d_q, discrete
protohalo overlaps are normalized to the actual particle pair count. For each
mass node Q=M/m_particle, the nonzero-separation overlap is multiplied by

\[
s_M=\frac{Q-1}{\sum_{q>0}d_q O(q/R_L)}.
\]

At q=0 the same-halo probability is f_H; at nonzero q it is the mass-weighted
sum of s_M O(q/R_L). Consequently

\[
a^3\sum_q d_q F_{1h}^{\rm lattice}(q)
=\int_{\rm resolved}d\ln M\,\frac{dn}{d\ln M}
  \left(\frac{M}{\bar\rho}\right)^2\equiv S_H.
\]

This normalizes a halo's particle pair count, **not its abundance or bias**.
The lattice correction is the discrete-minus-continuous integral of the
connected displacement characteristic. The candidate background is

\[
P_{\rm bg}=P_{\rm LPT}+\Delta P_{\rm lattice}
-a^3\sum_q d_q F_{1h}^{\rm lattice}(q)J(k,q).
\]

Adding the actual halo self term restores S_H at k=0 and cancels its subtraction.
At high k, decorrelated lattice sites leave f_U a^3, the uncollapsed-particle
self term. No additional white particle noise is added to this background.
Constant cancellation does not establish a k^4 stochastic tail; that stronger
momentum-conservation claim is not made for this approximate closure.

The angular implementation separates this white self term before applying
pixel windows. It combines an existing exact/finite-width linear shell
projection with a projected correction from P_bg-P_linear-f_U*a^3, the
uncollapsed self term, and the population-averaged actual pixel-pair D_ell.
Only the distinct-particle correction receives an additional pixel window.
The current nonlinear correction uses Limber projection and must be checked
against finite-width projection before acceptance.

### Numerical Audit Status

The local research bundle is `outputs/two_regime_research/`. Its inputs are
the real L3870N2160 cosmology/CAMB tables, finite HMF files, and painting
manifest. The linear evolution is scale dependent; the native lattice spacing
is 1.7916667 Mpc/h. Profile self power uses the production adaptive stencil and
JAX painter, averages independently sampled subpixel directions, and integrates
the HMF with its original finite mass grid. A finely spaced pair-angle histogram
accelerates the Legendre sum; histogram and orientation convergence are separate
numerical checks, not fitted physical parameters.

The initial midpoint screen using an unrestricted input spectrum beat both
schemes in all four panels. **That result is not accepted.** Inspection of
PINOCCHIO's `src/GenIC.c` and `src/pinocchio.h` on Leonardo shows a spherical
Nyquist cutoff (`NYQUIST=1`), not an unrestricted spectrum or a Fourier-cube
average. Enforcing that physical input changes the intermediate-redshift
prediction appreciably. A better numerical fit with the wrong initial spectrum
is not grounds to adopt it.

The source inspected on Leonardo has PINOCCHIO revision
`e1df32e92f6a5ea419812376d0f932185186b191`; the cutoff is enforced in
`GenIC.c:280` and defined in `pinocchio.h:56`. The candidate also removes the
perturbation library's default Gaussian input cutoff when matching these initial
conditions. This source inspection is not a substitute for preserving the
original simulation binary/build provenance.

The 8-to-16-direction check changes the integrated halo self spectrum below
ell=2000 by at most 0.193%, 0.226%, 0.036%, and 0.0023% in segments 23, 9, 3,
and 0. Halving the pair-angle grid resolution changes it by at most 2.1e-7
fractionally. The largest single-halo numerical mass-closure error is 1.6e-11.
These bounds validate parts of the numerical assignment calculation, not the
accuracy of the LPT/exclusion closure or its radial projection.

A second audit found that the old NaMaster theory reference used `maps=None`,
which skips its template initialization. The measured maps did remove the mean.
The reference now supplies a zero map, checks that one template is retained,
and is regression-tested with the installed NaMaster on Leonardo. Comparison
plots must re-couple **both** baseline schemes and the candidate consistently.
This operator accounts for linear constant removal at a fixed shell
normalization; it does not model correlations with the random measured mean
in the denominator of a density estimator.

`plot_painting_matched_audit.py` produces a common-estimator comparison without
fitting parameters. Its RMS metric is descriptive, not covariance weighted.
The remaining acceptance work is to validate the displacement and exclusion
closure against the unpainted backbone, quantify finite-width corrections and
quadrature/subpixel convergence, and then repeat the four-panel comparison.
The painting operator can be correct while the backbone model is inaccurate.

### Common-Estimator Comparison

The completed corrected-mask audit is available in
`examples/angular_power_painting_matched_audit/`, including PNG/PDF plots,
numerical bandpowers, and CSV/JSON summaries. It uses the spherical initial
Nyquist cutoff without an extra Gaussian filter, q_max=128 Mpc/h, three radial
nodes, and 16 subpixel orientations per resolved mass node.

| Segment | Redshift range | Scheme 5 RMS | Scheme 6 RMS | Candidate RMS |
| ---: | --- | ---: | ---: | ---: |
| 23 | 0.174--0.211 | 17.024% | 10.411% | 4.757% |
| 9 | 0.779--0.832 | 3.771% | 13.159% | 7.108% |
| 3 | 1.399--1.554 | 2.414% | 4.248% | 1.769% |
| 0 | 1.907--2.000 | 3.203% | 2.839% | 1.719% |

The candidate improves three panels but **fails the all-four-panel criterion**.
In segment 9 it overpredicts the smaller-scale power. It is not promoted to the
production submission script. No amplitude or transition parameter is adjusted
to remove that failure.

The corrected binned mask operator agrees with three independent direct
NaMaster probe calls to maximum relative error 1.3e-10. Relative to the legacy
operator, its largest change in the archived linear/two-halo bandpowers over
20<=ell<2000 is below 0.01%. The mask bug is real but does not explain the
remaining model disagreement. Leonardo job 56990124 completed in 19m35s, with
maximum step RSS 5,402,936 KiB.

Doubling the lattice integral cutoff from 64 to 128 Mpc/h changes the predicted
comparison bandpowers by at most 1.1e-5 fractionally. Multipoles above 3000
contribute at most 5.2e-5 of these bandpowers in the tested model, so deleting
that entire modeled tail has a negligible effect on the present residuals.
This is a sensitivity test of the current tail, not a general bound on an
unknown true spectrum.

The next scientific task is to check the unpainted particle/halo covariance
and PINOCCHIO displacement statistics independently. In particular, the
unconditional one-loop LPT approximation for collapse-selected pairs should
not be tuned against the final painted spectrum and then presented as a
first-principles prediction.

### Subsequent Backbone Diagnostics

A second, still experimental calculation retains the full covariance and
third displacement cumulant in the characteristic function, following the
CLPT organization in [Vlah, Seljak & Baldauf](https://arxiv.org/abs/1410.1617).
The initial spherical Nyquist cutoff, measured HMF, actual painting weights,
and comparison estimator remain unchanged. Direct band-limited Bessel
integrals avoid small-separation covariance errors from the sharp-cutoff
FFTLog transform. This is not an exact prediction of PINOCCHIO's 3LPT field.

Its four RMS errors are 2.802%, 5.230%, 1.699%, and 1.719%, respectively.
It still fails to outperform scheme 5 in segment 9. These results are in
`outputs/two_regime_research/full_cumulant_comparison/`; the published audit
directory above retains the earlier expanded-cumulant calculation.

An independent catalogue audit, `examples/audit_pinocchio_shell_hmf.py`, reads
each PLC part once and bins halo count, M, and M^2 by the original shells.
It checks the selected counts and total masses against the painting manifest
before writing. It does not read or fit the painted angular spectra.
Leonardo job 56996028 passed those checks. The catalogue-to-box-HMF ratios of
the zero-wavenumber one-halo amplitude are approximately:

| Segment | Catalogue / box-HMF one-halo amplitude |
| ---: | ---: |
| 23 | 1.0244 |
| 9 | 1.0330 |
| 3 | 1.0417 |
| 0 | 1.1035 |

These historical numbers use a common theoretical shell-density normalization
and the **previous trapezoidal** box-HMF quadrature. They are not ratios of
observed C_ell and must not all be attributed to abundance fluctuations: the
count-quadrature correction described below changes their denominators.
Changing the population requires recomputing both the profile self term and
its particle/halo model; rescaling only the one-halo spectrum is inconsistent.

### Conditional-Pair Screen

A further research calculation conditions the linear displacement on
delta_R=1.686 in a spherical Lagrangian top-hat. It averages pairs over the
overlap lens of two protohalo spheres and retains the resulting conditional
mean, covariance, and third cumulant. Intrinsic higher-order LPT cumulants
are left unchanged. No concentration, abundance, or transition parameter is
fitted to the painted spectra.

This is exact Gaussian conditioning only for the linear displacement and
the stated single overdensity constraint. Uniform spherical protohaloes,
a fixed spherical-collapse threshold, and the treatment of higher-order
displacements are approximations. They are not PINOCCHIO's full halo
selection or fragmentation algorithm.

The preliminary common-estimator comparison is in
`examples/angular_power_painting_conditioned_audit/`:

| Segment | Scheme 5 RMS | Scheme 6 RMS | Conditional-pair RMS |
| ---: | ---: | ---: | ---: |
| 23 | 17.024% | 10.411% | 2.407% |
| 9 | 3.771% | 13.159% | 1.930% |
| 3 | 2.414% | 4.248% | 1.689% |
| 0 | 3.203% | 2.839% | 1.728% |

These numbers pass the descriptive four-panel RMS comparison, **not the
physical acceptance of a complete painting-matched prediction**. The
experiment currently changes only the same-protohalo subtraction in a
scalar background spectrum and adds the actual painted self term. It does
not yet supply the full component covariance needed to propagate A_ell
through distinct-halo and particle/halo terms, including their concentration
dependence. Consequently it must not be advertised as a validated analytical
replacement for the full painter.

Remaining checks are explicit:

1. Test the conditional displacement statistics against the unpainted
   PINOCCHIO backbone independently of these four painted spectra.
2. Construct the particle/halo covariance inputs and apply both A and D
   using the component model above. Check concentration derivatives of the
   complete prediction, not just its self term.
3. Quantify finite-width nonlinear projection and spectral, radial, lens,
   and angular quadrature errors. In this screen the conditioning integral
   ends at the final input k node below the spherical Nyquist boundary;
   interpolation to the exact boundary still needs a convergence check.
4. Repeat the comparison on independent realizations, using their covariance
   before interpreting a smaller descriptive RMS as statistical preference.

The production theory and submission script are unchanged by this screen.

## Explicit Constrained Covariance

The follow-up implementation uses an explicit particle/halo covariance, so
both assignment moments participate in every relevant term. Its stochastic
part is a mass-weighted rank-one model, motivated by the form in
[Schmidt (2016), equation 33](https://arxiv.org/abs/1511.02231), extended here
to retain uncollapsed particles as a separate component. Our common window
and pair calculation are not that paper's mass-dependent interpolation.

For a radial quadrature node, define N=(N_U,S_1,...,S_n), where N_U is the
uncollapsed-particle self amplitude and S_i is the resolved halo-bin self
amplitude. With N_T=sum_i N_i, use

\[
E_{ij}(\ell)=\delta_{ij}N_i-
  T_\ell\frac{N_iN_j}{N_T}.
\]

At T=1 the total-mass stochastic mode vanishes. At T=0 this becomes diagonal
self noise. For nonnegative N and T<=1 the matrix is positive semidefinite.
This condition is checked on the input pair tables; values are not clipped
to make the model pass. The k^0 cancellation does not establish a k^4 tail.

Let beta_i be the finite-HMF mass-weighted Castro-corrected bias of halo bin
i, beta_U=1-sum_i beta_i, and L_ell the common clustering spectrum. This
closure approximates the correlated component by beta_i*beta_j*L_ell; it
does not include independent nonlinear or tidal halo-bias operators. The
total painted prediction at the node is

\[
C_\ell =
\left[\beta_U+\sum_i\beta_i A_{i\ell}\right]^2 L_\ell
+N_U+\sum_i S_iD_{i\ell}
-\frac{T_\ell}{N_T}
 \left[N_U+\sum_i S_i A_{i\ell}\right]^2.
\]

The particle/halo cross term is explicit and depends on A. D supplies the
same-halo term and must not be replaced by A squared. The sum can equivalently
be evaluated as a weighted response variance plus the nonnegative
orientation variance D-A^2. That form preserves a small conserved total
when its individual component terms are much larger, including in float32.

For the experimental Lagrangian tables, before pixelization,

\[
T(k,z)=\frac{B(k,z)+N_U(z)-\Delta P_{\rm lattice}(k,z)}
                  {N_U(z)+S_H(z)},
\]

where B is the conditional same-protohalo subtraction and S_H=sum_i S_i.
Thus the scale dependence comes from the displacement/lattice calculation,
not a transition fitted to C_ell. The angular node uses the usual pixel
window squared on L and on T, but not on the self amplitudes N_U or S_i*D_i.
It uses native-host-referenced A. With A=D=1 this construction reproduces the
point-halo pair-model spectrum; with actual weights it supplies the missing
profile changes in the cross terms. This covariance allocation is an
additional physical approximation, not uniquely fixed by the scalar pair
spectrum or by the painting operation.

The typed JAX API is `ConstrainedBackboneAngularModel` and
`assemble_constrained_painted_angular_power` in `geppetto.painting_theory`.
It contracts without allocating a dense multipole-by-mass-by-mass matrix.
Tests compare against that explicit matrix, check both asymptotic limits,
positive semidefiniteness, empty populations, float32 cancellation, and
finite-difference/JAX concentration derivatives through the actual painter,
including the particle/halo and distinct-halo terms. The existing
`validate_painting_theory_operator.py` example also exercises the contraction
with a clearly labelled algebraic covariance fixture in all three branches.

`HistogramAngularGeometry` and `histogram_angular_assignment_moments` keep
large-population moment accumulation inside JAX as well. The host supplies
fixed native-row pairs and angular interpolation brackets; the JAX kernel
accumulates weights and weight products into per-group histograms, then
evaluates the Legendre recurrence. It does not build a pair-by-multipole
array. NGP groups can be handled analytically. The angular grid is a numerical
quadrature and must be converged for the requested multipole range; a host
histogram of already-painted weights is not a differentiable substitute.

`geppetto.io.build_angular_histogram_geometry` constructs those fixed
brackets from global native rows, halo row offsets, reference vectors, and
orientation weights. It includes only same-halo ordered pairs, retains
diagonal pairs, and rejects insufficient angular coverage. The geometry is
selected from stencil rows, not from nonzero painted values. The runnable
operator example differentiates the entire painter-to-histogram-to-covariance
path for all three concentration parameters and compares every component
against finite differences in all three assignment branches.

The population audit also passes all twelve radial nodes of the four
representative shells at NSIDE=2048, with sixteen fixed orientations per
resolved native-count mass node. JAX moments agree with the previous
fixed-value histogram calculation (rtol=1e-8, atol=2e-10). All three
concentration derivatives pass a two-step central Richardson check
(steps 1e-3 and 5e-4; rtol=3e-4, atol=5e-8), and their zero-mode derivatives
vanish to absolute tolerance 1e-9. These are numerical assignment checks,
not validation of the statistical backbone.

## Resolved Population Quadrature

The experimental model uses native HMF counts rather than a fitted HMF
extrapolated below the simulation threshold. At snapshot a_j and native
representative mass M_ij, the number-density integration weight is

\[
q_{ij}=\frac{N_{ij}}{L_{\rm box}^3},\qquad
f_H(a_j)=\sum_i q_{ij}\frac{M_{ij}}{\bar\rho},\qquad
S_H(a_j)=\sum_i q_{ij}\left(\frac{M_{ij}}{\bar\rho}\right)^2.
\]

No further mass-bin width belongs in these sums. The uncollapsed mass
fraction is 1-f_H; the measured halo fraction is not rescaled to one. Both
the halo self term and the same-protohalo subtraction use the same q_ij.

The API is `HaloCountQuadrature`, `halo_count_weights`, and
`read_pinocchio_halo_count_quadrature`. Native representative masses form
a union grid with zero weight outside each snapshot's native nodes. Linear
interpolation in scale factor preserves its count and representative-mass
moments exactly. Masses are Msun/h, box sides are comoving Mpc/h, and q has
units (Mpc/h)^-3. No unmeasured low-mass population is introduced.

This distinction matters because PINOCCHIO's `compute_mf` divides counts by
particle-rounded bin widths, while its first column contains mean halo mass
for bins with more than one object and a nominal centre otherwise.
Trapezoidal integration of dn/dlnM at these irregular locations does not
recover the native count or mass budget. For the four representative lower
shell boundaries, count-based resolved mass fractions are respectively
3.43%, 6.61%, 11.82%, and 17.47% larger than the previous trapezoid estimates.
Schemes 5 and 6 are kept unchanged as comparison baselines.

The count moment is exact for the input files. The first mass moment is
limited by singleton-bin centres and printed precision; S_H also neglects
the unavailable within-bin mass variance. Therefore this quadrature must not
be described as an exact sum of catalogue M^2. A direct PLC catalogue moment
is a useful independent diagnostic, not an amplitude to fit to the target
C_ell. Box-HMF and PLC-HMF fluctuations must also be distinguished from this
numerical quadrature difference.

## Updated Numerical Screen

With the count-based HMF, actual A and D moments, the complete constrained
covariance, and the finite-width nonlinear projection screen, the common
mask/constant-deprojection comparison gives:

| Segment | Scheme 5 RMS | Scheme 6 RMS | Constrained candidate RMS |
| ---: | ---: | ---: | ---: |
| 23 | 17.024% | 10.411% | 2.074% |
| 9 | 3.771% | 13.159% | 2.100% |
| 3 | 2.414% | 4.248% | 1.687% |
| 0 | 3.203% | 2.839% | 1.729% |

The statistic is unweighted RMS(measured/theory-1) over 99 mode-count-weighted
bands of width 20 within 20 <= ell < 2000. It is not chi-squared and does not
establish statistical preference without a realization covariance. No target
angular spectrum is input to the calculation of the candidate's components.

The stationary finite-width screen shifts bandpowers by at most 0.32%
relative to the nonlinear Limber projection. Doubling its line-of-sight
quadrature changes the pre-count-correction comparison by at most 2.2e-7 in
fractional bandpower. This is a thin-shell numerical check, not a full
unequal-time nonlinear projection. The protohalo-conditioning integral now
includes an interpolated point at the exact spherical Nyquist boundary;
the earlier endpoint check changed the subtraction by less than 7.3e-5.

The remaining scientific limitations are the uniform spherical protohalo
conditioning approximation, the rank-one allocation of stochastic covariance,
independent-orientation factorization, bin-representative masses, and the
finite-width/unequal-time approximation. They require component-level and
independent-realization validation before replacing production theory.
The mask response describes a pixel mask on an isotropic sky. If the halo
catalogue itself is restricted by centre position, objects outside the
catalogue boundary cannot contribute profiles across that boundary; this
additional selection does not commute with painting. That edge response is
not explicitly modeled by the current population forecast.

### Independent Component Warning

A preliminary comparison with the existing eight-seed full-sky component
audit at 0.174 < z < 0.211 exposes compensating errors in the candidate.
For 200 <= ell <= 512, its predicted/measured uncollapsed auto power is
1.18-1.22, cross power is 0.86-0.94, and halo auto power is 0.69-0.96.
The corresponding total ratio is much closer to one, 0.96-0.99.

These use the candidate's direct finite-width component projection, before
the exact-linear replacement used for its total. They are a warning screen,
not a final covariance-weighted rejection statistic. Nevertheless the
component discrepancies are appreciably larger than the preceding projection
quadrature errors. The eight manifests were checked on Leonardo: they use
the same hard finite-LOS profile, angular threshold, n_resolution=4, and
particle mass. Their halo mass fractions range from 0.1286 to 0.1317, close
to the count-based box prediction. The native HMF peak heights also agree
with a direct top-hat integral of the CAMB P(k,z) to better than 0.05% in
this shell. Neither a large missing-mass factor nor the old growth error
explains the component mismatch.

Therefore **the good four-panel total does not yet validate the analytical
backbone**. The next check must resolve the separately biased particle and
halo fields and their exclusion/cross statistics, not tune the total spectrum.
In particular, a mass-weighted Lagrangian bias expansion can give different
scale dependence to the components while enforcing zero summed Lagrangian
bias. A screen retaining the linear Lagrangian-bias operators from
[velocileptors](https://github.com/sfschen/velocileptors) improves some of
these component ratios but does not close the test: the uncollapsed ratio
falls from 1.03 near ell=210 to 0.74 near ell=506, while the cross ratio
rises from 1.02 to 1.10. This truncation is not a complete nonlinear bias or
conditional-selection model, and is **not adopted**. It illustrates why
passing a total-spectrum comparison is insufficient for this scientific aim.

The next implementation should supply independently checked component
statistics, including their large-scale biases, scale dependence, and
particle/halo exclusion covariance. The exact painting operator above can
consume those statistics without changing the profile, introducing unresolved
haloes, or compensating errors by fitting the final C_ell.

A second screen added quadratic local Lagrangian bias from the fitted HMF's
peak-background-split derivative, without fitting angular spectra. It still
fails component closure: near ell=500 the predicted/measured uncollapsed
power is about 0.62, cross power 1.01, and halo power 1.11, while the total
ratio is 0.98. The inferred second-order bias is uncorrected PBS, not a
Castro-calibrated second-order bias. This variant is also **not adopted**.

## Portable Node Projection

The experimental contraction no longer needs the research script's absolute
paths. `examples/project_painting_covariance.py` accepts explicit numerical
backbone and assignment tables. It does not read measured spectra, choose a
model using comparison residuals, or infer all component statistics from a
HMF. In particular, this is a **replay/projector API**, not yet an end-to-end
replacement for the production cosmological predictor. Generating the
conditioned LPT backbone is still a research calculation.

The JAX API in `geppetto.painting_theory` now also supplies:

- `ResolvedPaintingPopulation` and `resolved_painting_population`: native
  count weights to finite-population mass, bias, and self-power budgets.
- `StationaryLineOfSightRule` and `stationary_shell_average`: explicitly
  normalized sinc-squared quadrature of supplied three-dimensional tables.
- `project_constrained_painting_node`: radial projection and the complete
  covariance contraction, including both concentration-dependent moments.

For q_i = N_i / V_box, these helpers use

\[
f_i=q_iM_i/\bar\rho,\quad \beta_i=f_i b_i,\quad
S_i=q_i(M_i/\bar\rho)^2,\quad
N_U=(1-\sum_i f_i)m_p/\bar\rho.
\]

There is no additional mass-bin width and no rescaling of the resolved HMF
or bias integral to unity. The uncollapsed bias amplitude is
1-sum(beta_i), not necessarily its mass fraction 1-sum(f_i). The radial
factor is dchi*chi^2/V_sr^2, with V_sr=(chi_hi^3-chi_lo^3)/3. Smooth
correlations and the stochastic subtraction receive the native pixel
window; the discrete particle and native-pixel halo self terms do not.

### Replay Input Contract

Pass a JSON specification with this structure. Paths are relative to the
JSON file; numbers below illustrate one radial midpoint, not a calibrated
cosmology or precision recommendation.

```json
{
  "schema_version": 1,
  "grid": "grid.npz",
  "mean_density_msun_h_mpch3": 108603516000.0,
  "particle_mass_msun_h": 624599734600.0,
  "backbone_description": "State the supplied statistical approximation",
  "assignment_description": "State NSIDE, profile, concentration, resolution and orientation quadrature",
  "nodes": [
    {
      "segment_index": 23,
      "redshift": 0.2,
      "chi_lo_mpc_h": 500.0,
      "chi_hi_mpc_h": 600.0,
      "chi_mpc_h": 550.0,
      "dchi_weight_mpc_h": 100.0,
      "table": "node_00.npz"
    }
  ]
}
```

`grid.npz` contains integer `ell` and dimensionless `pixel_window` vectors.
It may also contain an independently calculated `shell_linear` array
(already pixel-windowed) and matching `segment_indices`. Nodes are grouped
in their first-appearance segment order. Their positive dchi weights must
sum to the shell width; duplicate nodes are rejected.

Each node NPZ contains:

| Key | Shape | Convention |
| --- | --- | --- |
| `mass_msun_h` | n_mass | Positive, increasing native representative masses |
| `number_density_weight_mpc_h3` | n_mass | Counts/volume, already integrated over mass bins |
| `linear_halo_bias` | n_mass | Eulerian bias in the same mass convention |
| `k_h_mpc` | n_k | Positive, increasing wave numbers |
| `coherent_power_mpc_h3` | n_k | Supplied nonnegative coherent backbone |
| `linear_power_mpc_h3` | n_k | Linear reference at this node's redshift |
| `same_protohalo_power_mpc_h3` | n_k | Same-protohalo subtraction B, including collapsed-particle self pairs |
| `lattice_correction_mpc_h3` | n_k | Discrete-particle correction to the continuous backbone |
| `response` | n_ell, n_mass | Actual global native-host A moment |
| `self_pair` | n_ell, n_mass | Actual global native-pixel D moment, not A squared |

The pair-derived constraint is (B+N_U-lattice_correction)/(N_U+sum(S_i)).
Invalid mass fractions, negative coherent power, non-finite arrays, invalid
assignment moments, and constraints outside [0,1] fail rather than being
clipped. Optional `response_jacobian` and `self_pair_jacobian` arrays must
both have shape (3,n_ell,n_mass), ordered as amplitude, mass slope, redshift
slope. They must be differentiated through the global painter normalization;
the projector propagates them through all covariance terms with JAX JVPs.

```bash
python examples/project_painting_covariance.py \
  --inputs outputs/two_regime_research/covariance_replay/inputs.json \
  --output outputs/two_regime_research/covariance_replay/prediction.npz

python examples/plot_painting_matched_audit.py \
  --projection outputs/two_regime_research/covariance_replay/prediction.npz \
  --mask-response outputs/two_regime_research/mask_response_deprojected.npz \
  --output-dir outputs/two_regime_research/covariance_replay/comparison
```

These commands replay locally generated audit inputs; they do not download
private PINOCCHIO files or create the backbone inputs from scratch. Defaults
are 256 pi-wide line-of-sight intervals with 16 Gauss nodes each. Refine these
numerical settings for a new cosmology/shell configuration. `--projection
limber` provides the corresponding direct-k check.

The single output NPZ contains five stationary component arrays ordered as
uncollapsed, twice particle/halo cross, distinct halo, same halo, and total.
If an independent linear projection is supplied, the difference from the
stationary linear reference is stored separately as
`linear_projection_correction` and added once to `shell_total`. It is not
silently allocated to separately exact-projected components. Optional
Jacobians retain the same component order. Input hashes, model descriptions,
mass fractions, and numerical settings are stored in `metadata_json`.

Outside the supplied k range the replay uses constant endpoint continuation;
the high-k endpoint also covers the omitted sinc-squared tail. Its reported
contributions are **not physical error bounds**. Neither this tail choice nor
the stationary approximation should be mistaken for an exact unequal-time
projection. Production defaults remain unchanged.
