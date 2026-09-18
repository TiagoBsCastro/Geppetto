# Conditional-Pair Research Comparison

This is a preliminary comparison, not an accepted production theory release.
It compares the same measured maps against schemas 5 and 6 and a candidate
using spherical initial-condition cutoff, full one-loop displacement
cumulants, and Gaussian-conditioned spherical protohalo pairs. The halo
self term uses the actual production adaptive painting weights.

All predictions use the same verified mask and constant-deprojection
operator. The metric is unweighted RMS(measured/predicted - 1) over 99
mode-count-weighted bands spanning 20 <= ell < 2000; it is not chi-squared.

| Segment | Scheme 5 RMS | Scheme 6 RMS | Candidate RMS |
| ---: | ---: | ---: | ---: |
| 23 | 17.024% | 10.411% | 2.407% |
| 9 | 3.771% | 13.159% | 1.930% |
| 3 | 2.414% | 4.248% | 1.689% |
| 0 | 3.203% | 2.839% | 1.728% |

The candidate passes this descriptive comparison. It still requires
independent physical and numerical validation; in particular, its scalar
background does not yet implement the full concentration-dependent
particle/halo covariance contraction. This is not a fit of concentration,
abundance, or a multipole-transition parameter to the painted spectra.

See [the equations and remaining checks](../../docs/painting_matched_halo_model.md).
Local reproduction inputs and commands are in
`outputs/two_regime_research/README.md`, which is ignored by git. The NPZ
here stores comparison spectra, not a duplicate of the input maps.
