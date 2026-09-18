# Painting-Matched Model Audit

These are experimental comparisons, not a new accepted production scheme.
The candidate uses the actual globally normalized adaptive pixel weights and
a finite-HMF Lagrangian particle-pair closure for the PINOCCHIO backbone.
There is no fit of a transition, amplitude, or damping parameter to these maps.

All three theories have been forward-coupled with the same actual mask and
constant-deprojection response. The old theory reference skipped initialization
of its deprojection template; that bug is fixed separately. Correcting it changes
the compared bandpowers by less than 0.01%, and does not fix the model mismatch.

| Representative segment | Scheme 5 RMS | Scheme 6 RMS | Candidate RMS |
| ---: | ---: | ---: | ---: |
| 23 | 17.024% | 10.411% | 4.757% |
| 9 | 3.771% | 13.159% | 7.108% |
| 3 | 2.414% | 4.248% | 1.769% |
| 0 | 3.203% | 2.839% | 1.719% |

RMS means the unweighted RMS of measured/predicted minus one across 99
mode-count-weighted bands covering 20<=ell<2000. It is not a covariance-weighted
statistical test. The candidate fails the all-four-panel criterion and remains
experimental. The main unresolved approximation is the collapse-conditioned
particle/halo correlation model, not the discrete NFW painting operator.

See [the derivation and limitations](../../docs/painting_matched_halo_model.md).
The local reproduction bundle is `outputs/two_regime_research/README.md`.

Files:

- `painting_matched_representative_shells.png` and `.pdf`: common-estimator plots.
- `painting_matched_audit.csv` and `.json`: descriptive RMS and provenance.
- `painting_matched_audit.npz`: bandpowers and additive full-sky candidate terms.
