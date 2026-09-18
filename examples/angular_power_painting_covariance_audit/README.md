# Experimental Painting-Matched Covariance Audit

These plots use the actual adaptive painting moments A and D, a native-count
HMF quadrature, and an explicit constrained particle/halo covariance. There is
no fitted multipole transition or rescaling of the measured halo abundance.

The total-spectrum RMS is lower than both schemes 5 and 6 in the four
representative shells over 20 <= ell < 2000. However, the separate low-redshift
components do not yet agree with the independent full-sky ensemble. Errors
partially cancel in their sum. This is **not an accepted production model**.

See [the derivation and limitations](../../docs/painting_matched_halo_model.md).
The CSV/JSON files state the metric and input provenance. The NPZ contains
the binned comparison and full-sky candidate components used by these figures.

The figures are generated with the raw-input
`examples/predict_painting_matched_power.py` workflow: three radial nodes,
16 orientations per native mass, the actual PINOCCHIO higher-order growth
ratios, and the production adaptive assignment. The linear reference is
theory-only; measured spectra enter only the comparison plotting command.
The total-spectrum RMS values are 2.069%, 2.106%, 1.686%, and 1.728%, in panel
order. This is not a claim of smaller error in every individual multipole bin.

`linear_projection_correction` is kept separate from the stationary
uncollapsed/cross/halo components; it does not turn those components into
independently exact-projected predictions. The public
`examples/project_painting_covariance.py` remains available to replay explicit
node tables without regenerating the particle-pair backbone.

The preliminary component CSV and population finite-difference audit JSON
retain the earlier replay's diagnostic checks. The low-redshift component
warning remains in force: adding the tested linear and quadratic
Lagrangian-bias terms, including the consistent-displacement resummation,
did not resolve it. Those variants are not adopted on the strength of a good
total spectrum.

The raw-input workflow was also run with all three concentration derivatives
enabled. Its `(4,3,5,4095)` component Jacobian is finite and sums consistently
to the total derivative. Enabling derivatives changes the predicted spectra
by at most 6.3e-14 fractionally. The focused population tests additionally
compare all three JAX derivatives with Richardson-extrapolated finite
differences, including global normalization and native-pixel aggregation.
