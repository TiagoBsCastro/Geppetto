# Full-Sky Pair-Model Candidate

This directory contains comparisons of the fixed painting-matched candidate
with the eight existing full-sky NGRID2160 realizations. All 13 predictions
from Leonardo array 57018928 completed successfully (exit code 0:0), verified
on 2026-09-09. No measurements or maps were regenerated.

## Complete Comparison

The files in this directory's root cover all eight seeds and all 13 shells,
from redshift 0 to 0.49256256. They were generated locally from the downloaded
array predictions while the dependent comparison job 57019582 was still
pending. The local comparison script and scheme-5/6 theory archives have the
same SHA256 hashes as the staged Leonardo inputs.

`fullsky_pair_model_all_shells.{pdf,png}` compares the ensemble mean against
the fixed candidate and schemes 5 and 6. The NPZ contains 8 realizations,
13 shells, and 99 bands. Bandpowers, shell means, component spectra, input
hashes, and per-shell residuals are included in the accompanying CSV/JSON
files. The comparison checks shell alignment, finite spectra, and component
closure before plotting.

Using theoretical-mean normalization and equal-bin RMS over
`20 <= ell < 2000`, the candidate residuals span 0.91% to 2.05% for the 12
shells above redshift 0.03362193. The nearest shell has a 6.57% RMS residual
(6.26% after excluding reference seed 1386). These are descriptive residuals,
not a covariance-based acceptance test or validation of every component.

Reproduce the complete local comparison from the repository root:

```bash
python examples/compare_fullsky_pair_model.py \
    --prediction-glob 'outputs/fullsky_pair_model/predictions/shell_*.npz' \
    --ensemble-root outputs/fullsky_pair_model/inputs \
    --scheme5 examples/angular_power_validation_schema5 \
    --scheme6 examples/angular_power_fullsky_ensemble_schema6 \
    --output-dir examples/angular_power_fullsky_pair_model
```

## First Completed Shell

`preview_shell009/` is a **one-shell preview**, not the completed campaign.
It uses the actual array output `predictions/shell_009.npz` and all eight
seed observation caches. The redshift bounds are 0.10256817 to 0.13796899.
The original candidate's HMF, concentration parameters, corrected growth,
and full-sky linear reference are unchanged.

With theoretical-mean normalization, mode-count-weighted bins of width
20, and equal-bin RMS over `20 <= ell < 2000`, the fractional RMS residuals
are 1.60% (pair candidate), 14.91% (scheme 5), and 10.25% (scheme 6).
Excluding the original reference seed 1386 gives 1.63% for the candidate.
These are descriptive residuals, not covariance-based goodness-of-fit
probabilities. The displayed uncertainty is the ensemble mean's standard
error. Inspect `fullsky_pair_model_components.csv` separately; accurate
total power does not establish accurate component predictions.

Input hashes, bandpowers, shell means, component spectra and residuals are
saved beside the preview figure. The complete campaign setup and transfer
command are documented in
`scripts/leonardo/l3870_n2160_fullsky_ensemble/README.md`.
