# Verification of the frozen diagnostic report

Executed on 2026-09-17 from the GEPPETTO repository root. This record refers
to the frozen input hashes in `manifest.json`, not to an arbitrary future
catalogue or model configuration.

## Commands executed

```bash
/home/tcastro/miniforge3/envs/geppetto-dev/bin/python examples/diagnose_pinocchio_galaxies.py --config examples/pinocchio_galaxy_diagnostics_config.json
/home/tcastro/miniforge3/envs/geppetto-dev/bin/python examples/diagnose_pinocchio_galaxies.py --config examples/pinocchio_galaxy_diagnostics_config.json --plot-only
/usr/bin/python3 examples/render_pinocchio_galaxy_report.py --input-dir examples/pinocchio_galaxy_diagnostics
/home/tcastro/miniforge3/envs/geppetto-dev/bin/python -m pytest tests/test_galaxy_diagnostics.py -q
/home/tcastro/miniforge3/envs/geppetto-dev/bin/python -m pytest
/home/tcastro/miniforge3/envs/geppetto-dev/bin/ruff check .
git diff --check
```

## Results

| Verification | Result |
|---|---|
| Full real-data audit | Completed; 660,873 galaxies, 5,435,239 native halo occurrences |
| Frozen products | 16 input/source hashes verified before and after the audit |
| Catalogue SHA256 | `c6e6ee70a646e2b4cba34eb1593fd1e33a265d93e6072d3d03540c11f1c53c51` |
| Archived HOD table | `frozen_hod.npz` is byte-identical to the existing calibration cache |
| Replot reproducibility | SHA256 unchanged for all 10 PNGs and 16 CSVs after `--plot-only` |
| Focused diagnostic tests | 5 passed |
| Full test suite | 419 passed, 7 optional tests skipped; 173.26 seconds |
| Ruff | All checks passed |
| Diff whitespace check | Passed |
| Visual inspection | All 10 figures inspected; 12-page PDF rendered with Poppler; cover, plots and conclusion layout checked |
| Model changes | None; no catalogue generation or calibration called |

The focused tests cover half-open histogram boundaries, conditional
Poisson-binomial uncertainties, exact integration of the colour-dependent
piecewise-linear k-correction including clipped tails, Landy-Szalay pair
normalization against explicit pairs, and a plot-only path that cannot
invoke model computation.

Scientific tolerances and residuals are in `audit_summary.json`, `checks.csv`
and `report.md`. Passing these internal sampler checks does not establish
agreement with observed clustering, colours or cluster richness. The LF is
the independently specified observational calibration target; its mock
comparison is restricted to the documented completeness domain.
