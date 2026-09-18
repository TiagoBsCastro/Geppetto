# Schema-6 Cut-Sky Validation

Completed on Leonardo as job `56884283`: the original NGRID=2160 simulation,
29 shells spanning `0 <= z <= 2`, NSIDE=2048, and `2 <= ell <= 4096`.
The retained sky fraction is 0.3290201823. Measurements are unchanged from
schema 5; the fitted-HMF theory has been recomputed and convolved with the
actual mask, including the constant-deprojection convention.

## Figures

- [Summed map](figures/angular_power_summed.png)
  ([PDF](figures/angular_power_summed.pdf)).
- [Four representative shells](figures/angular_power_representative_shells.png)
  ([PDF](figures/angular_power_representative_shells.pdf)).
- [Both one-halo conventions, all 29 shells](figures/angular_power_all_shells_schema6.png)
  ([PDF](figures/angular_power_all_shells_schema6.pdf)).
- [Shell residuals](figures/angular_power_shell_residuals.png).
- [Normalized HMF and bias closure](figures/normalized_hmf_closure.png)
  ([PDF](figures/normalized_hmf_closure.pdf)).

The primary total is standard two-halo power plus compensated one-halo power
and particle shot noise. The alternative replaces only the one-halo term
with its ordinary, uncompensated form. The linear baseline is a diagnostic,
not an additional contribution to either total. The representative and
summed residual panels use the primary total. Their Gaussian guide is not
an ensemble covariance estimate.

## Model And Checks

The 30 native HMF files were fitted independently, without fitting measured
angular spectra. The model uses the manifest concentration relation,
continuum hard-truncated NFW, scale-dependent PINOCCHIO CAMB growth, and the
HEALPix pixel window. Exact projection was evaluated through `ell=512`;
the existing exact/finite-width/Limber matching checks passed at 1% tolerance.

| Check | Result |
| --- | ---: |
| Maximum count-weighted log fit residual | 0.03050 |
| Maximum mass-closure residual | 2.41e-10 |
| Maximum bias-closure residual | 2.23e-16 |
| Final integration relative change | 0.04823% |
| Maximum omitted low-tail white-power fraction bound | 3.83e-14 |

The final integration uses 1024 log-mass Gauss nodes from `0.01` to
`1e18 Msun/h`, with analytic tail completion. The formal low-peak-height
completion contains approximately 26-47% of the mass across the fitted
redshifts. This is an extrapolation assumption, not a physically calibrated
population; numerical convergence alone does not validate it.

For `20 <= ell < 200`, the mode-count-weighted ratio of summed measured power
to primary theory is 0.9818. This aggregate statistic does not establish
agreement shell by shell or at higher multipoles; the plots retain those
scale-dependent residuals.

See [the model documentation](../../docs/normalized_halo_model.md) for
equations, assumptions, API and reproduction commands, and the
[eight-seed full-sky comparison](../angular_power_fullsky_ensemble_schema6/README.md)
for an unmasked ensemble test. The NPZ spectra and quadrature are available
locally but ignored by Git, following repository policy.

The initial production job needed an in-place CPU-affinity repair; its
7h50m elapsed time is not a benchmark of the corrected submission script.
Diagnostic volume weights and the mass-integral column were corrected after
completion to match the current runner. The spectral archive was unchanged.
