# Schema-6 Full-Sky Ensemble

Completed on Leonardo as job `56884299`: eight independent NGRID=2160 seeds,
13 full-sky shells spanning `0 <= z <= 0.49256256`, NSIDE=2048, and
`2 <= ell <= 4096`. The measured maps were reused; the theory was recomputed.

## Figures

- [Both one-halo conventions, all shells](figures/angular_power_all_shells_schema6.png)
  ([PDF](figures/angular_power_all_shells_schema6.pdf)). Shading is the standard
  error of the eight measured realizations for the compensated comparison.
- [Normalized HMF and bias closure](figures/normalized_hmf_closure.png)
  ([PDF](figures/normalized_hmf_closure.pdf)). Points are the native HMF bins;
  curves show the fitted abundance or normalized bias.
- [Primary comparison, with both mean-density conventions](standard-halo-model/angular_power_fullsky_ensemble_all_shells.png).
- [Uncompensated one-halo comparison](standard-halo-model-uncompensated/angular_power_fullsky_ensemble_all_shells.png).

Both totals contain standard two-halo power and the separate particle shot
noise. The primary total uses compensated one-halo power; the alternative
uses ordinary one-halo power. The blue and pink curves are not two different
two-halo prescriptions.

## Model And Checks

Each of the 112 native HMF files was fitted independently. Mass and
bias-weighted abundances were averaged before evaluating the ensemble
response and squaring it. The model uses the manifest concentration relation,
continuum hard-truncated NFW, scale-dependent PINOCCHIO CAMB growth, and the
HEALPix pixel window. No cut-sky mask correction is involved.

| Check | Result |
| --- | ---: |
| Maximum count-weighted log fit residual | 0.03097 |
| Maximum mass-closure residual | 9.92e-11 |
| Maximum bias-closure residual | 3.34e-16 |
| Final integration relative change | 0.02789% |
| Maximum omitted low-tail white-power fraction bound | 1.49e-15 |

The final integration uses 1024 log-mass Gauss nodes from `0.01` to
`1e18 Msun/h`, with analytic tail completion. Approximately 46-47% of the
mass is in that formal low-peak-height completion because the finite input
power spectrum makes the small-mass variance plateau. Numerical convergence
does not validate this extrapolation physically.

For `20 <= ell < 200`, the mode-count-weighted measured-to-primary-theory
ratios are 0.806 and 0.944 for the first two shells. The remaining shells span
0.982-1.031. Normalization therefore does not eliminate every discrepancy.

See [the model documentation](../../docs/normalized_halo_model.md) for
equations, assumptions, API and reproduction commands. The NPZ spectra and
quadrature are present locally but ignored by Git, following repository
policy; the CSV/JSON diagnostics record fit and input provenance.

The initial production job needed an in-place CPU-affinity repair. The
submission script now prevents the inherited single-CPU binding. Diagnostic
volume weights and the mass-integral column were corrected after completion;
the spectral archive was not modified.
