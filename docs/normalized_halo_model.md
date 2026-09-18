# Schema-6 Normalized Halo Model

This is a theory-only benchmark. It does not change GEPPETTO painting or add
a two-halo map. Schema 5 used a compensated correction to the linear response;
schema 6 instead implements the standard mass-weighted halo-model response.
Old schema-4/5 archives and their APIs remain readable.

## Abundance And Bias

`geppetto.hmf.fit_normalized_hmf` fits each PINOCCHIO HMF independently, using
the measured abundance and halo counts, never the measured angular spectrum:

\[
F(\nu)=A\nu^q[1+(a\nu^2)^{-p}]e^{-a\nu^2/2},\qquad q=2p+s,
\]

where F is the mass fraction **per d ln(nu)**. A is analytic:

\[
A^{-1}=\frac12(2/a)^{q/2}
 [\Gamma(q/2)+2^{-p}\Gamma(q/2-p)].
\]

The constraints are `0.05 <= a <= 5`, `0 <= p <= 2`, and
`0.01 <= s <= delta_c-0.001`, with `delta_c=1.686`. They enforce integrable
tails and positive low-nu PBS bias; positivity is also required over the
complete numerical bias grid. Fits require eight populated bins and a
count-weighted log residual at most 0.05. No post-fit rescaling of the measured
bins is performed. Failed fits stop the calculation.

Peak heights and their mass slopes are computed from the top-hat variance of
the PINOCCHIO input P(k,z), in h/Mpc and (Mpc/h)^3. When available, the actual
scale-dependent CAMB series is used, not the background scalar-growth column.
Native HMF peak heights must agree to 1%. Shape parameters are interpolated
with PCHIP in scale factor; normalization is recomputed at projection nodes.

The PBS bias is `1 - dln(F)/dln(nu)/delta_c`. It is multiplied by
CCToolkit's `bias_correction_PBS`, pinned at
`ac16ab613eb93f795f562f928ad145597d981b2f`, then divided by the complete
mass-weighted bias integral. This is an **adapted, normalized Castro bias**,
not the unmodified Castro calibration. Castro's calibration uses virial
haloes; the numerical PINOCCHIO HMF and the manifest's profile mass definition
are retained and their possible mismatch remains a modeling uncertainty.

References: [Castro et al. halo-bias calibration](https://arxiv.org/abs/2409.01877),
[CCToolkit](https://github.com/TiagoBsCastro/CCToolkit), and the
[standard halo-model review](https://arxiv.org/abs/astro-ph/0206508).

## Mass Completion

The mass quadrature starts at `1e6..1e18 Msun/h` with 512 Gauss-Legendre
nodes in log mass. Analytic incomplete-gamma tails complete the integral to
zero/infinite peak height. The low-tail PBS integral is
`mass_low - F(nu_min)/delta_c`; the upper-tail sign is positive.
The Castro correction is held at the respective endpoint in each tail.
The upper-tail mass fraction must be below `1e-12`.

The low tail is a point-limit (`u=1`) completion, not an extrapolation of the
input P(k). Its omitted standard one-halo power is bounded by
`mass_low * Mmin / rho_mean`. The starting cutoff is not assumed converged:
a preliminary cutoff scan requires stable bias normalization and small
point-limit error, then the complete power grids are recomputed with Mmin/10
and twice the quadrature order. Both one-halo bandpowers and the squared
two-halo response must change by less than 0.1%. Mass closure must be within
`1e-5`; bias closure is enforced using the complete weighted integral.

**Numerical convergence is not evidence for the physical low-mass extrapolation.**
The finite input k range can leave a substantial analytic tail. Normalization
does not determine the missing HMF uniquely. This all-halo benchmark can
differ from the actual PINOCCHIO mixture of resolved haloes and uncollapsed
particles, even with a good fit to its resolved HMF.

## Power And Projection

The JAX API `normalized_halo_matter_power` returns three shared profile integrals:

\[
I(k,z)=\int d\ln M\,\frac{dn}{d\ln M}\frac{M}{\bar\rho}b_1u,
\quad P_{2h}=I^2P_{\rm lin},
\]
\[
P_{1h}^{\rm standard}=\int d\ln M\,\frac{dn}{d\ln M}
 \left(\frac{M}{\bar\rho}\right)^2 u^2,
\quad
P_{1h}^{\rm compensated}=\int d\ln M\,\frac{dn}{d\ln M}
 \left(\frac{M}{\bar\rho}\right)^2(u-W_L)^2.
\]

There is no additional linear baseline or Lagrangian subtraction in I.
The standard one-halo term has a white low-k limit; the compensated variant
has a k^4 limit and is the primary comparison. Both are reported, not mixed.
All terms use the manifest's concentration relation and hard-truncated NFW
profile. Concentration derivatives remain inside JAX; fits/bias are fixed.

The default is continuum NFW, an explicitly labeled standard halo-model
benchmark. `--match-ngp` applies the manifest's concentration-independent NGP
threshold (`u=1` for unresolved haloes). This is optional because it changes
the benchmark's profile convention. Supersampled/native branches use the
continuum transform, not a realization-specific pixel stencil.

Profile radii retain the painter's existing flat-LCDM density helper, including
when the supplied background tables describe evolving dark energy. This keeps
the theory profile consistent with the painted maps; it is not a new
general-dark-energy calibration of halo radii. The linear growth and distances
still come from the supplied PINOCCHIO tables.

Linear and two-halo spectra retain the existing exact/finite-width/Limber
projection, scale-dependent growth, pixel window and actual cut-sky mask
coupling. Both one-halo terms use Limber and share their profile evaluations.
Particle shot noise is kept separate from halo discreteness. In the full-sky
ensemble, its level is averaged per seed using each seed's uncollapsed count
and theoretical total mean. No two-halo contribution is counted twice.

## Running And Products

Install the optional `.[validation]` dependencies. The new entry point is
`examples/validate_normalized_halo_model.py`; `--help` lists its controls.
Use one `--hmf-glob` per equal-volume realization. The ensemble averages
abundances and bias-weighted abundances before squaring the response; it does
not average fitted parameters or reuse the original seed's HMF.

On Leonardo, with the existing measured caches:

```bash
sbatch --job-name=geppetto_s6_cut submit_theory.sh
SCHEMA6_CASE=ensemble sbatch --job-name=geppetto_s6_ens submit_theory.sh
```

The default cut-sky directory is `000/geppetto_reduced/angular_power_validation_schema6`.
The ensemble directory is `fullsky_z0493_ensemble/angular_power_fullsky_ensemble_schema6`.
`OUTDIR` can override either destination. Each job requests one full DCGP node.
`--preflight-only` stops after fitting, closure, convergence and diagnostic tables.

Keep `OMP_PROC_BIND=FALSE` and leave `OMP_PLACES` unset in this process-pool
workflow. OpenMP initialization can otherwise pin the Python parent to one
CPU; spawned processes inherit that actual affinity even after their OpenMP
environment variables are changed. The submission script now prevents this.
For an already-running allocation, `scripts/leonardo/restore_job_affinity.py`
can repair an explicitly selected, owned process tree from an overlapping
full-CPU Slurm step. It verifies both processes belong to the same job and
never expands beyond the invoking step's allowed CPUs.

- `angular_power_theory.npz`: explicit linear, two-halo, standard one-halo,
  compensated one-halo and particle-shot components; shell and summed theory;
  actual mask-coupled components for cut-sky data. Totals are derived, not
  duplicated. The full-sky cache lacks a measured sum, so none is fabricated.
- `normalized_hmf_audit.json`: fits, tail weights, closure, convergence,
  model conventions and input provenance.
- `normalized_hmf_quadrature.npz`: mass and bias-weighted integration tables.
- `angular_power_bias_fit.csv`: one row per HMF fit and seed.
- `figures/`: HMF/bias closure and all-shell comparisons; cut-sky plots also
  include the original summed, representative-shell and residual figures.

Only measurements are reused from the schema-5 archive. Their geometry is
checked against the manifest/mask and their archive hash is recorded. This
does not retrospectively provide missing map-content provenance in old caches.
Theory fingerprints include the new model and fitted tables. Old exact
two-halo checkpoints cannot be reused for schema 6.

Tests cover analytic tails, peak-height closure and growth, intermediate-z
mass/bias closure, low-k limits, concentration autodiff versus finite
differences, NGP zero derivatives, shapes, ensemble averaging and archive
compatibility. Run `ruff check .` and `python -m pytest`.
