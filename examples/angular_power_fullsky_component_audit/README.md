# Full-sky component two-halo audit

This directory contains the eight-seed decomposition audit for the
`L3870N2160` full-sky ensemble over `0 < z < 0.49256256`. The input component
spectra were measured at `NSIDE=2048`, and the theory is the schema-5
scale-dependent CAMB calculation.

The primary `20 <= ell <= 199` result is:

- the measured component-coherent power remains within about 4 percent of the
  linear shell prediction for `z >= 0.068`;
- the corrected two-halo response rises from `1.01` times linear at `z=0.47`
  to `1.21` at `z=0.085`, `1.32` at `z=0.051`, and `1.50` in the nearest
  shell;
- the measured `U-H` correlation falls from about `0.90` at `z=0.47` to
  `0.61`, `0.43`, and `0.21` in those three low-redshift shells;
- the corrected response exceeds the measured coherent power by about 24
  percent at `z=0.085`, 49 percent at `z=0.051`, and 142 percent in the
  nearest shell.

The final percentage differs slightly from ratios formed before subtracting
uncollapsed-particle shot noise. The nearest shell is also underdense in the
eight-realization mean and has the smallest volume, so it should not be used
alone to calibrate a replacement model. The discrepancy is already clear in
the next two shells, where those limitations are substantially weaker.

`HH_parallel = UH^2 / (UU - N_U)` is a regression onto the measured
uncollapsed field. It establishes that the current total response is too
large, but it does not uniquely identify whether the cause is the assumed
Lagrangian-window subtraction, imperfect field coherence, scale-dependent
bias, or halo exclusion. A definitive transfer calibration requires a map of
the corresponding initial linear density mode.
