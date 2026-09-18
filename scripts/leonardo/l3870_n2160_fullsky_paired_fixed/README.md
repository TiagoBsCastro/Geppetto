# L3870N2160 full-sky paired-and-fixed diagnostic

This diagnostic runs four opposite-phase pairs with fixed Fourier amplitudes.
It is intended only to determine whether the residual large-scale power
deficit in the ordinary eight-seed ensemble is realization variance or a
systematic map/theory mismatch. It does not alter GEPPETTO's painter or theory.

The geometry, shell boundaries, cosmology, and numerical resources match the
ordinary full-sky ensemble. `FixedIC` is enabled for every realization;
`PairedIC` is enabled for the `_b` member of each pair.
