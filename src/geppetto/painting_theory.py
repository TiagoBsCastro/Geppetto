"""Analytical angular-power operators for the actual discrete painting rule.

These functions describe halo-profile assignment, not the statistics of the
PINOCCHIO backbone. Particle/halo cross-correlations and distinct-halo
correlations are explicit inputs. No fitted multipole transition is used.
HEALPix indexing, stencil construction, and covariance estimation stay outside
these differentiable kernels.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import lax

from geppetto.types import Array


class AngularAssignmentMoments(NamedTuple):
    """Dimensionless assignment moments for ell=0,...,lmax.

    ``response`` is sum_p w_p P_ell(n_reference dot n_p). ``self_pair`` is
    sum_pq w_p w_q P_ell(n_p dot n_q). The latter, not response squared, is
    the same-halo angular-power transfer. They retain all native-pixel
    anisotropy. Both equal one at ell=0 for a globally normalized assignment.
    Population calculations may prepend a group dimension.
    """

    response: Array
    self_pair: Array


class HistogramAngularGeometry(NamedTuple):
    """Fixed host-built geometry for grouped angular-moment quadrature.

    All bin indices address a flattened (n_group,n_angle) histogram. A lower
    bin and its successor bracket the sample angle; fractions lie in [0,1].
    ``cosine_grid`` has shape (n_angle,), with the first entry exactly one.
    ``analytic_ngp_weight`` has shape (n_group,) and supplies the orientation
    fraction handled analytically as native-host NGP. It fixes n_group.

    The three response arrays have shape (n_native_row,). The five pair
    arrays have shape (n_pair,), and their row indices address the supplied
    native assignment weights. Pair rows must belong to the SAME halo;
    include ordered pairs, including diagonal pairs. ``*_average_weight``
    supplies the halo/orientation averaging weight, not a second factor of
    profile mass. Each input row weight is already normalized per halo.

    All geometry is concentration independent. Construct HEALPix indices,
    child-to-native aggregation, and interpolation brackets outside JAX.
    Native rows must cover global halo support, not a clipped compact mask.
    A zero pair_average_weight can pad a valid pair index. The host must
    validate index bounds and interpolation fractions before JIT execution.
    """

    cosine_grid: Array
    analytic_ngp_weight: Array
    response_lower_bin: Array
    response_bin_fraction: Array
    response_average_weight: Array
    pair_row_a: Array
    pair_row_b: Array
    pair_lower_bin: Array
    pair_bin_fraction: Array
    pair_average_weight: Array


class HaloBackboneAngularSpectra(NamedTuple):
    """Unpainted component spectra, all normalized by the *total* shell mean.

    All C_ell values are dimensionless. Halo-bin maps contain the halo masses
    (or particle-equivalent counts) at reference positions, not number-density
    contrasts normalized separately by each bin's mean.

    ``uncollapsed_auto`` has shape (n_ell,); ``uncollapsed_halo_cross`` has
    shape (n_ell,n_bin), without a factor of two. ``halo_auto_cross`` has
    shape (n_ell,n_bin,n_bin) and includes the halo same-object diagonal.
    ``halo_self_pair`` has shape (n_bin,) and gives that white diagonal's
    amplitude. It is subtracted once before applying distinct-halo transfers.
    Particle discreteness and particle/halo cross-covariance are already in
    the component inputs: no independent extra shot noise is added.
    """

    uncollapsed_auto: Array
    uncollapsed_halo_cross: Array
    halo_auto_cross: Array
    halo_self_pair: Array


class PaintedAngularSpectra(NamedTuple):
    """Additive dimensionless C_ell components and their total (n_ell,).

    ``particle_halo_cross`` includes the factor of two. Distinct-halo and
    cross components may be negative; only their complete covariance is
    constrained to yield a non-negative total. ``one_halo`` is the actual
    same-object term, not a compensated phenomenological substitute.
    """

    uncollapsed: Array
    particle_halo_cross: Array
    distinct_halo: Array
    one_halo: Array
    total: Array


class ResolvedPaintingPopulation(NamedTuple):
    """Native-count mass weights, without extrapolating the resolved HMF.

    ``halo_mass_fraction`` and ``halo_bias_weight`` have shape (n_mass,)
    and are dimensionless. ``halo_self_power`` has the same shape and units
    (Mpc/h)^3. The uncollapsed fraction and self power are scalars. All
    fields use the theoretical TOTAL mean density, not the component means.
    """

    halo_mass_fraction: Array
    halo_bias_weight: Array
    halo_self_power: Array
    uncollapsed_mass_fraction: Array
    uncollapsed_self_power: Array


class StationaryLineOfSightRule(NamedTuple):
    """Fixed quadrature for (2/pi) integral_0^infinity sinc(t)^2 dt = 1.

    ``nodes`` are positive dimensionless t = k_parallel * shell_width / 2.
    ``weights`` already include (2/pi)*sinc(t)^2*dt, with sinc(t)=sin(t)/t.
    The scalar ``tail_weight`` is the unintegrated probability beyond the
    last node. All weights must be nonnegative and sum to one including
    the tail. Construct and validate this numerical rule outside JAX.
    """

    nodes: Array
    weights: Array
    tail_weight: Array


class StationaryShellAverage(NamedTuple):
    """A stationary projection and its explicit endpoint continuations.

    Arrays have shape (n_ell,) and retain the input table's units. The low
    continuation is the contribution below the table's smallest k. The high
    continuation includes the unresolved line-of-sight tail. These are
    contributions under the specified constant continuations, NOT rigorous
    bounds on the unknown physical power outside the table.
    """

    value: Array
    low_k_continuation: Array
    high_k_continuation: Array


class ConstrainedBackboneAngularModel(NamedTuple):
    """Rank-one mass-constrained covariance for one radial quadrature node.

    ``coherent_cl`` and ``constraint_window`` have shape (n_ell,).
    ``halo_mass_weighted_bias`` and ``halo_self_cl`` have shape (n_bin,).
    ``uncollapsed_self_cl`` is scalar. Spectra are dimensionless C_ell;
    self amplitudes include the node's radial integration weight. Bias
    amplitudes include only the finite resolved HMF mass weights, not that
    radial weight. The uncollapsed bias amplitude is one minus their sum.

    For N=(N_U,S_1,...), the residual covariance is
    diag(N)-constraint_window*N*N.T/sum(N). Nonnegative N, coherent_cl >= 0,
    and constraint_window <= 1 guarantee positive semidefiniteness. The
    intended physical window goes from one on large scales to zero on small
    scales; its derivation is an explicit host-side model input. No clipping
    or transition fitting is performed here. Window=1 removes the total-mass
    stochastic mode; this alone does not establish a k^4 momentum constraint.

    This is a statistical closure, not an identity for PINOCCHIO components.
    It generalizes the mass-weighted rank-one form of Schmidt (2016), Eq. 33,
    arXiv:1511.02231, to include uncollapsed particle counts. It is NOT that
    paper's mass-dependent interpolation or a full nonlinear bias model.
    """

    coherent_cl: Array
    halo_mass_weighted_bias: Array
    uncollapsed_self_cl: Array
    halo_self_cl: Array
    constraint_window: Array


def resolved_painting_population(
    mass_msun_h: Array,
    number_density_weight_mpc_h3: Array,
    linear_halo_bias: Array,
    mean_density_msun_h_mpch3: Array,
    particle_mass_msun_h: Array,
) -> ResolvedPaintingPopulation:
    """Form the mass budget of the objects actually replaced by painting.

    Mass and particle mass are Msun/h; mean density is (Msun/h)/(Mpc/h)^3.
    The three vector inputs have shape (n_mass,). Number-density weights
    are counts / box_volume, in (Mpc/h)^-3, such as ``halo_count_weights``
    returns. They ALREADY contain the mass quadrature: never multiply by a
    further dM or dlnM. Halo bias is dimensionless Eulerian linear bias.

    f_i = q_i*M_i/rho, S_i = q_i*(M_i/rho)^2, beta_i = f_i*b_i.
    Uncollapsed particles retain fraction 1-sum(f_i) and self power
    (1-sum(f_i))*m_particle/rho. Neither halo masses nor halo biases are
    rescaled to make their resolved-only integrals equal one. The missing
    bias amplitude 1-sum(beta_i) belongs to the uncollapsed field.

    The host must validate positive masses/density, nonnegative q, finite
    bias, and sum(f_i) <= 1. No clipping hides an inconsistent mass budget.
    Empty resolved populations are supported. This pure JAX operation is
    differentiable in its numeric inputs; catalogue selection is fixed.
    """

    mass, number, bias, density, particle = map(jnp.asarray, (
        mass_msun_h, number_density_weight_mpc_h3, linear_halo_bias,
        mean_density_msun_h_mpch3, particle_mass_msun_h,
    ))
    if mass.ndim != 1 or number.shape != mass.shape or bias.shape != mass.shape:
        raise ValueError("mass, number-density weights, and bias must have shape (n_mass,)")
    if density.ndim != 0 or particle.ndim != 0:
        raise ValueError("mean density and particle mass must be scalars")
    volume = mass / density
    fraction = number * volume
    uncollapsed = 1 - jnp.sum(fraction)
    return ResolvedPaintingPopulation(
        fraction, fraction * bias, number * volume**2,
        uncollapsed, uncollapsed * particle / density,
    )


def stationary_shell_average(
    transverse_k_h_mpc: Array,
    table_k_h_mpc: Array,
    table_values: Array,
    shell_width_mpc_h: Array,
    rule: StationaryLineOfSightRule,
    *,
    low_k_value: Array,
    high_k_value: Array,
) -> StationaryShellAverage:
    """Average F(sqrt(k_perp^2+k_parallel^2)) through a stationary top-hat.

    Returns (2/pi)*integral dt sinc(t)^2 F(sqrt(k_perp^2+(2t/width)^2)).
    Wave numbers are h/Mpc and width is comoving Mpc/h. Values can be power
    in (Mpc/h)^3 or a dimensionless covariance window. This is a stationary
    thin-shell approximation, NOT an exact spherical/unequal-time solver.

    All arrays are one-dimensional except scalar width and continuation
    values. The table must have increasing positive k; the host validates
    its values and the quadrature. Piecewise-linear interpolation and both
    explicit endpoint continuations remain inside JAX. The omitted tail is
    assigned ``high_k_value``; its contribution is reported, not hidden.

    Forward working storage is O(n_los+n_ell), not n_los*n_ell. Differentiable
    in table values, width and endpoint values at fixed quadrature nodes;
    derivatives of interpolation are piecewise smooth. Prefer parameter JVPs
    to full table Jacobians when the quadrature is large.
    """

    transverse, wave, values, width, lower, upper = map(jnp.asarray, (
        transverse_k_h_mpc, table_k_h_mpc, table_values, shell_width_mpc_h,
        low_k_value, high_k_value,
    ))
    nodes, weights, tail = map(jnp.asarray, rule)
    if transverse.ndim != 1 or wave.ndim != 1 or values.shape != wave.shape or wave.size < 2:
        raise ValueError("transverse k must be a vector and the table needs >= 2 matching nodes")
    if nodes.ndim != 1 or weights.shape != nodes.shape or nodes.size == 0:
        raise ValueError("line-of-sight nodes and weights must be matching nonempty vectors")
    if any(value.ndim != 0 for value in (width, lower, upper, tail)):
        raise ValueError("width, endpoint values, and tail weight must be scalars")

    def average_one(k_perp):
        k = jnp.sqrt(k_perp**2 + (2 * nodes / width)**2)
        value = jnp.interp(k, wave, values, left=lower, right=upper)
        high = upper * (jnp.sum(jnp.where(k > wave[-1], weights, 0)) + tail)
        low = lower * jnp.sum(jnp.where(k < wave[0], weights, 0))
        return StationaryShellAverage(jnp.sum(value * weights) + tail * upper, low, high)

    return lax.map(average_one, transverse)


def project_constrained_painting_node(
    population: ResolvedPaintingPopulation,
    coherent_power_mpc_h3: Array,
    constraint_window: Array,
    radial_weight_mpch_minus3: Array,
    pixel_window: Array,
    response: Array,
    self_pair: Array,
) -> PaintedAngularSpectra:
    """Project and paint one radial node of the experimental covariance.

    For a count shell, radial_weight = dchi*chi^2/V_sr^2, where
    V_sr=(chi_hi^3-chi_lo^3)/3. Distances are comoving Mpc/h. The coherent
    power has units (Mpc/h)^3 and may be evaluated at Limber k or averaged
    with ``stationary_shell_average`` first. Other spectral inputs are
    dimensionless vectors (n_ell,); assignment moments are (n_ell,n_mass).

    The native pixel window multiplies smooth correlations, including the
    constrained subtraction, but NOT the discrete halo/particle self terms.
    Supersampled children must already be aggregated to native pixels in A,D.
    No additional linear spectrum, halo self noise or particle noise is added.
    Sum returned nodes before applying a survey mask/estimator response.

    This is the rank-one statistical approximation documented by
    ``ConstrainedBackboneAngularModel``, not an exact PINOCCHIO covariance.
    Concentration derivatives flow through BOTH moments and all cross terms.
    """

    coherent, window, weight, pixels = map(jnp.asarray, (
        coherent_power_mpc_h3, constraint_window, radial_weight_mpch_minus3, pixel_window,
    ))
    if coherent.ndim != 1 or window.shape != coherent.shape or pixels.shape != coherent.shape:
        raise ValueError("coherent power, constraint, and pixel window must have shape (n_ell,)")
    if weight.ndim != 0:
        raise ValueError("radial weight must be scalar")
    backbone = ConstrainedBackboneAngularModel(
        weight * pixels**2 * coherent, population.halo_bias_weight,
        weight * population.uncollapsed_self_power, weight * population.halo_self_power,
        pixels**2 * window,
    )
    return assemble_constrained_painted_angular_power(backbone, response, self_pair)


def _weighted_legendre_sum(cosine: Array, weight: Array, lmax: int) -> Array:
    p0 = jnp.ones_like(cosine)
    s0 = jnp.sum(weight)
    if lmax == 0:
        return s0[None]
    s1 = jnp.sum(weight * cosine)

    def step(previous: tuple[Array, Array], order: Array):
        lower, current = previous
        following = ((2 * order - 1) * cosine * current - (order - 1) * lower) / order
        return (current, following), jnp.sum(weight * following)

    _, higher = lax.scan(step, (p0, cosine), jnp.arange(2, lmax + 1))
    return jnp.concatenate((s0[None], s1[None], higher))


def angular_assignment_moments(
    pixel_unit_vectors: Array,
    assignment_weights: Array,
    reference_unit_vector: Array,
    *,
    lmax: int,
    pair_chunk_size: int = 64,
) -> AngularAssignmentMoments:
    """Compute exact single-halo moments of a finite native-pixel assignment.

    Vectors have shapes (n_pixel,3) and (3,) and must be unit normalized.
    Non-negative, dimensionless weights have shape (n_pixel,). For the global
    profile transfer, pass the painter's globally normalized weights BEFORE
    compact-map filtering. Do not renormalize clipped weights here. Children
    from supersampling must first be aggregated onto their native pixels.

    The reference can be the true halo direction or its NGP host-pixel centre,
    but it must match the reference used to define the halo backbone spectra.
    The NGP-host convention gives response=self_pair=1 for an unresolved halo.

    The spherical-harmonic addition theorem gives its exact unmasked self
    power: C_ell = self_pair / (4*pi) for a unit-mass delta-pixel distribution.
    No HEALPix pixel-window factor belongs on top of these discrete moments.
    A harmonic estimator's quadrature/iteration effects need separate testing.

    Differentiable in weights (and hence concentration when fed by the JAX
    painter). Geometry, lmax, and pair_chunk_size are fixed. The double sum is
    O(n_pixel^2*lmax), with O(pair_chunk_size*n_pixel) forward working storage,
    not a full pixel-pair-by-multipole allocation. Reverse-mode autodiff may
    retain recurrence history; concentration JVPs avoid that storage cost.
    Zero weights can pad profiles. Use float64 for high ell and small angles.
    """

    if lmax < 0 or pair_chunk_size < 1:
        raise ValueError("lmax must be non-negative and pair_chunk_size positive")
    vectors = jnp.asarray(pixel_unit_vectors)
    weights = jnp.asarray(assignment_weights)
    reference = jnp.asarray(reference_unit_vector)
    if vectors.ndim != 2 or vectors.shape[1] != 3 or weights.shape != (vectors.shape[0],):
        raise ValueError("pixel vectors and weights must have shapes (n_pixel,3) and (n_pixel,)")
    if reference.shape != (3,):
        raise ValueError("reference_unit_vector must have shape (3,)")
    if weights.size == 0:
        zero = jnp.zeros((lmax + 1,), dtype=weights.dtype)
        return AngularAssignmentMoments(zero, zero)
    response = _weighted_legendre_sum(jnp.clip(vectors @ reference, -1, 1), weights, lmax)
    size = min(pair_chunk_size, weights.size)
    padding = (-weights.size) % size
    blocks = jnp.pad(vectors, ((0, padding), (0, 0))).reshape(-1, size, 3)
    block_weights = jnp.pad(weights, (0, padding)).reshape(-1, size)

    def add_block(total: Array, block: tuple[Array, Array]):
        positions, mass = block
        cosine = jnp.clip(positions @ vectors.T, -1, 1)
        pair_weight = mass[:, None] * weights[None, :]
        return total + _weighted_legendre_sum(cosine, pair_weight, lmax), None

    self_pair, _ = lax.scan(add_block, jnp.zeros_like(response), (blocks, block_weights))
    return AngularAssignmentMoments(response, self_pair)


def histogram_angular_assignment_moments(
    native_assignment_weights: Array,
    geometry: HistogramAngularGeometry,
    *,
    lmax: int,
    pair_chunk_size: int = 65536,
) -> AngularAssignmentMoments:
    """Differentiate grouped A and D through an angular histogram quadrature.

    Weights are dimensionless, normalized per halo, and have shape
    (n_native_row,). Output moments have shape (n_group,lmax+1). Geometry
    defines their concentration-independent halo/orientation averaging.
    All dependence on weights, including products in the self term, remains
    in JAX. In particular, no NumPy histogram of painted weights is allowed
    on a path intended for concentration differentiation.

    Linear interpolation between fixed angular nodes approximates Legendre
    polynomials at the actual sample angles. Refine the angle grid and test
    against :func:`angular_assignment_moments` for the requested lmax; this
    is not an exact addition-theorem evaluation away from grid nodes. Check
    D >= A^2 to the required quadrature accuracy. The ell=0 normalization is
    retained by the interpolation, and native-host NGP has A=D=1 exactly.

    Pair accumulation is statically chunked and never allocates an
    n_pair*n_ell array. Forward storage is O(n_group*n_angle+pair_chunk_size)
    beyond the supplied geometry and output. Prefer JVPs for a small number
    of concentration parameters; reverse mode can retain scan history.
    """

    if lmax < 0 or pair_chunk_size < 1:
        raise ValueError("lmax must be non-negative and pair_chunk_size positive")
    weights = jnp.asarray(native_assignment_weights)
    values = HistogramAngularGeometry(*(jnp.asarray(value) for value in geometry))
    if weights.ndim != 1 or values.cosine_grid.ndim != 1 or values.cosine_grid.size < 2:
        raise ValueError("native weights and cosine grid must be vectors, with at least two angles")
    if values.analytic_ngp_weight.ndim != 1:
        raise ValueError("analytic_ngp_weight must be a group vector")
    if any(x.shape != weights.shape for x in (
        values.response_lower_bin, values.response_bin_fraction, values.response_average_weight,
    )):
        raise ValueError("response geometry must match the native row weights")
    if values.pair_row_a.ndim != 1 or any(x.shape != values.pair_row_a.shape for x in (
        values.pair_row_b, values.pair_lower_bin, values.pair_bin_fraction, values.pair_average_weight,
    )):
        raise ValueError("pair geometry arrays must be matching vectors")
    n_group, n_angle = values.analytic_ngp_weight.size, values.cosine_grid.size
    histogram_dtype = jnp.result_type(
        weights, values.analytic_ngp_weight, values.response_bin_fraction,
        values.response_average_weight, values.pair_bin_fraction, values.pair_average_weight,
    )
    initial = jnp.zeros((n_group, n_angle), dtype=histogram_dtype)
    initial = initial.at[:, 0].set(values.analytic_ngp_weight)

    def deposit(histogram: Array, lower: Array, fraction: Array, contribution: Array):
        histogram = histogram.at[lower].add((1-fraction)*contribution)
        return histogram.at[lower+1].add(fraction*contribution)

    response_histogram = deposit(
        initial.ravel(), values.response_lower_bin, values.response_bin_fraction,
        values.response_average_weight * weights,
    ).reshape(n_group, n_angle)
    self_histogram = initial.ravel()
    if values.pair_row_a.size:
        size = min(pair_chunk_size, values.pair_row_a.size)
        padding = (-values.pair_row_a.size) % size
        blocks = tuple(jnp.pad(value, (0, padding)).reshape(-1, size) for value in (
            values.pair_row_a, values.pair_row_b, values.pair_lower_bin,
            values.pair_bin_fraction, values.pair_average_weight,
        ))

        def accumulate(histogram, block):
            row_a, row_b, lower, fraction, average = block
            contribution = average * weights[row_a] * weights[row_b]
            return deposit(histogram, lower, fraction, contribution), None

        self_histogram, _ = lax.scan(accumulate, self_histogram, blocks)

    def transform(histogram):
        return lax.map(lambda row: _weighted_legendre_sum(values.cosine_grid, row, lmax), histogram)

    return AngularAssignmentMoments(
        transform(response_histogram), transform(self_histogram.reshape(n_group, n_angle)),
    )


def catalogue_self_pair_amplitude(
    halo_particle_counts: Array,
    mean_total_counts_per_pixel: Array,
    pixel_area_sr: Array,
) -> Array:
    """White same-halo C_ell amplitude for a full-sky catalogue or mass bin.

    Counts are dimensionless M_h/m_particle; the mean is the total shell's
    counts per pixel and area is in steradians. Sum over all global haloes.
    A cut-sky count cannot be substituted without a footprint/selection model.
    Does not include a pixel window, profile transfer, or Poisson assumption.
    """

    return (
        (jnp.asarray(pixel_area_sr) / mean_total_counts_per_pixel) ** 2
        * jnp.sum(jnp.asarray(halo_particle_counts) ** 2) / (4 * jnp.pi)
    )


def assemble_painted_angular_power(
    backbone: HaloBackboneAngularSpectra,
    response: Array,
    self_pair: Array,
) -> PaintedAngularSpectra:
    """Apply population-averaged assignment moments to backbone covariances.

    Inputs and output C_ell are dimensionless, before any survey mask.
    ``response`` and ``self_pair`` have shape (n_ell,n_bin). Average response
    using the field's mass weighting and self_pair using its mass-SQUARED
    weighting; the binning in mass/redshift must resolve those variations.

    The same-halo term is exact given its averaged moments. Factorization of
    distinct-halo and particle/halo terms assumes independent, statistically
    isotropic profile orientations at fixed bin, independent of environment.
    Finite pixels and coarse bins can violate this; test against actual
    painting rather than claiming the factorization is a realization identity.

    This operator does not infer backbone correlations from a HMF. Linear
    bias alone supplies only their large-scale deterministic limit. It also
    does not add a separate P_linear term to an already complete covariance.
    All profile dependence stays differentiable through the moment arrays.
    """

    response, self_pair = jnp.asarray(response), jnp.asarray(self_pair)
    if response.ndim != 2 or self_pair.shape != response.shape:
        raise ValueError("response and self_pair must have shape (n_ell,n_bin)")
    n_ell, n_bin = response.shape
    if (
        backbone.uncollapsed_auto.shape != (n_ell,)
        or backbone.uncollapsed_halo_cross.shape != (n_ell, n_bin)
        or backbone.halo_auto_cross.shape != (n_ell, n_bin, n_bin)
        or backbone.halo_self_pair.shape != (n_bin,)
    ):
        raise ValueError("backbone covariance dimensions do not match assignment moments")
    self_diagonal = jnp.diag(backbone.halo_self_pair)
    distinct = backbone.halo_auto_cross - self_diagonal[None, :, :]
    cross = 2 * jnp.sum(response * backbone.uncollapsed_halo_cross, axis=-1)
    halo_cross = jnp.einsum("li,lij,lj->l", response, distinct, response)
    one_halo = jnp.sum(self_pair * backbone.halo_self_pair, axis=-1)
    uncollapsed = backbone.uncollapsed_auto
    return PaintedAngularSpectra(
        uncollapsed, cross, halo_cross, one_halo,
        uncollapsed + cross + halo_cross + one_halo,
    )


def assemble_constrained_painted_angular_power(
    backbone: ConstrainedBackboneAngularModel,
    response: Array,
    self_pair: Array,
) -> PaintedAngularSpectra:
    """Apply actual assignment moments to a constrained component covariance.

    Inputs/output are dimensionless angular spectra. Both moments have shape
    (n_ell,n_bin). The assumptions about binning, orientation independence,
    and total-shell normalization are the same as in
    :func:`assemble_painted_angular_power`. Concentration dependence must
    enter BOTH response and self_pair; every cross term remains inside JAX.

    This contracts the rank-one covariance without constructing an
    (n_ell,n_bin,n_bin) array. Working storage scales as n_ell*n_bin. Each
    radial node can be computed and accumulated separately. The self term
    uses D, not A^2, including orientation scatter and pixel anisotropy.
    For valid normalized assignments D >= A^2. Together with the backbone
    constraints this guarantees nonnegative total power, before any separate
    radial-projection approximation or mask estimator is applied.

    For A=D=1 the result is coherent_cl + sum(N)*(1-window). At window=1 it
    recovers coherent_cl, independently of the resolved mass fraction. At
    window=0 and negligible coherent_cl it gives N_U + sum(S_i*D_i), the
    particle plus actual one-halo self terms, with no extra noise added.
    """

    response, self_pair = jnp.asarray(response), jnp.asarray(self_pair)
    coherent, bias, particle, halo_self, window = map(jnp.asarray, backbone)
    if response.ndim != 2 or self_pair.shape != response.shape:
        raise ValueError("response and self_pair must have shape (n_ell,n_bin)")
    n_ell, n_bin = response.shape
    if (
        coherent.shape != (n_ell,) or window.shape != (n_ell,)
        or bias.shape != (n_bin,) or halo_self.shape != (n_bin,)
        or particle.ndim != 0
    ):
        raise ValueError("constrained covariance dimensions do not match assignment moments")
    total_self = particle + jnp.sum(halo_self)
    constraint = window / jnp.where(total_self > 0, total_self, 1)
    uncollapsed_bias = 1 - jnp.sum(bias)
    halo_response = jnp.sum(response * bias[None, :], axis=-1)
    self_response = jnp.sum(response * halo_self[None, :], axis=-1)
    uncollapsed = uncollapsed_bias**2 * coherent + particle - constraint * particle**2
    cross = 2 * (uncollapsed_bias * halo_response * coherent
                 - constraint * particle * self_response)
    distinct = halo_response**2 * coherent - constraint * self_response**2
    one_halo = jnp.sum(self_pair * halo_self[None, :], axis=-1)
    # Weighted-variance form avoids subtracting a large white self amplitude
    # from itself when the conserved total is much smaller than its components.
    inverse_total = 1 / jnp.where(total_self > 0, total_self, 1)
    mean_response = (particle + self_response) * inverse_total
    stochastic = (
        particle * (1 - mean_response)**2
        + jnp.sum(halo_self[None, :] * (response - mean_response[:, None])**2, axis=-1)
        + jnp.sum(halo_self[None, :] * (self_pair - response**2), axis=-1)
        + (1 - window) * (particle + self_response)**2 * inverse_total
    )
    total_response = 1 + jnp.sum(bias[None, :] * (response - 1), axis=-1)
    return PaintedAngularSpectra(
        uncollapsed, cross, distinct, one_halo,
        total_response**2 * coherent + stochastic,
    )
