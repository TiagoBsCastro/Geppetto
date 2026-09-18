"""Experimental PINOCCHIO particle-pair backbone, evaluated on the host.

This module has no concentration dependence and does not paint maps. It uses
velocileptors' one-loop LPT displacement correlators, their full covariance/
third-cumulant characteristic, a cubic particle lattice, and Gaussian density
conditioning inside uniform spherical protohaloes. The last two ingredients
describe replacement of the resolved particle population, not a fitted C_ell
transition. They are approximations to PINOCCHIO fragmentation and exclusion.

The resummation follows the cumulant construction discussed by Vlah, Seljak &
Baldauf (2015), arXiv:1410.1617. The scalar pair-replacement strategy is related
to Valageas & Nishimichi (2011), arXiv:1009.0597. Neither reference validates
this particular conditional-selection or component-covariance closure.

Distances are comoving Mpc/h, masses Msun/h, k h/Mpc and P(k) (Mpc/h)^3.
The optional ``lpt`` extra is required only when calculating a backbone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np


class LPTGrowthRatios(NamedTuple):
    """Dimensionless growth relative to the Einstein-de Sitter kernels.

    PINOCCHIO conventions give second_order = D2 / [(3/7) D1^2] and
    third_order_b = D3b / [(5/42) D1^3]. Both are one in EdS. The third-order
    determinant kernel vanishes for the (k,p,-p) configuration entering
    Gaussian one-loop P13; D3a therefore does not enter these correlators.
    The supplied linear P(k,z) already includes D1 and is NOT grown again.
    """

    second_order: float
    third_order_b: float


@dataclass(frozen=True)
class LPTBackboneParams:
    """Explicit physical conditioning and numerical quadrature settings.

    ``collapse_threshold`` is the linear overdensity used in the Gaussian
    protohalo constraint. ``q_max_mpc_h`` is a numerical integration window,
    NOT a physical exclusion radius; convergence must be checked. The
    spherical Fourier cutoff is fixed to the simulation's pi/grid_spacing.
    All other orders/chunk sizes are dimensionless positive integers.
    """

    collapse_threshold: float = 1.686
    k_min_h_mpc: float = .001
    k_max_h_mpc: float = 6.
    k_nodes: int = 192
    q_max_mpc_h: float = 128.
    continuous_order: int = 512
    spectral_order: int = 512
    angular_order: int = 128
    fft_size: int = 4096
    bessel_orders: int = 20
    lens_order: int = 16
    conditioning_radial_nodes: int = 257
    k_chunk_size: int = 4
    threads: int = 1


@dataclass(frozen=True)
class LPTBackboneSpectrum:
    """Concentration-independent 3D tables, all powers in (Mpc/h)^3.

    ``same_protohalo_power`` includes the collapsed particles' own diagonal.
    Its zero-mode normalization is ``halo_self_power``. ``lattice_correction``
    is discrete minus continuous displaced-particle power, before removal.
    For unpainted native-host haloes the total before angular pixelization is
    coherent_power + lattice_correction - same_protohalo_power + halo_self.
    Do not add an independent particle Poisson term to this combination.
    """

    k_h_mpc: np.ndarray
    coherent_power: np.ndarray
    linear_power: np.ndarray
    lattice_correction: np.ndarray
    same_protohalo_power: np.ndarray
    resolved_mass_fraction: float
    halo_self_power: float
    same_protohalo_zero_mode: float


def _validate_params(params: LPTBackboneParams) -> None:
    for name in ("collapse_threshold", "k_min_h_mpc", "k_max_h_mpc", "q_max_mpc_h"):
        value = getattr(params, name)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if params.k_min_h_mpc >= params.k_max_h_mpc:
        raise ValueError("backbone k range must be increasing")
    for name, minimum in (
        ("k_nodes", 2), ("continuous_order", 4), ("spectral_order", 4), ("angular_order", 4),
        ("fft_size", 128), ("bessel_orders", 2), ("lens_order", 2),
        ("conditioning_radial_nodes", 3), ("k_chunk_size", 1), ("threads", 1),
    ):
        value = getattr(params, name)
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")


def cubic_lattice_shells(spacing_mpc_h: float, radius_mpc_h: float) -> tuple[np.ndarray, np.ndarray]:
    """Distinct cubic-lattice distances and exact multiplicities in a sphere.

    Includes the origin once. Uses integer-square convolution rather than
    allocating a three-dimensional lattice. This is an isotropic shell sum,
    not a calculation of the lattice's directional anisotropy.
    """

    from scipy.signal import fftconvolve

    if (not np.isfinite(spacing_mpc_h) or not np.isfinite(radius_mpc_h)
            or spacing_mpc_h <= 0 or radius_mpc_h < 0):
        raise ValueError("lattice spacing must be positive and radius nonnegative")
    n = int(np.floor(radius_mpc_h / spacing_mpc_h))
    one_dimensional = np.bincount(np.arange(n+1)**2, weights=np.r_[1., np.full(n, 2.)])
    counts = np.rint(fftconvolve(fftconvolve(one_dimensional, one_dimensional), one_dimensional))
    distance = np.sqrt(np.arange(counts.size)) * spacing_mpc_h
    chosen = (counts > 0) & (distance <= radius_mpc_h)
    return distance[chosen], counts[chosen]


def spherical_protohalo_overlap(separation_mpc_h: np.ndarray, radius_mpc_h: np.ndarray) -> np.ndarray:
    """Fractional overlap of equal spherical protohaloes at separation q.

    Inputs are nonnegative q and positive R in comoving Mpc/h; broadcasting
    is allowed. This is a geometric probability assumption, not a measured
    halo exclusion function or an NFW support radius.
    """

    q, radius = np.asarray(separation_mpc_h), np.asarray(radius_mpc_h)
    if np.any(~np.isfinite(q)) or np.any(~np.isfinite(radius)) or np.any(q < 0) or np.any(radius <= 0):
        raise ValueError("protohalo separations/radii must be finite with q >= 0 and R > 0")
    ratio = q / radius
    return np.where(ratio < 2, (2-ratio)**2 * (4+ratio) / 16, 0.)


def _direct_cumulants(model, q: np.ndarray, nyquist: float, growth: LPTGrowthRatios, order: int):
    """Band-separated Bessel integrals avoid sharp-cutoff FFTLog ringing."""

    from scipy.special import spherical_jn

    nodes, weights = np.polynomial.legendre.leggauss(order)
    wave = np.r_[(nodes+1)*nyquist/2, (nodes+3)*nyquist/2]
    dk = np.tile(weights*nyquist/2, 2) / (2*np.pi**2)
    linear = np.interp(wave, model.kint, model.pint)
    linear[wave > nyquist] = 0.
    r1, r2, q1, q2 = [np.interp(wave, model.kint, getattr(model.qf, name))
                      for name in ("R1", "R2", "Q1", "Q2")]
    r1[wave > nyquist] = 0.
    r2[wave > nyquist] = 0.
    loop = growth.third_order_b*(10/21)*r1 + growth.second_order**2*(9/98)*q1
    vsource = growth.second_order*(3/35)*(-4*r1+12*r2-2*q1+6*q2)
    tsource = growth.second_order*(-3/7)*(2*r1+4*r2+q1+2*q2)
    outputs = [[] for _ in range(6)]
    for first in range(0, len(q), 128):
        phase = q[first:first+128, None]*wave[None, :]
        j2 = spherical_jn(2, phase)
        shape = 1-spherical_jn(0, phase)-j2
        shape = np.where(phase < .001, phase**2/10-phase**4/280+phase**6/15120, shape)
        xx, yy = (2/3)*(shape @ (dk*linear)), 2*(j2 @ (dk*linear))
        xl, yl = (2/3)*(shape @ (dk*loop)), 2*(j2 @ (dk*loop))
        tl = 3*(spherical_jn(3, phase) @ (dk*tsource/wave))
        vl = 3*(spherical_jn(1, phase) @ (dk*vsource/wave))-.6*tl
        for destination, value in zip(outputs, (xx, yy, xl, yl, vl, tl), strict=True):
            destination.append(value)
    return [np.concatenate(value)[None, :, None] for value in outputs], (2/3)*np.sum(dk*(linear+loop))


def _conditioned_pairs(
    radius, fraction, pair_normalization, q, multiplicity, wave, power,
    *, particle_volume, params,
):
    """Gaussian delta_R conditioning, averaged over the spherical lens.

    Linear conditioning is Gaussian-exact at fixed endpoints. Spherical
    geometry, uniform endpoint sampling, the fixed barrier and the unchanged
    intrinsic higher-LPT cumulants are separate physical approximations.
    """

    from scipy.special import spherical_jn

    nodes, weights = np.polynomial.legendre.leggauss(params.spectral_order)
    k = (nodes+1)*wave[-1]/2
    dk = weights*wave[-1]/2/(2*np.pi**2)
    pk = np.interp(k, wave, power)
    ln, lw = np.polynomial.legendre.leggauss(params.lens_order)
    unit = (ln+1)/2
    result = {key: [] for key in ("q_index", "weight", "mean", "x", "y", "v", "t")}
    for index, rl in enumerate(radius):
        if fraction[index] == 0:
            continue
        selected = np.flatnonzero((q > 0) & (q < 2*rl))
        if not selected.size:
            raise ValueError("a populated protohalo has no distinct lattice pairs")
        qr = q[selected]
        kr = k*rl
        window = 3*spherical_jn(1, kr)/kr
        sigma2 = np.sum(dk*k**2*pk*window**2)
        if not np.isfinite(sigma2) or sigma2 <= 0:
            raise ValueError("conditional protohalo variance must be finite and positive")
        radial = np.linspace(0, rl, params.conditioning_radial_nodes)
        j1_over_r = spherical_jn(1, radial[:, None]*k[None, :])/np.where(radial == 0, 1., radial)[:, None]
        j1_over_r[0] = k/3
        coefficient = -(j1_over_r @ (dk*k*pk*window))
        if not np.isclose(coefficient[-1], -sigma2/3, rtol=2.e-12, atol=0):
            raise ValueError("conditional displacement violates its spherical boundary identity")
        zmax = rl-qr/2
        z = zmax[:, None, None]*unit[None, :, None]
        edge = z+qr[:, None, None]/2
        smax2 = (rl-edge)*(rl+edge)
        s2 = smax2*unit[None, None, :]
        rm = np.sqrt(s2+(z-qr[:, None, None]/2)**2)
        rp = np.sqrt(s2+(z+qr[:, None, None]/2)**2)
        fm = np.interp(rm.ravel(), radial, coefficient).reshape(rm.shape)
        fp = np.interp(rp.ravel(), radial, coefficient).reshape(rp.shape)
        bz = fp*(z+qr[:, None, None]/2)-fm*(z-qr[:, None, None]/2)
        bs2 = s2*(fp-fm)**2
        overlap = spherical_protohalo_overlap(qr, rl)
        lens_volume = 4*np.pi*rl**3/3*overlap
        measure = (np.pi/2)*zmax[:, None, None]*smax2*lw[None, :, None]*lw[None, None, :]
        measure /= lens_volume[:, None, None]
        if not np.allclose(measure.sum(axis=(1, 2)), 1., rtol=2.e-9, atol=0):
            raise ValueError("conditional protohalo lens quadrature does not normalize")

        def average(value, measure=measure):
            return np.sum(measure*value, axis=(1, 2))

        b = average(bz)
        bzz, bxx = average(bz**2), average(bs2)/2
        bzzz, bzxx = average(bz**3), average(bz*bs2)/2
        alpha = params.collapse_threshold/sigma2
        coefficient = alpha**2-1/sigma2
        xx = coefficient*bxx
        zz = coefficient*bzz-alpha**2*b**2
        kzzz = alpha**3*(bzzz-3*b*bzz+2*b**3)-3*alpha/sigma2*(bzzz-b*bzz)
        kzxx = alpha**3*(bzxx-b*bxx)-alpha/sigma2*(3*bzxx-b*bxx)
        probability = particle_volume*multiplicity[selected]*overlap*fraction[index]*pair_normalization[index]
        for name, values in zip(result, (selected, probability, alpha*b, xx, zz-xx,
                                        3*kzxx, kzzz-3*kzxx), strict=True):
            result[name].append(values)
    return {name: np.concatenate(values) if values else np.empty(0, dtype=int if name == "q_index" else float)
            for name, values in result.items()}


def build_lpt_particle_backbone(
    linear_k_h_mpc: np.ndarray,
    linear_power_mpc_h3: np.ndarray,
    mass_msun_h: np.ndarray,
    number_density_weight_mpc_h3: np.ndarray,
    *,
    mean_density_msun_h_mpch3: float,
    particle_mass_msun_h: float,
    grid_spacing_mpc_h: float,
    growth: LPTGrowthRatios,
    params: LPTBackboneParams | None = None,
) -> LPTBackboneSpectrum:
    """Calculate a shell-redshift backbone without an observed angular map.

    Input P(k,z) must already have the correct growth and normalization at
    the requested redshift. It is band-limited at the simulation's spherical
    Nyquist cutoff for displacement construction. The uncut input is retained
    as the linear reference. Native number weights have units (Mpc/h)^-3 and
    already integrate over mass bins. Only resolved catalogue haloes enter;
    their abundance is never rescaled to contain all matter.

    This host-side calculation is NOT differentiable in cosmology, HMF or
    growth. Its arrays are fixed inputs to the differentiable painting
    operator. No concentration/profile parameter belongs in this function.
    k integration is chunked by ``k_chunk_size``; no pixel maps are allocated.
    The isotropically averaged lattice and protohalo assumptions require
    separate scientific validation. No fitted EFT or C_ell parameters enter.
    """

    params = LPTBackboneParams() if params is None else params
    _validate_params(params)
    wave, power, mass, number = [np.asarray(value, dtype=np.float64) for value in (
        linear_k_h_mpc, linear_power_mpc_h3, mass_msun_h, number_density_weight_mpc_h3,
    )]
    if (wave.ndim != 1 or wave.size < 4 or power.shape != wave.shape or np.any(np.diff(wave) <= 0)
            or np.any(wave <= 0) or np.any(power <= 0) or not np.all(np.isfinite(wave+power))):
        raise ValueError("input linear k/P must be finite, positive matching vectors with increasing k")
    if (mass.ndim != 1 or number.shape != mass.shape or np.any(mass <= 0) or np.any(number < 0)
            or not np.all(np.isfinite(mass+number))):
        raise ValueError("native mass/count weights must be finite matching vectors with M > 0 and q >= 0")
    rho, particle, spacing = mean_density_msun_h_mpch3, particle_mass_msun_h, grid_spacing_mpc_h
    if any(not np.isfinite(value) or value <= 0 for value in (rho, particle, spacing, *growth)):
        raise ValueError("density, particle mass, lattice spacing and growth ratios must be positive")
    if not np.isclose(particle/rho, spacing**3, rtol=1.e-8, atol=0):
        raise ValueError("particle mass / mean density must equal lattice cell volume")
    if np.any(mass[number > 0] < particle):
        raise ValueError("a resolved halo cannot have less mass than one simulation particle")
    populated = number > 0
    mass, number = mass[populated], number[populated]
    fraction = number*mass/rho
    fhalo = float(fraction.sum())
    if not 0 <= fhalo <= 1:
        raise ValueError("resolved HMF mass fraction must lie in [0,1]")
    halo_self = float(np.sum(number*(mass/rho)**2))
    radius = (3*mass/(4*np.pi*rho))**(1/3)
    if radius.size and 2*radius.max() > params.q_max_mpc_h:
        raise ValueError("q_max_mpc_h must cover the complete largest protohalo diameter")
    nyquist = np.pi/spacing
    if not wave[0] < nyquist < wave[-1]:
        raise ValueError("input linear k grid must bracket the simulation Nyquist cutoff")
    if not wave[0] <= params.k_min_h_mpc < params.k_max_h_mpc <= wave[-1]:
        raise ValueError("requested output k range lies outside the supplied linear spectrum")
    try:
        from velocileptors.LPT.cleft_fftw import CLEFT
    except ImportError as error:
        raise ImportError("install geppetto[lpt] to calculate the experimental LPT backbone") from error
    model = CLEFT(wave, power, one_loop=True, shear=False, third_order=False,
                  cutoff=np.inf, N=params.fft_size, jn=params.bessel_orders, threads=params.threads)
    model.pint *= model.kint <= nyquist
    model.setup_powerspectrum()
    model.Xloop = 2*growth.third_order_b*model.qf.Xloop13+growth.second_order**2*model.qf.Xloop22
    model.Yloop = 2*growth.third_order_b*model.qf.Yloop13+growth.second_order**2*model.qf.Yloop22
    model.sigmaloop = model.Xloop[-1]
    model.Vloop *= growth.second_order
    model.Tloop *= growth.second_order
    model.make_ptable(kmin=params.k_min_h_mpc, kmax=params.k_max_h_mpc, nk=params.k_nodes)
    k, expanded = model.pktable[:, :2].T
    discrete, multiplicity = cubic_lattice_shells(spacing, params.q_max_mpc_h)
    nodes, weights = np.polynomial.legendre.leggauss(params.continuous_order)
    continuous = np.r_[(nodes+1)*params.q_max_mpc_h/4, (nodes+3)*params.q_max_mpc_h/4]
    continuous_weight = np.tile(weights*params.q_max_mpc_h/4, 2)
    q = np.r_[discrete, continuous]
    nd = discrete.size
    numerical_window = np.where(q <= params.q_max_mpc_h/2, 1.,
                                .5*(1+np.cos(np.pi*(2*q/params.q_max_mpc_h-1))))
    mu, mu_weight = np.polynomial.legendre.leggauss(params.angular_order)
    mu = mu[None, None, :]
    old = [np.interp(q, model.qint, getattr(model, name)) for name in (
        "Xlin", "Ylin", "Xloop", "Yloop", "Vloop", "Tloop",
    )]
    for value in old:
        value[q == 0] = 0.
    x, y, xl, yl, vl, tl = [value[None, :, None] for value in old]
    direct, sigma = _direct_cumulants(model, q, nyquist, growth, params.spectral_order)
    xd, yd, xld, yld, vld, tld = direct
    overlap = spherical_protohalo_overlap(discrete[1:, None], radius[None, :])
    pair_denominator = multiplicity[1:] @ overlap
    if np.any(pair_denominator <= 0):
        raise ValueError("protohalo has no nonzero overlap with the particle lattice")
    pair_normalization = (mass/particle-1)/pair_denominator
    selected = wave < nyquist
    conditional = _conditioned_pairs(
        radius, fraction, pair_normalization, discrete, multiplicity,
        np.r_[wave[selected], nyquist], np.r_[power[selected], np.interp(nyquist, wave, power)],
        particle_volume=spacing**3, params=params,
    )
    zero_mode = spacing**3*fhalo+conditional["weight"].sum()
    if not np.isclose(zero_mode, halo_self, rtol=3.e-12, atol=1.e-14):
        raise ValueError("same-protohalo subtraction fails the native mass-squared zero-mode closure")
    qi = conditional["q_index"]
    variance = xd+xld+mu**2*(yd+yld)
    conditional_variance = (xd[:, qi]+xld[:, qi]+conditional["x"][None, :, None]
                            +mu**2*(yd[:, qi]+yld[:, qi]+conditional["y"][None, :, None]))
    if np.min(variance) < -1.e-10 or np.min(conditional_variance, initial=0.) < -1.e-10:
        raise ValueError("LPT/conditional displacement covariance is not positive")
    aliases, counters, corrections = [], [], []
    for first in range(0, k.size, params.k_chunk_size):
        kv = k[first:first+params.k_chunk_size, None, None]
        phase = kv*q[None, :, None]*mu
        old_angular = np.exp(-.5*kv**2*(x+mu**2*y))*(
            (1-.5*kv**2*(xl+mu**2*yl))*np.cos(phase)
            +kv**3/6*(mu*vl+mu**3*tl)*np.sin(phase))
        old_characteristic = .5*np.sum(old_angular*mu_weight, axis=-1)
        third = kv**3/6*(mu*vld+mu**3*tld)
        characteristic = .5*np.sum(np.exp(-.5*kv**2*variance)*np.cos(phase-third)*mu_weight, axis=-1)
        kk = kv[:, :, 0]
        old_asymptote = np.exp(-.5*kk**2*model.sigma)*(1-.5*kk**2*model.Xloop[-1])
        asymptote = np.exp(-.5*kk**2*sigma)
        free = np.sinc(kk*q[None, :]/np.pi)
        changed = (characteristic-old_characteristic-(asymptote-old_asymptote)*free)*numerical_window
        corrections.extend(4*np.pi*np.sum(changed[:, nd:]*continuous**2*continuous_weight, axis=1))
        connected = (characteristic-asymptote*free)*numerical_window
        aliases.extend(spacing**3*np.sum(connected[:, :nd]*multiplicity, axis=1)
                       -4*np.pi*np.sum(connected[:, nd:]*continuous**2*continuous_weight, axis=1))
        phase_c = kv*(discrete[qi]+conditional["mean"])[None, :, None]*mu
        third_c = kv**3/6*(mu*(vld[:, qi]+conditional["v"][None, :, None])
                           +mu**3*(tld[:, qi]+conditional["t"][None, :, None]))
        characteristic_c = .5*np.sum(np.exp(-.5*kv**2*conditional_variance)
                                     *np.cos(phase_c-third_c)*mu_weight, axis=-1)
        counters.extend(spacing**3*fhalo+characteristic_c @ conditional["weight"])
    coherent = expanded+np.asarray(corrections)
    if not np.all(np.isfinite(coherent)) or np.any(coherent < 0):
        raise ValueError("the approximate LPT coherent spectrum is non-finite or negative")
    return LPTBackboneSpectrum(
        k.copy(), coherent, np.exp(np.interp(np.log(k), np.log(wave), np.log(power))),
        np.asarray(aliases), np.asarray(counters), fhalo, halo_self, float(zero_mode),
    )
