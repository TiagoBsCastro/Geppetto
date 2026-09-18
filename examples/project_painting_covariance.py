#!/usr/bin/env python3
"""Replay the experimental painting-matched covariance from explicit node tables.

No measured C_ell, fitted angular transition or target-spectrum amplitude enters
this program. It projects a supplied statistical backbone and actual native-pixel
assignment moments; it does NOT infer PINOCCHIO correlations from a HMF alone.
See docs/painting_matched_halo_model.md for the input contract and limitations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from geppetto.painting_theory import (
    StationaryLineOfSightRule,
    project_constrained_painting_node,
    resolved_painting_population,
    stationary_shell_average,
)


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        result = {}
        for key in source.files:
            value = np.asarray(source[key])
            # FITS-derived pixel windows can retain big-endian storage in NPZ.
            result[key] = value.astype(value.dtype.newbyteorder("="), copy=False)
        return result


def _positive_scalar(value, label: str, *, allow_zero: bool = False) -> float:
    result = np.asarray(value, dtype=float)
    if result.ndim != 0 or not np.isfinite(result) or (result < 0 if allow_zero else result <= 0):
        raise ValueError(f"{label} must be a finite {'nonnegative' if allow_zero else 'positive'} scalar")
    return float(result)


def _integer(value, label: str, *, minimum: int = 0) -> int:
    scalar = _positive_scalar(value, label, allow_zero=True)
    if scalar != int(scalar) or scalar < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return int(scalar)


def line_of_sight_rule(periods: int, order: int) -> StationaryLineOfSightRule:
    """Gauss-Legendre integration of each pi-wide sinc-squared interval."""

    periods = _integer(periods, "los_periods", minimum=1)
    order = _integer(order, "los_order", minimum=4)
    nodes, weights = np.polynomial.legendre.leggauss(order)
    t = (np.arange(periods)[:, None] + .5 * (nodes[None, :] + 1)) * np.pi
    quadrature = weights[None, :] * np.sinc(t / np.pi)**2
    tail = 1 - quadrature.sum()
    if not 0 <= tail < 1:
        raise ValueError("line-of-sight quadrature has an invalid remaining tail; increase order")
    return StationaryLineOfSightRule(
        jnp.asarray(t.ravel()), jnp.asarray(quadrature.ravel()), jnp.asarray(tail),
    )


def validate_node_arrays(data: dict, n_ell: int) -> None:
    """Reject inconsistent mass, backbone and global-assignment tables."""

    keys = ("mass_msun_h", "number_density_weight_mpc_h3", "linear_halo_bias",
            "k_h_mpc", "coherent_power_mpc_h3", "linear_power_mpc_h3",
            "same_protohalo_power_mpc_h3", "lattice_correction_mpc_h3",
            "response", "self_pair")
    for key in keys:
        if key not in data or not np.all(np.isfinite(data[key])):
            raise ValueError(f"node table requires finite {key}")
    mass, number, bias = (data[key] for key in keys[:3])
    if mass.ndim != 1 or number.shape != mass.shape or bias.shape != mass.shape:
        raise ValueError("node mass, number weights, and bias dimensions differ")
    if np.any(mass <= 0) or np.any(np.diff(mass) <= 0) or np.any(number < 0):
        raise ValueError("node masses must be positive/increasing and number weights nonnegative")
    wave = data["k_h_mpc"]
    if wave.ndim != 1 or wave.size < 2 or np.any(wave <= 0) or np.any(np.diff(wave) <= 0):
        raise ValueError("node k grid must be positive/increasing with at least two entries")
    for key in keys[4:8]:
        if data[key].shape != wave.shape:
            raise ValueError(f"node {key} does not match its k grid")
    for key in ("coherent_power_mpc_h3", "linear_power_mpc_h3"):
        if np.any(data[key] < 0):
            raise ValueError(f"node {key} must be nonnegative")
    response, self_pair = data["response"], data["self_pair"]
    if response.shape != (n_ell, mass.size) or self_pair.shape != response.shape:
        raise ValueError("assignment moments must have shape (n_ell,n_mass)")
    tolerance = 1.e-7
    if (np.any(abs(response) > 1 + tolerance) or np.any(self_pair > 1 + tolerance)
            or np.any(self_pair < response**2 - tolerance)):
        raise ValueError("global assignment moments must satisfy |A| <= 1 and A^2 <= D <= 1")
    gradients = [key in data for key in ("response_jacobian", "self_pair_jacobian")]
    if any(gradients) and not all(gradients):
        raise ValueError("both assignment Jacobians are required together")
    if all(gradients):
        for key in ("response_jacobian", "self_pair_jacobian"):
            if data[key].shape != (3, n_ell, mass.size) or not np.all(np.isfinite(data[key])):
                raise ValueError(f"{key} must have finite shape (3,n_ell,n_mass)")


def run(args: argparse.Namespace) -> dict:
    if not jax.config.jax_enable_x64:
        raise ValueError("enable JAX float64 for the high-ell painting covariance projection")
    if args.output.suffix != ".npz":
        raise ValueError("output must have a .npz suffix")
    if args.projection not in {"stationary", "limber"}:
        raise ValueError("projection must be stationary or limber")
    root = args.inputs.resolve().parent
    specification = json.loads(args.inputs.read_text())
    if specification.get("schema_version") != 1:
        raise ValueError("unsupported painting covariance input schema_version")
    for key in ("backbone_description", "assignment_description"):
        if not isinstance(specification.get(key), str) or not specification[key].strip():
            raise ValueError(f"input specification requires {key}")
    density = _positive_scalar(specification["mean_density_msun_h_mpch3"], "mean density")
    particle_mass = _positive_scalar(specification["particle_mass_msun_h"], "particle mass")
    grid_path = root / specification["grid"]
    grid = _load(grid_path)
    ell, pixel = grid["ell"], grid["pixel_window"]
    if (ell.ndim != 1 or not len(ell) or not np.all(np.isfinite(ell))
            or np.any(ell != np.floor(ell)) or np.any(ell < 0) or np.any(np.diff(ell) <= 0)):
        raise ValueError("ell must contain increasing nonnegative integers")
    if pixel.shape != ell.shape or not np.all(np.isfinite(pixel)) or np.any(pixel <= 0) or np.any(pixel > 1):
        raise ValueError("pixel_window must match ell and lie in (0,1]")
    rule = line_of_sight_rule(args.los_periods, args.los_order)
    project = jax.jit(project_constrained_painting_node)
    average = jax.jit(stationary_shell_average)
    shell_data, seen, hashes = {}, set(), {}
    hashes[str(grid_path)] = hashlib.sha256(grid_path.read_bytes()).hexdigest()
    gradient_presence = set()
    continuation = []
    for ordinal, node in enumerate(specification["nodes"]):
        segment = _integer(node["segment_index"], "segment_index")
        redshift = _positive_scalar(node["redshift"], "redshift", allow_zero=True)
        lo = _positive_scalar(node["chi_lo_mpc_h"], "chi_lo", allow_zero=True)
        hi = _positive_scalar(node["chi_hi_mpc_h"], "chi_hi")
        chi = _positive_scalar(node["chi_mpc_h"], "chi")
        dchi = _positive_scalar(node["dchi_weight_mpc_h"], "dchi_weight")
        if not lo < chi < hi:
            raise ValueError("radial nodes must lie strictly inside their shell")
        if (segment, chi) in seen:
            raise ValueError("duplicate radial node would double-count a shell contribution")
        seen.add((segment, chi))
        data_path = root / node["table"]
        data = _load(data_path)
        hashes[str(data_path)] = hashlib.sha256(data_path.read_bytes()).hexdigest()
        validate_node_arrays(data, len(ell))
        if ell[0] == 0 and not np.allclose(
            np.array([data["response"][0], data["self_pair"][0]]), 1., rtol=0, atol=1.e-9,
        ):
            raise ValueError("global assignment moments must have unit monopoles")
        population = resolved_painting_population(
            data["mass_msun_h"], data["number_density_weight_mpc_h3"], data["linear_halo_bias"],
            density, particle_mass,
        )
        uncollapsed = float(population.uncollapsed_mass_fraction)
        if not 0 <= uncollapsed <= 1:
            raise ValueError(f"segment {segment}: resolved HMF mass fraction is outside [0,1]")
        particle_self = float(population.uncollapsed_self_power)
        total_self = particle_self + float(jnp.sum(population.halo_self_power))
        constraint = (data["same_protohalo_power_mpc_h3"] + particle_self
                      - data["lattice_correction_mpc_h3"]) / total_self
        if np.any(constraint < 0) or np.any(constraint > 1):
            raise ValueError(f"segment {segment}: pair-derived constraint lies outside [0,1]; no clipping")
        transverse = (ell + .5) / chi
        projected, continued = [], {}
        for name, values in (("coherent", data["coherent_power_mpc_h3"]),
                             ("constraint", constraint), ("linear", data["linear_power_mpc_h3"])):
            if args.projection == "stationary":
                result = average(transverse, data["k_h_mpc"], values, hi-lo, rule,
                                 low_k_value=values[0], high_k_value=values[-1])
                projected.append(result.value)
                continued[name] = dict(
                    low_k_max_absolute=float(jnp.max(jnp.abs(result.low_k_continuation))),
                    high_k_max_absolute=float(jnp.max(jnp.abs(result.high_k_continuation))),
                )
            else:
                projected.append(jnp.asarray(np.interp(transverse, data["k_h_mpc"], values)))
                continued[name] = dict(
                    low_k_multipoles=int(np.count_nonzero(transverse < data["k_h_mpc"][0])),
                    high_k_multipoles=int(np.count_nonzero(transverse > data["k_h_mpc"][-1])),
                )
        coherent, projected_constraint, linear = projected
        radial_weight = dchi * chi**2 / ((hi**3-lo**3)/3)**2
        moments = (jnp.asarray(data["response"]), jnp.asarray(data["self_pair"]))
        fixed = (population, coherent, projected_constraint, radial_weight, jnp.asarray(pixel))
        result = project(*fixed, *moments)
        if not np.all(np.isfinite(result)) or np.any(np.asarray(result.total) < 0):
            raise ValueError(f"segment {segment}: non-finite or negative covariance prediction")
        arrays = dict(components=np.asarray(result), linear=radial_weight*pixel**2*np.asarray(linear))
        has_gradients = "response_jacobian" in data
        gradient_presence.add(has_gradients)
        if has_gradients:
            arrays["jacobian"] = np.stack([
                np.asarray(jax.jvp(lambda a, d, fixed=fixed: project(*fixed, a, d), moments, (
                    jnp.asarray(data["response_jacobian"][parameter]),
                    jnp.asarray(data["self_pair_jacobian"][parameter]),
                ))[1]) for parameter in range(3)
            ])
        if segment not in shell_data:
            shell_data[segment] = dict(bounds=(lo, hi), dchi=0., **{
                key: np.zeros_like(value) for key, value in arrays.items()
            })
        accumulator = shell_data[segment]
        if accumulator["bounds"] != (lo, hi):
            raise ValueError("nodes of one segment must agree on shell boundaries")
        for key, value in arrays.items():
            if key not in accumulator:
                raise ValueError("all nodes must consistently supply or omit assignment Jacobians")
            accumulator[key] += value
        accumulator["dchi"] += dchi
        continuation.append(dict(segment=segment, redshift=redshift,
                                 uncollapsed_mass_fraction=uncollapsed, endpoint_continuations=continued))
        print(f"[painting theory] node {ordinal+1}/{len(specification['nodes'])}, segment={segment}", flush=True)
    if not shell_data or len(gradient_presence) > 1:
        raise ValueError("require nonempty nodes with consistent assignment Jacobian availability")
    for segment, shell in shell_data.items():
        if not np.isclose(shell["dchi"], shell["bounds"][1]-shell["bounds"][0], rtol=1.e-10, atol=1.e-10):
            raise ValueError(f"segment {segment}: radial weights do not integrate the shell width")
    indices = np.array(list(shell_data))
    components = np.stack([row["components"] for row in shell_data.values()])
    stationary_linear = np.stack([row["linear"] for row in shell_data.values()])
    if "shell_linear" in grid:
        if (not np.array_equal(grid.get("segment_indices"), indices)
                or grid["shell_linear"].shape != stationary_linear.shape
                or not np.all(np.isfinite(grid["shell_linear"])) or np.any(grid["shell_linear"] < 0)):
            raise ValueError("independent linear projection must match the requested segment/ell grid")
        linear_reference = grid["shell_linear"]
    else:
        linear_reference = stationary_linear
    linear_correction = linear_reference - stationary_linear
    total = components[:, -1] + linear_correction
    if not np.all(np.isfinite(total)) or np.any(total < 0):
        raise ValueError("linear projection replacement produced invalid total power")
    report = dict(
        status="experimental; total agreement does not validate separate backbone components",
        projection=args.projection, los_periods=args.los_periods, los_order=args.los_order,
        los_tail_weight=float(rule.tail_weight), input_sha256=hashes,
        backbone_description=specification["backbone_description"],
        assignment_description=specification["assignment_description"],
        linear_projection_replaced="shell_linear" in grid, nodes=continuation,
        endpoint_policy="constant; reported continuation contributions are not physical error bounds",
    )
    output = dict(
        ell=ell, segment_indices=indices, pixel_window=pixel, stationary_components=components,
        stationary_linear=stationary_linear, linear_projection_correction=linear_correction,
        shell_total=total, metadata_json=np.asarray(json.dumps(report, sort_keys=True)),
    )
    if gradient_presence == {True}:
        output["component_jacobian"] = np.stack([row["jacobian"] for row in shell_data.values()])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.resolve() in {Path(path).resolve() for path in hashes} | {args.inputs.resolve()}:
        raise ValueError("output must not overwrite an input")
    np.savez_compressed(args.output, **output)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True, help="JSON node specification")
    parser.add_argument("--output", type=Path, required=True, help="output NPZ")
    parser.add_argument("--projection", choices=("stationary", "limber"), default="stationary")
    parser.add_argument("--los-periods", type=int, default=256)
    parser.add_argument("--los-order", type=int, default=16)
    return parser.parse_args()


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    run(parse_args())
