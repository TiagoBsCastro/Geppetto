#!/usr/bin/env python3
"""Plot normalized HMF closure and both schema-6 one-halo conventions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from geppetto.cosmology import RHO_CRIT0_MSUNH_PER_MPCH3
from geppetto.io import read_pinocchio_mass_function


def _bin(values, ell, width=20):
    selected = np.flatnonzero(ell >= 20)
    bins = [selected[start : start + width] for start in range(0, selected.size, width)]
    centers = np.array([np.average(ell[b], weights=2 * ell[b] + 1) for b in bins])
    data = np.stack(
        [np.average(values[..., b], axis=-1, weights=2 * ell[b] + 1) for b in bins], axis=-1
    )
    return centers, data


def render(input_dir: Path, output_dir: Path) -> None:
    """Write PDF/PNG diagnostics without recomputing any theory."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    audit = json.loads((input_dir / "normalized_hmf_audit.json").read_text())
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 10,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "savefig.bbox": "tight",
        }
    )
    with np.load(input_dir / "normalized_hmf_quadrature.npz", allow_pickle=False) as table:
        scale = table["scale_factor"]
        mass = table["mass_msun_h"]
        weight = table["mass_fraction_weight"]
        bias_weight = table["biased_mass_fraction_weight"]
    controls = audit["convergence"][-1]
    _, integration_weight = np.polynomial.legendre.leggauss(controls["mass_order"])
    integration_weight *= np.log(controls["maximum_mass"] / controls["minimum_mass"]) / 2
    figure, axes = plt.subplots(2, 2, figsize=(8, 6), constrained_layout=True)
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
    chosen = [
        min(audit["fit_parameters"][0], key=lambda fit, z=z: abs(fit["redshift"] - z))
        for z in np.linspace(0, 1 / scale[0] - 1, 4)
    ]
    for i, fit in enumerate(chosen):
        z = fit["redshift"]
        index = int(np.argmin(np.abs(scale - 1 / (1 + z))))
        positive = weight[index] > 1e-14
        label = f"z = {1 / scale[index] - 1:.2f}"
        axes[0, 0].loglog(mass, weight[index] / integration_weight, color=colors[i], label=label)
        axes[0, 1].semilogx(
            mass[positive],
            bias_weight[index, positive] / weight[index, positive],
            color=colors[i],
            label=label,
        )
        density = RHO_CRIT0_MSUNH_PER_MPCH3 * audit["omega_m0"]
        for seed in audit["fit_parameters"]:
            matching = min(seed, key=lambda row: abs(row["redshift"] - z))
            source = Path(matching["source"])
            if source.exists():
                native = read_pinocchio_mass_function(source)
                valid = native.halo_counts > 0
                axes[0, 0].scatter(
                    native.mass_msun_h[valid],
                    native.number_density[valid] * native.mass_msun_h[valid] ** 2 / density,
                    s=6,
                    color=colors[i],
                    alpha=0.4,
                    linewidths=0,
                )
    axes[0, 0].set(xlabel=r"$M\ [M_\odot/h]$", ylabel=r"$d f_{\rm mass}/d\ln M$", ylim=(1e-5, 1))
    axes[0, 1].set(xlabel=r"$M\ [M_\odot/h]$", ylabel=r"Normalized $b_1(M)$", ylim=(0, 12))
    axes[0, 0].legend(frameon=False, fontsize=8)
    for seed in audit["closure"]:
        z = [row["redshift"] for row in seed]
        axes[1, 0].plot(z, [row["mass_integral"] - 1 for row in seed], color=colors[0], alpha=0.7)
        axes[1, 0].plot(z, [row["bias_integral"] - 1 for row in seed], color=colors[1], alpha=0.7)
        axes[1, 1].plot(
            z, [row["bias_renormalization"] for row in seed], color=colors[0], alpha=0.6
        )
        axes[1, 1].plot(z, [row["low_mass_fraction"] for row in seed], color=colors[1], alpha=0.6)
    axes[1, 0].plot([], [], color=colors[0], label=r"$\int f\,d\ln M -1$")
    axes[1, 0].plot([], [], color=colors[1], label=r"$\int fb_1\,d\ln M -1$")
    axes[1, 0].set(xlabel="Redshift", ylabel="Closure residual")
    axes[1, 0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[1, 0].legend(frameon=False, fontsize=8)
    axes[1, 1].plot([], [], color=colors[0], label="Bias normalization factor")
    axes[1, 1].plot([], [], color=colors[1], label="Analytic low-tail mass fraction")
    axes[1, 1].set(xlabel="Redshift", ylabel="Dimensionless weight")
    axes[1, 1].legend(frameon=False, fontsize=8)
    for suffix in ("pdf", "png"):
        figure.savefig(output_dir / f"normalized_hmf_closure.{suffix}", dpi=240)
    plt.close(figure)
    archive = input_dir / "angular_power_theory.npz"
    if not archive.exists():
        return
    with np.load(archive, allow_pickle=False) as source:
        if int(source["validation_schema_version"]) != 6:
            raise ValueError("schema-6 theory is required")
        ell, zlo, zhi = source["ell"], source["z_lo"], source["z_hi"]
        suffix = "_pseudo_over_fsky" if "observed_realizations" not in source.files else ""
        two = source[f"shell_two_halo{suffix}"]
        shot = source[f"shell_particle_shot_noise{suffix}"]
        standard = two + shot + source[f"shell_one_halo_standard{suffix}"]
        compensated = two + shot + source[f"shell_one_halo_compensated{suffix}"]
        centers, observed = _bin(source["observed_shell"], ell)
        _, standard = _bin(standard, ell)
        _, compensated = _bin(compensated, ell)
        sem = None
        if "observed_realizations" in source.files:
            _, real = _bin(source["observed_realizations"], ell)
            sem = np.std(real, axis=0, ddof=1) / np.sqrt(real.shape[0])
    ncols, nrows = 4, int(np.ceil(zlo.size / 4))
    figure, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(11, 2.15 * nrows),
        sharex=True,
        constrained_layout=True,
        squeeze=False,
    )
    for panel, index in enumerate(np.argsort(zlo)):
        ax = axes.flat[panel]
        ax.axhline(1, color="0.4", lw=0.6)
        ax.plot(
            centers,
            observed[index] / compensated[index],
            color="#0072B2",
            lw=1,
            label="Standard 2h + compensated 1h",
        )
        ax.plot(
            centers,
            observed[index] / standard[index],
            color="#CC79A7",
            lw=1,
            ls="--",
            label="Standard 2h + standard 1h",
        )
        if sem is not None:
            ax.fill_between(
                centers,
                (observed[index] - sem[index]) / compensated[index],
                (observed[index] + sem[index]) / compensated[index],
                color="#0072B2",
                alpha=0.18,
                linewidth=0,
            )
        ax.set(
            xscale="log",
            xlim=(20, ell[-1]),
            ylim=(0, 1.5),
            title=f"{zlo[index]:.3f} < z < {zhi[index]:.3f}",
        )
        if panel % ncols == 0:
            ax.set_ylabel("Measured / theory")
        if panel + ncols >= zlo.size:
            ax.set_xlabel(r"$\ell$")
            ax.tick_params(axis="x", labelbottom=True)
    for ax in axes.flat[zlo.size :]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncols=2, frameon=False)
    for suffix in ("pdf", "png"):
        figure.savefig(output_dir / f"angular_power_all_shells_schema6.{suffix}", dpi=240)
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    render(args.input_dir, args.output_dir)
