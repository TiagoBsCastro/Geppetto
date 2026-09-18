#!/usr/bin/env python
"""Plot true galaxy members of the most massive staged PINOCCHIO halo.

Read-only input: native M_PIN in Msun/h and observer-relative comoving Mpc/h.
The displayed coordinates are gnomonic cone-frame offsets, in arcminutes,
not ICRS coordinates. Marker colours encode rest-frame ^0.1(g-r), not RGB.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path

import h5py
import numpy as np

ARCMIN_PER_RADIAN = 180.*60./np.pi


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def tangent_offsets(positions, centre, basis):
    """Return (N,2) gnomonic offsets in arcmin along local cone east/north.

    Positions and centre are observer-relative comoving Mpc/h; basis rows
    rotate simulation coordinates into the catalogue's cone frame. Distance
    cancels in the projection. Reject the opposite observer hemisphere.
    """
    positions, centre, basis = map(np.asarray, (positions, centre, basis))
    if positions.ndim != 2 or positions.shape[1] != 3 or centre.shape != (3,):
        raise ValueError("Expected positions (N,3) and centre (3,)")
    if basis.shape != (3, 3) or not np.allclose(basis@basis.T, np.eye(3), atol=1.e-12):
        raise ValueError("Sky basis must be an orthonormal (3,3) matrix")
    distance = np.linalg.norm(centre)
    if not np.isfinite(distance) or distance <= 0 or np.any(~np.isfinite(positions)):
        raise ValueError("Centre and positions must be finite, with centre away from observer")
    direction = basis@centre/distance
    longitude = np.arctan2(direction[1], direction[0])
    latitude = np.arcsin(np.clip(direction[2], -1., 1.))
    east = np.array([-np.sin(longitude), np.cos(longitude), 0.])
    north = np.array([-np.sin(latitude)*np.cos(longitude),
                      -np.sin(latitude)*np.sin(longitude), np.cos(latitude)])
    rotated = positions@basis.T
    depth = rotated@direction
    if np.any(depth <= 0):
        raise ValueError("Gnomonic projection requires the same observer hemisphere")
    return np.column_stack((rotated@east/depth, rotated@north/depth))*ARCMIN_PER_RADIAN


def support_radius_arcmin(radius, distance):
    """Gnomonic silhouette radius of a sphere, for comoving Mpc/h inputs."""
    if not (np.isfinite(radius) and np.isfinite(distance) and 0 < radius < distance):
        raise ValueError("Require finite 0 < support radius < observer distance")
    return radius/np.sqrt(distance**2-radius**2)*ARCMIN_PER_RADIAN


def load_members(catalogue):
    """Select by unique native PLC occurrence, never persistent group ID."""
    before = sha256(catalogue)
    with h5py.File(catalogue, "r") as handle:
        metadata = json.loads(handle.attrs["metadata_json"])
        group = handle["galaxies"]
        mass = group["M_PIN"][:]
        if mass.size == 0 or np.any(~np.isfinite(mass)):
            raise ValueError("Galaxy catalogue has no finite host-mass maximum")
        peak = int(np.argmax(mass))
        host_index = int(group["host_index"][peak])
        host_id = int(group["host_halo_id"][peak])
        rows = np.flatnonzero(group["host_halo_id"][:] == host_id)
        members = {name: values[rows] for name, values in group.items()}
    native = Path(metadata["config"]["halo_input"])
    with h5py.File(native, "r") as handle:
        group = handle["halos"]
        maximum = float(np.max(group["M_PIN"][:]))
        if maximum != float(mass[peak]) or int(group["occurrence_id"][host_index]) != host_id:
            raise ValueError("Most massive native halo is not the selected galaxy host")
        centre = group["position"][host_index]
        group_id = int(group["group_id"][host_index])
        host_z = float(group["z_cos"][host_index])
    if not np.all(members["host_index"] == host_index) or not np.all(members["M_PIN"] == maximum):
        raise ValueError("Member rows disagree on their host identity or mass")
    if np.count_nonzero(members["is_central"]) > 1:
        raise ValueError("More than one central in a host occurrence")
    radius = float(members["R_sat_comoving"][0])
    if np.any(np.linalg.norm(members["position"]-centre, axis=1) > radius*(1+1.e-10)):
        raise ValueError("A member lies beyond the effective satellite radius")
    offsets = tangent_offsets(members["position"], centre, np.asarray(metadata["sky_basis_rows"]))
    distance = float(np.linalg.norm(centre))
    info = dict(catalogue=str(catalogue), catalogue_sha256=before, native_input=str(native),
                host_halo_id=host_id, host_group_id=group_id, host_index=host_index,
                most_massive_in_native_input=True, M_PIN_msun_h=maximum, host_z_cos=host_z,
                centre_comoving_mpc_h=centre.tolist(), distance_comoving_mpc_h=distance,
                R_sat_comoving_mpc_h=radius, support_radius_arcmin=support_radius_arcmin(radius, distance),
                n_members=len(rows), n_centrals=int(members["is_central"].sum()),
                n_satellites=int((~members["is_central"]).sum()),
                n_selected_r_limit=int(members["selected_r_limit"].sum()),
                magnitude_faint=metadata["config"]["calibration"]["magnitude_faint"],
                r_limit=metadata["config"]["survey_r_limit"],
                colour_field="g_r_rest_0p1", colour_reference_redshift=.1,
                projection="gnomonic local east/north in catalogue cone frame; arcmin, not ICRS",
                selection="All underlying-sample members of one host_halo_id; no additional flux cut",
                mass_definition=metadata["mass_definition"], model_parameters_changed=False)
    return members, offsets, info


def plot_members(members, offsets, info, output):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/geppetto-galaxy-halo-mpl")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle

    colours = members["g_r_rest_0p1"]
    colour_min = np.floor(colours.min()*20)/20
    colour_max = max(colour_min+.05, np.ceil(colours.max()*20)/20)
    info["colour_limits"] = [float(colour_min), float(colour_max)]
    figure = plt.figure(figsize=(9.4, 9.6))
    axis = figure.add_axes([.12, .21, .70, .685])
    radius = info["support_radius_arcmin"]
    axis.add_patch(Circle((0., 0.), radius, fill=False, edgecolor=".55", lw=1.3,
                          linestyle=(0, (5, 4)), zorder=1))
    options = dict(cmap="RdBu_r", vmin=colour_min, vmax=colour_max, edgecolors="#242424", linewidths=.65)
    satellite = ~members["is_central"]
    scatter = axis.scatter(*offsets[satellite].T, c=colours[satellite], s=110, zorder=3, **options)
    axis.scatter(*offsets[~satellite].T, c=colours[~satellite], marker="*", s=370, zorder=4, **options)
    axis.set(xlim=(-1.14*radius, 1.14*radius), ylim=(-1.14*radius, 1.14*radius), aspect="equal",
             xlabel=r"Tangent-plane east offset $\xi$ [arcmin]",
             ylabel=r"Tangent-plane north offset $\eta$ [arcmin]")
    axis.tick_params(direction="in", top=True, right=True)
    axis.grid(alpha=.14, lw=.6)
    colour_axis = figure.add_axes([.855, .32, .022, .46])
    colourbar = figure.colorbar(scatter, cax=colour_axis)
    colourbar.set_label(r"Rest-frame $^{0.1}(g-r)$ [mag]", labelpad=10)
    colour_axis.text(.5, -.055, "Bluer", transform=colour_axis.transAxes, ha="center", fontsize=9)
    colour_axis.text(.5, 1.035, "Redder", transform=colour_axis.transAxes, ha="center", fontsize=9)
    figure.text(.12, .972, "Most massive PINOCCHIO halo", fontsize=18, weight="bold", va="top")
    figure.text(.12, .935, rf"$M_{{\rm PIN}} = {info['M_PIN_msun_h']/1.e14:.2f}\times10^{{14}}\,M_\odot/h$"
                rf"     $z_{{\rm host}}={info['host_z_cos']:.4f}$     {info['n_members']} catalogue members", fontsize=11)
    legend = [Line2D([], [], marker="*", color="none", markerfacecolor=".65", markeredgecolor=".15",
                     markersize=14, label=f"Central ({info['n_centrals']})"),
              Line2D([], [], marker="o", color="none", markerfacecolor=".65", markeredgecolor=".15",
                     markersize=8, label=f"Satellites ({info['n_satellites']})"),
              Line2D([], [], color=".55", ls="--", label=rf"$R_{{\rm sat}}={info['R_sat_comoving_mpc_h']:.2f}$ comoving Mpc/$h$")]
    figure.legend(handles=legend, loc="lower left", bbox_to_anchor=(.115, .114), ncol=3,
                  frameon=False, fontsize=9, handlelength=2.)
    figure.text(.12, .029, "\n".join([
        rf"All host members with $^{{0.1}}M_r-5\log_{{10}}h<{info['magnitude_faint']:g}$ AB; {info['n_selected_r_limit']}/{info['n_members']} also pass $r<{info['r_limit']:g}$. Fainter galaxies are not shown.",
        "Colour encodes rest-frame g-r, not observed RGB. Individual simulated objects; no error bars.",
        "M_PIN is native fragmentation-group mass; the dashed radius is a model choice, not measured R200m.",
        "Cone-frame sky projection, not ICRS. True members only; no foreground/background objects.",
    ]), fontsize=8, linespacing=1.5, va="bottom")
    figure.savefig(output, dpi=250)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalogue", type=Path,
                        default=Path("outputs/galaxy_lightcone/validation/galaxies_dc3fc74d63c89809.hdf5"))
    args = parser.parse_args()
    members, offsets, info = load_members(args.catalogue)
    output = args.catalogue.parent/"most_massive_halo_galaxies"
    plot_members(members, offsets, info, output.with_suffix(".png"))
    with output.with_suffix(".csv").open("w", newline="") as handle:
        fields = ["galaxy_id", "host_halo_id", "is_central", "M_PIN", "host_z_cos", "z_cos", "z_obs",
                  "absolute_magnitude_r", "apparent_magnitude_r", "g_r_rest_0p1", "selected_r_limit",
                  "satellite_radius_comoving"]
        writer = csv.DictWriter(handle, fieldnames=fields+["xi_arcmin", "eta_arcmin"])
        writer.writeheader()
        for i in range(len(offsets)):
            writer.writerow({**{name: members[name][i].item() for name in fields},
                             "xi_arcmin": offsets[i, 0], "eta_arcmin": offsets[i, 1]})
    if sha256(args.catalogue) != info["catalogue_sha256"]:
        raise RuntimeError("Input catalogue changed during plotting")
    info["plot_code_sha256"] = sha256(__file__)
    info["catalogue_unchanged"] = True
    output.with_suffix(".json").write_text(json.dumps(info, indent=2, allow_nan=False)+"\n")
    copy = args.catalogue.parent/Path(__file__).name
    if copy.resolve() != Path(__file__).resolve():
        shutil.copy2(__file__, copy)
    print(json.dumps(info, indent=2))
    print(f"Plot: {output.with_suffix('.png')}")


if __name__ == "__main__":
    main()
