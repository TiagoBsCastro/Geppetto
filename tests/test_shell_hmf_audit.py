import csv
import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    return importlib.import_module("audit_pinocchio_shell_hmf")


def test_mass_moments_and_selection(module):
    count, mass, squared = module.halo_mass_moments(
        np.array([2., 3., 5., 8.]), np.array([.1, .2, .3, .4]),
        np.array([.1, .25, .4]), np.log([1., 4., 10.]),
    )
    np.testing.assert_array_equal(count, [[2, 0], [0, 1]])
    np.testing.assert_array_equal(mass, [[5, 0], [0, 5]])
    np.testing.assert_array_equal(squared, [[13, 0], [0, 25]])


def test_inclusive_boundary_and_part_additivity(module):
    kwargs = dict(z_edges=np.array([.1, .3]), log_mass_edges=np.log([1., 10.]), inclusive_upper=True)
    together = np.array(module.halo_mass_moments(np.array([2., 4.]), np.array([.1, .3]), **kwargs))
    separately = sum(np.array(module.halo_mass_moments(np.array([m]), np.array([z]), **kwargs))
                     for m, z in [(2., .1), (4., .3)])
    np.testing.assert_array_equal(together, separately)
    np.testing.assert_array_equal(together[:, 0, 0], [2, 6, 20])


def test_mass_grid_must_cover_selected_haloes(module):
    with pytest.raises(ValueError, match="cover"):
        module.halo_mass_moments(np.array([100.]), np.array([.2]), np.array([.1, .3]), np.log([1., 10.]))


def _audit_inputs(tmp_path, monkeypatch, module, mode, mismatch=None):
    monkeypatch.setattr(module, "read_pinocchio_parameter_file", lambda _: SimpleNamespace(
        particle_mass_msun_h=1.,
    ))
    catalogue = np.zeros((3, 13))
    catalogue[:, 0] = [1, 2, 3]
    catalogue[:, 1] = [.1, .3, .2]
    catalogue[:, 8] = [2., 3., 5.]
    catalogue[:, 12] = [.3, .2, .4]
    base = tmp_path / "pinocchio.test.plc.out"
    np.savetxt(f"{base}.0", catalogue[:2])
    np.savetxt(f"{base}.1", catalogue[2:])
    counts, masses = ([2, 1], [7., 3.]) if mode == "true" else ([1, 2], [3., 7.])
    rows = [dict(
        segment_index=index, redshift_mode=mode, z_lo=lo, z_hi=hi,
        inclusive_upper=index == 1, selected_halo_count=counts[index],
        selected_halo_mass_msun_h=masses[index],
    ) for index, (lo, hi) in enumerate([(.1, .25), (.25, .4)])]
    if mismatch is not None:
        rows[0][mismatch] += 1
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(reversed(rows))
    return SimpleNamespace(
        params=tmp_path / "params.txt", manifest=manifest, plc_catalog=base,
        output=tmp_path / "result.npz", redshift_bins=2, mass_bins=20,
    )


@pytest.mark.parametrize("mode", ["true", "observed"])
def test_catalogue_audit_reads_each_part_once_and_uses_manifest_selection(
    tmp_path, monkeypatch, module, mode,
):
    args = _audit_inputs(tmp_path, monkeypatch, module, mode)
    original_reader = module.read_pinocchio_lightcone_catalog
    paths = []

    def read(path):
        paths.append(path)
        return original_reader(path)

    monkeypatch.setattr(module, "read_pinocchio_lightcone_catalog", read)
    assert module.run(args) == args.output
    assert paths == [Path(f"{args.plc_catalog}.0"), Path(f"{args.plc_catalog}.1")]
    with np.load(args.output, allow_pickle=False) as result:
        np.testing.assert_array_equal(result["segment_indices"], [0, 1])
        expected = ([2, 1], [7., 3.], [29., 9.]) if mode == "true" else (
            [1, 2], [3., 7.], [9., 29.],
        )
        for field, values in zip(("count", "mass_sum", "mass_squared_sum"), expected, strict=True):
            assert result[field].shape == (2, 2, 20)
            np.testing.assert_array_equal(result[field].sum(axis=(1, 2)), values)


@pytest.mark.parametrize("mismatch", ["selected_halo_count", "selected_halo_mass_msun_h"])
def test_catalogue_audit_rejects_mismatched_manifest_before_writing(
    tmp_path, monkeypatch, module, mismatch,
):
    args = _audit_inputs(tmp_path, monkeypatch, module, "true", mismatch)
    with pytest.raises(ValueError, match="manifest"):
        module.run(args)
    assert not args.output.exists()
