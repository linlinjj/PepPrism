"""Unit tests verifying PepPrism first-principles biophysical calculations against Biopython."""

from Bio.SeqUtils.ProtParam import ProteinAnalysis
import numpy as np
import pytest

from pepprism.biophysics.descriptors import (
    compute_biophysical_profile,
    compute_dpc,
    compute_pseaac,
    extract_feature_matrix,
)
from pepprism.biophysics.helical_wheel import compute_helical_wheel


@pytest.mark.parametrize(
    "seq",
    [
        "GIGKFLHSAKKFGKAFVGEIMNS",
        "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES",
        "ILPWKWPWWPWRR",
        "KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK",
    ],
)
def test_biophysics_matches_biopython(seq: str) -> None:
    prof = compute_biophysical_profile(seq)
    pa = ProteinAnalysis(seq)

    assert prof.length == len(seq)
    assert prof.molecular_weight == pytest.approx(pa.molecular_weight(), rel=1e-3)
    assert prof.net_charge_ph74 == pytest.approx(pa.charge_at_pH(7.4), abs=1e-3)
    assert prof.isoelectric_point == pytest.approx(pa.isoelectric_point(), abs=0.05)
    assert prof.gravy_kd == pytest.approx(pa.gravy(), abs=1e-3)
    assert prof.instability_index == pytest.approx(pa.instability_index(), abs=1e-2)
    assert prof.aromaticity == pytest.approx(pa.aromaticity(), abs=1e-4)


def test_helical_wheel_matches_hydrophobic_moment() -> None:
    seq = "GIGKFLHSAKKFGKAFVGEIMNS"
    prof = compute_biophysical_profile(seq)
    wheel = compute_helical_wheel(seq)
    assert len(wheel.points) == len(seq)
    assert wheel.moment_magnitude == pytest.approx(prof.hydrophobic_moment_alpha, abs=1e-3)


def test_pseaac_dpc_and_feature_matrix_shapes() -> None:
    seqs = ["GIGKFLHSAKKFGKAFVGEIMNS", "ILPWKWPWWPWRR"]
    dpc = compute_dpc(seqs[0])
    pseaac = compute_pseaac(seqs[0], lam=5)
    assert dpc.shape == (400,)
    assert dpc.sum() == pytest.approx(1.0, abs=1e-5)
    assert pseaac.shape == (25,)
    assert pseaac.sum() == pytest.approx(1.0, abs=1e-5)

    bio_mat, classic_mat = extract_feature_matrix(seqs)
    assert bio_mat.shape == (2, 17)
    assert classic_mat.shape == (2, 462)
    assert np.isfinite(bio_mat).all()
    assert np.isfinite(classic_mat).all()


def test_invalid_sequence_raises() -> None:
    with pytest.raises(ValueError):
        compute_biophysical_profile("")
    with pytest.raises(ValueError):
        compute_biophysical_profile("GIGKFLX123")
