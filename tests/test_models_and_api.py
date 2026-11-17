"""End-to-end tests for deep models, explanation, optimization, checkpointing, and FastAPI."""

from pathlib import Path

from fastapi.testclient import TestClient
import numpy as np
import pytest

from pepprism.biophysics.descriptors import extract_feature_matrix
from pepprism.design.optimizer import optimize_peptide
from pepprism.explain.attribution import explain_peptide
from pepprism.models.trainer import (
    TrainedPepPrismBundle,
    compute_classification_metrics,
    train_deep_model,
)
from pepprism.web.app import create_app


def _synthetic_toy_dataset():
    pos = [
        "GIGKFLHSAKKFGKAFVGEIMNS",
        "KWKLFKKIEKVGQNIRDGIIKAG",
        "LLGDFFRKSKEKIGKEFKRIVQR",
        "FAKKFAKKFKKFAKKFAKFAFAF",
        "GLLDTFKNLALNAAKSAGVSVLN",
        "GLFDIVKKVVGAIGSL",
        "ILPWKWPWWPWRR",
        "FLGALFKALSKLL",
        "KLLKLLKKLLKLLK",
        "RWRWRWRWRW",
    ]
    neg = [
        "DDDDDEEEEEGGGGSSSS",
        "EEEEEDDDDDTTTTNNNN",
        "QQQQQSSSSSTTTTGGGG",
        "PPPPPGSSSSQQQQNNNN",
        "DDDTTTSSSEEEQQQNNN",
        "GGGGSSSTTTNNNQQQDD",
        "EEEDDDNNNQQQSSSGGG",
        "AAATTTSSSGGGDDDQQQ",
        "NNNQQQSSSTTTDDDEEE",
        "SSSTTTGGGDDDQQQNNN",
    ]
    seqs = pos + neg
    labels = np.array([1] * len(pos) + [0] * len(neg), dtype=np.int64)
    bio_mat, _ = extract_feature_matrix(seqs, include_dpc=False)
    return seqs, labels, bio_mat


@pytest.mark.parametrize(
    "arch",
    [
        "SeqCNN1D",
        "BiGRUAttention",
        "PepPrism_SeqOnly",
        "PepPrism_BioOnly",
        "PepPrism_Concat",
        "PepPrismNet",
    ],
)
def test_deep_architectures_train_and_predict(arch: str) -> None:
    seqs, labels, bio_mat = _synthetic_toy_dataset()
    bundle = train_deep_model(
        arch=arch,
        train_sequences=seqs,
        train_labels=labels,
        train_bio_raw=bio_mat,
        max_len=48,
        epochs=3,
        batch_size=8,
        seed=42,
    )
    out = bundle.predict_sequences(seqs[:4], mc_samples=5)
    probs = out["probabilities"]
    stds = out["uncertainty_std"]
    attn = out["attention"]

    assert probs.shape == (4,)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    assert stds.shape == (4,)
    assert np.all(stds >= 0.0)
    assert attn.shape == (4, 48)


def test_explain_optimize_save_load_and_api(tmp_path: Path) -> None:
    seqs, labels, bio_mat = _synthetic_toy_dataset()
    bundle = train_deep_model(
        arch="PepPrismNet",
        train_sequences=seqs,
        train_labels=labels,
        train_bio_raw=bio_mat,
        max_len=48,
        epochs=4,
        batch_size=8,
        seed=42,
    )

    ckpt_dir = tmp_path / "ckpts"
    ckpt_path = ckpt_dir / "pepprism_amp.pt"
    bundle.save(ckpt_path)
    reloaded = TrainedPepPrismBundle.load(ckpt_path)

    p_orig = bundle.predict_sequences([seqs[0]], mc_samples=1)["probabilities"][0]
    p_reload = reloaded.predict_sequences([seqs[0]], mc_samples=1)["probabilities"][0]
    assert p_orig == pytest.approx(p_reload, abs=1e-6)

    expl = explain_peptide(reloaded, seqs[0], mc_samples=5)
    assert len(expl.residue_attributions) == len(seqs[0])
    assert 0.0 <= expl.seq_gate_ratio <= 1.0
    assert len(expl.top_hotspots) == 5

    opt = optimize_peptide(reloaded, seqs[0], max_mutations=2, top_k=6, beam_width=4)
    assert len(opt.candidates) == 6
    assert any(c.pareto_optimal for c in opt.candidates)

    metrics = compute_classification_metrics(labels, bundle.predict_sequences(seqs)["probabilities"])
    assert 0.0 <= metrics.auroc <= 1.0
    assert 0.0 <= metrics.ece <= 1.0

    api = create_app(checkpoint_dir=ckpt_dir)
    client = TestClient(api)
    resp_home = client.get("/")
    assert resp_home.status_code == 200
    assert "PepPrism" in resp_home.text
    assert client.get("/api/health").json()["available_tasks"] == ["amp"]
    assert len(client.get("/api/presets").json()) >= 3

    resp_an = client.post("/api/analyze", json={"sequence": seqs[0], "task": "amp", "mc_samples": 4})
    assert resp_an.status_code == 200
    assert "helical_wheel" in resp_an.json()

    resp_opt = client.post("/api/optimize", json={"sequence": seqs[0], "task": "amp", "max_mutations": 1, "top_k": 5})
    assert resp_opt.status_code == 200
    assert len(resp_opt.json()["candidates"]) == 5

    resp_bad = client.post("/api/analyze", json={"sequence": "INVALID123", "task": "amp"})
    assert resp_bad.status_code == 400
