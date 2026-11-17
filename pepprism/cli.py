"""Command-line interface for PepPrism therapeutic peptide analysis and design."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from pepprism.biophysics.descriptors import compute_biophysical_profile
from pepprism.biophysics.helical_wheel import compute_helical_wheel
from pepprism.data.datasets import parse_fasta_or_lines
from pepprism.design.optimizer import optimize_peptide
from pepprism.explain.attribution import explain_peptide
from pepprism.models.trainer import TrainedPepPrismBundle

app = typer.Typer(
    name="pepprism",
    help="PepPrism: Interpretable multi-scale biophysical-sequence framework for therapeutic peptides.",
    add_completion=False,
)

PKG_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CHECKPOINTS = {
    "amp": Path("checkpoints/pepprism_amp.pt"),
    "acp": Path("checkpoints/pepprism_acp.pt"),
}


def _resolve_bundle(task: str, checkpoint: Optional[Path]) -> TrainedPepPrismBundle:
    if checkpoint is not None:
        ckpt_path = checkpoint
    else:
        rel_path = DEFAULT_CHECKPOINTS.get(task.lower())
        if rel_path is None:
            ckpt_path = None
        elif rel_path.exists():
            ckpt_path = rel_path
        else:
            ckpt_path = PKG_ROOT / rel_path
    if ckpt_path is None or not ckpt_path.exists():
        raise typer.BadParameter(
            f"Checkpoint not found at '{ckpt_path}'. Run experiments/run_benchmarks.py first or pass --checkpoint."
        )
    return TrainedPepPrismBundle.load(ckpt_path)


def _load_sequences_input(sequence: Optional[str], fasta: Optional[Path]) -> list[tuple[str, str]]:
    if sequence:
        return [("query_1", sequence.strip())]
    if fasta:
        if not fasta.exists():
            raise typer.BadParameter(f"FASTA file not found: {fasta}")
        text = fasta.read_text(encoding="utf-8")
        seqs = parse_fasta_or_lines(text)
        return [(f"seq_{i+1}", s) for i, s in enumerate(seqs)]
    raise typer.BadParameter("Provide either --sequence or --fasta.")


@app.command()
def predict(
    sequence: Optional[str] = typer.Option(None, "--sequence", "-s", help="Single peptide sequence."),
    fasta: Optional[Path] = typer.Option(None, "--fasta", "-f", help="Path to FASTA or line-delimited sequence file."),
    task: str = typer.Option("amp", "--task", "-t", help="Prediction task: amp or acp."),
    checkpoint: Optional[Path] = typer.Option(None, "--checkpoint", "-c", help="Custom model checkpoint path."),
    mc_samples: int = typer.Option(20, "--mc-samples", help="MC-Dropout samples for uncertainty estimation."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Optional JSON output path."),
) -> None:
    """Predict therapeutic peptide activity probability and epistemic uncertainty."""
    bundle = _resolve_bundle(task, checkpoint)
    items = _load_sequences_input(sequence, fasta)
    seqs = [s for _, s in items]
    res = bundle.predict_sequences(seqs, mc_samples=mc_samples)

    records = []
    for i, (sid, seq) in enumerate(items):
        prof = compute_biophysical_profile(seq)
        records.append(
            {
                "id": sid,
                "sequence": prof.sequence,
                "length": prof.length,
                "task": task.upper(),
                "probability": round(float(res["probabilities"][i]), 4),
                "uncertainty_std": round(float(res["uncertainty_std"][i]), 4),
                "seq_gate_ratio": round(float(res["seq_gate_ratio"][i]), 4),
                "net_charge_ph74": prof.net_charge_ph74,
                "hydrophobic_moment_alpha": prof.hydrophobic_moment_alpha,
                "hemolysis_risk_heuristic": prof.hemolysis_risk_heuristic,
            }
        )

    payload = json.dumps(records, indent=2, ensure_ascii=False)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
        typer.echo(f"Saved {len(records)} prediction(s) to {output}")
    else:
        typer.echo(payload)


@app.command()
def analyze(
    sequence: str = typer.Option(..., "--sequence", "-s", help="Peptide sequence to analyze."),
    task: str = typer.Option("amp", "--task", "-t", help="Prediction task: amp or acp."),
    checkpoint: Optional[Path] = typer.Option(None, "--checkpoint", "-c", help="Custom model checkpoint path."),
    mc_samples: int = typer.Option(20, "--mc-samples", help="MC-Dropout samples."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Optional JSON output path."),
) -> None:
    """Compute biophysical profile, helical wheel coordinates, and residue-level attribution."""
    profile = compute_biophysical_profile(sequence)
    wheel = compute_helical_wheel(sequence)
    bundle = _resolve_bundle(task, checkpoint)
    explanation = explain_peptide(bundle, sequence, mc_samples=mc_samples)

    result = {
        "biophysics": profile.to_dict(),
        "helical_wheel": wheel.to_dict(),
        "explanation": explanation.to_dict(),
    }
    payload = json.dumps(result, indent=2, ensure_ascii=False)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
        typer.echo(f"Saved analysis report to {output}")
    else:
        typer.echo(payload)


@app.command()
def optimize(
    sequence: str = typer.Option(..., "--sequence", "-s", help="Parent peptide sequence."),
    task: str = typer.Option("amp", "--task", "-t", help="Optimization target task: amp or acp."),
    checkpoint: Optional[Path] = typer.Option(None, "--checkpoint", "-c", help="Custom model checkpoint path."),
    max_mutations: int = typer.Option(2, "--max-mutations", "-m", help="Maximum mutation depth (1 or 2)."),
    top_k: int = typer.Option(10, "--top-k", "-k", help="Number of top candidates to return."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Optional JSON output path."),
) -> None:
    """Run constrained in silico directed evolution and Pareto optimization."""
    bundle = _resolve_bundle(task, checkpoint)
    report = optimize_peptide(
        bundle,
        sequence,
        max_mutations=max_mutations,
        top_k=top_k,
    )
    payload = json.dumps(report.to_dict(), indent=2, ensure_ascii=False)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
        typer.echo(f"Saved optimization report to {output}")
    else:
        typer.echo(payload)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind host."),
    port: int = typer.Option(8000, "--port", help="Bind port."),
) -> None:
    """Launch the interactive PepPrism FastAPI web workbench."""
    import uvicorn

    uvicorn.run("pepprism.web.app:app", host=host, port=port, reload=False)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
