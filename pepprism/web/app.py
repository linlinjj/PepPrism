"""FastAPI backend serving PepPrism predictions, explanations, and optimization."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from pepprism.biophysics.descriptors import compute_biophysical_profile
from pepprism.biophysics.helical_wheel import compute_helical_wheel
from pepprism.design.optimizer import optimize_peptide
from pepprism.explain.attribution import explain_peptide
from pepprism.models.trainer import TrainedPepPrismBundle

STATIC_DIR = Path(__file__).parent / "static"
PKG_ROOT = Path(__file__).resolve().parents[2]

PRESET_PEPTIDES: list[dict[str, str]] = [
    {
        "name": "Magainin-2 (经典两亲性抗菌肽)",
        "sequence": "GIGKFLHSAKKFGKAFVGEIMNS",
        "note": "非洲爪蟾皮肤分泌的典型阳离子 α-螺旋抗菌肽，具有优异膜选择性。",
    },
    {
        "name": "LL-37 (人源 Cathelicidin 抗菌肽)",
        "sequence": "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES",
        "note": "人体先天免疫核心多肽，兼具广谱抗菌、免疫调节及抗肿瘤活性。",
    },
    {
        "name": "Cecropin A (天蚕素 A)",
        "sequence": "KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK",
        "note": "昆虫源高电荷螺旋抗菌肽，兼具抗革兰氏阴性菌与抑癌潜能。",
    },
    {
        "name": "Melittin (蜂毒肽 - 高溶血警示案例)",
        "sequence": "GIGAVLKVLTTGLPALISWIKRKRQQ",
        "note": "强效但高溶血毒性的两亲性肽，用于演示溶血风险预警与低毒优化。",
    },
    {
        "name": "Indolicidin (富含 Trp 抗菌肽)",
        "sequence": "ILPWKWPWWPWRR",
        "note": "牛中性粒细胞分泌的 13 肽短序列，富含色氨酸和精氨酸锚定基序。",
    },
]


class AnalyzeRequest(BaseModel):
    sequence: str = Field(..., min_length=5, max_length=200)
    task: Literal["amp", "acp"] = "amp"
    mc_samples: int = Field(default=20, ge=1, le=50)


class OptimizeRequest(BaseModel):
    sequence: str = Field(..., min_length=5, max_length=200)
    task: Literal["amp", "acp"] = "amp"
    max_mutations: int = Field(default=2, ge=1, le=2)
    top_k: int = Field(default=10, ge=1, le=30)
    min_charge: float = Field(default=2.0, ge=-5.0, le=15.0)
    max_charge: float = Field(default=9.0, ge=0.0, le=20.0)
    max_instability: float = Field(default=40.0, ge=10.0, le=120.0)


def _get_checkpoint_path(task: str, checkpoint_dir: Path | None = None) -> Path:
    base = checkpoint_dir or (PKG_ROOT / "checkpoints")
    return base / f"pepprism_{task.lower()}.pt"


def create_app(checkpoint_dir: Path | None = None) -> FastAPI:
    """Create configured FastAPI application instance."""
    api = FastAPI(
        title="PepPrism Web Workbench",
        description="Interpretable Multi-Scale Biophysical-Sequence Framework for Therapeutic Peptides",
        version="0.1.0",
    )
    bundle_cache: dict[str, TrainedPepPrismBundle] = {}

    def load_bundle(task: str) -> TrainedPepPrismBundle:
        key = task.lower()
        if key in bundle_cache:
            return bundle_cache[key]
        path = _get_checkpoint_path(key, checkpoint_dir=checkpoint_dir)
        if not path.exists():
            raise HTTPException(
                status_code=503,
                detail=f"Model checkpoint '{path}' not found. Run experiments/run_benchmarks.py first.",
            )
        bundle = TrainedPepPrismBundle.load(path)
        bundle_cache[key] = bundle
        return bundle

    @api.get("/", response_class=HTMLResponse)
    def home() -> HTMLResponse:
        html_path = STATIC_DIR / "index.html"
        return HTMLResponse(html_path.read_text(encoding="utf-8"))

    @api.get("/api/health")
    def health() -> dict[str, object]:
        available = [
            t for t in ("amp", "acp") if _get_checkpoint_path(t, checkpoint_dir).exists()
        ]
        return {"status": "ok", "available_tasks": available}

    @api.get("/api/presets")
    def presets() -> list[dict[str, str]]:
        return PRESET_PEPTIDES

    @api.post("/api/analyze")
    def analyze(req: AnalyzeRequest) -> dict[str, object]:
        try:
            profile = compute_biophysical_profile(req.sequence)
            wheel = compute_helical_wheel(req.sequence)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        bundle = load_bundle(req.task)
        explanation = explain_peptide(bundle, profile.sequence, mc_samples=req.mc_samples)
        return {
            "task": req.task.upper(),
            "biophysics": profile.to_dict(),
            "helical_wheel": wheel.to_dict(),
            "explanation": explanation.to_dict(),
        }

    @api.post("/api/optimize")
    def optimize(req: OptimizeRequest) -> dict[str, object]:
        if req.min_charge > req.max_charge:
            raise HTTPException(status_code=400, detail="min_charge must be <= max_charge.")
        try:
            compute_biophysical_profile(req.sequence)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        bundle = load_bundle(req.task)
        report = optimize_peptide(
            bundle,
            req.sequence,
            max_mutations=req.max_mutations,
            top_k=req.top_k,
            charge_bounds=(req.min_charge, req.max_charge),
            max_instability=req.max_instability,
        )
        return report.to_dict()

    return api


app = create_app()
