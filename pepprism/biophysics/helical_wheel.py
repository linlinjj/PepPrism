"""Alpha-helical wheel projection geometry and amphipathic vector analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from pepprism.biophysics.descriptors import sanitize_sequence
from pepprism.biophysics.scales import AA_GROUP, EISENBERG_HYDROPHOBICITY

HELIX_STEP_DEG: float = 100.0


@dataclass(frozen=True)
class HelicalResiduePoint:
    """Single residue projected onto an alpha-helical wheel plane."""

    position: int
    amino_acid: str
    group: str
    hydrophobicity: float
    angle_deg: float
    x: float
    y: float
    ring: int


@dataclass(frozen=True)
class HelicalWheelData:
    """Complete alpha-helical wheel geometric projection and moment vector."""

    sequence: str
    points: list[HelicalResiduePoint]
    moment_magnitude: float
    moment_angle_deg: float
    moment_vx: float
    moment_vy: float

    def to_dict(self) -> dict[str, object]:
        """Serialize helical wheel geometry for API/JSON responses."""
        return asdict(self)


def compute_helical_wheel(sequence: str, max_residues: int = 36) -> HelicalWheelData:
    """Compute 2D alpha-helical wheel coordinates (100 deg/residue) and moment vector."""
    seq = sanitize_sequence(sequence, strict=True)[:max_residues]
    points: list[HelicalResiduePoint] = []
    sum_cos = 0.0
    sum_sin = 0.0

    for idx, aa in enumerate(seq):
        # Rotate clockwise from top (90 degrees in standard Cartesian coordinates)
        angle_deg = (idx * HELIX_STEP_DEG) % 360.0
        rad = math.radians(90.0 - angle_deg)
        ring = idx // 18  # 18 residues complete 5 helical turns (1800 deg)
        radius = 1.0 + 0.24 * ring
        x = radius * math.cos(rad)
        y = radius * math.sin(rad)

        h = EISENBERG_HYDROPHOBICITY[aa]
        sum_cos += h * math.cos(rad)
        sum_sin += h * math.sin(rad)

        points.append(
            HelicalResiduePoint(
                position=idx + 1,
                amino_acid=aa,
                group=AA_GROUP[aa],
                hydrophobicity=round(h, 3),
                angle_deg=round(angle_deg, 1),
                x=round(x, 4),
                y=round(y, 4),
                ring=ring,
            )
        )

    n = max(len(seq), 1)
    vx = sum_cos / n
    vy = sum_sin / n
    mag = math.hypot(vx, vy)
    moment_angle = (90.0 - math.degrees(math.atan2(vy, vx))) % 360.0

    return HelicalWheelData(
        sequence=seq,
        points=points,
        moment_magnitude=round(mag, 4),
        moment_angle_deg=round(moment_angle, 2),
        moment_vx=round(vx, 4),
        moment_vy=round(vy, 4),
    )
