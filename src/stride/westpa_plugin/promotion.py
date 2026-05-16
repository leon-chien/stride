from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class PromotionDecision:
    promote: bool
    reason: str
    challenger: dict[str, float]
    baseline: dict[str, float]
    champion: dict[str, float] | None = None

    def write_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n")
        return path


def decide_promotion(
    challenger: dict[str, float],
    baseline: dict[str, float],
    champion: dict[str, float] | None = None,
    min_delta: float = 0.0,
    max_occupancy_cv: float = 2.0,
) -> PromotionDecision:
    """
    Decide whether an online challenger is safe to promote.
    """
    checks = (
        ("auprc", "AUPRC"),
        ("top25_enrichment", "top-25 enrichment"),
        ("bin_event_rate_gradient", "bin event-rate gradient"),
    )
    for key, label in checks:
        if float(challenger.get(key, float("-inf"))) <= float(baseline.get(key, float("-inf"))) + min_delta:
            return PromotionDecision(
                promote=False,
                reason=f"challenger does not beat baseline on {label}",
                challenger=dict(challenger),
                baseline=dict(baseline),
                champion=dict(champion) if champion is not None else None,
            )
        if champion is not None and float(challenger.get(key, float("-inf"))) <= float(
            champion.get(key, float("-inf"))
        ) + min_delta:
            return PromotionDecision(
                promote=False,
                reason=f"challenger does not beat champion on {label}",
                challenger=dict(challenger),
                baseline=dict(baseline),
                champion=dict(champion),
            )

    if float(challenger.get("occupancy_cv", 0.0)) > max_occupancy_cv:
        return PromotionDecision(
            promote=False,
            reason="challenger bins are too imbalanced",
            challenger=dict(challenger),
            baseline=dict(baseline),
            champion=dict(champion) if champion is not None else None,
        )

    return PromotionDecision(
        promote=True,
        reason="challenger beats baseline and champion promotion criteria",
        challenger=dict(challenger),
        baseline=dict(baseline),
        champion=dict(champion) if champion is not None else None,
    )
