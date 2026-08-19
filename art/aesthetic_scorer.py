"""Interpretable deterministic aesthetic scoring."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class AestheticReport:
    composition_score: float
    color_score: float
    overall_score: float
    findings: list[str]
    suggestions: list[str]


def score_aesthetics(description: dict) -> AestheticReport:
    """Score a structured art brief with interpretable composition and color criteria."""
    findings: list[str] = [] ; suggestions: list[str] = []
    focal = description.get("focal_point", (0.5, 0.5))
    thirds = min(abs(focal[0]-1/3), abs(focal[0]-2/3)) + min(abs(focal[1]-1/3), abs(focal[1]-2/3))
    rule = max(0.0, 1 - thirds * 2.2)
    symmetry = float(description.get("symmetry", 0.5))
    balance = float(description.get("balance", 0.5))
    negative = float(description.get("negative_space", 0.3))
    neg_score = max(0.0, 1 - abs(negative - 0.32) * 2)
    comp = (rule + symmetry + balance + neg_score) / 4
    findings.append(f"Composition: rule-of-thirds={rule:.2f}, symmetry={symmetry:.2f}, balance={balance:.2f}, negative-space={neg_score:.2f}.")
    if rule < .55: suggestions.append("Move the focal point closer to a one-third intersection for stronger hierarchy.")
    if balance < .55: suggestions.append("Redistribute visual weight to improve left/right or top/bottom balance.")

    harmony = str(description.get("harmony", "analogous")).lower()
    harmony_score = {"complementary": .9, "analogous": .85, "triadic": .82, "monochrome": .75}.get(harmony, .55)
    contrast = float(description.get("contrast", 0.5))
    temperature = str(description.get("temperature", "balanced")).lower()
    temp_score = .85 if temperature in {"warm", "cool", "balanced"} else .6
    saturation = float(description.get("saturation", 0.55))
    sat_score = max(0.0, 1 - abs(saturation - .58) * 1.4)
    color = (harmony_score + contrast + temp_score + sat_score) / 4
    findings.append(f"Color: harmony={harmony_score:.2f} ({harmony}), contrast={contrast:.2f}, temperature={temp_score:.2f}, saturation-control={sat_score:.2f}.")
    if contrast < .5: suggestions.append("Increase value contrast to improve readability and focal separation.")
    if harmony_score < .7: suggestions.append("Choose a clearer complementary, analogous, triadic, or monochrome palette relationship.")
    overall = round((comp * .55 + color * .45) * 100, 1)
    return AestheticReport(round(comp*100,1), round(color*100,1), overall, findings, suggestions)
