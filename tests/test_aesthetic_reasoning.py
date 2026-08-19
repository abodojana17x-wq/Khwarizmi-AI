from art.aesthetic_scorer import score_aesthetics


def test_aesthetic_report_is_interpretable():
    report = score_aesthetics({"focal_point": (0.33, 0.66), "symmetry": 0.8, "balance": 0.75, "negative_space": 0.3, "harmony": "complementary", "contrast": 0.85, "temperature": "warm", "saturation": 0.6})
    assert report.overall_score > 75
    assert any("Composition" in item for item in report.findings)
    assert any("Color" in item for item in report.findings)


def test_low_contrast_gets_suggestion():
    report = score_aesthetics({"contrast": 0.2, "balance": 0.4})
    assert any("contrast" in s.lower() for s in report.suggestions)
