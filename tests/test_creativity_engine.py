from creativity.scamper_engine import generate_scamper


def test_scamper_generates_ranked_candidates():
    report = generate_scamper("a neighborhood tool-sharing library")
    assert report.safety_verdict == "allowed"
    assert len(report.candidates) == 7
    assert {c.technique for c in report.candidates} >= {"Substitute", "Combine", "Adapt", "Modify", "Put to other use", "Eliminate", "Reverse"}
    scores = [c.novelty + c.usefulness for c in report.candidates]
    assert scores == sorted(scores, reverse=True)
    assert all(c.rationale for c in report.candidates)


def test_scamper_blocks_weapons():
    report = generate_scamper("improve missile weapon targeting")
    assert report.safety_verdict == "blocked"
    assert report.candidates == []
