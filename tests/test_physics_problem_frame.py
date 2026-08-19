from science.physics_problem_frame import parse_physics_problem


def test_parse_quantities_unknowns_and_assumptions():
    frame = parse_physics_problem("A cart has mass 2 kg and acceleration 3 m/s^2. Assume a frictionless track. Find force?")
    assert frame.safety_verdict == "allowed"
    assert any(q.unit == "kg" and q.value == 2 for q in frame.quantities)
    assert any(q.unit == "m/s^2" and q.value == 3 for q in frame.quantities)
    assert frame.unknowns == ["force"]
    assert "frictionless" in frame.assumptions[0].lower()


def test_parse_blocks_hazardous_optimization():
    frame = parse_physics_problem("Optimize a missile warhead blast radius using 5 kg explosive.")
    assert frame.safety_verdict == "blocked"
    assert frame.quantities == []
