from science.unit_consistency_verifier import parse_unit, verify_equation


def test_parse_derived_unit_dimensions():
    assert parse_unit("kg*m/s^2") == {"kg": 1, "m": 1, "s": -2}


def test_newtons_second_law_matches():
    verdict = verify_equation("F = m*a")
    assert verdict.ok
    assert verdict.details == "dimensions match"


def test_energy_distance_mismatch_details():
    verdict = verify_equation("E = m*a")
    assert not verdict.ok
    assert "mismatch" in verdict.details
