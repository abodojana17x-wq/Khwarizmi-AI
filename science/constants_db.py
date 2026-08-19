"""Local offline physical constants database."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional


@dataclass(frozen=True)
class PhysicalConstant:
    name: str
    symbol: str
    value: float
    unit: str
    dimensions: dict[str, int]
    description: str


_CONSTANTS: Dict[str, PhysicalConstant] = {
    "c": PhysicalConstant("speed of light in vacuum", "c", 299_792_458.0, "m/s", {"m": 1, "s": -1}, "Exact SI value."),
    "G": PhysicalConstant("Newtonian gravitational constant", "G", 6.67430e-11, "m^3/(kg s^2)", {"m": 3, "kg": -1, "s": -2}, "CODATA conventional value."),
    "h": PhysicalConstant("Planck constant", "h", 6.62607015e-34, "J s", {"kg": 1, "m": 2, "s": -1}, "Exact SI value."),
    "k_B": PhysicalConstant("Boltzmann constant", "k_B", 1.380649e-23, "J/K", {"kg": 1, "m": 2, "s": -2, "K": -1}, "Exact SI value."),
    "e": PhysicalConstant("elementary charge", "e", 1.602176634e-19, "C", {"A": 1, "s": 1}, "Exact SI value."),
    "g": PhysicalConstant("standard gravity", "g", 9.80665, "m/s^2", {"m": 1, "s": -2}, "Standard acceleration due to gravity."),
    "epsilon_0": PhysicalConstant("vacuum electric permittivity", "epsilon_0", 8.8541878128e-12, "F/m", {"kg": -1, "m": -3, "s": 4, "A": 2}, "Electric constant."),
    "mu_0": PhysicalConstant("vacuum magnetic permeability", "mu_0", 1.25663706212e-6, "N/A^2", {"kg": 1, "m": 1, "s": -2, "A": -2}, "Magnetic constant."),
    "N_A": PhysicalConstant("Avogadro constant", "N_A", 6.02214076e23, "1/mol", {"mol": -1}, "Exact SI value."),
    "R": PhysicalConstant("molar gas constant", "R", 8.31446261815324, "J/(mol K)", {"kg": 1, "m": 2, "s": -2, "mol": -1, "K": -1}, "Ideal gas constant."),
}
_ALIASES = {c.name.lower(): key for key, c in _CONSTANTS.items()} | {key.lower(): key for key in _CONSTANTS}


def get_constant(name_or_symbol: str) -> Optional[PhysicalConstant]:
    """Return a constant by symbol or lowercase name without network access."""
    key = _ALIASES.get(name_or_symbol.strip().lower())
    return _CONSTANTS.get(key) if key else None


def list_constants() -> Iterable[PhysicalConstant]:
    return tuple(_CONSTANTS.values())
