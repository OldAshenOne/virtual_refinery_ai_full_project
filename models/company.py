
from dataclasses import dataclass, field
from typing import Dict, Set
from .material import Material
from .unit import UnitData

@dataclass
class CompanyModel:
    name: str
    year: int
    materials: Dict[str, Material] = field(default_factory=dict)
    fixed_factors: Dict[str, float] = field(default_factory=dict)
    units: Dict[str, UnitData] = field(default_factory=dict)
    carry_init: Dict[str, float] = field(default_factory=dict)
    routing: dict = field(default_factory=dict)
    def produced_material_ids(self) -> Set[str]:
        out = set()
        for u in self.units.values():
            out.update(u.production.keys())
        return out
