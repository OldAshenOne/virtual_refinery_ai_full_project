
from dataclasses import dataclass, field
from typing import Dict
@dataclass
class UnitData:
    unit_id: str
    unit_name: str
    consumption: Dict[str, float] = field(default_factory=dict)
    production: Dict[str, float] = field(default_factory=dict)
