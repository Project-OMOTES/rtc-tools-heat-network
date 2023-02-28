import math
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class PipeClass:
    name: str
    inner_diameter: float
    maximum_velocity: float
    u_values: Tuple[float, float]
    investment_costs: float

    @property
    def maximum_discharge(self):
        return self.area * self.maximum_velocity

    @property
    def area(self):
        return 0.25 * math.pi * self.inner_diameter**2
