import math
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class PipeClass:
    """
    This class defines a certain pipe. The optimization can select different pipe classes for each
    pipe, depending on the available pipe classes.
    - name: Name of the pipe class e.g. DN200-S1
    - maximum_velocity: Maximum velocity magnitude in the pipe in m/s.
    - u_values: Insulative properties.
    - investment_costs: Cost for the pipe in euro/meter
    """

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
