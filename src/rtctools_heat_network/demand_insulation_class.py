from dataclasses import dataclass


@dataclass(frozen=True)
class DemandInsulationClass:
    name_insulation_level: str
    name_demand: str
    minimum_temperature_deg: float
    demand_scaling_factor: float  # value between 0.0 and 1.0
    investment_cost_euro: float
