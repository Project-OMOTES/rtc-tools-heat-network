import numpy as np

from rtctools.optimization.collocated_integrated_optimization_problem import (
    CollocatedIntegratedOptimizationProblem,
)
from rtctools.optimization.goal_programming_mixin import GoalProgrammingMixin
from rtctools.optimization.goal_programming_mixin_base import Goal
from rtctools.optimization.linearized_order_goal_programming_mixin import (
    LinearizedOrderGoalProgrammingMixin,
)
from rtctools.util import run_optimization_problem

from rtctools_heat_network.esdl.esdl_mixin import ESDLMixin
from rtctools_heat_network.physics_mixin import PhysicsMixin
from rtctools_heat_network.techno_economic_mixin import TechnoEconomicMixin


class TargetDemandGoal(Goal):
    priority = 1

    order = 2

    def __init__(self, state, target):
        self.state = state

        self.target_min = target
        self.target_max = target
        self.function_range = (0.0, 2.0 * max(target.values))
        self.function_nominal = np.median(target.values)

    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.state(self.state)


class _GoalsAndOptions:
    def path_goals(self):
        goals = super().path_goals().copy()

        for demand in self.heat_network_components["electricity_demand"]:
            target = self.get_timeseries(f"{demand}.target_electricity_demand")
            state = f"{demand}.Electricity_demand"

            goals.append(TargetDemandGoal(state, target))

        return goals

    def heat_network_options(self):
        options = super().heat_network_options()
        options["include_electric_cable_power_loss"] = True

        return options


class ElectricityProblem(
    _GoalsAndOptions,
    TechnoEconomicMixin,
    LinearizedOrderGoalProgrammingMixin,
    GoalProgrammingMixin,
    ESDLMixin,
    CollocatedIntegratedOptimizationProblem,
):
    def path_goals(self):
        goals = super().path_goals().copy()

        return goals


class ElectricityProblemMaxCurr(
    PhysicsMixin,
    LinearizedOrderGoalProgrammingMixin,
    GoalProgrammingMixin,
    ESDLMixin,
    CollocatedIntegratedOptimizationProblem,
):
    def read(self):
        super().read()

        for d in self.heat_network_components["electricity_demand"]:
            new_timeseries = self.get_timeseries(f"{d}.target_electricity_demand").values * 50
            self.set_timeseries(f"{d}.target_electricity_demand", new_timeseries)

    def path_goals(self):
        goals = super().path_goals().copy()

        for demand in self.heat_network_components["electricity_demand"]:
            target = self.get_timeseries(f"{demand}.target_electricity_demand")
            state = f"{demand}.Electricity_demand"

            goals.append(TargetDemandGoal(state, target))

        return goals

    def bounds(self):
        bounds = super().bounds()
        bounds["ElectricityProducer_b95d.Electricity_source"] = (0.0, 100000.0)
        bounds["ElectricityCable_238f.ElectricityIn.Power"] = (0.0, 100000.0)
        bounds["ElectricityCable_238f.ElectricityOut.Power"] = (0.0, 100000.0)
        return bounds


if __name__ == "__main__":
    elect = run_optimization_problem(ElectricityProblem)
    r = elect.extract_results()
    print(r["ElectricityDemand_2af6.Electricity_demand"])
    print(r["ElectricityDemand_2af6.ElectricityIn.Power"])
    print(r["ElectricityDemand_2af6.ElectricityIn.V"])
    print(r["ElectricityDemand_2af6.ElectricityIn.I"])
