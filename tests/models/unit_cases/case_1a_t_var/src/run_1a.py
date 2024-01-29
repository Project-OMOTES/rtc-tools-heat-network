import numpy as np

from rtctools.optimization.collocated_integrated_optimization_problem import (
    CollocatedIntegratedOptimizationProblem,
)
from rtctools.optimization.goal_programming_mixin import Goal, GoalProgrammingMixin
from rtctools.optimization.linearized_order_goal_programming_mixin import (
    LinearizedOrderGoalProgrammingMixin,
)
from rtctools.util import run_optimization_problem

from rtctools_heat_network.esdl.esdl_mixin import ESDLMixin
from rtctools_heat_network.physics_mixin import PhysicsMixin


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

        for demand in self.heat_network_components["demand"]:
            target = self.get_timeseries(f"{demand}.target_heat_demand")
            state = f"{demand}.Heat_demand"

            goals.append(TargetDemandGoal(state, target))

        return goals


class HeatProblem(
    _GoalsAndOptions,
    PhysicsMixin,
    LinearizedOrderGoalProgrammingMixin,
    GoalProgrammingMixin,
    ESDLMixin,
    CollocatedIntegratedOptimizationProblem,
):

    __temperature_options = {}

    def solver_options(self):
        options = super().solver_options()
        options["solver"] = "highs"
        return options

    def temperature_carriers(self):
        return self.esdl_carriers

    def temperature_regimes(self, carrier):

        temperature_options = []

        try:
            temperature_options = self.__temperature_options[carrier]
        except KeyError:
            for asset in [
                *self.heat_network_components.get("source", []),
                *self.heat_network_components.get("ates", []),
            ]:
                esdl_asset = self.esdl_assets[self.esdl_asset_name_to_id_map[asset]]
                parameters = self.parameters(0)
                for i in range(len(esdl_asset.attributes["constraint"].items)):
                    constraint = esdl_asset.attributes["constraint"].items[i]
                    if (
                        constraint.name == "supply_temperature"
                        and carrier == parameters[f"{asset}.T_supply_id"]
                    ):
                        lb = constraint.range.minValue
                        ub = constraint.range.maxValue
                        n_options = round((ub - lb) / 2.5)
                        temperature_options = np.linspace(lb, ub, n_options + 1)
                        self.__temperature_options[carrier] = temperature_options

        return temperature_options


if __name__ == "__main__":
    sol = run_optimization_problem(HeatProblem)
    results = sol.extract_results()
    a = 1
