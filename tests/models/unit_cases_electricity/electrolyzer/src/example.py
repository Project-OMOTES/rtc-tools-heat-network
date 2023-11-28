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
from rtctools_heat_network.heat_mixin import HeatMixin


class RevenueGoal(Goal):
    priority = 1

    order = 1

    def __init__(self, state, price_profile):
        self.state = state

        self.price_profile = price_profile
        # self.function_range = (-np.inf, 0.)
        self.function_nominal = 1.e6

    def function(self, optimization_problem, ensemble_member):
        return -optimization_problem.state(self.state)


class _GoalsAndOptions:
    def path_goals(self):
        goals = super().path_goals().copy()

        for demand in self.heat_network_components["electricity_demand"]:
            price_profile = f"{demand}.electricity_price"
            state = f"{demand}.Electricity_demand"

            goals.append(RevenueGoal(state, price_profile))

        for demand in self.heat_network_components["gas_demand"]:
            price_profile = f"{demand}.gas_price"
            state = f"{demand}.Gas_demand_flow"

            goals.append(RevenueGoal(state, price_profile))

        return goals


class ElectricityProblem(
    _GoalsAndOptions,
    HeatMixin,
    LinearizedOrderGoalProgrammingMixin,
    GoalProgrammingMixin,
    ESDLMixin,
    CollocatedIntegratedOptimizationProblem,
):
    def path_goals(self):
        goals = super().path_goals().copy()

        return goals


if __name__ == "__main__":
    elect = run_optimization_problem(ElectricityProblem)
    r = elect.extract_results()
    print(r["Electrolyzer_fc66.ElectricityIn.Power"])
