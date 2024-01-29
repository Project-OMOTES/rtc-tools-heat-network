import datetime

import esdl

import numpy as np

from rtctools.data.storage import DataStore
from rtctools.optimization.collocated_integrated_optimization_problem import (
    CollocatedIntegratedOptimizationProblem,
)
from rtctools.optimization.goal_programming_mixin import Goal, GoalProgrammingMixin
from rtctools.optimization.linearized_order_goal_programming_mixin import (
    LinearizedOrderGoalProgrammingMixin,
)
from rtctools.optimization.single_pass_goal_programming_mixin import (
    CachingQPSol,
    SinglePassGoalProgrammingMixin,
)
from rtctools.util import run_optimization_problem

from rtctools_heat_network.esdl.esdl_mixin import ESDLMixin
from rtctools_heat_network.techno_economic_mixin import TechnoEconomicMixin

from rtctools_heat_network.workflows.goals.minimize_tco_goal import MinimizeTCO


class TargetDemandGoal(Goal):
    priority = 1

    order = 2

    def __init__(self, state, target):
        self.state = state

        self.target_min = target
        self.target_max = target
        self.function_range = (-1.0, 2.0 * max(target.values))
        self.function_nominal = np.median(target.values)

    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.state(self.state)


class MinimizeCostHeatGoal(Goal):
    priority = 2

    order = 1

    def __init__(self, source):
        self.target_max = 0.0
        self.function_range = (0.0, 10e8)
        self.source = source
        self.function_nominal = 1e7

    def function(self, optimization_problem, ensemble_member):
        return (
            optimization_problem.state(f"{self.source}.Heat_source")
            * optimization_problem.parameters(0)[
                f"{self.source}.variable_operational_cost_coefficient"]
        )


class _GoalsAndOptions:
    def path_goals(self):
        goals = super().path_goals().copy()

        for demand in self.heat_network_components.get("demand"):
            target = self.get_timeseries(f"{demand}.target_heat_demand")
            state = f"{demand}.Heat_demand"

            goals.append(TargetDemandGoal(state, target))

        for s in self.heat_network_components.get("source"):
            goals.append(MinimizeCostHeatGoal(s))
        # goals.append(MinimizeTCO)

        return goals

    def solver_options(self):
        options = super().solver_options()
        # options["solver"] = "gurobi"
        # gurobi_options = options["gurobi"] = {}
        # gurobi_options["MIPgap"] = 0.02

        return options


class HeatProblem(
    _GoalsAndOptions,
    TechnoEconomicMixin,
    LinearizedOrderGoalProgrammingMixin,
    SinglePassGoalProgrammingMixin,
    ESDLMixin,
    CollocatedIntegratedOptimizationProblem,
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def path_goals(self):
        goals = super().path_goals().copy()

        return goals

    def heat_network_options(self):
        options = super().heat_network_options()
        options["minimum_velocity"] = 0.0001
        return options

    def temperature_carriers(self):
        return self.esdl_carriers  # geeft terug de carriers met multiple temperature options

    def temperature_regimes(self, carrier):
        temperatures = []
        if carrier == 41770304791669983859190:
            # supply
            temperatures = [70.0, 50.0]

        return temperatures

    def constraints(self, ensemble_member):
        constraints = super().constraints(ensemble_member)

        for a in self.heat_network_components.get("ates", []):
            stored_heat = self.state_vector(f"{a}.Stored_heat")
            heat_ates = self.state_vector(f"{a}.Heat_ates")
            constraints.append((stored_heat[0] - stored_heat[-1], 0.0, 0.0))
            constraints.append((heat_ates[0], 0.0, 0.0))
            ates_temperature = self.state_vector(f"{a}.Temperature_ates")
            constraints.append(((ates_temperature[-1] - ates_temperature[0]), 0.0, np.inf))

        return constraints


if __name__ == "__main__":
    sol = run_optimization_problem(HeatProblem)
    results = sol.extract_results()
