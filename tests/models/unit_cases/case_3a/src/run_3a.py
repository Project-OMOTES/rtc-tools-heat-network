import numpy as np

from rtctools.optimization.collocated_integrated_optimization_problem import (
    CollocatedIntegratedOptimizationProblem,
)
from rtctools.optimization.goal_programming_mixin import Goal, GoalProgrammingMixin
from rtctools.optimization.homotopy_mixin import HomotopyMixin
from rtctools.optimization.linearized_order_goal_programming_mixin import (
    LinearizedOrderGoalProgrammingMixin,
)

from rtctools_heat_network.esdl.esdl_mixin import ESDLMixin
from rtctools_heat_network.heat_mixin import HeatMixin
from rtctools_heat_network.qth_mixin import QTHMixin
from rtctools_heat_network.util import run_heat_network_optimization


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


class ConstantGeothermalSource(Goal):

    priority = 2

    order = 2

    def __init__(self, optimization_problem, source, target, lower_fac=0.9, upper_fac=1.1):
        self.target_min = lower_fac * target
        self.target_max = upper_fac * target
        self.function_range = (0.0, 2.0 * target)
        self.state = f"{source}.Q"
        self.function_nominal = optimization_problem.variable_nominal(self.state)

    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.state(self.state)


class MinimizeSourcesHeatGoal(Goal):

    priority = 3

    order = 2

    def __init__(self, source):
        self.target_max = 0.0
        self.function_range = (0.0, 10e6)
        self.source = source
        self.function_nominal = 1e6

    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.state(f"{self.source}.Heat_source")


class MinimizeSourcesQTHGoal(Goal):

    priority = 3

    order = 2

    def __init__(self, source):
        self.source = source
        self.function_nominal = 1e6

    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.state(f"{self.source}.Heat_source")


class _GoalsAndOptions:
    def path_goals(self):
        goals = super().path_goals().copy()
        parameters = self.parameters(0)

        for demand in self.heat_network_components["demand"]:
            target = self.get_timeseries(f"{demand}.target_heat_demand")
            state = f"{demand}.Heat_demand"

            goals.append(TargetDemandGoal(state, target))

        for s in self.heat_network_components["source"]:
            try:
                target_flow_rate = parameters[f"{s}.target_flow_rate"]
                goals.append(ConstantGeothermalSource(self, s, target_flow_rate))
            except KeyError:
                pass

        return goals


class HeatProblem(
    _GoalsAndOptions,
    HeatMixin,
    LinearizedOrderGoalProgrammingMixin,
    GoalProgrammingMixin,
    ESDLMixin,
    CollocatedIntegratedOptimizationProblem,
):
    def path_goals(self):
        goals = super().path_goals().copy()

        for s in self.heat_network_components["source"]:
            goals.append(MinimizeSourcesHeatGoal(s))

        return goals


class QTHProblem(
    _GoalsAndOptions,
    QTHMixin,
    HomotopyMixin,
    GoalProgrammingMixin,
    ESDLMixin,
    CollocatedIntegratedOptimizationProblem,
):
    def path_goals(self):
        goals = super().path_goals().copy()

        for s in self.heat_network_components["source"]:
            goals.append(MinimizeSourcesQTHGoal(s))

        return goals


if __name__ == "__main__":
    run_heat_network_optimization(HeatProblem, QTHProblem)
