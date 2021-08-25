import numpy as np

from rtctools.optimization.collocated_integrated_optimization_problem import (
    CollocatedIntegratedOptimizationProblem,
)
from rtctools.optimization.goal_programming_mixin import (
    Goal,
    GoalProgrammingMixin,
)
from rtctools.optimization.linearized_order_goal_programming_mixin import (
    LinearizedOrderGoalProgrammingMixin,
)
from rtctools.util import run_optimization_problem

from rtctools_heat_network.esdl.esdl_mixin import ESDLMixin
from rtctools_heat_network.heat_mixin import HeatMixin
from rtctools_heat_network.pipe_class import PipeClass


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


class MinimizeLDGoal(Goal):

    priority = 2

    order = 1

    def __init__(self, source):
        self.source = source

    def function(self, optimization_problem, ensemble_member):
        obj = 0.0
        parameters = optimization_problem.parameters(ensemble_member)
        nominal = 0.0

        for p in optimization_problem.hot_pipes:
            length = parameters[f"{p}.length"]
            var_name = optimization_problem.pipe_diameter_symbol_name(p)

            nominal += length * optimization_problem.variable_nominal(var_name)

            obj += optimization_problem.extra_variable(var_name, ensemble_member) * length

        return obj / nominal


class PipeDiameterSizingProblem(
    HeatMixin,
    LinearizedOrderGoalProgrammingMixin,
    GoalProgrammingMixin,
    ESDLMixin,
    CollocatedIntegratedOptimizationProblem,
):
    def heat_network_options(self):
        options = super().heat_network_options()

        options["minimum_velocity"] = 0.0
        options["neglect_pipe_heat_losses"] = True

        return options

    def pipe_classes(self, pipe):
        return [
            PipeClass("None", 0.0, 0.0),
            PipeClass("DN40", 0.0431, 1.5),
            PipeClass("DN50", 0.0545, 1.7),
            PipeClass("DN65", 0.0703, 1.9),
            PipeClass("DN80", 0.0825, 2.2),
            PipeClass("DN100", 0.1071, 2.4),
            PipeClass("DN125", 0.1325, 2.6),
            PipeClass("DN150", 0.1603, 2.8),
            PipeClass("DN200", 0.2101, 3.0),
            PipeClass("DN250", 0.263, 3.0),
            PipeClass("DN300", 0.3127, 3.0),
            PipeClass("DN350", 0.3444, 3.0),
            PipeClass("DN400", 0.3938, 3.0),
            PipeClass("DN450", 0.4444, 3.0),
            PipeClass("DN500", 0.4954, 3.0),
            PipeClass("DN600", 0.5954, 3.0),
        ]

    def path_goals(self):
        goals = super().path_goals().copy()

        for demand in self.heat_network_components["demand"]:
            target = self.get_timeseries(f"{demand}.target_heat_demand")
            state = f"{demand}.Heat_demand"

            goals.append(TargetDemandGoal(state, target))

        return goals

    def goals(self):
        goals = super().goals().copy()
        goals.append(MinimizeLDGoal(self))
        return goals

    def path_constraints(self, ensemble_member):
        constraints = super().path_constraints(ensemble_member)

        # Apparently there is freedom on the cold side, which results in
        # warnings. We force the cold pipes to have zero heat at all times.
        for p in self.cold_pipes:
            constraints.append(
                (self.state(f"{p}.Heat_in") / self.variable_nominal(f"{p}.Heat_in"), 0.0, 0.0)
            )
            constraints.append(
                (self.state(f"{p}.Heat_out") / self.variable_nominal(f"{p}.Heat_out"), 0.0, 0.0)
            )

        return constraints

    def priority_completed(self, priority):
        super().priority_completed(priority)
        self._hot_start = True

    def solver_options(self):
        options = super().solver_options()
        options["hot_start"] = getattr(self, "_hot_start", False)
        return options


if __name__ == "__main__":
    import time

    start_time = time.time()

    heat_problem = run_optimization_problem(PipeDiameterSizingProblem)

    print("Execution time: " + time.strftime("%M:%S", time.gmtime(time.time() - start_time)))
