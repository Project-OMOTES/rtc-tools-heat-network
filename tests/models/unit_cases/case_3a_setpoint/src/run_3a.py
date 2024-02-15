import numpy as np

from rtctools.optimization.collocated_integrated_optimization_problem import (
    CollocatedIntegratedOptimizationProblem,
)
from rtctools.optimization.goal_programming_mixin import Goal
from rtctools.optimization.linearized_order_goal_programming_mixin import (
    LinearizedOrderGoalProgrammingMixin,
)
from rtctools.optimization.single_pass_goal_programming_mixin import SinglePassGoalProgrammingMixin

from rtctools_heat_network.esdl.esdl_additional_vars_mixin import ESDLAdditionalVarsMixin
from rtctools_heat_network.esdl.esdl_mixin import ESDLMixin
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

    order = 1

    def __init__(self, source):
        self.target_max = 0.0
        self.function_range = (0.0, 10e6)
        self.source = source
        self.function_nominal = 1e6

    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.state(f"{self.source}.Heat_source")


class MinimizeSourcesFlowGoal(Goal):
    priority = 4

    order = 2

    def __init__(self, source, nominal=1.0):
        self.target_max = 0.0
        self.function_range = (0.0, 1.0e3)
        self.source = source
        self.function_nominal = nominal

    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.state(f"{self.source}.Q") * 1.0e3


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

        for demand in self.heat_network_components["demand"]:
            target = self.get_timeseries(f"{demand}.target_heat_demand")
            state = f"{demand}.Heat_demand"

            goals.append(TargetDemandGoal(state, target))

        # for s in self.heat_network_components["source"]:
        #     try:
        #         target_flow_rate = parameters[f"{s}.target_flow_rate"]
        #         goals.append(ConstantGeothermalSource(self, s, target_flow_rate))
        #     except KeyError:
        #         pass

        return goals

    def solver_options(self):
        options = super().solver_options()
        options["solver"] = "highs"
        # highs_options = options["highs"] = {}
        # highs_options["mip_rel_gap"] = 0.0025
        # options["gurobi"] = gurobi_options = {}
        # gurobi_options["MIPgap"] = 0.001
        return options

    def heat_network_options(self):
        options = super().heat_network_options()
        options["minimum_velocity"] = 0.0001
        # options["heat_loss_disconnected_pipe"] = False
        # options["neglect_pipe_heat_losses"] = False
        return options


class HeatProblem(
    _GoalsAndOptions,
    ESDLAdditionalVarsMixin,
    TechnoEconomicMixin,
    LinearizedOrderGoalProgrammingMixin,
    SinglePassGoalProgrammingMixin,
    ESDLMixin,
    CollocatedIntegratedOptimizationProblem,
):
    def path_goals(self):
        goals = super().path_goals().copy()

        for s in self.heat_network_components["source"]:
            goals.append(MinimizeSourcesHeatGoal(s))

        return goals

    def solver_options(self):
        options = super().solver_options()
        options["solver"] = "highs"
        # highs_options = options["highs"] = {}
        # highs_options["mip_rel_gap"] = 0.0025
        # options["gurobi"] = gurobi_options = {}
        # gurobi_options["MIPgap"] = 0.0001
        return options


if __name__ == "__main__":
    from rtctools.util import run_optimization_problem

    sol = run_optimization_problem(HeatProblem)

    results = sol.extract_results()
    print(results["GeothermalSource_b702.Heat_source"])

    a = 1
