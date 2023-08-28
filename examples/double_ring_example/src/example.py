import time

import numpy as np

from rtctools.optimization.collocated_integrated_optimization_problem import (
    CollocatedIntegratedOptimizationProblem,
)
from rtctools.optimization.csv_mixin import CSVMixin
from rtctools.optimization.goal_programming_mixin import Goal, GoalProgrammingMixin
from rtctools.optimization.homotopy_mixin import HomotopyMixin
from rtctools.optimization.linearized_order_goal_programming_mixin import (
    LinearizedOrderGoalProgrammingMixin,
)
from rtctools.optimization.modelica_mixin import ModelicaMixin
from rtctools.optimization.timeseries import Timeseries

from rtctools_heat_network.heat_mixin import HeatMixin
from rtctools_heat_network.modelica_component_type_mixin import ModelicaComponentTypeMixin
from rtctools_heat_network.qth_mixin import QTHMixin
from rtctools_heat_network.util import run_heat_network_optimization


class RangeGoal(Goal):
    def __init__(
        self,
        optimization_problem,
        state,
        target_min,
        target_max,
        priority,
        state_bounds=None,
        order=1,
        weight=1.0,
    ):
        self.state = state
        self.target_min = target_min
        self.target_max = target_max
        self.priority = priority
        self.order = order
        self.weight = weight
        if state_bounds is None:
            state_bounds = optimization_problem.bounds()[state]
        self.function_range = state_bounds
        self.function_nominal = max((abs(state_bounds[1]) + abs(state_bounds[0])) / 2.0, 1.0)

    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.state(self.state)


class MinimizeSource(Goal):
    def __init__(
        self,
        optimization_problem,
        state,
        target_min,
        target_max,
        priority,
        state_bounds=None,
        order=2,
        weight=1.0,
    ):
        self.state = state
        self.target_min = target_min
        self.target_max = target_max
        self.priority = priority
        self.order = order
        self.weight = weight
        if state_bounds is None:
            state_bounds = (
                optimization_problem.bounds()[state][0] * len(optimization_problem.times()),
                optimization_problem.bounds()[state][1] * len(optimization_problem.times()),
            )
        self.function_range = state_bounds
        self.function_nominal = max((abs(state_bounds[1]) + abs(state_bounds[0])) / 2.0, 1.0)

    def function(self, optimization_problem, ensemble_member):
        t = optimization_problem.times()
        sum_over_t = 0.0
        for i in range(len(t)):
            sum_over_t += optimization_problem.state_at(self.state, t[i])
        return sum_over_t


class GoalsAndOptions:
    def heat_network_options(self):
        options = super().heat_network_options()
        options["maximum_temperature_der"] = 1.5
        options["maximum_flow_der"] = 0.001
        return options

    def pre(self):
        super().pre()

        tot_demand = 0.0
        for d in self.heat_network_components["demand"]:
            ts = self.get_timeseries("Heat_demand_" + d[-1])
            dem = ts.values
            dem /= 10
            self.set_timeseries("Heat_demand_" + d[-1], Timeseries(ts.times, dem))
            tot_demand += np.mean(dem)

    def path_goals(self):
        goals = super().path_goals()

        # Goal 1: Match the demand target heat
        for d in self.heat_network_components["demand"]:
            k = d[6:]
            var = d + ".Heat_demand"
            target_heat = self.get_timeseries(f"Heat_demand_{k}")
            lb = min(target_heat.values) * 0.5
            ub = max(target_heat.values) * 1.5

            goals.append(
                RangeGoal(
                    self,
                    state=var,
                    target_min=target_heat,
                    target_max=target_heat,
                    priority=1,
                    state_bounds=(lb, ub),
                )
            )

        return goals

    def goals(self):
        goals = super().goals()

        # Goal 2: Minimize usage of all sources
        # Effectively we are minimizing: sum_s[sum_t(Heat(t,s))^2], i.e, minimize the square
        # of the sum of source produced by each source and loop over all the sources.
        # Thus the optimization is unbiased (over the all time horizon) wrt to the source.
        for s in self.heat_network_components["source"]:
            goals.append(
                MinimizeSource(
                    self,
                    target_min=np.nan,
                    target_max=0.0,
                    state=s + ".Heat_source",
                    priority=2,
                    order=2,
                    weight=1.0,
                )
            )

        return goals


class HeatProblem(
    GoalsAndOptions,
    HeatMixin,
    ModelicaComponentTypeMixin,
    LinearizedOrderGoalProgrammingMixin,
    GoalProgrammingMixin,
    CSVMixin,
    ModelicaMixin,
    CollocatedIntegratedOptimizationProblem,
):
    model_name = "Westland_Heat"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__hot_start = False

    def solver_options(self):
        options = super().solver_options()
        options["hot_start"] = self.__hot_start
        return options

    def priority_completed(self, priority):
        super().priority_completed(priority)
        self.__hot_start = True


class QTHProblem(
    GoalsAndOptions,
    QTHMixin,
    ModelicaComponentTypeMixin,
    HomotopyMixin,
    GoalProgrammingMixin,
    CSVMixin,
    ModelicaMixin,
    CollocatedIntegratedOptimizationProblem,
):
    model_name = "Westland_QTH"

    def goal_programming_options(self):
        options = super().goal_programming_options()

        if self.parameters(0)["theta"] > 0.0:
            options["constraint_relaxation"] = 1e-4

        return options

    def heat_network_options(self):
        options = super().heat_network_options()

        options["minimum_pressure_far_point"] = 0.0
        # Note: True also works
        options["max_t_der_bidirect_pipe"] = False

        return options

    def solver_options(self):
        options = super().solver_options()

        options["expand"] = True

        solver = options["solver"]
        options[solver]["max_iter"] = 500
        options[solver]["tol"] = 1e-5

        options[solver]["acceptable_tol"] = 1e-5
        options[solver]["nlp_scaling_method"] = "none"
        options[solver]["linear_system_scaling"] = "none"
        options[solver]["linear_scaling_on_demand"] = "no"

        return options


if __name__ == "__main__":
    start_time = time.time()
    heat_problem, qth_problem = run_heat_network_optimization(HeatProblem, QTHProblem)
    print("Execution time: " + time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
