import numpy as np

from rtctools.optimization.collocated_integrated_optimization_problem import (
    CollocatedIntegratedOptimizationProblem,
)
from rtctools.optimization.csv_mixin import CSVMixin
from rtctools.optimization.goal_programming_mixin import Goal, GoalProgrammingMixin
from rtctools.optimization.linearized_order_goal_programming_mixin import (
    LinearizedOrderGoalProgrammingMixin,
)
from rtctools.optimization.timeseries import Timeseries
from rtctools.util import run_optimization_problem

from rtctools_heat_network.heat_mixin import HeatMixin
from rtctools_heat_network.modelica_component_type_mixin import ModelicaComponentTypeMixin
from rtctools_heat_network.pycml.pycml_mixin import PyCMLMixin

if __name__ == "__main__":
    from model import Model
else:
    from .model import Model


class TargetDemandGoal(Goal):
    priority = 1

    order = 1

    def __init__(self, optimization_problem):
        self.target_min = optimization_problem.get_timeseries("Heat_demand")
        self.target_max = optimization_problem.get_timeseries("Heat_demand")
        self.function_range = (0.0, 2e6)
        self.function_nominal = 1e6

    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.state("demand.Heat_demand")


class MinimizeSourceGoal(Goal):
    priority = 2

    order = 1

    def __init__(self, optimization_problem):
        self.target_min = np.nan
        self.target_max = 0.0
        self.function_range = (0.0, 2e6)
        self.function_nominal = 1e6

    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.state("source.Heat_source")


class HeatBuffer(
    HeatMixin,
    ModelicaComponentTypeMixin,
    LinearizedOrderGoalProgrammingMixin,
    GoalProgrammingMixin,
    CSVMixin,
    PyCMLMixin,
    CollocatedIntegratedOptimizationProblem,
):
    def __init__(self, *args, **kwargs):
        self.__model = Model()
        super().__init__(*args, **kwargs)

    def path_goals(self):
        return [TargetDemandGoal(self), MinimizeSourceGoal(self)]

    def post(self):
        super().post()

    def solver_options(self):
        options = super().solver_options()
        options["solver"] = "highs"
        highs_options = options["highs"] = {}
        return options

    def pycml_model(self):
        return self.__model


class HeatBufferNoHistory(HeatBuffer):
    pass


class HeatBufferHistory(HeatBuffer):
    def history(self, ensemble_member):
        history = {}

        initial_time = np.array([self.initial_time])
        history["buffer.Heat_buffer"] = Timeseries(initial_time, [-12000.0])

        return history


class HeatBufferHistoryStoredHeat(HeatBuffer):
    def history(self, ensemble_member):
        history = {}

        initial_time = self.initial_time
        stored_heat_lb = self._HeatMixin__buffer_t0_bounds["buffer.Stored_heat"][0].values[0]
        history["buffer.Stored_heat"] = Timeseries(
            np.array([initial_time - 1, initial_time]),
            np.array([stored_heat_lb + 12000.0, stored_heat_lb]),
        )

        return history


if __name__ == "__main__":
    buffer_nohistory = run_optimization_problem(HeatBufferNoHistory)
    buffer_history = run_optimization_problem(HeatBufferHistory)
    buffer_historystoredheat = run_optimization_problem(HeatBufferHistoryStoredHeat)
