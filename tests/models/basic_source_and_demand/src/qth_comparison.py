from rtctools.optimization.collocated_integrated_optimization_problem import (
    CollocatedIntegratedOptimizationProblem,
)
from rtctools.optimization.csv_mixin import CSVMixin
from rtctools.optimization.goal_programming_mixin import Goal, GoalProgrammingMixin
from rtctools.optimization.homotopy_mixin import HomotopyMixin
from rtctools.optimization.modelica_mixin import ModelicaMixin
from rtctools.util import run_optimization_problem

from rtctools_heat_network.bounds_to_pipe_flow_directions_mixin import (
    BoundsToPipeFlowDirectionsMixin,
)
from rtctools_heat_network.modelica_component_type_mixin import ModelicaComponentTypeMixin
from rtctools_heat_network.pycml.pycml_mixin import PyCMLMixin
from rtctools_heat_network.qth_mixin import QTHMixin

if __name__ == "__main__":
    from model_qth import Model
else:
    from .model_qth import Model


class TargetDemandGoal(Goal):

    priority = 1

    order = 1

    def __init__(self, optimization_problem):
        self.target_min = optimization_problem.get_timeseries("Heat_demand")
        self.target_max = optimization_problem.get_timeseries("Heat_demand")
        self.function_range = (0.0, 2e5)
        self.function_nominal = 1e5

    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.state("demand.Heat_demand")


class _GoalsAndOptions:
    def path_goals(self):
        goals = super().path_goals().copy()
        goals.append(TargetDemandGoal(self))
        return goals

    def priority_completed(self, priority):
        super().priority_completed(priority)

        if not hasattr(self, "_objective_values"):
            self._objective_values = []
        self._objective_values.append(self.objective_value)


class QTHModelica(
    _GoalsAndOptions,
    BoundsToPipeFlowDirectionsMixin,
    QTHMixin,
    ModelicaComponentTypeMixin,
    HomotopyMixin,
    GoalProgrammingMixin,
    CSVMixin,
    ModelicaMixin,
    CollocatedIntegratedOptimizationProblem,
):
    pass


class QTHPython(
    _GoalsAndOptions,
    BoundsToPipeFlowDirectionsMixin,
    QTHMixin,
    ModelicaComponentTypeMixin,
    HomotopyMixin,
    GoalProgrammingMixin,
    CSVMixin,
    PyCMLMixin,
    CollocatedIntegratedOptimizationProblem,
):
    def __init__(self, *args, **kwargs):
        self.__model = Model()
        super().__init__(*args, **kwargs)

    def pycml_model(self):
        return self.__model


if __name__ == "__main__":
    a = run_optimization_problem(QTHModelica)
    b = run_optimization_problem(QTHPython)
