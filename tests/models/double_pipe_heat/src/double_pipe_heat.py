from rtctools.optimization.collocated_integrated_optimization_problem import (
    CollocatedIntegratedOptimizationProblem,
)
from rtctools.optimization.csv_mixin import CSVMixin
from rtctools.optimization.goal_programming_mixin import Goal, GoalProgrammingMixin
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
        self.function_range = (0.0, 2e5)
        self.function_nominal = 1e5

    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.state("demand.Heat_demand")


class MinimizeProduction(Goal):

    priority = 2

    order = 1

    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.state("source.Heat_source")


class DoublePipeEqualHeat(
    HeatMixin,
    ModelicaComponentTypeMixin,
    GoalProgrammingMixin,
    CSVMixin,
    PyCMLMixin,
    CollocatedIntegratedOptimizationProblem,
):
    def __init__(self, *args, **kwargs):
        self.__model = Model()
        super().__init__(*args, **kwargs)

    def path_goals(self):
        return [TargetDemandGoal(self), MinimizeProduction()]

    def post(self):
        super().post()

    def pycml_model(self):
        return self.__model


if __name__ == "__main__":
    heat_problem = run_optimization_problem(DoublePipeEqualHeat)

    results = heat_problem.extract_results()
    times = heat_problem.times()

    directions = {}

    hot_pipes = [p for p in heat_problem.heat_network_components["pipe"] if p.endswith("_hot")]

    for p in hot_pipes:
        heat_in = results[p + ".HeatIn.Heat"]
        heat_out = results[p + ".HeatOut.Heat"]

        if not heat_problem.parameters(0)[p + ".disconnectable"]:
            # Flow direction is directly related to the sign of the heat
            direction_pipe = (heat_in >= 0.0).astype(int) * 2 - 1
        elif heat_problem.parameters(0)[p + ".disconnectable"]:
            direction_pipe = (heat_in >= 0.0).astype(int) * 2 - 1
            # Disconnect a pipe when the heat entering the component is only used
            # to account for its heat losses. There are three cases in which this
            # can happen.
            direction_pipe[((heat_in > 0.0) & (heat_out < 0.0))] = 0
            direction_pipe[((heat_in < 0.0) & (heat_out > 0.0))] = 0
            direction_pipe[((heat_in == 0.0) | (heat_out == 0.0))] = 0
        directions[p] = Timeseries(times, direction_pipe)

        # NOTE: The assumption is that the orientation of the cold pipes is such that the flow
        # is always in the same direction as its "hot" pipe companion.
        cold_pipe = f"{p[:-4]}_cold"
        directions[cold_pipe] = directions[p]

    directions_values = {k: v.values for k, v in directions.items()}

    a = 1
