from rtctools.optimization.collocated_integrated_optimization_problem import (
    CollocatedIntegratedOptimizationProblem,
)
from rtctools.optimization.goal_programming_mixin import Goal
from rtctools.optimization.linearized_order_goal_programming_mixin import (
    LinearizedOrderGoalProgrammingMixin,
)
from rtctools.optimization.single_pass_goal_programming_mixin import SinglePassGoalProgrammingMixin
from rtctools.util import run_optimization_problem

from rtctools_heat_network.esdl.esdl_additional_vars_mixin import ESDLAdditionalVarsMixin
from rtctools_heat_network.esdl.esdl_mixin import ESDLMixin
from rtctools_heat_network.esdl.esdl_parser import ESDLFileParser
from rtctools_heat_network.esdl.profile_parser import ProfileReaderFromFile
from rtctools_heat_network.head_loss_class import HeadLossOption
from rtctools_heat_network.techno_economic_mixin import TechnoEconomicMixin


class TargetDemandGoal(Goal):
    priority = 1

    order = 2

    def __init__(self, optimization_problem):
        self.target_min = optimization_problem.get_timeseries("demand.target_heat_demand")
        self.target_max = optimization_problem.get_timeseries("demand.target_heat_demand")
        self.function_range = (0.0, 2e5)
        self.function_nominal = 1e5

    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.state("demand.Heat_demand")


class MinimizeProduction(Goal):
    priority = 2

    order = 1

    def __init__(self):
        self.function_nominal = 1e6

    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.state("source.Heat_source")


class SourcePipeSink(
    TechnoEconomicMixin,
    LinearizedOrderGoalProgrammingMixin,
    SinglePassGoalProgrammingMixin,
    ESDLMixin,
    CollocatedIntegratedOptimizationProblem,
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def path_goals(self):
        g = super().path_goals().copy()
        g.append(TargetDemandGoal(self))
        g.append(MinimizeProduction())
        return g

    def post(self):
        super().post()


class HeatProblemHydraulic(ESDLAdditionalVarsMixin, SourcePipeSink):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.heat_network_settings["head_loss_option"] = (
            HeadLossOption.LINEARIZED_N_LINES_WEAK_INEQUALITY
        )
        self.heat_network_settings["n_linearization_lines"] = 5
        self.heat_network_settings["minimize_head_losses"] = True

    def heat_network_options(self):
        options = super().heat_network_options()

        return options

    def solver_options(self):
        options = super().solver_options()
        # options["solver"] = "gurobi"

        return options


if __name__ == "__main__":
    sol = run_optimization_problem(
        HeatProblemHydraulic,
        esdl_file_name="sourcesink.esdl",
        esdl_parser=ESDLFileParser,
        profile_reader=ProfileReaderFromFile,
        input_timeseries_file="timeseries_import.csv",
    )
    results = sol.extract_results()
    a = 1
