import numpy as np

from rtctools.optimization.collocated_integrated_optimization_problem import (
    CollocatedIntegratedOptimizationProblem,
)
from rtctools.optimization.goal_programming_mixin import Goal, GoalProgrammingMixin
from rtctools.optimization.homotopy_mixin import HomotopyMixin
from rtctools.optimization.linearized_order_goal_programming_mixin import (
    LinearizedOrderGoalProgrammingMixin,
)
from rtctools.optimization.single_pass_goal_programming_mixin import SinglePassGoalProgrammingMixin

from rtctools_heat_network.esdl.esdl_mixin import ESDLMixin
from rtctools_heat_network.heat_mixin import HeatMixin
from rtctools_heat_network.qth_mixin import QTHMixin


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
        parameters = self.parameters(0)

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


class HeatProblem(
    _GoalsAndOptions,
    HeatMixin,
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
        return options

    def heat_network_options(self):
        options = super().heat_network_options()
        options["minimum_velocity"] = 0.
        options["heat_loss_disconnected_pipe"] = False
        options["neglect_pipe_heat_losses"] = False
        return options


class HeatProblemSetPointConstraints(
    HeatMixin,
    LinearizedOrderGoalProgrammingMixin,
    SinglePassGoalProgrammingMixin,
    ESDLMixin,
    CollocatedIntegratedOptimizationProblem,
):
    def path_goals(self):
        goals = super().path_goals().copy()

        for demand in self.heat_network_components["demand"]:
            target = self.get_timeseries(f"{demand}.target_heat_demand")
            state = f"{demand}.Heat_demand"

            goals.append(TargetDemandGoal(state, target))

        for s in self.heat_network_components["source"]:
            goals.append(MinimizeSourcesHeatGoal(s))

        return goals


class HeatProblemTvarsup(
    HeatMixin,
    LinearizedOrderGoalProgrammingMixin,
    SinglePassGoalProgrammingMixin,
    ESDLMixin,
):
    def path_goals(self):
        goals = super().path_goals().copy()

        for demand in self.heat_network_components["demand"]:
            target = self.get_timeseries(f"{demand}.target_heat_demand")
            state = f"{demand}.Heat_demand"

            goals.append(TargetDemandGoal(state, target))

        for s in self.heat_network_components["source"]:
            goals.append(MinimizeSourcesHeatGoal(s))

        return goals

    def temperature_carriers(self):
        return self.esdl_carriers  # geeft terug de carriers met multiple temperature options

    def temperature_regimes(self, carrier):
        temperatures = []
        if carrier == 4195016129475469474608:
            # supply
            temperatures = [80.0, 120.0]

        return temperatures

    def times(self, variable=None):
        times = super().times(variable)
        return times[:2]

    def constraints(self, ensemble_member):
        constraints = super().constraints(ensemble_member)
        # These constraints are added to allow for a quicker solve
        for carrier, temperatures in self.temperature_carriers().items():
            number_list = [int(s) for s in carrier if s.isdigit()]
            number = ""
            for nr in number_list:
                number = number + str(nr)
            carrier_type = temperatures["__rtc_type"]
            if carrier_type == "return":
                number = number + "000"
            carrier_id_number_mapping = number
            temperature_regimes = self.temperature_regimes(int(carrier_id_number_mapping))
            if len(temperature_regimes) > 0:
                for temperature in temperature_regimes:
                    selected_temp_vec = self.state_vector(
                        f"{int(carrier_id_number_mapping)}__{carrier_type}_{temperature}"
                    )
                    for i in range(1, len(self.times())):
                        constraints.append(
                            (selected_temp_vec[i] - selected_temp_vec[i - 1], 0.0, 0.0)
                        )

        return constraints


class HeatProblemTvarret(
    HeatMixin,
    LinearizedOrderGoalProgrammingMixin,
    SinglePassGoalProgrammingMixin,
    ESDLMixin,
    CollocatedIntegratedOptimizationProblem,
):
    def path_goals(self):
        goals = super().path_goals().copy()

        for demand in self.heat_network_components["demand"]:
            target = self.get_timeseries(f"{demand}.target_heat_demand")
            state = f"{demand}.Heat_demand"

            goals.append(TargetDemandGoal(state, target))

        for s in self.heat_network_components["source"]:
            goals.append(MinimizeSourcesFlowGoal(s))

        return goals

    def temperature_carriers(self):
        return self.esdl_carriers  # geeft terug de carriers met multiple temperature options

    def temperature_regimes(self, carrier):
        temperatures = []
        if carrier == 4195016129475469474608000:
            # return
            temperatures = [30.0, 40.0]

        return temperatures

    def times(self, variable=None):
        times = super().times(variable)
        return times[:2]

    def constraints(self, ensemble_member):
        constraints = super().constraints(ensemble_member)
        # These constraints are added to allow for a quicker solve
        for carrier, temperatures in self.temperature_carriers().items():
            number_list = [int(s) for s in carrier if s.isdigit()]
            number = ""
            for nr in number_list:
                number = number + str(nr)
            carrier_type = temperatures["__rtc_type"]
            if carrier_type == "return":
                number = number + "000"
            carrier_id_number_mapping = number
            temperature_regimes = self.temperature_regimes(int(carrier_id_number_mapping))
            if len(temperature_regimes) > 0:
                for temperature in temperature_regimes:
                    selected_temp_vec = self.state_vector(
                        f"{int(carrier_id_number_mapping)}__{carrier_type}_{temperature}"
                    )
                    for i in range(1, len(self.times())):
                        constraints.append(
                            (selected_temp_vec[i] - selected_temp_vec[i - 1], 0.0, 0.0)
                        )

        return constraints


class HeatProblemProdProfile(
    HeatMixin,
    LinearizedOrderGoalProgrammingMixin,
    SinglePassGoalProgrammingMixin,
    ESDLMixin,
    CollocatedIntegratedOptimizationProblem,
):
    def read(self):
        super().read()

        for s in self.heat_network_components["source"]:
            demand_timeseries = self.get_timeseries("HeatingDemand_a3b8.target_heat_demand")
            new_timeseries = np.ones(len(demand_timeseries.values)) * 1
            ind_hlf = int(len(demand_timeseries.values) / 2)
            new_timeseries[ind_hlf : ind_hlf + 4] = np.ones(4) * 0.05
            self.set_timeseries(f"{s}.target_heat_source", new_timeseries)

    def path_goals(self):
        goals = super().path_goals().copy()

        for demand in self.heat_network_components["demand"]:
            target = self.get_timeseries(f"{demand}.target_heat_demand")
            state = f"{demand}.Heat_demand"

            goals.append(TargetDemandGoal(state, target))

        for s in self.heat_network_components["source"]:
            goals.append(MinimizeSourcesHeatGoal(s))

        return goals


class QTHProblem(
    _GoalsAndOptions,
    QTHMixin,
    HomotopyMixin,
    SinglePassGoalProgrammingMixin,
    ESDLMixin,
    CollocatedIntegratedOptimizationProblem,
):
    def path_goals(self):
        goals = super().path_goals().copy()

        for s in self.heat_network_components["source"]:
            goals.append(MinimizeSourcesQTHGoal(s))

        return goals

    def heat_network_options(self):
        options = super().heat_network_options()
        from rtctools_heat_network.head_loss_mixin import HeadLossOption

        options["head_loss_option"] = HeadLossOption.NO_HEADLOSS
        return options


if __name__ == "__main__":
    from rtctools.util import run_optimization_problem

    sol = run_optimization_problem(HeatProblem)
    results = sol.extract_results()
    a = 1
    # run_heat_network_optimization(HeatProblem, QTHProblem)
