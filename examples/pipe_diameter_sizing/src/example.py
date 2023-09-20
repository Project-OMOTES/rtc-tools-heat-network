"""
This example shows a simplified example of an optimization problem for heat
network design. The goal is to optimize the selection of pipe diameters for
various segments of the heat network while considering multiple goals. Here's an
explanation of the problem that this code addresses:

1. **Heat Network Design**: The code optimizes the design of a
   heat network, which typically involves the transportation of hot water or
   steam to meet the heating demands of various consumers (e.g., buildings).

2. **Objective**: The primary objective of the optimization is to determine the
   appropriate optimal combination of pipe diameters for the heat network,
   considering both minimizing heat loss and meeting demand targets.

3. **Components**: - `Pipe Classes`: The code defines different classes of pipes
   with varying diameters, lengths, and thermal properties. These pipe classes
   represent the available options for constructing the heat network. - `Target
   Demand Goals`: The code considers the heating demands of various consumers
   and defines goals to ensure that the heat supplied matches the desired target
   demand. - `MinimizeLDGoal`: Another goal is to minimize heat losses in the
   system by selecting appropriate pipe diameters for different segments. -
   `Solver Options`: The code specifies options for the optimization solver,
   including parameters like maximum runtime.

4. **Constraints**: This example does not specify constraints. However, in
   some applications it is necessary to define constraints to represent
   physical priperties (fluid temperature, flow rates, etc.) or other
   limitations of the system.

"""

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
    """
    Represents an optimization problem for sizing pipe diameters in a heat
    network.

    Methods:
        heat_network_options(): Specifies heat network options.
        pipe_classes(pipe): Defines pipe classes with characteristics.
        path_goals(): Includes the goal related to the heat demand in the
        network. goals(): Includes the goal related to minimization of losses.
        priority_completed(priority): Called when a priority level of goals is
        completed to activate the hot start solver_options(): Specifies solver
        options, including hot start.

    """
    def heat_network_options(self):
        options = super().heat_network_options()
        options["minimum_velocity"] = 0.0
        return options

    def pipe_classes(self, pipe):
        return [
            PipeClass("None", 0.0, 0.0, (0.0, 0.0), 0.0),
            PipeClass("DN40", 0.0431, 1.5, (0.179091, 0.005049), 1.0),
            PipeClass("DN50", 0.0545, 1.7, (0.201377, 0.006086), 1.0),
            PipeClass("DN65", 0.0703, 1.9, (0.227114, 0.007300), 1.0),
            PipeClass("DN80", 0.0825, 2.2, (0.238244, 0.007611), 1.0),
            PipeClass("DN100", 0.1071, 2.4, (0.247804, 0.007386), 1.0),
            PipeClass("DN125", 0.1325, 2.6, (0.287779, 0.009431), 1.0),
            PipeClass("DN150", 0.1603, 2.8, (0.328592, 0.011567), 1.0),
            PipeClass("DN200", 0.2101, 3.0, (0.346285, 0.011215), 1.0),
            PipeClass("DN250", 0.263, 3.0, (0.334606, 0.009037), 1.0),
            PipeClass("DN300", 0.3127, 3.0, (0.384640, 0.011141), 1.0),
            PipeClass("DN350", 0.3444, 3.0, (0.368061, 0.009447), 1.0),
            PipeClass("DN400", 0.3938, 3.0, (0.381603, 0.009349), 1.0),
            PipeClass("DN450", 0.4444, 3.0, (0.380070, 0.008506), 1.0),
            PipeClass("DN500", 0.4954, 3.0, (0.369282, 0.007349), 1.0),
            PipeClass("DN600", 0.5954, 3.0, (0.431023, 0.009155), 1.0),
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
        goals.append(MinimizeLDGoal())
        return goals

    def priority_completed(self, priority):
        super().priority_completed(priority)
        self._hot_start = True

    def solver_options(self):
        options = super().solver_options()
        # options["solver"] = "gurobi"
        options["hot_start"] = getattr(self, "_hot_start", False)
        cbc_options = options["cbc"] = {}
        cbc_options["seconds"] = 500.0
        return options


class PipeDiameterSizingProblemTvar(PipeDiameterSizingProblem):
    """
    Represents an extension of PipeDiameterSizingProblem with temperature variations.

    Methods:
        temperature_carriers(): Define temperature carriers.
        temperature_regimes(carrier): Specify temperature regimes for carriers.
        solver_options(): Specify solver options, including hot start.

    """
    def temperature_carriers(self):
        return self.esdl_carriers  # geeft terug de carriers met multiple temperature options

    def solver_options(self):
        options = super().solver_options()
        options["hot_start"] = getattr(self, "_hot_start", True)
        return options

    def temperature_regimes(self, carrier):
        temperatures = []
        if carrier == 761602374459208051248:
            # supply
            temperatures = [70.0, 90.0]

        if carrier == 761602374459208051248000:
            # return
            temperatures = [30.0, 40.0]

        return temperatures

    # def constraints(self, ensemble_member):
    #     constraints = super().constraints(ensemble_member)
    #     # These constraints are added to allow for a quicker solve
    #     for carrier, temperatures in self.temperature_carriers().items():
    #         number_list = [int(s) for s in carrier if s.isdigit()]
    #         number = ""
    #         for nr in number_list:
    #             number = number + str(nr)
    #         carrier_type = temperatures["__rtc_type"]
    #         if carrier_type == "return":
    #             number = number + "000"
    #         carrier_id_number_mapping = number
    #         temperature_regimes = self.temperature_regimes(int(carrier_id_number_mapping))
    #         if len(temperature_regimes) > 0:
    #             for temperature in temperature_regimes:
    #                 selected_temp_vec = self.state_vector(
    #                     f"{int(carrier_id_number_mapping)}__{carrier_type}_{temperature}"
    #                 )
    #                 for i in range(1, len(self.times())):
    #                     constraints.append(
    #                         (selected_temp_vec[i] - selected_temp_vec[i - 1], 0.0, 0.0)
    #                     )
    #
    #     return constraints


if __name__ == "__main__":
    import time

    start_time = time.time()

    heat_problem = run_optimization_problem(PipeDiameterSizingProblem)

    print("Execution time: " + time.strftime("%M:%S", time.gmtime(time.time() - start_time)))
