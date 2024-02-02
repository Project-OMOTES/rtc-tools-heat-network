import datetime

import esdl

import numpy as np

from rtctools.data.storage import DataStore
from rtctools.optimization.collocated_integrated_optimization_problem import (
    CollocatedIntegratedOptimizationProblem,
)
from rtctools.optimization.goal_programming_mixin import Goal, GoalProgrammingMixin
from rtctools.optimization.linearized_order_goal_programming_mixin import (
    LinearizedOrderGoalProgrammingMixin,
)
from rtctools.optimization.single_pass_goal_programming_mixin import (
    CachingQPSol,
    SinglePassGoalProgrammingMixin,
)
from rtctools.util import run_optimization_problem

from rtctools_heat_network.esdl.esdl_mixin import ESDLMixin
from rtctools_heat_network.techno_economic_mixin import TechnoEconomicMixin

from rtctools_heat_network.workflows.goals.minimize_tco_goal import MinimizeTCO


class TargetDemandGoal(Goal):
    priority = 1

    order = 2

    def __init__(self, state, target):
        self.state = state

        self.target_min = target
        self.target_max = target
        self.function_range = (-1.0, 2.0 * max(target.values))
        self.function_nominal = np.median(target.values)

    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.state(self.state)


class MinimizeCostHeatGoal(Goal):
    priority = 2

    order = 1

    def __init__(self, source):
        self.target_max = 0.0
        self.function_range = (0.0, 10e4)
        self.source = source
        self.function_nominal = 1e3

    def function(self, optimization_problem, ensemble_member):
        try:
            state = optimization_problem.state(f"{self.source}.Heat_source")
        except KeyError:
            state = optimization_problem.state(f"{self.source}.Power_elec") #heatpumps are not yet in the variable_operational_costs in financial_mixin
        return (state
            * optimization_problem.parameters(0)[
                f"{self.source}.variable_operational_cost_coefficient"]
        )


class _GoalsAndOptions:
    def path_goals(self):
        goals = super().path_goals().copy()

        for demand in self.heat_network_components.get("demand"):
            target = self.get_timeseries(f"{demand}.target_heat_demand")
            state = f"{demand}.Heat_demand"

            goals.append(TargetDemandGoal(state, target))

        for s in [*self.heat_network_components.get("source"),
                  *self.heat_network_components.get("heat_pump")]:
            goals.append(MinimizeCostHeatGoal(s))
        # goals.append(MinimizeTCO)

        return goals

    def solver_options(self):
        options = super().solver_options()
        # options["solver"] = "highs"
        # highs_options = options["highs"] = {}
        # highs_options["mip_rel_gap"] = 0.05
        # options["solver"]="cbc"
        # cbc_options = options["cbc"] = {}
        # cbc_options["seconds"] = 300.0
        options["solver"] = "gurobi"
        gurobi_options = options["gurobi"] = {}
        gurobi_options["MIPgap"] = 0.01

        return options


class HeatProblem(
    _GoalsAndOptions,
    TechnoEconomicMixin,
    LinearizedOrderGoalProgrammingMixin,
    SinglePassGoalProgrammingMixin,
    ESDLMixin,
    CollocatedIntegratedOptimizationProblem,
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def path_goals(self):
        goals = super().path_goals().copy()

        return goals

    def heat_network_options(self):
        options = super().heat_network_options()
        options["minimum_velocity"] = 0.0001
        options["heat_loss_disconnected_pipe"] = False #required since we want to disconnect HP & HEX
        return options

    def temperature_carriers(self):
        return self.esdl_carriers

    def temperature_regimes(self, carrier):
        temperatures = []
        if carrier == 41770304791669983859190:
            # supply
            temperatures = [70.0, 55.0, 50.0, 45.0]#, 37.5, 35.0]

        return temperatures

    def path_constraints(self, ensemble_member):
        constraints = super().path_constraints(ensemble_member)

        #To prevent heat being consumer by hex to upgrade it (add heat) by heatpump to match demand without loading/unloading ates.
        sum_disabled_vars = 0
        for asset in [*self.heat_network_components.get("heat_pump"),
            *self.heat_network_components.get("heat_exchanger")]:
            disabled_var = self.state(f"{asset}__disabled")
            sum_disabled_vars +=disabled_var

        constraints.append((sum_disabled_vars, 1.0, 2.0))
        #
        # sum_disabled_vars = 0
        # for asset in [*self.heat_network_components.get("heat_pump"),
        #         *self.heat_network_components.get("heat_exchanger")]:
        #     disabled_var = self.state(f"{asset}__disabled")
        #     sum_disabled_vars += disabled_var
        #
        # constraints.append((1-sum_disabled_vars, 0.0, 0.0))

        #maybe replace by only required to have one disabled instead of at least one of them
        # e.g. 1-dis_hp-dis_hex ==0 (actually takes longer assumption because trying to balance heatlosses for which lower minimum velocity is required) instead of 1<=dis_hp+dis_hex<=2
        #when using compound asset instead of separate assets, one could still use this constraint
        # but potentially add the constraint that if hex is enabled, ates is loading and if hp is
        # enabled ates is unloading (dis_hex-ates_charging, 0.0, 0.0)

        return constraints

    def constraints(self, ensemble_member):
        constraints = super().constraints(ensemble_member)

        for a in self.heat_network_components.get("ates", []):
            stored_heat = self.state_vector(f"{a}.Stored_heat")
            heat_ates = self.state_vector(f"{a}.Heat_ates")
            constraints.append((stored_heat[0] - stored_heat[-1], 0.0, 0.0))
            constraints.append((heat_ates[0], 0.0, 0.0))
            ates_temperature = self.state_vector(f"{a}.Temperature_ates")
            constraints.append(((ates_temperature[-1] - ates_temperature[0]), 0.0, np.inf))

        return constraints


if __name__ == "__main__":
    sol = run_optimization_problem(HeatProblem)
    results = sol.extract_results()
