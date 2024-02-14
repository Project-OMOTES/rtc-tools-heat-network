import numpy as np

from rtctools.data.storage import DataStore
from rtctools.optimization.collocated_integrated_optimization_problem import (
    CollocatedIntegratedOptimizationProblem,
)
from rtctools.optimization.goal_programming_mixin import Goal
from rtctools.optimization.linearized_order_goal_programming_mixin import (
    LinearizedOrderGoalProgrammingMixin,
)
from rtctools.optimization.single_pass_goal_programming_mixin import (
    SinglePassGoalProgrammingMixin, GoalProgrammingMixin
)
from rtctools.util import run_optimization_problem

from rtctools_heat_network.esdl.esdl_mixin import ESDLMixin
from rtctools_heat_network.head_loss_class import HeadLossOption
from rtctools_heat_network.techno_economic_mixin import TechnoEconomicMixin


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
        self.function_range = (0.0, 1.e4)
        self.source = source
        self.function_nominal = 1e0

    def function(self, optimization_problem, ensemble_member):
        try:
            state = optimization_problem.state(f"{self.source}.Heat_source")
        except KeyError:
            state = optimization_problem.state(
                f"{self.source}.Power_elec"
            )  # heatpumps are not yet in the variable_operational_costs in financial_mixin
        return (
            state
            * optimization_problem.parameters(0)[
                f"{self.source}.variable_operational_cost_coefficient"
            ]
        )


class MinimizeATESTemperature(Goal):
    priority = 3

    order = 1

    def __init__(self, ates):
        self.target_max = 0.
        self.function_range = (0., 2.e3)
        self.ates = ates
        # self.function_nominal = 1e0

    def function(self, optimization_problem, ensemble_member):
        # return optimization_problem.state(f"{self.ates}.Temperature_ates") - optimization_problem.state(f"{self.ates}__temperature_ates_disc")
        return optimization_problem.state(f"{self.ates}__temperature_ates_disc")
        # return optimization_problem.state(f"{self.ates}.Temperature_ates")


class _GoalsAndOptions:
    def path_goals(self):
        goals = super().path_goals().copy()

        for demand in self.heat_network_components.get("demand"):
            target = self.get_timeseries(f"{demand}.target_heat_demand")
            state = f"{demand}.Heat_demand"

            goals.append(TargetDemandGoal(state, target))

        for s in [
            *self.heat_network_components.get("source"),
            *self.heat_network_components.get("heat_pump"),
        ]:
            goals.append(MinimizeCostHeatGoal(s))

        # for ates in self.heat_network_components.get("ates", []):
        #     goals.append(MinimizeATESTemperature(ates))

        return goals

    def solver_options(self):
        options = super().solver_options()
        # options["solver"] = "highs"
        # highs_options = options["highs"] = {}
        # highs_options["mip_rel_gap"] = 0.01
        options["solver"] = "gurobi"
        gurobi_options = options["gurobi"] = {}
        gurobi_options["MIPgap"] = 0.1
        # gurobi_options["OptimalityTol"] = 1.e-3

        return options


class HeatProblem(
    _GoalsAndOptions,
    TechnoEconomicMixin,
    LinearizedOrderGoalProgrammingMixin,
    GoalProgrammingMixin,
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
        options["heat_loss_disconnected_pipe"] = (
            False  # required since we want to disconnect HP & HEX
        )
        options["head_loss_option"] = HeadLossOption.NO_HEADLOSS
        options["neglect_pipe_heat_losses"] = True
        return options

    def temperature_carriers(self):
        return self.esdl_carriers

    def temperature_regimes(self, carrier):
        temperatures = []
        if carrier == 41770304791669983859190:
            # supply
            # temperatures = np.linspace(50, 70, 9).tolist()[::-1]
            # temperatures = np.linspace(52.5, 65, 6).tolist()[::-1]
            # temperatures.extend(np.linspace(45, 50, 6).tolist()[::-1])

            temperatures = np.linspace(40, 60, 9).tolist()[::-1]

        return temperatures

    def path_constraints(self, ensemble_member):
        constraints = super().path_constraints(ensemble_member)

        # To prevent heat being consumer by hex to upgrade it (add heat) by heatpump to match
        # demand without loading/unloading ates.
        sum_disabled_vars = 0
        for asset in [
            *self.heat_network_components.get("heat_pump"),
            *self.heat_network_components.get("heat_exchanger"),
        ]:
            disabled_var = self.state(f"{asset}__disabled")
            sum_disabled_vars += disabled_var

        constraints.append((sum_disabled_vars, 1.0, 2.0))

        # when using compound asset instead of separate assets, one could still use this constraint
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
            ates_temperature = self.__state_vector_scaled(f"{a}.Temperature_ates", ensemble_member)
            ates_temperature_disc = self.__state_vector_scaled(f"{a}__temperature_ates_disc", ensemble_member)
            constraints.append(((ates_temperature_disc[-1] - ates_temperature_disc[0]), 0.0, 0.0))
            # constraints.append(((ates_temperature_disc[-1] - ates_temperature_disc[0]), 0.0, 0.0))
            # constraints.append(((ates_temperature_disc[-1] - ates_temperature[-1]), 0.0, 0.0))
            # constraints.append(((ates_temperature_disc[0] - ates_temperature[0]), 0.0, 0.0))


        return constraints

    def __state_vector_scaled(self, variable, ensemble_member):
        """
        This functions returns the casadi symbols scaled with their nominal for the entire time
        horizon.
        """
        canonical, sign = self.alias_relation.canonical_signed(variable)
        return (
            self.state_vector(canonical, ensemble_member) * self.variable_nominal(canonical) * sign
        )

    def read(self):
        """
        Reads the yearly profile with hourly time steps and adapt to a 5 day averaged profile.
        """
        super().read()

        demands = self.heat_network_components.get("demand", [])
        new_datastore = DataStore(self)
        new_datastore.reference_datetime = self.io.datetimes[0]

        for ensemble_member in range(self.ensemble_size):
            total_demand = sum(
                self.get_timeseries(f"{demand}.target_heat_demand", ensemble_member).values
                for demand in demands
            )

            # TODO: the approach of picking one peak day was introduced for a network with a tree
            #  layout and all big sources situated at the root of the tree. It is not guaranteed
            #  that an optimal solution is reached in different network topologies.
            nr_of_days = len(total_demand) // (24 * 5)
            new_date_times = list()
            for day in range(0, nr_of_days):
                new_date_times.append(self.io.datetimes[day * 24 * 5])
            new_date_times = np.asarray(new_date_times)

            for demand in demands:
                var_name = f"{demand}.target_heat_demand"
                data = self.get_timeseries(
                    variable=var_name, ensemble_member=ensemble_member
                ).values
                new_data = list()
                for day in range(0, nr_of_days):
                    data_for_day = data[day * 24 * 5 : (day + 1) * 24 * 5]
                    new_data.append(np.mean(data_for_day))
                new_datastore.set_timeseries(
                    variable=var_name,
                    datetimes=new_date_times,
                    values=np.asarray(new_data) * 2.,
                    ensemble_member=ensemble_member,
                    check_duplicates=True,
                )

            self.io = new_datastore


class HeatProblemMaxFlow(HeatProblem):

    def read(self):
        super().read()

        demand_timeseries = self.get_timeseries("HeatingDemand_1.target_heat_demand")
        demand_timeseries.values[2] = demand_timeseries.values[2] * 2
        self.set_timeseries("HeatingDemand_1.target_heat_demand", demand_timeseries)


if __name__ == "__main__":
    import time
    t0 = time.time()

    sol = run_optimization_problem(HeatProblem)
    results = sol.extract_results()
    print("T_ates: ", results["ATES_cb47.Temperature_ates"])
    print("T_ates_disc: ", results["ATES_cb47__temperature_ates_disc"])
    print("T_ates: ", results["ATES_cb47__temperature_disc_40.0"])
    print(f"time: {time.time() - t0}")
    a = 1


