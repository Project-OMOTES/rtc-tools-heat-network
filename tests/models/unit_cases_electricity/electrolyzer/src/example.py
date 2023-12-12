import numpy as np

from rtctools.optimization.collocated_integrated_optimization_problem import (
    CollocatedIntegratedOptimizationProblem,
)
from rtctools.optimization.goal_programming_mixin import GoalProgrammingMixin
from rtctools.optimization.goal_programming_mixin_base import Goal
from rtctools.optimization.linearized_order_goal_programming_mixin import (
    LinearizedOrderGoalProgrammingMixin,
)
from rtctools.util import run_optimization_problem

from rtctools_heat_network.esdl.esdl_mixin import ESDLMixin
from rtctools_heat_network.heat_mixin import HeatMixin


class RevenueGoal(Goal):
    priority = 1

    order = 1

    def __init__(self, state, price_profile, nominal):
        self.state = state

        self.price_profile = price_profile
        self.function_nominal = nominal

    def function(self, optimization_problem, ensemble_member):
        canonical, sign = optimization_problem.alias_relation.canonical_signed(self.state)
        symbols = (
            sign * optimization_problem.state_vector(canonical, ensemble_member)
            * optimization_problem.variable_nominal(self.state)
        )
        price_profile = optimization_problem.get_timeseries(self.price_profile).values
        sum = 0.
        for i in range(len(price_profile)):
            sum += symbols[i] * price_profile[i]

        # TODO: subtract the costs
        # for asset in []:
        #     sum -= optimization_problem.extra_variable(f"{asset}__investment_cost", ensemble_member)
        #     sum -= optimization_problem.extra_variable(f"{asset}__installation_cost", ensemble_member)
        #     sum -= optimization_problem.extra_variable(f"{asset}__variable_operational_cost", ensemble_member)
        #     sum -= optimization_problem.extra_variable(f"{asset}__fixed_operational_cost", ensemble_member)

        return -sum


class _GoalsAndOptions:
    def goals(self):
        goals = super().goals().copy()

        # TODO: these goals should incorperate the timestep
        for demand in self.heat_network_components.get("electricity_demand", []):
            carrier_name = self.esdl_assets[self.esdl_asset_name_to_id_map[demand]].in_ports[0].carrier.name
            price_profile = f"{carrier_name}.price_profile"
            state = f"{demand}.Electricity_demand"
            nominal = self.variable_nominal(state) * np.median(
                self.get_timeseries(price_profile).values)

            goals.append(RevenueGoal(state, price_profile, nominal))

        for demand in self.heat_network_components.get("gas_demand", []):
            carrier_name = self.esdl_assets[self.esdl_asset_name_to_id_map[demand]].in_ports[
                0].carrier.name
            price_profile = f"{carrier_name}.price_profile"
            state = f"{demand}.Gas_demand_mass_flow"
            nominal= self.variable_nominal(state)*np.median(
                self.get_timeseries(price_profile).values)

            goals.append(RevenueGoal(state, price_profile, nominal))

        return goals

    def constraints(self, ensemble_member):
        constraints = super().constraints(ensemble_member)

        for gs in self.heat_network_components.get("gas_tank_storage", []):
            canonical, sign = self.alias_relation.canonical_signed(f"{gs}.Stored_gas_mass")
            storage_t0 = sign * self.state_vector(canonical, ensemble_member)[0]
            constraints.append((storage_t0, 0., 0.))
            canonical, sign = self.alias_relation.canonical_signed(f"{gs}.Gas_tank_flow")
            gas_flow_t0 = sign * self.state_vector(canonical, ensemble_member)[0]
            constraints.append((gas_flow_t0, 0., 0.))

        # for elec in self.heat_network_components.get("electrolyzer", []):
        #     canonical, sign = self.alias_relation.canonical_signed(f"{elec}.Power_consumed")
        #     power = sign * self.state_vector(canonical, ensemble_member)[0]
        #     nominal = self.variable_nominal(f"{elec}.Power_consumed")
        #     constraints.append(((power * nominal - 1.e8) / nominal, 0., 0.))
        #     power = sign * self.state_vector(canonical, ensemble_member)[1]
        #     nominal = self.variable_nominal(f"{elec}.Power_consumed")
        #     constraints.append(((power * nominal) / nominal, 0., 0.))

        return constraints



class MILPProblem(
    _GoalsAndOptions,
    HeatMixin,
    LinearizedOrderGoalProgrammingMixin,
    GoalProgrammingMixin,
    ESDLMixin,
    CollocatedIntegratedOptimizationProblem,
):
    def path_goals(self):
        goals = super().path_goals().copy()

        return goals

    def solver_options(self):
        options = super().solver_options()
        # options["solver"] = "gurobi"

        return options

    def heat_network_options(self):
        options = super().heat_network_options()
        options["include_asset_is_switched_on"] = True

        return options

    # def times(self, variable=None) -> np.ndarray:
    #     return super().times(variable)[:5]


if __name__ == "__main__":
    elect = run_optimization_problem(MILPProblem)
    r = elect.extract_results()
    print(r["Electrolyzer_fc66.ElectricityIn.Power"])
    print(r["Electrolyzer_fc66.Gas_mass_flow_out"])

    # KvR
    print(r["ElectricityCable_09d1.ElectricityOut.Power"])  # Elect being sold
    print(r["GasStorage_e492.Stored_gas_mass"])
    print(r["Pipe_6ba6.GasOut.mass_flow"])  # gas being sold, kg/s

    carrier_name = "gas"
    price_profile = f"{carrier_name}.price_profile"
    gas_rate = elect.get_timeseries(price_profile).values  # currently euro/kg
    # h2_energy_content_per_mass = 118.8 * 1e6  # J/kg
    hydrogen_income = r["Pipe_6ba6.GasOut.mass_flow"][1:] * (
        elect.get_timeseries(price_profile).times[1:]
        - elect.get_timeseries(price_profile).times[0:-1]
    ) * gas_rate[1:]
    # hydrogen_income = r["Pipe_6ba6.GasOut.mass_flow"] * gas_rate
    print("Hydrogen income MEuro: %0.1f" % (sum(hydrogen_income) / 1.0e6))

    carrier_name = "elec"
    price_profile = f"{carrier_name}.price_profile"
    elect_rate = elect.get_timeseries(price_profile).values  # euro/Wh
    elect_energy = (
        r["ElectricityCable_09d1.ElectricityOut.Power"][1:]
        * (
            elect.get_timeseries(price_profile).times[1:]
            - elect.get_timeseries(price_profile).times[0:-1]
        ) / 3600.0
    )  # Wh
    elect_income = elect_energy * elect_rate[1:]
    print("Electricity income MEuro: %0.1f" % (sum(elect_income) / 1.0e6))

    print("Total income MEuro: %0.1f" % ((sum(elect_income) + sum(hydrogen_income)) / 1.0e6))

    print(
        sum(
            (
                elect.get_timeseries(price_profile).times[1:]
                - elect.get_timeseries(price_profile).times[0:-1]
            )
            * r["Pipe_6ba6.GasOut.mass_flow"][1:] * 0.16
        )
    )
    print(r['GasDemand_0cf3__variable_operational_cost'][0])
    temp = 0.0
