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
            sign
            * optimization_problem.state_vector(canonical, ensemble_member)
            * optimization_problem.variable_nominal(self.state)
        )
        price_profile = optimization_problem.get_timeseries(self.price_profile).values
        sum = 0.0
        for i in range(len(price_profile)):
            sum += symbols[i] * price_profile[i]

        for asset in [
            *optimization_problem.heat_network_components.get("gas_demand", []),
            *optimization_problem.heat_network_components.get("gas_source", []),
            *optimization_problem.heat_network_components.get("electrolyzer", []),
            *optimization_problem.heat_network_components.get("gas_tank_storage", []),
            *optimization_problem.heat_network_components.get("wind_park", []),
            *optimization_problem.heat_network_components.get("electricity_demand", []),
            *optimization_problem.heat_network_components.get("electricity_source", []),
        ]:
            sum -= optimization_problem.extra_variable(
                f"{asset}__variable_operational_cost", ensemble_member
            )
            sum -= optimization_problem.extra_variable(
                f"{asset}__fixed_operational_cost", ensemble_member
            )

        return -sum


class _GoalsAndOptions:
    def goals(self):
        goals = super().goals().copy()

        # TODO: these goals should incorperate the timestep
        for demand in self.heat_network_components.get("electricity_demand", []):
            carrier_name = (
                self.esdl_assets[self.esdl_asset_name_to_id_map[demand]].in_ports[0].carrier.name
            )
            price_profile = f"{carrier_name}.price_profile"
            state = f"{demand}.Electricity_demand"
            nominal = self.variable_nominal(state) * np.median(
                self.get_timeseries(price_profile).values
            )

            goals.append(RevenueGoal(state, price_profile, nominal))

        for demand in self.heat_network_components.get("gas_demand", []):
            carrier_name = (
                self.esdl_assets[self.esdl_asset_name_to_id_map[demand]].in_ports[0].carrier.name
            )
            price_profile = f"{carrier_name}.price_profile"
            state = f"{demand}.Gas_demand_mass_flow"
            nominal = self.variable_nominal(state) * np.median(
                self.get_timeseries(price_profile).values
            )

            goals.append(RevenueGoal(state, price_profile, nominal))

        return goals

    def constraints(self, ensemble_member):
        constraints = super().constraints(ensemble_member)

        for gs in self.heat_network_components.get("gas_tank_storage", []):
            canonical, sign = self.alias_relation.canonical_signed(f"{gs}.Stored_gas_mass")
            storage_t0 = sign * self.state_vector(canonical, ensemble_member)[0]
            constraints.append((storage_t0, 0.0, 0.0))
            canonical, sign = self.alias_relation.canonical_signed(f"{gs}.Gas_tank_flow")
            gas_flow_t0 = sign * self.state_vector(canonical, ensemble_member)[0]
            constraints.append((gas_flow_t0, 0.0, 0.0))

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
        options["include_electric_cable_power_loss"] = False

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
    ) / 3600.0 * gas_rate[1:]
    # hydrogen_income = r["Pipe_6ba6.GasOut.mass_flow"] * gas_rate
    print("Hydrogen income MEuro: %0.6f" % (sum(hydrogen_income) / 1.0e6))

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
    print("Electricity income MEuro: %0.6f" % (sum(elect_income) / 1.0e6))
    print("Total income MEuro: %0.6f" % ((sum(elect_income) + sum(hydrogen_income)) / 1.0e6))

    # ----------------------------------------------------------------------------------------------
    # Manual checks
    tot_expense = 0.0
    # Check transport cost 0.16 euro/kg H2
    print("\nTranport cost variable opex")
    print(
        sum(
            (
                elect.get_timeseries(price_profile).times[1:]
                - elect.get_timeseries(price_profile).times[0:-1]
            ) / 3600.0
            * r["Pipe_6ba6.GasOut.mass_flow"][1:] * 0.16
        )
    )
    print(r['GasDemand_0cf3__variable_operational_cost'][0])

    tot_expense += sum(r['GasDemand_0cf3__variable_operational_cost'])

    # Check storage costs fix opex 5 euro/kgH2/year -> 118.575euro/m3
    # Storage resreved size = 500ton -> 500e3 kg / 23.715 (@350bar) kg/m3 = 21083 m3
    # still to add
    print("Storage fixed opex")
    temp_calc = 118.575 * 21083.0
    print(r['GasStorage_e492__fixed_operational_cost'][0])
    print(temp_calc)

    tot_expense += sum(r['GasStorage_e492__fixed_operational_cost'])

    # Still to add this functionality to MILP
    # Check storage costs variable opex 1MWh/tonH2 =
    # 52 euro/MWh (median price) / 1000kgH2 = 0.052 euro/kgH2
    # print(
    #     0.052 * sum(
    #         (
    #             elect.get_timeseries(price_profile).times[1:]
    #             - elect.get_timeseries(price_profile).times[0:-1]
    #         ) / 3600.0 * np.clip(r['GasStorage_e492.GasIn.mass_flow'][1:], 0.0, np.inf)
    #     )
    # )
    # print(r['GasStorage_e492__variable_operational_cost'][0])

    # Check electolyzer variable OPEX 0.05 euro / kg H2 -> should have left this one out
    # Left out as suggested by Javier
    # print("Electrolyzer variable opex")
    # print(
    #     sum(
    #         (
    #             elect.get_timeseries(price_profile).times[1:]
    #             - elect.get_timeseries(price_profile).times[0:-1]
    #         ) / 3600.0
    #         * r["Electrolyzer_fc66.Gas_mass_flow_out"][1:] * 0.05
    #     )
    # )
    # print(r['Electrolyzer_fc66__variable_operational_cost'][0])

    # Check electrolyzer fixed opex, based on installed size of 500MW
    print("Electrolyzer fixed opex")
    print(15.0 * 500.0e6 / 1.0e3)
    print(r['Electrolyzer_fc66__fixed_operational_cost'][0])

    tot_expense += sum(r['Electrolyzer_fc66__fixed_operational_cost'])

    # Check electrolyzer investment cost, based on installed size of 500MW
    print("Electrolyzer investment cost")
    print(2000.0 / 1.e3 * 500.0e6)
    print(r['Electrolyzer_fc66__investment_cost'][0])

    tot_expense += sum(r['Electrolyzer_fc66__investment_cost'])

    # ----------------------------------------------------------------------------------------------
    # Do not delete the temp code below for now: main purpose was to create comparison plots
    # Create some plots

    # import matplotlib.pyplot as plt

    # f1 = plt.figure()
    # ax1 = f1.add_subplot(111)
    # ax1.plot(elect.get_timeseries(price_profile).times, r["GasStorage_e492.Stored_gas_mass"])
    # plt.xlabel('Time steps [s]')
    # plt.ylabel('Gas storage [kg]')
    # plt.show()

    # f2 = plt.figure()
    # ax2 = f2.add_subplot(111)
    # ax2.plot(elect.get_timeseries(price_profile).times, gas_rate)
    # plt.xlabel('Time steps [s]')
    # plt.ylabel('Gas price [euro/kg]')
    # plt.show()

    # f3 = plt.figure()
    # ax3 = f3.add_subplot(111)
    # ax3.plot(elect.get_timeseries(price_profile).times, r["Electrolyzer_fc66.Gas_mass_flow_out"])
    # plt.xlabel('Time steps [s]')
    # plt.ylabel('Electrolyzer gas mass flow out [kg/hr]')
    # plt.show()

    # f4 = plt.figure()
    # ax4 = f4.add_subplot(111)
    # ax4.plot(
    #     elect.get_timeseries(price_profile).times, r["Electrolyzer_fc66.Power_consumed"]
    # )
    # plt.xlabel('Time steps [s]')
    # plt.ylabel('Electrolyzer power usage [W]')
    # plt.show()

    # f5 = plt.figure()
    # ax5 = f5.add_subplot(111)
    # ax5.plot(
    #     elect.get_timeseries(price_profile).times, r["Pipe_6ba6.GasOut.mass_flow"] / 3600.0
    # )
    # plt.xlabel('Time steps [s]')
    # plt.ylabel('Gass sold [kg/s]')
    # plt.show()

    # The code below was used for storing data of multiple runs (different settings like with
    # without mass storage)

    # import pandas as pd

    # storage = "storage_included"
    # data_name = "Pipe_6ba6.GasOut.mass_flow"
    # test = {data_name: r[data_name] / 3600.0}
    # df_data = pd.DataFrame(test)
    # df_data.to_pickle(
    #     "C:\\Projects_gitlab\\NWN_dev\\rtc-tools-heat-network\\tests\\models\\"
    #     f"unit_cases_electricity\\electrolyzer\\df_data{storage}.pkl"
    # )
    # df_read_1 = pd.read_pickle(
    #     "C:\\Projects_gitlab\\NWN_dev\\rtc-tools-heat-network\\tests\\models\\"
    #     f"unit_cases_electricity\\electrolyzer\\df_data{storage}.pkl"
    # )

    # storage = "storage_excluded"
    # data_name = "Pipe_6ba6.GasOut.mass_flow"
    # test = {data_name: r[data_name] / 3600.0}
    # df_data = pd.DataFrame(test)
    # df_data.to_pickle(
    #     "C:\\Projects_gitlab\\NWN_dev\\rtc-tools-heat-network\\tests\\models\\"
    #     f"unit_cases_electricity\\electrolyzer\\df_data{storage}.pkl"
    # )
    # df_read_2 = pd.read_pickle(
    #     "C:\\Projects_gitlab\\NWN_dev\\rtc-tools-heat-network\\tests\\models\\"
    #     f"unit_cases_electricity\\electrolyzer\\df_data{storage}.pkl"
    # )

    # plt.xlabel('Number of time steps')
    # plt.ylabel('Gass sold [kg/s]')
    # plt.show()
    # temp = 0.0
    # ----------------------------------------------------------------------------------------------
