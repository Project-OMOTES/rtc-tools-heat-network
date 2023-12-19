from pathlib import Path
from unittest import TestCase

import numpy as np

from rtctools.util import run_optimization_problem

# from utils_tests import demand_matching_test, energy_conservation_test, heat_to_discharge_test


class TestElectrolyzer(TestCase):
    def test_electrolyzer(self):
        """
        ...
        """
        import models.unit_cases_electricity.electrolyzer.src.example as example
        from models.unit_cases_electricity.electrolyzer.src.example import MILPProblem

        base_folder = Path(example.__file__).resolve().parent.parent

        milp_problem = run_optimization_problem(MILPProblem, base_folder=base_folder)

        results = milp_problem.extract_results()

        price_profile = "GasDemand_0cf3.gas_price"
        state = "GasDemand_0cf3.Gas_demand_mass_flow"
        nominal = milp_problem.variable_nominal(state) * np.median(
            milp_problem.get_timeseries(price_profile).values
        )
        gas_revenue = np.sum(
            milp_problem.get_timeseries("GasDemand_0cf3.gas_price").values
            * results["GasDemand_0cf3.Gas_demand_mass_flow"]
        ) / nominal

        price_profile = "ElectricityDemand_9d15.electricity_price"
        state = "ElectricityDemand_9d15.ElectricityIn.Power"
        nominal = milp_problem.variable_nominal(state) * np.median(
            milp_problem.get_timeseries(price_profile).values
        )
        electricity_revenue = np.sum(
            milp_problem.get_timeseries("ElectricityDemand_9d15.electricity_price").values
            * results["ElectricityDemand_9d15.ElectricityIn.Power"]
        ) / nominal

        # Check that goal is as expected
        np.testing.assert_allclose(
            milp_problem.objective_value, -(gas_revenue + electricity_revenue),
            atol=1e-2,
            rtol=1e-2,
        )
        tol = 1.e-6
        # Check that the electrolyzer only consumes electricity and does not produce.
        np.testing.assert_array_less(-results["Electrolyzer_fc66.ElectricityIn.Power"], tol)

        # Check that windfarm does not produce more than the specified maximum profile
        ub = milp_problem.get_timeseries("WindPark_7f14.maximum_production").values
        np.testing.assert_array_less(results["WindPark_7f14.ElectricityOut.Power"], ub + tol)

        # Check that the wind farm setpoint matches with the production
        np.testing.assert_allclose(
            results["WindPark_7f14.ElectricityOut.Power"],
            ub * results["WindPark_7f14.Set_point"]
        )

        # Checks on the storage
        timestep = 3600.0
        rho = milp_problem.parameters(0)["GasStorage_e492.density_max_storage"]
        np.testing.assert_allclose(
            np.diff(results["GasStorage_e492.Stored_gas_mass"]),
            results["GasStorage_e492.Gas_tank_flow"][1:] * rho * timestep
        )
        np.testing.assert_allclose(results["GasStorage_e492.Stored_gas_mass"][0], 0.0)
        np.testing.assert_allclose(results["GasStorage_e492.Gas_tank_flow"][0], 0.0)

        for cable in milp_problem.heat_network_components.get("electricity_cable", []):
            ub = milp_problem.esdl_assets[
                milp_problem.esdl_asset_name_to_id_map[f"{cable}"]
            ].attributes["capacity"]
            np.testing.assert_array_less(results[f"{cable}.ElectricityOut.Power"], ub + tol)
            lb = \
                milp_problem.esdl_assets[
                    milp_problem.esdl_asset_name_to_id_map[f"{cable}"]
                ].in_ports[0].carrier.voltage
            tol = 1.e-2
            np.testing.assert_array_less(lb - tol, results[f"{cable}.ElectricityOut.V"])
            np.testing.assert_array_less(
                results[f"{cable}.ElectricityOut.Power"],
                results[f"{cable}.ElectricityOut.V"] * results[f"{cable}.ElectricityOut.I"] + tol
            )

        # Electrolyser
        coef_a = milp_problem.parameters(0)["Electrolyzer_fc66.a_eff_coefficient"]
        coef_b = milp_problem.parameters(0)["Electrolyzer_fc66.b_eff_coefficient"]
        coef_c = milp_problem.parameters(0)["Electrolyzer_fc66.c_eff_coefficient"]
        a, b = milp_problem._get_linear_coef_electrolyzer_mass_vs_epower_fit(
            coef_a,
            coef_b,
            coef_c,
            n_lines=3,
            electrical_power_min=0.0,
            electrical_power_max=milp_problem.bounds()["Electrolyzer_fc66.ElectricityIn.Power"][1]
        )
        # TODO: Add test below once the mass flow is coupled to the volumetric flow rate. Currently
        # the gas network is non-limiting (mass flow not coupled to volumetric flow rate)
        # np.testing.assert_allclose(results["Electrolyzer_fc66.Gas_mass_flow_out"],
        #                            results["Electrolyzer_fc66.GasOut.Q"] *
        #                            milp_problem.parameters(0)["Electrolyzer_fc66.density"])
        for i in range(len(a)):
            np.testing.assert_array_less(
                results["Electrolyzer_fc66.Gas_mass_flow_out"],
                results["Electrolyzer_fc66.ElectricityIn.Power"] * a[i] + b[i] + 1.0e-3
            )

        # print(results["Electrolyzer_fc66.ElectricityIn.Power"])
        # print(results["Electrolyzer_fc66.Gas_mass_flow_out"])

        #  -----------------------------------------------------------------------------------------
        # Do cost checks

        # Check variable opex: transport cost 0.1 euro/kg H2
        gas_tranport_cost = sum(
            (
                milp_problem.get_timeseries(price_profile).times[1:]
                - milp_problem.get_timeseries(price_profile).times[0:-1]
            ) / 3600.0
            * results["Pipe_6ba6.GasOut.mass_flow"][1:] * 0.1
        )
        np.testing.assert_allclose(
            gas_tranport_cost,
            results["GasDemand_0cf3__variable_operational_cost"],
        )

        # Check storage cost fix opex 10 euro/kgH2/year -> 10*23.715 = 237.15euro/m3
        # Storage reserved size = 500m3
        storage_fixed_opex = 237.15 * 500.0
        np.testing.assert_allclose(
            storage_fixed_opex,
            sum(results['GasStorage_e492__fixed_operational_cost']),
        )

        # Check electrolyzer fixed opex, based on installed size of 500MW and 10euro/kW
        electrolyzer_fixed_opex = 1.0 * 500.0e6 / 1.0e3
        np.testing.assert_allclose(
            electrolyzer_fixed_opex,
            sum(results['Electrolyzer_fc66__fixed_operational_cost']),
        )

        # Check electrolyzer investment cost, based on installed size of 500MW and 20euro/kW
        electrolyzer_investment_cost = 20.0 * 500.0e6 / 1.e3
        np.testing.assert_allclose(
            electrolyzer_investment_cost,
            sum(results['Electrolyzer_fc66__investment_cost']),
        )
        #  -----------------------------------------------------------------------------------------


if __name__ == "__main__":
    import time

    start_time = time.time()
    test = TestElectrolyzer()
    sol = test.test_electrolyzer()
    print("Execution time: " + time.strftime("%M:%S", time.gmtime(time.time() - start_time)))
