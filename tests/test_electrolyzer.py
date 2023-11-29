from pathlib import Path
from unittest import TestCase

import numpy as np

from rtctools.util import run_optimization_problem

from utils_tests import demand_matching_test, energy_conservation_test, heat_to_discharge_test


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

        gas_revenue = np.sum(milp_problem.get_timeseries("GasDemand_0cf3.gas_price").values * results["GasDemand_0cf3.Gas_demand_mass_flow"])

        electricity_revenue = np.sum(milp_problem.get_timeseries("ElectricityDemand_9d15.electricity_price").values * results["ElectricityDemand_9d15.ElectricityIn.Power"])

        # Check that goal is as expected
        np.testing.assert_allclose(
            milp_problem.objective_value, -(gas_revenue + electricity_revenue) / 1.e6
        )
        tol = 1.e-6
        # Check that the electrolyzer only consumes electricity and does not produce.
        np.testing.assert_array_less(-results["Electrolyzer_fc66.ElectricityIn.Power"], tol)

        # Check that windfarm does not produce more than the specified maximum profile
        ub = milp_problem.get_timeseries("WindPark_7f14.maximum_production").values
        np.testing.assert_array_less(results["WindPark_7f14.ElectricityOut.Power"], ub + tol)

        # Check that the wind farm setpoint matches with the production
        np.testing.assert_allclose(results["WindPark_7f14.ElectricityOut.Power"], ub * results["WindPark_7f14.Set_point"])

        # Checks on the storage
        timestep = 3600.
        rho = milp_problem.parameters(0)["GasStorage_e492.density_max_storage"]
        np.testing.assert_allclose(np.diff(results["GasStorage_e492.Stored_gas_mass"]), results["GasStorage_e492.Gas_tank_flow"][1:] * rho * timestep)
        np.testing.assert_allclose(results["GasStorage_e492.Stored_gas_mass"][0], 0.)
        np.testing.assert_allclose(results["GasStorage_e492.Gas_tank_flow"][0], 0.)

        for cable in milp_problem.heat_network_components.get("electricity_cable", []):
            ub = milp_problem.esdl_assets[milp_problem.esdl_asset_name_to_id_map[f"{cable}"]].attributes["capacity"]
            np.testing.assert_array_less(results[f"{cable}.ElectricityOut.Power"], ub + tol)
            lb = \
            milp_problem.esdl_assets[milp_problem.esdl_asset_name_to_id_map[f"{cable}"]].in_ports[0].carrier.voltage
            tol = 1.e-2
            np.testing.assert_array_less(lb - tol, results[f"{cable}.ElectricityOut.V"])
            np.testing.assert_array_less(results[f"{cable}.ElectricityOut.Power"],
                                         results[f"{cable}.ElectricityOut.V"] *
                                         results[f"{cable}.ElectricityOut.I"] + tol)

        # Electrolyser
        coef_a = milp_problem.parameters(0)["Electrolyzer_fc66.a_eff_coefficient"]
        coef_b = milp_problem.parameters(0)["Electrolyzer_fc66.b_eff_coefficient"]
        coef_c = milp_problem.parameters(0)["Electrolyzer_fc66.c_eff_coefficient"]
        a, b = milp_problem._get_linear_coef_electrolyzer_mass_vs_epower_fit(
                coef_a,
                coef_b,
                coef_c,
                1.,
                n_lines=1,
                electrical_power_min=milp_problem.parameters(0)["Electrolyzer_fc66.minimum_load"],
                electrical_power_max=milp_problem.bounds()["Electrolyzer_fc66.ElectricityIn.Power"]
        )

        np.testing.assert_allclose(results["Electrolyzer_fc66.Gas_mass_out"],
                                   results["Electrolyzer_fc66.GasOut.Q"] *
                                   milp_problem.parameters(0)["Electrolyzer_fc66.density"])
        for i in range(len(a)):
            np.testing.assert_array_less(results["Electrolyzer_fc66.Gas_mass_out"],
                                         results["Electrolyzer_fc66.ElectricityIn.Power"]*b[i] + a[i])

        a=1
