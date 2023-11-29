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

        timestep = 3600.
        rho = milp_problem.parameters(0)["GasStorage_e492.density_max_storage"]
        np.testing.assert_allclose(np.diff(results["GasStorage_e492.Stored_gas_mass"]), results["GasStorage_e492.Gas_tank_flow"][1:] * rho * timestep)





        a=1
