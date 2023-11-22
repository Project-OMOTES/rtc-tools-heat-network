from pathlib import Path
from unittest import TestCase

import numpy as np

from rtctools.util import run_optimization_problem

from utils_tests import demand_matching_test, energy_conservation_test, heat_to_discharge_test


class TestHEX(TestCase):
    def test_heat_exchanger(self):
        import models.heat_exchange.src.run_heat_exchanger as run_heat_exchanger
        from models.heat_exchange.src.run_heat_exchanger import (
            HeatProblem,
        )

        base_folder = Path(run_heat_exchanger.__file__).resolve().parent.parent

        solution = run_optimization_problem(HeatProblem, base_folder=base_folder)

        results = solution.extract_results()

        prim_heat = results["HeatExchange_39ed.Primary_heat"]
        sec_heat = results["HeatExchange_39ed.Secondary_heat"]
        disabled = results["HeatExchange_39ed__disabled"]

        # We check the energy converted betweeen the commodities
        eff = solution.parameters(0)["HeatExchange_39ed.efficiency"]

        demand_matching_test(solution, results)
        heat_to_discharge_test(solution, results)
        energy_conservation_test(solution, results)

        np.testing.assert_allclose(prim_heat * eff, sec_heat)

        # Note that we are not testing the last element as we exploit the last timestep for
        # checking the disabled boolean and the assert statement doesn't work for a difference of
        # zero
        np.testing.assert_allclose(prim_heat[-1], 0.0, atol=1e-8)
        np.testing.assert_allclose(disabled[-1], 1.0)
        np.testing.assert_allclose(disabled[:-1], 0.0)
        # Check that heat is flowing through the hex
        np.testing.assert_array_less(-prim_heat[:-1], 0.0)


class TestHP(TestCase):
    def test_heat_pump(self):
        import models.heatpump.src.run_heat_pump as run_heat_pump
        from models.heatpump.src.run_heat_pump import (
            HeatProblem,
        )

        base_folder = Path(run_heat_pump.__file__).resolve().parent.parent

        solution = run_optimization_problem(HeatProblem, base_folder=base_folder)

        results = solution.extract_results()

        prim_heat = results["GenericConversion_3d3f.Primary_heat"]
        sec_heat = results["GenericConversion_3d3f.Secondary_heat"]
        power_elec = results["GenericConversion_3d3f.Power_elec"]

        # TODO: we should also check if heatdemand target is matched
        # TODO: check if the primary source utilisisation is maximised and secondary minimised
        # I think it is badly scaled or some scaling issues in constraints. (possible in combination
        # with disconnected)

        demand_matching_test(solution, results)
        heat_to_discharge_test(solution, results)
        energy_conservation_test(solution, results)

        # We check the energy converted betweeen the commodities
        np.testing.assert_allclose(power_elec * 4.0, sec_heat)
        np.testing.assert_allclose(power_elec + prim_heat, sec_heat)
