from pathlib import Path
from unittest import TestCase

import numpy as np

from rtctools.util import run_optimization_problem


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
        prim_q = results["HeatExchange_39ed.Primary.HeatIn.Q"]
        sec_q = results["HeatExchange_39ed.Secondary.HeatOut.Q"]
        disabled = results["HeatExchange_39ed__disabled"]

        # Values used in non_storage_component.py
        cp = 4200.0
        rho = 988.0

        # We check the energy converted betweeen the commodities
        # 0.9 efficiency specified in esdl
        np.testing.assert_allclose(prim_heat * 0.9, sec_heat)
        # We check the energy being properly linked to the flows, this should already be satisfied
        # by the multiple commodity test but we do it anyway.
        np.testing.assert_allclose(prim_heat, prim_q * cp * rho * 40.0)
        # Note that we are not testing the last element as we exploit the last timestep for
        # checking the disabled boolean and the assert statement doesn't work for a difference of
        # zero
        np.testing.assert_array_less(-np.abs(sec_q[:-1] * cp * rho * 30.0 - sec_heat[:-1]), 1.0e-6)
        np.testing.assert_allclose(prim_heat[-1], 0.0)
        np.testing.assert_allclose(disabled[-1], 1.0)
        np.testing.assert_allclose(disabled[:-1], 0.0)


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
        prim_q = results["GenericConversion_3d3f.Primary.HeatIn.Q"]
        sec_q = results["GenericConversion_3d3f.Secondary.HeatOut.Q"]

        # TODO: we should also check if heatdemand target is matched
        # TODO: check if the primary source utilisisation is maximised and secondary minimised

        # Values used in non_storage_component.py
        cp = 4200.0
        rho = 988.0

        # We check the energy converted betweeen the commodities
        # 0.9 efficiency specified in esdl
        np.testing.assert_allclose(power_elec * 4.0, sec_heat)
        np.testing.assert_allclose(power_elec + prim_heat, sec_heat)
        # We check the energy being properly linked to the flows, this should already be satisfied
        # by the multiple commodity test but we do it anyway.
        np.testing.assert_allclose(prim_heat, prim_q * cp * rho * 30.0)
        np.testing.assert_array_less(sec_q * cp * rho * 40.0, sec_heat)
