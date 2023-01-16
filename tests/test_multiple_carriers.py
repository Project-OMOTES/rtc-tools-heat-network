from pathlib import Path
from unittest import TestCase

import numpy as np

from rtctools.util import run_optimization_problem


class TestMultipleCarriers(TestCase):
    def test_multiple_carriers(self):
        import models.multiple_carriers.src.run_multiple_carriers as run_multiple_carriers
        from models.multiple_carriers.src.run_multiple_carriers import (
            HeatProblem,
        )

        base_folder = Path(run_multiple_carriers.__file__).resolve().parent.parent

        solution = run_optimization_problem(HeatProblem, base_folder=base_folder)

        results = solution.extract_results()

        heat_demand_3222 = results["HeatingDemand_3322.Heat_demand"]
        heat_demand_18aa = results["HeatingDemand_18aa.Heat_demand"]
        heat_demand_3222_q = results["HeatingDemand_3322.Q"]
        heat_demand_18aa_q = results["HeatingDemand_18aa.Q"]

        # Values used in non_storage_component.py
        cp = 4200.0
        rho = 988.0

        # We check for a system consisting out of 2 hydraulically decoupled networks that the energy
        # balance equations are done with the correct carrier.
        np.testing.assert_allclose(heat_demand_3222, heat_demand_3222_q * cp * rho * 30.0)
        np.testing.assert_allclose(heat_demand_18aa, heat_demand_18aa_q * cp * rho * 40.0)
