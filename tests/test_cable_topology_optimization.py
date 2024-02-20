from pathlib import Path
from unittest import TestCase

import numpy as np

from rtctools.util import run_optimization_problem


class TestAbsoluteHeat(TestCase):
    def test_absolute_heat(self):
        """
        This test checks the functioning of topology optimization of electricity cables. It uses
        a symmetrical network where the left side should stay in as those cables are shorter.

        Checks:
        1. Demand is matched
        2. That intended cables are removed

        """
        import models.electricity_cable_topology.src.example as example
        from models.electricity_cable_topology.src.example import HeatProblem

        base_folder = Path(example.__file__).resolve().parent.parent

        heat_problem = run_optimization_problem(HeatProblem, base_folder=base_folder)

        results = heat_problem.extract_results()

        for demand in heat_problem.heat_network_components.get("electricity_demand", []):
            target = heat_problem.get_timeseries(f"{demand}.target_electricity_demand").values
            np.testing.assert_allclose(target, results[f"{demand}.Electricity_demand"])

        removed_cables = [
            "ElectricityCable_6dce",
            "ElectricityCable_6aeb",
            "ElectricityCable_fbfd",
            "ElectricityCable_ef84",
        ]

        for cable in removed_cables:
            np.testing.assert_allclose(results[f"{cable}__investment_cost"], 0.0)
            np.testing.assert_allclose(results[f"{cable}__en_max_current"], 0.0)
