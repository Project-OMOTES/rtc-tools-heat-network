from pathlib import Path
from unittest import TestCase

import numpy as np

from rtctools.util import run_optimization_problem


class TestAtes(TestCase):
    def test_ates(self):
        import models.test_case_small_network_with_ates.src.run_ates as run_ates
        from models.test_case_small_network_with_ates.src.run_ates import (
            HeatProblem,
        )

        base_folder = Path(run_ates.__file__).resolve().parent.parent

        # This is an optimization done over a full year with 365 day timesteps
        solution = run_optimization_problem(HeatProblem, base_folder=base_folder)

        results = solution.extract_results()

        stored_heat = results["ATES_033c.Stored_heat"]
        heat_loss = results["ATES_033c.Heat_loss"]
        heat_ates = results["ATES_033c.Heat_ates"]
        summed_charge = np.sum(np.clip(heat_ates, 0.0, np.inf))
        summed_discharge = np.abs(np.sum(np.clip(heat_ates, -np.inf, 0.0)))

        # Test if the heat loss is as expected
        coeff = solution.parameters(0)["ATES_033c.heat_loss_coeff"]
        np.testing.assert_allclose(heat_loss, coeff * stored_heat)
        np.testing.assert_array_less(summed_discharge, summed_charge)

        # Test begin and end same state
        np.testing.assert_allclose(stored_heat[0], stored_heat[-1])


if __name__ == "__main__":
    import time

    start_time = time.time()
    a = TestAtes()
    a.test_ates()
    temp = 0
