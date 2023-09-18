from pathlib import Path
from unittest import TestCase

import numpy as np

from rtctools.util import run_optimization_problem

from rtctools_heat_network.workflows import EndScenarioSizingHIGHS


class TestEndScenarioSizing(TestCase):
    def test_end_scenario_sizing(self):
        import models.test_case_small_network_ates_buffer_optional_assets.src.run_ates as run_ates

        base_folder = Path(run_ates.__file__).resolve().parent.parent

        # This is an optimization done over a full year with timesteps of 5 days and hour timesteps
        # for the peak day
        solution = run_optimization_problem(EndScenarioSizingHIGHS, base_folder=base_folder)

        results = solution.extract_results()
        # In the future we want to check the following
        # Is the timeline correctly converted, correct peak day, correct amount of timesteps, etc.
        # Check whether expected assets are disabled
        # Check the optimal size of assets
        # Check the cost breakdown, check whether all the enabled assets are in the cost breakdown
        # Check that computation time is within expected bounds

        # Check whehter the heat demand is matched
        for d in solution.heat_network_components.get("demand", []):
            target = solution.get_timeseries(f"{d}.target_heat_demand").values
            np.testing.assert_allclose(target, results[f"{d}.Heat_demand"])

        # Check whether cyclic ates constraint is working
        for a in solution.heat_network_components.get("ates", []):
            stored_heat = results[f"{a}.Stored_heat"]
            np.testing.assert_allclose(stored_heat[0], stored_heat[-1], atol=1.0)

        # Check whether buffer tank is only active in peak day
        peak_day_indx = solution.parameters(0)["peak_day_index"]
        for b in solution.heat_network_components.get("buffer", []):
            heat_buffer = results[f"{b}.Heat_buffer"]
            for i in range(len(solution.times())):
                if i < peak_day_indx or i > (peak_day_indx + 23):
                    np.testing.assert_allclose(heat_buffer, 0.0)


if __name__ == "__main__":
    import time

    start_time = time.time()
    a = TestEndScenarioSizing()
    a.test_end_scenario_sizing()
    print("Execution time: " + time.strftime("%M:%S", time.gmtime(time.time() - start_time)))
