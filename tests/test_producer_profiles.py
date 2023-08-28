from pathlib import Path
from unittest import TestCase

import numpy as np


from rtctools.util import run_optimization_problem

from utils_tests import demand_matching_test, energy_conservation_test, heat_to_discharge_test


class TestProducerMaxProfile(TestCase):
    """To verify that the producer can have a given scaled profile, where the producer will always
    produce equal or less than said profile.
    The constraint for the producer profile is checked by ensuring that the producer is temporarily
    less available (reducing the profile value at a few time steps)."""

    def test_max_producer_profile(self):
        import models.unit_cases.case_3a.src.run_3a as run_3a
        from models.unit_cases.case_3a.src.run_3a import HeatProblemProdProfile

        base_folder = Path(run_3a.__file__).resolve().parent.parent

        solution = run_optimization_problem(HeatProblemProdProfile, base_folder=base_folder)
        results = solution.extract_results()

        demand_matching_test(solution, results)
        energy_conservation_test(solution, results)
        heat_to_discharge_test(solution, results)
        tol = 1e-10
        heat_demand1 = results["HeatingDemand_a3b8.Heat_demand"]
        heat_producer = results["GeothermalSource_b702.Heat_source"]
        size_producer = results["GeothermalSource_b702__max_size"]

        heatdemand1_target = solution.get_timeseries("HeatingDemand_a3b8.target_heat_demand").values
        heat_producer_profile_scaled = solution.get_timeseries(
            "GeothermalSource_b702.target_heat_source"
        ).values
        heat_producer_profile_full = heat_producer_profile_scaled * size_producer

        # check that heat produced is smaller than the profile
        biggerthen = all(heat_producer_profile_full + tol >= heat_producer)
        self.assertTrue(biggerthen)

        # heat should be stored in buffer and thus heatdemand should still be able to be matched.
        np.testing.assert_allclose(heat_demand1, heatdemand1_target)
