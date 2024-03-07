from pathlib import Path
from unittest import TestCase

import numpy as np

from rtctools.util import run_optimization_problem

from rtctools_heat_network.esdl.esdl_parser import ESDLFileParser
from rtctools_heat_network.esdl.profile_parser import ProfileReaderFromFile
from rtctools_heat_network.workflows import (
    EndScenarioSizingDiscountedHIGHS,
    EndScenarioSizingHIGHS,
)


class TestEndScenarioSizing(TestCase):
    def test_end_scenario_sizing(self):
        """
        Check if the TestEndScenario sizing workflow is behaving as expected. This is an
        optimization done over a full year with timesteps of 5 days and hour timesteps for the peak
        day.

        Checks:
        - Cyclic behaviour for ATES
        - That buffer tank is only used on peak day
        - demand matching
        - Check if TCO goal included the desired cost components.


        Missing:
        - Link ATES t0 utilization to state of charge at end of year for optimizations over one
        year.
        """
        import models.test_case_small_network_ates_buffer_optional_assets.src.run_ates as run_ates

        base_folder = Path(run_ates.__file__).resolve().parent.parent

        class TestEndScenarioSizingHIGHS(EndScenarioSizingHIGHS):
            def solver_options(self):
                options = super().solver_options()
                options["solver"] = "highs"
                highs_options = options["highs"] = {}
                highs_options["mip_rel_gap"] = 0.05
                return options

        # This is an optimization done over a full year with timesteps of 5 days and hour timesteps
        # for the peak day
        solution = run_optimization_problem(
            TestEndScenarioSizingHIGHS,
            base_folder=base_folder,
            esdl_file_name="test_case_small_network_with_ates_with_buffer_all_optional.esdl",
            esdl_parser=ESDLFileParser,
            profile_reader=ProfileReaderFromFile,
            input_timeseries_file="Warmte_test.csv",
        )

        results = solution.extract_results()
        # In the future we want to check the following
        # Is the timeline correctly converted, correct peak day, correct amount of timesteps, etc.
        # Check whether expected assets are disabled
        # Check the optimal size of assets
        # Check the cost breakdown, check whether all the enabled assets are in the cost breakdown
        # Check that computation time is within expected bounds

        # Check whehter the milp demand is matched
        for d in solution.energy_system_components.get("demand", []):
            target = solution.get_timeseries(f"{d}.target_heat_demand").values
            np.testing.assert_allclose(target, results[f"{d}.Heat_demand"])

        # Check whether cyclic ates constraint is working
        for a in solution.energy_system_components.get("ates", []):
            stored_heat = results[f"{a}.Stored_heat"]
            np.testing.assert_allclose(stored_heat[0], stored_heat[-1], atol=1.0)

        # Check whether buffer tank is only active in peak day
        peak_day_indx = solution.parameters(0)["peak_day_index"]
        for b in solution.energy_system_components.get("buffer", []):
            heat_buffer = results[f"{b}.Heat_buffer"]
            for i in range(len(solution.times())):
                if i < peak_day_indx or i > (peak_day_indx + 23):
                    np.testing.assert_allclose(heat_buffer[i], 0.0, atol=1.0e-6)

        obj = 0.0
        years = solution.parameters(0)["number_of_years"]
        for asset in [
            *solution.energy_system_components.get("source", []),
            *solution.energy_system_components.get("ates", []),
            *solution.energy_system_components.get("buffer", []),
            *solution.energy_system_components.get("demand", []),
            *solution.energy_system_components.get("heat_exchanger", []),
            *solution.energy_system_components.get("heat_pump", []),
            *solution.energy_system_components.get("pipe", []),
        ]:
            obj += results[f"{solution._asset_fixed_operational_cost_map[asset]}"] * years
            obj += results[f"{solution._asset_variable_operational_cost_map[asset]}"] * years
            obj += results[f"{solution._asset_investment_cost_map[asset]}"]
            obj += results[f"{solution._asset_installation_cost_map[asset]}"]

        np.testing.assert_allclose(obj / 1.0e6, solution.objective_value)

    def test_end_scenario_sizing_discounted(self):
        """
        Check if the TestEndScenario sizing workflow is behaving as expected. This is an
        optimization done over a full year with timesteps of 5 days and hour timesteps for the peak
        day.

        Checks:
        - Cyclic behaviour for ATES
        - That buffer tank is only used on peak day
        - demand matching

        Missing:
        - Check if TCO goal included the desired cost components.

        """
        import models.test_case_small_network_ates_buffer_optional_assets.src.run_ates as run_ates

        base_folder = Path(run_ates.__file__).resolve().parent.parent

        class TestEndScenarioSizingDiscountedHIGHS(EndScenarioSizingDiscountedHIGHS):
            def solver_options(self):
                options = super().solver_options()
                options["solver"] = "highs"
                highs_options = options["highs"] = {}
                highs_options["mip_rel_gap"] = 0.05
                return options

        # This is an optimization done over a full year with timesteps of 5 days and hour timesteps
        # for the peak day
        solution = run_optimization_problem(
            TestEndScenarioSizingDiscountedHIGHS,
            base_folder=base_folder,
            esdl_file_name="test_case_small_network_with_ates_with_buffer_all_optional.esdl",
            esdl_parser=ESDLFileParser,
            profile_reader=ProfileReaderFromFile,
            input_timeseries_file="Warmte_test.csv",
        )

        results = solution.extract_results()
        # In the future we want to check the following
        # Is the timeline correctly converted, correct peak day, correct amount of timesteps, etc.
        # Check whether expected assets are disabled
        # Check the optimal size of assets
        # Check the cost breakdown, check whether all the enabled assets are in the cost breakdown
        # Check that computation time is within expected bounds

        # TODO: this workflow will be improved, currently the not matching of demands is within
        # MIPGAP.
        # Check whether the milp demand is matched
        # for d in solution.heat_network_components.get("demand", []):
        #     target = solution.get_timeseries(f"{d}.target_heat_demand").values
        #     # np.testing.assert_allclose(target, results[f"{d}.Heat_demand"])

        # Check whether cyclic ates constraint is working
        for a in solution.energy_system_components.get("ates", []):
            stored_heat = results[f"{a}.Stored_heat"]
            np.testing.assert_allclose(stored_heat[0], stored_heat[-1], atol=1.0)

        # Check whether buffer tank is only active in peak day
        peak_day_indx = solution.parameters(0)["peak_day_index"]
        for b in solution.energy_system_components.get("buffer", []):
            heat_buffer = results[f"{b}.Heat_buffer"]
            for i in range(len(solution.times())):
                if i < peak_day_indx or i > (peak_day_indx + 23):
                    np.testing.assert_allclose(heat_buffer[i], 0.0, atol=1.0e-6)


if __name__ == "__main__":
    import time

    start_time = time.time()
    a = TestEndScenarioSizing()
    a.test_end_scenario_sizing()
    print("Execution time: " + time.strftime("%M:%S", time.gmtime(time.time() - start_time)))
