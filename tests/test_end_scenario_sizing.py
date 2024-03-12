from pathlib import Path
from unittest import TestCase

import numpy as np

from rtctools.util import run_optimization_problem

from rtctools_heat_network.esdl.esdl_parser import ESDLFileParser
from rtctools_heat_network.esdl.profile_parser import ProfileReaderFromFile
from rtctools_heat_network.workflows import (
    EndScenarioSizingDiscountedHIGHS,
    EndScenarioSizingHIGHS,
    EndScenarioSizingStagedHIGHS,
    run_end_scenario_sizing,
)


class TestEndScenarioSizing(TestCase):
    def test_end_scenario_sizing(self):
        """
        Check if the EndScenarioSizingHIGHS sizing workflow is behaving as expected. This is an
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

        # This is an optimization done over a full year with timesteps of 5 days and hour timesteps
        # for the peak day
        solution = run_optimization_problem(
            EndScenarioSizingHIGHS,
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
        for d in solution.energy_system_components.get("heat_demand", []):
            target = solution.get_timeseries(f"{d}.target_heat_demand").values
            np.testing.assert_allclose(target, results[f"{d}.Heat_demand"])

        # Check whether cyclic ates constraint is working
        for a in solution.energy_system_components.get("ates", []):
            stored_heat = results[f"{a}.Stored_heat"]
            np.testing.assert_allclose(stored_heat[0], stored_heat[-1], atol=1.0)

        # Check whether buffer tank is only active in peak day
        peak_day_indx = solution.parameters(0)["peak_day_index"]
        for b in solution.energy_system_components.get("heat_buffer", []):
            heat_buffer = results[f"{b}.Heat_buffer"]
            for i in range(len(solution.times())):
                if i < peak_day_indx or i > (peak_day_indx + 23):
                    np.testing.assert_allclose(heat_buffer[i], 0.0, atol=1.0e-6)

        obj = 0.0
        years = solution.parameters(0)["number_of_years"]
        for asset in [
            *solution.energy_system_components.get("heat_source", []),
            *solution.energy_system_components.get("ates", []),
            *solution.energy_system_components.get("heat_buffer", []),
            *solution.energy_system_components.get("heat_demand", []),
            *solution.energy_system_components.get("heat_exchanger", []),
            *solution.energy_system_components.get("heat_pump", []),
            *solution.energy_system_components.get("heat_pipe", []),
        ]:
            obj += results[f"{solution._asset_fixed_operational_cost_map[asset]}"] * years
            obj += results[f"{solution._asset_variable_operational_cost_map[asset]}"] * years
            obj += results[f"{solution._asset_investment_cost_map[asset]}"]
            obj += results[f"{solution._asset_installation_cost_map[asset]}"]

        np.testing.assert_allclose(obj / 1.0e6, solution.objective_value)

    def test_end_scenario_sizing_staged(self):
        """
        Check if the EndScenarioSizingStagedHIGHS workflow is behaving as expected. This is an
        optimization done over a full year with timesteps of 5 days and hour timesteps for the peak
        day.

        Checks:
        - Cyclic behaviour for ATES
        - That buffer tank is only used on peak day
        - demand matching
        - Check if TCO goal included the desired cost components.

        - Compare objective value of staged approach wit non-staged approach

        Compare solution time of 3 scenarios on how to run the optimisation:
        - Staged approach should be solved faster than the unstaged approaches
        - Unstaged approaches, using the general function run_optimization_problem and the
        function run_end_scenario_sizing with staged_pipe_optimization to False should have
        comparable computation times.

        Missing:
        - Link ATES t0 utilization to state of charge at end of year for optimizations over one
        year.
        """
        import models.test_case_small_network_ates_buffer_optional_assets.src.run_ates as run_ates

        base_folder = Path(run_ates.__file__).resolve().parent.parent

        # This is an optimization done over a full year with timesteps of 5 days and hour timesteps
        # for the peak day

        solution_unstaged = run_optimization_problem(
            EndScenarioSizingHIGHS,
            base_folder=base_folder,
            esdl_file_name="test_case_small_network_with_ates_with_buffer_all_optional.esdl",
            esdl_parser=ESDLFileParser,
            profile_reader=ProfileReaderFromFile,
            input_timeseries_file="Warmte_test.csv",
        )
        solution_unstaged_2 = run_end_scenario_sizing(
            EndScenarioSizingHIGHS,
            staged_pipe_optimization=False,
            base_folder=base_folder,
            esdl_file_name="test_case_small_network_with_ates_with_buffer_all_optional.esdl",
            esdl_parser=ESDLFileParser,
            profile_reader=ProfileReaderFromFile,
            input_timeseries_file="Warmte_test.csv",
        )

        solution = run_end_scenario_sizing(
            EndScenarioSizingStagedHIGHS,
            base_folder=base_folder,
            esdl_file_name="test_case_small_network_with_ates_with_buffer_all_optional.esdl",
            esdl_parser=ESDLFileParser,
            profile_reader=ProfileReaderFromFile,
            input_timeseries_file="Warmte_test.csv",
        )

        results = solution.extract_results()

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
                    np.testing.assert_allclose(heat_buffer[i], 0.0, atol=1.0e-6)

        obj = 0.0
        years = solution.parameters(0)["number_of_years"]
        for asset in [
            *solution.heat_network_components.get("source", []),
            *solution.heat_network_components.get("ates", []),
            *solution.heat_network_components.get("buffer", []),
            *solution.heat_network_components.get("demand", []),
            *solution.heat_network_components.get("heat_exchanger", []),
            *solution.heat_network_components.get("heat_pump", []),
            *solution.heat_network_components.get("pipe", []),
        ]:
            obj += results[f"{solution._asset_fixed_operational_cost_map[asset]}"] * years
            obj += results[f"{solution._asset_variable_operational_cost_map[asset]}"] * years
            obj += results[f"{solution._asset_investment_cost_map[asset]}"]
            obj += results[f"{solution._asset_installation_cost_map[asset]}"]

        np.testing.assert_allclose(obj / 1.0e6, solution.objective_value)

        # comparing results of staged and unstaged problem definition. For larger systems there
        # might be a difference in the value but that would either be a difference within the
        # MIPgap (thus checking best bound objective is still smaller or equal than objective of
        # the other problem) or because of some tighter constraints in the staged problem e.g.
        # staged problem slightly higher objective value
        if (
            solution.solver_stats["mip_gap"] == 0.0
            and solution_unstaged.solver_stats["mip_gap"] == 0.0
        ):
            np.testing.assert_allclose(solution.objective_value, solution_unstaged.objective_value)
        else:
            np.testing.assert_array_less(
                solution_unstaged._priorities_output[1][4]["mip_dual_bound"] - 1e-6,
                solution.objective_value,
            )
            np.testing.assert_array_less(
                solution._priorities_output[3][4]["mip_dual_bound"] - 1e-6,
                solution_unstaged.objective_value,
            )
        # checking time spend on optimisation approaches, the difference between the two unstaged
        # approaches should be smaller than the difference with the staged and unstaged approach.
        # The staged approach should be quickest in solving. The two unstaged approaches should
        # have comparable computation time.
        solution_time_unstaged = sum([i[1] for i in solution_unstaged._priorities_output])
        solution_time_unstaged_2 = sum([i[1] for i in solution_unstaged_2._priorities_output])
        solution_time_staged = sum([i[1] for i in solution._priorities_output])
        np.testing.assert_array_less(
            abs(solution_time_unstaged_2 - solution_time_unstaged),
            abs(solution_time_staged - solution_time_unstaged),
        )
        np.testing.assert_array_less(solution_time_staged, solution_time_unstaged)

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

        # Check whether the heat demand is matched
        for d in solution.energy_system_components.get("demand", []):
            target = solution.get_timeseries(f"{d}.target_heat_demand").values
            np.testing.assert_allclose(target, results[f"{d}.Heat_demand"])

        # Check whether cyclic ates constraint is working
        for a in solution.energy_system_components.get("ates", []):
            stored_heat = results[f"{a}.Stored_heat"]
            np.testing.assert_allclose(stored_heat[0], stored_heat[-1], atol=1.0)

        # Check whether buffer tank is only active in peak day
        peak_day_indx = solution.parameters(0)["peak_day_index"]
        for b in solution.energy_system_components.get("heat_buffer", []):
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
