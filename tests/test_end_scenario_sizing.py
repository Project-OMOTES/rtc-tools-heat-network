from pathlib import Path
from unittest import TestCase

import numpy as np

from rtctools.util import run_optimization_problem

import rtctools_heat_network._darcy_weisbach as darcy_weisbach
from rtctools_heat_network.esdl.esdl_parser import ESDLFileParser
from rtctools_heat_network.esdl.profile_parser import ProfileReaderFromFile
from rtctools_heat_network.workflows import (
    EndScenarioSizingDiscountedHIGHS,
    EndScenarioSizingHIGHS,
    EndScenarioSizingStagedHIGHS,
    run_end_scenario_sizing,
)
from rtctools_heat_network.workflows.grow_workflow import EndScenarioSizingHeadLossStaged

from tests.utils_tests import demand_matching_test


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

        - Compare objective value of staged approach without staged approach

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
        # checking time spend on optimisation approaches, the difference between the unstaged
        # approaches should be smaller than the difference with the staged approach. The staged
        # approach should be quickest in solving.
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

    def test_end_scenario_sizing_head_loss(self):
        """
        Test is EndScenarioSizingHeadLoss class is behaving as expected. E.g. should behave
        similarly to EndScenarioSizing class but now the linearised inequality Darcy Weisbach
        equations should be included for the head loss.

        Checks:
        - objective (TCO) same order of magnitude
        - head loss is calculated, currently only checked if it is equal or larger than the
        DW head_loss
        """

        import models.test_case_small_network_ates_buffer_optional_assets.src.run_ates as run_ates

        base_folder = Path(run_ates.__file__).resolve().parent.parent

        solution = run_end_scenario_sizing(
            EndScenarioSizingHeadLossStaged,
            base_folder=base_folder,
            esdl_file_name="test_case_small_network_with_ates_with_buffer_all_optional.esdl",
            esdl_parser=ESDLFileParser,
            profile_reader=ProfileReaderFromFile,
            input_timeseries_file="Warmte_test.csv",
        )

        results = solution.extract_results()

        demand_matching_test(solution, results)

        pipes = solution.heat_network_components.get("pipe")
        for pipe in pipes:
            pipe_diameter = solution.parameters(0)[f"{pipes[0]}.diameter"]
            pipe_wall_roughness = solution.heat_network_options()["wall_roughness"]
            temperature = solution.parameters(0)[f"{pipes[0]}.temperature"]
            pipe_length = solution.parameters(0)[f"{pipes[0]}.length"]
            velocities = results[f"{pipe}.Q"] / solution.parameters(0)[f"{pipe}.area"]
            for ii in range(len(results[f"{pipe}.dH"])):
                if velocities[ii] > 0 and pipe_diameter > 0:
                    np.testing.assert_array_less(
                        darcy_weisbach.head_loss(
                            velocities[ii],
                            pipe_diameter,
                            pipe_length,
                            pipe_wall_roughness,
                            temperature,
                        ),
                        abs(results[f"{pipe}.dH"][ii]),
                    )


if __name__ == "__main__":
    import time

    start_time = time.time()
    a = TestEndScenarioSizing()
    a.test_end_scenario_sizing()
    print("Execution time: " + time.strftime("%M:%S", time.gmtime(time.time() - start_time)))
