from pathlib import Path
from unittest import TestCase

import numpy as np

from rtctools.util import run_optimization_problem

from rtctools_heat_network.workflows import EndScenarioSizingHIGHS


class TestEndScenarioSizingAnnualized(TestCase):
    def test_end_scenario_sizing_annualized(self):
        from models.test_case_small_network_optional_assets_annualized.src.run_annualized import HeatProblemDiscAnnualizedCost, HeatProblemNoDiscTotalCost
        from models.test_case_small_network_optional_assets_annualized.src.run_ates import HeatProblem
        import models.test_case_small_network_optional_assets_annualized.src.run_annualized as run_annualized

        base_folder = Path(run_annualized.__file__).resolve().parent.parent

        # First we check that the non annualized solution with discount=0 equals the one without discount computations and for one year
        solution_1 = run_optimization_problem(HeatProblemDiscAnnualizedCost, base_folder=base_folder)
        solution_2 = run_optimization_problem(HeatProblemNoDiscTotalCost, base_folder=base_folder)

        np.testing.assert_allclose(solution_1.objective_value, solution_2.objective_value)


        # Here we check that the discounted solution cheaper than the non-discounted one
        # check the values of the total investment.
        solution = run_optimization_problem(HeatProblemDiscAnnualizedCost, base_folder=base_folder)

        results = solution.extract_results()

        # 

        a=1



if __name__ == "__main__":
    import time

    start_time = time.time()
    a = TestEndScenarioSizingAnnualized()
    a.test_end_scenario_sizing_annualized()
    print("Execution time: " + time.strftime("%M:%S", time.gmtime(time.time() - start_time)))
