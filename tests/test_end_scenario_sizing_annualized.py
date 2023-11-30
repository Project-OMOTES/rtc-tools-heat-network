from pathlib import Path
from unittest import TestCase

import numpy as np

from rtctools.util import run_optimization_problem

from rtctools_heat_network.workflows import EndScenarioSizingHIGHS


class TestEndScenarioSizingAnnualized(TestCase):
    def test_end_scenario_sizing_annualized(self):
        from models.test_case_small_network_optional_assets_annualized.src.run_annualized import HeatProblemDiscAnnualizedCost
        from models.test_case_small_network_optional_assets_annualized.src.run_ates import HeatProblem
        import models.test_case_small_network_optional_assets_annualized.src.run_annualized as run_annualized

        base_folder = Path(run_annualized.__file__).resolve().parent.parent

        # Non annualized objective value with discount=0 and technical life 1 
        # year matches the objective value of the discounted problem
        solution_1 = run_optimization_problem(HeatProblemDiscAnnualizedCost,
                                              base_folder=base_folder)
        solution_2 = run_optimization_problem(HeatProblem,
                                              base_folder=base_folder)
        np.testing.assert_allclose(solution_1.objective_value,
                                   solution_2.objective_value)

        # To minimize investment cost, max heat supply from HeatProducer
        # matches the installed capacity (max size).
        # This also verifies that the constraint is respected
        results = solution_1.extract_results()
        heat_producers = [1, 2]
        decimal = 3
        for i in heat_producers:
            capacity = np.max(results[f"HeatProducer_{i}.Heat_source"])
            max_size = results[f"HeatProducer_{i}__max_size"]
            err_msg_assert_2 = f"Max heat supply from HeatProducer_{i} is not sufficiently close to its supply capacity"
            np.testing.assert_almost_equal(capacity, max_size, decimal, 
                                           err_msg_assert_2)


        # Here we check that the discounted solution cheaper than the non-discounted one
        # check the values of the total investment.
        solution = run_optimization_problem(HeatProblemDiscAnnualizedCost, base_folder=base_folder)

        results = solution.extract_results()
        


if __name__ == "__main__":
    import time

    start_time = time.time()
    a = TestEndScenarioSizingAnnualized()
    a.test_end_scenario_sizing_annualized()
    print("Execution time: " + time.strftime("%M:%S", time.gmtime(time.time() - start_time)))
