from pathlib import Path
from unittest import TestCase

import numpy as np

from rtctools.util import run_optimization_problem


class TestEndScenarioSizingAnnualized(TestCase):
    def test_end_scenario_sizing_annualized(self):
        from models.test_case_small_network_optional_assets_annualized.src.run_annualized import (
            HeatProblemDiscAnnualizedCost,
            HeatProblemDiscAnnualizedCost_Modified_Param,
        )
        from models.test_case_small_network_optional_assets_annualized.src.run_ates import (
            HeatProblem,
        )
        import models.test_case_small_network_optional_assets_annualized.src.run_annualized as run_annualized
        from rtctools_heat_network.heat_mixin import calculate_annuity_factor

        base_folder = Path(run_annualized.__file__).resolve().parent.parent

        # Assertion 1: Non annualized objective value with discount=0 and technical life 1
        # year matches the objective value of the discounted problem
        solution_1 = run_optimization_problem(
            HeatProblemDiscAnnualizedCost_Modified_Param, base_folder=base_folder
        )
        solution_2 = run_optimization_problem(HeatProblem, base_folder=base_folder)

        np.testing.assert_allclose(solution_1.objective_value, solution_2.objective_value)

        # To minimize investment cost, max heat supply from HeatProducer
        # matches the installed capacity (max size).
        # This also verifies that the constraint is respected
        results = solution_1.extract_results()
        heat_producers = [1, 2]
        decimal = 3
        discount_rate = 0.0
        years_asset_life = 1
        for i in heat_producers:
            # Assertion 3:
            capacity = np.max(results[f"HeatProducer_{i}.Heat_source"])
            max_size = results[f"HeatProducer_{i}__max_size"]
            np.testing.assert_almost_equal(capacity, max_size, decimal)
            # Assertion 3: Non-discounted total cost
            investment_and_installation_cost = (
                results[f"HeatProducer_{i}__investment_cost"]
                + results[f"HeatProducer_{i}__installation_cost"]
            )
            discount_factor = calculate_annuity_factor(discount_rate, years_asset_life)
            annualized_capex = results[f"HeatProducer_{i}__annualized_capex"]
            np.testing.assert_almost_equal(
                annualized_capex,
                investment_and_installation_cost * discount_factor,
                decimal,
            )

        # Here we check that the discounted solution cheaper than the non-discounted one
        # check the values of the total investment.
        solution = run_optimization_problem(HeatProblemDiscAnnualizedCost, base_folder=base_folder)
        results = solution.extract_results()
        for i in heat_producers:
            capacity = np.max(results[f"HeatProducer_{i}.Heat_source"])
            max_size = results[f"HeatProducer_{i}__max_size"]
            np.testing.assert_almost_equal(capacity, max_size, decimal)
            investment_and_installation_cost = (
                results[f"HeatProducer_{i}__investment_cost"]
                + results[f"HeatProducer_{i}__installation_cost"]
            )

            years_asset_life = 25
            discount_rate = 0.05
            discount_factor = calculate_annuity_factor(discount_rate, years_asset_life)
            annualized_capex = results[f"HeatProducer_{i}__annualized_capex"]
            np.testing.assert_almost_equal(
                annualized_capex,
                investment_and_installation_cost * discount_factor,
                decimal,
            )


if __name__ == "__main__":
    import time

    start_time = time.time()
    a = TestEndScenarioSizingAnnualized()
    a.test_end_scenario_sizing_annualized()
    print("Execution time: " + time.strftime("%M:%S", time.gmtime(time.time() - start_time)))

# TODO:
# Include attributes discount factor and technical life in each asset
# Include all assets in  __annualized_capex_constraints (heat_mixin)
