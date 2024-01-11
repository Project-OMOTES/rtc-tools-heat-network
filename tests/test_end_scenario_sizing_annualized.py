from pathlib import Path
from unittest import TestCase

import numpy as np

from rtctools.util import run_optimization_problem


class TestEndScenarioSizingAnnualized(TestCase):
    """
    Tests for end scenario sizing with annualized costs in a small network with optional assets.

    This class tests various scenarios using different models for a heat network, both with
    and without annualized costs. It asserts the following:
    1. Under some conditions, the objective value of the annualized model is equal to the solution
    from the non annualized one.
    2. The effect of the discount rate on the objective value.
    3. The correctness of annualized capital expenditures calculations.
    """

    def test_end_scenario_sizing_annualized(self):
        from models.test_case_small_network_optional_assets_annualized.src.run_annualized import (
            HeatProblemDiscAnnualizedCost,
            HeatProblemDiscAnnualizedCostModifiedParam,
            HeatProblemDiscAnnualizedCostModifiedDiscountRate,
        )
        from models.test_case_small_network_optional_assets_annualized.src.run_ates import (
            HeatProblem,
        )
        from models.test_case_small_network_optional_assets_annualized.src import (
            run_annualized,
        )
        from rtctools_heat_network.heat_mixin import calculate_annuity_factor

        base_folder = Path(run_annualized.__file__).resolve().parent.parent

        solution_run_ates = run_optimization_problem(HeatProblem, base_folder=base_folder)

        solution_annualized_cost = run_optimization_problem(
            HeatProblemDiscAnnualizedCost, base_folder=base_folder
        )

        solution__annualized_modified_param = run_optimization_problem(
            HeatProblemDiscAnnualizedCostModifiedParam, base_folder=base_folder
        )

        solution__annualized_modified_discount = run_optimization_problem(
            HeatProblemDiscAnnualizedCostModifiedDiscountRate, base_folder=base_folder
        )

        # Assertion 1: Model for annualized objective value with discount=0 and
        # technical life 1 year matches the objective value of the non-discounted problem
        np.testing.assert_allclose(
            solution__annualized_modified_param.objective_value,
            solution_run_ates.objective_value,
        )

        # Assertion 2: Undiscounted problem has a lower objective value than the discocunted one
        # due to cost of capital not considered in undiscounted problem
        np.testing.assert_array_less(
            solution__annualized_modified_discount.objective_value,
            solution_annualized_cost.objective_value,
        )

        results = solution_annualized_cost.extract_results()
        heat_producers = [1, 2]
        decimal = 3
        discount_rate = 0.0
        years_asset_life = 1
        for i in heat_producers:
            investment_and_installation_cost = (
                results[f"HeatProducer_{i}__investment_cost"]
                + results[f"HeatProducer_{i}__installation_cost"]
            )
            # These are the same parameters used by the model from the ESDL file:
            years_asset_life = 25
            discount_rate = 0.05
            discount_factor = calculate_annuity_factor(discount_rate, years_asset_life)
            annualized_capex = results[f"HeatProducer_{i}__annualized_capex"]
            # Assertion 3: annualized capex matches the discounted investment
            # and installation cost
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
