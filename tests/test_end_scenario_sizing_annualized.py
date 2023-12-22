from pathlib import Path
from unittest import TestCase

import numpy as np

from rtctools.util import run_optimization_problem


class TestEndScenarioSizingAnnualized(TestCase):
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

        # # Assertion 1: Non annualized objective value with discount=0 and
        # # technical life 1 year matches the objective value of the discounted problem
        # np.testing.assert_allclose(
        #     solution__annualized_modified_param.objective_value,
        #     solution_run_ates.objective_value,
        # )

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
            capacity = np.max(results[f"HeatProducer_{i}.Heat_source"])
            max_size = results[f"HeatProducer_{i}__max_size"]
            # Asertion 3: installed capacity matches peak demand
            # To minimize investment cost, max heat supply from HeatProducer
            # matches the installed capacity (max size).
            # This also verifies that the capacity constraint is respected
            np.testing.assert_almost_equal(capacity, max_size, decimal)

            investment_and_installation_cost = (
                results[f"HeatProducer_{i}__investment_cost"]
                + results[f"HeatProducer_{i}__installation_cost"]
            )
            # These are the same parameters used by the model from the ESDL file:
            years_asset_life = 25
            discount_rate = 0.05
            discount_factor = calculate_annuity_factor(discount_rate, years_asset_life)
            annualized_capex = results[f"HeatProducer_{i}__annualized_capex"]
            # Assertion 4: annualized capex matches the discounted investment
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

# TODO:
# Include attributes discount factor and technical life in each asset
# Include all assets in  __annualized_capex_constraints (heat_mixin)
