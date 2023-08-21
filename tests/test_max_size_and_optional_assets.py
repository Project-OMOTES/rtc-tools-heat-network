from pathlib import Path
from unittest import TestCase

import numpy as np

from rtctools.util import run_optimization_problem


class TestMaxSizeAggregationCount(TestCase):
    def test_max_size_and_aggr_count(self):
        import models.test_case_small_network_with_ates_with_buffer.src.run_ates as run_ates
        from models.test_case_small_network_with_ates_with_buffer.src.run_ates import (
            HeatProblem,
        )

        base_folder = Path(run_ates.__file__).resolve().parent.parent

        # This is an optimization done over a few days
        solution = run_optimization_problem(HeatProblem, base_folder=base_folder)

        results = solution.extract_results()

        # Producer 1 should not produce due to higher cost
        # Producer 2 should produce
        heat_1 = results["HeatProducer_1.Heat_source"]
        heat_2 = results["HeatProducer_2.Heat_source"]
        prod_1_placed = results["HeatProducer_1_aggregation_count"]
        prod_2_placed = results["HeatProducer_2_aggregation_count"]
        var_cost_1 = results["HeatProducer_1__variable_operational_cost"]
        var_cost_2 = results["HeatProducer_2__variable_operational_cost"]
        inst_cost_1 = results["HeatProducer_1__installation_cost"]
        inst_cost_2 = results["HeatProducer_2__installation_cost"]
        inv_cost_1 = results["HeatProducer_1__investment_cost"]
        inv_cost_2 = results["HeatProducer_2__investment_cost"]
        max_size_1 = results["HeatProducer_1__max_size"]
        max_size_2 = results["HeatProducer_2__max_size"]

        # Test if source 1 is off and 2 is producing
        np.testing.assert_allclose(heat_1, 0.0)
        np.testing.assert_equal(True, heat_2[1:] > 0.0)

        # Test if source 1 is not placed and 2 is placed
        np.testing.assert_allclose(prod_1_placed, 0.0)
        np.testing.assert_equal(prod_2_placed, 1.0)

        # Test that max size is correct, note that we use an equality check as due to the cost
        # minimization they should be equal.
        np.testing.assert_allclose(max(heat_2), max_size_2)
        np.testing.assert_allclose(max_size_1, 0.0)

        # Test that investmentcost is correctly linked to max size
        np.testing.assert_allclose(
            inv_cost_2,
            solution.parameters(0)["HeatProducer_2.investment_cost_coefficient"] * max_size_2,
        )

        # Test that cost only exist for 2 and not for 1. Note the tolerances
        # to avoid test failing when heat losses slightly change
        np.testing.assert_allclose(var_cost_1, 0.0)
        np.testing.assert_allclose(var_cost_2, 6174.920222, atol=1000.0, rtol=1.0e-2)
        np.testing.assert_allclose(inst_cost_1, 0.0)
        np.testing.assert_allclose(inst_cost_2, 100000.0)
        np.testing.assert_allclose(inv_cost_1, 0.0)
        np.testing.assert_allclose(inv_cost_2, 476459.893686, atol=1000.0, rtol=1.0e-2)

        # Since the buffer and ates are not optional they must consume some heat to compensate
        # losses. Therefore we can check the max_size constraint
        np.testing.assert_allclose(
            True, results["HeatStorage_74c1.Stored_heat"] <= results["HeatStorage_74c1__max_size"]
        )
        np.testing.assert_allclose(
            True, abs(results["ATES_033c.Heat_ates"]) <= results["ATES_033c__max_size"]
        )
        np.testing.assert_allclose(results["ATES_033c_aggregation_count"], 1.0)
        np.testing.assert_allclose(results["HeatStorage_74c1_aggregation_count"], 1.0)

        import models.test_case_small_network_ates_buffer_optional_assets.src.run_ates as run_ates
        from models.test_case_small_network_ates_buffer_optional_assets.src.run_ates import (
            HeatProblem,
        )

        base_folder = Path(run_ates.__file__).resolve().parent.parent

        # This is the same problem, but now with the buffer and ates also optional.
        # Therefore we expect that the ates and buffer are no longer placed to avoid their heat
        # losses. This allows us to check if their placement constraints are proper.
        solution = run_optimization_problem(
            HeatProblem,
            base_folder=base_folder,
        )

        results = solution.extract_results()

        np.testing.assert_allclose(results["ATES_033c.Heat_ates"], 0.0)
        np.testing.assert_allclose(results["HeatStorage_74c1.Stored_heat"], 0.0)
        np.testing.assert_allclose(results["ATES_033c_aggregation_count"], 0.0)
        np.testing.assert_allclose(results["HeatStorage_74c1_aggregation_count"], 0.0)
