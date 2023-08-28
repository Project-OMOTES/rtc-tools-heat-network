from pathlib import Path
from unittest import TestCase

import numpy as np

from rtctools.util import run_optimization_problem


class TestAssetIsRealized(TestCase):
    def test_asset_is_realized(self):
        r"""
        This is a test to check the behaviour of the cumulative investments made and the
        asset is realized variable. We want the asset only to become available once sufficient
        investments are made.

        In this specific test we optimize to match the heat demand. However, the sources are not
        available from the start as the cumulative invesments made at timestep 0 is 0. Furthermore,
        there is a cap on the investments that can be done per timestep. We expect the optimizer
        to find a solution that releases the sources as soon as possible in order to match demand
        and the demand not to be matched until that point in time.
        """
        import models.test_case_small_network_with_ates.src.run_ates as run_ates
        from models.test_case_small_network_with_ates.src.run_ates import (
            HeatProblemPlacingOverTime,
        )

        base_folder = Path(run_ates.__file__).resolve().parent.parent

        # This is an optimization done over 25 timesteps with a cap on how quickly the cost
        # for the 2 producers can be realized
        solution = run_optimization_problem(HeatProblemPlacingOverTime, base_folder=base_folder)

        results = solution.extract_results()

        # First we test whether the investments made are below cap
        cap = 2.5e5 + 1.0e-3  # some small tolerance, CBC...
        np.testing.assert_allclose(
            True, np.diff(results["HeatProducer_1__cumulative_investments_made_in_eur"]) <= cap
        )
        np.testing.assert_allclose(
            True, np.diff(results["HeatProducer_2__cumulative_investments_made_in_eur"]) <= cap
        )

        # Now we test if the investments made are greater then the needed investments once the
        # asset is realized
        inds = np.where(np.round(results["HeatProducer_1__asset_is_realized"]) == 1)
        np.testing.assert_allclose(
            True,
            (
                results["HeatProducer_1__cumulative_investments_made_in_eur"][inds]
                >= results["HeatProducer_1__investment_cost"]
                + results["HeatProducer_1__installation_cost"]
                - 1.0e-3
            ),
        )
        np.testing.assert_allclose(
            True,
            (
                results["HeatProducer_2__cumulative_investments_made_in_eur"][inds]
                >= results["HeatProducer_2__investment_cost"]
                + results["HeatProducer_2__installation_cost"]
                - 1.0e-3
            ),
        )

        # Here we test that the asset is not used until it is actually realized
        inds_not = np.where(np.round(results["HeatProducer_1__asset_is_realized"]) == 0)
        np.testing.assert_allclose(results["HeatProducer_1.Heat_source"][inds_not], 0.0)
        np.testing.assert_allclose(results["HeatProducer_1.Heat_source"][inds_not], 0.0)

        # Here we test that the asset is actually used once it is realized
        np.testing.assert_allclose(results["HeatProducer_1.Heat_source"][inds] > 0.0, True)
        np.testing.assert_allclose(results["HeatProducer_1.Heat_source"][inds] > 0.0, True)


if __name__ == "__main__":
    import time

    start_time = time.time()
    a = TestAssetIsRealized()
    a.test_asset_is_realized()
    print("Execution time: " + time.strftime("%M:%S", time.gmtime(time.time() - start_time)))
