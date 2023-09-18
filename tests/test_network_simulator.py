from pathlib import Path
from unittest import TestCase

import numpy as np

from rtctools.util import run_optimization_problem

from rtctools_heat_network.workflows import NetworkSimulatorHIGHSTestCase

from utils_tests import demand_matching_test, energy_conservation_test, heat_to_discharge_test


class TestNetworkSimulator(TestCase):
    """
    In this test case 2 heat producers and an ATES is used to supply 3 heating demands. A merit
    order (preference of 1st use) is given to the producers: Producer_1 = 2 and Producer_2 = 1.

    Testing:
    - General checks namely demand matching, energy conservation and asset heat variable vs
      calculated heat (based on flow rate)
    - Check that producer 1 (merit oder = 2) is only used for the supply of heat lossed in the
      connected and is does not contribute to the heating demands 1, 2 and 3
    - Check that the ATES is not delivering any heat to the network during the 1st time step
    """

    def test_network_simulator(self):
        import models.test_case_small_network_with_ates.src.run_ates as run_ates

        base_folder = Path(run_ates.__file__).resolve().parent.parent

        solution = run_optimization_problem(NetworkSimulatorHIGHSTestCase, base_folder=base_folder)

        results = solution.extract_results()

        # General checks
        demand_matching_test(solution, results)
        energy_conservation_test(solution, results)
        heat_to_discharge_test(solution, results)

        # Check that producer 1 (merit oder = 2) is only used for the supply of heat lossed in the
        # connected and is does not contribute to the heating demands 1, 2 and 3
        np.testing.assert_allclose(
            results["Pipe1__hn_heat_loss"] - results["HeatProducer_1.Heat_source"],
            0.0,
            err_msg="Heat producer 1 should only cater for the heat loss "
            "in the connected pipe, due to producer 2 being sufficient for "
            "the total heat demand and it has the 1st priority for usage",
            rtol=1.0e-3,
            atol=1.0e-3,
        )
        # Check ATES
        np.testing.assert_array_less(
            results["ATES_033c.Heat_ates"][0],
            0.0,
            err_msg="ATES should not be delivering heat to the network in the 1st time step",
        )


if __name__ == "__main__":
    import time

    start_time = time.time()
    a = TestNetworkSimulator()
    a.test_network_simulator()
    print("Execution time: " + time.strftime("%M:%S", time.gmtime(time.time() - start_time)))
