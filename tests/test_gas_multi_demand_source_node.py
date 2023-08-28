from pathlib import Path
from unittest import TestCase

import numpy as np


from rtctools.util import run_optimization_problem


class TestMILPGasMultiDemandSourceNode(TestCase):
    """Unit tests for the MILP test case of 2 demands, 2 sources and a node"""

    def test_multi_demand_source_node(self):
        """Test to verify that head is equal for all ports at a node. And throughout the network
        the flow (mass) balance is maintained"""
        import models.unit_cases_gas.multi_demand_source_node.src.run_test as example
        from models.unit_cases_gas.multi_demand_source_node.src.run_test import GasProblem

        base_folder = Path(example.__file__).resolve().parent.parent

        results = run_optimization_problem(GasProblem, base_folder=base_folder).extract_results()

        # Test head at node
        for i in range(1, 5):
            np.testing.assert_allclose(
                results[f"Joint_17c4.GasConn[{i}].H"], results["Joint_17c4.H"]
            )

        # Test if head is going down
        np.testing.assert_allclose(
            results["GasDemand_47d0.Gas_demand_flow"] + results["GasDemand_7978.Gas_demand_flow"],
            results["GasProducer_3573.Gas_source_flow"]
            + results["GasProducer_a977.Gas_source_flow"],
        )
