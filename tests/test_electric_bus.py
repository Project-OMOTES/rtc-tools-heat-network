"""All test for a electric node/bus

Currently both the MILP and NLP tests

What at least was implement
- voltages on all connections are the same
- power in == power out
"""

from pathlib import Path
from unittest import TestCase

import numpy as np


from rtctools.util import run_optimization_problem


class TestMILPbus(TestCase):
    """MILP problem test and NLP problem test. In same test to not duplicate code"""

    def test_voltages_and_power_network1(self):
        """Test the voltage levels relative to each other. The absolute value
        (230V for instance) is determined by other components than the bus"""
        import models.unit_cases_electricity.bus_networks.src.example as example
        from models.unit_cases_electricity.bus_networks.src.example import ElectricityProblem

        base_folder = Path(example.__file__).resolve().parent.parent

        # Run the problem
        results = run_optimization_problem(
            ElectricityProblem, base_folder=base_folder
        ).extract_results()
        v1 = results["Bus_f262.ElectricityConn[1].V"]
        v2 = results["Bus_f262.ElectricityConn[2].V"]
        v_outgoing_cable = results["ElectricityCable_de9a.ElectricityIn.V"]
        v_incoming_cable = results["ElectricityCable_1ad0.ElectricityOut.V"]
        v_demand = results["ElectricityDemand_e527.ElectricityIn.V"]
        p_demand = results["ElectricityDemand_e527.ElectricityIn.Power"]
        i_demand = results["ElectricityDemand_e527.ElectricityIn.I"]
        p1 = results["Bus_f262.ElectricityConn[1].Power"]
        p2 = results["Bus_f262.ElectricityConn[2].Power"]
        p3 = results["Bus_f262.ElectricityConn[3].Power"]
        p4 = results["Bus_f262.ElectricityConn[4].Power"]

        # Incoming voltage == outgoing voltage of bus
        self.assertTrue(all(v1 == v2))
        # Ingoing voltage of bus == voltage of incoming cable
        self.assertTrue(all(v1 == v_incoming_cable))
        # Outgoing voltage of bus == voltage of outgoing cable
        self.assertTrue(all(v1 == v_outgoing_cable))
        # Power in == power out = no dissipation of power
        np.testing.assert_allclose(p1 + p2 - p3 - p4, 0.0, rtol=1.0e-6, atol=1.0e-6)
        # check if minimum voltage is reached
        np.testing.assert_array_less(230.0 - 1.0e-3, v_demand)
        # Check that current is high enough to carry the power
        np.testing.assert_array_less(p_demand - 1e-12, v_demand * i_demand)
