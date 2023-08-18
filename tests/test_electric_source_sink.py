from pathlib import Path
from unittest import TestCase

import numpy as np


from rtctools.util import run_optimization_problem

# TODO: still have to make test where elecitricity direction is switched:
# e.g. 2 nodes, with at each node a producer and consumer, first one node medium demand, second
# small demand and then increase the demand of the second node such that direction changes


class TestMILPElectricSourceSink(TestCase):
    """Unit tests for the MILP test case of a source, a cable, a sink"""

    def test_source_sink(self):
        """Test to verify if when a target it set that is more or less than
        then max or min of the respective component it is capped. 1000"""
        import models.unit_cases_electricity.source_sink_cable.src.example as example
        from models.unit_cases_electricity.source_sink_cable.src.example import ElectricityProblem

        base_folder = Path(example.__file__).resolve().parent.parent
        max_ = 1000.0  # This max is set in the esdl file
        v_min = 230  # set as minimum voltage for cables
        tol = 1e-10

        results = run_optimization_problem(
            ElectricityProblem, base_folder=base_folder
        ).extract_results()

        # Test if capping is ok
        power_consumed = results["ElectricityDemand_2af6.ElectricityIn.Power"]
        smallerthen = all(power_consumed <= np.ones(len(power_consumed)) * max_)
        self.assertTrue(smallerthen)
        biggerthen = all(power_consumed >= np.zeros(len(power_consumed)))
        self.assertTrue(biggerthen)

        # Test energy conservation
        power_consumed = results["ElectricityDemand_2af6.ElectricityIn.Power"]
        power_delivered = results["ElectricityProducer_b95d.ElectricityOut.Power"]
        power_loss = results["ElectricityCable_238f.Power_loss"]
        total_power_dissipation = power_consumed + power_loss
        self.assertIsNone(
            np.testing.assert_allclose(total_power_dissipation, power_delivered, rtol=1e-4),
            msg="No energy conservation. Total demand is not equal to total delivery.",
        )
        biggerthen = all(power_loss >= np.zeros(len(power_loss)))
        self.assertTrue(biggerthen)

        # Test that voltage goes down
        v_in = results["ElectricityCable_238f.ElectricityIn.V"]
        v_out = results["ElectricityCable_238f.ElectricityOut.V"]
        np.testing.assert_array_less(v_out, v_in)
        biggerthen = all(v_out >= (v_min - tol) * np.ones(len(v_out)))
        self.assertTrue(biggerthen)

    def test_source_sink_max_curr(self):
        """Test to verify if when a target it set that is more or less than
        then max or min of the respective component it is capped. 1000"""
        import models.unit_cases_electricity.source_sink_cable.src.example as example
        from models.unit_cases_electricity.source_sink_cable.src.example import (
            ElectricityProblemMaxCurr,
        )

        base_folder = Path(example.__file__).resolve().parent.parent
        max_ = 32660.0  # This max is based on max current and voltage requirement at consumer
        v_min = 230  # set as minimum voltage for cables

        results = run_optimization_problem(
            ElectricityProblemMaxCurr, base_folder=base_folder
        ).extract_results()

        tolerance = 1e-10  # due to computational comparison

        # Test if capping is ok (capping based on max power as result of v_min*Imax)
        power_consumed = results["ElectricityDemand_2af6.ElectricityIn.Power"]
        smallerthen = all(power_consumed - tolerance <= np.ones(len(power_consumed)) * max_)
        self.assertTrue(smallerthen)
        biggerthen = all(power_consumed >= np.zeros(len(power_consumed)))
        self.assertTrue(biggerthen)

        # Test energy conservation
        power_consumed = results["ElectricityDemand_2af6.ElectricityIn.Power"]
        power_delivered = results["ElectricityProducer_b95d.ElectricityOut.Power"]
        power_loss = results["ElectricityCable_238f.Power_loss"]
        total_power_dissipation = power_consumed + power_loss
        self.assertIsNone(
            np.testing.assert_allclose(total_power_dissipation, power_delivered, rtol=1e-4),
            msg="No energy conservation. Total demand is not equal to total delivery.",
        )
        biggerthen = all(power_loss >= np.zeros(len(power_loss)))
        self.assertTrue(biggerthen)

        # Test that voltage goes down
        v_in = results["ElectricityCable_238f.ElectricityIn.V"]
        v_out = results["ElectricityCable_238f.ElectricityOut.V"]
        np.testing.assert_array_less(v_out, v_in)
        biggerthen = all(v_out >= v_min * np.ones(len(v_out)) - tolerance)
        self.assertTrue(biggerthen)

        # Test that max current is not exceeded and is constant along path (since no nodes included)
        current_demand = results["ElectricityDemand_2af6.ElectricityIn.I"]
        current_producer = results["ElectricityProducer_b95d.ElectricityOut.I"]
        current_cable = results["ElectricityCable_238f.ElectricityOut.I"]
        np.testing.assert_allclose(current_demand, current_cable)
        np.testing.assert_allclose(current_cable, current_producer)
        biggerthen = all(142.0 * np.ones(len(current_demand)) >= current_demand - tolerance)
        self.assertTrue(biggerthen)
