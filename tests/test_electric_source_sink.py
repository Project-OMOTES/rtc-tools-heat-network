from pathlib import Path
from unittest import TestCase

import numpy as np


from rtctools.util import run_optimization_problem


class TestMILPElectricSourceSink(TestCase):
    """Unit tests for the MILP test case of a source, a cable, a sink"""

    def test_source_sink(self):
        """Test to verify if when a target it set that is more or less than
        then max or min of the respective component it is capped. 1000"""
        import models.unit_cases_electricity.source_sink_cable.src.example as example
        from models.unit_cases_electricity.source_sink_cable.src.example import ElectricityProblem

        base_folder = Path(example.__file__).resolve().parent.parent
        max_ = 1000.0  # This max is set in the esdl file

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
