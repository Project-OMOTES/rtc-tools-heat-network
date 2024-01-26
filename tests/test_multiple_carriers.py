from pathlib import Path
from unittest import TestCase

from rtctools.util import run_optimization_problem

from rtctools_heat_network.esdl.esdl_parser import ESDLFileParser
from rtctools_heat_network.esdl.profile_parser import ProfileReaderFromFile


class TestMultipleCarriers(TestCase):
    def test_multiple_carriers(self):
        """
        Test to check optimzation is functioning as expected for a problem where two hydraulically
        decoupled networks which have no interaction.

        Checks:
        - Heat to discharge relation for both networks are not interferring with each other

        Missing:
        - This check is based upon the old relative heat formulation should be updated
        - energy conservation check.

        """
        import models.multiple_carriers.src.run_multiple_carriers as run_multiple_carriers
        from models.multiple_carriers.src.run_multiple_carriers import (
            HeatProblem,
        )

        base_folder = Path(run_multiple_carriers.__file__).resolve().parent.parent

        solution = run_optimization_problem(
            HeatProblem, base_folder=base_folder, esdl_file_name="MultipleCarrierTest.esdl",
            esdl_parser=ESDLFileParser, profile_reader=ProfileReaderFromFile,
            input_timeseries_file="timeseries_import.xml"
        )

        results = solution.extract_results()

        heat_demand_3222 = results["HeatingDemand_3322.Heat_demand"]
        heat_demand_18aa = results["HeatingDemand_18aa.Heat_demand"]
        heat_demand_3222_q = results["HeatingDemand_3322.Q"]
        heat_demand_18aa_q = results["HeatingDemand_18aa.Q"]

        # Values used in non_storage_component.py
        cp = 4200.0
        rho = 988.0

        # We check for a system consisting out of 2 hydraulically decoupled networks that the energy
        # balance equations are done with the correct carrier.
        test = TestCase()
        test.assertTrue(expr=all(heat_demand_3222 <= heat_demand_3222_q * rho * cp * 30))
        test.assertTrue(expr=all(heat_demand_18aa <= heat_demand_18aa_q * rho * cp * 40))
