from pathlib import Path
from unittest import TestCase

from rtctools.util import run_optimization_problem

from rtctools_heat_network.esdl.esdl_parser import ESDLFileParser
from rtctools_heat_network.esdl.profile_parser import ProfileReaderFromFile

from utils_tests import demand_matching_test, energy_conservation_test, heat_to_discharge_test


class TestAbsoluteHeat(TestCase):
    def test_absolute_heat(self):
        """
        This is a single line ring model, meaning that there are no dedicated supply or return
        lines. This means that this model pipes are not related (no relation between hot and cold
        pipes exists).

        Checks:
        1. demand is matched
        2. energy conservation in the network
        3. heat to discharge

        """
        import models.absolute_heat.src.example as example
        from models.absolute_heat.src.example import HeatProblem

        base_folder = Path(example.__file__).resolve().parent.parent

        heat_problem = run_optimization_problem(
            HeatProblem,
            base_folder=base_folder,
            esdl_file_name="absolute_heat.esdl",
            esdl_parser=ESDLFileParser,
            profile_reader=ProfileReaderFromFile,
            input_timeseries_file="timeseries.csv",
        )

        demand_matching_test(heat_problem, heat_problem.extract_results())
        energy_conservation_test(heat_problem, heat_problem.extract_results())
        heat_to_discharge_test(heat_problem, heat_problem.extract_results())
