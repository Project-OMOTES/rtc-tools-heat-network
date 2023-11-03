from pathlib import Path
from unittest import TestCase

from rtctools.util import run_optimization_problem

from utils_tests import demand_matching_test, energy_conservation_test, heat_to_discharge_test


class TestAbsoluteHeat(TestCase):
    def test_absolute_heat(self):
        """
        This is a single line ring model, meaning that there are no dedicated supply or return
        lines. This means that this test checks whether this model is correctly parsed from esdl
        and meets the logical energy and heat to discharge constraints.
        """
        import models.absolute_heat.src.example as example
        from models.absolute_heat.src.example import HeatProblem

        base_folder = Path(example.__file__).resolve().parent.parent

        heat_problem = run_optimization_problem(HeatProblem, base_folder=base_folder)

        demand_matching_test(heat_problem, heat_problem.extract_results())
        energy_conservation_test(heat_problem, heat_problem.extract_results())
        heat_to_discharge_test(heat_problem, heat_problem.extract_results())
