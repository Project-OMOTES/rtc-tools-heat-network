from pathlib import Path
from unittest import TestCase

from rtctools_heat_network.util import run_heat_network_optimization


class TestWarmingUpUnitCases(TestCase):
    def test_1a(self):
        import models.unit_cases.case_1a.src.run_1a as run_1a
        from models.unit_cases.case_1a.src.run_1a import HeatProblem, QTHProblem

        base_folder = Path(run_1a.__file__).resolve().parent.parent

        # Just a "problem is not infeasible"
        _heat_problem, _qth_problem = run_heat_network_optimization(
            HeatProblem, QTHProblem, base_folder=base_folder
        )
