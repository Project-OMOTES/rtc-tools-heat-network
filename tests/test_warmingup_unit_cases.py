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

    def test_2a(self):
        import models.unit_cases.case_2a.src.run_2a as run_2a
        from models.unit_cases.case_2a.src.run_2a import HeatProblem, QTHProblem

        base_folder = Path(run_2a.__file__).resolve().parent.parent

        # Just a "problem is not infeasible"
        _heat_problem, _qth_problem = run_heat_network_optimization(
            HeatProblem, QTHProblem, base_folder=base_folder
        )

    def test_3a(self):
        import models.unit_cases.case_3a.src.run_3a as run_3a
        from models.unit_cases.case_3a.src.run_3a import HeatProblem, QTHProblem

        base_folder = Path(run_3a.__file__).resolve().parent.parent

        # Just a "problem is not infeasible"
        _heat_problem, _qth_problem = run_heat_network_optimization(
            HeatProblem, QTHProblem, base_folder=base_folder
        )
