from pathlib import Path
from unittest import TestCase

from rtctools.util import run_optimization_problem

from utils_tests import demand_matching_test, energy_conservation_test, heat_to_discharge_test


class TestWarmingUpUnitCases(TestCase):
    def test_1a(self):
        """
        This is the most basic check where we have a simple network and check for the basic physics.
        This simple network includes one source, pipes, a node, and 3 demands.

        Checks;
        - Demand matching
        - Energy conservation
        - Heat to discharge

        """
        import models.unit_cases.case_1a.src.run_1a as run_1a
        from models.unit_cases.case_1a.src.run_1a import HeatProblem

        base_folder = Path(run_1a.__file__).resolve().parent.parent

        # Just a "problem is not infeasible"
        heat_problem = run_optimization_problem(HeatProblem, base_folder=base_folder)

        demand_matching_test(heat_problem, heat_problem.extract_results())
        energy_conservation_test(heat_problem, heat_problem.extract_results())
        heat_to_discharge_test(heat_problem, heat_problem.extract_results())

    def test_2a(self):
        """
        This is the most basic check where we have a simple network and check for the basic physics.
        This simple network includes two source, pipes, nodes, and 3 demands.

        Checks;
        - Demand matching
        - Energy conservation
        - Heat to discharge

        """
        import models.unit_cases.case_2a.src.run_2a as run_2a
        from models.unit_cases.case_2a.src.run_2a import HeatProblem

        base_folder = Path(run_2a.__file__).resolve().parent.parent

        # Just a "problem is not infeasible"
        heat_problem = run_optimization_problem(HeatProblem, base_folder=base_folder)

        demand_matching_test(heat_problem, heat_problem.extract_results())
        energy_conservation_test(heat_problem, heat_problem.extract_results())
        heat_to_discharge_test(heat_problem, heat_problem.extract_results())

    def test_3a(self):
        """
        This is the most basic check where we have a simple network and check for the basic physics.
        This simple network includes one source, pipes, node, a tank storage and 3 demands.

        Checks;
        - Demand matching
        - Energy conservation
        - Heat to discharge

        """
        import models.unit_cases.case_3a.src.run_3a as run_3a
        from models.unit_cases.case_3a.src.run_3a import HeatProblem

        base_folder = Path(run_3a.__file__).resolve().parent.parent

        # Just a "problem is not infeasible"
        heat_problem = run_optimization_problem(HeatProblem, base_folder=base_folder)

        demand_matching_test(heat_problem, heat_problem.extract_results())
        energy_conservation_test(heat_problem, heat_problem.extract_results())
        heat_to_discharge_test(heat_problem, heat_problem.extract_results())
