import sys
from pathlib import Path
from unittest import TestCase

from rtctools.util import run_optimization_problem


class TestPipeDiameterSizingExample(TestCase):
    def test_half_network_gone(self):
        root_folder = str(Path(__file__).resolve().parent.parent)
        sys.path.insert(1, root_folder)

        import examples.pipe_diameter_sizing.src.example  # noqa: E402, I100
        from examples.pipe_diameter_sizing.src.example import (
            PipeDiameterSizingProblem,
        )  # noqa: E402, I100

        base_folder = (
            Path(examples.pipe_diameter_sizing.src.example.__file__).resolve().parent.parent
        )

        del root_folder
        sys.path.pop(1)

        problem = run_optimization_problem(PipeDiameterSizingProblem, base_folder=base_folder)

        parameters = problem.parameters(0)
        diameters = {p: parameters[f"{p}.diameter"] for p in problem.hot_pipes}

        # Check that half the network is removed, i.e. 4 pipes. Note that it
        # is equally possible for the left or right side of the network to be
        # removed.
        self.assertEqual(len([d for d in diameters.values() if d == 0.0]), 4)
