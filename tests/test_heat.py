from pathlib import Path
from unittest import TestCase

import numpy as np

from rtctools.util import run_optimization_problem


class TestDoublePipeHeat(TestCase):
    def test_heat_loss(self):
        import models.double_pipe_heat.src.double_pipe_heat as double_pipe_heat
        from models.double_pipe_heat.src.double_pipe_heat import DoublePipeEqualHeat

        base_folder = Path(double_pipe_heat.__file__).resolve().parent.parent

        case = run_optimization_problem(DoublePipeEqualHeat, base_folder=base_folder)
        results = case.extract_results()

        source = results["source.Heat_source"]
        demand = results["demand.Heat_demand"]

        # With non-zero heat losses in pipes, the demand should always be
        # strictly lower than what is produced.
        np.testing.assert_array_less(demand, source)
