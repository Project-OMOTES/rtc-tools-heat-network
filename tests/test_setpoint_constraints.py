from pathlib import Path
from unittest import TestCase

import numpy as np

from rtctools_heat_network.util import run_optimization_problem


class TestSetpointConstraints(TestCase):
    def test_setpoint_constraints(self):
        import models.unit_cases.case_3a.src.run_3a as run_3a
        from models.unit_cases.case_3a.src.run_3a import HeatProblemSetPointConstraints

        base_folder = Path(run_3a.__file__).resolve().parent.parent

        _heat_problem_3 = run_optimization_problem(
            HeatProblemSetPointConstraints,
            base_folder=base_folder,
            **{"timed_setpoints": {"GeothermalSource_b702": (45, 1)}},
        )
        results_3 = _heat_problem_3.extract_results()

        _heat_problem_4 = run_optimization_problem(
            HeatProblemSetPointConstraints,
            base_folder=base_folder,
            **{"timed_setpoints": {"GeothermalSource_b702": (45, 0)}},
        )
        results_4 = _heat_problem_4.extract_results()

        # Check that solution has one setpoint change
        a = abs(
            results_3["GeothermalSource_b702.Heat_source"][2:]
            - results_3["GeothermalSource_b702.Heat_source"][1:-1]
        )
        np.testing.assert_allclose((a >= 1.0e-6).sum(), 1)

        # Check that solution has no setpoint change
        np.testing.assert_array_less(
            abs(
                results_4["GeothermalSource_b702.Heat_source"][2:]
                - results_4["GeothermalSource_b702.Heat_source"][1:-1]
            ),
            1.0e-6,
        )
