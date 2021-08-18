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


class TestMinMaxPressureOptions(TestCase):

    import models.basic_source_and_demand.src.heat_comparison as heat_comparison
    from models.basic_source_and_demand.src.heat_comparison import HeatPython

    base_folder = Path(heat_comparison.__file__).resolve().parent.parent
    min_pressure = 4.0
    max_pressure = 12.0

    class SmallerPipes(HeatPython):
        # We want to force the dynamic pressure in the system to be higher
        # than 12 - 4 = 8 bar (typical upper and lower bounds). We make the
        # pipes smaller in diameter/area to accomplish this. We also adjust
        # the minimum heat delivered by the source, such that a maximum
        # pressure range can force this to go lower than usual.
        def parameters(self, ensemble_member):
            parameters = super().parameters(ensemble_member)
            for p in self.heat_network_components["pipe"]:
                parameters[f"{p}.diameter"] = 0.04
                parameters[f"{p}.area"] = 0.25 * 3.14159265 * parameters[f"{p}.diameter"] ** 2
            return parameters

        def bounds(self):
            bounds = super().bounds()
            bounds["source.Heat_source"] = (0.0, 125_000.0)
            return bounds

    class MinPressure(SmallerPipes):
        def heat_network_options(self):
            options = super().heat_network_options()
            assert "pipe_minimum_pressure" in options
            options["pipe_minimum_pressure"] = TestMinMaxPressureOptions.min_pressure
            return options

    class MaxPressure(SmallerPipes):
        def heat_network_options(self):
            options = super().heat_network_options()
            assert "pipe_maximum_pressure" in options
            options["pipe_maximum_pressure"] = TestMinMaxPressureOptions.max_pressure
            return options

    class MinMaxPressure(SmallerPipes):
        def heat_network_options(self):
            options = super().heat_network_options()
            options["pipe_minimum_pressure"] = TestMinMaxPressureOptions.min_pressure
            options["pipe_maximum_pressure"] = TestMinMaxPressureOptions.max_pressure
            return options

    def test_min_max_pressure_options(self):
        case_default = run_optimization_problem(self.SmallerPipes, base_folder=self.base_folder)
        case_min_pressure = run_optimization_problem(self.MinPressure, base_folder=self.base_folder)
        case_max_pressure = run_optimization_problem(self.MaxPressure, base_folder=self.base_folder)
        case_min_max_pressure = run_optimization_problem(
            self.MinMaxPressure, base_folder=self.base_folder
        )

        def _get_min_max_pressure(case):
            min_head = np.inf
            max_head = -np.inf

            results = case.extract_results()
            for p in case.heat_network_components["pipe"]:
                min_head_in = min(results[f"{p}.H_in"])
                min_head_out = min(results[f"{p}.H_out"])
                min_head = min([min_head, min_head_in, min_head_out])

                max_head_in = max(results[f"{p}.H_in"])
                max_head_out = max(results[f"{p}.H_out"])
                max_head = max([max_head, max_head_in, max_head_out])

            return min_head / 10.2, max_head / 10.2

        min_, max_ = _get_min_max_pressure(case_default)
        self.assertGreater(max_ - min_, self.max_pressure - self.min_pressure)
        base_objective_value = case_default.objective_value

        min_, max_ = _get_min_max_pressure(case_min_pressure)
        self.assertGreater(min_, self.min_pressure * 0.99)
        self.assertGreater(max_, self.max_pressure)
        self.assertAlmostEqual(case_min_pressure.objective_value, base_objective_value, 4)

        min_, max_ = _get_min_max_pressure(case_max_pressure)
        self.assertLess(min_, self.min_pressure)
        self.assertLess(max_, self.max_pressure * 1.01)
        self.assertAlmostEqual(case_max_pressure.objective_value, base_objective_value, 4)

        min_, max_ = _get_min_max_pressure(case_min_max_pressure)
        self.assertGreater(min_, self.min_pressure * 0.99)
        self.assertLess(max_, self.max_pressure * 1.01)
        self.assertGreater(case_min_max_pressure.objective_value, base_objective_value * 1.5)
