import numpy as np
import numpy.testing
from pathlib import Path
import sys
from unittest import TestCase

from rtctools.util import run_optimization_problem


MIP_TOLERANCE = 1e-10


class TestTopoConstraintsOnPipeDiameterSizingExample(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
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

        cls.problem = run_optimization_problem(PipeDiameterSizingProblem, base_folder=base_folder)
        cls.results = cls.problem.extract_results()

    def test_pipe_class_var(self):
        for p in self.problem.hot_pipes:
            given_pipe_classes = self.problem.pipe_classes(p)
            expected_class_vars = [f"{p}__hn_pipe_class_{pc.name}"
                                   for pc in given_pipe_classes]
            for var_name in expected_class_vars:
                self.assertTrue(var_name in self.results,
                                msg=f"{var_name} not in results")
            class_vars = \
                {var_name: value for var_name, value in self.results.items()
                 if var_name in expected_class_vars}
            for value in class_vars.values():
                self.assertTrue(
                    abs(value - 0.0) < MIP_TOLERANCE or abs(value - 1.0) < MIP_TOLERANCE
                )
            np.testing.assert_almost_equal(
                1.0, np.sum(val for val in class_vars.values()), 6,
                err_msg=f"Not exactly 1 pipe class selected for {p}"
            )
            chosen_var = None
            for var_name, value in class_vars.items():
                if abs(value - 1.0) < MIP_TOLERANCE:
                    chosen_var = var_name
            assert chosen_var is not None
            class_name = chosen_var.split("_")[-1]
            chosen_pc = [pc for pc in given_pipe_classes if pc.name == class_name]
            assert len(chosen_pc) == 1
            chosen_pc = chosen_pc[0]
            np.testing.assert_array_almost_equal(
                chosen_pc.inner_diameter, self.results[f"{p}__hn_diameter"], 5,
                err_msg=f"{p} inner diameter doesn't match expected"
            )
            np.testing.assert_array_almost_equal(
                chosen_pc.investment_costs, self.results[f"{p}__hn_cost"], 5,
                err_msg=f"{p} investment costs doesn't match expected"
            )

            expected_heat_losses = self.problem._pipe_heat_loss(
                options=self.problem.heat_network_options(),
                parameters=self.problem.parameters(0),
                p=p,
                u_values=chosen_pc.u_values
            )
            np.testing.assert_almost_equal(
                self.results[f"{p}__hn_heat_loss"], expected_heat_losses, 5
            )
            cold_pipe = self.problem.hot_to_cold_pipe(p)
            expected_heat_losses_return = self.problem._pipe_heat_loss(
                options=self.problem.heat_network_options(),
                parameters=self.problem.parameters(0),
                p=cold_pipe,
                u_values=chosen_pc.u_values
            )
            np.testing.assert_almost_equal(
                self.results[f"{cold_pipe}__hn_heat_loss"], expected_heat_losses_return, 5
            )
