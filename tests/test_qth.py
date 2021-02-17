import logging
from pathlib import Path
from unittest import TestCase

import numpy as np

from rtctools.util import run_optimization_problem


logger = logging.getLogger("rtctools_heat_network")


class TestArtificalHeadLoss(TestCase):
    @classmethod
    def setUpClass(cls):
        import models.double_pipe_qth.src.double_pipe_qth as double_pipe_qth

        cls.base_folder = Path(double_pipe_qth.__file__).resolve().parent.parent

    def test_single_pipe(self):
        from models.double_pipe_qth.src.double_pipe_qth import SinglePipeQTH

        with self.assertLogs(logger, level="WARNING") as cm:
            run_optimization_problem(SinglePipeQTH, base_folder=self.base_folder)

            self.assertNotIn("artificial head loss", str(cm.output))

            if not cm.output:
                # Prevent __exit__ from failing on AssertionError
                logger.warning("Test succeeded")

    def test_double_pipe_unequal(self):
        from models.double_pipe_qth.src.double_pipe_qth import DoublePipeUnequalQTH

        with self.assertLogs(logger, level="WARNING") as cm:
            run_optimization_problem(DoublePipeUnequalQTH, base_folder=self.base_folder)

            self.assertIn("Pipe pipe_2_hot has artificial head loss", str(cm.output))
            self.assertIn("Pipe pipe_2_cold has artificial head loss", str(cm.output))

            self.assertNotIn("Pipe pipe_1_hot has artificial head loss", str(cm.output))
            self.assertNotIn("Pipe pipe_1_cold has artificial head loss", str(cm.output))

    def test_double_pipe_unequal_with_valve(self):
        from models.double_pipe_qth.src.double_pipe_qth import DoublePipeUnequalWithValveQTH

        with self.assertLogs(logger, level="WARNING") as cm:
            run_optimization_problem(DoublePipeUnequalWithValveQTH, base_folder=self.base_folder)

            self.assertNotIn("artificial head loss", str(cm.output))

            if not cm.output:
                # Prevent __exit__ from failing on AssertionError
                logger.warning("Test succeeded")


class TestHeadLossEqualities(TestCase):
    @classmethod
    def setUpClass(cls):
        import models.double_pipe_qth.src.cq2_inequality_vs_equality as cq2_inequality_vs_equality

        cls.base_folder = Path(cq2_inequality_vs_equality.__file__).resolve().parent.parent

    def test_equal_length(self):

        from models.double_pipe_qth.src.cq2_inequality_vs_equality import (
            EqualLengthLinearEquality,
            EqualLengthQuadraticEquality,
        )

        with self.assertLogs(logger, level="WARNING") as cm:
            equal_length_linear_equality = run_optimization_problem(
                EqualLengthLinearEquality, base_folder=self.base_folder
            )

            self.assertNotIn("artificial head loss", str(cm.output))

            if not cm.output:
                # Prevent __exit__ from failing on AssertionError
                logger.warning("Test succeeded")

        with self.assertLogs(logger, level="WARNING") as cm:
            equal_length_quadratic_equality = run_optimization_problem(
                EqualLengthQuadraticEquality, base_folder=self.base_folder
            )

            self.assertNotIn("artificial head loss", str(cm.output))

            if not cm.output:
                # Prevent __exit__ from failing on AssertionError
                logger.warning("Test succeeded")

        # Check results of linear head loss equation
        results_linear = equal_length_linear_equality.extract_results()

        linear_q_1 = results_linear["pipe_1_hot.Q"]
        linear_q_2 = results_linear["pipe_2_hot.Q"]
        linear_ratio = linear_q_1 / linear_q_2

        np.testing.assert_allclose(linear_ratio, 1.0)

        # Check results of quadratic head loss equation
        results_quadratic = equal_length_quadratic_equality.extract_results()

        quadratic_q_1 = results_quadratic["pipe_1_hot.Q"]
        quadratic_q_2 = results_quadratic["pipe_2_hot.Q"]
        quadratic_ratio = quadratic_q_1 / quadratic_q_2

        np.testing.assert_allclose(quadratic_ratio, 1.0)

    def test_unequal_length(self):

        from models.double_pipe_qth.src.cq2_inequality_vs_equality import (
            UnequalLengthLinearEquality,
            UnequalLengthQuadraticEquality,
        )

        with self.assertLogs(logger, level="WARNING") as cm:
            equal_length_linear_equality = run_optimization_problem(
                UnequalLengthLinearEquality, base_folder=self.base_folder
            )

            self.assertNotIn("artificial head loss", str(cm.output))

            if not cm.output:
                # Prevent __exit__ from failing on AssertionError
                logger.warning("Test succeeded")

        with self.assertLogs(logger, level="WARNING") as cm:
            equal_length_quadratic_equality = run_optimization_problem(
                UnequalLengthQuadraticEquality, base_folder=self.base_folder
            )

            self.assertNotIn("artificial head loss", str(cm.output))

            if not cm.output:
                # Prevent __exit__ from failing on AssertionError
                logger.warning("Test succeeded")

        # Check results of linear head loss equation
        results_linear = equal_length_linear_equality.extract_results()

        linear_q_1 = results_linear["pipe_1_hot.Q"]
        linear_q_2 = results_linear["pipe_2_hot.Q"]
        linear_ratio = linear_q_1 / linear_q_2

        np.testing.assert_allclose(linear_ratio, 4.0)

        # Check results of quadratic head loss equation
        results_quadratic = equal_length_quadratic_equality.extract_results()

        quadratic_q_1 = results_quadratic["pipe_1_hot.Q"]
        quadratic_q_2 = results_quadratic["pipe_2_hot.Q"]
        quadratic_ratio = quadratic_q_1 / quadratic_q_2

        np.testing.assert_allclose(quadratic_ratio, 2.0)

    def test_unequal_length_valve(self):

        from models.double_pipe_qth.src.cq2_inequality_vs_equality import (
            UnequalLengthValveLinearEquality,
            UnequalLengthValveQuadraticEquality,
        )

        with self.assertLogs(logger, level="WARNING") as cm:
            equal_length_linear_equality = run_optimization_problem(
                UnequalLengthValveLinearEquality, base_folder=self.base_folder
            )

            self.assertNotIn("artificial head loss", str(cm.output))

            if not cm.output:
                # Prevent __exit__ from failing on AssertionError
                logger.warning("Test succeeded")

        with self.assertLogs(logger, level="WARNING") as cm:
            equal_length_quadratic_equality = run_optimization_problem(
                UnequalLengthValveQuadraticEquality, base_folder=self.base_folder
            )

            self.assertNotIn("artificial head loss", str(cm.output))

            if not cm.output:
                # Prevent __exit__ from failing on AssertionError
                logger.warning("Test succeeded")

        # Check results of linear head loss equation
        results_linear = equal_length_linear_equality.extract_results()

        linear_q_1 = results_linear["pipe_1_hot.Q"]
        linear_q_2 = results_linear["pipe_2_hot.Q"]
        linear_ratio = linear_q_1 / linear_q_2

        # The optimization should prefer the shorter pipe much more than would
        # follow from hydraulic equality
        np.testing.assert_array_less(10.0, linear_ratio)

        # Check results of quadratic head loss equation
        results_quadratic = equal_length_quadratic_equality.extract_results()

        quadratic_q_1 = results_quadratic["pipe_1_hot.Q"]
        quadratic_q_2 = results_quadratic["pipe_2_hot.Q"]
        quadratic_ratio = quadratic_q_1 / quadratic_q_2

        # The optimization should prefer the shorter pipe much more than would
        # follow from hydraulic equality
        np.testing.assert_array_less(10.0, quadratic_ratio)

        # Because the hydraulics (equality constraints) no longer determine
        # the results with the presence of the control valve, we expect the
        # quadratic and linear formulations to have the same result.
        np.testing.assert_allclose(linear_q_1, quadratic_q_1)
        np.testing.assert_allclose(linear_q_2, quadratic_q_2)
