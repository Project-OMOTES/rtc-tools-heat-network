import logging
import sys
from pathlib import Path
from unittest import TestCase

from rtctools.util import run_optimization_problem

logger = logging.getLogger("rtctools_heat_network")


class TestArtificalHeadLoss(TestCase):
    def test_single_pipe(self):

        base_folder = (Path(__file__).parent / "models" / "double_pipe_qth").absolute()
        sys.path.insert(0, str(base_folder / "src"))
        from double_pipe_qth import SinglePipeQTH

        with self.assertLogs(logger, level="WARNING") as cm:
            run_optimization_problem(SinglePipeQTH, base_folder=base_folder)

            self.assertNotIn("artificial head loss", str(cm.output))

            if not cm.output:
                # Prevent __exit__ from failing on AssertionError
                logger.warning("Test succeeded")

        sys.path.pop(0)

    def test_double_pipe_unequal(self):

        base_folder = (Path(__file__).parent / "models" / "double_pipe_qth").absolute()
        sys.path.insert(0, str(base_folder / "src"))
        from double_pipe_qth import DoublePipeUnequalQTH

        with self.assertLogs(logger, level="WARNING") as cm:
            run_optimization_problem(DoublePipeUnequalQTH, base_folder=base_folder)

            self.assertIn("Pipe pipe_2_hot has artificial head loss", str(cm.output))
            self.assertIn("Pipe pipe_2_cold has artificial head loss", str(cm.output))

            self.assertNotIn("Pipe pipe_1_hot has artificial head loss", str(cm.output))
            self.assertNotIn("Pipe pipe_1_cold has artificial head loss", str(cm.output))

        sys.path.pop(0)

    def test_double_pipe_unequal_with_valve(self):

        base_folder = (Path(__file__).parent / "models" / "double_pipe_qth").absolute()
        sys.path.insert(0, str(base_folder / "src"))
        from double_pipe_qth import DoublePipeUnequalWithValveQTH

        with self.assertLogs(logger, level="WARNING") as cm:
            run_optimization_problem(DoublePipeUnequalWithValveQTH, base_folder=base_folder)

            self.assertNotIn("artificial head loss", str(cm.output))

            if not cm.output:
                # Prevent __exit__ from failing on AssertionError
                logger.warning("Test succeeded")

        sys.path.pop(0)
