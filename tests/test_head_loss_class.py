from pathlib import Path
from unittest import TestCase

import numpy as np

from rtctools.util import run_optimization_problem

from rtctools_heat_network.head_loss_class import HeadLossOption


class TestHeadLossCalculation(TestCase):
    def test_scalar_return_type(self):
        import models.basic_source_and_demand.src.heat_comparison as heat_comparison
        from models.basic_source_and_demand.src.heat_comparison import HeatPython

        class Model(HeatPython):
            def __init__(self, head_loss_option, *args, **kwargs):
                self.__head_loss_option = head_loss_option
                super().__init__(*args, **kwargs)

            def _hn_get_pipe_head_loss_option(self, *args, **kwargs):
                return self.__head_loss_option

            def optimize(self):
                # Just pre, we don't care about anything else
                self.pre()

        base_folder = Path(heat_comparison.__file__).resolve().parent.parent

        for h in [
            HeadLossOption.LINEAR,
            HeadLossOption.CQ2_INEQUALITY,
            HeadLossOption.LINEARIZED_DW,
        ]:
            m = run_optimization_problem(Model, head_loss_option=h, base_folder=base_folder)

            options = m.heat_network_options()
            parameters = m.parameters(0)

            ret = m._head_loss_class._hn_pipe_head_loss("pipe_hot", m, options, parameters, 0.1)
            self.assertIsInstance(ret, float)

            ret = m._head_loss_class._hn_pipe_head_loss(
                "pipe_hot", m, options, parameters, np.array([0.1])
            )
            self.assertIsInstance(ret, np.ndarray)
            self.assertEqual(len(ret), 1)

            ret = m._head_loss_class._hn_pipe_head_loss(
                "pipe_hot", m, options, parameters, np.array([0.05, 0.1, 0.2])
            )
            self.assertIsInstance(ret, np.ndarray)
            self.assertEqual(len(ret), 3)
