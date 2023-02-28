from pathlib import Path
from unittest import TestCase

import numpy as np

from rtctools.util import run_optimization_problem

from rtctools_heat_network.head_loss_mixin import HeadLossOption
from rtctools_heat_network.util import run_heat_network_optimization


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

            ret = m._hn_pipe_head_loss("pipe_hot", options, parameters, 0.1)
            self.assertIsInstance(ret, float)

            ret = m._hn_pipe_head_loss("pipe_hot", options, parameters, np.array([0.1]))
            self.assertIsInstance(ret, np.ndarray)
            self.assertEqual(len(ret), 1)

            ret = m._hn_pipe_head_loss("pipe_hot", options, parameters, np.array([0.05, 0.1, 0.2]))
            self.assertIsInstance(ret, np.ndarray)
            self.assertEqual(len(ret), 3)


class TestHeadLossOptions(TestCase):
    def test_no_head_loss_mixing_options(self):
        import models.basic_source_and_demand.src.heat_comparison as heat_comparison
        from models.basic_source_and_demand.src.heat_comparison import HeatPython

        base_folder = Path(heat_comparison.__file__).resolve().parent.parent

        class Model(HeatPython):
            def heat_network_options(self):
                options = super().heat_network_options()
                options["head_loss_option"] = HeadLossOption.LINEAR
                return options

            def _hn_get_pipe_head_loss_option(self, *args, **kwargs):
                return HeadLossOption.NO_HEADLOSS

        with self.assertRaisesRegex(
            Exception, "Mixing .NO_HEADLOSS with other head loss options is not allowed"
        ):
            run_optimization_problem(Model, base_folder=base_folder)

    def test_no_head_loss(self):
        # Test if a model with NO_HEADLOSS set runs without issues
        from models.basic_buffer.src.compare import (
            HeatProblemPyCML,
            QTHProblemPyCML,
            base_folder,
        )

        class ModelHeat(HeatProblemPyCML):
            def heat_network_options(self):
                options = super().heat_network_options()
                options["head_loss_option"] = HeadLossOption.NO_HEADLOSS
                return options

        class ModelQTH(QTHProblemPyCML):
            def heat_network_options(self):
                options = super().heat_network_options()
                options["head_loss_option"] = HeadLossOption.NO_HEADLOSS
                return options

        run_heat_network_optimization(ModelHeat, ModelQTH, base_folder=base_folder)
