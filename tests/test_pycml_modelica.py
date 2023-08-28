from pathlib import Path
from unittest import TestCase

import numpy as np

from rtctools.util import run_optimization_problem


class TestPyCMLvsModelica(TestCase):
    def test_basic_source_and_demand_heat(self):
        import models.basic_source_and_demand.src.heat_comparison as heat_comparison
        from models.basic_source_and_demand.src.heat_comparison import HeatModelica, HeatPython

        base_folder = Path(heat_comparison.__file__).resolve().parent.parent

        case_modelica = run_optimization_problem(HeatModelica, base_folder=base_folder)
        case_python = run_optimization_problem(HeatPython, base_folder=base_folder)

        self.assertAlmostEqual(case_modelica.objective_value, case_python.objective_value, 6)

    def test_basic_source_and_demand_qth(self):
        import models.basic_source_and_demand.src.qth_comparison as qth_comparison
        from models.basic_source_and_demand.src.qth_comparison import QTHModelica, QTHPython

        base_folder = Path(qth_comparison.__file__).resolve().parent.parent

        case_modelica = run_optimization_problem(QTHModelica, base_folder=base_folder)
        case_python = run_optimization_problem(QTHPython, base_folder=base_folder)

        np.testing.assert_allclose(case_modelica._objective_values, case_python._objective_values)

    # def test_basic_buffer_example(self):
    #     from models.basic_buffer.src.compare import (
    #         HeatProblemPyCML,
    #         QTHProblemPyCML,
    #         base_folder,
    #     )
    #
    #     # TODO: this test should be moved as the comparsion with modellica is no longer relevant.
    #     #
    #     # This test should only check whether the QTH problem is seeded with a feasible set of
    #     # flow directions. Note that the current example is flawed as there seem to be multiple
    #     # equivalent MILP solutions to the problem. Which do not all give feasible directions for
    #     # QTH.
    #     pycml_heat, pycml_qth = run_heat_network_optimization(
    #         HeatProblemPyCML, QTHProblemPyCML, base_folder=base_folder
    #     )
    #     self.assertTrue(np.asarray(pycml_heat._objective_values) >= 0.0)
    #     self.assertTrue(all(np.asarray(pycml_qth._objective_values) >= 0.0))
