from pathlib import Path
from unittest import TestCase

import numpy as np

from rtctools.util import run_optimization_problem

from rtctools_heat_network.util import run_heat_network_optimization


class TestPyCML(TestCase):
    def test_basic_source_and_demand(self):
        import models.basic_source_and_demand.src.heat_comparison as heat_comparison
        from models.basic_source_and_demand.src.heat_comparison import HeatModelica, HeatPython

        base_folder = Path(heat_comparison.__file__).resolve().parent.parent

        case_modelica = run_optimization_problem(HeatModelica, base_folder=base_folder)
        case_python = run_optimization_problem(HeatPython, base_folder=base_folder)

        self.assertAlmostEqual(case_modelica.objective_value, case_python.objective_value, 6)

    def test_basic_buffer_example(self):
        from models.basic_buffer.src.compare import (
            HeatProblemModelica,
            QTHProblemModelica,
            HeatProblemPyCML,
            QTHProblemPyCML,
            base_folder,
        )

        modelica_heat, modelica_qth = run_heat_network_optimization(
            HeatProblemModelica, QTHProblemModelica, base_folder=base_folder
        )
        pycml_heat, pycml_qth = run_heat_network_optimization(
            HeatProblemPyCML, QTHProblemPyCML, base_folder=base_folder
        )

        np.testing.assert_allclose(modelica_heat._objective_values, pycml_heat._objective_values)
        np.testing.assert_allclose(
            modelica_qth._objective_values, pycml_qth._objective_values, rtol=1e-4
        )