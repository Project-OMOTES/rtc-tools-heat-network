from pathlib import Path
from unittest import TestCase

import numpy as np

from rtctools.util import run_optimization_problem


class TestESDL(TestCase):
    def test_basic_source_and_demand_heat(self):
        import models.basic_source_and_demand.src.heat_comparison as heat_comparison
        from models.basic_source_and_demand.src.heat_comparison import HeatESDL, HeatPython

        base_folder = Path(heat_comparison.__file__).resolve().parent.parent

        case_python = run_optimization_problem(HeatPython, base_folder=base_folder)
        case_esdl = run_optimization_problem(HeatESDL, base_folder=base_folder)

        self.assertAlmostEqual(case_python.objective_value, case_esdl.objective_value, 6)

        np.testing.assert_allclose(
            case_python.extract_results()["demand.Heat_demand"],
            case_esdl.extract_results()["demand.Heat_demand"],
        )

    def test_basic_source_and_demand_qth(self):
        import models.basic_source_and_demand.src.qth_comparison as qth_comparison
        from models.basic_source_and_demand.src.qth_comparison import QTHPython, QTHESDL

        base_folder = Path(qth_comparison.__file__).resolve().parent.parent

        case_python = run_optimization_problem(QTHPython, base_folder=base_folder)
        case_esdl = run_optimization_problem(QTHESDL, base_folder=base_folder)

        np.testing.assert_allclose(
            case_python._objective_values, case_esdl._objective_values, rtol=1e-5, atol=1e-5
        )
