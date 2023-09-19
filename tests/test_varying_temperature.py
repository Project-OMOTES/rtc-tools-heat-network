from pathlib import Path
from unittest import TestCase

import numpy as np

from rtctools.util import run_optimization_problem


class TestVaryingTemperature(TestCase):
    def test_1a_temperature_variation(self):
        import models.unit_cases.case_1a.src.run_1a as run_1a
        from models.unit_cases.case_1a.src.run_1a import HeatProblemTvar

        base_folder = Path(run_1a.__file__).resolve().parent.parent

        heat_problem = run_optimization_problem(HeatProblemTvar, base_folder=base_folder)

        # This optimization problem is to see whether the correct minimum delta temperaute is
        # chose by the optimization, minimum of 21 deg is needed, only the 75/60 option is
        # feasible, note that a maximum velocity is set low to achieve this.
        results = heat_problem.extract_results()

        # Check that the highest supply temperature is selected
        np.testing.assert_allclose(results[f"{3625334968694477359}__supply_temperature"], 75.0)

        # Check that the lowest return temperature is selected
        np.testing.assert_allclose(results[f"{3625334968694477359000}__return_temperature"], 60.0)

    def test_3a_temperature_variation(self):
        import models.unit_cases.case_3a.src.run_3a as run_3a
        from models.unit_cases.case_3a.src.run_3a import HeatProblemTvarsup, HeatProblemTvarret

        base_folder = Path(run_3a.__file__).resolve().parent.parent

        heat_problem = run_optimization_problem(HeatProblemTvarsup, base_folder=base_folder)

        # optimization with two choices in supply temp 80 and 120 deg
        # lowest temperature should be selected because of lower heat losses and
        # heat production minimization goal
        results = heat_problem.extract_results()

        # Check whehter the heat demand is matched
        for d in heat_problem.heat_network_components.get("demand", []):
            target = heat_problem.get_timeseries(f"{d}.target_heat_demand").values[
                : len(heat_problem.times())
            ]
            np.testing.assert_allclose(target, results[f"{d}.Heat_demand"])

        # Check that the lowest temperature (80.0) is the outputted temperature
        np.testing.assert_allclose(results[f"{4195016129475469474608}__supply_temperature"], 80.0)
        # Verify that also the integer is correctly set
        np.testing.assert_allclose(results[f"{4195016129475469474608}__supply_80.0"], 1.0)
        np.testing.assert_allclose(results[f"{4195016129475469474608}__supply_120.0"], 0.0)

        heat_problem = run_optimization_problem(HeatProblemTvarret, base_folder=base_folder)

        # optimization with two choices in return temp 30 and 40 deg
        # lowest temperature should be selected because of larger dT causing lowest flow rates
        # and we apply source Q minimization goal
        results = heat_problem.extract_results()

        # Check whehter the heat demand is matched
        for d in heat_problem.heat_network_components.get("demand", []):
            target = heat_problem.get_timeseries(f"{d}.target_heat_demand").values[
                : len(heat_problem.times())
            ]
            np.testing.assert_allclose(target, results[f"{d}.Heat_demand"])

        # Check that the lowest temperature (30.0) is the outputted temperature
        np.testing.assert_allclose(
            results[f"{4195016129475469474608000}__return_temperature"], 30.0
        )
        # Verify that also the integer is correctly set
        np.testing.assert_allclose(results[f"{4195016129475469474608000}__return_30.0"], 1.0)
        np.testing.assert_allclose(results[f"{4195016129475469474608000}__return_40.0"], 0.0)

    def test_hex_temperature_variation(self):
        import models.heat_exchange.src.run_heat_exchanger as run_heat_exchanger
        from models.heat_exchange.src.run_heat_exchanger import (
            HeatProblemTvar,
            HeatProblemTvarDisableHEX,
            HeatProblemTvarSecondary,
        )

        base_folder = Path(run_heat_exchanger.__file__).resolve().parent.parent

        heat_problem = run_optimization_problem(HeatProblemTvar, base_folder=base_folder)

        # optimization with three choices of primary supply temperature of which the lowest is
        # infeasible. Therefore optimization should select second lowest option of 80.
        # lowest feasible temperature should be selected due to heatlosses and the minimization
        # goal of the sources
        results = heat_problem.extract_results()

        # Check that the lowest feasible temperature (80.0) is the outputted temperature and not
        # the 69 which is below the secondary side supply temperature
        np.testing.assert_allclose(results[f"{33638164429859421}__supply_temperature"], 80.0)
        # Verify that also the integer is correctly set
        np.testing.assert_allclose(results[f"{33638164429859421}__supply_69.0"], 0.0)
        np.testing.assert_allclose(results[f"{33638164429859421}__supply_80.0"], 1.0)
        np.testing.assert_allclose(results[f"{33638164429859421}__supply_90.0"], 0.0)

        heat_problem = run_optimization_problem(HeatProblemTvarDisableHEX, base_folder=base_folder)

        # optimization with only one option in temperature which is infeasible for the hex.
        # therefore optimization should disable the heat exchanger
        results = heat_problem.extract_results()

        # Check that the problem has an infeasible temperature for the hex
        np.testing.assert_allclose(results[f"{33638164429859421}__supply_temperature"], 69.0)
        # Verify that the hex is disabled
        np.testing.assert_allclose(results["HeatExchange_39ed__disabled"], 1.0)
        np.testing.assert_allclose(results["HeatExchange_39ed.Primary_heat"], 0.0)

        heat_problem = run_optimization_problem(HeatProblemTvarSecondary, base_folder=base_folder)

        # optimization with two choices in secondary supply temp 70 and 90 deg
        # lowest temperature should be selected because of larger dT causing lowest flow rates
        # and we apply source dH minimization goal
        results = heat_problem.extract_results()

        # Check that the lowest temperature (70.0) is the outputted temperature
        np.testing.assert_allclose(results[f"{7212673879469902607010}__supply_temperature"], 70.0)
        # Verify that also the integer is correctly set
        np.testing.assert_allclose(results[f"{7212673879469902607010}__supply_70.0"], 1.0)
        np.testing.assert_allclose(results[f"{7212673879469902607010}__supply_90.0"], 0.0)

    # Note that CBC struggles heavily and tends to crash, therefore excluded from pipeline
    # def test_varying_temperature_with_pipe_sizing(self):
    #     root_folder = str(Path(__file__).resolve().parent.parent)
    #     import sys
    #     sys.path.insert(1, root_folder)
    #
    #     import examples.pipe_diameter_sizing.src.example  # noqa: E402, I100
    #     from examples.pipe_diameter_sizing.src.example import (
    #         PipeDiameterSizingProblemTvar,
    #     )  # noqa: E402, I100
    #
    #     base_folder = (
    #         Path(examples.pipe_diameter_sizing.src.example.__file__).resolve().parent.parent
    #     )
    #
    #     del root_folder
    #     sys.path.pop(1)
    #
    #     # The optimization should select the highest delta-temperature as
    #     # this minimizes the length
    #     # times diameter cost function
    #
    #     # problem = run_optimization_problem(
    #     PipeDiameterSizingProblemTvar, base_folder=base_folder)
    #     # results = problem.extract_results()
    #
    #
    #     # # Check that the highest supply temperature is selected
    #     # np.testing.assert_allclose(
    #     results[f"{761602374459208051248}__supply_temperature"], 90.0)
    #     #
    #     # # Check that the lowest return temperature is selected
    #     # np.testing.assert_allclose(
    #     results[f"{761602374459208051248000}__return_temperature"], 30.0)
