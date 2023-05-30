from pathlib import Path
from unittest import TestCase

import numpy as np

import pandas as pd

from rtctools.util import run_optimization_problem

from rtctools_heat_network.head_loss_mixin import HeadLossOption


class TestHydraulicPower(TestCase):
    def test_hydraulic_power(self):
        import models.pipe_test.src.run_hydraulic_power as run_hydraulic_power
        from models.pipe_test.src.run_hydraulic_power import (
            HeatProblem,
        )

        # Settings
        base_folder = Path(run_hydraulic_power.__file__).resolve().parent.parent
        run_hydraulic_power.comp_vars_vals = {
            "pipe_length": [25000.0],  # [m]
        }
        run_hydraulic_power.comp_vars_init = {
            "pipe_length": 0.0,  # [m]
            "heat_demand": [3.95 * 10**6, 3.95 * 10**6],  # [W]
            "pipe_DN_MILP": 300,  # [mm]
        }
        standard_columns_specified = [
            "Pipe1_supply_dPress",
            "Pipe1_return_dPress",
            "Pipe1_supply_Q",
            "Pipe1_return_Q",
            "Pipe1_supply_mass_flow",
            "Pipe1_return_mass_flow",
            "Pipe1_supply_flow_vel",
            "Pipe1_return_flow_vel",
            "Pipe1_supply_dT",
            "Pipe1_return_dT",
            "Heat_source",
            "Heat_demand",
            "Heat_loss",
            "pipe_length",
        ]

        # Initialize variables
        run_hydraulic_power.ThermalDemand = run_hydraulic_power.comp_vars_init["heat_demand"]
        run_hydraulic_power.manual_set_pipe_length = run_hydraulic_power.comp_vars_init[
            "pipe_length"
        ]
        run_hydraulic_power.manual_set_pipe_DN_diam_MILP = run_hydraulic_power.comp_vars_init[
            "pipe_DN_MILP"
        ]
        # ----------------------------------------------------------------------------------------
        # 3 MILP simulations with the only difference being the linear head loss setting:
        # - LINEARIZED_DW (1 line segment)
        # - LINEAR
        # - LINEARIZED_DW (default line segments = 5)
        # ----------------------------------------------------------------------------------------
        # Run MILP with LINEARIZED_DW head loss setting and 1 line segement
        run_hydraulic_power.df_MILP = pd.DataFrame(columns=standard_columns_specified)
        run_hydraulic_power.head_loss_setting = HeadLossOption.LINEARIZED_DW
        run_hydraulic_power.n_linearization_lines_setting = 1

        for val in range(0, len(run_hydraulic_power.comp_vars_vals["pipe_length"])):
            run_hydraulic_power.manual_set_pipe_length = run_hydraulic_power.comp_vars_vals[
                "pipe_length"
            ][val]
            run_optimization_problem(HeatProblem, base_folder=base_folder)

        hydraulic_power_post_process_dw_1 = run_hydraulic_power.df_MILP["Pipe1_supply_Q"][0] * abs(
            run_hydraulic_power.df_MILP["Pipe1_supply_dPress"][0]
        )
        hydraulic_power_dw_1 = run_hydraulic_power.df_MILP["Pipe1_supply_Hydraulic_power"][0]
        # Hydraulic power = delta pressure * Q = f(Q^3), where delta pressure = f(Q^2)
        # The linear approximation (hydraulic_power_dw_1) of the 3rd order function should
        # overestimate the hydraulic power when compared to the product of Q and the linear
        # approximation of 2nd order function (delta pressure).
        np.testing.assert_array_less(
            hydraulic_power_post_process_dw_1,
            hydraulic_power_dw_1,
            "Post process hydraulic power must be < hydraulic_power",
        )
        # Compare hydraulic power, for an one hour timeseries with a specific demand, to a hard
        # coded value which originates from runnning MILP without big_m method being implemented,
        # during the comparison of MILP and a high-fidelity code
        np.testing.assert_allclose(
            104829.66021214866,
            hydraulic_power_dw_1,
        )
        np.testing.assert_allclose(
            run_hydraulic_power.df_MILP["Pipe1_return_Hydraulic_power"][0],
            run_hydraulic_power.df_MILP["Pipe1_supply_Hydraulic_power"][0],
            rtol=1e-2,
        )
        # ----------------------------------------------------------------------------------------
        # Rerun MILP with LINEAR head loss setting
        run_hydraulic_power.df_MILP = pd.DataFrame(columns=standard_columns_specified)  # empty df
        run_hydraulic_power.head_loss_setting = HeadLossOption.LINEAR

        for val in range(0, len(run_hydraulic_power.comp_vars_vals["pipe_length"])):
            run_hydraulic_power.manual_set_pipe_length = run_hydraulic_power.comp_vars_vals[
                "pipe_length"
            ][val]
            run_optimization_problem(HeatProblem, base_folder=base_folder)

        hydraulic_power_post_process_linear = run_hydraulic_power.df_MILP["Pipe1_supply_Q"][
            0
        ] * abs(run_hydraulic_power.df_MILP["Pipe1_supply_dPress"][0])
        hydraulic_power_linear = run_hydraulic_power.df_MILP["Pipe1_supply_Hydraulic_power"][0]
        # Hydraulic power = delta pressure * Q = f(Q^3), where delta pressure = f(Q^2)
        # The linear approximation (1 line segment) of the 3rd order function should
        # overestimate the hydraulic power when compared to the product of Q and the linear
        # approximation of 2nd order function (delta pressure).
        np.testing.assert_array_less(
            hydraulic_power_post_process_linear,
            hydraulic_power_linear,
            "Post process hydraulic power must be < hydraulic_power",
        )
        # Hydraulic hydraulic =  delta pressure * Q = f(Q^3), where delta pressure = f(Q^2)
        # The predicted hydraulic power should be the same when the delta pressure is approximated
        # by a linear segment, via 2 different head loss setting options. Head loss setting =
        # HeadLossOption.LINEAR and HeadLossOption.LINEARIZED_DW (with 1 linear segment)
        np.testing.assert_allclose(
            hydraulic_power_post_process_linear,
            hydraulic_power_post_process_dw_1,
            rtol=1e-7,
            err_msg="Values should be the same",
        )
        # Hydraulic hydraulic =  delta pressure * Q = f(Q^3)
        # The predicted hydraulic power should be the same if it is approximated by a linear segment
        # , via 2 different head loss setting options. Head loss setting =
        # HeadLossOption.LINEAR and HeadLossOption.LINEARIZED_DW (with 1 linear segment)
        np.testing.assert_allclose(
            hydraulic_power_linear, hydraulic_power_dw_1, err_msg="Values should be the same"
        )
        # ----------------------------------------------------------------------------------------
        # Rerun MILP with DW head loss setting, and default line segments
        run_hydraulic_power.df_MILP = pd.DataFrame(columns=standard_columns_specified)  # empty df
        run_hydraulic_power.head_loss_setting = HeadLossOption.LINEARIZED_DW
        run_hydraulic_power.n_linearization_lines_setting = 5

        for val in range(0, len(run_hydraulic_power.comp_vars_vals["pipe_length"])):
            run_hydraulic_power.manual_set_pipe_length = run_hydraulic_power.comp_vars_vals[
                "pipe_length"
            ][val]
            run_optimization_problem(HeatProblem, base_folder=base_folder)

        hydraulic_power_post_process_dw = run_hydraulic_power.df_MILP["Pipe1_supply_Q"][0] * abs(
            run_hydraulic_power.df_MILP["Pipe1_supply_dPress"][0]
        )
        hydraulic_power_dw = run_hydraulic_power.df_MILP["Pipe1_supply_Hydraulic_power"][0]
        # Hydraulic power = delta pressure * Q = f(Q^3), where delta pressure = f(Q^2)
        # The linear approximation (default number of line segments) of the 3rd order function
        # should overestimate the hydraulic power when compared to the product of Q and the linear
        # approximation (default number of line segments) of 2nd order function (delta pressure).
        np.testing.assert_array_less(
            hydraulic_power_post_process_dw,
            hydraulic_power_dw,
            "Post process hydraulic power must be < hydraulic_power",
        )
        # Hydraulic power = delta pressure * Q = f(Q^3), where delta pressure = f(Q^2)
        # The approximation of the 3rd order function via 5 line segments (dafault value) should be
        # better compared to 1 line segment approximation thereof. The latter will result in an
        # overstimated prediction
        np.testing.assert_array_less(
            hydraulic_power_dw,
            hydraulic_power_dw_1,
            "5 line segments predicted hydraulic power > hydraulic_power with 1 line segment",
        )
        # Compare hydraulic power, for an one hour timeseries with a specific demand, to a hard
        # coded value which originates from runnning MILP without big_m method being implemented,
        # during the comparison of MILP and a high-fidelity code
        np.testing.assert_allclose(
            4367.240507173596,
            hydraulic_power_dw,
        )


if __name__ == "__main__":
    import time

    start_time = time.time()
    a = TestHydraulicPower()
    a.test_hydraulic_power()
    print("Execution time: " + time.strftime("%M:%S", time.gmtime(time.time() - start_time)))
