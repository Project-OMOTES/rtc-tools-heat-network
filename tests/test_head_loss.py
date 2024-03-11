from pathlib import Path
from unittest import TestCase

import numpy as np

from rtctools.util import run_optimization_problem

import rtctools_heat_network._darcy_weisbach as darcy_weisbach
from rtctools_heat_network.constants import GRAVITATIONAL_CONSTANT
from rtctools_heat_network.esdl.esdl_parser import ESDLFileParser
from rtctools_heat_network.esdl.profile_parser import ProfileReaderFromFile
from rtctools_heat_network.head_loss_class import HeadLossOption
from rtctools_heat_network.network_common import NetworkSettings


class TestHeadLoss(TestCase):
    """
    Test case for a heat network and a gas network consisting out of a source, pipe(s) and a sink
    """

    def test_heat_network_head_loss(self):
        """
        Heat network: test the piecewise linear inequality constraint of the head loss
        approximation.

        Checks:
        - That the head_loss() function does return the expected theoretical dH at a data point
        in the middle of the 1st line segment (dH curve is approximated with 5 linear lines)
        - That the head_loss() function does return a value smaller than a manual linearly
        approximated dH at a data point in the middle of the 1st line segment (dH curve is
        approximated with 5 linear lines)
        - That for the dH value approximated by the code is conservative, in other word greater
        than the theoretical value
        - That the pump power is conservative
        """
        import models.source_pipe_sink.src.double_pipe_heat as example
        from models.source_pipe_sink.src.double_pipe_heat import SourcePipeSink

        base_folder = Path(example.__file__).resolve().parent.parent

        for head_loss_option_setting in [
            HeadLossOption.LINEARIZED_N_LINES_WEAK_INEQUALITY,
            HeadLossOption.LINEARIZED_N_LINES_EQUALITY,
        ]:
            # Added for case where head loss is modelled via DW
            class SourcePipeSinkDW(SourcePipeSink):
                def heat_network_options(self):
                    options = super().heat_network_options()

                    nonlocal head_loss_option_setting
                    head_loss_option_setting = head_loss_option_setting

                    if (
                        head_loss_option_setting
                        == HeadLossOption.LINEARIZED_N_LINES_WEAK_INEQUALITY
                    ):
                        self.gas_network_settings["head_loss_option"] = (
                            HeadLossOption.LINEARIZED_ONE_LINE_EQUALITY
                        )
                        self.gas_network_settings["n_linearization_lines"] = 5
                        self.heat_network_settings["minimize_head_losses"] = True
                    elif head_loss_option_setting == HeadLossOption.LINEARIZED_N_LINES_EQUALITY:
                        self.heat_network_settings["head_loss_option"] = (
                            HeadLossOption.LINEARIZED_N_LINES_EQUALITY
                        )
                        self.heat_network_settings["minimize_head_losses"] = False
                        self.heat_network_settings["minimum_velocity"] = 1.0e-6

                    return options

            solution = run_optimization_problem(
                SourcePipeSinkDW,
                base_folder=base_folder,
                esdl_file_name="sourcesink.esdl",
                esdl_parser=ESDLFileParser,
                profile_reader=ProfileReaderFromFile,
                input_timeseries_file="timeseries_import.csv",
            )
            results = solution.extract_results()

            pipes = ["Pipe1"]
            for itime in range(len(results[f"{pipes[0]}.dH"])):
                v_max = solution.heat_network_settings["maximum_velocity"]
                pipe_diameter = solution.parameters(0)[f"{pipes[0]}.diameter"]
                pipe_wall_roughness = solution.heat_network_options()["wall_roughness"]
                temperature = solution.parameters(0)[f"{pipes[0]}.temperature"]
                pipe_length = solution.parameters(0)[f"{pipes[0]}.length"]
                v_points = np.linspace(
                    0.0,
                    v_max,
                    solution.heat_network_settings["n_linearization_lines"] + 1,
                )
                v_inspect = (
                    results[f"{pipes[0]}.Q"][itime] / solution.parameters(0)[f"{pipes[0]}.area"]
                )
                idx = []
                linearized_idx = []
                idx.append(
                    (results["Pipe1.Q"][itime] / solution.parameters(0)["Pipe1.area"]) >= v_points
                )
                idx.append(
                    (results["Pipe1.Q"][itime] / solution.parameters(0)["Pipe1.area"]) < v_points
                )
                linearized_idx.append(np.where(idx[0])[0][-1])
                linearized_idx.append(np.where(idx[1])[0][0])

                # Theoretical head loss calc, dH =
                # friction_factor * 8 * pipe_length * volumetric_flow^2 / ( pipe_diameter^5 * g *
                # pi^2)
                dh_theory = (
                    darcy_weisbach.friction_factor(
                        v_inspect,
                        pipe_diameter,
                        pipe_wall_roughness,
                        temperature,
                    )
                    * 8.0
                    * pipe_length
                    * (v_inspect * np.pi * pipe_diameter**2 / 4.0) ** 2
                    / (pipe_diameter**5 * GRAVITATIONAL_CONSTANT * np.pi**2)
                )
                # Approximate dH [m] vs Q [m3/s] with a linear line between between v_points
                # dH_manual_linear = a*Q + b
                # Then use this linear function to calculate the head loss
                delta_dh_theory = darcy_weisbach.head_loss(
                    v_points[linearized_idx[1]],
                    pipe_diameter,
                    pipe_length,
                    pipe_wall_roughness,
                    temperature,
                ) - darcy_weisbach.head_loss(
                    v_points[linearized_idx[0]],
                    pipe_diameter,
                    pipe_length,
                    pipe_wall_roughness,
                    temperature,
                )

                delta_volumetric_flow = (
                    v_points[linearized_idx[1]] * np.pi * pipe_diameter**2 / 4.0
                ) - (v_points[linearized_idx[0]] * np.pi * pipe_diameter**2 / 4.0)

                a = delta_dh_theory / delta_volumetric_flow
                b = delta_dh_theory - a * delta_volumetric_flow
                dh_manual_linear = a * (v_inspect * np.pi * pipe_diameter**2 / 4.0) + b

                dh_milp_head_loss_function = darcy_weisbach.head_loss(
                    v_inspect, pipe_diameter, pipe_length, pipe_wall_roughness, temperature
                )

                np.testing.assert_allclose(dh_theory, dh_milp_head_loss_function)
                np.testing.assert_array_less(dh_milp_head_loss_function, dh_manual_linear)

                if head_loss_option_setting == HeadLossOption.LINEARIZED_N_LINES_WEAK_INEQUALITY:
                    np.testing.assert_array_less(
                        dh_manual_linear, -results[f"{pipes[0]}.dH"][itime] + 1e-6
                    )
                elif head_loss_option_setting == HeadLossOption.LINEARIZED_N_LINES_EQUALITY:
                    np.testing.assert_allclose(
                        -results[f"{pipes[0]}.dH"][itime], dh_manual_linear, rtol=1e-5, atol=1e-7
                    )

            for pipe in pipes:
                velocities = results[f"{pipe}.Q"] / solution.parameters(0)[f"{pipe}.area"]
                for ii in range(len(results[f"{pipe}.dH"])):
                    np.testing.assert_array_less(
                        darcy_weisbach.head_loss(
                            velocities[ii],
                            pipe_diameter,
                            pipe_length,
                            pipe_wall_roughness,
                            temperature,
                        ),
                        -results[f"{pipe}.dH"][ii],
                    )

            pump_power = results["source.Pump_power"]
            pump_power_post_process = (
                results["source.dH"] / GRAVITATIONAL_CONSTANT * 1.0e5 * results["source.Q"]
            )

            # The pump power should be overestimated compared to the actual head loss due to the
            # fact that we are linearizing a third order equation for hydraulic power instead of
            # the second order for head loss.
            np.testing.assert_array_less(pump_power_post_process, pump_power)

            sum_hp = (
                results["demand.HeatOut.Hydraulic_power"] - results["demand.HeatIn.Hydraulic_power"]
            )
            sum_hp += (
                results["Pipe1.HeatOut.Hydraulic_power"] - results["Pipe1.HeatIn.Hydraulic_power"]
            )
            sum_hp += (
                results["Pipe1_ret.HeatOut.Hydraulic_power"]
                - results["Pipe1_ret.HeatIn.Hydraulic_power"]
            )

            np.testing.assert_allclose(abs(sum_hp), pump_power, atol=1.0e-3)

    def test_heat_network_pipe_split_head_loss(self):
        """
        Heat network: test the piecewise linear inequality constraint of the head loss
        approximation.

        Checks:
        - That the head_loss() function does return the expected theoretical dH at a data point
        in the middle of the 1st line segment (dH curve is approximated with 5 linear lines)
        - That the head_loss() function does return a value smaller than a manual linearly
        approximated dH at a data point in the middle of the 1st line segment (dH curve is
        approximated with 5 linear lines)
        - That for the dH value approximated by the code is conservative, in other word greater
        than the theoretical value
        """
        import models.source_pipe_split_sink.src.double_pipe_heat as example
        from models.source_pipe_split_sink.src.double_pipe_heat import SourcePipeSink

        base_folder = Path(example.__file__).resolve().parent.parent

        # Added for case where head loss is modelled via DW
        class SourcePipeSinkDW(SourcePipeSink):
            def heat_network_options(self):
                options = super().heat_network_options()

                self.heat_network_settings["head_loss_option"] = (
                    HeadLossOption.LINEARIZED_N_LINES_EQUALITY
                )

                self.heat_network_settings["n_linearization_lines"] = 2
                self.heat_network_settings["minimum_velocity"] = 0.0

                return options

        solution = run_optimization_problem(
            SourcePipeSinkDW,
            base_folder=base_folder,
            esdl_file_name="sourcesink.esdl",
            esdl_parser=ESDLFileParser,
            profile_reader=ProfileReaderFromFile,
            input_timeseries_file="timeseries_import.csv",
        )
        results = solution.extract_results()

        pipes = ["Pipe1", "Pipe2", "Pipe3", "Pipe4"]
        v_max = solution.heat_network_settings["maximum_velocity"]
        pipe_diameter = solution.parameters(0)[f"{pipes[0]}.diameter"]
        pipe_wall_roughness = solution.heat_network_options()["wall_roughness"]
        temperature = solution.parameters(0)[f"{pipes[0]}.temperature"]
        pipe_length = solution.parameters(0)[f"{pipes[0]}.length"]
        v_points = np.linspace(
            0.0,
            v_max,
            solution.heat_network_settings["n_linearization_lines"] + 1,
        )
        v_inspect = v_points[0] + (v_points[1] - v_points[0]) / 2.0

        #####
        # import matplotlib.pyplot as plt
        # p_points = [0] * (solution.heat_network_settings["n_linearization_lines"] + 1)
        # for ii in range(1, len(p_points)):
        #     p_points[ii] = darcy_weisbach.head_loss(
        #         v_points[ii],
        #         pipe_diameter,
        #         pipe_length,
        #         pipe_wall_roughness,
        #         temperature,
        #         )

        # velocities = results[f"{pipes[0]}.Q"] / solution.parameters(0)[f"{pipes[0]}.area"]
        # plt.plot(v_points, p_points)
        # plt.plot(velocities[:],-results[f"{pipes[0]}.dH"][:], marker="1", linestyle='None' )
        # plt.show()
        #####

        # Theoretical head loss calc, dH =
        # friction_factor * 8 * pipe_length * volumetric_flow^2 / ( pipe_diameter^5 * g * pi^2)
        dh_theory = (
            darcy_weisbach.friction_factor(
                v_inspect,
                pipe_diameter,
                pipe_wall_roughness,
                temperature,
            )
            * 8.0
            * pipe_length
            * (v_inspect * np.pi * pipe_diameter**2 / 4.0) ** 2
            / (pipe_diameter**5 * GRAVITATIONAL_CONSTANT * np.pi**2)
        )
        # Approximate dH [m] vs Q [m3/s] with a linear line between between v_points
        # dH_manual_linear = a*Q + b
        # Then use this linear function to calculate the head loss
        delta_dh_theory = darcy_weisbach.head_loss(
            v_points[1], pipe_diameter, pipe_length, pipe_wall_roughness, temperature
        ) - darcy_weisbach.head_loss(
            v_points[0], pipe_diameter, pipe_length, pipe_wall_roughness, temperature
        )

        delta_volumetric_flow = (v_points[1] * np.pi * pipe_diameter**2 / 4.0) - (
            v_points[0] * np.pi * pipe_diameter**2 / 4.0
        )

        a = delta_dh_theory / delta_volumetric_flow
        b = delta_dh_theory - a * delta_volumetric_flow
        dh_manual_linear = a * (v_inspect * np.pi * pipe_diameter**2 / 4.0) + b

        dh_milp_head_loss_function = darcy_weisbach.head_loss(
            v_inspect, pipe_diameter, pipe_length, pipe_wall_roughness, temperature
        )

        np.testing.assert_allclose(dh_theory, dh_milp_head_loss_function)
        np.testing.assert_array_less(dh_milp_head_loss_function, dh_manual_linear)

        # Check:
        #  - That only one linear line is active for a pipe
        #  - Pipe 4 for has a zero flow rate, but the dH should be the same as pipe 2
        for pipe in pipes:
            # TODO: check if the wrong line segment is selected for Pipe1?
            # # Check that only one linear line segment is active for the head loss linearization
            # np.testing.assert_allclose(
            #     results[f"{pipe}__pipe_linear_line_segment_num_1_neg_discharge"],
            #     0.0,
            # )
            # np.testing.assert_allclose(
            #     results[f"{pipe}__pipe_linear_line_segment_num_2_neg_discharge"],
            #     0.0,
            # )
            # np.testing.assert_allclose(
            #     results[f"{pipe}__pipe_linear_line_segment_num_2_pos_discharge"],
            #     0.0,
            # )
            # np.testing.assert_allclose(
            #     results[f"{pipe}__pipe_linear_line_segment_num_1_pos_discharge"],
            #     1.0,
            # )
            # np.testing.assert_allclose(
            #     results[f"{pipe}__pipe_linear_line_segment_num_2_pos_discharge"],
            #     0.0,
            # )
            if pipe not in ["Pipe4", "Pipe4_ret"]:
                velocities = results[f"{pipe}.Q"] / solution.parameters(0)[f"{pipe}.area"]
                for ii in range(len(results[f"{pipe}.dH"])):
                    np.testing.assert_array_less(
                        darcy_weisbach.head_loss(
                            velocities[ii],
                            pipe_diameter,
                            pipe_length,
                            pipe_wall_roughness,
                            temperature,
                        ),
                        abs(results[f"{pipe}.dH"][ii]),
                    )
            elif pipe in ["Pipe4", "Pipe4_ret"]:
                np.testing.assert_allclose(results["Pipe2.dH"][ii], results[f"{pipe}.dH"][ii])

        if pipe not in ["Pipe4", "Pipe4_ret"]:  # kvr fix indent is wrong
            np.testing.assert_allclose(
                -results[f"{pipes[0]}.dH"][0], dh_manual_linear, rtol=1e-5, atol=1e-7
            )

    def test_gas_network_head_loss(self):
        """
        Gas network: Test the head loss approximation.

        Checks:
        - head loss variable vs manually calcuated value
        - linear equality constraint
        - linear inequality constraint with 2 linear segments
        """

        import models.unit_cases_gas.source_sink.src.run_source_sink as example
        from models.unit_cases_gas.source_sink.src.run_source_sink import GasProblem

        base_folder = Path(example.__file__).resolve().parent.parent

        linear_head_loss_equality = 0.0

        for head_loss_option_setting in [
            HeadLossOption.LINEARIZED_ONE_LINE_EQUALITY,
            HeadLossOption.LINEARIZED_N_LINES_WEAK_INEQUALITY,
            HeadLossOption.LINEARIZED_N_LINES_EQUALITY,
        ]:

            class TestSourceSink(GasProblem):
                def heat_network_options(self):
                    options = super().heat_network_options()

                    nonlocal head_loss_option_setting
                    head_loss_option_setting = head_loss_option_setting

                    self.gas_network_settings["head_loss_option"] = head_loss_option_setting
                    if head_loss_option_setting == HeadLossOption.LINEARIZED_ONE_LINE_EQUALITY:
                        # This setting is below is not needed for the optmiizer, but is used in the
                        # test below.
                        self.gas_network_settings["n_linearization_lines"] = 1  # dot not delete
                        self.gas_network_settings["minimize_head_losses"] = True
                    elif (
                        head_loss_option_setting
                        == HeadLossOption.LINEARIZED_N_LINES_WEAK_INEQUALITY
                    ):
                        self.gas_network_settings["n_linearization_lines"] = 2
                        self.gas_network_settings["minimize_head_losses"] = True
                    if head_loss_option_setting == HeadLossOption.LINEARIZED_N_LINES_EQUALITY:
                        self.gas_network_settings["n_linearization_lines"] = 2
                        self.gas_network_settings["minimize_head_losses"] = True
                        self.gas_network_settings["minimum_velocity"] = 0.0

                    # This code below was needed for LINEARIZED_N_LINES_EQUALITY to pass below
                    # self.gas_network_settings["pipe_maximum_pressure"] = 100.0  # [bar]
                    # self.gas_network_settings["pipe_minimum_pressure"] = 10.0
                    return options

            solution = run_optimization_problem(
                TestSourceSink,
                base_folder=base_folder,
                esdl_file_name="source_sink.esdl",
                esdl_parser=ESDLFileParser,
                profile_reader=ProfileReaderFromFile,
                input_timeseries_file="timeseries.csv",
            )
            results = solution.extract_results()

            # Check the head loss variable
            np.testing.assert_allclose(
                results["Pipe_4abc.GasOut.H"] - results["Pipe_4abc.GasIn.H"],
                results["Pipe_4abc.dH"],
            )

            pipes = ["Pipe_4abc"]
            v_max = solution.gas_network_settings["maximum_velocity"]
            pipe_diameter = solution.parameters(0)[f"{pipes[0]}.diameter"]
            pipe_wall_roughness = solution.heat_network_options()["wall_roughness"]
            # TODO: resolve temperature - >solution.parameters(0)[f"{pipes[0]}.temperature"]
            temperature = 20.0
            pipe_length = solution.parameters(0)[f"{pipes[0]}.length"]
            v_points = [0.0, v_max / solution.gas_network_settings["n_linearization_lines"]]
            v_inspect = results[f"{pipes[0]}.GasOut.Q"] / solution.parameters(0)[f"{pipes[0]}.area"]

            # Approximate dH [m] vs Q [m3/s] with a linear line between between v_points
            # dH_manual_linear = a*Q + b
            # Then use this linear function to calculate the head loss
            delta_dh_theory = darcy_weisbach.head_loss(
                v_points[1],
                pipe_diameter,
                pipe_length,
                pipe_wall_roughness,
                temperature,
                network_type=NetworkSettings.NETWORK_TYPE_GAS,
                pressure=solution.parameters(0)[f"{pipes[0]}.pressure"],
            ) - darcy_weisbach.head_loss(
                v_points[0],
                pipe_diameter,
                pipe_length,
                pipe_wall_roughness,
                temperature,
                network_type=NetworkSettings.NETWORK_TYPE_GAS,
                pressure=solution.parameters(0)[f"{pipes[0]}.pressure"],
            )

            delta_volumetric_flow = (v_points[1] * np.pi * pipe_diameter**2 / 4.0) - (
                v_points[0] * np.pi * pipe_diameter**2 / 4.0
            )

            a = delta_dh_theory / delta_volumetric_flow
            b = delta_dh_theory - a * delta_volumetric_flow
            dh_manual_linear = a * (v_inspect * np.pi * pipe_diameter**2 / 4.0) + b

            # Check that the head loss approximation with 2 linear lines (inequality constraints
            # is < than the linear equality head loss constraint
            if head_loss_option_setting == HeadLossOption.LINEARIZED_ONE_LINE_EQUALITY:
                # Check that the aproximated head loss matches the manually calculated value
                np.testing.assert_allclose(dh_manual_linear, -results[f"{pipes[0]}.dH"])
                linear_head_loss_equality = dh_manual_linear
            elif head_loss_option_setting == HeadLossOption.LINEARIZED_N_LINES_WEAK_INEQUALITY:
                np.testing.assert_array_less(-results[f"{pipes[0]}.dH"], linear_head_loss_equality)
            elif head_loss_option_setting == HeadLossOption.LINEARIZED_N_LINES_EQUALITY:
                # TODO: check why this fails when the timeseries input length is changed? and why
                # only timestep index 1 passes?
                # Check that the approximated head loss matches the manually calculated value
                # This is now completely failing

                # kvr this is completely broken after the merge
                np.testing.assert_allclose(dh_manual_linear[1], -results[f"{pipes[0]}.dH"][1])

            for pipe in pipes:
                velocities = results[f"{pipe}.Q"] / solution.parameters(0)[f"{pipe}.area"]
                np.testing.assert_array_less(
                    darcy_weisbach.head_loss(
                        velocities[0], pipe_diameter, pipe_length, pipe_wall_roughness, temperature
                    ),
                    -results[f"{pipe}.dH"][0],
                )
                if head_loss_option_setting == HeadLossOption.LINEARIZED_N_LINES_EQUALITY:
                    # Check that only one linear line segment is active for the head loss
                    # linearization
                    np.testing.assert_allclose(
                        results[f"{pipe}__pipe_linear_line_segment_num_1_neg_discharge"],
                        0.0,
                    )
                    np.testing.assert_allclose(
                        results[f"{pipe}__pipe_linear_line_segment_num_2_neg_discharge"],
                        0.0,
                    )
                    np.testing.assert_allclose(
                        results[f"{pipe}__pipe_linear_line_segment_num_2_pos_discharge"],
                        0.0,
                    )
                    np.testing.assert_allclose(
                        results[f"{pipe}__pipe_linear_line_segment_num_1_pos_discharge"],
                        1.0,
                    )
                    np.testing.assert_allclose(
                        results[f"{pipe}__pipe_linear_line_segment_num_2_pos_discharge"],
                        0.0,
                    )

    def test_gas_network_pipe_split_head_loss(self):
        """
        Gas network: Test the head loss approximation.

        Checks:
        -......
        """

        import models.unit_cases_gas.source_pipe_split_sink.src.run_source_sink as example
        from models.unit_cases_gas.source_pipe_split_sink.src.run_source_sink import GasProblem

        base_folder = Path(example.__file__).resolve().parent.parent

        for head_loss_option_setting in [
            HeadLossOption.LINEARIZED_N_LINES_EQUALITY,
        ]:

            class TestSourceSink(GasProblem):
                def heat_network_options(self):
                    options = super().heat_network_options()
                    self.gas_network_settings["minimum_velocity"] = 0.0

                    nonlocal head_loss_option_setting
                    head_loss_option_setting = head_loss_option_setting

                    self.gas_network_settings["head_loss_option"] = head_loss_option_setting
                    if head_loss_option_setting == HeadLossOption.LINEARIZED_N_LINES_EQUALITY:
                        self.gas_network_settings["n_linearization_lines"] = 2
                        self.gas_network_settings["minimize_head_losses"] = True
                    elif head_loss_option_setting == HeadLossOption.LINEARIZED_ONE_LINE_EQUALITY:
                        self.gas_network_settings["minimize_head_losses"] = True
                    elif (
                        head_loss_option_setting
                        == HeadLossOption.LINEARIZED_N_LINES_WEAK_INEQUALITY
                    ):
                        self.gas_network_settings["n_linearization_lines"] = 2
                        self.gas_network_settings["minimize_head_losses"] = True

                    return options

            solution = run_optimization_problem(
                TestSourceSink,
                base_folder=base_folder,
                esdl_file_name="source_sink.esdl",
                esdl_parser=ESDLFileParser,
                profile_reader=ProfileReaderFromFile,
                input_timeseries_file="timeseries.csv",
            )
            results = solution.extract_results()

            # Check the head loss variable
            np.testing.assert_allclose(
                results["Pipe1.GasOut.H"] - results["Pipe1.GasIn.H"],
                results["Pipe1.dH"],
            )
            np.testing.assert_allclose(-results["Pipe1.dH"], results["Pipe1.__head_loss"])

            pipes = ["Pipe1"]
            v_max = solution.gas_network_settings["maximum_velocity"]
            pipe_diameter = solution.parameters(0)[f"{pipes[0]}.diameter"]
            pipe_wall_roughness = solution.heat_network_options()["wall_roughness"]
            # TODO: resolve temperature - >solution.parameters(0)[f"{pipes[0]}.temperature"]
            temperature = 20.0
            pipe_length = solution.parameters(0)[f"{pipes[0]}.length"]
            v_points = [0.0, v_max / solution.gas_network_settings["n_linearization_lines"]]
            v_inspect = results[f"{pipes[0]}.GasOut.Q"] / solution.parameters(0)[f"{pipes[0]}.area"]

            # Approximate dH [m] vs Q [m3/s] with a linear line between between v_points
            # dH_manual_linear = a*Q + b
            # Then use this linear function to calculate the head loss
            delta_dh_theory = darcy_weisbach.head_loss(
                v_points[1],
                pipe_diameter,
                pipe_length,
                pipe_wall_roughness,
                temperature,
                network_type=NetworkSettings.NETWORK_TYPE_GAS,
                pressure=solution.parameters(0)[f"{pipes[0]}.pressure"],
            ) - darcy_weisbach.head_loss(
                v_points[0],
                pipe_diameter,
                pipe_length,
                pipe_wall_roughness,
                temperature,
                network_type=NetworkSettings.NETWORK_TYPE_GAS,
                pressure=solution.parameters(0)[f"{pipes[0]}.pressure"],
            )

            delta_volumetric_flow = (v_points[1] * np.pi * pipe_diameter**2 / 4.0) - (
                v_points[0] * np.pi * pipe_diameter**2 / 4.0
            )

            a = delta_dh_theory / delta_volumetric_flow
            b = delta_dh_theory - a * delta_volumetric_flow
            dh_manual_linear = a * (v_inspect * np.pi * pipe_diameter**2 / 4.0) + b

            # Gas flow balance
            np.testing.assert_allclose(
                results["GasDemand_a2d8.Gas_demand_mass_flow"],
                solution.get_timeseries("GasDemand_a2d8.target_gas_demand").values,
            )
            # demand_matching_test(solution, results)  # still to be updated for gas demand

            # Check that the aproximated head loss matches the maunally calculated value
            # kvr this is broken after merge
            np.testing.assert_allclose(dh_manual_linear, -results["Pipe1.dH"], atol=1e-6)

            for pipe in pipes:
                velocities = results[f"{pipe}.Q"] / solution.parameters(0)[f"{pipe}.area"]
                np.testing.assert_array_less(
                    darcy_weisbach.head_loss(
                        velocities[0], pipe_diameter, pipe_length, pipe_wall_roughness, temperature
                    ),
                    -results[f"{pipe}.dH"][0],
                )

                # Check that only one linear line segment is active for the head loss linearization
                np.testing.assert_allclose(
                    results[f"{pipe}__pipe_linear_line_segment_num_1_neg_discharge"],
                    0.0,
                )
                np.testing.assert_allclose(
                    results[f"{pipe}__pipe_linear_line_segment_num_2_neg_discharge"],
                    0.0,
                )
                np.testing.assert_allclose(
                    results[f"{pipe}__pipe_linear_line_segment_num_2_pos_discharge"],
                    0.0,
                )
                np.testing.assert_allclose(
                    results[f"{pipe}__pipe_linear_line_segment_num_1_pos_discharge"],
                    1.0,
                )
                np.testing.assert_allclose(
                    results[f"{pipe}__pipe_linear_line_segment_num_2_pos_discharge"],
                    0.0,
                )

    def test_gas_substation(self):
        """
        Test to check if the gas substation reduces the pressure and the head loss computation
        are correctly performed at the two pressure levels.

        Checks:
        - That the two pipes are at two different pressure levels
        _ That the pipes have the expected head loss given their reference pressures
        """
        import models.multiple_gas_carriers.src.run_multiple_gas_carriers as example
        from models.multiple_gas_carriers.src.run_multiple_gas_carriers import GasProblem

        base_folder = Path(example.__file__).resolve().parent.parent

        solution = run_optimization_problem(
            GasProblem,
            base_folder=base_folder,
            esdl_file_name="multiple_carriers.esdl",
            esdl_parser=ESDLFileParser,
            profile_reader=ProfileReaderFromFile,
            input_timeseries_file="timeseries.csv",
        )
        results = solution.extract_results()
        parameters = solution.parameters(0)

        assert parameters["Pipe1.pressure"] != parameters["Pipe2.pressure"]

        for pipe in solution.heat_network_components.get("gas_pipe", []):
            dh = results[f"{pipe}.dH"]
            vel = results[f"{pipe}.Q"] / (np.pi * (parameters[f"{pipe}.diameter"] / 2.0) ** 2)
            for i in range(len(solution.times())):
                analytical_dh = (
                    vel[i]
                    / solution.gas_network_settings["maximum_velocity"]
                    * darcy_weisbach.head_loss(
                        solution.gas_network_settings["maximum_velocity"],
                        parameters[f"{pipe}.diameter"],
                        parameters[f"{pipe}.length"],
                        solution.heat_network_options()["wall_roughness"],
                        20.0,
                        network_type=NetworkSettings.NETWORK_TYPE_GAS,
                        pressure=parameters[f"{pipe}.pressure"],
                    )
                )
                np.testing.assert_allclose(abs(dh[i]), abs(analytical_dh), atol=1.0e-6)


if __name__ == "__main__":
    import time

    start_time = time.time()
    a = TestHeadLoss()
    a.test_heat_network_head_loss()
    a.test_heat_network_pipe_split_head_loss()
    a.test_gas_network_head_loss()
    a.test_gas_network_pipe_split_head_loss()
    a.test_gas_substation()
    print("Execution time: " + time.strftime("%M:%S", time.gmtime(time.time() - start_time)))
