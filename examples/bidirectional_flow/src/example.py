import time

import numpy as np

from rtctools.optimization.collocated_integrated_optimization_problem import (
    CollocatedIntegratedOptimizationProblem,
)
from rtctools.optimization.csv_mixin import CSVMixin
from rtctools.optimization.goal_programming_mixin import Goal, GoalProgrammingMixin
from rtctools.optimization.homotopy_mixin import HomotopyMixin
from rtctools.optimization.linearized_order_goal_programming_mixin import (
    LinearizedOrderGoalProgrammingMixin,
)
from rtctools.optimization.modelica_mixin import ModelicaMixin
from rtctools.optimization.timeseries import Timeseries

from rtctools_heat_network.heat_mixin import HeatMixin
from rtctools_heat_network.modelica_component_type_mixin import ModelicaComponentTypeMixin
from rtctools_heat_network.qth_mixin import QTHMixin
from rtctools_heat_network.util import run_heat_network_optimization


class RangeGoal(Goal):
    def __init__(
        self,
        optimization_problem,
        state,
        target_min,
        target_max,
        priority,
        state_bounds=None,
        order=2,
    ):
        self.state = state
        self.target_min = target_min
        self.target_max = target_max
        self.priority = priority
        self.order = order
        if state_bounds is None:
            state_bounds = optimization_problem.bounds()[state]
        self.function_range = state_bounds
        self.function_nominal = max((abs(state_bounds[1]) + abs(state_bounds[0])) / 2.0, 1.0)

    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.state(self.state)


class GoalsAndOptions:
    def heat_network_options(self):
        options = super().heat_network_options()
        options["maximum_temperature_der"] = 1.5
        options["maximum_flow_der"] = 0.01
        return options

    def path_goals(self):
        goals = super().path_goals()

        # Goal 1: Match the demand target heat
        for d in self.heat_network_components["demand"]:
            k = d[6:]
            var = d + ".Heat_demand"
            target_heat = self.get_timeseries(f"Heat_demand_{k}")
            lb = min(target_heat.values) * 0.9
            ub = max(target_heat.values) * 1.1

            goals.append(
                RangeGoal(
                    self,
                    state=var,
                    target_min=target_heat,
                    target_max=target_heat,
                    priority=1,
                    state_bounds=(lb, ub),
                )
            )

        # Goal 2: Try to keep Source 1 heat constant
        goals.append(
            RangeGoal(
                self,
                state="source1.Heat_source",
                target_min=1.8e5,
                target_max=1.8e5,
                priority=2,
            )
        )

        # Goal 3: Minimize usage of Source 2
        goals.append(
            RangeGoal(
                self,
                state="source2.Heat_source",
                target_min=np.nan,
                target_max=0.0,
                priority=3,
            )
        )

        return goals


class HeatProblem(
    GoalsAndOptions,
    HeatMixin,
    ModelicaComponentTypeMixin,
    LinearizedOrderGoalProgrammingMixin,
    GoalProgrammingMixin,
    CSVMixin,
    ModelicaMixin,
    CollocatedIntegratedOptimizationProblem,
):

    model_name = "Example_Heat"


class QTHProblem(
    GoalsAndOptions,
    QTHMixin,
    ModelicaComponentTypeMixin,
    HomotopyMixin,
    GoalProgrammingMixin,
    CSVMixin,
    ModelicaMixin,
    CollocatedIntegratedOptimizationProblem,
):
    model_name = "Example_QTH"

    # Set whether show plots and print pipe info or not
    plots = False
    print_pipe_info = False

    def __init__(self, *args, **kwargs):
        self.__q_bounds = {}

        super().__init__(*args, **kwargs)

    def pre(self):
        super().pre()

        # We force a minimum flow requirement through all pipes.
        # NOTE: These limits are higher than in the Modelica file.
        q_pumps_min = 0.0002778
        q_pumps_max = 0.01111

        assert self.ensemble_size == 1
        times = self.times()
        constant_inputs = self.constant_inputs(0)

        for p in [p for p in self.heat_network_components["pipe"] if self.is_hot_pipe(p)]:
            q_ub = np.full_like(times, 0.0)
            q_lb = np.full_like(times, 0.0)

            flow_directions = self.heat_network_flow_directions

            d = constant_inputs[flow_directions[p]].values

            q_ub = np.where(d == 1, q_pumps_max, -q_pumps_min)
            q_lb = np.where(d == 1, q_pumps_min, -q_pumps_max)

            self.__q_bounds[f"{p}.QTHIn.Q"] = (Timeseries(times, q_lb), Timeseries(times, q_ub))

    def bounds(self):
        bounds = super().bounds()
        bounds.update(self.__q_bounds)
        return bounds

    def goal_programming_options(self):
        options = super().goal_programming_options()

        if self.parameters(0)["theta"] > 0.0:
            options["constraint_relaxation"] = 1e-4

        return options

    # Overwrite default solver options
    def solver_options(self):
        options = super().solver_options()

        solver = options["solver"]
        options[solver]["max_iter"] = 1000
        options[solver]["tol"] = 1e-5
        options[solver]["acceptable_tol"] = 1e-5
        options[solver]["nlp_scaling_method"] = "none"
        options[solver]["linear_system_scaling"] = "none"
        options[solver]["linear_scaling_on_demand"] = "no"

        return options

    def post(self):
        times = self.times()
        results = self.extract_results()

        pipe_profile_hot = [
            "pipe1a_hot",
            "pipe1b_hot",
            "pipe579_hot",
            "pipe15_hot",
            "pipe25_hot",
            "pipe27_hot",
            "pipe29_hot",
        ]

        pipe_profile_cold = [
            "pipe29_cold",
            "pipe27_cold",
            "pipe25_cold",
            "pipe15_cold",
            "pipe579_cold",
            "pipe1b_cold",
            "pipe1a_cold",
        ]

        # Check change in temperature in pipes
        for p in self.heat_network_components["pipe"]:
            t_in = results[p + ".QTHIn.T"]
            max_der_change = max(np.abs(t_in[1:] - t_in[:-1]))
            print("Max temperature change in pipe {}: {}".format(p, max_der_change))

        # Print pipe info
        if self.print_pipe_info:
            for p in pipe_profile_hot:
                q_in = results[p + ".QTHIn.Q"]
                t_in = results[p + ".QTHIn.T"]
                print(p)
                print(q_in)
                print(t_in)

            for p in pipe_profile_cold:
                q_in = results[p + ".QTHIn.Q"]
                t_in = results[p + ".QTHIn.T"]
                print(p)
                print(q_in)
                print(t_in)

        # ****** PLOTS ******

        if self.plots:

            import matplotlib.pyplot as plt

            sum_demands = np.full_like(self.times(), 0.0)
            for d in self.heat_network_components["demand"]:
                k = d[6:]
                sum_demands += self.get_timeseries("Heat_demand_" + k).values

            # Generate Heat Plot
            # This plot illustrates:
            # * upper plot - heat of the sources and in/out from the storage;
            # * lower plot - heat demand requested versus result of the optimization

            n_subplots = 2
            fig, axarr = plt.subplots(n_subplots, sharex=True, figsize=(8, 3 * n_subplots))
            axarr[0].set_title("Heat")

            # Upper subplot
            axarr[0].set_ylabel("Heat Sources")
            axarr[0].ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
            axarr[0].plot(
                times,
                results["source1.Heat_source"],
                label="source1",
                linewidth=2,
                color="b",
                linestyle="--",
            )
            axarr[0].plot(
                times,
                results["source2.Heat_source"],
                label="source2",
                linewidth=2,
                color="r",
                linestyle="--",
            )

            # Lower Subplot
            axarr[1].set_ylabel("Demand")
            axarr[1].plot(
                times,
                self.get_timeseries("Heat_demand_7").values,
                label="demand7 req",
                linewidth=2,
                color="r",
            )
            axarr[1].plot(
                times,
                results["demand7.Heat_demand"],
                label="demand7 opt",
                linewidth=2,
                color="g",
                linestyle="--",
            )
            axarr[1].plot(
                times,
                self.get_timeseries("Heat_demand_91").values,
                label="demand91&92 req",
                linewidth=2,
                color="r",
            )
            axarr[1].plot(
                times,
                results["demand91.Heat_demand"],
                label="demand91 opt",
                linewidth=2,
                color="g",
                linestyle="--",
            )
            axarr[1].plot(
                times,
                results["demand92.Heat_demand"],
                label="demand92 opt",
                linewidth=2,
                color="g",
                linestyle="--",
            )

            # Shrink each axis and put a legend to the right of the axis
            for i in range(n_subplots):
                box = axarr[i].get_position()
                axarr[i].set_position([box.x0, box.y0, box.width * 0.8, box.height])
                axarr[i].legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)

            # Output Plot
            plt.show()

            n_subplots = 2
            fig, axarr = plt.subplots(n_subplots, sharex=True, figsize=(8, 3 * n_subplots))
            axarr[0].set_title("Bidirectional pipes")

            # Upper subplot
            axarr[0].set_ylabel("Bidirectional pipes temperature")
            axarr[0].ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
            axarr[0].plot(
                times,
                results["pipe579_hot.QTHIn.T"],
                label="pipe579_hot_in",
                linewidth=2,
                color="b",
                linestyle="--",
            )
            axarr[0].plot(
                times,
                results["pipe579_hot.QTHOut.T"],
                label="pipe579_hot_out",
                linewidth=2,
                color="r",
                linestyle="--",
            )

            # Lower Subplot
            axarr[1].set_ylabel("Bidirectional pipes Q")
            axarr[1].ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
            axarr[1].plot(
                times,
                results["pipe579_hot.QTHIn.Q"],
                label="pipe579_hot",
                linewidth=2,
                color="b",
                linestyle="--",
            )

            # Shrink each axis and put a legend to the right of the axis
            for i in range(n_subplots):
                box = axarr[i].get_position()
                axarr[i].set_position([box.x0, box.y0, box.width * 0.8, box.height])
                axarr[i].legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)

            # Output Plot
            plt.show()

            # Generate Route Plots
            # This plot illustrates:
            # * upper plot - Temperature from source to demand and back to source;
            # * middle plot - Heat from source to demand (with buffer in between);
            # * lower plot - Discharge from source to demand with buffer in middle of network;

            # generate x-axis (length network)
            network_length = []
            length = 0
            for pipe in pipe_profile_hot:
                # route (x coordinate for T_in en T_out)
                network_length.append(length)
                length += self.parameters(0)[pipe + ".length"]
                network_length.append(length)

            # Temperature in feed line
            temperature_route_feed = []
            for pipe in pipe_profile_hot:
                temperature_pipe_hot_in = np.mean(results[pipe + ".QTHIn.T"])
                temperature_route_feed.append((temperature_pipe_hot_in))
                temperature_pipe_hot_out = np.mean(results[pipe + ".QTHOut.T"])
                temperature_route_feed.append((temperature_pipe_hot_out))

            # Heat in feed line
            heat_route_feed = []
            for pipe in pipe_profile_hot:
                q = np.mean(results[pipe + ".Q"])
                cp = self.parameters(0)[pipe + ".cp"]
                rho = self.parameters(0)[pipe + ".rho"]
                temperature_pipe_hot_in = np.mean(results[pipe + ".QTHIn.T"])
                heat_in = q * cp * rho * temperature_pipe_hot_in
                heat_route_feed.append(heat_in)
                temperature_pipe_hot_out = np.mean(results[pipe + ".QTHOut.T"])
                heat_out = q * cp * rho * temperature_pipe_hot_out

                heat_route_feed.append(heat_out)

            # Heat in return line
            heat_route_return = []
            # route same as hot, from source to demand
            # temperature along route (cold is reversed)
            for pipe in list(reversed(pipe_profile_cold)):
                q = np.mean(results[pipe + ".Q"])
                cp = self.parameters(0)[pipe + ".cp"]
                rho = self.parameters(0)[pipe + ".rho"]
                temperature_pipe_cold_out = np.mean(results[pipe + ".QTHOut.T"])
                heat_out = q * cp * rho * temperature_pipe_cold_out
                heat_route_return.append(heat_out)
                temperature_pipe_cold_in = np.mean(results[pipe + ".QTHIn.T"])
                heat_in = q * cp * rho * temperature_pipe_cold_in

                heat_route_return.append(heat_in)

            # Temperature in retour line
            temperature_route_return = []
            for pipe in list(reversed(pipe_profile_cold)):
                # route same as hot, from source to demand
                # temperature along route (cold is reversed)
                temperature_pipe_cold_out = np.mean(results[pipe + ".QTHOut.T"])
                temperature_route_return.append((temperature_pipe_cold_out))
                temperature_pipe_cold_in = np.mean(results[pipe + ".QTHIn.T"])
                temperature_route_return.append((temperature_pipe_cold_in))

            # # Locations for components (sources, demands and buffers)
            # components = [self.source_connections, self.demand_connections]
            # comp_locations = []
            # for comp in components:
            #     locations = []
            #     for con in comp:
            #         position = 0.0
            #         for pipe in pipe_profile_hot:
            #             if pipe == con:
            #                 break
            #             position += self.parameters(0)[pipe + ".length"]
            #         locations.append(position)
            #     comp_locations.append(locations)

            n_subplots = 2
            fig, axarr = plt.subplots(n_subplots, sharex=True, figsize=(8, 3 * n_subplots))
            axarr[0].set_title("Route")

            # Upper subplot
            axarr[1].set_ylabel("Temperature [degC]")
            axarr[1].ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
            axarr[1].plot(
                network_length, temperature_route_feed, label="Feed T", linewidth=2, color="r"
            )
            axarr[1].plot(
                network_length, temperature_route_return, label="Return T", linewidth=2, color="b"
            )

            # color = ["k", "g", "y"]
            # types = ["source", "demand"]
            # for i in range(len(comp_locations)):
            #     # print(types[i], comp_locations[i])
            #     for xc in comp_locations[i]:
            #         # axarr[0].axvline(x=xc, color=color[i], label=types[i], linestyle="--")
            #         # axarr[1].axvline(x=xc, color=color[i], label=types[i], linestyle="--")
            #         axarr[0].axvline(x=xc, color=color[i], label=types[i], linestyle="--")

            # Lower subplot
            axarr[0].set_ylabel("Heat")
            axarr[0].plot(
                network_length, heat_route_feed, label="heat feed", linewidth=2, color="r"
            )
            axarr[0].plot(
                network_length, heat_route_return, label="heat return", linewidth=2, color="b"
            )
            axarr[0].set_xlabel("Route [m]")

            # Shrink each axis and put a legend to the right of the axis
            for i in range(n_subplots):
                box = axarr[i].get_position()
                axarr[i].set_position([box.x0, box.y0, box.width * 0.8, box.height])
                axarr[i].legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
            plt.show()

            n_subplots = 2
            fig, axarr = plt.subplots(n_subplots, sharex=True, figsize=(8, 3 * n_subplots))
            axarr[0].set_title("Q / Temperature profile")

            # Upper subplot
            axarr[0].set_ylabel("Q sources")
            axarr[0].ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
            axarr[0].plot(
                times,
                results["source1.QTHOut.Q"],
                label="source 1",
                linewidth=2,
                color="r",
                linestyle="--",
            )
            axarr[0].plot(
                times,
                results["source2.QTHOut.Q"],
                label="source 2",
                linewidth=2,
                color="g",
                linestyle="--",
            )
            axarr[0].plot(
                times,
                results["demand7.QTHOut.Q"],
                label="demand7",
                linewidth=2,
                color="b",
                linestyle="--",
            )

            axarr[0].plot(
                times,
                results["demand91.QTHOut.Q"],
                label="demand91",
                linewidth=2,
                color="c",
                linestyle="--",
            )

            axarr[0].plot(
                times,
                results["demand92.QTHOut.Q"],
                label="demand92",
                linewidth=2,
                color="y",
                linestyle="--",
            )

            # Lower Subplot
            axarr[1].set_ylabel("Temperature sources")
            axarr[1].plot(
                times,
                results["source1.QTHIn.T"],
                label="source1 In",
                linewidth=2,
                color="r",
            )
            axarr[1].plot(
                times,
                results["source1.QTHOut.T"],
                label="source1 Out",
                linewidth=2,
                color="r",
                linestyle="--",
            )

            axarr[1].plot(
                times,
                results["source2.QTHIn.T"],
                label="source2 In",
                linewidth=2,
                color="g",
            )
            axarr[1].plot(
                times,
                results["source2.QTHOut.T"],
                label="source2 Out",
                linewidth=2,
                color="g",
                linestyle="--",
            )

            axarr[1].plot(
                times,
                results["demand7.QTHIn.T"],
                label="demand7 In",
                linewidth=2,
                color="b",
            )
            axarr[1].plot(
                times,
                results["demand7.QTHOut.T"],
                label="demand7 Out",
                linewidth=2,
                color="b",
                linestyle="--",
            )

            axarr[1].plot(
                times,
                results["demand92.QTHIn.T"],
                label="demand92 In",
                linewidth=2,
                color="y",
            )
            axarr[1].plot(
                times,
                results["demand92.QTHOut.T"],
                label="demand92 Out",
                linewidth=2,
                color="y",
                linestyle="--",
            )
            axarr[1].plot(
                times,
                results["demand91.QTHIn.T"],
                label="demand91 In",
                linewidth=2,
                color="c",
            )
            axarr[1].plot(
                times,
                results["demand91.QTHOut.T"],
                label="demand91 Out",
                linewidth=2,
                color="c",
                linestyle="--",
            )

            # Shrink each axis and put a legend to the right of the axis
            for i in range(n_subplots):
                box = axarr[i].get_position()
                axarr[i].set_position([box.x0, box.y0, box.width * 0.8, box.height])
                axarr[i].legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)

            # Output Plot
            plt.show()

        super().post()


# Run
start_time = time.time()

run_heat_network_optimization(HeatProblem, QTHProblem)

# Output runtime
print("Execution time: " + time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
