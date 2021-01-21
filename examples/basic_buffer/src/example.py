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

    def pre(self):
        super().pre()

        # Extract info on how much heat is expected to be needed from the system
        # TMP fix
        self.tot_heat = 0.0
        for p in self.heat_network_components["pipe"]:
            try:
                self.tot_heat += self.parameters(0)[p + ".Heat_loss"]
            except KeyError:
                pass

        for d in self.heat_network_components["demand"]:
            k = d[6:]
            self.tot_heat += max(self.get_timeseries(f"Heat_demand_{k}").values)

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
                    # order=1,
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
                priority=10,
                # order=1,
            )
        )

        # Goal 3: Minimize usage of Source 2
        goals.append(
            RangeGoal(
                self,
                state="source2.Heat_source",
                target_min=np.nan,
                target_max=0.0,
                priority=20,
                # TMP
                state_bounds=(0.0, self.tot_heat),
                order=2,
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

        # This function is called after the optimization run. Thus can be used to do analysis of the
        # results, make plots etc. To get the results of the optimization for a certain variable
        # use: results[name_variable]

        # Compute dH for each pipe.
        # (When Temperature is fixed throughout, computing dH can be easily done via an optimization
        # goal. However, the minimization of the head loss implies a minimization of Q which is
        # problematic as Heat = Q*Temperature. Thus, we compute dH is post-processing. To compute H
        # also the pipe profile needs to be taken into acocunt. This is doable, but unnecessary for
        # now.)

        hn_components = self.heat_network_components

        # for pipe in hn_components["pipe"]:
        #     gravitational_constant = 9.81
        #     friction_factor = 0.04
        #     diameter = self.parameters(0)[pipe + ".diameter"]
        #     length = self.parameters(0)[pipe + ".length"]
        #     # Compute c_v constant (where |dH| ~ c_v*v^2)
        #     c_v = length * friction_factor / (2 * gravitational_constant) / diameter
        #     area = math.pi * diameter ** 2 / 4
        #     q = results[pipe + ".Q"]
        #     v = q / area
        #     dh = -c_v * v ** 2
        #     # Overwrite dH
        #     results[pipe + ".dH"] = dh

        # ****** RESULTS ANALYSIS ******

        # # (Possibly) Useful info for debugging purposes
        # print('Pumps')
        # for pump in self.pumps:
        #     print('Q')
        #     print(np.mean(results[pump+'.Q']))
        #     print('dH')
        #     print(np.mean(results[pump+'.dH']))

        # print("Demands")
        # for d in self.demands:
        #     print(d)
        #     print("dH")
        #     print(np.mean(results[d+'.QTHOut.H']-results[d+'.QTHIn.H']))

        # print("STATS PIPES")
        # tot_pipe_heat_loss = 0.0
        # for p in self.pipes:
        #     print(p)
        #     print('dH')
        #     print(np.mean(results[p+'.dH']))
        #     print('Q')
        #     print(np.mean(results[p+'.Q']))
        #     print('T in')
        #     print(np.mean(results[p+'.QTHIn.T']))
        #     print('T out')
        #     print(np.mean(results[p+'.QTHOut.T']))
        #     t_out = results[p+'.QTHOut.T']
        #     t_in = results[p+'.QTHIn.T']
        #     q = results[p+'.Q']
        #     cp = self.parameters(0)[p+'.cp']
        #     rho = self.parameters(0)[p+'.rho']
        #     length = self.parameters(0)[p+'.length']
        #     U_1 = self.parameters(0)[p+'.U_1']
        #     U_2 = self.parameters(0)[p+'.U_2']
        #     T_g = self.parameters(0)[p+'.T_g']
        #     sign_dT = self.parameters(0)[p+'.sign_dT']
        #     dT = self.parameters(0)[p+'.T_supply'] - self.parameters(0)[p+'.T_return']
        #     heat_loss = (length*(U_1-U_2)*(t_in + t_out)/2 -
        #                     (length*(U_1-U_2)*T_g)+(length*U_2*(sign_dT*dT)))
        #     temp_loss = heat_loss/(cp*rho*q)
        #     print("Avg heat loss in pipe")
        #     print(np.mean(heat_loss))
        #     tot_pipe_heat_loss += np.mean(heat_loss)
        #     print("Avg temperature loss in pipe")
        #     print(np.mean(temp_loss))
        #     print()

        # print("STATS SYSTEM WIDE")
        # # Heat
        # heat_sources = np.mean(results['source1.Heat']) + np.mean(results['source2.Heat'])
        # heat_demands = 0.0
        # for d in self.demands:
        #     k = d[6:]
        #     heat_demands += np.mean(self.get_timeseries('Heat_demand_'+k).values)
        # print("Avg tot heat from sources")
        # print(heat_sources)
        # print("Avg tot heat demand")
        # print(heat_demands)
        # print("Avg tot heat loss in pipes")
        # print(tot_pipe_heat_loss)
        # print("Differences in (total) conservation of heat")
        # print(heat_sources - (heat_demands+tot_pipe_heat_loss))
        # # Temperatures
        # t_supply = []
        # for p in self.pipe_profile_hot:
        #     t_supply_avg = (np.mean(results[p+'.QTHIn.T'])+np.mean(results[p+'.QTHOut.T']))/2
        #     t_supply.append((t_supply_avg))
        # print("Avg supply temperature system profile")
        # print(t_supply)
        # t_return = []
        # for p in self.pipe_profile_cold:
        #     t_return_avg = (np.mean(results[p+'.QTHIn.T'])+np.mean(results[p+'.QTHOut.T']))/2
        #     t_return.append((t_return_avg))
        # print("Avg return temperature system profile")
        # print(t_return)

        # ****** PLOTS ******

        if self.plots:

            import matplotlib.pyplot as plt

            sum_demands = np.full_like(self.times(), 0.0)
            for d in hn_components["demand"]:
                k = d[6:]
                sum_demands += self.get_timeseries("Heat_demand_" + k).values

            # Generate Heat Plot
            # This plot illustrates:
            # * upper plot - heat of the sources and in/out from the storage;
            # * middle plot - stored heat in the buffer;
            # * lower plot - heat demand requested versus result of the optimization

            n_subplots = 3
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

            # Middle Subplot
            axarr[1].set_ylabel("Hot volume buffer")
            axarr[1].plot(
                times,
                results["buffer1.V_hot_tank"],
                label="Volume hot tank",
                linewidth=2,
                color="g",
            )

            # Lower Subplot
            axarr[2].set_ylabel("Demand")
            axarr[2].plot(
                times,
                self.get_timeseries("Heat_demand_7").values,
                label="demand7 req",
                linewidth=2,
                color="r",
            )
            axarr[2].plot(
                times,
                results["demand7.Heat_demand"],
                label="demand7 opt",
                linewidth=2,
                color="g",
                linestyle="--",
            )
            axarr[2].plot(
                times,
                self.get_timeseries("Heat_demand_91").values,
                label="demand91&92 req",
                linewidth=2,
                color="r",
            )
            axarr[2].plot(
                times,
                results["demand91.Heat_demand"],
                label="demand91 opt",
                linewidth=2,
                color="g",
                linestyle="--",
            )
            axarr[2].plot(
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

            # # Generate Route Plots
            # # This plot illustrates:
            # # * upper plot - Temperature from source to demand and back to source;
            # # * middle plot - Heat from source to demand (with buffer in between);
            # # * lower plot - Head from source to demand;

            # # generate x-axis (length network)
            # network_length = []
            # length = 0
            # for pipe in self.pipe_profile_hot:
            #     # route (x coordinate for T_in en T_out)
            #     network_length.append(length)
            #     length += self.parameters(0)[pipe + ".length"]
            #     network_length.append(length)

            # # Temperature in feed line
            # temperature_route_feed = []
            # head_route_feed = []
            # for pipe in self.pipe_profile_hot:
            #     temperature_pipe_hot_in = np.mean(results[pipe + ".QTHIn.T"])
            #     temperature_route_feed.append((temperature_pipe_hot_in))
            #     temperature_pipe_hot_out = np.mean(results[pipe + ".QTHOut.T"])
            #     temperature_route_feed.append((temperature_pipe_hot_out))
            #     # Head along the selected route (feed)
            #     head_route_feed.append(np.mean(results[pipe + ".QTHIn.H"]))
            #     head_route_feed.append(np.mean(results[pipe + ".QTHOut.H"]))

            # # Heat in feed line
            # heat_route_feed = []
            # for pipe in self.pipe_profile_hot:
            #     q = np.mean(results[pipe + ".Q"])
            #     cp = self.parameters(0)[pipe + ".cp"]
            #     rho = self.parameters(0)[pipe + ".rho"]
            #     temperature_pipe_hot_in = np.mean(results[pipe + ".QTHIn.T"])
            #     heat_in = q * cp * rho * temperature_pipe_hot_in
            #     heat_route_feed.append(heat_in)
            #     temperature_pipe_hot_out = np.mean(results[pipe + ".QTHOut.T"])
            #     heat_out = q * cp * rho * temperature_pipe_hot_out
            #     heat_route_feed.append(heat_out)

            # # Heat in return line
            # heat_route_return = []
            # # route same as hot, from source to demand
            # # temperature along route (cold is reversed)
            # for pipe in list(reversed(self.pipe_profile_cold)):
            #     q = np.mean(results[pipe + ".Q"])
            #     cp = self.parameters(0)[pipe + ".cp"]
            #     rho = self.parameters(0)[pipe + ".rho"]
            #     temperature_pipe_cold_out = np.mean(results[pipe + ".QTHOut.T"])
            #     heat_out = q * cp * rho * temperature_pipe_cold_out
            #     heat_route_return.append(heat_out)
            #     temperature_pipe_cold_in = np.mean(results[pipe + ".QTHIn.T"])
            #     heat_in = q * cp * rho * temperature_pipe_cold_in
            #     heat_route_return.append(heat_in)

            # # Temperature and headloss in retour line
            # temperature_route_return = []
            # head_route_return = []
            # for pipe in list(reversed(self.pipe_profile_cold)):
            #     # route same as hot, from source to demand
            #     # temperature along route (cold is reversed)
            #     temperature_pipe_cold_out = np.mean(results[pipe + ".QTHOut.T"])
            #     temperature_route_return.append((temperature_pipe_cold_out))
            #     temperature_pipe_cold_in = np.mean(results[pipe + ".QTHIn.T"])
            #     temperature_route_return.append((temperature_pipe_cold_in))
            #     # Head along the selected route (return)
            #     head_route_return.append(np.mean(results[pipe + ".QTHOut.H"]))
            #     head_route_return.append(np.mean(results[pipe + ".QTHIn.H"]))

            # # Locations for components (sources, demands and buffers)
            # components =
            # [self.source_connections, self.demand_connections, self.buffer_connections]
            # comp_locations = []
            # for comp in components:
            #     locations = []
            #     for con in comp:
            #         position = 0.0
            #         for pipe in self.pipe_profile_hot:
            #             if pipe == con:
            #                 break
            #             position += self.parameters(0)[pipe + ".length"]
            #         locations.append(position)
            #     comp_locations.append(locations)

            # n_subplots = 3
            # fig, axarr = plt.subplots(n_subplots, sharex=True, figsize=(8, 3 * n_subplots))
            # axarr[0].set_title("Route")

            # color = ["k", "g", "y"]
            # types = ["source", "demand", "buffer"]
            # for i in range(len(comp_locations)):
            #     # print(types[i], comp_locations[i])
            #     for xc in comp_locations[i]:
            #         axarr[0].axvline(x=xc, color=color[i], label=types[i], linestyle="--")
            #         axarr[1].axvline(x=xc, color=color[i], label=types[i], linestyle="--")
            #         axarr[2].axvline(x=xc, color=color[i], label=types[i], linestyle="--")

            # # Upper subplot - Temperatures in the pipelines
            # axarr[0].set_ylabel("Temperature [degC]")
            # axarr[0].ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
            # axarr[0].plot(
            #     network_length, temperature_route_feed, label="Feed T", linewidth=2, color="r"
            # )
            # axarr[0].plot(
            #     network_length, temperature_route_return, label="Retour T", linewidth=2, color="b"
            # )

            # # Middle subplot - Heat
            # axarr[1].set_ylabel("Heat")
            # axarr[1].plot(
            #     network_length, heat_route_feed, label="heat feed", linewidth=2, color="r"
            # )
            # axarr[1].plot(
            #     network_length, heat_route_return, label="heat return", linewidth=2, color="b"
            # )
            # axarr[1].set_xlabel("Route [m]")

            # # Bottom subplot - head along the selected route
            # axarr[2].set_ylabel("Head")
            # axarr[2].plot(
            #     network_length,
            #     head_route_return,
            #     label="Retour head",
            #     linewidth=2,
            #     color="c",
            #     linestyle="-",
            # )
            # axarr[2].plot(
            #     network_length,
            #     head_route_feed,
            #     label="feed head",
            #     linewidth=2,
            #     color="c",
            #     linestyle="--",
            # )

            # # Shrink each axis and put a legend to the right of the axis
            # for i in range(n_subplots):
            #     box = axarr[i].get_position()
            #     axarr[i].set_position([box.x0, box.y0, box.width * 0.8, box.height])
            #     axarr[i].legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
            # plt.show()

            n_subplots = 2
            fig, axarr = plt.subplots(n_subplots, sharex=True, figsize=(8, 3 * n_subplots))
            axarr[0].set_title("Buffer pipes")

            # Upper subplot
            axarr[0].set_ylabel("Buffer pipes temperature")
            axarr[0].ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
            axarr[0].plot(
                times,
                results["buffer1.T_hot_pipe"],
                label="T hot pipe",
                linewidth=2,
                color="b",
                linestyle="--",
            )
            axarr[0].plot(
                times,
                results["buffer1.T_cold_pipe"],
                label="T cold pipe",
                linewidth=2,
                color="r",
                linestyle="--",
            )
            axarr[0].plot(
                times,
                results["buffer1.T_hot_tank"],
                label="buffer hot T",
                linewidth=2,
                color="g",
                linestyle="--",
            )
            axarr[0].plot(
                times,
                results["buffer1.T_cold_tank"],
                label="buffer cold T",
                linewidth=2,
                color="c",
                linestyle="--",
            )

            # Lower Subplot
            axarr[1].set_ylabel("Buffer pipes Q")
            axarr[1].ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
            axarr[1].plot(
                times,
                results["buffer1.Q_hot_pipe"],
                label="buffer hot Q",
                linewidth=2,
                color="r",
                linestyle="--",
            )
            axarr[1].plot(
                times,
                results["buffer1.Q_cold_pipe"],
                label="buffer cold Q",
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
            axarr[1].plot(
                times,
                results["buffer1.T_hot_tank"],
                label="buffer1 hot tank",
                linewidth=2,
                color="m",
            )
            axarr[1].plot(
                times,
                results["buffer1.T_cold_tank"],
                label="buffer1 cold tank",
                linewidth=2,
                color="m",
                linestyle="--",
            )

            # Shrink each axis and put a legend to the right of the axis
            for i in range(n_subplots):
                box = axarr[i].get_position()
                axarr[i].set_position([box.x0, box.y0, box.width * 0.8, box.height])
                axarr[i].legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)

            # Output Plot
            plt.show()


if __name__ == "__main__":
    # Run
    start_time = time.time()

    run_heat_network_optimization(HeatProblem, QTHProblem)

    # Output runtime
    print("Execution time: " + time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
