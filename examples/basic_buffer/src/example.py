import logging
import math
import time

import numpy as np

from rtctools.optimization.collocated_integrated_optimization_problem import (
    CollocatedIntegratedOptimizationProblem,
)
from rtctools.optimization.csv_mixin import CSVMixin
from rtctools.optimization.goal_programming_mixin import Goal, GoalProgrammingMixin
from rtctools.optimization.homotopy_mixin import HomotopyMixin
from rtctools.optimization.modelica_mixin import ModelicaMixin
from rtctools.optimization.timeseries import Timeseries
from rtctools.util import run_optimization_problem

from rtctools_heat_network.bounds_to_pipe_flow_directions_mixin import (
    BoundsToPipeFlowDirectionsMixin,
)
from rtctools_heat_network.modelica_component_type_mixin import ModelicaComponentTypeMixin
from rtctools_heat_network.qth_mixin import QTHMixin


logger = logging.getLogger("rtctools")


class RangeGoal(Goal):
    def __init__(
        self,
        optimization_problem,
        state,
        state_bounds,
        target_min,
        target_max,
        priority,
        weight=1.0,
        order=2,
    ):
        self.state = state
        self.target_min = target_min
        self.target_max = target_max
        self.priority = priority
        self.weight = weight
        self.order = order
        self.function_range = state_bounds
        self.function_nominal = abs((state_bounds[1] + state_bounds[0]) / 2.0)

    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.state(self.state)


class MinimizeGoal(Goal):
    def __init__(self, optimization_problem, state, priority, function_nominal=1.0, order=1):
        self.state = state
        self.function_nominal = function_nominal
        self.priority = priority
        self.order = order

    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.state(self.state) / self.function_nominal


class MaximizeGoal(Goal):
    def __init__(self, optimization_problem, state, priority, function_nominal=1.0, order=1):
        self.state = state
        self.function_nominal = function_nominal
        self.priority = priority
        self.order = order

    def function(self, optimization_problem, ensemble_member):
        return -optimization_problem.state(self.state) / self.function_nominal


class Example(
    BoundsToPipeFlowDirectionsMixin,
    QTHMixin,
    ModelicaComponentTypeMixin,
    HomotopyMixin,
    GoalProgrammingMixin,
    CSVMixin,
    ModelicaMixin,
    CollocatedIntegratedOptimizationProblem,
):

    # Set whether flow in/out the buffer should be bidirectional
    bidirectional_flow_buffer = False

    # Set whether show plots or not
    plots = False

    # Create useful lists

    # list of pipe-routes for plots
    pipe_profile_hot = [
        "pipe1a_hot",
        "pipe1b_hot",
        "pipe5_hot",
        "pipe7_hot",
        "pipe9_hot",
        "pipe15_hot",
        "pipe25_hot",
        "pipe27_hot",
        "pipe29_hot",
        "pipe31_hot",
        "pipe32_hot",
    ]

    pipe_profile_cold = [
        "pipe32_cold",
        "pipe31_cold",
        "pipe29_cold",
        "pipe27_cold",
        "pipe25_cold",
        "pipe15_cold",
        "pipe9_cold",
        "pipe7_cold",
        "pipe5_cold",
        "pipe1b_cold",
        "pipe1a_cold",
    ]

    demand_connections = [
        "pipe27_hot",
        "pipe31_hot",
        "demand91",
    ]

    source_connections = ["pipe1a_hot", "pipe5_hot"]

    buffer_connections = ["pipe15_hot"]

    def constraints(self, ensemble_member):
        constraints = super().constraints(ensemble_member)

        # Amount of heat stored in the buffer at the beginning of the time horizon is set to 0.0
        t0 = self.times()[0]
        constraints.append(
            (
                self.state_at("buffer1.Stored_heat", t0)
                / self.variable_nominal("buffer1.Stored_heat"),
                0.0,
                0.0,
            )
        )

        return constraints

    def path_goals(self):
        goals = super().path_goals()
        # Similarly to path_constraints, path goals are goals that will be applied at every time
        # step.

        # There are two types of goals in rtc-tools:
        # * set a minimum and(/or) maximum target
        # * minimize a certain function

        # You can see the goals classes in the beginning of this code.
        # RangeGoal: sets a minimum and/or a maximum target on a certain state.
        # One has to provide the state, the target_min and target_max (set np.nan if it doesn't
        # apply), state_bounds which are the physical lower and upper bounds to the variable, the
        # priority of the goal and (optionally) the order. Order=2 means that target violations will
        # be punished quadratically. Order=1 means that violations are punished linearly. (If you
        # play around with the order of the goal at priority 3 you will see the effect kicking in.)

        # MaximizeGoal: maximizes the given state.

        # We have four different goals:
        # * priority 1: match the Heat demand
        # * priority 2: extract a certain (constant) heat from source1
        # * priority 3: minimize the usage of source2

        components = self.heat_network_components

        # Match the demand target heat
        for d in components["demand"]:
            k = d[6:]
            var = d + ".Heat_demand"
            target_heat = self.get_timeseries("Heat_demand_" + k)
            # Note: with the self.get_timeseries function you can extract the timeseries that are
            # in the timeseries_import file in the input folder.
            # Timeseries objects have (times, values) as property.
            target_heat_val = target_heat.values

            target_heat_ts = Timeseries(self.times(), target_heat_val)
            lb = min(target_heat_ts.values) * 0.9
            ub = max(target_heat_ts.values) * 1.1
            goals.append(
                RangeGoal(
                    self,
                    state=var,
                    target_min=target_heat_ts,
                    target_max=target_heat_ts,
                    state_bounds=(lb, ub),
                    priority=1,
                    order=1,
                )
            )

        # Extract certain heat from source1
        goals.append(
            RangeGoal(
                self,
                state="source1.Heat_source",
                target_min=1e5,
                target_max=1e5,
                state_bounds=(0.0, 1.5e6),
                priority=2,
                order=2,
            )
        )

        # Minimize the usage of source2
        goals.append(
            RangeGoal(
                self,
                state="source2.Heat_source",
                target_max=0.0,
                target_min=np.nan,
                state_bounds=(0.0, 1.5e6),
                priority=3,
                order=2,
            )
        )

        # In/out pipes to and from the buffer should have bidirectional flow
        if self.bidirectional_flow_buffer:
            for p in components["pipe"]:
                if "in" in p:
                    goals.append(
                        RangeGoal(
                            self,
                            state=p + ".Q",
                            target_min=np.nan,
                            target_max=0.0,
                            state_bounds=(0.0, 0.023),
                            priority=4,
                            weight=1.0,
                            order=1,
                        )
                    )

        return goals

    def goal_programming_options(self):
        options = super().goal_programming_options()
        if self.bidirectional_flow_buffer:
            # To ensure bidirectional flow, need to introduce some slack
            options["constraint_relaxation"] = 1e-5
        return options

    # Overwrite default solver options
    def solver_options(self):
        options = super().solver_options()
        solver = options["solver"]
        options[solver]["nlp_scaling_method"] = "none"
        options[solver]["linear_system_scaling"] = "none"
        options[solver]["linear_scaling_on_demand"] = "no"
        options[solver]["max_iter"] = 1000
        options[solver]["tol"] = 1e-5
        options[solver]["acceptable_tol"] = 1e-5
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

        for pipe in hn_components["pipe"]:
            gravitational_constant = 9.81
            friction_factor = 0.04
            diameter = self.parameters(0)[pipe + ".diameter"]
            length = self.parameters(0)[pipe + ".length"]
            # Compute c_v constant (where |dH| ~ c_v*v^2)
            c_v = length * friction_factor / (2 * gravitational_constant) / diameter
            area = math.pi * diameter ** 2 / 4
            q = results[pipe + ".Q"]
            v = q / area
            dh = -c_v * v ** 2
            # Overwrite dH
            results[pipe + ".dH"] = dh

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
                results["source1.Heat"],
                label="source1",
                linewidth=2,
                color="b",
                linestyle="--",
            )
            axarr[0].plot(
                times,
                results["source2.Heat"],
                label="source2",
                linewidth=2,
                color="r",
                linestyle="--",
            )
            axarr[0].plot(
                times,
                results["buffer1.Heat"],
                label="buffer",
                linewidth=2,
                color="g",
                linestyle="--",
            )

            # Middle Subplot
            axarr[1].set_ylabel("Stored Heat Buffer")
            axarr[1].plot(
                times,
                results["StoredHeat_buffer"] / 3600.0,
                label="Stored heat buffer",
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
                results["demand7.Heat"],
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
                results["demand91.Heat"],
                label="demand91 opt",
                linewidth=2,
                color="g",
                linestyle="--",
            )
            axarr[2].plot(
                times,
                results["demand92.Heat"],
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

            # Generate Route Plots
            # This plot illustrates:
            # * upper plot - Temperature from source to demand and back to source;
            # * middle plot - Heat from source to demand (with buffer in between);
            # * lower plot - Head from source to demand;

            # generate x-axis (length network)
            network_length = []
            length = 0
            for pipe in self.pipe_profile_hot:
                # route (x coordinate for T_in en T_out)
                network_length.append(length)
                length += self.parameters(0)[pipe + ".length"]
                network_length.append(length)

            # Temperature in feed line
            temperature_route_feed = []
            head_route_feed = []
            for pipe in self.pipe_profile_hot:
                temperature_pipe_hot_in = np.mean(results[pipe + ".QTHIn.T"])
                temperature_route_feed.append((temperature_pipe_hot_in))
                temperature_pipe_hot_out = np.mean(results[pipe + ".QTHOut.T"])
                temperature_route_feed.append((temperature_pipe_hot_out))
                # Head along the selected route (feed)
                head_route_feed.append(np.mean(results[pipe + ".QTHIn.H"]))
                head_route_feed.append(np.mean(results[pipe + ".QTHOut.H"]))

            # Heat in feed line
            heat_route_feed = []
            for pipe in self.pipe_profile_hot:
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
            for pipe in list(reversed(self.pipe_profile_cold)):
                q = np.mean(results[pipe + ".Q"])
                cp = self.parameters(0)[pipe + ".cp"]
                rho = self.parameters(0)[pipe + ".rho"]
                temperature_pipe_cold_out = np.mean(results[pipe + ".QTHOut.T"])
                heat_out = q * cp * rho * temperature_pipe_cold_out
                heat_route_return.append(heat_out)
                temperature_pipe_cold_in = np.mean(results[pipe + ".QTHIn.T"])
                heat_in = q * cp * rho * temperature_pipe_cold_in
                heat_route_return.append(heat_in)

            # Temperature and headloss in retour line
            temperature_route_return = []
            head_route_return = []
            for pipe in list(reversed(self.pipe_profile_cold)):
                # route same as hot, from source to demand
                # temperature along route (cold is reversed)
                temperature_pipe_cold_out = np.mean(results[pipe + ".QTHOut.T"])
                temperature_route_return.append((temperature_pipe_cold_out))
                temperature_pipe_cold_in = np.mean(results[pipe + ".QTHIn.T"])
                temperature_route_return.append((temperature_pipe_cold_in))
                # Head along the selected route (return)
                head_route_return.append(np.mean(results[pipe + ".QTHOut.H"]))
                head_route_return.append(np.mean(results[pipe + ".QTHIn.H"]))

            # Locations for components (sources, demands and buffers)
            components = [self.source_connections, self.demand_connections, self.buffer_connections]
            comp_locations = []
            for comp in components:
                locations = []
                for con in comp:
                    position = 0.0
                    for pipe in self.pipe_profile_hot:
                        if pipe == con:
                            break
                        position += self.parameters(0)[pipe + ".length"]
                    locations.append(position)
                comp_locations.append(locations)

            n_subplots = 3
            fig, axarr = plt.subplots(n_subplots, sharex=True, figsize=(8, 3 * n_subplots))
            axarr[0].set_title("Route")

            color = ["k", "g", "y"]
            types = ["source", "demand", "buffer"]
            for i in range(len(comp_locations)):
                # print(types[i], comp_locations[i])
                for xc in comp_locations[i]:
                    axarr[0].axvline(x=xc, color=color[i], label=types[i], linestyle="--")
                    axarr[1].axvline(x=xc, color=color[i], label=types[i], linestyle="--")
                    axarr[2].axvline(x=xc, color=color[i], label=types[i], linestyle="--")

            # Upper subplot - Temperatures in the pipelines
            axarr[0].set_ylabel("Temperature [degC]")
            axarr[0].ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
            axarr[0].plot(
                network_length, temperature_route_feed, label="Feed T", linewidth=2, color="r"
            )
            axarr[0].plot(
                network_length, temperature_route_return, label="Retour T", linewidth=2, color="b"
            )

            # Middle subplot - Heat
            axarr[1].set_ylabel("Heat")
            axarr[1].plot(
                network_length, heat_route_feed, label="heat feed", linewidth=2, color="r"
            )
            axarr[1].plot(
                network_length, heat_route_return, label="heat return", linewidth=2, color="b"
            )
            axarr[1].set_xlabel("Route [m]")

            # Bottom subplot - head along the selected route
            axarr[2].set_ylabel("Head")
            axarr[2].plot(
                network_length,
                head_route_return,
                label="Retour head",
                linewidth=2,
                color="c",
                linestyle="-",
            )
            axarr[2].plot(
                network_length,
                head_route_feed,
                label="feed head",
                linewidth=2,
                color="c",
                linestyle="--",
            )

            # Shrink each axis and put a legend to the right of the axis
            for i in range(n_subplots):
                box = axarr[i].get_position()
                axarr[i].set_position([box.x0, box.y0, box.width * 0.8, box.height])
                axarr[i].legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
            plt.show()

        super().post()


# Run
start_time = time.time()

run_optimization_problem(Example)

# Output runtime
print("Execution time: " + time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
