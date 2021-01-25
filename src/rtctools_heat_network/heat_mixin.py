import logging
import math

import casadi as ca

import numpy as np

from rtctools.optimization.collocated_integrated_optimization_problem import (
    CollocatedIntegratedOptimizationProblem,
)
from rtctools.optimization.timeseries import Timeseries

from .base_component_type_mixin import BaseComponentTypeMixin


logger = logging.getLogger("rtctools_heat_network")


class HeatMixin(BaseComponentTypeMixin, CollocatedIntegratedOptimizationProblem):
    def __init__(self, *args, **kwargs):
        # Prepare dicts for additional variables
        self.__flow_direct_var = {}
        self.__flow_direct_bounds = {}
        self.__pipe_to_flow_direct_map = {}

        super().__init__(*args, **kwargs)

    def pre(self):
        super().pre()

        def _get_max_bound(bound):
            if isinstance(bound, np.ndarray):
                return max(bound)
            elif isinstance(bound, Timeseries):
                return max(bound.values)
            else:
                return bound

        def _get_min_bound(bound):
            if isinstance(bound, np.ndarray):
                return min(bound)
            elif isinstance(bound, Timeseries):
                return min(bound.values)
            else:
                return bound

        bounds = self.bounds()

        # Mixed-integer formulation applies only to hot pipes.
        for p in self.hot_pipes:
            flow_dir_var = f"{p}__flow_direct_var"

            self.__pipe_to_flow_direct_map[p] = flow_dir_var
            self.__flow_direct_var[flow_dir_var] = ca.MX.sym(flow_dir_var)

            # Fix the directions that are already implied by the bounds on heat
            # Nonnegative heat implies that flow direction Boolean is equal to one.
            # Nonpositive heat implies that flow direction Boolean is equal to zero.

            heat_in_lb = _get_min_bound(bounds[f"{p}.HeatIn.Heat"][0])
            heat_in_ub = _get_max_bound(bounds[f"{p}.HeatIn.Heat"][1])
            heat_out_lb = _get_min_bound(bounds[f"{p}.HeatOut.Heat"][0])
            heat_out_ub = _get_max_bound(bounds[f"{p}.HeatOut.Heat"][1])

            if (heat_in_lb >= 0.0 and heat_in_ub >= 0.0) or (
                heat_out_lb >= 0.0 and heat_out_ub >= 0.0
            ):
                self.__flow_direct_bounds[flow_dir_var] = (1.0, 1.0)
            elif (heat_in_lb <= 0.0 and heat_in_ub <= 0.0) or (
                heat_out_lb <= 0.0 and heat_out_ub <= 0.0
            ):
                self.__flow_direct_bounds[flow_dir_var] = (0.0, 0.0)
            else:
                self.__flow_direct_bounds[flow_dir_var] = (0.0, 1.0)

            if heat_in_ub <= 0.0 and heat_out_lb >= 0.0:
                raise Exception(f"Heat flow rate in/out of pipe '{p}' cannot be zero.")

    def heat_network_options(self):
        r"""
        Returns a dictionary of heat network specific options.

        +--------------------------------+-----------+-----------------------------+
        | Option                         | Type      | Default value               |
        +================================+===========+========================-----+
        | ``maximum_temperature_der``    | ``float`` | ``2.0`` °C/hour             |
        +--------------------------------+-----------+-----------------------------+
        | ``maximum_flow_der``           | ``float`` | ``np.inf`` m3/s/hour        |
        +--------------------------------+-----------+-----------------------------+

        The ``maximum_temperature_der`` gives the maximum temperature change
        per hour. Similarly, the ``maximum_flow_der`` parameter gives the
        maximum flow change per hour. These options together are used to
        constrain the maximum heat change per hour allowed in the entire
        network. Note the unit for flow is m3/s, but the change is expressed
        on an hourly basis leading to the ``m3/s/hour`` unit."""

        options = {}

        options["maximum_temperature_der"] = 2.0
        options["maximum_flow_der"] = np.inf

        return options

    @property
    def path_variables(self):
        variables = super().path_variables.copy()
        variables.extend(self.__flow_direct_var.values())
        return variables

    def variable_is_discrete(self, variable):
        if variable in self.__flow_direct_var:
            return True
        else:
            return super().variable_is_discrete(variable)

    def bounds(self):
        bounds = super().bounds()
        bounds.update(self.__flow_direct_bounds)
        return bounds

    def parameters(self, ensemble_member):
        parameters = super().parameters(ensemble_member)

        # Compute the heat loss of each pipe
        # The heat losses have three components:
        # - dependency on the pipe temperature
        # - dependency on the ground temperature
        # - dependency on temperature difference between the supply/return line.
        # This latter term assumes that the supply and return lines lie close
        # to, and thus influence, each other. I.e., the supply line loses
        # heat that is absorbed by the return line. Note that the term dtemp is
        # positive when the pipe is in the supply line and negative otherwise.
        for p in self.heat_network_components["pipe"]:
            length = parameters[f"{p}.length"]
            u_1 = parameters[f"{p}.U_1"]
            u_2 = parameters[f"{p}.U_2"]
            temperature = parameters[f"{p}.temperature"]
            temperature_ground = parameters[f"{p}.T_ground"]
            sign_dtemp = 1 if self.is_hot_pipe(p) else -1
            dtemp = sign_dtemp * (parameters[f"{p}.T_supply"] - parameters[f"{p}.T_return"])

            heat_loss = (
                length * (u_1 - u_2) * temperature
                - (length * (u_1 - u_2) * temperature_ground)
                + (length * u_2 * dtemp)
            )

            if heat_loss < 0:
                raise Exception(f"Heat loss of pipe {p} should be nonnegative.")

            parameters[f"{p}.Heat_loss"] = heat_loss

        return parameters

    def __pipe_rate_heat_change_constraints(self, ensemble_member):
        # To avoid sudden change in heat from a timestep to the next,
        # constraints on d(Heat)/dt are introduced.
        # Information of restrictions on dQ/dt and dT/dt are used, as d(Heat)/dt is
        # proportional to average_temperature * dQ/dt + average_discharge * dT/dt.
        # The average discharge is computed using the assumption that the average velocity is 1.
        constraints = []

        parameters = self.parameters(ensemble_member)
        hn_options = self.heat_network_options()

        t_change = hn_options["maximum_temperature_der"]
        q_change = hn_options["maximum_flow_der"]

        for p in self.hot_pipes:
            variable = f"{p}.HeatIn.Heat"
            dt = np.diff(self.times(variable))

            canonical, sign = self.alias_relation.canonical_signed(variable)
            source_temperature_out = sign * self.state_vector(canonical, ensemble_member)

            # Maximum differences are expressed per hour. We scale appropriately.
            cp = parameters[f"{p}.cp"]
            rho = parameters[f"{p}.rho"]
            diameter = parameters[f"{p}.diameter"]
            area = 0.25 * math.pi * diameter ** 2
            avg_t = parameters[f"{p}.temperature"]
            # Assumption: average velocity is 1 m/s
            avg_v = 1
            avg_q = avg_v * area
            heat_change = cp * rho * (t_change / 3600 * avg_q + q_change / 3600 * avg_t)

            if heat_change < 0:
                raise Exception(f"Heat change of pipe {p} should be nonnegative.")
            elif not np.isfinite(heat_change):
                continue

            var_cur = source_temperature_out[1:]
            var_prev = source_temperature_out[:-1]
            variable_nominal = self.variable_nominal(variable)

            constraints.append(
                (
                    var_cur - var_prev,
                    -heat_change * dt / variable_nominal,
                    heat_change * dt / variable_nominal,
                )
            )

        return constraints

    def __node_mixing_path_constraints(self, ensemble_member):
        constraints = []

        for node, connected_pipes in self.heat_network_topology.nodes.items():
            heat_sum = 0.0
            heat_nominals = []

            for i_conn, (_pipe, orientation) in connected_pipes.items():
                heat_conn = f"{node}.HeatConn[{i_conn + 1}].Heat"
                heat_sum += orientation * self.state(heat_conn)
                heat_nominals.append(self.variable_nominal(heat_conn))

            heat_nominal = np.median(heat_nominals)
            constraints.append((heat_sum / heat_nominal, 0.0, 0.0))

        return constraints

    def __heat_loss_path_constraints(self, ensemble_member):
        constraints = []
        parameters = self.parameters(ensemble_member)

        for p in self.cold_pipes:
            heat_in = self.state(f"{p}.HeatIn.Heat")
            heat_out = self.state(f"{p}.HeatOut.Heat")
            heat_nominal = self.variable_nominal(f"{p}.HeatOut.Heat")

            constraints.append(((heat_in - heat_out) / heat_nominal, 0.0, 0.0))

        for p in self.hot_pipes:
            heat_loss = parameters[f"{p}.Heat_loss"]
            heat_nominal = self.variable_nominal(f"{p}.HeatIn.Heat")

            heat_in = self.state(f"{p}.HeatIn.Heat")
            heat_out = self.state(f"{p}.HeatOut.Heat")

            # Heat loss constraint
            constraints.append(((heat_in - heat_out - heat_loss) / heat_nominal, 0.0, 0.0))

        return constraints

    def __flow_direction_path_constraints(self, ensemble_member):
        constraints = []

        def _get_abs_max_bounds(*bounds):
            max_ = 0.0

            for b in bounds:
                if isinstance(b, np.ndarray):
                    max_ = max(max_, max(abs(b)))
                elif isinstance(b, Timeseries):
                    max_ = max(max_, max(abs(b.values)))
                else:
                    max_ = max(max_, abs(b))

            return max_

        bounds = self.bounds()

        for p in self.hot_pipes:
            flow_dir_var = self.__pipe_to_flow_direct_map[p]

            heat_in = self.state(f"{p}.HeatIn.Heat")
            heat_out = self.state(f"{p}.HeatOut.Heat")
            flow_dir = self.state(flow_dir_var)

            big_m = _get_abs_max_bounds(
                *self.merge_bounds(bounds[f"{p}.HeatIn.Heat"], bounds[f"{p}.HeatOut.Heat"])
            )

            if not np.isfinite(big_m):
                raise Exception(f"Heat in pipe {p} must be bounded")

            # Fix flow direction
            constraints.append(((heat_in - big_m * flow_dir) / big_m, -np.inf, 0.0))
            constraints.append(((heat_in + big_m * (1 - flow_dir)) / big_m, 0.0, np.inf))

            # Flow direction is the same for In and Out. Note that this
            # ensures that the heat going in and/or out of a pipe is more than
            # its heat losses.
            constraints.append(((heat_out - big_m * flow_dir) / big_m, -np.inf, 0.0))
            constraints.append(((heat_out + big_m * (1 - flow_dir)) / big_m, 0.0, np.inf))

        # Pipes that are connected in series should have the same heat direction.
        for pipes in self.heat_network_topology.pipe_series:
            if len(pipes) <= 1:
                continue

            assert (
                len({p for p in pipes if self.is_cold_pipe(p)}) == 0
            ), "Pipe series for Heat models should only contain hot pipes"

            base_flow_dir_var = self.state(self.__pipe_to_flow_direct_map[pipes[0]])

            for p in pipes[1:]:
                flow_dir_var = self.state(self.__pipe_to_flow_direct_map[p])
                constraints.append((base_flow_dir_var - flow_dir_var, 0.0, 0.0))

        return constraints

    def __buffer_path_constraints(self, ensemble_member):
        constraints = []

        for b, ((_, hot_orient), (_, cold_orient)) in self.heat_network_topology.buffers.items():
            heat_nominal = self.variable_nominal(f"{b}.HeatIn.Heat")

            heat_in = self.state(f"{b}.HeatIn.Heat")
            heat_out = self.state(f"{b}.HeatOut.Heat")
            heat_hot = self.state(f"{b}.HeatHot")
            heat_cold = self.state(f"{b}.HeatCold")

            # Note that in the conventional scenario, where the hot pipe out-port is connected
            # to the buffer's in-port and the buffer's out-port is connected to the cold pipe
            # in-port, the orientation of the hot/cold pipe is 1/-1 respectively.
            constraints.append(((heat_hot - hot_orient * heat_in) / heat_nominal, 0.0, 0.0))
            constraints.append(((heat_cold + cold_orient * heat_out) / heat_nominal, 0.0, 0.0))

        return constraints

    def path_constraints(self, ensemble_member):
        constraints = super().path_constraints(ensemble_member)

        constraints.extend(self.__node_mixing_path_constraints(ensemble_member))
        constraints.extend(self.__heat_loss_path_constraints(ensemble_member))
        constraints.extend(self.__flow_direction_path_constraints(ensemble_member))
        constraints.extend(self.__buffer_path_constraints(ensemble_member))

        return constraints

    def constraints(self, ensemble_member):
        constraints = super().constraints(ensemble_member)

        constraints.extend(self.__pipe_rate_heat_change_constraints(ensemble_member))

        return constraints

    def history(self, ensemble_member):
        history = super().history(ensemble_member)

        initial_time = np.array([self.initial_time])
        empty_timeseries = Timeseries(initial_time, [np.nan])
        buffers = self.heat_network_components.get("buffer", [])

        for b in buffers:

            hist_heat_buffer = history.get(f"{b}.Heat_buffer", empty_timeseries).values
            hist_stored_heat = history.get(f"{b}.Stored_heat", empty_timeseries).values

            # One has to provide information of what Heat_buffer (i.e., the heat
            # added/extracted from the buffer at that timestep) is at t0.
            # Else the solution will always extract heat from the buffer at t0.
            # This information can be passed in two ways:
            # - non-trivial history of Heat_buffer at t0;
            # - non-trivial history of Stored_heat.
            # If not known, we assume that Heat_buffer is 0.0 at t0.

            if (len(hist_heat_buffer) < 1 or np.isnan(hist_heat_buffer[0])) and (
                len(hist_stored_heat) <= 1 or np.any(np.isnan(hist_stored_heat[-2:]))
            ):
                history[f"{b}.Heat_buffer"] = Timeseries(initial_time, [0.0])

        # TODO: add ATES when component is available

        return history

    def goal_programming_options(self):
        options = super().goal_programming_options()
        options["keep_soft_constraints"] = True
        return options

    def solver_options(self):
        options = super().solver_options()
        options["casadi_solver"] = "qpsol"
        options["solver"] = "cbc"
        return options

    def post(self):
        super().post()

        results = self.extract_results()

        # The flow directions are the same as the heat directions if the
        # return (i.e. cold) line has zero heat throughout. Here we check that
        # this is indeed the case.
        for p in self.cold_pipes:
            heat_in = results[f"{p}.HeatIn.Heat"]
            heat_out = results[f"{p}.HeatOut.Heat"]
            if np.any(heat_in > 1.0) or np.any(heat_out > 1.0):
                logger.warning(f"Heat directions of pipes might be wrong. Check {p}.")
