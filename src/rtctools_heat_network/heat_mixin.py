import logging
import math

import casadi as ca

import numpy as np

from rtctools.optimization.collocated_integrated_optimization_problem import (
    CollocatedIntegratedOptimizationProblem,
)
from rtctools.optimization.timeseries import Timeseries

from rtctools_heat_network._heat_loss_u_values_pipe import heat_loss_u_values_pipe

from .base_component_type_mixin import BaseComponentTypeMixin
from .head_loss_mixin import HeadLossOption, _HeadLossMixin


logger = logging.getLogger("rtctools_heat_network")


class HeatMixin(_HeadLossMixin, BaseComponentTypeMixin, CollocatedIntegratedOptimizationProblem):

    __allowed_head_loss_options = {
        HeadLossOption.NO_HEADLOSS,
        HeadLossOption.LINEAR,
        HeadLossOption.LINEARIZED_DW,
    }

    _hn_prefix = "Heat"

    def __init__(self, *args, **kwargs):
        # Prepare dicts for additional variables
        self.__flow_direct_var = {}
        self.__flow_direct_bounds = {}
        self.__pipe_to_flow_direct_map = {}

        self.__pipe_disconnect_var = {}
        self.__pipe_disconnect_var_bounds = {}
        self.__pipe_disconnect_map = {}

        super().__init__(*args, **kwargs)

    def pre(self):
        super().pre()

        options = self.heat_network_options()
        parameters = self.parameters(0)

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

            if options["minimum_velocity"] > 0.0 and parameters[f"{p}.disconnectable"]:
                disconnected_var = f"{p}__is_disconnected"

                self.__pipe_disconnect_map[p] = disconnected_var
                self.__pipe_disconnect_var[disconnected_var] = ca.MX.sym(disconnected_var)
                self.__pipe_disconnect_var_bounds[disconnected_var] = (0.0, 1.0)

            if heat_in_ub <= 0.0 and heat_out_lb >= 0.0:
                raise Exception(f"Heat flow rate in/out of pipe '{p}' cannot be zero.")

    def heat_network_options(self):
        r"""
        Returns a dictionary of heat network specific options.

        +--------------------------------------+-----------+-----------------------------+
        | Option                               | Type      | Default value               |
        +======================================+===========+=============================+
        | ``minimum_pressure_far_point``       | ``float`` | ``1.0`` bar                 |
        +--------------------------------------+-----------+-----------------------------+
        | ``maximum_temperature_der``          | ``float`` | ``2.0`` °C/hour             |
        +--------------------------------------+-----------+-----------------------------+
        | ``maximum_flow_der``                 | ``float`` | ``np.inf`` m3/s/hour        |
        +--------------------------------------+-----------+-----------------------------+
        | ``minimum_velocity``                 | ``float`` | ``0.005`` m/s               |
        +--------------------------------------+-----------+-----------------------------+
        | ``head_loss_option`` (inherited)     | ``enum``  | ``HeadLossOption.LINEAR``   |
        +--------------------------------------+-----------+-----------------------------+
        | ``minimize_head_losses`` (inherited) | ``bool``  | ``False``                   |
        +--------------------------------------+-----------+-----------------------------+

        The ``maximum_temperature_der`` gives the maximum temperature change
        per hour. Similarly, the ``maximum_flow_der`` parameter gives the
        maximum flow change per hour. These options together are used to
        constrain the maximum heat change per hour allowed in the entire
        network. Note the unit for flow is m3/s, but the change is expressed
        on an hourly basis leading to the ``m3/s/hour`` unit.

        The ``minimum_velocity`` is the minimum absolute value of the velocity
        in every pipe. It is mostly an option to improve the stability of the
        solver in a possibly subsequent QTH problem: the default value of
        `0.005` m/s helps the solver by avoiding the difficult case where
        discharges get close to zero.

        Note that the inherited options ``head_loss_option`` and
        ``minimize_head_losses`` are changed from their default values to
        ``HeadLossOption.LINEAR`` and ``False`` respectively.
        """

        options = super().heat_network_options()

        options["minimum_pressure_far_point"] = 1.0
        options["maximum_temperature_der"] = 2.0
        options["maximum_flow_der"] = np.inf
        options["minimum_velocity"] = 0.005
        options["head_loss_option"] = HeadLossOption.LINEAR
        options["minimize_head_losses"] = False

        return options

    @property
    def path_variables(self):
        variables = super().path_variables.copy()
        variables.extend(self.__flow_direct_var.values())
        variables.extend(self.__pipe_disconnect_var.values())
        return variables

    def variable_is_discrete(self, variable):
        if variable in self.__flow_direct_var or variable in self.__pipe_disconnect_var:
            return True
        else:
            return super().variable_is_discrete(variable)

    def bounds(self):
        bounds = super().bounds()
        bounds.update(self.__flow_direct_bounds)
        bounds.update(self.__pipe_disconnect_var_bounds)
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
            u_kwargs = {
                "inner_diameter": parameters[f"{p}.diameter"],
                "insulation_thicknesses": parameters[f"{p}.insulation_thickness"],
                "conductivities_insulation": parameters[f"{p}.conductivity_insulation"],
                "conductivity_subsoil": parameters[f"{p}.conductivity_subsoil"],
                "depth": parameters[f"{p}.depth"],
                "h_surface": parameters[f"{p}.h_surface"],
                "pipe_distance": parameters[f"{p}.pipe_pair_distance"],
            }

            # NaN values mean we use the function default
            u_kwargs = {k: v for k, v in u_kwargs.items() if not np.all(np.isnan(v))}
            u_1, u_2 = heat_loss_u_values_pipe(**u_kwargs)

            length = parameters[f"{p}.length"]
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

    def __node_heat_mixing_path_constraints(self, ensemble_member):
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

    def __node_discharge_mixing_path_constraints(self, ensemble_member):
        constraints = []

        for node, connected_pipes in self.heat_network_topology.nodes.items():
            q_sum = 0.0
            q_nominals = []

            for i_conn, (_pipe, orientation) in connected_pipes.items():
                q_conn = f"{node}.HeatConn[{i_conn + 1}].Q"
                q_sum += orientation * self.state(q_conn)
                q_nominals.append(self.variable_nominal(q_conn))

            q_nominal = np.median(q_nominals)
            constraints.append((q_sum / q_nominal, 0.0, 0.0))

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

    @staticmethod
    def __get_abs_max_bounds(*bounds):
        max_ = 0.0

        for b in bounds:
            if isinstance(b, np.ndarray):
                max_ = max(max_, max(abs(b)))
            elif isinstance(b, Timeseries):
                max_ = max(max_, max(abs(b.values)))
            else:
                max_ = max(max_, abs(b))

        return max_

    def __flow_direction_path_constraints(self, ensemble_member):
        constraints = []
        options = self.heat_network_options()
        parameters = self.parameters(ensemble_member)

        bounds = self.bounds()

        # These constraints are redundant with the discharge ones. However,
        # CBC tends to get confused and return significantly infeasible
        # results if we remove them.
        for p in self.hot_pipes:
            flow_dir_var = self.__pipe_to_flow_direct_map[p]

            heat_in = self.state(f"{p}.HeatIn.Heat")
            heat_out = self.state(f"{p}.HeatOut.Heat")
            flow_dir = self.state(flow_dir_var)

            big_m = self.__get_abs_max_bounds(
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

        minimum_velocity = options["minimum_velocity"]
        maximum_velocity = options["maximum_velocity"]

        # Also ensure that the discharge has the same sign as the heat.
        for p in self.heat_network_components["pipe"]:
            # FIXME: Enable heat in cold pipes as well.
            if self.is_cold_pipe(p):
                hot_pipe = self.cold_to_hot_pipe(p)
            else:
                hot_pipe = p

            flow_dir_var = self.__pipe_to_flow_direct_map[hot_pipe]
            flow_dir = self.state(flow_dir_var)

            is_disconnected_var = self.__pipe_disconnect_map.get(hot_pipe)

            if is_disconnected_var is None:
                is_disconnected = 0.0
            else:
                is_disconnected = self.state(is_disconnected_var)

            q_pipe = self.state(f"{p}.Q")
            diameter = parameters[f"{p}.diameter"]
            area = 0.25 * math.pi * diameter ** 2
            maximum_discharge = maximum_velocity * area

            if math.isfinite(minimum_velocity):
                minimum_discharge = minimum_velocity * area
            else:
                minimum_discharge = 0.0

            big_m = maximum_discharge + minimum_discharge

            if minimum_discharge > 0.0 and is_disconnected_var is not None:
                constraint_nominal = (minimum_discharge * big_m) ** 0.5
            else:
                constraint_nominal = big_m

            constraints.append(
                (
                    (q_pipe - big_m * flow_dir + (1 - is_disconnected) * minimum_discharge)
                    / constraint_nominal,
                    -np.inf,
                    0.0,
                )
            )
            constraints.append(
                (
                    (q_pipe + big_m * (1 - flow_dir) - (1 - is_disconnected) * minimum_discharge)
                    / constraint_nominal,
                    0.0,
                    np.inf,
                )
            )

            # If a pipe is disconnected, the discharge should be zero
            if is_disconnected_var is not None:
                constraints.append(((q_pipe - (1 - is_disconnected) * big_m) / big_m, -np.inf, 0.0))

                constraints.append(((q_pipe + (1 - is_disconnected) * big_m) / big_m, 0.0, np.inf))

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

    def __demand_heat_to_discharge_path_constraints(self, ensemble_member):
        constraints = []
        parameters = self.parameters(ensemble_member)

        for d in self.heat_network_components["demand"]:
            q_nominal = parameters[f"{d}.Q_nominal"]
            cp = parameters[f"{d}.cp"]
            rho = parameters[f"{d}.rho"]
            dt = parameters[f"{d}.T_supply"] - parameters[f"{d}.T_return"]
            heat_nominal = cp * rho * dt * q_nominal

            discharge = self.state(f"{d}.Q")
            heat_consumed = self.state(f"{d}.Heat_demand")

            constraints.append(
                ((heat_consumed - cp * rho * dt * discharge) / heat_nominal, 0.0, 0.0)
            )

        return constraints

    def __source_heat_to_discharge_path_constraints(self, ensemble_member):
        constraints = []
        parameters = self.parameters(ensemble_member)

        for s in self.heat_network_components["source"]:
            q_nominal = parameters[f"{s}.Q_nominal"]
            cp = parameters[f"{s}.cp"]
            rho = parameters[f"{s}.rho"]
            dt = parameters[f"{s}.T_supply"] - parameters[f"{s}.T_return"]
            heat_nominal = cp * rho * dt * q_nominal

            discharge = self.state(f"{s}.Q")
            heat_production = self.state(f"{s}.Heat_source")

            constraints.append(
                ((heat_production - cp * rho * dt * discharge) / heat_nominal, 0.0, np.inf)
            )

        return constraints

    def __pipe_heat_to_discharge_path_constraints(self, ensemble_member):
        constraints = []
        parameters = self.parameters(ensemble_member)

        sum_heat_losses = sum(
            parameters[f"{p}.Heat_loss"] for p in self.heat_network_components["pipe"]
        )

        for p in self.hot_pipes:
            cp = parameters[f"{p}.cp"]
            rho = parameters[f"{p}.rho"]
            dt = parameters[f"{p}.T_supply"] - parameters[f"{p}.T_return"]
            heat_to_discharge_fac = 1.0 / (cp * rho * dt)

            flow_dir_var = self.__pipe_to_flow_direct_map[p]
            flow_dir = self.state(flow_dir_var)
            scaled_heat_in = self.state(f"{p}.HeatIn.Heat") * heat_to_discharge_fac
            scaled_heat_out = self.state(f"{p}.HeatOut.Heat") * heat_to_discharge_fac
            pipe_q = self.state(f"{p}.Q")

            # We do not want Big M to be too tight in this case, as it results
            # in a rather hard yes/no constraint as far as feasibility on e.g.
            # a single source system is concerned. Use a factor of 2 to give
            # some slack.
            big_m = 2 * sum_heat_losses * heat_to_discharge_fac

            for heat in (scaled_heat_in, scaled_heat_out):
                constraints.append(((heat - pipe_q + big_m * (1 - flow_dir)) / big_m, 0.0, np.inf))
                constraints.append(((heat - pipe_q - big_m * flow_dir) / big_m, -np.inf, 0.0))

        return constraints

    def __buffer_heat_to_discharge_path_constraints(self, ensemble_member):
        constraints = []
        parameters = self.parameters(ensemble_member)
        bounds = self.bounds()

        for b, ((hot_pipe, hot_pipe_orientation), _) in self.heat_network_topology.buffers.items():
            q_nominal = parameters[f"{b}.Q_nominal"]
            cp = parameters[f"{b}.cp"]
            rho = parameters[f"{b}.rho"]
            dt = parameters[f"{b}.T_supply"] - parameters[f"{b}.T_return"]
            heat_nominal = cp * rho * dt * q_nominal

            discharge = self.state(f"{b}.HeatIn.Q") * hot_pipe_orientation
            # Note that `heat_consumed` can be negative for the buffer; in that case we
            # are extracting heat from it.
            heat_consumed = self.state(f"{b}.Heat_buffer")

            # We want an _equality_ constraint between discharge and heat if the buffer is
            # consuming (i.e. behaving like a "demand"). We want an _inequality_
            # constraint (`|heat| >= |f(Q)|`) just like a "source" component if heat is
            # extracted from the buffer. We accomplish this by disabling one of
            # the constraints with a boolean. Note that `discharge` and `heat_consumed`
            # are guaranteed to have the same sign.
            flow_dir_var = self.__pipe_to_flow_direct_map[hot_pipe]
            is_buffer_charging = hot_pipe_orientation * self.state(flow_dir_var)

            big_m = self.__get_abs_max_bounds(
                *self.merge_bounds(bounds[f"{b}.HeatIn.Heat"], bounds[f"{b}.HeatOut.Heat"])
            )

            constraints.append(
                (
                    (heat_consumed - cp * rho * dt * discharge + (1 - is_buffer_charging) * big_m)
                    / heat_nominal,
                    0.0,
                    np.inf,
                )
            )
            constraints.append(
                ((heat_consumed - cp * rho * dt * discharge) / heat_nominal, -np.inf, 0.0)
            )

        return constraints

    def __state_vector_scaled(self, variable, ensemble_member):
        canonical, sign = self.alias_relation.canonical_signed(variable)
        return (
            self.state_vector(canonical, ensemble_member) * self.variable_nominal(canonical) * sign
        )

    @staticmethod
    def _hn_get_pipe_head_loss_option(pipe, heat_network_options, parameters):
        head_loss_option = heat_network_options["head_loss_option"]

        if head_loss_option == HeadLossOption.LINEAR and parameters[f"{pipe}.has_control_valve"]:
            # If there is a control valve present, we use the more accurate
            # Darcy-Weisbach inequality formulation.
            head_loss_option = HeadLossOption.LINEARIZED_DW

        return head_loss_option

    def _hn_pipe_head_loss_constraints(self, ensemble_member):
        constraints = []

        options = self.heat_network_options()
        parameters = self.parameters(ensemble_member)
        components = self.heat_network_components

        # Set the head loss according to the direction in the pipes. Note that
        # the `.__head_loss` symbol is always positive by definition, but that
        # `.dH` is not (positive when flow is negative, and vice versa).
        # If the pipe is disconnected, we leave the .__head_loss symbol free
        # (and it has no physical meaning). We also do not set any discharge
        # relationship in this case (but dH is still equal to Out - In of
        # course).
        for pipe in components["pipe"]:
            if parameters[f"{pipe}.length"] == 0.0:
                # If the pipe does not have a control valve, the head loss is
                # forced to zero via bounds. If the pipe _does_ have a control
                # valve, then there still is no relationship between the
                # discharge and the head loss/dH.
                continue

            if self.is_cold_pipe(pipe):
                hot_pipe = self.cold_to_hot_pipe(pipe)
            else:
                hot_pipe = pipe

            head_loss_sym = self._hn_pipe_to_head_loss_map[hot_pipe]

            dh = self.__state_vector_scaled(f"{pipe}.dH", ensemble_member)
            head_loss = self.__state_vector_scaled(head_loss_sym, ensemble_member)
            discharge = self.__state_vector_scaled(f"{pipe}.Q", ensemble_member)

            constraints.extend(
                self._hn_pipe_head_loss(pipe, options, parameters, discharge, head_loss, dh)
            )

            # Relate the head loss symbol to the pipe's dH symbol.
            area = 0.25 * math.pi * parameters[f"{pipe}.diameter"] ** 2
            max_discharge = options["maximum_velocity"] * area
            max_head_loss = self._hn_pipe_head_loss(pipe, options, parameters, max_discharge)

            # FIXME: Ugly hack. Cold pipes should be modelled completely with
            # their own integers as well.
            flow_dir = self.__state_vector_scaled(
                self.__pipe_to_flow_direct_map[hot_pipe], ensemble_member
            )

            constraints.append(
                (
                    (-dh - head_loss + (1 - flow_dir) * max_head_loss) / max_head_loss,
                    0.0,
                    np.inf,
                )
            )
            constraints.append(
                ((dh - head_loss + flow_dir * max_head_loss) / max_head_loss, 0.0, np.inf)
            )

        return constraints

    def path_constraints(self, ensemble_member):
        constraints = super().path_constraints(ensemble_member)

        constraints.extend(self.__node_heat_mixing_path_constraints(ensemble_member))
        constraints.extend(self.__heat_loss_path_constraints(ensemble_member))
        constraints.extend(self.__flow_direction_path_constraints(ensemble_member))
        constraints.extend(self.__buffer_path_constraints(ensemble_member))
        constraints.extend(self.__node_discharge_mixing_path_constraints(ensemble_member))
        constraints.extend(self.__demand_heat_to_discharge_path_constraints(ensemble_member))
        constraints.extend(self.__source_heat_to_discharge_path_constraints(ensemble_member))
        constraints.extend(self.__pipe_heat_to_discharge_path_constraints(ensemble_member))
        constraints.extend(self.__buffer_heat_to_discharge_path_constraints(ensemble_member))

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

    def compiler_options(self):
        options = super().compiler_options()
        options["resolve_parameter_values"] = True
        return options

    def post(self):
        super().post()

        results = self.extract_results()
        parameters = self.parameters(0)
        options = self.heat_network_options()

        # The flow directions are the same as the heat directions if the
        # return (i.e. cold) line has zero heat throughout. Here we check that
        # this is indeed the case.
        for p in self.cold_pipes:
            heat_in = results[f"{p}.HeatIn.Heat"]
            heat_out = results[f"{p}.HeatOut.Heat"]
            if np.any(heat_in > 1.0) or np.any(heat_out > 1.0):
                logger.warning(f"Heat directions of pipes might be wrong. Check {p}.")

        for p in self.heat_network_components["pipe"]:
            head_diff = results[f"{p}.HeatIn.H"] - results[f"{p}.HeatOut.H"]
            if parameters[f"{p}.length"] == 0.0 and not parameters[f"{p}.has_control_valve"]:
                atol = self.variable_nominal(f"{p}.HeatIn.H") * 1e-5
                assert np.allclose(head_diff, 0.0, atol=atol)
            else:
                q = results[f"{p}.Q"]
                q_nominal = self.variable_nominal(self.alias_relation.canonical_signed(f"{p}.Q")[0])
                inds = np.abs(q) / q_nominal > 1e-4
                assert np.all(np.sign(head_diff[inds]) == np.sign(q[inds]))

        minimum_velocity = options["minimum_velocity"]
        for p in self.heat_network_components["pipe"]:
            if self.is_cold_pipe(p):
                hot_pipe = self.cold_to_hot_pipe(p)
            else:
                hot_pipe = p
            area = 0.25 * math.pi * parameters[f"{p}.diameter"] ** 2
            q = results[f"{p}.Q"]
            q_w_error_margin = q + 1e-5 * np.sign(q) * self.variable_nominal(f"{p}.Q")
            v = q_w_error_margin / area
            flow_dir = np.round(results[self.__pipe_to_flow_direct_map[hot_pipe]])
            try:
                is_disconnected = np.round(results[self.__pipe_disconnect_map[hot_pipe]])
            except KeyError:
                is_disconnected = np.zeros_like(q)

            inds_disconnected = is_disconnected == 1
            inds_positive = (flow_dir == 1) & ~inds_disconnected
            inds_negative = (flow_dir == 0) & ~inds_disconnected

            assert np.all(v[inds_positive] >= minimum_velocity)
            assert np.all(v[inds_negative] <= -1 * minimum_velocity)

            assert np.all(np.abs(q[inds_disconnected]) / self.variable_nominal(f"{p}.Q") <= 1e-5)

        for p in self.hot_pipes:
            heat_in = results[f"{p}.HeatIn.Heat"]
            heat_out = results[f"{p}.HeatOut.Heat"]
            inds = np.abs(heat_out) > np.abs(heat_in)

            heat = heat_in.copy()
            heat[inds] = heat_out[inds]

            flow_dir_var = np.round(results[self.__pipe_to_flow_direct_map[p]])

            assert np.all(np.sign(heat) == 2 * flow_dir_var - 1)
