import logging
import math
from math import isfinite
from typing import Dict

import casadi as ca

import numpy as np

from rtctools.optimization.collocated_integrated_optimization_problem import (
    CollocatedIntegratedOptimizationProblem,
)
from rtctools.optimization.homotopy_mixin import HomotopyMixin
from rtctools.optimization.timeseries import Timeseries

from rtctools_heat_network._heat_loss_u_values_pipe import heat_loss_u_values_pipe

from .base_component_type_mixin import BaseComponentTypeMixin
from .head_loss_mixin import (
    HeadLossOption,
    _HeadLossMixin,
    _MinimizeHeadLosses as _MinimizeHeadLossesBase,
)
from .heat_network_common import NodeConnectionDirection, PipeFlowDirection

logger = logging.getLogger("rtctools_heat_network")


class _MinimizeHeadLosses(_MinimizeHeadLossesBase):
    @property
    def is_empty(self):
        parameters = self.optimization_problem.parameters(0)
        theta = parameters[self.optimization_problem.homotopy_options()["homotopy_parameter"]]

        # Check if there are any goals before us, otherwise there is not much to do
        all_goals = [*self.optimization_problem.goals(), *self.optimization_problem.path_goals()]
        other_goals = [
            g for g in all_goals if not isinstance(g, _MinimizeHeadLosses) and not g.is_empty
        ]

        return (theta < 1.0) or (not other_goals)


class QTHMixin(_HeadLossMixin, BaseComponentTypeMixin, CollocatedIntegratedOptimizationProblem):
    """
    Adds handling of QTH heat network objects in your model to your
    optimization problem.

    Relevant parameters and variables are read from the model, and from this
    data a set of constraints and objectives are automatically generated, e.g.
    for the head loss and temperature losses in pipes.
    """

    _hn_minimization_goal_class = _MinimizeHeadLosses

    def __init__(self, *args, flow_directions=None, **kwargs):
        """
        :param flow_directions: A dictionary mapping a pipe name to a
            Timeseries of the flow directions of type :py:class:`PipeFlowDirection`.
        """
        super().__init__(*args, **kwargs)

        if flow_directions is not None:
            self.__flow_directions_name_map = {
                p: f"{p}__flow_direction" for p in flow_directions.keys()
            }
            self.__flow_directions = {
                self.__flow_directions_name_map[p]: v for p, v in flow_directions.items()
            }
        else:
            self.__flow_directions_name_map = None
            self.__flow_directions = None

        self.__implied_directions = None
        self.__direction_bounds = None

        if not isinstance(self, HomotopyMixin):
            # Note that we inherit ourselves, as there is a certain in which
            # inheritance is required.
            raise Exception("Class needs inherit from HomotopyMixin")

        self.__flow_direction_bounds = None

    def pre(self):
        self.__flow_direction_bounds = None
        self.__temperature_pipe_theta_zero = None

        super().pre()

        self.__update_temperature_pipe_theta_zero_bounds()

    def heat_network_options(self):
        r"""
        Returns a dictionary of heat network specific options.

        +--------------------------------+-----------+-----------------------------------+
        | Option                         | Type      | Default value                     |
        +================================+===========+===================================+
        | ``maximum_temperature_der``    | ``float`` | ``2.0`` °C/hour                   |
        +--------------------------------+-----------+-----------------------------------+
        | ``max_t_der_bidirect_pipe``    | ``bool``  | ``True``                          |
        +--------------------------------+-----------+-----------------------------------+
        | ``minimum_velocity``           | ``float`` | ``0.005`` m/s                     |
        +--------------------------------+-----------+-----------------------------------+

        The ``maximum_temperature_der`` is the maximum temperature change
        allowed in the network. It is expressed in °C per hour. Note that this
        is a proxy constraint as it does not imply that temperature in the
        entire network is within the wanted limits.

        When the flag ``max_t_der_bidirect_pipe`` is False, the maximum
        temperature change set with ``maximum_temperature_der`` is _not_
        imposed on pipes when the flow direction changes. When it is True (the
        default), it is imposed in cases of flow reversal.

        The ``minimum_velocity`` is the minimum absolute value of the velocity
        in every pipe. It is mostly an option to improve the stability of the
        solver: the default value of `0.005` m/s helps the solver by avoiding
        the difficult case where discharges get close to zero.
        """

        options = super().heat_network_options()

        options["maximum_temperature_der"] = 2.0
        options["max_t_der_bidirect_pipe"] = True
        options["minimum_velocity"] = 0.005

        return options

    @property
    def heat_network_pipe_flow_directions(self) -> Dict[str, str]:
        """
        Maps a pipe name to its corresponding `constant_inputs` Timeseries
        name for the direction.
        """
        if self.__flow_directions_name_map is not None:
            return self.__flow_directions_name_map
        else:
            raise NotImplementedError(
                "Please implement/set the `heat_network_pipe_flow_directions` property"
            )

    def constant_inputs(self, ensemble_member):
        inputs = super().constant_inputs(ensemble_member)
        if self.__flow_directions is not None:
            for k, v in self.__flow_directions.items():
                inputs[k] = v
        return inputs

    def interpolation_method(self, variable=None):
        try:
            if variable in self.__pipe_flow_dir_symbols:
                return self.INTERPOLATION_PIECEWISE_CONSTANT_BACKWARD
            else:
                return super().interpolation_method(variable)
        except AttributeError:
            self.__pipe_flow_dir_symbols = set(self.heat_network_pipe_flow_directions.values())
            # Try again
            return self.interpolation_method(variable)

    def __state_vector_scaled(self, variable, ensemble_member):
        canonical, sign = self.alias_relation.canonical_signed(variable)
        return (
            self.state_vector(canonical, ensemble_member) * self.variable_nominal(canonical) * sign
        )

    def __pipe_heat_loss_constraints(self, ensemble_member):
        parameters = self.parameters(ensemble_member)
        constraints = []

        theta = parameters[self.homotopy_options()["homotopy_parameter"]]
        components = self.heat_network_components

        # At theta=0, the temperature of the hot/cold pipes are constant
        # and equal to the design ones. Thus heat loss equations do not apply.
        if theta == 0.0:
            return []

        interpolated_flow_dir_values = self.__get_interpolated_flow_directions(ensemble_member)

        for p in components["pipe"]:
            temp_in_sym = self.__state_vector_scaled(f"{p}.QTHIn.T", ensemble_member)
            temp_out_sym = self.__state_vector_scaled(f"{p}.QTHOut.T", ensemble_member)
            q_sym = self.__state_vector_scaled(f"{p}.Q", ensemble_member)

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

            cp = parameters[f"{p}.cp"]
            rho = parameters[f"{p}.rho"]
            length = parameters[f"{p}.length"]

            temp_ground = parameters[f"{p}.T_ground"]
            temp_supply = parameters[f"{p}.T_supply"]
            temp_return = parameters[f"{p}.T_return"]
            sign_dtemp = 1 if self.is_hot_pipe(p) else -1
            dtemp = sign_dtemp * (temp_supply - temp_return)

            flow_direction = interpolated_flow_dir_values[p]
            heat_loss_eq = []
            no_heat_loss_eq = []

            # We want to scale the equation appropriately. We therefore find
            # the (approximate) geometric mean of the coefficients in the
            # jacobian.
            heat_nominal = (
                rho * cp * self.variable_nominal(f"{p}.Q") * self.variable_nominal(f"{p}.QTHIn.T")
            )
            heat_loss_nominal = length * (u_1 - u_2) * self.variable_nominal(f"{p}.QTHIn.T")
            equation_nominal = (heat_nominal * heat_loss_nominal) ** 0.5

            # If pipe is connected, add heat losses
            # The heat losses have three components:
            # - dependency on the pipe temperature
            # - dependency on the ground temperature
            # - dependency on temperature difference between the supply/return line.
            # This latter term assumes that the supply and return lines lie close
            # to, and thus influence, each other. I.e., the supply line loses
            # heat that is absorbed by the return line. Note that the term dtemp is
            # positive when the pipe is in the supply line and negative otherwise.
            heat_loss_inds = np.flatnonzero((flow_direction != 0).astype(int)).tolist()
            heat_loss_eq.append(
                (
                    (1 - theta) * (temp_out_sym - temp_in_sym)
                    + theta
                    * (
                        (temp_out_sym - temp_in_sym) * q_sym * cp * rho
                        + length * (u_1 - u_2) * (temp_in_sym + temp_out_sym) / 2
                        - (length * (u_1 - u_2) * temp_ground)
                        + (length * u_2 * dtemp)
                    )
                    / equation_nominal
                )[heat_loss_inds]
            )

            if len(heat_loss_inds) > 0:
                constraints.append((ca.vertcat(*heat_loss_eq), 0.0, 0.0))

            # If pipe is disabled, no heat equations
            no_heat_loss_inds = np.flatnonzero((flow_direction == 0).astype(int)).tolist()
            no_heat_loss_eq.append((temp_out_sym - temp_in_sym)[no_heat_loss_inds])

            if len(no_heat_loss_inds) > 0:
                constraints.append((ca.vertcat(*no_heat_loss_eq), 0.0, 0.0))

        return constraints

    def __node_mixing_constraints(self, ensemble_member):
        parameters = self.parameters(ensemble_member)
        constraints = []

        theta = parameters[self.homotopy_options()["homotopy_parameter"]]

        interpolated_flow_dir_values = self.__get_interpolated_flow_directions(ensemble_member)

        for node, connected_pipes in self.heat_network_topology.nodes.items():
            temperature_node_sym = self.__state_vector_scaled(f"{node}.Tnode", ensemble_member)
            temperature_estimate = parameters[f"{node}.temperature"]

            # Definition of in/outflows
            q_in_sum = 0.0
            q_t_in_sum = 0.0
            q_out_sum = 0.0

            t_out_conn = []

            for i_conn, (pipe, orientation) in connected_pipes.items():
                flow_direction = interpolated_flow_dir_values[pipe]

                assert (
                    len(flow_direction) == temperature_node_sym.size1()
                ), "Collocation times mismatch"

                # The direction at the node is the product of the flow direction and whether
                # the orientation pipe is in or out of the node.
                # A positive flow in a pipe at any time step (= 1) and the pipe orientation
                # into the node (= 1) mean that flow is going into the node (1 * 1 = 1).
                # Similarly, a negative flow in a pipe at any time step (= -1), combined with
                # an orientation _out of_ the node (-1), also means flow going into the node
                # (-1 * -1 = 1)
                node_in_or_out = orientation * flow_direction

                node_in = (node_in_or_out == NodeConnectionDirection.IN).astype(int)
                node_out = (node_in_or_out == NodeConnectionDirection.OUT).astype(int)

                conn_base = f"{node}.QTHConn[{i_conn + 1}]"
                conn_q = self.__state_vector_scaled(f"{conn_base}.Q", ensemble_member)
                conn_q_abs = conn_q * flow_direction
                conn_t = self.__state_vector_scaled(f"{conn_base}.T", ensemble_member)

                # In
                q_in_sum += conn_q_abs * node_in
                q_t_in_sum += conn_q_abs * conn_t * node_in

                # Out
                q_out_sum += conn_q_abs * node_out

                inds = np.flatnonzero(node_out).tolist()
                t_out_conn.append((conn_t - temperature_node_sym)[inds])

            assert q_in_sum.size1() == len(self.times())

            q_nominal = np.median(
                [
                    self.variable_nominal(f"{node}.QTHConn[{i_conn + 1}].Q")
                    for i in range(len(connected_pipes))
                ]
            )
            t_nominal = np.median(
                [
                    self.variable_nominal(f"{node}.QTHConn[{i_conn + 1}].T")
                    for i in range(len(connected_pipes))
                ]
            )
            qt_nominal = q_nominal * t_nominal

            # Conservation of mass
            constraints.append(((q_in_sum - q_out_sum) / q_nominal, 0.0, 0.0))

            # Conservation of heat
            constraints.append(
                (
                    (
                        (1 - theta) * (temperature_node_sym - temperature_estimate) / t_nominal
                        + theta * (q_in_sum * temperature_node_sym - q_t_in_sum) / qt_nominal
                    ),
                    0.0,
                    0.0,
                )
            )

            if theta > 0.0:
                # Temperature of outgoing flows is equal to mixing temperature
                # At theta zero this is implied by the bounds on temperature.
                constraints.append((ca.vertcat(*t_out_conn) / t_nominal, 0.0, 0.0))

        return constraints

    def __buffer_constraints(self, ensemble_member):

        parameters = self.parameters(ensemble_member)
        constraints = []

        theta = parameters[self.homotopy_options()["homotopy_parameter"]]

        interpolated_flow_dir_values = self.__get_interpolated_flow_directions(ensemble_member)

        for b, (hot_pair, cold_pair) in self.heat_network_topology.buffers.items():
            hot_pipe, hot_pipe_orientation = hot_pair
            _, cold_pipe_orientation = cold_pair

            # buffer_is_charging:
            #   1 if buffer is charging (flow into buffer on hot side)
            #  -1 if discharging (flow out of buffer on hot side)
            #   0 if no flow going in/out of buffer
            buffer_is_charging = hot_pipe_orientation * interpolated_flow_dir_values[hot_pipe]
            e = ensemble_member

            # Flows going in/out of the buffer. We want Q_hot_pipe and
            # Q_cold_pipe to be positive when the buffer is charging.
            # Note that in the conventional scenario, where the hot pipe out-port is connected
            # to the buffer's in-port and the buffer's out-port is connected to the cold pipe
            # in-port, the orientation of the hot/cold pipe is 1/-1 respectively.
            q_in = self.__state_vector_scaled(f"{b}.QTHIn.Q", e)
            q_out = self.__state_vector_scaled(f"{b}.QTHOut.Q", e)
            q_hot_pipe = self.__state_vector_scaled(f"{b}.Q_hot_pipe", e)
            q_cold_pipe = self.__state_vector_scaled(f"{b}.Q_cold_pipe", e)

            q_nominal = self.variable_nominal(f"{b}.QTHIn.Q")

            constraints.append(((hot_pipe_orientation * q_in - q_hot_pipe) / q_nominal, 0.0, 0.0))
            constraints.append(
                ((cold_pipe_orientation * q_out + q_cold_pipe) / q_nominal, 0.0, 0.0)
            )

            # Temperature of outgoing flows is equal to buffer temperature
            temp_hot_tank_sym = self.__state_vector_scaled(f"{b}.T_hot_tank", e)
            temp_hot_pipe_sym = self.__state_vector_scaled(f"{b}.T_hot_pipe", e)
            temp_cold_tank_sym = self.__state_vector_scaled(f"{b}.T_cold_tank", e)
            temp_cold_pipe_sym = self.__state_vector_scaled(f"{b}.T_cold_pipe", e)

            t_hot_nominal = self.variable_nominal(f"{b}.T_hot_tank")
            t_cold_nominal = self.variable_nominal(f"{b}.T_cold_tank")

            # At theta=0, the temperature of the pipes is equal to the design temperature.
            # We fix the temperature of the buffer as well.
            # Note that at temperature of the buffer at t0 is already fixed through the model.
            if theta == 0.0:
                constraints.append(
                    ((temp_hot_pipe_sym[1:] - temp_hot_tank_sym[1:]) / t_hot_nominal, 0.0, 0.0)
                )
                constraints.append(
                    ((temp_cold_pipe_sym[1:] - temp_cold_tank_sym[1:]) / t_hot_nominal, 0.0, 0.0)
                )
                break

            # At theta>0, we need to model the buffer temperature and how it is related
            # to the network.
            # There are two part to model:
            # - the water going out of a tank must have the same temperature as the tank itself.
            # - temperature of the each tank

            # Temperature of outgoing flows is equal to buffer temperature

            # Hot tank
            # In this case, there is outgoing flow only when the buffer is discharging.
            # When the buffer is neither charging or discharging, the buffer is disconnected
            # from the network and thus the temperature is not relevant. However, for consistency
            # purposes, we do not want the temperature to take arbitrary values in this case.
            # Thus, we also set it to be equal to the temperature of the buffer.
            # A similar reasoning holds for the cold tank.
            pipe_temp_as_buffer_hot = (buffer_is_charging != 1).astype(int)
            inds_hot = np.flatnonzero(pipe_temp_as_buffer_hot).tolist()
            t_out_conn = (temp_hot_pipe_sym - temp_hot_tank_sym)[inds_hot]
            if len(inds_hot) > 0:
                constraints.append((t_out_conn / t_hot_nominal, 0.0, 0.0))

            # Cold tank
            t_cold_nominal = self.variable_nominal(f"{b}.T_cold_tank")
            pipe_temp_as_buffer_cold = (buffer_is_charging != -1).astype(int)
            inds_cold = np.flatnonzero(pipe_temp_as_buffer_cold).tolist()
            t_out_conn = (temp_cold_pipe_sym - temp_cold_tank_sym)[inds_cold]
            if len(inds_cold) > 0:
                constraints.append((t_out_conn / t_cold_nominal, 0.0, 0.0))

            # Temperature mixing in the buffer
            # There are two set of equations, depending on whether the tank is charging
            # or not charging (i.e., discharging or not used).
            # If tank is charging, there is temperature mixing and thus:
            # * der(T_tank * V_tank) - T_pipe * Q_pipe + heat losses = 0.
            # If the tank is discharging or is not used, there is no temperature mixing and thus:
            # * der(T_tank) + heat losses = 0.
            # Where heat losses are:
            # surface area * heat transfer coefficient * (T_tank - T_outside) / (rho * cp)
            # Surface area is 2 * pi * r * (r + h). It is approximated with an average height
            # when there is no temperature mixing.
            # Note: the equations are not apply at t0

            radius = parameters[f"{b}.radius"]
            height = parameters[f"{b}.height"]
            volume = math.pi * radius ** 2 * height
            avg_surface = math.pi * radius * (radius + height)
            cp = parameters[f"{b}.cp"]
            rho = parameters[f"{b}.rho"]
            heat_transfer_coeff = parameters[f"{b}.heat_transfer_coeff"]
            temp_outside = parameters[f"{b}.T_outside"]

            # Collocation times of those variables must same as the global ones
            assert np.all(self.times() == self.times(f"{b}.T_hot_tank"))
            assert np.all(self.times() == self.times(f"{b}.V_hot_tank"))
            assert np.all(self.times() == self.times(f"{b}.Q_hot_pipe"))
            assert np.all(self.times() == self.times(f"{b}.T_hot_pipe"))
            assert np.all(self.times() == self.times(f"{b}.T_cold_tank"))
            assert np.all(self.times() == self.times(f"{b}.V_cold_tank"))
            assert np.all(self.times() == self.times(f"{b}.Q_cold_pipe"))
            assert np.all(self.times() == self.times(f"{b}.T_cold_pipe"))

            t_hot_tank_curr = self.__state_vector_scaled(f"{b}.T_hot_tank", e)[1:]
            t_hot_tank_prev = self.__state_vector_scaled(f"{b}.T_hot_tank", e)[:-1]
            v_hot_tank_curr = self.__state_vector_scaled(f"{b}.V_hot_tank", e)[1:]
            v_hot_tank_prev = self.__state_vector_scaled(f"{b}.V_hot_tank", e)[:-1]
            q_hot_pipe_curr = self.__state_vector_scaled(f"{b}.Q_hot_pipe", e)[1:]
            t_hot_pipe_curr = self.__state_vector_scaled(f"{b}.T_hot_pipe", e)[1:]

            t_cold_tank_curr = self.__state_vector_scaled(f"{b}.T_cold_tank", e)[1:]
            t_cold_tank_prev = self.__state_vector_scaled(f"{b}.T_cold_tank", e)[:-1]
            v_cold_tank_curr = self.__state_vector_scaled(f"{b}.V_cold_tank", e)[1:]
            v_cold_tank_prev = self.__state_vector_scaled(f"{b}.V_cold_tank", e)[:-1]
            q_cold_pipe_curr = self.__state_vector_scaled(f"{b}.Q_cold_pipe", e)[1:]
            t_cold_pipe_curr = self.__state_vector_scaled(f"{b}.T_cold_pipe", e)[1:]

            dt = np.diff(self.times())

            t_mix_hot = []
            t_mix_cold = []

            hot_mix_inds = np.flatnonzero((buffer_is_charging[1:] == 1).astype(int)).tolist()
            cold_mix_inds = np.flatnonzero((buffer_is_charging[1:] == -1).astype(int)).tolist()
            inactive_inds = np.flatnonzero((buffer_is_charging[1:] == 0).astype(int)).tolist()

            # Hot tank
            t_mix_hot.append(
                (
                    (1 - theta) * (t_hot_tank_curr - t_hot_tank_prev)
                    + theta
                    * (
                        (t_hot_tank_curr * v_hot_tank_curr - t_hot_tank_prev * v_hot_tank_prev) / dt
                        - q_hot_pipe_curr * t_hot_pipe_curr
                        + (math.pi * radius ** 2 + 2 / radius * v_hot_tank_curr)
                        * heat_transfer_coeff
                        * (t_hot_tank_prev - temp_outside)
                        / (rho * cp)
                    )
                )[hot_mix_inds]
            )

            t_mix_hot.append(
                (
                    (
                        (1 - theta) * (t_hot_tank_curr - t_hot_tank_prev)
                        + theta
                        * (
                            (t_hot_tank_curr - t_hot_tank_prev) / dt
                            + avg_surface
                            / (volume / 2)
                            * heat_transfer_coeff
                            * (t_hot_tank_prev - temp_outside)
                            / (rho * cp)
                        )
                    )[cold_mix_inds + inactive_inds]
                )
            )

            # Cold tank
            t_mix_cold.append(
                (
                    (1 - theta) * (t_cold_tank_curr - t_cold_tank_prev)
                    + theta
                    * (
                        (t_cold_tank_curr * v_cold_tank_curr - t_cold_tank_prev * v_cold_tank_prev)
                        / dt
                        - q_cold_pipe_curr * t_cold_pipe_curr
                        + (math.pi * radius ** 2 + 2 / radius * v_cold_tank_curr)
                        * heat_transfer_coeff
                        * (t_cold_tank_curr - temp_outside)
                        / (rho * cp)
                    )
                )[cold_mix_inds]
            )

            t_mix_cold.append(
                (
                    (
                        (1 - theta) * (t_cold_tank_curr - t_cold_tank_prev)
                        + theta
                        * (
                            (t_cold_tank_curr - t_cold_tank_prev) / dt
                            + avg_surface
                            / (volume / 2)
                            * heat_transfer_coeff
                            * (t_cold_tank_prev - temp_outside)
                            / (rho * cp)
                        )
                    )[hot_mix_inds + inactive_inds]
                )
            )

            # Can't vertcat 1x0 symbols that would occur when running with only
            # two time steps (due to the slicing above). Remove them first.
            t_mix_hot = [x for x in t_mix_hot if x.nnz() > 0]
            t_mix_cold = [x for x in t_mix_cold if x.nnz() > 0]

            constraints.append((ca.vertcat(*t_mix_hot), 0.0, 0.0))
            constraints.append((ca.vertcat(*t_mix_cold), 0.0, 0.0))

        return constraints

    def __max_temp_rate_of_change_constraints(self, ensemble_member):
        constraints = []

        options = self.heat_network_options()
        components = self.heat_network_components

        parameters = self.parameters(ensemble_member)
        theta = parameters[self.homotopy_options()["homotopy_parameter"]]

        # Maximum allowed rate of change of temperature in pipes
        maximum_temperature_der = options["maximum_temperature_der"]

        if (
            maximum_temperature_der is not None
            and isfinite(maximum_temperature_der)
            and theta > 0.0
        ):
            # Temperature rate of change constraints are relevant only for the nonlinear
            # model. Impose the rate of change constraints on the out temperature of the
            # sources and on the pipes with bidirectional flow.
            # NOTE: We impose them here (not as path_constraints), because we want to skip
            # the initial derivative.
            bounds = self.bounds()
            times = self.times()
            dt = np.diff(times)
            avg_dt = np.mean(np.diff(times))  # For scaling purposes

            # Note that maximum temperature change is expressed in °C per
            # hour. RTC-Tools uses seconds, so we have to scale accordingly.
            max_der_sec = maximum_temperature_der / 3600.0

            for s in components["source"]:
                # NOTE: Imposing the constraint on the outflow of the sources does mean that
                # the temperature change for cold pipes might be higher than the target limit.
                # Because the temperature on the cold side is generally less in our control
                # anyway, we assume this is OK.
                variable = f"{s}.QTHOut.T"
                np.testing.assert_array_equal(self.times(variable), times)

                source_temperature_out = self.__state_vector_scaled(variable, ensemble_member)
                interp_mode = self.interpolation_method(variable)

                # Get the bounds. We want to avoid setting a constraint on times/variables
                # where the lower bound and upper bound are equal, as that would create empty,
                # trivial constraints and thus break the linear independence requirement
                lb, ub = bounds[variable]
                if isinstance(lb, Timeseries):
                    lb_values = self.interpolate(times[1:], lb.times, lb.values, mode=interp_mode)
                else:
                    assert np.isscalar(lb)
                    lb_values = np.full(len(times[1:]), lb)
                if isinstance(ub, Timeseries):
                    ub_values = self.interpolate(times[1:], ub.times, ub.values, mode=interp_mode)
                else:
                    assert np.isscalar(ub)
                    ub_values = np.full(len(times[1:]), ub)
                inds = np.flatnonzero(lb_values != ub_values)

                if len(inds) > 0:
                    # Make the derivative constraints for the time steps we want.
                    var_cur = source_temperature_out[1:][inds]
                    var_prev = source_temperature_out[:-1][inds]
                    roc = (var_cur - var_prev) / dt[inds]
                    constraints.append((roc * avg_dt, -max_der_sec * avg_dt, max_der_sec * avg_dt))

            if options["max_t_der_bidirect_pipe"]:
                # Applies only to the pipes with bidirectional flow
                flow_directions = self.__get_interpolated_flow_directions(ensemble_member)
                for p in components["pipe"]:
                    d_val = flow_directions[p]
                    if not np.all(d_val == d_val[0]):
                        for in_out in ["In", "Out"]:
                            variable = f"{p}.QTH{in_out}.T"
                            np.testing.assert_array_equal(self.times(variable), times)

                            temperature = self.__state_vector_scaled(variable, ensemble_member)
                            var_cur = temperature[1:]
                            var_prev = temperature[:-1]
                            roc = (var_cur - var_prev) / dt

                            constraints.append(
                                (roc * avg_dt, -max_der_sec * avg_dt, max_der_sec * avg_dt)
                            )

        return constraints

    def _hn_get_pipe_head_loss_option(self, pipe, heat_network_options, parameters):
        """
        The global user head loss option is not necessarily the same as the
        head loss option for a specific pipe. For example, when a control
        valve is present, a .LINEAR global head loss option would mean a
        .CQ2_INEQUALITY formulation should be used instead.

        See also the explanation of `head_loss_option` (and its values) in
        :py:meth:`.heat_network_options`.
        """

        head_loss_option = heat_network_options["head_loss_option"]

        if head_loss_option == HeadLossOption.LINEAR and parameters[f"{pipe}.has_control_valve"]:
            # If there is a control valve present, we use the more accurate
            # C*Q^2 inequality formulation.
            head_loss_option = HeadLossOption.CQ2_INEQUALITY
        elif head_loss_option == HeadLossOption.CQ2_EQUALITY:
            theta = parameters[self.homotopy_options()["homotopy_parameter"]]

            if parameters[f"{pipe}.has_control_valve"]:
                # An equality would be wrong when there is a control valve
                # present, so use the inequality formulation
                head_loss_option = HeadLossOption.CQ2_INEQUALITY
            elif np.isnan(theta):
                # In pre()
                pass
            elif theta < 1.0:
                # Not fully non-linear yet, so use the linear formulation instead
                head_loss_option = HeadLossOption.LINEAR

        return head_loss_option

    def _hn_pipe_head_loss_constraints(self, ensemble_member):
        constraints = []

        options = self.heat_network_options()
        parameters = self.parameters(ensemble_member)
        components = self.heat_network_components

        # Set the head loss according to the direction in the pipes.
        interpolated_flow_dir_values = self.__get_interpolated_flow_directions(ensemble_member)

        # Set the head loss according to the direction in the pipes. Note that
        # the `.__head_loss` symbol is always positive by definition, but that
        # `.dH` is not (positive when flow is negative, and vice versa).
        # If the flow direction is zero (i.e. pipe is disconnected), we leave
        # the .__head_loss symbol free (and it has no physical meaning). We
        # also do not set any discharge relationship in this case (but dH is
        # still equal to Out - In of course).
        for pipe in components["pipe"]:
            dir_values = interpolated_flow_dir_values[pipe]
            inds = np.flatnonzero(dir_values != PipeFlowDirection.DISABLED)

            if len(inds) == 0:
                continue

            head_loss_sym = self._hn_pipe_to_head_loss_map[pipe]

            dh = self.__state_vector_scaled(f"{pipe}.dH", ensemble_member)[inds]
            head_loss = self.__state_vector_scaled(head_loss_sym, ensemble_member)[inds]
            discharge = self.__state_vector_scaled(f"{pipe}.Q", ensemble_member)[inds]

            constraints.extend(
                self._hn_pipe_head_loss(pipe, options, parameters, discharge, head_loss, dh)
            )

            # Relate the absolute value of the head loss to the dH symbol (if
            # pipe is not disconnected).
            constraint_nominal = self.variable_nominal(f"{pipe}.dH")
            constraints.append(((dir_values[inds] * dh + head_loss) / constraint_nominal, 0.0, 0.0))

        return constraints

    def constraints(self, ensemble_member):
        constraints = super().constraints(ensemble_member)
        constraints.extend(self.__pipe_heat_loss_constraints(ensemble_member))
        constraints.extend(self.__node_mixing_constraints(ensemble_member))
        constraints.extend(self.__buffer_constraints(ensemble_member))
        constraints.extend(self.__max_temp_rate_of_change_constraints(ensemble_member))
        return constraints

    def path_constraints(self, ensemble_member):
        constraints = super().path_constraints(ensemble_member).copy()

        components = self.heat_network_components
        parameters = self.parameters(ensemble_member)
        theta = parameters[self.homotopy_options()["homotopy_parameter"]]

        if theta > 0.0:
            # Fix dT at demand nodes otherwise
            # Note that for theta == 0.0, this is trivially satisfied as the temperature
            # of the cold/hot line are constant.
            for d in components["demand"]:
                dt = parameters[d + ".T_supply"] - parameters[d + ".T_return"]
                constraints.append(
                    (self.state(d + ".QTHIn.T") - self.state(d + ".QTHOut.T"), dt, dt)
                )

        return constraints

    def priority_started(self, priority):
        super().priority_started(priority)
        self.__priority = priority

    def __get_interpolated_flow_directions(self, ensemble_member) -> Dict[str, np.ndarray]:
        """
        Interpolates the flow directions of all pipes to the collocation
        times. Returns a dictionary that maps from pipe name to NumPy array of
        direction values (dtype: PipeFlowDirection)
        """
        times = self.times()
        constant_inputs = self.constant_inputs(ensemble_member)

        flow_dirs = self.heat_network_pipe_flow_directions

        interpolated_flow_dir_values = {}

        for p in self.heat_network_components["pipe"]:
            try:
                direction_ts = constant_inputs[flow_dirs[p]]
            except KeyError:
                raise KeyError(
                    f"Could not find the direction of pipe {p} for ensemble member "
                    f"{ensemble_member}. Please extend or override the "
                    f"`heat_network_pipe_flow_directions` method. Note that this information "
                    f"is necessary before calling `super().pre()`, and cannot change afterwards."
                )

            interpolated_flow_dir_values[p] = self.interpolate(
                times,
                direction_ts.times,
                direction_ts.values,
                self.INTERPOLATION_PIECEWISE_CONSTANT_BACKWARD,
            )

        return interpolated_flow_dir_values

    def __update_flow_direction_bounds(self, ensemble_member):
        times = self.times()
        bounds = self.bounds()
        parameters = self.parameters(ensemble_member)
        options = self.heat_network_options()

        direction_bounds = {}
        interpolated_flow_dir_values = self.__get_interpolated_flow_directions(ensemble_member)

        minimum_velocity = options["minimum_velocity"]

        for p in self.heat_network_components["pipe"]:
            dir_values = interpolated_flow_dir_values[p]

            if isfinite(minimum_velocity):
                diameter = parameters[f"{p}.diameter"]
                area = 0.25 * math.pi * diameter ** 2
                minimum_discharge = minimum_velocity * area
            else:
                minimum_discharge = 0.0

            lb = np.where(dir_values == PipeFlowDirection.NEGATIVE, -np.inf, minimum_discharge)
            ub = np.where(dir_values == PipeFlowDirection.POSITIVE, np.inf, -1 * minimum_discharge)
            b = self.merge_bounds(bounds[f"{p}.Q"], (Timeseries(times, lb), Timeseries(times, ub)))
            # Pipes' bounds can take both negative and positive values.
            # To force bounds to be zero, they need to be explicitely overwritten.
            b[0].values[(dir_values == PipeFlowDirection.DISABLED)] = 0.0
            b[1].values[(dir_values == PipeFlowDirection.DISABLED)] = 0.0

            direction_bounds[f"{p}.Q"] = b

            # Head loss
            lb = np.where(dir_values == PipeFlowDirection.NEGATIVE, 0.0, -np.inf)
            ub = np.where(dir_values == PipeFlowDirection.POSITIVE, 0.0, np.inf)
            b = self.merge_bounds(bounds[f"{p}.dH"], (Timeseries(times, lb), Timeseries(times, ub)))

            direction_bounds[f"{p}.dH"] = b

        if self.__flow_direction_bounds is None:
            self.__flow_direction_bounds = [{} for _ in range(self.ensemble_size)]

        self.__flow_direction_bounds[ensemble_member] = direction_bounds.copy()

    def __update_temperature_pipe_theta_zero_bounds(self):
        # At theta=0, the temperature of the pipes must be equal to its design temperature.
        # Here we create a dictionary which will be used to update the bounds.
        # Note that the design temperature is not ensemble dependent.
        parameters = self.parameters(0)
        temperature_bounds = {}

        for p in self.heat_network_components["pipe"]:
            temperature = parameters[f"{p}.temperature"]
            b = (temperature, temperature)

            temperature_bounds[f"{p}.QTHIn.T"] = b
            temperature_bounds[f"{p}.QTHOut.T"] = b

        self.__temperature_pipe_theta_zero = temperature_bounds.copy()

    def bounds(self):
        bounds = super().bounds().copy()

        parameters = self.parameters(0)
        theta = parameters[self.homotopy_options()["homotopy_parameter"]]

        if self.__flow_direction_bounds is not None:
            # TODO: Per ensemble member
            bounds.update(self.__flow_direction_bounds[0])

        # Set the temperature of the pipes in the linear problem
        if theta == 0.0:
            bounds.update(self.__temperature_pipe_theta_zero)

        return bounds

    def __start_transcribe_checks(self):
        ens_interpolated_flow_dir_values = [
            self.__get_interpolated_flow_directions(e) for e in range(self.ensemble_size)
        ]

        for p in self.heat_network_components["pipe"]:
            cur_pipe_flow_dir_values = [
                ens_interpolated_flow_dir_values[e][p] for e in range(self.ensemble_size)
            ]
            if not np.array_equal(
                np.amin(cur_pipe_flow_dir_values, 0), np.amax(cur_pipe_flow_dir_values, 0)
            ):
                raise Exception(
                    f"Pipe direction of pipe '{p}' differs based on ensemble member. "
                    f"This is not properly supported yet."
                )

    def transcribe(self):
        self.__start_transcribe_checks()

        # Update flow direction bounds
        for e in range(self.ensemble_size):
            self.__update_flow_direction_bounds(e)

        discrete, lbx, ubx, lbg, ubg, x0, nlp = super().transcribe()

        if self.__priority == self._hn_minimization_goal_class.priority:
            # We overrule here instead of in bounds(), because bounds() does
            # not support per-ensemble-member bounds. The collocation indices
            # are private for now, but will become part of the public API soon.

            parameters = self.parameters(0)
            theta = parameters[self.homotopy_options()["homotopy_parameter"]]
            assert (
                theta == 1.0
            )  # Minimization should be skipped (via `Goal.is_empty`) if theta < 1.0

            lb = np.full_like(lbx, -np.inf)
            ub = np.full_like(ubx, np.inf)

            fix_value_variables = set()
            for p in self.heat_network_components["pipe"]:
                fix_value_variables.add(f"{p}.QTHIn.Q")
            fix_value_variables = {
                self.alias_relation.canonical_signed(v)[0] for v in fix_value_variables
            }

            output = self.solver_output

            previous_indices = self.__previous_indices
            current_indices = self._CollocatedIntegratedOptimizationProblem__indices_as_lists

            for ensemble_member in range(self.ensemble_size):
                for v in fix_value_variables:
                    cur_inds = current_indices[ensemble_member][v]
                    prev_inds = previous_indices[ensemble_member][v]
                    ub[cur_inds] = lb[cur_inds] = output[prev_inds]

            lbx = np.maximum(lbx, lb)
            ubx = np.minimum(ubx, ub)

            # Sometimes rounding errors can change the ulp, leading to lbx
            # becomes slightly (~1E-15) larger than ubx. Fix by setting
            # equality entries explicitly.
            inds = lb == ub
            lbx[inds] = ubx[inds] = lb[inds]

            assert np.all(lbx <= ubx)

        self.__previous_indices = self._CollocatedIntegratedOptimizationProblem__indices_as_lists

        return discrete, lbx, ubx, lbg, ubg, x0, nlp

    def solver_options(self):
        options = super().solver_options()

        solver = options["solver"]
        options[solver]["nlp_scaling_method"] = "none"
        options[solver]["linear_system_scaling"] = "none"
        options[solver]["linear_scaling_on_demand"] = "no"

        return options

    def homotopy_options(self):
        options = super().homotopy_options()
        options["delta_theta_min"] = 1.0
        return options
