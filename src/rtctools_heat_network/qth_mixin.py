import logging
import math
from enum import IntEnum
from math import isfinite
from typing import Dict, Optional, Union

import casadi as ca

import numpy as np

from rtctools.optimization.collocated_integrated_optimization_problem import (
    CollocatedIntegratedOptimizationProblem,
)
from rtctools.optimization.goal_programming_mixin_base import Goal
from rtctools.optimization.homotopy_mixin import HomotopyMixin
from rtctools.optimization.timeseries import Timeseries

import rtctools_heat_network._darcy_weisbach as darcy_weisbach
from rtctools_heat_network._heat_loss_u_values_pipe import heat_loss_u_values_pipe

from .base_component_type_mixin import BaseComponentTypeMixin
from .constants import GRAVITATIONAL_CONSTANT
from .heat_network_common import NodeConnectionDirection, PipeFlowDirection


logger = logging.getLogger("rtctools_heat_network")


class HeadLossOption(IntEnum):
    """
    Enumeration for the possible options to take head loss in pipes into
    account.
    """

    NO_HEADLOSS = 1
    CQ2_INEQUALITY = 2
    LINEARIZED_DW = 3
    LINEAR = 4
    CQ2_EQUALITY = 5


class _MinimizeHeadLosses(Goal):
    order = 1

    priority = 2 ** 31 - 1

    def __init__(self, optimization_problem, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimization_problem = optimization_problem
        self.function_nominal = len(optimization_problem.times())

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

    def function(self, optimization_problem, ensemble_member):
        sum_ = 0.0

        parameters = self.optimization_problem.parameters(ensemble_member)

        pumps = optimization_problem.heat_network_components.get("pump", [])

        for p in pumps:
            sum_ += optimization_problem.state(f"{p}.dH")

        for p in optimization_problem.heat_network_components["pipe"]:
            if not parameters[f"{p}.has_control_valve"]:
                sum_ += -1 * optimization_problem.state(f"{p}.dH")

        return sum_


class QTHMixin(BaseComponentTypeMixin, CollocatedIntegratedOptimizationProblem):
    """
    Adds handling of QTH heat network objects in your model to your
    optimization problem.

    Relevant parameters and variables are read from the model, and from this
    data a set of constraints and objectives are automatically generated, e.g.
    for the head loss and temperature losses in pipes.
    """

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

        self.__update_temperature_pipe_theta_zero_bounds()

        super().pre()

    def heat_network_options(self):
        r"""
        Returns a dictionary of heat network specific options.

        +--------------------------------+-----------+-----------------------------------+
        | Option                         | Type      | Default value                     |
        +================================+===========+===================================+
        | ``minimum_pressure_far_point`` | ``float`` | ``1.0`` bar                       |
        +--------------------------------+-----------+-----------------------------------+
        | ``dtemp_demand``               | ``float`` | ``30`` °C                         |
        +--------------------------------+-----------+-----------------------------------+
        | ``maximum_temperature_der``    | ``float`` | ``2.0`` °C/hour                   |
        +--------------------------------+-----------+-----------------------------------+
        | ``max_t_der_bidirect_pipe``    | ``bool``  | ``True``                          |
        +--------------------------------+-----------+-----------------------------------+
        | ``minimum_velocity``           | ``float`` | ``0.005`` m/s                     |
        +--------------------------------+-----------+-----------------------------------+
        | ``wall_roughness``             | ``float`` | ``0.002`` m                       |
        +--------------------------------+-----------+-----------------------------------+
        | ``head_loss_option``           | ``enum``  | ``HeadLossOption.CQ2_INEQUALITY`` |
        +--------------------------------+-----------+-----------------------------------+
        | ``estimated_velocity``         | ``float`` | ``1.0`` m/s (CQ2_* & LINEAR)      |
        +--------------------------------+-----------+-----------------------------------+
        | ``maximum_velocity``           | ``float`` | ``2.5`` m/s (LINEARIZED_DW)       |
        +--------------------------------+-----------+-----------------------------------+
        | ``n_linearization_lines``      | ``int``   | ``10`` (LINEARIZED_DW)            |
        +--------------------------------+-----------+-----------------------------------+
        | ``minimize_head_losses``       | ``bool``  | ``True``                          |
        +--------------------------------+-----------+-----------------------------------+

        The ``minimum_pressure_far_point`` gives the minimum pressure
        requirement at any demand node, which means that the pressure at the
        furthest point is also satisfied without inspecting the topology.

        The ``dtemp_demand`` parameter specifies what the (fixed) temperature
        drop is  over the demand nodes.

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

        The ``wall_roughness`` of the pipes plays a role in determining the
        resistance of the pipes.

        To model the head loss in pipes, the ``head_lost_option`` refers to
        one of the ways this can be done. See :class:`HeadLossOption` for more
        explanation on what each option entails. Note that some options model
        the head loss as an inequality, i.e. :math:`\Delta H \ge f(Q)`, whereas
        others model it as an equality.

        When ``HeadLossOption.CQ2_INEQUALITY`` is used, the wall roughness at
        ``estimated_velocity`` determines the `C` in :math:`\Delta H \ge C
        \cdot Q^2`.

        When ``HeadLossOption.LINEARIZED_DW`` is used, the
        ``maximum_velocity`` needs to be set. The Darcy-Weisbach head loss
        relationship from :math:`v = 0` until :math:`v = maximum_velocity`
        will then be linearized using ``n_linearization`` lines.

        When ``HeadLossOption.LINEAR`` is used, the wall roughness at
        ``estimated_velocity`` determines the `C` in :math:`\Delta H = C \cdot
        Q`. For pipes that contain a control valve, the formulation of
        ``HeadLossOption.CQ2_INEQUALITY`` is used.

        When ``HeadLossOption.CQ2_EQUALITY`` is used, the wall roughness at
        ``estimated_velocity`` determines the `C` in :math:`\Delta H = C \cdot
        Q^2`. Note that this formulation is non-convex. At `theta < 1` we
        therefore use the formulation ``HeadLossOption.LINEAR``. For pipes
        that contain a control valve, the formulation of
        ``HeadLossOption.CQ2_INEQUALITY`` is used.

        When ``minimize_head_losses`` is set to True (default), a last
        priority is inserted where the head losses in the system are
        minimized. This is related to the assumption that control valves are
        present in the system to steer water in the right direction the case
        of multiple routes. If such control valves are not present, enabling
        this option will give warnings in case the found solution is not
        feasible. In case the option is False, both the minimization and
        checks are skipped.
        """

        options = {}

        options["minimum_pressure_far_point"] = 1.0
        options["dtemp_demand"] = 30
        options["maximum_temperature_der"] = 2.0
        options["max_t_der_bidirect_pipe"] = True
        options["minimum_velocity"] = 0.005
        options["wall_roughness"] = 2e-3
        options["head_loss_option"] = HeadLossOption.CQ2_INEQUALITY
        options["estimated_velocity"] = 1.0
        options["maximum_velocity"] = 2.5
        options["n_linearization_lines"] = 10
        options["minimize_head_losses"] = True

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

            constraints.append((hot_pipe_orientation * q_in - q_hot_pipe, 0.0, 0.0))
            constraints.append((cold_pipe_orientation * q_out + q_cold_pipe, 0.0, 0.0))

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

    def __pipe_head_loss(
        self,
        pipe: str,
        heat_network_options,
        parameters,
        discharge: Union[ca.MX, np.ndarray],
        head_loss: Optional[ca.MX],
    ):
        """
        This function has two purposes:
        - return the head loss constraint expression(s) or
        - compute the head loss numerically.

        Note that there are different head loss formulations (see
        :class:`HeadLossOption`), that are possible. When `head_loss` is its
        corresponding MX symbol/expression, the appropriate constraint expression is
        returned. When `head_loss` is None, the `discharge` is assumed numerical, and
        the numerical computation of the appropriate heat loss formulation is
        returned.
        """

        if head_loss is None:
            symbolic = False
        else:
            symbolic = True

        head_loss_option = heat_network_options["head_loss_option"]

        if head_loss_option == HeadLossOption.LINEAR and parameters[f"{pipe}.has_control_valve"]:
            # If there is a control valve present, we use the more accurate
            # C*Q^2 inequality formulation.
            head_loss_option = HeadLossOption.CQ2_INEQUALITY
        elif head_loss_option == HeadLossOption.CQ2_EQUALITY:
            if parameters[f"{pipe}.has_control_valve"]:
                # An equality would be wrong when there is a control valve
                # present, so use the inequality formulation
                head_loss_option = HeadLossOption.CQ2_INEQUALITY
            elif parameters[self.homotopy_options()["homotopy_parameter"]] < 1.0:
                # Not fully non-linear yet, so use the linear formulation instead
                head_loss_option = HeadLossOption.LINEAR

        # Apply head loss constraints in pipes depending on the option set by
        # the user.
        if head_loss_option == HeadLossOption.NO_HEADLOSS:
            # No constraints to be set. Note that we will set the bounds of
            # all the dH symbols to zero.
            if symbolic:
                return []
            else:
                return 0.0

        elif head_loss_option in {
            HeadLossOption.CQ2_INEQUALITY,
            HeadLossOption.LINEAR,
            HeadLossOption.CQ2_EQUALITY,
        }:
            estimated_velocity = heat_network_options["estimated_velocity"]
            wall_roughness = heat_network_options["wall_roughness"]

            diameter = parameters[f"{pipe}.diameter"]
            length = parameters[f"{pipe}.length"]
            temperature = parameters[f"{pipe}.temperature"]
            has_control_valve = parameters[f"{pipe}.has_control_valve"]

            ff = darcy_weisbach.friction_factor(
                estimated_velocity, diameter, length, wall_roughness, temperature
            )

            # Compute c_v constant (where |dH| ~ c_v * v^2)
            c_v = length * ff / (2 * GRAVITATIONAL_CONSTANT) / diameter
            area = 0.25 * math.pi * diameter ** 2

            v = discharge / area

            if head_loss_option == HeadLossOption.CQ2_INEQUALITY:
                expr = c_v * v ** 2
                ub = np.inf
            elif head_loss_option == HeadLossOption.CQ2_EQUALITY:
                expr = c_v * v ** 2
                ub = np.inf if has_control_valve else 0.0
            else:
                if not symbolic:
                    # We are supposed to only return a positive head loss,
                    # regardless of the sign of the discharge.
                    v = np.abs(v)

                assert head_loss_option == HeadLossOption.LINEAR
                expr = c_v * v
                ub = np.inf if has_control_valve else 0.0

            if symbolic:
                return [(head_loss - expr, 0.0, ub)]
            else:
                return expr

        elif head_loss_option == HeadLossOption.LINEARIZED_DW:
            wall_roughness = heat_network_options["wall_roughness"]
            v_max = heat_network_options["maximum_velocity"]
            n_lines = heat_network_options["n_linearization_lines"]

            diameter = parameters[f"{pipe}.diameter"]
            area = math.pi * diameter ** 2 / 4
            length = parameters[f"{pipe}.length"]
            temperature = parameters[f"{pipe}.temperature"]

            a, b = darcy_weisbach.get_linear_pipe_dh_vs_q_fit(
                diameter,
                length,
                wall_roughness,
                temperature=temperature,
                n_lines=n_lines,
                v_max=v_max,
            )

            # The function above only gives result in the positive quadrant
            # (positive head loss, positive discharge). We also need a
            # positive head loss for _negative_ discharges.
            a = np.hstack([-a, a])
            b = np.hstack([b, b])

            # Vectorize constraint for speed
            if symbolic:
                head_loss_vec = ca.repmat(head_loss, len(a))
                discharge_vec = ca.repmat(discharge, len(a))
                return [(head_loss_vec - (a * discharge_vec + b), 0.0, np.inf)]
            else:
                return np.amax(a * np.tile(discharge, (len(a), 1)).transpose() + b, axis=1)

    def __pipe_head_loss_path_constraints(self, ensemble_member):
        constraints = []

        parameters = self.parameters(ensemble_member)
        components = self.heat_network_components

        # Check if head_loss_option is correct
        options = self.heat_network_options()
        head_loss_option = options["head_loss_option"]

        if head_loss_option not in HeadLossOption.__members__.values():
            raise Exception(f"Head loss option '{head_loss_option}' does not exist")

        if head_loss_option == HeadLossOption.NO_HEADLOSS:
            # No relationships between H and dh, Q are set. H values are completely free.
            return []

        # Set the head loss according to the direction in the pipes
        flow_dirs = self.heat_network_pipe_flow_directions

        for pipe in components["pipe"]:
            if head_loss_option == HeadLossOption.LINEAR:
                # When dealing with a linear head loss relationship, we do not
                # need to have a head symbol that is always a certain sign
                # regardless of flow direction (e.g. the pipe's dH symbol will
                # always be negative). A need for such a symbol only arises
                # when dealing with inequality constraints or if the discharge
                # symbol is squared.
                head_loss = self.state(f"{pipe}.QTHIn.H") - self.state(f"{pipe}.QTHOut.H")
            else:
                # Note that the dH (always negative for pipes) is the opposite
                # sign of the head loss (which is positive). See also the
                # definition/constraints of a pipe's dH symbol.
                head_loss = -1 * self.state(f"{pipe}.dH")

            constraints.extend(
                self.__pipe_head_loss(
                    pipe,
                    options,
                    parameters,
                    self.state(f"{pipe}.Q"),
                    head_loss,
                )
            )

        for pipe in components["pipe"]:
            dh = self.state(f"{pipe}.dH")
            flow_dir = self.variable(flow_dirs[pipe])
            h_down = self.state(f"{pipe}.QTHOut.H")
            h_up = self.state(f"{pipe}.QTHIn.H")

            constraints.append((dh - flow_dir * (h_down - h_up), 0.0, 0.0))

        return constraints

    def __source_head_loss_path_constraints(self, ensemble_member):
        constraints = []

        parameters = self.parameters(ensemble_member)
        components = self.heat_network_components

        for source in components["source"]:
            c = parameters[f"{source}.head_loss"]

            if c == 0.0:
                constraints.append(
                    (self.state(f"{source}.QTHIn.H") - self.state(f"{source}.QTHOut.H"), 0.0, 0.0)
                )
            else:
                constraints.append(
                    (
                        self.state(f"{source}.QTHIn.H")
                        - self.state(f"{source}.QTHOut.H")
                        - c * self.state(f"{source}.QTHIn.Q") ** 2,
                        0.0,
                        np.inf,
                    )
                )

        return constraints

    def __demand_head_loss_path_constraints(self, ensemble_member):
        constraints = []

        options = self.heat_network_options()
        components = self.heat_network_components

        # Convert minimum pressure at far point from bar to meter (water) head
        min_head_loss = options["minimum_pressure_far_point"] * 10.2

        for d in components["demand"]:
            constraints.append(
                (self.state(d + ".QTHIn.H") - self.state(d + ".QTHOut.H"), min_head_loss, np.inf)
            )

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

        options = self.heat_network_options()
        components = self.heat_network_components
        parameters = self.parameters(ensemble_member)
        theta = parameters[self.homotopy_options()["homotopy_parameter"]]

        # Head (loss) constraints
        constraints.extend(self.__pipe_head_loss_path_constraints(ensemble_member))
        # Add source/demand head loss constrains only if head loss is non-zero
        if options["head_loss_option"] != HeadLossOption.NO_HEADLOSS:
            constraints.extend(self.__source_head_loss_path_constraints(ensemble_member))
            constraints.extend(self.__demand_head_loss_path_constraints(ensemble_member))

        if theta > 0.0:
            # Fix dT at demand nodes otherwise
            # Note that for theta == 0.0, this is trivially satisfied as the temperature
            # of the cold/hot line are constant.
            dtemp = options["dtemp_demand"]
            for d in components["demand"]:
                constraints.append(
                    (self.state(d + ".QTHIn.T") - self.state(d + ".QTHOut.T"), dtemp, dtemp)
                )

        return constraints

    def priority_started(self, priority):
        super().priority_started(priority)
        self.__priority = priority

    def priority_completed(self, priority):
        super().priority_completed(priority)

        options = self.heat_network_options()

        if options["minimize_head_losses"] and priority == _MinimizeHeadLosses.priority:
            parameters = self.parameters(0)
            theta = parameters[self.homotopy_options()["homotopy_parameter"]]
            assert (
                theta == 1.0
            )  # Minimization should be skipped (via `Goal.is_empty`) if theta < 1.0

            components = self.heat_network_components

            rtol = 1e-5
            atol = 1e-4

            for ensemble_member in range(self.ensemble_size):
                parameters = self.parameters(ensemble_member)
                results = self.extract_results(ensemble_member)

                for pipe in components["pipe"]:
                    if parameters[f"{pipe}.has_control_valve"]:
                        continue
                    q = results[f"{pipe}.Q"]
                    head_loss = -1 * results[f"{pipe}.dH"]
                    head_loss_target = self.__pipe_head_loss(pipe, options, parameters, q, None)

                    if not np.allclose(head_loss, head_loss_target, rtol=rtol, atol=atol):
                        logger.warning(
                            f"Pipe {pipe} has artificial head loss; "
                            f"at least one more control valve should be added to the network."
                        )

                for source in components["source"]:
                    c = parameters[f"{source}.head_loss"]
                    head_loss = results[f"{source}.QTHIn.H"] - results[f"{source}.QTHOut.H"]
                    head_loss_target = c * results[f"{source}.QTHIn.Q"] ** 2

                    if not np.allclose(head_loss, head_loss_target, rtol=rtol, atol=atol):
                        logger.warning(f"Source {source} has artificial head loss.")

                min_head_loss_target = options["minimum_pressure_far_point"] * 10.2
                min_head_loss = None

                for demand in components["demand"]:
                    head_loss = results[f"{demand}.QTHIn.H"] - results[f"{demand}.QTHOut.H"]
                    if min_head_loss is None:
                        min_head_loss = head_loss
                    else:
                        min_head_loss = np.minimum(min_head_loss, head_loss)

                if not np.allclose(min_head_loss, min_head_loss_target, rtol=rtol, atol=atol):
                    logger.warning("Minimum head at demands is higher than target minimum.")

    def path_goals(self):
        g = super().path_goals().copy()

        options = self.heat_network_options()
        if options["minimize_head_losses"]:
            g.append(_MinimizeHeadLosses(self))

        return g

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

        min_abs_velocity = abs(options["minimum_velocity"])

        for p in self.heat_network_components["pipe"]:
            dir_values = interpolated_flow_dir_values[p]

            if isfinite(min_abs_velocity):
                diameter = parameters[f"{p}.diameter"]
                area = 0.25 * math.pi * diameter ** 2
                min_abs_discharge = min_abs_velocity * area
            else:
                min_abs_discharge = 0.0

            lb = np.where(dir_values == PipeFlowDirection.NEGATIVE, -np.inf, min_abs_discharge)
            ub = np.where(dir_values == PipeFlowDirection.POSITIVE, np.inf, -1 * min_abs_discharge)
            b = self.merge_bounds(bounds[f"{p}.Q"], (Timeseries(times, lb), Timeseries(times, ub)))
            # Pipes' bounds can take both negative and positive values.
            # To force bounds to be zero, they need to be explicitely overwritten.
            b[0].values[(dir_values == PipeFlowDirection.DISABLED)] = 0.0
            b[1].values[(dir_values == PipeFlowDirection.DISABLED)] = 0.0

            direction_bounds[f"{p}.Q"] = b

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

        options = self.heat_network_options()
        parameters = self.parameters(0)
        theta = parameters[self.homotopy_options()["homotopy_parameter"]]

        if self.__flow_direction_bounds is not None:
            # TODO: Per ensemble member
            bounds.update(self.__flow_direction_bounds[0])

        if options["head_loss_option"] == HeadLossOption.NO_HEADLOSS:
            for pipe in self.heat_network_components["pipe"]:
                bounds[f"{pipe}.dH"] = (0.0, 0.0)

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

        if self.__priority == _MinimizeHeadLosses.priority:
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

    def post(self):
        super().post()
