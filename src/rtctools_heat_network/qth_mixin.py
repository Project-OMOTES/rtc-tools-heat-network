import math
from enum import IntEnum
from math import isfinite
from typing import Dict

import casadi as ca

import numpy as np

from rtctools.optimization.collocated_integrated_optimization_problem import (
    CollocatedIntegratedOptimizationProblem,
)
from rtctools.optimization.homotopy_mixin import HomotopyMixin
from rtctools.optimization.timeseries import Timeseries

import rtctools_heat_network._darcy_weisbach as darcy_weisbach

from .base_component_type_mixin import BaseComponentTypeMixin
from .constants import GRAVITATIONAL_CONSTANT
from .heat_network_common import NodeConnectionDirection, PipeFlowDirection


class HeadLossOption(IntEnum):
    """
    Enumeration for the possible options to take head loss in pipes into
    account.
    """

    NO_HEADLOSS = 1
    CQ2 = 2
    LINEARIZED_DW = 3


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
        # Check that all pipes have a corresponding hot and cold version
        components = self.heat_network_components

        pipes = components["pipe"]
        hot_pipes_no_suf = {p[:-4] for p in pipes if p.endswith("_hot")}
        cold_pipes_no_suf = {p[:-5] for p in pipes if p.endswith("_cold")}

        if hot_pipes_no_suf != cold_pipes_no_suf:
            raise Exception("Every hot pipe should have a corresponding cold pipe and vice versa.")

        self.__flow_direction_bounds = None

        super().pre()

    def heat_network_options(self):
        r"""
        Returns a dictionary of heat network specific options.

        +--------------------------------+-----------+-----------------------------+
        | Option                         | Type      | Default value               |
        +================================+===========+========================-----+
        | ``minimum_pressure_far_point`` | ``float`` | ``1.0`` bar                 |
        +--------------------------------+-----------+-----------------------------+
        | ``dtemp_demand``               | ``float`` | ``30`` °C                   |
        +--------------------------------+-----------+-----------------------------+
        | ``maximum_temperature_der``    | ``float`` | ``2.0`` °C/hour             |
        +--------------------------------+-----------+-----------------------------+
        | ``max_t_der_bidirect_pipe``    | ``bool``  | ``True``                    |
        +--------------------------------+-----------+-----------------------------+
        | ``wall_roughness``             | ``float`` | ``0.002`` m                 |
        +--------------------------------+-----------+-----------------------------+
        | ``head_loss_option``           | ``enum``  | ``HeadLossOption.CQ2``      |
        +--------------------------------+-----------+-----------------------------+
        | ``estimated_velocity``         | ``float`` | ``1.0`` m/s (CQ2)           |
        +--------------------------------+-----------+-----------------------------+
        | ``maximum_velocity``           | ``float`` | ``2.0`` m/s (LINEARIZED_DW) |
        +--------------------------------+-----------+-----------------------------+
        | ``n_linearization_lines``      | ``int``   | ``10`` (LINEARIZED_DW)      |
        +--------------------------------+-----------+-----------------------------+

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

        The ``wall_roughness`` of the pipes plays a role in determining the
        resistance of the pipes.

        To model the head loss in pipes, the ``head_lost_option`` refers to
        one of the ways this can be done. See :class:`HeadLossOption` for more
        explanation on what each option entails. Note that all options model
        the head loss as an inequality, i.e. :math:`\Delta H \ge f(Q)`.

        When ``HeadLossOption.CQ2`` is used, the wall roughness at
        ``estimated_velocity`` determines the `C` in :math:`\Delta H \ge C
        \cdot Q^2`.

        When ``HeadLossOption.LINEARIZED_DW`` is used, the
        ``maximum_velocity`` needs to be set. The Darcy-Weisbach head loss
        relationship from :math:`v = 0` until :math:`v = maximum_velocity`
        will then be linearized using ``n_linearization`` lines.
        """

        options = {}

        options["minimum_pressure_far_point"] = 1.0
        options["dtemp_demand"] = 30
        options["maximum_temperature_der"] = 2.0
        options["max_t_der_bidirect_pipe"] = True
        options["wall_roughness"] = 2e-3
        options["head_loss_option"] = HeadLossOption.CQ2
        options["estimated_velocity"] = 1.0
        options["maximum_velocity"] = 2.0
        options["n_linearization_lines"] = 10

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

            # Temperature of outgoing flows is equal to mixing temperature
            constraints.append((ca.vertcat(*t_out_conn) / t_nominal, 0.0, 0.0))

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

    def __pipe_head_loss_path_constraints(self, ensemble_member):
        constraints = []

        parameters = self.parameters(ensemble_member)
        components = self.heat_network_components

        # Check if head_loss_option is correct
        options = self.heat_network_options()
        head_loss_option = options["head_loss_option"]

        if head_loss_option not in HeadLossOption.__members__.values():
            raise Exception(f"Head loss option '{head_loss_option}' does not exist")

        # Set the head loss according to the direction in the pipes
        flow_dirs = self.heat_network_pipe_flow_directions

        for pipe in components["pipe"]:
            dh = self.state(f"{pipe}.dH")
            flow_dir = self.variable(flow_dirs[pipe])
            h_down = self.state(f"{pipe}.QTHOut.H")
            h_up = self.state(f"{pipe}.QTHIn.H")

            constraints.append((dh - flow_dir * (h_down - h_up), 0.0, 0.0))

        # Apply head loss constraints in pipes depending on the option set by
        # the user.
        if head_loss_option == HeadLossOption.NO_HEADLOSS:
            for pipe in components["pipe"]:
                constraints.append((self.state(f"{pipe}.dH"), 0.0, 0.0))

        elif head_loss_option == HeadLossOption.CQ2:
            estimated_velocity = options["estimated_velocity"]
            wall_roughness = options["wall_roughness"]

            for pipe in components["pipe"]:
                diameter = parameters[f"{pipe}.diameter"]
                length = parameters[f"{pipe}.length"]
                temperature = parameters[f"{pipe}.temperature"]

                ff = darcy_weisbach.friction_factor(
                    estimated_velocity, diameter, length, wall_roughness, temperature
                )

                # Compute c_v constant (where |dH| ~ c_v * v^2)
                c_v = length * ff / (2 * GRAVITATIONAL_CONSTANT) / diameter
                area = 0.25 * math.pi * diameter ** 2

                v = self.state(f"{pipe}.Q") / area
                constraints.append((-self.state(f"{pipe}.dH") - c_v * v ** 2, 0.0, np.inf))

        elif head_loss_option == HeadLossOption.LINEARIZED_DW:
            wall_roughness = options["wall_roughness"]
            v_max = options["maximum_velocity"]
            n_lines = options["n_linearization_lines"]

            for pipe in components["pipe"]:
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

                # Vectorize constraint for speed
                dh = ca.repmat(self.state(f"{pipe}.dH"), len(a))
                q = ca.repmat(self.state(f"{pipe}.Q"), len(a))
                constraints.append((-dh - (a * q + b), 0.0, np.inf))

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
        constraints.extend(self.__node_mixing_constraints(ensemble_member))
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
        constraints.extend(self.__source_head_loss_path_constraints(ensemble_member))
        constraints.extend(self.__demand_head_loss_path_constraints(ensemble_member))

        if theta == 0.0:
            # Fix temperature in pipes for the fully linear model
            for pipe in components["pipe"]:
                constraints.append(
                    (self.state(f"{pipe}.QTHOut.T") - parameters[f"{pipe}.temperature"], 0.0, 0.0)
                )
        elif theta > 0.0:
            # Fix dT at demand nodes otherwise
            dtemp = options["dtemp_demand"]
            for d in components["demand"]:
                constraints.append(
                    (self.state(d + ".QTHIn.T") - self.state(d + ".QTHOut.T"), dtemp, dtemp)
                )

        return constraints

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

        direction_bounds = {}
        interpolated_flow_dir_values = self.__get_interpolated_flow_directions(ensemble_member)

        for p in self.heat_network_components["pipe"]:
            dir_values = interpolated_flow_dir_values[p]

            lb = np.where(dir_values == PipeFlowDirection.NEGATIVE, -np.inf, 0.0)
            ub = np.where(dir_values == PipeFlowDirection.POSITIVE, np.inf, 0.0)

            direction_bounds[f"{p}.Q"] = self.merge_bounds(
                bounds[f"{p}.Q"], (Timeseries(times, lb), Timeseries(times, ub))
            )

        if self.__flow_direction_bounds is None:
            self.__flow_direction_bounds = [{} for _ in range(self.ensemble_size)]

        self.__flow_direction_bounds[ensemble_member] = direction_bounds.copy()

    def bounds(self):
        bounds = super().bounds()

        if self.__flow_direction_bounds is not None:
            # TODO: Per ensemble member
            bounds.update(self.__flow_direction_bounds[0])

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

        return super().transcribe()
