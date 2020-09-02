import math
from abc import abstractproperty
from enum import IntEnum
from math import isfinite
from typing import Dict, Tuple

import casadi as ca

import numpy as np

from rtctools.optimization.collocated_integrated_optimization_problem import (
    CollocatedIntegratedOptimizationProblem,
)
from rtctools.optimization.homotopy_mixin import HomotopyMixin
from rtctools.optimization.timeseries import Timeseries

import rtctools_heat_network._darcy_weisbach as darcy_weisbach
from rtctools_heat_network.constants import GRAVITATIONAL_CONSTANT


class HeadLossOption(IntEnum):
    """
    Enumeration for the possible options to take head loss in pipes into
    account.
    """

    NO_HEADLOSS = 1
    CQ2 = 2
    LINEARIZED_DW = 3


class PipeFlowDirection(IntEnum):
    """
    Enumeration for the possible directions a pipe can have.
    """

    NEGATIVE = -1
    DISABLED = 0
    POSITIVE = 1


class NodeConnectionDirection(IntEnum):
    """
    Enumeration for the orientation of a pipe connected to a node, or of the
    flow into a node.
    """

    OUT = -1
    IN = 1


class Topology:
    def __init__(self, nodes=None, pipe_series=None):
        if nodes is not None:
            self._nodes = nodes
        if pipe_series is not None:
            self._pipe_series = pipe_series

    @property
    def nodes(self) -> Dict[str, Dict[int, Tuple[str, NodeConnectionDirection]]]:
        """
        Maps a node name to a dictionary of its connections. Written out using
        descriptive variable names the return type would be:
            Dict[node_name, Dict[connection_index, Tuple[connected_pipe_name, pipe_orientation]]]
        """
        try:
            return self._nodes
        except AttributeError:
            raise NotImplementedError

    @property
    def pipe_series(self):
        try:
            return self._pipe_series
        except AttributeError:
            raise NotImplementedError


class BaseComponentTypeMixin(CollocatedIntegratedOptimizationProblem):
    @abstractproperty
    def heat_network_components(self) -> Dict[str, str]:
        raise NotImplementedError

    @abstractproperty
    def heat_network_topology(self) -> Topology:
        raise NotImplementedError


class ModelicaComponentTypeMixin(BaseComponentTypeMixin):
    def pre(self):
        components = self.heat_network_components
        nodes = components["node"]
        pipes = components["pipe"]

        # Figure out which pipes are connected to which nodes, and which pipes
        # are connected in series.
        pipes_set = set(pipes)
        parameters = [self.parameters(e) for e in range(self.ensemble_size)]
        node_connections = {}

        for n in nodes:
            n_connections = [ens_params[f"{n}.n"] for ens_params in parameters]

            if len(set(n_connections)) > 1:
                raise Exception(
                    "Nodes cannot have differing number of connections per ensemble member"
                )

            n_connections = n_connections[0]

            # Note that we do this based on temperature, because discharge may
            # be an alias of yet some other further away connected pipe.
            node_connections[n] = connected_pipes = {}

            for i in range(n_connections):
                cur_port = f"{n}.QTHConn[{i + 1}]"
                aliases = [
                    x for x in self.alias_relation.aliases(f"{cur_port}.T") if not x.startswith(n)
                ]

                if len(aliases) > 1:
                    raise Exception(f"More than one connection to {cur_port}")
                elif len(aliases) == 0:
                    raise Exception(f"Found no connection to {cur_port}")

                if aliases[0].endswith(".QTHOut.T"):
                    pipe_w_orientation = (aliases[0][:-9], NodeConnectionDirection.IN)
                else:
                    assert aliases[0].endswith(".QTHIn.T")
                    pipe_w_orientation = (aliases[0][:-8], NodeConnectionDirection.OUT)

                assert pipe_w_orientation[0] in pipes_set

                connected_pipes[i] = pipe_w_orientation

        # Note that a pipe series can include both hot and cold pipes. It is
        # only about figuring out which pipes are related direction-wise.
        pipe_series = []

        canonical_pipe_qs = {p: self.alias_relation.canonical_signed(f"{p}.Q") for p in pipes}
        # Move sign from canonical to alias
        canonical_pipe_qs = {(p, d): c for p, (c, d) in canonical_pipe_qs.items()}
        # Reverse the dictionary from `Dict[alias, canonical]` to `Dict[canonical, Set[alias]]`
        pipe_sets = {}
        for a, c in canonical_pipe_qs.items():
            pipe_sets.setdefault(c, []).append(a)

        pipe_series = list(pipe_sets.values())

        self.__topology = Topology(node_connections, pipe_series)

        super().pre()

    @property
    def heat_network_components(self) -> Dict[str, str]:
        try:
            return self.__hn_component_types
        except AttributeError:
            string_parameters = self.string_parameters(0)

            # Find the components in model, detection by string
            # (name.component_type: type)
            component_types = sorted({v for k, v in string_parameters.items()})

            components = {}
            for c in component_types:
                components[c] = sorted({k[:-15] for k, v in string_parameters.items() if v == c})

            self.__hn_component_types = components

            return components

    @property
    def heat_network_topology(self) -> Topology:
        return self.__topology


class BoundsToPipeFlowDirectionsMixin(BaseComponentTypeMixin):
    def pre(self):
        super().pre()

        bounds = self.bounds()
        components = self.heat_network_components
        pipes = components["pipe"]

        # Determine implied pipe directions from model bounds (that are
        # already available at this time)
        self.__implied_directions = [{} for e in range(self.ensemble_size)]

        for e in range(self.ensemble_size):
            for p in pipes:
                lb, ub = bounds[f"{p}.Q"]

                if not isinstance(lb, float) or not isinstance(ub, float):
                    raise ValueError(
                        "`BoundsToPipeFlowDirectionsMixin` only works for scalar bounds"
                    )

                if lb == ub and lb == 0.0:
                    # Pipe is disabled
                    self.__implied_directions[e][p] = PipeFlowDirection.DISABLED
                elif lb >= 0.0:
                    self.__implied_directions[e][p] = PipeFlowDirection.POSITIVE
                elif ub <= 0.0:
                    self.__implied_directions[e][p] = PipeFlowDirection.NEGATIVE

    def constant_inputs(self, ensemble_member):
        inputs = super().constant_inputs(ensemble_member)
        for p, d in self.__implied_directions[ensemble_member].items():
            k = self.heat_network_pipe_flow_directions[p]
            inputs[k] = Timeseries([-np.inf, np.inf], [d, d])
        return inputs

    @property
    def heat_network_pipe_flow_directions(self) -> Dict[str, str]:
        pipes = self.heat_network_components["pipe"]
        return {p: f"{p}__implied_direction" for p in pipes}


class QTHMixin(BaseComponentTypeMixin, CollocatedIntegratedOptimizationProblem):
    """
    Adds handling of QTH heat network objects in your model to your
    optimization problem.

    Relevant parameters and variables are read from the model, and from this
    data a set of constraints and objectives are automatically generated, e.g.
    for the head loss and temperature losses in pipes.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
        options["wall_roughness"] = 2e-3
        options["head_loss_option"] = HeadLossOption.CQ2
        options["estimated_velocity"] = 1.0
        options["maximum_velocity"] = 2.0
        options["n_linearization_lines"] = 10

        return options

    @abstractproperty
    def heat_network_pipe_flow_directions(self) -> Dict[str, str]:
        """
        Maps a pipe name to its corresponding `constant_inputs` Timeseries
        name for the direction.
        """
        raise NotImplementedError

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

    def __node_mixing_constraints(self, ensemble_member):
        def state_vector_scaled(variable, ensemble_member):
            canonical, sign = self.alias_relation.canonical_signed(variable)
            return (
                self.state_vector(canonical, ensemble_member)
                * self.variable_nominal(canonical)
                * sign
            )

        parameters = self.parameters(ensemble_member)
        constraints = []

        theta = parameters[self.homotopy_options()["homotopy_parameter"]]

        interpolated_flow_dir_values = self.__get_interpolated_flow_directions(ensemble_member)

        for node, connected_pipes in self.heat_network_topology.nodes.items():
            temperature_node_sym = state_vector_scaled(f"{node}.Tnode", ensemble_member)
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
                conn_q = state_vector_scaled(f"{conn_base}.Q", ensemble_member)
                conn_q_abs = conn_q * flow_direction
                conn_t = state_vector_scaled(f"{conn_base}.T", ensemble_member)

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

    def __demand_head_loss_constraints(self, ensemble_member):
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
        constraints.extend(self.__demand_head_loss_constraints(ensemble_member))

        # Temperature gradient in pipes
        maximum_temperature_der = options["maximum_temperature_der"]

        if maximum_temperature_der is not None and isfinite(maximum_temperature_der):
            for pipe in components["pipe"]:
                # Note that maximum temperature change is expressed in °C per
                # hour. RTC-Tools uses seconds, so we have to scale the
                # derivative with 3600.
                dtemp_dt = self.der(f"{pipe}.QTHIn.T") * 3600.0
                constraints.append((dtemp_dt, -maximum_temperature_der, maximum_temperature_der))

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
