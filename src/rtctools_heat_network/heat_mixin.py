import logging
import math
from typing import List, Optional, Tuple

import casadi as ca

import numpy as np

from rtctools.optimization.collocated_integrated_optimization_problem import (
    CollocatedIntegratedOptimizationProblem,
)
from rtctools.optimization.timeseries import Timeseries

from rtctools_heat_network._heat_loss_u_values_pipe import heat_loss_u_values_pipe

from .base_component_type_mixin import BaseComponentTypeMixin
from .head_loss_mixin import HeadLossOption, _HeadLossMixin
from .pipe_class import PipeClass


logger = logging.getLogger("rtctools_heat_network")


class HeatMixin(_HeadLossMixin, BaseComponentTypeMixin, CollocatedIntegratedOptimizationProblem):

    __allowed_head_loss_options = {
        HeadLossOption.NO_HEADLOSS,
        HeadLossOption.LINEAR,
        HeadLossOption.LINEARIZED_DW,
    }

    def __init__(self, *args, **kwargs):
        # Prepare dicts for additional variables
        self.__flow_direct_var = {}
        self.__flow_direct_bounds = {}
        self.__pipe_to_flow_direct_map = {}

        self.__pipe_disconnect_var = {}
        self.__pipe_disconnect_var_bounds = {}
        self.__pipe_disconnect_map = {}

        self.__check_valve_status_var = {}
        self.__check_valve_status_var_bounds = {}
        self.__check_valve_status_map = {}

        self.__control_valve_direction_var = {}
        self.__control_valve_direction_var_bounds = {}
        self.__control_valve_direction_map = {}

        self.__buffer_t0_bounds = {}

        self.__pipe_topo_diameter_var = {}
        self.__pipe_topo_diameter_var_bounds = {}
        self.__pipe_topo_diameter_map = {}
        self.__pipe_topo_diameter_nominals = {}

        self.__pipe_topo_heat_loss_var = {}
        self.__pipe_topo_heat_loss_var_bounds = {}
        self.__pipe_topo_heat_loss_map = {}
        self.__pipe_topo_heat_loss_nominals = {}
        self.__pipe_topo_heat_losses = {}

        self.__pipe_topo_pipe_class_var = {}
        self.__pipe_topo_pipe_class_var_bounds = {}
        self.__pipe_topo_pipe_class_map = {}
        self.__pipe_topo_pipe_class_result = {}

        self.__pipe_topo_heat_discharge_bounds = {}

        self.__pipe_topo_diameter_area_parameters = []
        self.__pipe_topo_heat_loss_parameters = []

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

        # Mixed-integer formulation applies only to hot pipes, not to cold
        # pipes.
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

            if parameters[f"{p}.disconnectable"]:
                disconnected_var = f"{p}__is_disconnected"

                self.__pipe_disconnect_map[p] = disconnected_var
                self.__pipe_disconnect_var[disconnected_var] = ca.MX.sym(disconnected_var)
                self.__pipe_disconnect_var_bounds[disconnected_var] = (0.0, 1.0)

            if heat_in_ub <= 0.0 and heat_out_lb >= 0.0:
                raise Exception(f"Heat flow rate in/out of pipe '{p}' cannot be zero.")

        for v in self.heat_network_components.get("check_valve", []):
            status_var = f"{v}__status_var"

            self.__check_valve_status_map[v] = status_var
            self.__check_valve_status_var[status_var] = ca.MX.sym(status_var)
            self.__check_valve_status_var_bounds[status_var] = (0.0, 1.0)

        for v in self.heat_network_components.get("control_valve", []):
            flow_dir_var = f"{v}__flow_direct_var"

            self.__control_valve_direction_map[v] = flow_dir_var
            self.__control_valve_direction_var[flow_dir_var] = ca.MX.sym(flow_dir_var)
            self.__control_valve_direction_var_bounds[flow_dir_var] = (0.0, 1.0)

        # Pipe topology variables

        # In case the user overrides the pipe class of the pipe with a single
        # pipe class we update the diameter/area parameters. If there is more
        # than a single pipe class for a certain pipe, we set the diameter
        # and area to NaN to prevent erroneous constraints.
        for _ in range(self.ensemble_size):
            self.__pipe_topo_diameter_area_parameters.append({})
            self.__pipe_topo_heat_loss_parameters.append({})

        for pipe in self.hot_pipes:
            cold_pipe = self.hot_to_cold_pipe(pipe)
            pipe_classes = self.pipe_classes(pipe)
            cold_pipe = self.hot_to_cold_pipe(pipe)

            if len([c for c in pipe_classes if c.inner_diameter == 0]) > 1:
                raise Exception(
                    f"Pipe {pipe} should not have more than one `diameter = 0` pipe class"
                )

            # Note that we always make a diameter symbol, even if the diameter
            # is fixed. This can be convenient when playing around with
            # different pipe class options, and providing a uniform interface
            # to the user. Contrary to that, the pipe class booleans are very
            # much an internal affair.
            diam_var_name = f"{pipe}__hn_diameter"
            self.__pipe_topo_diameter_var[diam_var_name] = ca.MX.sym(diam_var_name)
            self.__pipe_topo_diameter_map[pipe] = diam_var_name

            if not pipe_classes:
                # No pipe class decision to make for this pipe w.r.t. diameter
                diameter = parameters[f"{pipe}.diameter"]
                self.__pipe_topo_diameter_var_bounds[diam_var_name] = (diameter, diameter)
                if diameter > 0.0:
                    self.__pipe_topo_diameter_nominals[diam_var_name] = diameter
            elif len(pipe_classes) == 1:
                # No pipe class decision to make for this pipe w.r.t. diameter
                diameter = pipe_classes[0].inner_diameter
                self.__pipe_topo_diameter_var_bounds[diam_var_name] = (diameter, diameter)
                if diameter > 0.0:
                    self.__pipe_topo_diameter_nominals[diam_var_name] = diameter

                for ensemble_member in range(self.ensemble_size):
                    d = self.__pipe_topo_diameter_area_parameters[ensemble_member]

                    for p in [pipe, cold_pipe]:
                        d[f"{p}.diameter"] = diameter
                        d[f"{p}.area"] = pipe_classes[0].area
            else:
                diameters = [c.inner_diameter for c in pipe_classes]
                self.__pipe_topo_diameter_var_bounds[diam_var_name] = (
                    min(diameters),
                    max(diameters),
                )
                self.__pipe_topo_diameter_nominals[diam_var_name] = min(
                    x for x in diameters if x > 0.0
                )

                for ensemble_member in range(self.ensemble_size):
                    d = self.__pipe_topo_diameter_area_parameters[ensemble_member]

                    for p in [pipe, cold_pipe]:
                        d[f"{p}.diameter"] = np.nan
                        d[f"{p}.area"] = np.nan

            # For similar reasons as for the diameter, we always make a heat
            # loss symbol, even if the heat loss is fixed. Note that we also
            # override the .Heat_loss parameter for cold pipes, even though
            # it is not actually used in the optimization problem.
            heat_loss_var_name = f"{pipe}__hn_heat_loss"
            self.__pipe_topo_heat_loss_var[heat_loss_var_name] = ca.MX.sym(heat_loss_var_name)
            self.__pipe_topo_heat_loss_map[pipe] = heat_loss_var_name

            heat_losses = [
                self.__pipe_heat_loss(options, parameters, pipe, c.u_values) for c in pipe_classes
            ]

            if not pipe_classes or options["neglect_pipe_heat_losses"]:
                # No pipe class decision to make for this pipe w.r.t. heat loss
                heat_loss = self.__pipe_heat_loss(options, parameters, pipe)
                self.__pipe_topo_heat_loss_var_bounds[heat_loss_var_name] = (heat_loss, heat_loss)
                if heat_loss > 0:
                    self.__pipe_topo_heat_loss_nominals[heat_loss_var_name] = heat_loss

                for ensemble_member in range(self.ensemble_size):
                    h = self.__pipe_topo_heat_loss_parameters[ensemble_member]
                    for p in [pipe, cold_pipe]:
                        h[f"{p}.Heat_loss"] = self.__pipe_heat_loss(options, parameters, p)
            elif len(pipe_classes) == 1:
                # No pipe class decision to make for this pipe w.r.t. heat loss
                u_values = pipe_classes[0].u_values
                heat_loss = self.__pipe_heat_loss(options, parameters, pipe, u_values)

                self.__pipe_topo_heat_loss_var_bounds[heat_loss_var_name] = (heat_loss, heat_loss)
                if heat_loss > 0:
                    self.__pipe_topo_heat_loss_nominals[heat_loss_var_name] = heat_loss

                for ensemble_member in range(self.ensemble_size):
                    h = self.__pipe_topo_heat_loss_parameters[ensemble_member]
                    h[f"{pipe}.Heat_loss"] = heat_loss
                    h[f"{cold_pipe}.Heat_loss"] = self.__pipe_heat_loss(
                        options, parameters, cold_pipe, u_values
                    )
            else:
                self.__pipe_topo_heat_losses[pipe] = heat_losses
                self.__pipe_topo_heat_loss_var_bounds[heat_loss_var_name] = (
                    min(heat_losses),
                    max(heat_losses),
                )
                self.__pipe_topo_heat_loss_nominals[heat_loss_var_name] = min(
                    x for x in heat_losses if x > 0
                )

                for ensemble_member in range(self.ensemble_size):
                    h = self.__pipe_topo_heat_loss_parameters[ensemble_member]
                    h[f"{pipe}.Heat_loss"] = np.nan
                    h[f"{cold_pipe}.Heat_loss"] = np.nan

            # Pipe class variables.
            if not pipe_classes or len(pipe_classes) == 1:
                # No pipe class decision to make for this pipe
                pass
            else:
                self.__pipe_topo_pipe_class_map[pipe] = {}

                for c in pipe_classes:
                    pipe_class_var_name = f"{pipe}__hn_pipe_class_{c.name}"

                    self.__pipe_topo_pipe_class_map[pipe][c] = pipe_class_var_name
                    self.__pipe_topo_pipe_class_var[pipe_class_var_name] = ca.MX.sym(
                        pipe_class_var_name
                    )
                    self.__pipe_topo_pipe_class_var_bounds[pipe_class_var_name] = (0.0, 1.0)

        # Update the bounds of the pipes that will have their diameter
        # optimized. Note that the flow direction may have already been fixed
        # based on the original bounds, if that was desired. We can therefore
        # naively override the bounds without taking this into account.
        for pipe in self.__pipe_topo_pipe_class_map:
            pipe_classes = self.__pipe_topo_pipe_class_map[pipe]
            max_discharge = max(c.maximum_discharge for c in pipe_classes)

            self.__pipe_topo_heat_discharge_bounds[f"{pipe}.Q"] = (-max_discharge, max_discharge)
            self.__pipe_topo_heat_discharge_bounds[f"{self.hot_to_cold_pipe(pipe)}.Q"] = (
                -max_discharge,
                max_discharge,
            )

            # Heat on cold side is zero, so no change needed
            cp = parameters[f"{pipe}.cp"]
            rho = parameters[f"{pipe}.rho"]
            dt = parameters[f"{pipe}.dT"]

            max_heat = cp * rho * dt * max_discharge

            self.__pipe_topo_heat_discharge_bounds[f"{pipe}.Heat_in"] = (-max_heat, max_heat)
            self.__pipe_topo_heat_discharge_bounds[f"{pipe}.Heat_out"] = (-max_heat, max_heat)

        # Note that all entries in self.__pipe_topo_heat_losses are guaranteed
        # to be in self.__pipe_topo_pipe_class_map, but not vice versa. If
        # e.g. all diameters have a heat loss of zero, we don't have any
        # decision to make w.r.t heat loss.
        for p in self.__pipe_topo_heat_losses:
            assert p in self.__pipe_topo_pipe_class_map

        # When optimizing for pipe size, we do not yet support all options
        if self.__pipe_topo_pipe_class_map:
            if options["minimum_velocity"] > 0.0:
                raise Exception(
                    "When optimizing pipe diameters, "
                    "the `maximum_velocity` option should be set to zero."
                )

            if np.isfinite(options["maximum_temperature_der"]) and np.isfinite(
                options["maximum_flow_der"]
            ):
                raise Exception(
                    "When optimizing pipe diameters, "
                    "the `maximum_temperature_der` or `maximum_flow_der` should be infinite."
                )

        # Check that buffer information is logical and
        # set the stored heat at t0 in the buffer(s) via bounds
        self.__check_buffer_values_and_set_bounds_at_t0()

        self.__maximum_total_head_loss = self.__get_maximum_total_head_loss()

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
        | ``neglect_pipe_heat_losses``         | ``bool``  | ``False``                   |
        +--------------------------------------+-----------+-----------------------------+
        | ``heat_loss_disconnected_pipe``      | ``bool``  | ``True``                    |
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

        The ``heat_loss_disconnected_pipe`` option decides whether a
        disconnectable pipe has heat loss or not when it is disconnected on
        that particular time step. By default, a pipe has heat loss even if
        it is disconnected, as it would still contain relatively hot water in
        reality. We also do not want minimization of heat production to lead
        to overly disconnecting pipes. In some scenarios it is hydraulically
        impossible to supply heat to these disconnected pipes (Q is forced to
        zero), in which case this option can be set to ``False``.

        The ``neglect_pipe_heat_losses`` option sets the heat loss in pipes to
        zero. This can be useful when the insulation properties are unknown.
        Note that other components can still have heat loss, e.g. a buffer.

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
        options["neglect_pipe_heat_losses"] = False
        options["heat_loss_disconnected_pipe"] = True
        options["minimum_velocity"] = 0.005
        options["head_loss_option"] = HeadLossOption.LINEAR
        options["minimize_head_losses"] = False

        return options

    def pipe_classes(self, pipe: str) -> List[PipeClass]:
        """
        Note that this method is only queried for _hot_ pipes. Their
        respective cold pipes are assumed to have the exact same properties.

        If the returned List is:
        - empty: use the pipe properties from the model
        - len() == 1: use these pipe properties to overrule that of the model
        - len() > 1: decide between the pipe class options.

        A pipe class with diameter 0 is interpreted as there being _no_ pipe.
        """
        return []

    def get_optimized_pipe_class(self, pipe: str) -> PipeClass:
        """
        Return the optimized pipe class for a specific pipe. If no
        optimized pipe class is available (yet), a `KeyError` is returned.
        """
        return self.__pipe_topo_pipe_class_result[pipe]

    def pipe_diameter_symbol_name(self, pipe: str) -> str:
        return self.__pipe_topo_diameter_map[pipe]

    @property
    def extra_variables(self):
        variables = super().extra_variables.copy()
        variables.extend(self.__pipe_topo_diameter_var.values())
        variables.extend(self.__pipe_topo_heat_loss_var.values())
        variables.extend(self.__pipe_topo_pipe_class_var.values())
        return variables

    @property
    def path_variables(self):
        variables = super().path_variables.copy()
        variables.extend(self.__flow_direct_var.values())
        variables.extend(self.__pipe_disconnect_var.values())
        variables.extend(self.__check_valve_status_var.values())
        variables.extend(self.__control_valve_direction_var.values())
        return variables

    def variable_is_discrete(self, variable):
        if (
            variable in self.__flow_direct_var
            or variable in self.__pipe_disconnect_var
            or variable in self.__check_valve_status_var
            or variable in self.__control_valve_direction_var
            or variable in self.__pipe_topo_pipe_class_var
        ):
            return True
        else:
            return super().variable_is_discrete(variable)

    def variable_nominal(self, variable):
        if variable in self.__pipe_topo_diameter_nominals:
            return self.__pipe_topo_diameter_nominals[variable]
        elif variable in self.__pipe_topo_heat_loss_nominals:
            return self.__pipe_topo_heat_loss_nominals[variable]
        else:
            return super().variable_nominal(variable)

    def bounds(self):
        bounds = super().bounds()
        bounds.update(self.__flow_direct_bounds)
        bounds.update(self.__pipe_disconnect_var_bounds)
        bounds.update(self.__check_valve_status_var_bounds)
        bounds.update(self.__control_valve_direction_var_bounds)
        bounds.update(self.__buffer_t0_bounds)
        bounds.update(self.__pipe_topo_pipe_class_var_bounds)
        bounds.update(self.__pipe_topo_diameter_var_bounds)
        bounds.update(self.__pipe_topo_heat_loss_var_bounds)
        bounds.update(self.__pipe_topo_heat_discharge_bounds)
        return bounds

    def __pipe_heat_loss(
        self, options, parameters, p: str, u_values: Optional[Tuple[float, float]] = None
    ):
        """
        The heat losses have three components:

        - dependency on the pipe temperature
        - dependency on the ground temperature
        - dependency on temperature difference between the supply/return line.

        This latter term assumes that the supply and return lines lie close
        to, and thus influence, each other. I.e., the supply line loses heat
        that is absorbed by the return line. Note that the term dtemp is
        positive when the pipe is in the supply line and negative otherwise.
        """
        if options["neglect_pipe_heat_losses"]:
            return 0.0

        if u_values is None:
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
        else:
            u_1, u_2 = u_values

        length = parameters[f"{p}.length"]
        temperature = parameters[f"{p}.temperature"]
        temperature_ground = parameters[f"{p}.T_ground"]
        sign_dtemp = 1 if self.is_hot_pipe(p) else -1
        dtemp = sign_dtemp * parameters[f"{p}.dT"]

        heat_loss = (
            length * (u_1 - u_2) * temperature
            - (length * (u_1 - u_2) * temperature_ground)
            + (length * u_2 * dtemp)
        )

        if heat_loss < 0:
            raise Exception(f"Heat loss of pipe {p} should be nonnegative.")

        return heat_loss

    def parameters(self, ensemble_member):
        parameters = super().parameters(ensemble_member)

        # To avoid mistakes by accidentally using the `diameter`, `area` and `Heat_loss`
        # parameters in e.g. constraints when those are variable, we set them
        # to NaN in that case. In post(), they are set to their resulting
        # values once again.
        if self.__pipe_topo_diameter_area_parameters:
            parameters.update(self.__pipe_topo_diameter_area_parameters[ensemble_member])
        if self.__pipe_topo_heat_loss_parameters:
            parameters.update(self.__pipe_topo_heat_loss_parameters[ensemble_member])

        return parameters

    def __get_maximum_total_head_loss(self):
        """
        Get an upper bound on the maximum total head loss that can be used in
        big-M formulations of e.g. check valves and disconnectable pipes.

        There are multiple ways to calculate this upper bound, depending on
        what options are set. We compute all these upper bounds, and return
        the lowest one of them.
        """

        options = self.heat_network_options()
        components = self.heat_network_components

        if options["head_loss_option"] == HeadLossOption.NO_HEADLOSS:
            # Undefined, and all constraints using this methods value should
            # be skipped.
            return np.nan

        # Summing head loss in pipes
        max_sum_dh_pipes = 0.0

        for ensemble_member in range(self.ensemble_size):
            parameters = self.parameters(ensemble_member)

            head_loss = 0.0

            for pipe in components["pipe"]:
                if self.is_cold_pipe(pipe):
                    hot_pipe = self.cold_to_hot_pipe(pipe)
                else:
                    hot_pipe = pipe

                try:
                    pipe_classes = self.__pipe_topo_pipe_class_map[hot_pipe].keys()
                    head_loss += max(
                        self._hn_pipe_head_loss(
                            pipe, options, parameters, pc.maximum_discharge, pipe_class=pc
                        )
                        for pc in pipe_classes
                        if pc.maximum_discharge > 0.0
                    )
                except KeyError:
                    area = parameters[f"{pipe}.area"]
                    max_discharge = options["maximum_velocity"] * area
                    head_loss += self._hn_pipe_head_loss(pipe, options, parameters, max_discharge)

            head_loss += options["minimum_pressure_far_point"] * 10.2

            max_sum_dh_pipes = max(max_sum_dh_pipes, head_loss)

        # Maximum pressure difference allowed with user options
        # NOTE: Does not yet take elevation differences into acccount
        max_dh_network_options = (
            options["pipe_maximum_pressure"] - options["pipe_minimum_pressure"]
        ) * 10.2

        return min(max_sum_dh_pipes, max_dh_network_options)

    def __check_buffer_values_and_set_bounds_at_t0(self):
        t = self.times()
        # We assume that t0 is always equal to self.times()[0]
        assert self.initial_time == self.times()[0]

        parameters = self.parameters(0)
        bounds = self.bounds()
        components = self.heat_network_components
        buffers = components.get("buffer", [])

        for b in buffers:
            min_fract_vol = parameters[f"{b}.min_fraction_tank_volume"]
            if min_fract_vol < 0.0 or min_fract_vol >= 1.0:
                raise Exception(
                    f"Minimum fraction of tank capacity of {b} must be smaller"
                    "than 1.0 and larger or equal to 0.0"
                )

            cp = parameters[f"{b}.cp"]
            rho = parameters[f"{b}.rho"]
            dt = parameters[f"{b}.dT"]
            heat_t0 = parameters[f"{b}.init_Heat"]
            vol_t0 = parameters[f"{b}.init_V_hot_tank"]
            stored_heat = f"{b}.Stored_heat"
            if not np.isnan(vol_t0) and not np.isnan(heat_t0):
                raise Exception(
                    f"At most one between the initial heat and volume of {b} should be prescribed."
                )

            if np.isnan(heat_t0):
                if not np.isnan(vol_t0):
                    # Extract information from volume
                    heat_t0 = vol_t0 * dt * cp * rho
                else:
                    # Set default value
                    volume = parameters[f"{b}.volume"]
                    default_vol_t0 = min_fract_vol * volume
                    heat_t0 = default_vol_t0 * dt * cp * rho

            # Check that volume/initial stored heat at t0 is within bounds
            lb_heat, ub_heat = bounds[stored_heat]
            lb_heat_t0 = np.inf
            ub_heat_t0 = -np.inf
            for bound in [lb_heat, ub_heat]:
                assert not isinstance(
                    bound, np.ndarray
                ), f"{b} stored heat cannot be a vector state"
                if isinstance(bound, Timeseries):
                    bound_t0 = bound.values[0]
                else:
                    bound_t0 = bound
                lb_heat_t0 = min(lb_heat_t0, bound_t0)
                ub_heat_t0 = max(ub_heat_t0, bound_t0)

            if heat_t0 < lb_heat_t0 or heat_t0 > ub_heat_t0:
                raise Exception(f"Initial heat of {b} is not within bounds.")

            # Set heat at t0
            lb = np.full_like(t, -np.inf)
            ub = np.full_like(t, np.inf)
            lb[0] = heat_t0
            ub[0] = heat_t0
            b_t0 = (Timeseries(t, lb), Timeseries(t, ub))
            self.__buffer_t0_bounds[stored_heat] = self.merge_bounds(bounds[stored_heat], b_t0)

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

        if np.isfinite(t_change) and np.isfinite(q_change):
            assert (
                not self.__pipe_topo_pipe_class_map
            ), "heat rate change constraints not allowed with topology optimization"

        for p in self.hot_pipes:

            variable = f"{p}.HeatIn.Heat"
            dt = np.diff(self.times(variable))

            canonical, sign = self.alias_relation.canonical_signed(variable)
            source_temperature_out = sign * self.state_vector(canonical, ensemble_member)

            # Maximum differences are expressed per hour. We scale appropriately.
            cp = parameters[f"{p}.cp"]
            rho = parameters[f"{p}.rho"]
            area = parameters[f"{p}.area"]
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
        options = self.heat_network_options()

        for p in self.cold_pipes:
            heat_in = self.state(f"{p}.HeatIn.Heat")
            heat_out = self.state(f"{p}.HeatOut.Heat")
            heat_nominal = self.variable_nominal(f"{p}.HeatOut.Heat")

            constraints.append(((heat_in - heat_out) / heat_nominal, 0.0, 0.0))

        for p in self.hot_pipes:
            heat_in = self.state(f"{p}.HeatIn.Heat")
            heat_out = self.state(f"{p}.HeatOut.Heat")
            heat_nominal = self.variable_nominal(f"{p}.HeatIn.Heat")

            is_disconnected_var = self.__pipe_disconnect_map.get(p)

            if is_disconnected_var is None:
                is_disconnected = 0.0
            else:
                is_disconnected = self.state(is_disconnected_var)

            if p in self.__pipe_topo_heat_losses:
                # Heat loss is variable depending on pipe class
                heat_loss_sym_name = self.__pipe_topo_heat_loss_map[p]
                heat_loss = self.__pipe_topo_heat_loss_var[heat_loss_sym_name]
                heat_loss_nominal = self.__pipe_topo_heat_loss_nominals[heat_loss_sym_name]
                constraint_nominal = (heat_nominal * heat_loss_nominal) ** 0.5

                if options["heat_loss_disconnected_pipe"]:
                    constraints.append(
                        ((heat_in - heat_out - heat_loss) / constraint_nominal, 0.0, 0.0)
                    )
                else:
                    # Force heat loss to `heat_loss` when pipe is connected, and zero otherwise.
                    big_m = 2 * max(self.__pipe_topo_heat_losses[p])
                    heat_loss_nominal = self.__pipe_topo_heat_loss_nominals[heat_loss_sym_name]
                    constraint_nominal = (heat_nominal * heat_loss_nominal) ** 0.5

                    # Force heat loss to `heat_loss` when pipe is connected.
                    constraints.append(
                        (
                            (heat_in - heat_out - heat_loss - is_disconnected * big_m)
                            / constraint_nominal,
                            -np.inf,
                            0.0,
                        )
                    )
                    constraints.append(
                        (
                            (heat_in - heat_out - heat_loss + is_disconnected * big_m)
                            / constraint_nominal,
                            0.0,
                            np.inf,
                        )
                    )

                    # Force heat loss to zero (heat_in = heat_out) when pipe is
                    # disconnected. Note that heat loss is never less than zero, so
                    # we can skip a Big-M formulation in the lower bound.
                    constraints.append(
                        (
                            (heat_in - heat_out - (1 - is_disconnected) * big_m)
                            / constraint_nominal,
                            -np.inf,
                            0.0,
                        )
                    )
                    constraints.append(
                        (
                            (heat_in - heat_out) / heat_nominal,
                            0.0,
                            np.inf,
                        )
                    )
            else:
                # Heat loss is constant, i.e. does not depend on pipe class
                heat_loss = parameters[f"{p}.Heat_loss"]

                if options["heat_loss_disconnected_pipe"]:
                    constraints.append(((heat_in - heat_out - heat_loss) / heat_nominal, 0.0, 0.0))
                else:
                    constraint_nominal = (heat_loss * heat_nominal) ** 0.5
                    constraints.append(
                        (
                            (heat_in - heat_out - heat_loss * (1 - is_disconnected))
                            / constraint_nominal,
                            0.0,
                            0.0,
                        )
                    )

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

            heat_nominal = self.variable_nominal(f"{p}.HeatIn.Heat")

            big_m = self.__get_abs_max_bounds(
                *self.merge_bounds(bounds[f"{p}.HeatIn.Heat"], bounds[f"{p}.HeatOut.Heat"])
            )

            if not np.isfinite(big_m):
                raise Exception(f"Heat in pipe {p} must be bounded")

            constraint_nominal = (big_m * heat_nominal) ** 0.5

            # Fix flow direction
            constraints.append(((heat_in - big_m * flow_dir) / constraint_nominal, -np.inf, 0.0))
            constraints.append(
                ((heat_in + big_m * (1 - flow_dir)) / constraint_nominal, 0.0, np.inf)
            )

            # Flow direction is the same for In and Out. Note that this
            # ensures that the heat going in and/or out of a pipe is more than
            # its heat losses.
            constraints.append(((heat_out - big_m * flow_dir) / constraint_nominal, -np.inf, 0.0))
            constraints.append(
                ((heat_out + big_m * (1 - flow_dir)) / constraint_nominal, 0.0, np.inf)
            )

            if not options["heat_loss_disconnected_pipe"]:
                # If this pipe is disconnected, the heat should be zero
                is_disconnected_var = self.__pipe_disconnect_map.get(p)

                if is_disconnected_var is not None:
                    is_disconnected = self.state(is_disconnected_var)
                    is_conn = 1 - is_disconnected

                    # Note that big_m should now cover the range from [-max, max],
                    # so we need to double it.
                    big_m_dbl = 2 * big_m
                    for heat in [heat_in, heat_out]:
                        constraints.append(
                            ((heat + big_m_dbl * is_conn) / constraint_nominal, 0.0, np.inf)
                        )
                        constraints.append(
                            ((heat - big_m_dbl * is_conn) / constraint_nominal, -np.inf, 0.0)
                        )

        minimum_velocity = options["minimum_velocity"]
        maximum_velocity = options["maximum_velocity"]

        if minimum_velocity > 0.0:
            assert (
                not self.__pipe_topo_pipe_class_map
            ), "non-zero minimum velocity not allowed with topology optimization"

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

            try:
                pipe_classes = self.__pipe_topo_pipe_class_map[hot_pipe].keys()
                maximum_discharge = max(c.maximum_discharge for c in pipe_classes)
                minimum_discharge = 0.0
            except KeyError:
                maximum_discharge = maximum_velocity * parameters[f"{p}.area"]

                if math.isfinite(minimum_velocity) and minimum_velocity > 0.0:
                    minimum_discharge = minimum_velocity * parameters[f"{p}.area"]
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
            heat_nominal = parameters[f"{d}.Heat_nominal"]
            q_nominal = self.variable_nominal(f"{d}.Q")
            cp = parameters[f"{d}.cp"]
            rho = parameters[f"{d}.rho"]
            dt = parameters[f"{d}.dT"]

            discharge = self.state(f"{d}.Q")
            heat_consumed = self.state(f"{d}.Heat_demand")

            constraint_nominal = (heat_nominal * cp * rho * dt * q_nominal) ** 0.5

            constraints.append(
                ((heat_consumed - cp * rho * dt * discharge) / constraint_nominal, 0.0, 0.0)
            )

        return constraints

    def __source_heat_to_discharge_path_constraints(self, ensemble_member):
        constraints = []
        parameters = self.parameters(ensemble_member)

        for s in self.heat_network_components["source"]:
            heat_nominal = parameters[f"{s}.Heat_nominal"]
            q_nominal = self.variable_nominal(f"{s}.Q")
            cp = parameters[f"{s}.cp"]
            rho = parameters[f"{s}.rho"]
            dt = parameters[f"{s}.dT"]

            discharge = self.state(f"{s}.Q")
            heat_production = self.state(f"{s}.Heat_source")

            constraint_nominal = (heat_nominal * cp * rho * dt * q_nominal) ** 0.5

            constraints.append(
                ((heat_production - cp * rho * dt * discharge) / constraint_nominal, 0.0, np.inf)
            )

        return constraints

    def __pipe_heat_to_discharge_path_constraints(self, ensemble_member):
        constraints = []
        parameters = self.parameters(ensemble_member)

        sum_heat_losses = 0.0

        for p in self.hot_pipes:
            if p in self.__pipe_topo_heat_losses:
                sum_heat_losses += max(self.__pipe_topo_heat_losses[p])
            else:
                sum_heat_losses += parameters[f"{p}.Heat_loss"]

        assert not np.isnan(sum_heat_losses)

        for p in self.hot_pipes:
            cp = parameters[f"{p}.cp"]
            rho = parameters[f"{p}.rho"]
            dt = parameters[f"{p}.dT"]
            heat_to_discharge_fac = 1.0 / (cp * rho * dt)

            flow_dir_var = self.__pipe_to_flow_direct_map[p]
            flow_dir = self.state(flow_dir_var)
            scaled_heat_in = self.state(f"{p}.HeatIn.Heat") * heat_to_discharge_fac
            scaled_heat_out = self.state(f"{p}.HeatOut.Heat") * heat_to_discharge_fac
            pipe_q = self.state(f"{p}.Q")
            q_nominal = self.variable_nominal(f"{p}.Q")

            # We do not want Big M to be too tight in this case, as it results
            # in a rather hard yes/no constraint as far as feasibility on e.g.
            # a single source system is concerned. Use a factor of 2 to give
            # some slack.
            big_m = 2 * sum_heat_losses * heat_to_discharge_fac

            for heat in (scaled_heat_in, scaled_heat_out):
                if sum_heat_losses == 0:
                    constraints.append(((heat - pipe_q) / q_nominal, 0.0, 0.0))
                else:
                    assert big_m > 0.0
                    constraints.append(
                        ((heat - pipe_q + big_m * (1 - flow_dir)) / big_m, 0.0, np.inf)
                    )
                    constraints.append(((heat - pipe_q - big_m * flow_dir) / big_m, -np.inf, 0.0))

        return constraints

    def __buffer_heat_to_discharge_path_constraints(self, ensemble_member):
        constraints = []
        parameters = self.parameters(ensemble_member)
        bounds = self.bounds()

        for b, ((hot_pipe, hot_pipe_orientation), _) in self.heat_network_topology.buffers.items():
            heat_nominal = parameters[f"{b}.Heat_nominal"]
            q_nominal = self.variable_nominal(f"{b}.Q")
            cp = parameters[f"{b}.cp"]
            rho = parameters[f"{b}.rho"]
            dt = parameters[f"{b}.dT"]

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

            coefficients = [heat_nominal, cp * rho * dt * q_nominal, big_m]
            constraint_nominal = (min(coefficients) * max(coefficients)) ** 0.5
            constraints.append(
                (
                    (heat_consumed - cp * rho * dt * discharge + (1 - is_buffer_charging) * big_m)
                    / constraint_nominal,
                    0.0,
                    np.inf,
                )
            )

            constraint_nominal = (heat_nominal * cp * rho * dt * q_nominal) ** 0.5
            constraints.append(
                ((heat_consumed - cp * rho * dt * discharge) / constraint_nominal, -np.inf, 0.0)
            )

        return constraints

    def __state_vector_scaled(self, variable, ensemble_member):
        canonical, sign = self.alias_relation.canonical_signed(variable)
        return (
            self.state_vector(canonical, ensemble_member) * self.variable_nominal(canonical) * sign
        )

    def _hn_pipe_nominal_discharge(self, heat_network_options, parameters, pipe: str) -> float:
        if self.is_cold_pipe(pipe):
            hot_pipe = self.cold_to_hot_pipe(pipe)
        else:
            hot_pipe = pipe

        try:
            pipe_classes = self.__pipe_topo_pipe_class_map[hot_pipe].keys()
            area = np.median(c.area for c in pipe_classes)
        except KeyError:
            area = parameters[f"{pipe}.area"]

        return area * heat_network_options["estimated_velocity"]

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

            head_loss_sym = self._hn_pipe_to_head_loss_map[pipe]

            dh = self.__state_vector_scaled(f"{pipe}.dH", ensemble_member)
            head_loss = self.__state_vector_scaled(head_loss_sym, ensemble_member)
            discharge = self.__state_vector_scaled(f"{pipe}.Q", ensemble_member)

            # We need to make sure the dH is decoupled from the discharge when
            # the pipe is disconnected. Simply put, this means making the
            # below constraints trivial.
            is_disconnected_var = self.__pipe_disconnect_map.get(hot_pipe)

            if is_disconnected_var is None:
                is_disconnected = 0.0
            else:
                is_disconnected = self.__state_vector_scaled(is_disconnected_var, ensemble_member)

            max_discharge = None
            max_head_loss = -np.inf

            if hot_pipe in self.__pipe_topo_pipe_class_map:
                # Multiple diameter options for this pipe
                pipe_classes = self.__pipe_topo_pipe_class_map[hot_pipe]
                max_discharge = max(c.maximum_discharge for c in pipe_classes)

                for pc, pc_var_name in pipe_classes.items():
                    if pc.inner_diameter == 0.0:
                        continue

                    head_loss_max_discharge = self._hn_pipe_head_loss(
                        pipe, options, parameters, max_discharge, pipe_class=pc
                    )

                    big_m = max(1.1 * self.__maximum_total_head_loss, 2 * head_loss_max_discharge)

                    is_topo_disconnected = 1 - self.extra_variable(pc_var_name, ensemble_member)
                    is_topo_disconnected = ca.repmat(is_topo_disconnected, dh.size1())

                    # Note that we add the two booleans `is_disconnected` and
                    # `is_topo_disconnected`. This is allowed because of the way the
                    # resulting expression is used in the Big-M formulation. We only care
                    # that the expression (i.e. a single boolean or the sum of the two
                    # booleans) is either 0 when the pipe is connected, or >= 1 when it
                    # is disconnected.
                    constraints.extend(
                        self._hn_pipe_head_loss(
                            pipe,
                            options,
                            parameters,
                            discharge,
                            head_loss,
                            dh,
                            is_disconnected + is_topo_disconnected,
                            big_m,
                            pc,
                        )
                    )

                    # Contrary to the Big-M calculation above, the relation
                    # between dH and the head loss symbol requires the
                    # maximum head loss that can be realized effectively. So
                    # we pass the current pipe class's maximum discharge.
                    max_head_loss = max(
                        max_head_loss,
                        self._hn_pipe_head_loss(
                            pipe, options, parameters, pc.maximum_discharge, pipe_class=pc
                        ),
                    )
            else:
                # Only a single diameter for this pipe. Note that we rely on
                # the diameter parameter being overridden automatically if a
                # single pipe class is set by the user.
                area = parameters[f"{pipe}.area"]
                max_discharge = options["maximum_velocity"] * area

                is_topo_disconnected = int(parameters[f"{pipe}.diameter"] == 0.0)

                constraints.extend(
                    self._hn_pipe_head_loss(
                        pipe,
                        options,
                        parameters,
                        discharge,
                        head_loss,
                        dh,
                        is_disconnected + is_topo_disconnected,
                        1.1 * self.__maximum_total_head_loss,
                    )
                )

                max_head_loss = self._hn_pipe_head_loss(pipe, options, parameters, max_discharge)

            # Relate the head loss symbol to the pipe's dH symbol.

            # FIXME: Ugly hack. Cold pipes should be modelled completely with
            # their own integers as well.
            flow_dir = self.__state_vector_scaled(
                self.__pipe_to_flow_direct_map[hot_pipe], ensemble_member
            )

            # Note that the Big-M should _at least_ cover the maximum
            # distance between `head_loss` and `dh`. If `head_loss` can be at
            # most 1.0 (= `max_head_loss`), that means our Big-M should be at
            # least double (i.e. >= 2.0). And because we do not want Big-Ms to
            # be overly tight, we include an additional factor of 2.
            big_m = 2 * 2 * max_head_loss

            constraints.append(
                (
                    (-dh - head_loss + (1 - flow_dir) * big_m) / big_m,
                    0.0,
                    np.inf,
                )
            )
            constraints.append(((dh - head_loss + flow_dir * big_m) / big_m, 0.0, np.inf))

        return constraints

    def __check_valve_head_discharge_path_constraints(self, ensemble_member):
        constraints = []
        parameters = self.parameters(ensemble_member)
        options = self.heat_network_options()

        all_pipes = set(self.heat_network_components["pipe"])
        maximum_velocity = options["maximum_velocity"]

        for v in self.heat_network_components.get("check_valve", []):
            status_var = self.__check_valve_status_map[v]
            status = self.state(status_var)

            q = self.state(f"{v}.Q")
            dh = self.state(f"{v}.dH")

            # Determine the maximum discharge that can go through the Valve
            # by looking at connected pipes.
            q_aliases = self.alias_relation.aliases(q.name())
            connected_pipes = {p for p in all_pipes if f"{p}.Q" in q_aliases}

            maximum_discharge = 0.0

            for p in connected_pipes:
                try:
                    pipe_classes = self.__pipe_topo_pipe_class_map[p].keys()
                    max_discharge_pipe = max(c.maximum_discharge for c in pipe_classes)
                except KeyError:
                    max_discharge_pipe = maximum_velocity * parameters[f"{p}.area"]

                maximum_discharge = max(maximum_discharge, max_discharge_pipe)

            maximum_head_loss = self.__maximum_total_head_loss

            # (Ideal) check valve status:
            # - 1 means "open", so positive discharge, and dH = 0
            # - 0 means "closed", so Q = 0 and positive dH
            # Note that the Q >= 0 and dH >= 0 constraints are part of the bounds.
            constraints.append((q - status * maximum_discharge, -np.inf, 0.0))

            if options["head_loss_option"] != HeadLossOption.NO_HEADLOSS:
                constraints.append((dh - (1 - status) * maximum_head_loss, -np.inf, 0.0))

        return constraints

    def __control_valve_head_discharge_path_constraints(self, ensemble_member):
        constraints = []
        parameters = self.parameters(ensemble_member)
        options = self.heat_network_options()

        all_pipes = set(self.heat_network_components["pipe"])
        maximum_velocity = options["maximum_velocity"]

        for v in self.heat_network_components.get("control_valve", []):
            flow_dir_var = self.__control_valve_direction_map[v]
            flow_dir = self.state(flow_dir_var)

            q = self.state(f"{v}.Q")
            dh = self.state(f"{v}.dH")

            # Determine the maximum discharge that can go through the Valve
            # by looking at connected pipes.
            q_aliases = self.alias_relation.aliases(q.name())
            connected_pipes = {p for p in all_pipes if f"{p}.Q" in q_aliases}

            maximum_discharge = 0.0

            for p in connected_pipes:
                try:
                    pipe_classes = self.__pipe_topo_pipe_class_map[p].keys()
                    max_discharge_pipe = max(c.maximum_discharge for c in pipe_classes)
                except KeyError:
                    max_discharge_pipe = maximum_velocity * parameters[f"{p}.area"]

                maximum_discharge = max(maximum_discharge, max_discharge_pipe)

            maximum_head_loss = self.__maximum_total_head_loss

            # Flow direction:
            # - 1 means positive discharge, and negative dH
            # - 0 means negative discharge, and positive dH
            # It's a control valve, so the dH is of arbitrary magnitude.
            constraints.append((q + (1 - flow_dir) * maximum_discharge, 0.0, np.inf))
            constraints.append((q - flow_dir * maximum_discharge, -np.inf, 0.0))

            if options["head_loss_option"] != HeadLossOption.NO_HEADLOSS:
                constraints.append((-dh + (1 - flow_dir) * maximum_head_loss, 0.0, np.inf))
                constraints.append((-dh - flow_dir * maximum_head_loss, -np.inf, 0.0))

        return constraints

    def __pipe_topology_constraints(self, ensemble_member):
        constraints = []

        for p, pipe_classes in self.__pipe_topo_pipe_class_map.items():
            v = []
            for pc_var_name in pipe_classes.values():
                v.append(self.extra_variable(pc_var_name, ensemble_member))

            # Make sure exactly one indicator is true
            constraints.append((sum(v), 1.0, 1.0))

            # Match the indicators to the diameter symbol
            diam_sym_name = self.__pipe_topo_diameter_map[p]
            diam_sym = self.extra_variable(diam_sym_name, ensemble_member)

            diameters = [c.inner_diameter for c in pipe_classes.keys()]

            diam_expr = sum(s * d for s, d in zip(v, diameters))
            constraint_nominal = self.variable_nominal(diam_sym_name)

            constraints.append(((diam_sym - diam_expr) / constraint_nominal, 0.0, 0.0))

        for p, heat_losses in self.__pipe_topo_heat_losses.items():
            assert self.is_hot_pipe(p)

            pipe_classes = self.__pipe_topo_pipe_class_map[p]
            v = []
            for pc_var_name in pipe_classes.values():
                v.append(self.extra_variable(pc_var_name, ensemble_member))

            heat_loss_sym_name = self.__pipe_topo_heat_loss_map[p]
            heat_loss_sym = self.extra_variable(heat_loss_sym_name, ensemble_member)

            heat_loss_expr = sum(s * h for s, h in zip(v, heat_losses))
            constraint_nominal = self.variable_nominal(heat_loss_sym_name)

            constraints.append(((heat_loss_sym - heat_loss_expr) / constraint_nominal, 0.0, 0.0))

        return constraints

    def __pipe_topology_path_constraints(self, ensemble_member):
        constraints = []

        # Clip discharge based on pipe class
        for p, pipe_classes in self.__pipe_topo_pipe_class_map.items():
            v = []
            for var_name in pipe_classes.values():
                v.append(self.__pipe_topo_pipe_class_var[var_name])

            # Match the indicators to the discharge symbol(s)
            discharge_sym_hot = self.state(f"{p}.Q")
            discharge_sym_cold = self.state(f"{self.hot_to_cold_pipe(p)}.Q")

            maximum_discharges = [c.maximum_discharge for c in pipe_classes.keys()]

            max_discharge_expr = sum(s * d for s, d in zip(v, maximum_discharges))
            constraint_nominal = self.variable_nominal(f"{p}.Q")

            constraints.append(
                ((discharge_sym_hot - max_discharge_expr) / constraint_nominal, -np.inf, 0.0)
            )
            constraints.append(
                ((discharge_sym_hot + max_discharge_expr) / constraint_nominal, 0.0, np.inf)
            )

            constraints.append(
                ((discharge_sym_cold - max_discharge_expr) / constraint_nominal, -np.inf, 0.0)
            )
            constraints.append(
                ((discharge_sym_cold + max_discharge_expr) / constraint_nominal, 0.0, np.inf)
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
        constraints.extend(self.__check_valve_head_discharge_path_constraints(ensemble_member))
        constraints.extend(self.__control_valve_head_discharge_path_constraints(ensemble_member))
        constraints.extend(self.__pipe_topology_path_constraints(ensemble_member))

        return constraints

    def constraints(self, ensemble_member):
        constraints = super().constraints(ensemble_member)

        constraints.extend(self.__pipe_rate_heat_change_constraints(ensemble_member))
        constraints.extend(self.__pipe_topology_constraints(ensemble_member))

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

    def __pipe_class_to_results(self):
        for ensemble_member in range(self.ensemble_size):
            results = self.extract_results(ensemble_member)

            for pipe in self.hot_pipes:
                pipe_classes = self.pipe_classes(pipe)

                if not pipe_classes:
                    continue
                elif len(pipe_classes) == 1:
                    pipe_class = pipe_classes[0]
                else:
                    pipe_class = next(
                        c
                        for c, s in self.__pipe_topo_pipe_class_map[pipe].items()
                        if round(results[s][0]) == 1.0
                    )

                for p in [pipe, self.hot_to_cold_pipe(pipe)]:
                    self.__pipe_topo_pipe_class_result[p] = pipe_class

    def __pipe_diameter_to_parameters(self):
        for ensemble_member in range(self.ensemble_size):
            d = self.__pipe_topo_diameter_area_parameters[ensemble_member]
            for pipe in self.__pipe_topo_pipe_class_map:
                pipe_class = self.get_optimized_pipe_class(pipe)

                for p in [pipe, self.hot_to_cold_pipe(pipe)]:
                    d[f"{p}.diameter"] = pipe_class.inner_diameter
                    d[f"{p}.area"] = pipe_class.area

    def __pipe_heat_loss_to_parameters(self):
        options = self.heat_network_options()

        for ensemble_member in range(self.ensemble_size):
            parameters = self.parameters(ensemble_member)

            h = self.__pipe_topo_heat_loss_parameters[ensemble_member]
            for pipe in self.__pipe_topo_heat_losses:
                pipe_class = self.get_optimized_pipe_class(pipe)

                cold_pipe = self.hot_to_cold_pipe(pipe)

                for p in [pipe, cold_pipe]:
                    h[f"{p}.Heat_loss"] = self.__pipe_heat_loss(
                        options, parameters, p, pipe_class.u_values
                    )

    def priority_completed(self, priority):
        options = self.heat_network_options()

        self.__pipe_class_to_results()

        # The head loss mixin wants to do some check for the head loss
        # minimization priority that involves the diameter/area. We assume
        # that we're sort of done minimizing/choosing the pipe diameter, and
        # that we can set the parameters to the optimized values.
        if (
            options["minimize_head_losses"]
            and options["head_loss_option"] != HeadLossOption.NO_HEADLOSS
            and priority == self._hn_minimization_goal_class.priority
        ):
            self.__pipe_diameter_to_parameters()

        super().priority_completed(priority)

    def post(self):
        super().post()

        self.__pipe_class_to_results()
        self.__pipe_diameter_to_parameters()
        self.__pipe_heat_loss_to_parameters()

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

        if options["head_loss_option"] != HeadLossOption.NO_HEADLOSS:
            for p in self.heat_network_components["pipe"]:
                head_diff = results[f"{p}.HeatIn.H"] - results[f"{p}.HeatOut.H"]
                if parameters[f"{p}.length"] == 0.0 and not parameters[f"{p}.has_control_valve"]:
                    atol = self.variable_nominal(f"{p}.HeatIn.H") * 1e-5
                    assert np.allclose(head_diff, 0.0, atol=atol)
                else:
                    q = results[f"{p}.Q"]

                    if self.is_cold_pipe(p):
                        hot_pipe = self.cold_to_hot_pipe(p)
                    else:
                        hot_pipe = p

                    try:
                        is_disconnected = np.round(results[self.__pipe_disconnect_map[hot_pipe]])
                    except KeyError:
                        is_disconnected = np.zeros_like(q)

                    q_nominal = self.variable_nominal(
                        self.alias_relation.canonical_signed(f"{p}.Q")[0]
                    )
                    inds = (np.abs(q) / q_nominal > 1e-4) & (is_disconnected == 0)
                    assert np.all(np.sign(head_diff[inds]) == np.sign(q[inds]))

        minimum_velocity = options["minimum_velocity"]
        for p in self.heat_network_components["pipe"]:
            if self.is_cold_pipe(p):
                hot_pipe = self.cold_to_hot_pipe(p)
            else:
                hot_pipe = p
            area = parameters[f"{p}.area"]

            if area == 0.0:
                continue

            q = results[f"{p}.Q"]
            v = q / area
            flow_dir = np.round(results[self.__pipe_to_flow_direct_map[hot_pipe]])
            try:
                is_disconnected = np.round(results[self.__pipe_disconnect_map[hot_pipe]])
            except KeyError:
                is_disconnected = np.zeros_like(q)

            inds_disconnected = is_disconnected == 1
            inds_positive = (flow_dir == 1) & ~inds_disconnected
            inds_negative = (flow_dir == 0) & ~inds_disconnected

            # We allow a bit of slack in the velocity. If the
            # exceedence/discrepancy is more than 0.1 mm/s, we log a warning,
            # if it's more than 1 cm/s, we log an error message.
            if np.any(inds_positive) or np.any(inds_negative):
                max_exceedence = max(
                    np.hstack(
                        [minimum_velocity - v[inds_positive], v[inds_negative] + minimum_velocity]
                    )
                )

                for criterion, log_level in [(0.01, logging.ERROR), (1e-4, logging.WARNING)]:
                    if max_exceedence > criterion:
                        logger.log(
                            log_level,
                            f"Velocity in {p} exceeds minimum velocity {minimum_velocity} "
                            f"by more than {criterion} m/s. ({max_exceedence} m/s)",
                        )

                        break

            # Similar check for disconnected pipes, where we want the velocity
            # to be zero but allow the same amount of slack.
            if np.any(inds_disconnected):
                max_exceedence = max(np.abs(v[inds_disconnected]))

                for criterion, log_level in [(0.01, logging.ERROR), (1e-4, logging.WARNING)]:
                    if max_exceedence > criterion:
                        logger.log(
                            log_level,
                            f"Velocity in disconnected pipe {p} exceeds {criterion} m/s. "
                            f"({max_exceedence} m/s)",
                        )

                        break

        for p in self.hot_pipes:
            if parameters[f"{p}.diameter"] == 0.0:
                continue

            heat_in = results[f"{p}.HeatIn.Heat"]
            heat_out = results[f"{p}.HeatOut.Heat"]
            inds = np.abs(heat_out) > np.abs(heat_in)
            nominal = self.variable_nominal(f"{p}.HeatIn.Heat")

            heat = heat_in.copy()
            heat[inds] = heat_out[inds]

            flow_dir_var = np.round(results[self.__pipe_to_flow_direct_map[p]])

            if options["heat_loss_disconnected_pipe"]:
                np.testing.assert_array_equal(np.sign(heat), 2 * flow_dir_var - 1)
            else:
                try:
                    is_disconnected = np.round(results[self.__pipe_disconnect_map[p]])
                except KeyError:
                    is_disconnected = np.zeros_like(heat_in)

                inds_disconnected = is_disconnected == 1

                np.testing.assert_allclose(
                    heat[inds_disconnected] / nominal, 0.0, atol=1e-5, rtol=0
                )

                np.testing.assert_array_equal(
                    np.sign(heat[~inds_disconnected]), 2 * flow_dir_var[~inds_disconnected] - 1
                )
