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
from .constants import GRAVITATIONAL_CONSTANT
from .demand_insulation_class import DemandInsulationClass
from .head_loss_mixin import HeadLossOption, _HeadLossMixin

logger = logging.getLogger("rtctools_heat_network")


class HeatPhysicsMixin(_HeadLossMixin, BaseComponentTypeMixin, CollocatedIntegratedOptimizationProblem):
    __allowed_head_loss_options = {
        HeadLossOption.NO_HEADLOSS,
        HeadLossOption.LINEAR,
        HeadLossOption.LINEARIZED_DW,
    }

    def __init__(self, *args, **kwargs):
        """
        In this __init__ we prepare the dicts for the variables added by the HeatMixin class
        """
        # Boolean path-variable for the direction of the flow, inport to outport is positive flow.
        self.__flow_direct_var = {}
        self.__flow_direct_bounds = {}
        self.__pipe_to_flow_direct_map = {}

        # Boolean path-variable to determine whether flow is going through a pipe.
        self.__pipe_disconnect_var = {}
        self.__pipe_disconnect_var_bounds = {}
        self.__pipe_disconnect_map = {}

        # Boolean path-variable for the status of the check valve
        self.__check_valve_status_var = {}
        self.__check_valve_status_var_bounds = {}
        self.__check_valve_status_map = {}

        # Boolean path-variable for the status of the control valve
        self.__control_valve_direction_var = {}
        self.__control_valve_direction_var_bounds = {}
        self.__control_valve_direction_map = {}

        # Boolean path-variable to disable the hex for certain moments in time
        self.__disabled_hex_map = {}
        self.__disabled_hex_var = {}
        self.__disabled_hex_var_bounds = {}

        # To avoid artificial energy generation at t0
        self.__buffer_t0_bounds = {}

        # Variable for the heat-loss (not specific for pipe-class optimization)
        self.__pipe_topo_heat_loss_var = {}
        self.__pipe_topo_heat_loss_path_var = {}
        self.__pipe_topo_heat_loss_var_bounds = {}
        self.__pipe_topo_heat_loss_map = {}
        self.__pipe_topo_heat_loss_nominals = {}
        self.__pipe_topo_heat_losses = {}

        # Boolean variables for the insulation options per demand.
        self.__demand_insulation_class_var = {}  # value 0/1: demand insulation - not active/active
        self.__demand_insulation_class_var_bounds = {}
        self.__demand_insulation_class_map = {}
        self.__demand_insulation_class_result = {}

        # Variable of selected network temperature
        self.__temperature_regime_var = {}
        self.__temperature_regime_var_bounds = {}

        # Integer variable whether discrete temperature option has been selected
        self.__carrier_selected_var = {}
        self.__carrier_selected_var_bounds = {}

        super().__init__(*args, **kwargs)

    def temperature_carriers(self):
        """
        This function should be overwritten by the problem and should give a dict with the
        carriers as keys and a list of temperatures as values.
        """
        return {}

    def temperature_regimes(self, carrier):
        """
        This funciton returns a list of temperatures that can be selected for a certain carrier.
        """
        return []

    def pre(self):
        """
        In this pre method we fill the dicts initiated in the __init__. This means that we create
        the Casadi variables and determine the bounds, nominals and create maps for easier
        retrieving of the variables.
        """
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

        # Integers for disabling the HEX temperature constraints
        for hex in [
            *self.heat_network_components.get("heat_exchanger", []),
            *self.heat_network_components.get("heat_pump", []),
            *self.heat_network_components.get("heat_pump_elec", []),
        ]:
            disabeld_hex_var = f"{hex}__disabled"
            self.__disabled_hex_map[hex] = disabeld_hex_var
            self.__disabled_hex_var[disabeld_hex_var] = ca.MX.sym(disabeld_hex_var)
            self.__disabled_hex_var_bounds[disabeld_hex_var] = (0, 1.0)

        for p in self.heat_network_components.get("pipe", []):
            neighbour = self.has_related_pipe(p)
            if neighbour and p not in self.hot_pipes:
                flow_dir_var = f"{self.cold_to_hot_pipe(p)}__flow_direct_var"
            else:
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
                neighbour = self.has_related_pipe(p)
                if neighbour and p not in self.hot_pipes:
                    disconnected_var = f"{self.cold_to_hot_pipe(p)}__is_disconnected"
                else:
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

        # In case the user overrides the pipe class of the pipe with a single
        # pipe class we update the diameter/area parameters. If there is more
        # than a single pipe class for a certain pipe, we set the diameter
        # and area to NaN to prevent erroneous constraints.
        for _ in range(self.ensemble_size):
            self.__pipe_topo_diameter_area_parameters.append({})
            self.__pipe_topo_heat_loss_parameters.append({})

        for _carrier, temperatures in self.temperature_carriers().items():
            carrier_id_number_mapping = str(temperatures["id_number_mapping"])
            temp_var_name = carrier_id_number_mapping + "_temperature"
            self.__temperature_regime_var[temp_var_name] = ca.MX.sym(temp_var_name)
            temperature_regimes = self.temperature_regimes(int(carrier_id_number_mapping))
            if len(temperature_regimes) == 0:
                temperature = temperatures["temperature"]
                self.__temperature_regime_var_bounds[temp_var_name] = (temperature, temperature)
            elif len(temperature_regimes) == 1:
                temperature = temperature_regimes[0]
                self.__temperature_regime_var_bounds[temp_var_name] = (temperature, temperature)
            else:
                self.__temperature_regime_var_bounds[temp_var_name] = (
                    min(temperature_regimes),
                    max(temperature_regimes),
                )

            for temperature_regime in temperature_regimes:
                carrier_selected_var = carrier_id_number_mapping + f"_{temperature_regime}"
                self.__carrier_selected_var[carrier_selected_var] = ca.MX.sym(carrier_selected_var)
                self.__carrier_selected_var_bounds[carrier_selected_var] = (0.0, 1.0)

        for pipe in self.heat_network_components.get("pipe", []):

            # For similar reasons as for the diameter, we always make a heat
            # loss symbol, even if the heat loss is fixed. Note that we also
            # override the .Heat_loss parameter for cold pipes, even though
            # it is not actually used in the optimization problem.
            heat_loss_var_name = f"{pipe}__hn_heat_loss"
            carrier_id = parameters[f"{pipe}.carrier_id"]
            if len(self.temperature_regimes(carrier_id)) == 0:
                self.__pipe_topo_heat_loss_var[heat_loss_var_name] = ca.MX.sym(heat_loss_var_name)
            else:
                self.__pipe_topo_heat_loss_path_var[heat_loss_var_name] = ca.MX.sym(
                    heat_loss_var_name
                )
            self.__pipe_topo_heat_loss_map[pipe] = heat_loss_var_name

            if options["neglect_pipe_heat_losses"]:
                # No decision to make for this pipe w.r.t. heat loss
                self.__pipe_topo_heat_loss_var_bounds[heat_loss_var_name] = (
                    0.,
                    0.,
                )
                self.__pipe_topo_heat_loss_nominals[heat_loss_var_name] = max(
                    self._pipe_heat_loss({"neglect_pipe_heat_losses": False}, parameters, pipe),
                    1.0,
                )

                for ensemble_member in range(self.ensemble_size):
                    h = self.__pipe_topo_heat_loss_parameters[ensemble_member]
                    h[f"{pipe}.Heat_loss"] = self._pipe_heat_loss(options, parameters, pipe)
            else:
                heat_loss = self._pipe_heat_loss(options, parameters, pipe)
                self.__pipe_topo_heat_loss_var_bounds[heat_loss_var_name] = (
                    heat_loss,
                    heat_loss,
                )
                self.__pipe_topo_heat_loss_nominals[heat_loss_var_name] = max(
                    heat_loss,
                    1.0,
                )

                for ensemble_member in range(self.ensemble_size):
                    h = self.__pipe_topo_heat_loss_parameters[ensemble_member]
                    h[f"{pipe}.Heat_loss"] = heat_loss


        # Demand insulation link
        for dmnd in self.heat_network_components.get("demand", []):
            demand_insulation_classes = self.demand_insulation_classes(dmnd)
            if not demand_insulation_classes or len(demand_insulation_classes) == 1:
                # No insulation options availabe for the demands
                pass
            else:
                self.__demand_insulation_class_map[dmnd] = {}

                for insl in demand_insulation_classes:
                    if dmnd == insl.name_demand:
                        demand_insulation_class_var_name = (
                            f"{dmnd}__demand_insulation_class_{insl.name_insulation_level}"
                        )
                        if demand_insulation_class_var_name in (
                            self.__demand_insulation_class_map[dmnd].values()
                        ):
                            raise Exception(f"Resolve duplicate insulation: {insl}.")
                        self.__demand_insulation_class_map[dmnd][
                            insl
                        ] = demand_insulation_class_var_name
                        self.__demand_insulation_class_var[
                            demand_insulation_class_var_name
                        ] = ca.MX.sym(demand_insulation_class_var_name)
                        self.__demand_insulation_class_var_bounds[
                            demand_insulation_class_var_name
                        ] = (0.0, 1.0)

        # Check that buffer information is logical and
        # set the stored heat at t0 in the buffer(s) via bounds
        if len(self.times()) > 2:
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
        | ``include_demand_insulation_options``| ``bool``  | ``False``                   |
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

        The ``include_demand_insulation_options`` options is used, when insulations options per
        demand is specificied, to include heat demand and supply matching via constraints for all
        possible insulation options.
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
        options["include_demand_insulation_options"] = False

        return options

    def demand_insulation_classes(self, demand_insulation: str) -> List[DemandInsulationClass]:
        """
        If the returned List is:
        - empty: use the demand insualtion properties from the model
        - len() == 1: use these demand insualtion properties to overrule that of the model
        - len() > 1: decide between the demand insualtion class options.

        """
        return []

    def get_optimized_deman_insulation_class(self, demand_insulation: str) -> DemandInsulationClass:
        """
        Return the optimized demand_insulation class for a specific pipe. If no
        optimized demand insulation class is available (yet), a `KeyError` is returned.
        """
        return self.__demand_insulation_class_result[demand_insulation]

    @property
    def extra_variables(self):
        """
        In this function we add all the variables defined in the HeatMixin to the optimization
        problem. Note that these are only the normal variables not path variables.
        """
        variables = super().extra_variables.copy()
        variables.extend(self.__pipe_topo_heat_loss_var.values())
        return variables

    @property
    def path_variables(self):
        """
        In this function we add all the path variables defined in the HeatMixin to the
        optimization problem. Note that path_variables are variables that are created for each
        time-step.
        """
        variables = super().path_variables.copy()
        variables.extend(self.__flow_direct_var.values())
        variables.extend(self.__pipe_disconnect_var.values())
        variables.extend(self.__check_valve_status_var.values())
        variables.extend(self.__control_valve_direction_var.values())
        variables.extend(self.__demand_insulation_class_var.values())
        variables.extend(self.__temperature_regime_var.values())
        variables.extend(self.__carrier_selected_var.values())
        variables.extend(self.__disabled_hex_var.values())
        variables.extend(self.__pipe_topo_heat_loss_path_var.values())
        return variables

    def variable_is_discrete(self, variable):
        """
        All variables that only can take integer values should be added to this function.
        """
        if (
            variable in self.__flow_direct_var
            or variable in self.__pipe_disconnect_var
            or variable in self.__check_valve_status_var
            or variable in self.__control_valve_direction_var
            or variable in self.__demand_insulation_class_var
            or variable in self.__carrier_selected_var
            or variable in self.__disabled_hex_var
        ):
            return True
        else:
            return super().variable_is_discrete(variable)

    def variable_nominal(self, variable):
        """
        In this function we add all the nominals for the variables defined/added in the HeatMixin.
        """
        if variable in self.__pipe_topo_heat_loss_nominals:
            return self.__pipe_topo_heat_loss_nominals[variable]
        else:
            return super().variable_nominal(variable)

    def bounds(self):
        """
        In this function we add the bounds to the problem for all the variables defined/added in
        the HeatMixin.
        """
        bounds = super().bounds()
        bounds.update(self.__flow_direct_bounds)
        bounds.update(self.__pipe_disconnect_var_bounds)
        bounds.update(self.__check_valve_status_var_bounds)
        bounds.update(self.__control_valve_direction_var_bounds)
        bounds.update(self.__buffer_t0_bounds)
        bounds.update(self.__demand_insulation_class_var_bounds)
        bounds.update(self.__pipe_topo_heat_loss_var_bounds)
        bounds.update(self.__temperature_regime_var_bounds)
        bounds.update(self.__carrier_selected_var_bounds)
        bounds.update(self.__disabled_hex_var_bounds)
        return bounds

    def _pipe_heat_loss(
        self,
        options,
        parameters,
        p: str,
        u_values: Optional[Tuple[float, float]] = None,
        temp: float = None,
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

        neighbour = self.has_related_pipe(p)

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
            u_1, u_2 = heat_loss_u_values_pipe(**u_kwargs, neighbour=neighbour)
        else:
            u_1, u_2 = u_values

        length = parameters[f"{p}.length"]
        temperature = parameters[f"{p}.temperature"]
        if temp is not None:
            temperature = temp
        temperature_ground = parameters[f"{p}.T_ground"]
        if neighbour:
            if self.is_hot_pipe(p):
                dtemp = temperature - parameters[f"{self.hot_to_cold_pipe(p)}.temperature"]
            else:
                dtemp = temperature - parameters[f"{self.cold_to_hot_pipe(p)}.temperature"]
        else:
            dtemp = 0

        # if no return/supply pipes can be linked to eachother, the influence of the heat of the
        # neighbouring pipes can also not be determined and thus no influence is assumed
        # (distance between pipes to infinity)
        # This results in Rneighbour -> 0 and therefore u2->0, u1-> 1/(Rsoil+Rins)

        heat_loss = (
            length * (u_1 - u_2) * temperature
            - (length * (u_1 - u_2) * temperature_ground)
            + (length * u_2 * dtemp)
        )

        if heat_loss < 0:
            raise Exception(f"Heat loss of pipe {p} should be nonnegative.")

        return heat_loss

    def parameters(self, ensemble_member):
        """
        In this function we adapt the parameters object to avoid issues with accidentally using
        variables as constants.
        """
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

            for pipe in components.get("pipe", []):
                try:
                    pipe_classes = self.__pipe_topo_pipe_class_map[pipe].keys()
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
        """
        In this function we force the buffer at t0 to have a certain amount of set energy in it.
        We do this via the bounds, by providing the bounds with a time-series where the first
        element is the initial heat in the buffer.
        """
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

    def __heat_matching_demand_insulation_constraints(self, ensemble_member):
        """
        Consider all possible heat demand insulation options (each options has an assosiated unique
        demand profile for the specific demand) and add constraints such that only one insulation
        options is activated per demand. The constraints will then ensure aim to match the supply
        and demand.

        Note:
        - This function is only active when the "include_demand_insulation_options" (False by
        default) has been set to True in the heat network options.
        - Currently this functional requires that all demands have at least one insualtion option is
        specified for every demand in the heat network.
        """

        constraints = []
        for dmnd in self.heat_network_components["demand"]:
            heat_demand = self.__state_vector_scaled(f"{dmnd}.Heat_demand", ensemble_member)
            target_demand = self.get_timeseries(f"{dmnd}.target_heat_demand")

            try:
                demand_insulation_classes = self.__demand_insulation_class_map[dmnd]
                is_insulation_active = []
                demand_profile_for_this_class = []
                big_m = []  # big_m for every insulation class
                nominal = []
                for pc, pc_var_name in demand_insulation_classes.items():
                    # Create a demand profile for every insulation level option for this demand
                    demand_profile_for_this_class.append(
                        pc.demand_scaling_factor * target_demand.values[: (len(self.times()))]
                    )
                    # There might be a large differnece between profiles of different insulation
                    # classes. Therefor create variables below per profile (per insulation class)
                    big_m.append(2.0 * max(demand_profile_for_this_class[-1]))
                    nominal.append(np.median(demand_profile_for_this_class[-1]))

                    # Create integer variable to activated/deactivate (1/0) a demand insulation
                    is_insulation_active_var = self.extra_variable(pc_var_name, ensemble_member)
                    # Demand insulation activation variable for each time step of demand profile
                    is_insulation_active.append(
                        ca.repmat(
                            is_insulation_active_var,
                            len(target_demand.values[: (len(self.times()))]),
                        )
                    )
                # Add constraint to enforce that only 1 demand insulation can be active
                # per time step for a specific demand
                for itstep in range(len(target_demand.values[: (len(self.times()))])):
                    is_insulation_active_sum_per_timestep = 0
                    for iclasses in range(len(demand_profile_for_this_class)):
                        is_insulation_active_sum_per_timestep = (
                            is_insulation_active_sum_per_timestep
                            + is_insulation_active[iclasses][itstep]
                        )
                    constraints.append((is_insulation_active_sum_per_timestep, 1.0, 1.0))

                # Adding constraints for the entire time horizon per demand insulation
                for iclasses in range(len(demand_profile_for_this_class)):
                    for itstep in range(len(demand_profile_for_this_class[iclasses])):
                        constraints.append(
                            (
                                (
                                    heat_demand[itstep]
                                    - demand_profile_for_this_class[iclasses][itstep]
                                    + big_m[iclasses]
                                    * (1.0 - is_insulation_active[iclasses][itstep])
                                )
                                / nominal[iclasses],
                                0.0,
                                np.inf,
                            )
                        )
                        constraints.append(
                            (
                                (
                                    heat_demand[itstep]
                                    - demand_profile_for_this_class[iclasses][itstep]
                                    - big_m[iclasses]
                                    * (1.0 - is_insulation_active[iclasses][itstep])
                                )
                                / nominal[iclasses],
                                -np.inf,
                                0.0,
                            )
                        )
            except KeyError:
                raise Exception(
                    "Add a DemandInsulationClass with a demand_scaling_factor value =1.0 for "
                    "demand: {dmnd}"
                )
        return constraints

    def __pipe_rate_heat_change_constraints(self, ensemble_member):
        """
        To avoid sudden change in heat from a timestep to the next,
        constraints on d(Heat)/dt are introduced.
        Information of restrictions on dQ/dt and dT/dt are used, as d(Heat)/dt is
        proportional to average_temperature * dQ/dt + average_discharge * dT/dt.
        The average discharge is computed using the assumption that the average velocity is 1 m/s.
        """
        constraints = []

        parameters = self.parameters(ensemble_member)
        hn_options = self.heat_network_options()

        t_change = hn_options["maximum_temperature_der"]
        q_change = hn_options["maximum_flow_der"]

        if np.isfinite(t_change) and np.isfinite(q_change):
            assert (
                not self.__pipe_topo_pipe_class_map
            ), "heat rate change constraints not allowed with topology optimization"

        for p in self.heat_network_components.get("pipe", []):
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
        """
        This function adds constraints for each heat network node/joint to have as much
        thermal power (Heat variable) going in as out. Effectively, it is setting the sum of
        thermal powers to zero.
        """
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
        """
        This function adds constraints to ensure that the incoming volumetric flow equals the
        outgoing volumetric flow. We assume constant density throughout a hydraulically coupled
        system and thus these constraints are needed for mass conservation.
        """
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
        """
        This function adds the constraints to subtract the heat losses from the thermal power that
        propagates through the network. Note that this is done only on the thermal power, as such
        only energy losses are accounted for, temperature losses are not considered.

        There are a few cases for the heat loss constraints
        - Heat losses are constant: This is the case when the pipe class is constant and the
        network temperature is constant. In this case the symbol for the heat-loss is fixed by its
        lower and upper bound to a value.
        - Heat losses depend on pipe_class: In this case the heat loss depend on the pipe class
        selected by the optimization. In this case the heat losses can vary due to the varying
        radiation surface and different insulation materials applied to the pipe. Note that the
        influences of varying pipe class are taken into account while setting the heat-loss
        variable in topology constraints.
        - Heat losses depend on varying network temperature: In this case the heat loss varies due
        to the different delta temperature with ambient. Note that the heat loss symbol does not
        account for the varying temperature. Therefore, the big_m formulation is needed in these
        constraints.
        - Heat losses depend both on varying network temperature and pipe classes: In this case
        both pipe class and delta temperature with ambient vary
        - neglect_pipe_heat_losses:
        """
        constraints = []
        options = self.heat_network_options()

        for p in self.heat_network_components.get("pipe", []):
            heat_in = self.state(f"{p}.HeatIn.Heat")
            heat_out = self.state(f"{p}.HeatOut.Heat")
            heat_nominal = self.variable_nominal(f"{p}.HeatIn.Heat")

            big_m = 2.0 * np.max(
                np.abs(
                    (
                        *self.bounds()[f"{p}.HeatIn.Heat"],
                        *self.bounds()[f"{p}.HeatOut.Heat"],
                    )
                )
            )

            is_disconnected_var = self.__pipe_disconnect_map.get(p)

            if is_disconnected_var is None:
                is_disconnected = 0.0
            else:
                is_disconnected = self.state(is_disconnected_var)

            if p in self.__pipe_topo_heat_loss_map:
                # Heat loss is variable depending on pipe class
                heat_loss_sym_name = self.__pipe_topo_heat_loss_map[p]
                try:
                    heat_loss = self.variable(heat_loss_sym_name)
                except KeyError:
                    heat_loss = self.__pipe_topo_heat_loss_path_var[heat_loss_sym_name]
                heat_loss_nominal = self.variable_nominal(heat_loss_sym_name)
                constraint_nominal = (heat_nominal * heat_loss_nominal) ** 0.5

                if options["heat_loss_disconnected_pipe"]:
                    constraints.append(
                        (
                            (heat_in - heat_out - heat_loss) / constraint_nominal,
                            0.0,
                            0.0,
                        )
                    )
                else:
                    # Force heat loss to `heat_loss` when pipe is connected, and zero otherwise.
                    heat_loss_nominal = self.__pipe_topo_heat_loss_nominals[heat_loss_sym_name]
                    constraint_nominal = (big_m * heat_loss_nominal) ** 0.5

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
        return constraints

    @staticmethod
    def __get_abs_max_bounds(*bounds):
        """
        This function returns the absolute maximum of the bounds given. Note that bounds can also be
        a timeseries.
        """
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
        """
        This function adds constraints to set the direction in pipes and determine whether a pipe
        is utilized at all (is_disconnected variable).

        Whether a pipe is connected is based upon whether flow passes through that pipe.

        The directions are set based upon the directions of how thermal power propegates. This is
        done based upon the sign of the Heat variable. Where positive Heat means a positive
        direction and negative heat means a negative direction. By default, positive is defined from
        HeatIn to HeatOut.

        Finally, a minimum flow can be set. This can sometimes be useful for numerical stability.
        """
        constraints = []
        options = self.heat_network_options()
        parameters = self.parameters(ensemble_member)

        minimum_velocity = options["minimum_velocity"]
        maximum_velocity = options["maximum_velocity"]

        # Also ensure that the discharge has the same sign as the heat.
        for p in self.heat_network_components.get("pipe", []):
            flow_dir_var = self.__pipe_to_flow_direct_map[p]
            flow_dir = self.state(flow_dir_var)

            is_disconnected_var = self.__pipe_disconnect_map.get(p)

            if is_disconnected_var is None:
                is_disconnected = 0.0
            else:
                is_disconnected = self.state(is_disconnected_var)

            q_pipe = self.state(f"{p}.Q")
            heat_in = self.state(f"{p}.HeatIn.Heat")
            heat_out = self.state(f"{p}.HeatOut.Heat")

            try:
                pipe_classes = self.__pipe_topo_pipe_class_map[p].keys()
                maximum_discharge = max([c.maximum_discharge for c in pipe_classes])
                var_names = self.__pipe_topo_pipe_class_map[p].values()
                dn_none = self.__pipe_topo_pipe_class_var[list(var_names)[0]]
                minimum_discharge = min(
                    [c.area * minimum_velocity for c in pipe_classes if c.area > 0.0]
                )
            except KeyError:
                maximum_discharge = maximum_velocity * parameters[f"{p}.area"]

                if math.isfinite(minimum_velocity) and minimum_velocity > 0.0:
                    minimum_discharge = minimum_velocity * parameters[f"{p}.area"]
                else:
                    minimum_discharge = 0.0
                dn_none = 0.0

            if maximum_discharge == 0.0:
                maximum_discharge = 1.0
            big_m = 2.0 * (maximum_discharge + minimum_discharge)

            if minimum_discharge > 0.0:
                constraint_nominal = (minimum_discharge * big_m) ** 0.5
            else:
                constraint_nominal = big_m

            # when DN=0 the flow_dir variable can be 0 or 1, thus these constraints then need to be
            # disabled
            constraints.append(
                (
                    (
                        q_pipe
                        - big_m * (flow_dir + dn_none)
                        + (1 - is_disconnected) * minimum_discharge
                    )
                    / constraint_nominal,
                    -np.inf,
                    0.0,
                )
            )
            constraints.append(
                (
                    (
                        q_pipe
                        + big_m * (1 - flow_dir + dn_none)
                        - (1 - is_disconnected) * minimum_discharge
                    )
                    / constraint_nominal,
                    0.0,
                    np.inf,
                )
            )
            big_m = 2.0 * np.max(
                np.abs(
                    (
                        *self.bounds()[f"{p}.HeatIn.Heat"],
                        *self.bounds()[f"{p}.HeatOut.Heat"],
                    )
                )
            )
            # Note we only need one on the heat as the desired behaviour is propegated by the
            # constraints heat_in - heat_out - heat_loss == 0.
            constraints.append(
                (
                    (heat_in - big_m * flow_dir) / big_m,
                    -np.inf,
                    0.0,
                )
            )
            constraints.append(
                (
                    (heat_in + big_m * (1 - flow_dir)) / big_m,
                    0.0,
                    np.inf,
                )
            )

            # If a pipe is disconnected, the discharge should be zero
            if is_disconnected_var is not None:
                big_m = 2.0 * (maximum_discharge + minimum_discharge)
                constraints.append(((q_pipe - (1 - is_disconnected) * big_m) / big_m, -np.inf, 0.0))
                constraints.append(((q_pipe + (1 - is_disconnected) * big_m) / big_m, 0.0, np.inf))
                big_m = 2.0 * np.max(
                    np.abs(
                        (
                            *self.bounds()[f"{p}.HeatIn.Heat"],
                            *self.bounds()[f"{p}.HeatOut.Heat"],
                        )
                    )
                )
                constraints.append(
                    ((heat_in - (1 - is_disconnected) * big_m) / big_m, -np.inf, 0.0)
                )
                constraints.append(((heat_in + (1 - is_disconnected) * big_m) / big_m, 0.0, np.inf))
                constraints.append(
                    ((heat_out - (1 - is_disconnected) * big_m) / big_m, -np.inf, 0.0)
                )
                constraints.append(
                    ((heat_out + (1 - is_disconnected) * big_m) / big_m, 0.0, np.inf)
                )

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

    def __demand_heat_to_discharge_path_constraints(self, ensemble_member):
        """
        This function adds constraints linking the flow to the thermal power at the demand assets.
        We use an equality constraint on the outgoing flow for every non-pipe asset. Meaning that we
        equate Q * rho * cp * T == Heat for outgoing flows, and inequalities for the heat carried
        in the pipes. This means that the heat can decrease in the network to compensate losses,
        but that the losses and thus flow will always be over-estimated with the temperature for
        which no temperature drops are modelled.
        """
        constraints = []
        parameters = self.parameters(ensemble_member)

        for d in self.heat_network_components.get("demand", []):
            heat_nominal = parameters[f"{d}.Heat_nominal"]
            cp = parameters[f"{d}.cp"]
            rho = parameters[f"{d}.rho"]
            # TODO: future work - some sort of correction factor to account for temp drop n pipe:
            # (maximum/average lenght fromm source to demand) * V_nominal * temperature_loss_factor
            discharge = self.state(f"{d}.Q")
            heat_out = self.state(f"{d}.HeatOut.Heat")

            ret_carrier = parameters[f"{d}.T_return_id"]
            return_temperatures = self.temperature_regimes(ret_carrier)
            big_m = 2.0 * self.bounds()[f"{d}.HeatOut.Heat"][1]

            if len(return_temperatures) == 0:
                constraints.append(
                    (
                        (heat_out - discharge * cp * rho * parameters[f"{d}.T_return"])
                        / heat_nominal,
                        0.0,
                        0.0,
                    )
                )
            else:
                for return_temperature in return_temperatures:
                    ret_temperature_is_selected = self.state(f"{ret_carrier}_{return_temperature}")
                    constraints.append(
                        (
                            (
                                heat_out
                                - discharge * cp * rho * return_temperature
                                + (1.0 - ret_temperature_is_selected) * big_m
                            )
                            / heat_nominal,
                            0.0,
                            np.inf,
                        )
                    )
                    constraints.append(
                        (
                            (
                                heat_out
                                - discharge * cp * rho * return_temperature
                                - (1.0 - ret_temperature_is_selected) * big_m
                            )
                            / heat_nominal,
                            -np.inf,
                            0.0,
                        )
                    )

        return constraints

    def __source_heat_to_discharge_path_constraints(self, ensemble_member):
        """
        This function adds constraints linking the flow to the thermal power at the demand assets.
        We use an equality constraint on the outgoing flow for every non-pipe asset. Meaning that we
        equate Q * rho * cp * T == Heat for outgoing flows, and inequalities for the heat carried
        in the pipes. This means that the heat can decrease in the network to compensate losses,
        but that the losses and thus flow will always be over-estimated with the temperature for
        which no temperature drops are modelled.
        """
        constraints = []
        parameters = self.parameters(ensemble_member)

        for s in self.heat_network_components.get("source", []):
            heat_nominal = parameters[f"{s}.Heat_nominal"]
            q_nominal = self.variable_nominal(f"{s}.Q")
            cp = parameters[f"{s}.cp"]
            rho = parameters[f"{s}.rho"]
            dt = parameters[f"{s}.dT"]

            discharge = self.state(f"{s}.Q")
            heat_out = self.state(f"{s}.HeatOut.Heat")

            constraint_nominal = (heat_nominal * cp * rho * dt * q_nominal) ** 0.5

            sup_carrier = parameters[f"{s}.T_supply_id"]
            supply_temperatures = self.temperature_regimes(sup_carrier)
            big_m = 2.0 * self.bounds()[f"{s}.HeatOut.Heat"][1]

            if len(supply_temperatures) == 0:
                constraints.append(
                    (
                        (heat_out - discharge * cp * rho * parameters[f"{s}.T_supply"])
                        / heat_nominal,
                        0.0,
                        0.0,
                    )
                )
            else:
                for supply_temperature in supply_temperatures:
                    sup_temperature_is_selected = self.state(f"{sup_carrier}_{supply_temperature}")

                    constraints.append(
                        (
                            (
                                heat_out
                                - discharge * cp * rho * supply_temperature
                                + (1.0 - sup_temperature_is_selected) * big_m
                            )
                            / constraint_nominal,
                            0.0,
                            np.inf,
                        )
                    )
                    constraints.append(
                        (
                            (
                                heat_out
                                - discharge * cp * rho * supply_temperature
                                - (1.0 - sup_temperature_is_selected) * big_m
                            )
                            / constraint_nominal,
                            -np.inf,
                            0.0,
                        )
                    )

        return constraints

    def __pipe_hydraulic_power_path_constraints(self, ensemble_member):
        """
        This function adds constraints to compute the hydraulic power that is needed to realize the
        flow, compensating the pressure drop through the pipe. Similar to the head loss constraints
        we allow two supported methods. 1) a single linear line between 0 to max velocity. 2) A
        multiple line inequality approach where one can use the minimize_head_losses == True option
        to drag down the solution to the actual physical solution.

        Note that the linearizations are made separately from the pressure drop constraints, this is
        done to avoid "stacked" overestimations.
        """
        constraints = []
        options = self.heat_network_options()

        if options["head_loss_option"] != HeadLossOption.NO_HEADLOSS:
            parameters = self.parameters(ensemble_member)
            components = self.heat_network_components

            for pipe in components.get("pipe", []):
                if parameters[f"{pipe}.length"] == 0.0:
                    # If the pipe does not have a control valve, the head loss is
                    # forced to zero via bounds. If the pipe _does_ have a control
                    # valve, then there still is no relationship between the
                    # discharge and the hydraulic_power.
                    continue

                head_loss_option = self._hn_get_pipe_head_loss_option(pipe, options, parameters)
                assert (
                    head_loss_option != HeadLossOption.NO_HEADLOSS
                ), "This method should be skipped when NO_HEADLOSS is set."

                discharge = self.state(f"{pipe}.Q")
                hydraulic_power = self.state(f"{pipe}.Hydraulic_power")
                rho = parameters[f"{pipe}.rho"]

                # 0: pipe is connected, 1: pipe is disconnected
                is_disconnected_var = self.__pipe_disconnect_map.get(pipe)
                if is_disconnected_var is None:
                    is_disconnected = 0.0
                else:
                    is_disconnected = self.state(is_disconnected_var)

                flow_dir_var = self.__pipe_to_flow_direct_map[pipe]
                flow_dir = self.state(flow_dir_var)  # 0/1: negative/positive flow direction

                if pipe in self.__pipe_topo_pipe_class_map:
                    # Multiple diameter options for this pipe
                    pipe_classes = self.__pipe_topo_pipe_class_map[pipe]
                    max_discharge = max(c.maximum_discharge for c in pipe_classes)
                    for pc, pc_var_name in pipe_classes.items():
                        if pc.inner_diameter == 0.0:
                            continue

                        # Calc max hydraulic power based on maximum_total_head_loss =
                        # f(max_sum_dh_pipes, max_dh_network_options)
                        max_total_hydraulic_power = 2.0 * (
                            rho
                            * GRAVITATIONAL_CONSTANT
                            * self.__maximum_total_head_loss
                            * max_discharge
                        )

                        # is_topo_disconnected - 0: pipe selected, 1: pipe disconnected/not selected
                        # self.__pipe_topo_pipe_class_var - value 0: pipe is not selected, 1: pipe
                        # is selected
                        is_topo_disconnected = 1 - self.__pipe_topo_pipe_class_var[pc_var_name]

                        constraints.extend(
                            self._hydraulic_power(
                                pipe,
                                options,
                                parameters,
                                discharge,
                                hydraulic_power,
                                is_disconnected=is_topo_disconnected + is_disconnected,
                                big_m=max_total_hydraulic_power,
                                pipe_class=pc,
                                flow_dir=flow_dir,
                            )
                        )
                else:
                    is_topo_disconnected = int(parameters[f"{pipe}.diameter"] == 0.0)
                    max_total_hydraulic_power = 2.0 * (
                        rho
                        * GRAVITATIONAL_CONSTANT
                        * self.__maximum_total_head_loss
                        * parameters[f"{pipe}.area"]
                        * options["maximum_velocity"]
                    )
                    constraints.extend(
                        self._hydraulic_power(
                            pipe,
                            options,
                            parameters,
                            discharge,
                            hydraulic_power,
                            is_disconnected=is_disconnected + is_topo_disconnected,
                            big_m=max_total_hydraulic_power,
                            flow_dir=flow_dir,
                        )
                    )
        return constraints

    def __pipe_heat_to_discharge_path_constraints(self, ensemble_member):
        """
        This function adds constraints linking the flow to the thermal power at the pipe assets.
        We use an equality constraint on the outgoing flow for every non-pipe asset. Meaning that we
        equate Q * rho * cp * T == Heat for outgoing flows, and inequalities for the heat carried
        in the pipes. This means that the heat can decrease in the network to compensate losses,
        but that the losses and thus flow will always be over-estimated with the temperature for
        which no temperature drops are modelled.

        There are three cases for the constraint, namely:
        - no heat losses: In this case a single equality constraint can be used.
        - constant network temperature: In this case there is a single set inequality constraints
        - varying network temperature: In this case a set of big_m constraints is used to
        "activate" only the constraints with the selected network temperature.
        """
        constraints = []
        parameters = self.parameters(ensemble_member)

        sum_heat_losses = 0.0

        for p in self.heat_network_components.get("pipe", []):
            if p in self.__pipe_topo_heat_losses:
                sum_heat_losses += max(self.__pipe_topo_heat_losses[p])
            else:
                sum_heat_losses += parameters[f"{p}.Heat_loss"]

        assert not np.isnan(sum_heat_losses)

        for p in self.heat_network_components.get("pipe", []):
            cp = parameters[f"{p}.cp"]
            rho = parameters[f"{p}.rho"]
            temp = parameters[f"{p}.temperature"]

            flow_dir_var = self.__pipe_to_flow_direct_map[p]
            flow_dir = self.state(flow_dir_var)
            scaled_heat_in = self.state(f"{p}.HeatIn.Heat")  # * heat_to_discharge_fac
            scaled_heat_out = self.state(f"{p}.HeatOut.Heat")  # * heat_to_discharge_fac
            pipe_q = self.state(f"{p}.Q")
            heat_nominal = self.variable_nominal(f"{p}.HeatIn.Heat")

            # We do not want Big M to be too tight in this case, as it results
            # in a rather hard yes/no constraint as far as feasibility on e.g.
            # a single source system is concerned. Use a factor of 2 to give
            # some slack.
            # big_m = 2 * sum_heat_losses * heat_to_discharge_fac
            big_m = 2.0 * np.max(
                np.abs((*self.bounds()[f"{p}.HeatIn.Heat"], *self.bounds()[f"{p}.HeatOut.Heat"]))
            )

            for heat in [scaled_heat_in, scaled_heat_out]:
                if self.heat_network_options()["neglect_pipe_heat_losses"]:
                    constraints.append(
                        (
                            (heat - pipe_q * (cp * rho * temp)) / heat_nominal,
                            0.0,
                            0.0,
                        )
                    )
                else:
                    assert big_m > 0.0

                    carrier = parameters[f"{p}.carrier_id"]
                    temperatures = self.temperature_regimes(carrier)
                    if len(temperatures) == 0:
                        constraints.append(
                            (
                                (heat - pipe_q * (cp * rho * temp) - big_m * (1 - flow_dir))
                                / big_m,
                                -np.inf,
                                0.0,
                            )
                        )
                        constraints.append(
                            (
                                (heat - pipe_q * (cp * rho * temp) + big_m * flow_dir) / big_m,
                                0.0,
                                np.inf,
                            )
                        )
                    elif len(temperatures) > 0:
                        for temperature in temperatures:
                            temperature_is_selected = self.state(f"{carrier}_{temperature}")
                            constraints.append(
                                (
                                    (
                                        heat
                                        - pipe_q * (cp * rho * temperature)
                                        + big_m * flow_dir
                                        + (1.0 - temperature_is_selected) * big_m
                                    )
                                    / big_m,
                                    0.0,
                                    np.inf,
                                )
                            )
                            constraints.append(
                                (
                                    (
                                        heat
                                        - pipe_q * (cp * rho * temperature)
                                        - big_m * (1 - flow_dir)
                                        - (1.0 - temperature_is_selected) * big_m
                                    )
                                    / big_m,
                                    -np.inf,
                                    0.0,
                                )
                            )
        return constraints

    def __storage_heat_to_discharge_path_constraints(self, ensemble_member):
        """
        This function adds constraints linking the flow to the thermal power at the pipe assets.
        We use an equality constraint on the outgoing flow for every non-pipe asset. Meaning that we
        equate Q * rho * cp * T == Heat for outgoing flows, and inequalities for the heat carried
        in the pipes. This means that the heat can decrease in the network to compensate losses,
        but that the losses and thus flow will always be over-estimated with the temperature for
        which no temperature drops are modelled.

        This function adds the constraints relating the discharge to the thermal power at the
        buffer component. This is done following the philosophy described at the heat_to_discharge
        for sources/demands. Where a big_m formulation is used to switch between source and demand
        logic depending on whether the buffer is discharging or charging. For this purpose the
        direction of the connecting supply pipe is used, where a positive direction is charging
        (demand logic) and a negative direction is discharging (source logic). This also means that
        buffers can only be connected with the supply pipe going in positive direction to the
        buffer.
        """
        constraints = []
        parameters = self.parameters(ensemble_member)

        for b, (
            (hot_pipe, _hot_pipe_orientation),
            (_cold_pipe, _cold_pipe_orientation),
        ) in {**self.heat_network_topology.buffers, **self.heat_network_topology.ates}.items():
            heat_nominal = parameters[f"{b}.Heat_nominal"]
            q_nominal = self.variable_nominal(f"{b}.Q")
            cp = parameters[f"{b}.cp"]
            rho = parameters[f"{b}.rho"]
            dt = parameters[f"{b}.dT"]

            discharge = self.state(f"{b}.HeatIn.Q")
            # Note that `heat_hot` can be negative for the buffer; in that case we
            # are extracting heat from it.
            heat_out = self.state(f"{b}.HeatOut.Heat")
            heat_in = self.state(f"{b}.HeatIn.Heat")

            # We want an _equality_ constraint between discharge and heat if the buffer is
            # consuming (i.e. behaving like a "demand"). We want an _inequality_
            # constraint (`|heat| >= |f(Q)|`) just like a "source" component if heat is
            # extracted from the buffer. We accomplish this by disabling one of
            # the constraints with a boolean. Note that `discharge` and `heat_hot`
            # are guaranteed to have the same sign.
            flow_dir_var = self.__pipe_to_flow_direct_map[hot_pipe]
            is_buffer_charging = self.state(flow_dir_var)

            big_m = 2.0 * np.max(
                np.abs((*self.bounds()[f"{b}.HeatIn.Heat"], *self.bounds()[f"{b}.HeatOut.Heat"]))
            )

            sup_carrier = parameters[f"{b}.T_supply_id"]
            ret_carrier = parameters[f"{b}.T_return_id"]
            supply_temperatures = self.temperature_regimes(sup_carrier)
            return_temperatures = self.temperature_regimes(ret_carrier)

            if len(supply_temperatures) == 0:
                constraint_nominal = (heat_nominal * cp * rho * dt * q_nominal) ** 0.5
                # only when discharging the heat_in should match the heat excactly (like producer)
                constraints.append(
                    (
                        (
                            heat_in
                            - discharge * cp * rho * parameters[f"{b}.T_supply"]
                            + is_buffer_charging * big_m
                        )
                        / constraint_nominal,
                        0.0,
                        np.inf,
                    )
                )
                constraints.append(
                    (
                        (heat_in - discharge * cp * rho * parameters[f"{b}.T_supply"])
                        / constraint_nominal,
                        -np.inf,
                        0.0,
                    )
                )
            else:
                for supply_temperature in supply_temperatures:
                    sup_temperature_is_selected = self.state(f"{sup_carrier}_{supply_temperature}")
                    constraint_nominal = (
                        heat_nominal * cp * rho * supply_temperature * q_nominal
                    ) ** 0.5
                    constraints.append(
                        (
                            (
                                heat_in
                                - discharge * cp * rho * supply_temperature
                                + (1.0 - sup_temperature_is_selected) * big_m
                                + is_buffer_charging * big_m
                            )
                            / constraint_nominal,
                            0.0,
                            np.inf,
                        )
                    )
                    constraints.append(
                        (
                            (
                                heat_in
                                - discharge * cp * rho * supply_temperature
                                - (1.0 - sup_temperature_is_selected) * big_m
                            )
                            / constraint_nominal,
                            -np.inf,
                            0.0,
                        )
                    )

            if len(return_temperatures) == 0:
                constraint_nominal = (heat_nominal * cp * rho * dt * q_nominal) ** 0.5
                # Only when consuming/charging the heatout should match the q rho cp Tret
                constraints.append(
                    (
                        (heat_out - discharge * cp * rho * parameters[f"{b}.T_return"])
                        / constraint_nominal,
                        0.0,
                        np.inf,
                    )
                )
                constraints.append(
                    (
                        (
                            heat_out
                            - discharge * cp * rho * parameters[f"{b}.T_return"]
                            - (1.0 - is_buffer_charging) * big_m
                        )
                        / constraint_nominal,
                        -np.inf,
                        0.0,
                    )
                )
            else:
                for return_temperature in return_temperatures:
                    ret_temperature_is_selected = self.state(f"{ret_carrier}_{return_temperature}")
                    constraint_nominal = (
                        heat_nominal * cp * rho * return_temperature * q_nominal
                    ) ** 0.5
                    constraints.append(
                        (
                            (
                                heat_out
                                - discharge * cp * rho * return_temperature
                                + (1.0 - ret_temperature_is_selected) * big_m
                            )
                            / constraint_nominal,
                            0.0,
                            np.inf,
                        )
                    )
                    constraints.append(
                        (
                            (
                                heat_out
                                - discharge * cp * rho * return_temperature
                                - (2.0 - ret_temperature_is_selected - is_buffer_charging) * big_m
                            )
                            / constraint_nominal,
                            -np.inf,
                            0.0,
                        )
                    )

        return constraints

    def __network_temperature_path_constraints(self, ensemble_member):
        """
        This function adds constraints to ensure that only one temperature level is active for the
        supply and return network within one hydraulically coupled system. Furthermore, it sets the
        temperature variable to the temperature associated with the temperature of the integer
        variable.
        """
        constraints = []

        for _carrier, temperatures in self.temperature_carriers().items():
            number = temperatures["id_number_mapping"]
            sum = 0.0
            temperature_regimes = self.temperature_regimes(int(number))
            for temperature in temperature_regimes:
                temp_selected = self.state(f"{int(number)}_{temperature}")
                sum += temp_selected
                temperature_var = self.state(f"{int(number)}_temperature")
                big_m = 2.0 * self.bounds()[f"{int(number)}_temperature"][1]
                # Constraints for setting the temperature variable to the chosen temperature
                constraints.append(
                    (temperature - temperature_var + (1.0 - temp_selected) * big_m, 0.0, np.inf)
                )
                constraints.append(
                    (temperature - temperature_var - (1.0 - temp_selected) * big_m, -np.inf, 0.0)
                )
            if len(temperature_regimes) > 0:
                # Constraint to ensure that one single temperature is chosen for every timestep
                constraints.append((sum, 1.0, 1.0))

        return constraints

    def __heat_exchanger_heat_to_discharge_path_constraints(self, ensemble_member):
        """
        This function adds the heat to discharge constraints of components that connect two
        hydraulically decoupled networks. We assume that there is a dedicated primary side and
        secondary side and that Thermal power can only flow from primary to secondary.

        Following this assumption we use the demand logic to relate heat to discharge at the
        primary side and the source logic for relating heat to discharge at the secondary side.

        This function also adds constraints to ensure physically logical temperatures between the
        primary and secondary side when varying temperature is applied to the optimization.
        Assuming a counter flow heat exchanger, this means that the secondary supply temperature
        will always be below the primary supply temperature and that the primary return temperature
        has to be above the secondary return temperature.

        Finally, an is disabled variable is set for when the heat exchanger is not used. This is
        needed to allow disabling of the HEX temperature constraints when no heat is flowing
        through the HEX.
        """
        constraints = []
        parameters = self.parameters(ensemble_member)

        # The primary side of the heat exchanger acts like a heat consumer, and the secondary side
        # acts as a heat producer. Essentially using equality constraints to set the heat leaving
        # the secondary side based on the secondary Supply temperature and the heat leaving the
        # primary side based on the primary Return temperature.

        for heat_exchanger in [
            *self.heat_network_components.get("heat_exchanger", []),
            *self.heat_network_components.get("heat_pump", []),
            *self.heat_network_components.get("heat_pump_elec", []),
        ]:
            cp_prim = parameters[f"{heat_exchanger}.Primary.cp"]
            rho_prim = parameters[f"{heat_exchanger}.Primary.rho"]
            cp_sec = parameters[f"{heat_exchanger}.Secondary.cp"]
            rho_sec = parameters[f"{heat_exchanger}.Secondary.rho"]
            dt_prim = parameters[f"{heat_exchanger}.Primary.dT"]
            dt_sec = parameters[f"{heat_exchanger}.Secondary.dT"]
            discharge_primary = self.state(f"{heat_exchanger}.Primary.HeatIn.Q")
            discharge_secondary = self.state(f"{heat_exchanger}.Secondary.HeatOut.Q")
            heat_primary = self.state(f"{heat_exchanger}.Primary_heat")
            heat_out_prim = self.state(f"{heat_exchanger}.Primary.HeatOut.Heat")
            heat_out_sec = self.state(f"{heat_exchanger}.Secondary.HeatOut.Heat")

            constraint_nominal = (
                cp_prim
                * rho_prim
                * dt_prim
                * self.variable_nominal(f"{heat_exchanger}.Primary.HeatIn.Q")
            )

            sup_carrier_prim = parameters[f"{heat_exchanger}.Primary.T_supply_id"]
            ret_carrier_prim = parameters[f"{heat_exchanger}.Primary.T_return_id"]

            supply_temperatures_prim = self.temperature_regimes(sup_carrier_prim)
            return_temperatures_prim = self.temperature_regimes(ret_carrier_prim)

            big_m = 2.0 * self.bounds()[f"{heat_exchanger}.Primary.HeatOut.Heat"][1]

            # primary side
            if len(return_temperatures_prim) == 0:
                constraints.append(
                    (
                        (
                            heat_out_prim
                            - discharge_primary
                            * cp_prim
                            * rho_prim
                            * parameters[f"{heat_exchanger}.Primary.T_return"]
                        )
                        / constraint_nominal,
                        0.0,
                        0.0,
                    )
                )
            else:
                for return_temperature in return_temperatures_prim:
                    ret_temperature_is_selected = self.state(
                        f"{ret_carrier_prim}_{return_temperature}"
                    )
                    constraints.append(
                        (
                            (
                                heat_out_prim
                                - discharge_primary * cp_prim * rho_prim * return_temperature
                                + (1.0 - ret_temperature_is_selected) * big_m
                            )
                            / constraint_nominal,
                            0.0,
                            np.inf,
                        )
                    )
                    constraints.append(
                        (
                            (
                                heat_out_prim
                                - discharge_primary * cp_prim * rho_prim * return_temperature
                                - (1.0 - ret_temperature_is_selected) * big_m
                            )
                            / constraint_nominal,
                            -np.inf,
                            0.0,
                        )
                    )

            # Secondary side
            sup_carrier_sec = parameters[f"{heat_exchanger}.Secondary.T_supply_id"]
            ret_carrier_sec = parameters[f"{heat_exchanger}.Secondary.T_return_id"]

            supply_temperatures_sec = self.temperature_regimes(sup_carrier_sec)
            return_temperatures_sec = self.temperature_regimes(ret_carrier_sec)
            big_m = 2.0 * self.bounds()[f"{heat_exchanger}.Secondary.HeatOut.Heat"][1]
            constraint_nominal = (
                cp_sec * rho_sec * dt_sec * self.bounds()[f"{heat_exchanger}.Secondary.HeatIn.Q"][1]
            )

            if len(supply_temperatures_sec) == 0:
                constraints.append(
                    (
                        (
                            heat_out_sec
                            - discharge_secondary
                            * cp_sec
                            * rho_sec
                            * parameters[f"{heat_exchanger}.Secondary.T_supply"]
                        )
                        / constraint_nominal,
                        0.0,
                        0.0,
                    )
                )
            else:
                for supply_temperature in supply_temperatures_sec:
                    sup_temperature_is_selected = self.state(
                        f"{sup_carrier_sec}_{supply_temperature}"
                    )
                    constraints.append(
                        (
                            (
                                heat_out_sec
                                - discharge_secondary * cp_sec * rho_sec * supply_temperature
                                - (1.0 - sup_temperature_is_selected) * big_m
                            )
                            / constraint_nominal,
                            -np.inf,
                            0.0,
                        )
                    )
                    constraints.append(
                        (
                            (
                                heat_out_sec
                                - discharge_secondary * cp_sec * rho_sec * supply_temperature
                                + (1.0 - sup_temperature_is_selected) * big_m
                            )
                            / constraint_nominal,
                            0.0,
                            np.inf,
                        )
                    )

            # disconnect HEX
            # Getting var for disabled constraints
            small_m = 0  # 0W
            tol = 1e-5 * big_m  # W
            is_disabled = self.state(self.__disabled_hex_map[heat_exchanger])

            # Constraints to set the disabled integer, note we only set it for the primary
            # side as the secondary side implicetly follows from the energy balance constraints.
            # similar logic in the other blocks
            # This constraints ensures that is_disabled is 0 when heat_primary > 0
            constraints.append(((heat_primary - (1.0 - is_disabled) * big_m) / big_m, -np.inf, 0.0))
            # This constraints ensures that is_disabled is 1 when heat_primary < tol
            constraints.append(
                (
                    (heat_primary - (tol + (small_m - tol) * is_disabled)) / (big_m * tol) ** 0.5,
                    0.0,
                    np.inf,
                )
            )

            if heat_exchanger in self.heat_network_components.get("heat_exchanger", []):
                # Note we don't have to add constraints for the case of no temperature options,
                # as that check is done in the esdl_heat_model
                # Check that secondary supply temperature is lower than that of the primary side
                if len(supply_temperatures_prim) > 0:
                    for t_sup_prim in supply_temperatures_prim:
                        sup_prim_t_is_selected = self.state(f"{sup_carrier_prim}_{t_sup_prim}")
                        if len(supply_temperatures_sec) == 0:
                            t_sup_sec = parameters[f"{heat_exchanger}.Secondary.T_supply"]
                            big_m = 2.0 * t_sup_sec
                            constraints.append(
                                (
                                    (
                                        t_sup_prim
                                        - t_sup_sec
                                        + (is_disabled + (1.0 - sup_prim_t_is_selected)) * big_m
                                    ),
                                    0.0,
                                    np.inf,
                                )
                            )
                        else:
                            for t_sup_sec in supply_temperatures_sec:
                                sup_sec_t_is_selected = self.state(f"{sup_carrier_sec}_{t_sup_sec}")
                                big_m = 2.0 * t_sup_sec
                                constraints.append(
                                    (
                                        (
                                            t_sup_prim * sup_prim_t_is_selected
                                            - t_sup_sec * sup_sec_t_is_selected
                                            + (
                                                is_disabled
                                                + (1.0 - sup_prim_t_is_selected)
                                                + (1.0 - sup_sec_t_is_selected)
                                            )
                                            * big_m
                                        ),
                                        0.0,
                                        np.inf,
                                    )
                                )
                elif len(supply_temperatures_sec) > 0 and len(supply_temperatures_prim) == 0:
                    for t_sup_sec in supply_temperatures_sec:
                        sup_sec_t_is_selected = self.state(f"{sup_carrier_sec}_{t_sup_sec}")
                        t_sup_prim = parameters[f"{heat_exchanger}.Primary.T_supply"]
                        big_m = 2.0 * t_sup_sec
                        constraints.append(
                            (
                                (
                                    t_sup_prim
                                    - t_sup_sec
                                    + (is_disabled + (1.0 - sup_sec_t_is_selected)) * big_m
                                ),
                                0.0,
                                np.inf,
                            )
                        )
                # The check that the chosen return temperature on the primary side is not lower
                # than that of the secondary side
                if len(return_temperatures_prim) > 0:
                    for t_ret_prim in return_temperatures_prim:
                        ret_prim_t_is_selected = self.state(f"{ret_carrier_prim}_{t_ret_prim}")
                        if len(return_temperatures_sec) == 0:
                            t_ret_sec = parameters[f"{heat_exchanger}.Secondary.T_return"]
                            big_m = 2.0 * t_ret_sec
                            constraints.append(
                                (
                                    (
                                        t_ret_prim
                                        - t_ret_sec
                                        + (is_disabled + (1.0 - ret_prim_t_is_selected)) * big_m
                                    ),
                                    0.0,
                                    np.inf,
                                )
                            )
                        else:
                            for t_ret_sec in return_temperatures_sec:
                                ret_sec_t_is_selected = self.state(f"{ret_carrier_sec}_{t_ret_sec}")
                                big_m = 2.0 * t_ret_sec
                                constraints.append(
                                    (
                                        (
                                            t_ret_prim
                                            - t_ret_sec
                                            + (
                                                is_disabled
                                                + (1.0 - ret_sec_t_is_selected)
                                                + (1.0 - ret_prim_t_is_selected)
                                            )
                                            * big_m
                                        ),
                                        0.0,
                                        np.inf,
                                    )
                                )
                elif len(return_temperatures_sec) > 0 and len(return_temperatures_prim) == 0:
                    for t_ret_sec in return_temperatures_sec:
                        ret_sec_t_is_selected = self.state(f"{ret_carrier_sec}_{t_ret_sec}")
                        t_ret_prim = parameters[f"{heat_exchanger}.Primary.T_return"]
                        big_m = 2.0 * t_ret_sec
                        constraints.append(
                            (
                                (
                                    t_ret_prim
                                    - t_ret_sec
                                    + (is_disabled + (1.0 - ret_sec_t_is_selected)) * big_m
                                ),
                                0.0,
                                np.inf,
                            )
                        )

        return constraints

    def __state_vector_scaled(self, variable, ensemble_member):
        """
        This functions returns the casadi symbols scaled with their nominal for the entire time
        horizon.
        """
        canonical, sign = self.alias_relation.canonical_signed(variable)
        return (
            self.state_vector(canonical, ensemble_member) * self.variable_nominal(canonical) * sign
        )

    def _hn_pipe_nominal_discharge(self, heat_network_options, parameters, pipe: str) -> float:
        """
        This functions returns a nominal for the discharge of pipes under topology optimization.
        """
        try:
            pipe_classes = self.__pipe_topo_pipe_class_map[pipe].keys()
            area = np.median(c.area for c in pipe_classes)
        except KeyError:
            area = parameters[f"{pipe}.area"]

        return area * heat_network_options["estimated_velocity"]

    @staticmethod
    def _hn_get_pipe_head_loss_option(pipe, heat_network_options, parameters):
        """
        This function returns the head loss option for a pipe. Note that we assume that we can use
        the more accurate DW linearized approximation when a pipe has a control valve.
        """
        head_loss_option = heat_network_options["head_loss_option"]

        if head_loss_option == HeadLossOption.LINEAR and parameters[f"{pipe}.has_control_valve"]:
            # If there is a control valve present, we use the more accurate
            # Darcy-Weisbach inequality formulation.
            head_loss_option = HeadLossOption.LINEARIZED_DW

        return head_loss_option

    def _hn_pipe_head_loss_constraints(self, ensemble_member):
        """
        This function adds the head loss constraints for pipes. There are two options namely with
        and without pipe class optimization. In both cases we assume that disconnected pipes, pipes
        without flow have no head loss.

        Under pipe-class optimization the head loss constraints per pipe class are added and
        applied with the big_m method (is_topo_disconnected) to only activate the correct
        constraints.

        Under constant pipe class constraints for only one diameter are added.
        """
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

        for pipe in components.get("pipe", []):
            if parameters[f"{pipe}.length"] == 0.0:
                # If the pipe does not have a control valve, the head loss is
                # forced to zero via bounds. If the pipe _does_ have a control
                # valve, then there still is no relationship between the
                # discharge and the head loss/dH.
                continue

            head_loss_sym = self._hn_pipe_to_head_loss_map[pipe]

            dh = self.__state_vector_scaled(f"{pipe}.dH", ensemble_member)
            head_loss = self.__state_vector_scaled(head_loss_sym, ensemble_member)
            discharge = self.__state_vector_scaled(f"{pipe}.Q", ensemble_member)

            # We need to make sure the dH is decoupled from the discharge when
            # the pipe is disconnected. Simply put, this means making the
            # below constraints trivial.
            is_disconnected_var = self.__pipe_disconnect_map.get(pipe)

            if is_disconnected_var is None:
                is_disconnected = 0.0
            else:
                is_disconnected = self.__state_vector_scaled(is_disconnected_var, ensemble_member)

            max_discharge = None
            max_head_loss = -np.inf

            if pipe in self.__pipe_topo_pipe_class_map:
                # Multiple diameter options for this pipe
                pipe_classes = self.__pipe_topo_pipe_class_map[pipe]
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
                self.__pipe_to_flow_direct_map[pipe], ensemble_member
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
        """
        This function adds constraints for the check valve functionality. Meaning that the flow can
        only go in positive direction of the valve. Depending on the status of the valve the flow is
        set to zero or bounded between zero and the maximum discharge.

        The head loss is also bounded to only act in one direction.
        """
        constraints = []
        parameters = self.parameters(ensemble_member)
        options = self.heat_network_options()

        all_pipes = set(self.heat_network_components.get("pipe", []))
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
        """
        This function adds the constraints for the control valve. In this case we allow the valve to
        produce head loss for flow in both directions.
        """
        constraints = []
        parameters = self.parameters(ensemble_member)
        options = self.heat_network_options()

        all_pipes = set(self.heat_network_components.get("pipe", []))
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

    def __heat_loss_variable_constraints(self, ensemble_member):
        """
        Furthermore, the __hn_heat_loss symbol is set, as the heat loss depends on the chosen pipe
        class and the selected temperature in the network.

        Parameters
        ----------
        ensemble_member : The ensemble of the optimizaton

        Returns
        -------
        list of the added constraints
        """
        constraints = []

        for p in self.heat_network_components.get("pipe", []):
            pipe_classes = []

            heat_loss_sym_name = self.__pipe_topo_heat_loss_map[p]

            constraint_nominal = self.variable_nominal(heat_loss_sym_name)

            carrier = self.parameters(ensemble_member)[f"{p}.carrier_id"]
            temperatures = self.temperature_regimes(carrier)

            if len(temperatures) == 0:
                heat_loss_sym = self.extra_variable(heat_loss_sym_name, ensemble_member)
                heat_loss = self._pipe_heat_loss(
                    self.heat_network_options(),
                    self.parameters(ensemble_member),
                    p,
                )
                constraints.append(
                    (
                        (heat_loss_sym - heat_loss) / constraint_nominal,
                        0.0,
                        0.0,
                    )
                )
            else:
                heat_loss_sym = self.__state_vector_scaled(heat_loss_sym_name, ensemble_member)
                for temperature in temperatures:
                    temperature_is_selected = self.state_vector(f"{carrier}_{temperature}")
                    heat_loss = self._pipe_heat_loss(
                        self.heat_network_options(),
                        self.parameters(ensemble_member),
                        p,
                        temp=temperature,
                    )
                    big_m = 2.0 * heat_loss
                    constraints.append(
                        (
                            (
                                heat_loss_sym
                                - heat_loss * np.ones((len(self.times()), 1))
                                + (1.0 - temperature_is_selected) * big_m
                            )
                            / constraint_nominal,
                            0.0,
                            np.inf,
                        )
                    )
                    constraints.append(
                        (
                            (
                                heat_loss_sym
                                - heat_loss * np.ones((len(self.times()), 1))
                                - (1.0 - temperature_is_selected) * big_m
                            )
                            / constraint_nominal,
                            -np.inf,
                            0.0,
                        )
                        )

        return constraints

    def path_constraints(self, ensemble_member):
        """
        Here we add all the path constraints to the optimization problem. Please note that the
        path constraints are the constraints that are applied to each time-step in the problem.
        """

        constraints = super().path_constraints(ensemble_member)

        constraints.extend(self.__node_heat_mixing_path_constraints(ensemble_member))
        constraints.extend(self.__heat_loss_path_constraints(ensemble_member))
        constraints.extend(self.__flow_direction_path_constraints(ensemble_member))
        constraints.extend(self.__node_discharge_mixing_path_constraints(ensemble_member))
        constraints.extend(self.__demand_heat_to_discharge_path_constraints(ensemble_member))
        constraints.extend(self.__source_heat_to_discharge_path_constraints(ensemble_member))
        constraints.extend(self.__pipe_heat_to_discharge_path_constraints(ensemble_member))
        constraints.extend(self.__storage_heat_to_discharge_path_constraints(ensemble_member))
        constraints.extend(
            self.__heat_exchanger_heat_to_discharge_path_constraints(ensemble_member)
        )
        constraints.extend(self.__check_valve_head_discharge_path_constraints(ensemble_member))
        constraints.extend(self.__control_valve_head_discharge_path_constraints(ensemble_member))
        constraints.extend(self.__network_temperature_path_constraints(ensemble_member))
        constraints.extend(self.__pipe_hydraulic_power_path_constraints(ensemble_member))

        return constraints

    def constraints(self, ensemble_member):
        """
        This function adds the normal constraints to the problem. Unlike the path constraints these
        are not applied to every time-step in the problem. Meaning that these constraints either
        consider global variables that are independent of time-step or that the relevant time-steps
        are indexed within the constraint formulation.
        """
        constraints = super().constraints(ensemble_member)

        constraints.extend(self.__pipe_rate_heat_change_constraints(ensemble_member))
        constraints.extend(self.__heat_loss_variable_constraints(ensemble_member))

        if self.heat_network_options()["include_demand_insulation_options"]:
            constraints.extend(self.__heat_matching_demand_insulation_constraints(ensemble_member))

        return constraints

    def history(self, ensemble_member):
        """
        In this history function we avoid the optimization using artificial energy for storage
        assets as the history is not defined.
        """
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
        """
        Here we set the goal programming configuration. We use soft constraints for consecutive
        goals.
        """
        options = super().goal_programming_options()
        options["keep_soft_constraints"] = True
        return options

    def solver_options(self):
        """
        Here we define the solver options. By default we use the open-source solver cbc and casadi
        solver qpsol.
        """
        options = super().solver_options()
        options["casadi_solver"] = "qpsol"
        options["solver"] = "highs"
        return options

    def compiler_options(self):
        """
        In this function we set the compiler configuration.
        """
        options = super().compiler_options()
        options["resolve_parameter_values"] = True
        return options


    def priority_completed(self, priority):
        """
        This function is called after a priority of goals is completed. This function is used to
        specify operations between consecutive goals. Here we set some parameter attributes after
        the optimization is completed.
        """
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
        """
        In this post function we check the optimization results for accurately solving the
        constraints. We do this for the head losses and check if they are consistent with the flow
        direction. Whether, the minimum velocity is actually met. And whether, the directions of
        heat match the directions of the flow.
        """
        super().post()

        self.__pipe_class_to_results()
        self.__pipe_diameter_to_parameters()
        self._pipe_heat_loss_to_parameters()

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
            for p in self.heat_network_components.get("pipe", []):
                head_diff = results[f"{p}.HeatIn.H"] - results[f"{p}.HeatOut.H"]
                if parameters[f"{p}.length"] == 0.0 and not parameters[f"{p}.has_control_valve"]:
                    atol = self.variable_nominal(f"{p}.HeatIn.H") * 1e-5
                    assert np.allclose(head_diff, 0.0, atol=atol)
                else:
                    q = results[f"{p}.Q"]

                    try:
                        is_disconnected = np.round(results[self.__pipe_disconnect_map[p]])
                    except KeyError:
                        is_disconnected = np.zeros_like(q)

                    q_nominal = self.variable_nominal(
                        self.alias_relation.canonical_signed(f"{p}.Q")[0]
                    )
                    inds = (np.abs(q) / q_nominal > 1e-4) & (is_disconnected == 0)
                    if not options["heat_loss_disconnected_pipe"]:
                        assert np.all(np.sign(head_diff[inds]) == np.sign(q[inds]))

        minimum_velocity = options["minimum_velocity"]
        for p in self.heat_network_components.get("pipe", []):
            area = parameters[f"{p}.area"]

            if area == 0.0:
                continue

            q = results[f"{p}.Q"]
            v = q / area
            flow_dir = np.round(results[self.__pipe_to_flow_direct_map[p]])
            try:
                is_disconnected = np.round(results[self.__pipe_disconnect_map[p]])
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
                            f"Velocity in {p} lower than minimum velocity {minimum_velocity} "
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
