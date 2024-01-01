import logging
import math
from typing import List, Optional, Set, Tuple

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
from .physics_mixin import PhysicsMixin
from .pipe_class import PipeClass

logger = logging.getLogger("rtctools_heat_network")


class AssetSizingMixin(PhysicsMixin, BaseComponentTypeMixin, CollocatedIntegratedOptimizationProblem):
    __allowed_head_loss_options = {
        HeadLossOption.NO_HEADLOSS,
        HeadLossOption.LINEAR,
        HeadLossOption.LINEARIZED_DW,
    }

    def __init__(self, *args, **kwargs):
        """
        In this __init__ we prepare the dicts for the variables added by the HeatMixin class
        """

        # Boolean variable to switch assets on/off or to increment their size for the entire time
        # horizon.
        self.__asset_aggregation_count_var = {}
        self.__asset_aggregation_count_var_bounds = {}
        self._asset_aggregation_count_var_map = {}

        # Variable for the maximum discharge under pipe class optimization
        self.__pipe_topo_max_discharge_var = {}
        self._pipe_topo_max_discharge_map = {}
        self.__pipe_topo_max_discharge_nominals = {}
        self.__pipe_topo_max_discharge_var_bounds = {}

        # Variable for the diameter of a pipe during pipe-class optimization
        self.__pipe_topo_diameter_var = {}
        self.__pipe_topo_diameter_var_bounds = {}
        self._pipe_topo_diameter_map = {}
        self.__pipe_topo_diameter_nominals = {}

        # Variable for the investmentcost in eur/m during pipe-class optimization
        self.__pipe_topo_cost_var = {}
        self.__pipe_topo_cost_var_bounds = {}
        self._pipe_topo_cost_map = {}
        self.__pipe_topo_cost_nominals = {}

        # Variable for the heat-loss (not specific for pipe-class optimization)
        self.__pipe_heat_loss_var = {}
        self.__pipe_heat_loss_path_var = {}
        self.__pipe_heat_loss_var_bounds = {}
        self._pipe_heat_loss_map = {}
        self.__pipe_heat_loss_nominals = {}
        self.__pipe_heat_losses = {}

        # Boolean variables for the various pipe class options per pipe
        self.__pipe_topo_pipe_class_var = {}
        self.__pipe_topo_pipe_class_var_bounds = {}
        self._pipe_topo_pipe_class_map = {}
        self.__pipe_topo_pipe_class_result = {}

        self.__pipe_topo_pipe_class_discharge_ordering_var = {}
        self.__pipe_topo_pipe_class_discharge_ordering_var_bounds = {}
        self.__pipe_topo_pipe_class_discharge_ordering_map = {}

        self.__pipe_topo_pipe_class_cost_ordering_map = {}
        self.__pipe_topo_pipe_class_cost_ordering_var = {}
        self.__pipe_topo_pipe_class_cost_ordering_var_bounds = {}

        self.__pipe_topo_pipe_class_heat_loss_ordering_map = {}
        self.__pipe_topo_pipe_class_heat_loss_ordering_var = {}
        self.__pipe_topo_pipe_class_heat_loss_ordering_var_bounds = {}

        self.__pipe_topo_global_pipe_class_count_var = {}
        self.__pipe_topo_global_pipe_class_count_map = {}
        self.__pipe_topo_global_pipe_class_count_var_bounds = {}

        # Dict to specifically update the discharge bounds under pipe-class optimization
        self.__pipe_topo_heat_discharge_bounds = {}

        # list with entry per ensemble member containing dicts of pipe parameter values for
        # diameter, area and heatloss.
        self.__pipe_topo_diameter_area_parameters = []
        self.__pipe_topo_heat_loss_parameters = []

        # Variable for the maximum size of an asset
        self._asset_max_size_map = {}
        self.__asset_max_size_var = {}
        self.__asset_max_size_bounds = {}
        self.__asset_max_size_nominals = {}

        if "timed_setpoints" in kwargs and isinstance(kwargs["timed_setpoints"], dict):
            self._timed_setpoints = kwargs["timed_setpoints"]

        super().__init__(*args, **kwargs)

    def pre(self):
        """
        In this pre method we fill the dicts initiated in the __init__. This means that we create
        the Casadi variables and determine the bounds, nominals and create maps for easier
        retrieving of the variables.
        """
        super().pre()

        options = self.heat_network_options()
        parameters = self.parameters(0)

        bounds = self.bounds()

        # Pipe topology variables

        # In case the user overrides the pipe class of the pipe with a single
        # pipe class we update the diameter/area parameters. If there is more
        # than a single pipe class for a certain pipe, we set the diameter
        # and area to NaN to prevent erroneous constraints.
        for _ in range(self.ensemble_size):
            self.__pipe_topo_diameter_area_parameters.append({})
            self.__pipe_topo_heat_loss_parameters.append({})

        unique_pipe_classes = self.get_unique_pipe_classes()
        for pc in unique_pipe_classes:
            pipe_class_count = f"{pc.name}__global_pipe_class_count"
            self.__pipe_topo_global_pipe_class_count_var[pipe_class_count] = ca.MX.sym(
                pipe_class_count
            )
            self.__pipe_topo_global_pipe_class_count_map[f"{pc.name}"] = pipe_class_count
            self.__pipe_topo_global_pipe_class_count_var_bounds[pipe_class_count] = (
                0.0,
                len(self.heat_network_components.get("pipe", [])),
            )

        for pipe in self.heat_network_components.get("pipe", []):
            pipe_classes = self.pipe_classes(pipe)
            # cold_pipe = self.hot_to_cold_pipe(pipe)

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
            self._pipe_topo_diameter_map[pipe] = diam_var_name

            cost_var_name = f"{pipe}__hn_cost"
            self.__pipe_topo_cost_var[cost_var_name] = ca.MX.sym(cost_var_name)
            self._pipe_topo_cost_map[pipe] = cost_var_name

            max_discharge_var_name = f"{pipe}__hn_max_discharge"
            max_discharges = [c.maximum_discharge for c in pipe_classes]
            self.__pipe_topo_max_discharge_var[max_discharge_var_name] = ca.MX.sym(
                max_discharge_var_name
            )
            self._pipe_topo_max_discharge_map[pipe] = max_discharge_var_name

            if len(pipe_classes) > 0:
                self.__pipe_topo_max_discharge_nominals[pipe] = np.median(max_discharges)
                self.__pipe_topo_max_discharge_var_bounds[pipe] = (
                    -max(max_discharges),
                    max(max_discharges),
                )
            else:
                max_velocity = self.heat_network_options()["maximum_velocity"]
                self.__pipe_topo_max_discharge_nominals[pipe] = (
                    parameters[f"{pipe}.area"] * max_velocity
                )
                self.__pipe_topo_max_discharge_var_bounds[pipe] = (
                    -parameters[f"{pipe}.area"] * max_velocity,
                    parameters[f"{pipe}.area"] * max_velocity,
                )

            if not pipe_classes:
                # No pipe class decision to make for this pipe w.r.t. diameter
                diameter = parameters[f"{pipe}.diameter"]
                investment_cost = parameters[f"{pipe}.investment_cost_coefficient"]
                self.__pipe_topo_diameter_var_bounds[diam_var_name] = (diameter, diameter)
                self.__pipe_topo_cost_var_bounds[cost_var_name] = (investment_cost, investment_cost)
                if diameter > 0.0:
                    self.__pipe_topo_diameter_nominals[diam_var_name] = diameter
                    self.__pipe_topo_cost_nominals[cost_var_name] = max(investment_cost, 1.0)
            elif len(pipe_classes) == 1:
                # No pipe class decision to make for this pipe w.r.t. diameter
                diameter = pipe_classes[0].inner_diameter
                investment_cost = pipe_classes[0].investment_costs
                self.__pipe_topo_diameter_var_bounds[diam_var_name] = (diameter, diameter)
                self.__pipe_topo_cost_var_bounds[cost_var_name] = (investment_cost, investment_cost)
                if diameter > 0.0:
                    self.__pipe_topo_diameter_nominals[diam_var_name] = diameter
                    self.__pipe_topo_cost_nominals[cost_var_name] = max(investment_cost, 1.0)
                    if investment_cost == 0.0:
                        RuntimeWarning(f"{pipe} has an investment cost of 0. €/m")

                for ensemble_member in range(self.ensemble_size):
                    d = self.__pipe_topo_diameter_area_parameters[ensemble_member]

                    d[f"{pipe}.diameter"] = diameter
                    d[f"{pipe}.area"] = pipe_classes[0].area
            else:
                diameters = [c.inner_diameter for c in pipe_classes]
                self.__pipe_topo_diameter_var_bounds[diam_var_name] = (
                    min(diameters),
                    max(diameters),
                )
                costs = [c.investment_costs for c in pipe_classes]
                self.__pipe_topo_cost_var_bounds[cost_var_name] = (
                    min(costs),
                    max(costs),
                )
                self.__pipe_topo_cost_nominals[cost_var_name] = np.median(costs)

                self.__pipe_topo_diameter_nominals[diam_var_name] = min(
                    x for x in diameters if x > 0.0
                )

                for ensemble_member in range(self.ensemble_size):
                    d = self.__pipe_topo_diameter_area_parameters[ensemble_member]

                    d[f"{pipe}.diameter"] = np.nan
                    d[f"{pipe}.area"] = np.nan

            # For similar reasons as for the diameter, we always make a heat
            # loss symbol, even if the heat loss is fixed. Note that we also
            # override the .Heat_loss parameter for cold pipes, even though
            # it is not actually used in the optimization problem.
            heat_loss_var_name = f"{pipe}__hn_heat_loss"
            carrier_id = parameters[f"{pipe}.carrier_id"]
            if len(self.temperature_regimes(carrier_id)) == 0:
                self.__pipe_heat_loss_var[heat_loss_var_name] = ca.MX.sym(heat_loss_var_name)
            else:
                self.__pipe_heat_loss_path_var[heat_loss_var_name] = ca.MX.sym(
                    heat_loss_var_name
                )
            self._pipe_heat_loss_map[pipe] = heat_loss_var_name

            if not pipe_classes or options["neglect_pipe_heat_losses"]:
                # No pipe class decision to make for this pipe w.r.t. heat loss
                heat_loss = self._pipe_heat_loss(options, parameters, pipe)
                self.__pipe_heat_loss_var_bounds[heat_loss_var_name] = (
                    0.0,
                    2.0 * heat_loss,
                )
                if heat_loss > 0:
                    self.__pipe_heat_loss_nominals[heat_loss_var_name] = heat_loss
                else:
                    self.__pipe_heat_loss_nominals[heat_loss_var_name] = max(
                        self._pipe_heat_loss({"neglect_pipe_heat_losses": False}, parameters, pipe),
                        1.0,
                    )

                for ensemble_member in range(self.ensemble_size):
                    h = self.__pipe_topo_heat_loss_parameters[ensemble_member]
                    h[f"{pipe}.Heat_loss"] = self._pipe_heat_loss(options, parameters, pipe)

            elif len(pipe_classes) == 1:
                # No pipe class decision to make for this pipe w.r.t. heat loss
                u_values = pipe_classes[0].u_values
                heat_loss = self._pipe_heat_loss(options, parameters, pipe, u_values)

                self.__pipe_heat_loss_var_bounds[heat_loss_var_name] = (
                    0.0,
                    2.0 * heat_loss,
                )
                if heat_loss > 0:
                    self.__pipe_heat_loss_nominals[heat_loss_var_name] = heat_loss
                else:
                    self.__pipe_heat_loss_nominals[heat_loss_var_name] = max(
                        self._pipe_heat_loss({"neglect_pipe_heat_losses": False}, parameters, pipe),
                        1.0,
                    )

                for ensemble_member in range(self.ensemble_size):
                    h = self.__pipe_topo_heat_loss_parameters[ensemble_member]
                    h[f"{pipe}.Heat_loss"] = heat_loss
            else:
                heat_losses = [
                    self._pipe_heat_loss(options, parameters, pipe, c.u_values)
                    for c in pipe_classes
                ]

                self.__pipe_heat_losses[pipe] = heat_losses
                self.__pipe_heat_loss_var_bounds[heat_loss_var_name] = (
                    min(heat_losses),
                    max(heat_losses),
                )
                self.__pipe_heat_loss_nominals[heat_loss_var_name] = np.median(
                    [x for x in heat_losses if x > 0]
                )

                for ensemble_member in range(self.ensemble_size):
                    h = self.__pipe_topo_heat_loss_parameters[ensemble_member]
                    h[f"{pipe}.Heat_loss"] = max(
                        self._pipe_heat_loss(options, parameters, pipe), 1.0
                    )

            # Pipe class variables.
            if not pipe_classes or len(pipe_classes) == 1:
                # No pipe class decision to make for this pipe
                pass
            else:
                self._pipe_topo_pipe_class_map[pipe] = {}
                self.__pipe_topo_pipe_class_discharge_ordering_map[pipe] = {}
                self.__pipe_topo_pipe_class_cost_ordering_map[pipe] = {}
                self.__pipe_topo_pipe_class_heat_loss_ordering_map[pipe] = {}

                for c in pipe_classes:
                    neighbour = self.has_related_pipe(pipe)
                    if neighbour and pipe not in self.hot_pipes:
                        cold_pipe = self.cold_to_hot_pipe(pipe)
                        pipe_class_var_name = f"{cold_pipe}__hn_pipe_class_{c.name}"
                        pipe_class_ordering_name = (
                            f"{cold_pipe}__hn_pipe_class_{c.name}_discharge_ordering"
                        )
                        pipe_class_cost_ordering_name = (
                            f"{cold_pipe}__hn_pipe_class_{c.name}_cost_ordering"
                        )
                        pipe_class_heat_loss_ordering_name = (
                            f"{cold_pipe}__hn_pipe_class_{c.name}_heat_loss_ordering"
                        )
                    else:
                        pipe_class_var_name = f"{pipe}__hn_pipe_class_{c.name}"
                        pipe_class_ordering_name = (
                            f"{pipe}__hn_pipe_class_{c.name}_discharge_ordering"
                        )
                        pipe_class_cost_ordering_name = (
                            f"{pipe}__hn_pipe_class_{c.name}_cost_ordering"
                        )
                        pipe_class_heat_loss_ordering_name = (
                            f"{pipe}__hn_pipe_class_{c.name}_heat_loss_ordering"
                        )

                    self._pipe_topo_pipe_class_map[pipe][c] = pipe_class_var_name
                    self.__pipe_topo_pipe_class_var[pipe_class_var_name] = ca.MX.sym(
                        pipe_class_var_name
                    )
                    self.__pipe_topo_pipe_class_var_bounds[pipe_class_var_name] = (0.0, 1.0)

                    self.__pipe_topo_pipe_class_discharge_ordering_map[pipe][
                        c
                    ] = pipe_class_ordering_name
                    self.__pipe_topo_pipe_class_discharge_ordering_var[
                        pipe_class_ordering_name
                    ] = ca.MX.sym(pipe_class_ordering_name)
                    self.__pipe_topo_pipe_class_discharge_ordering_var_bounds[
                        pipe_class_ordering_name
                    ] = (0.0, 1.0)

                    self.__pipe_topo_pipe_class_cost_ordering_map[pipe][
                        c
                    ] = pipe_class_cost_ordering_name
                    self.__pipe_topo_pipe_class_cost_ordering_var[
                        pipe_class_cost_ordering_name
                    ] = ca.MX.sym(pipe_class_cost_ordering_name)
                    self.__pipe_topo_pipe_class_cost_ordering_var_bounds[
                        pipe_class_cost_ordering_name
                    ] = (0.0, 1.0)

                    self.__pipe_topo_pipe_class_heat_loss_ordering_map[pipe][
                        c
                    ] = pipe_class_heat_loss_ordering_name
                    self.__pipe_topo_pipe_class_heat_loss_ordering_var[
                        pipe_class_heat_loss_ordering_name
                    ] = ca.MX.sym(pipe_class_heat_loss_ordering_name)
                    self.__pipe_topo_pipe_class_heat_loss_ordering_var_bounds[
                        pipe_class_heat_loss_ordering_name
                    ] = (0.0, 1.0)

        # Update the bounds of the pipes that will have their diameter
        # optimized. Note that the flow direction may have already been fixed
        # based on the original bounds, if that was desired. We can therefore
        # naively override the bounds without taking this into account.
        for pipe in self._pipe_topo_pipe_class_map:
            pipe_classes = self._pipe_topo_pipe_class_map[pipe]
            max_discharge = max([c.maximum_discharge for c in pipe_classes])

            self.__pipe_topo_heat_discharge_bounds[f"{pipe}.Q"] = (-max_discharge, max_discharge)

            # Heat on cold side is zero, so no change needed
            cp = parameters[f"{pipe}.cp"]
            rho = parameters[f"{pipe}.rho"]
            temperature = parameters[f"{pipe}.temperature"]

            # TODO: if temperature is variable these bounds should be set differently
            max_heat = 2.0 * cp * rho * temperature * max_discharge

            self.__pipe_topo_heat_discharge_bounds[f"{pipe}.HeatIn.Heat"] = (-max_heat, max_heat)
            self.__pipe_topo_heat_discharge_bounds[f"{pipe}.HeatOut.Heat"] = (-max_heat, max_heat)

        # When optimizing for pipe size, we do not yet support all options
        if self._pipe_topo_pipe_class_map:
            if np.isfinite(options["maximum_temperature_der"]) and np.isfinite(
                options["maximum_flow_der"]
            ):
                raise Exception(
                    "When optimizing pipe diameters, "
                    "the `maximum_temperature_der` or `maximum_flow_der` should be infinite."
                )

        self.__maximum_total_head_loss = self.__get_maximum_total_head_loss()

        # Making the variables for max size

        def _make_max_size_var(name, lb, ub, nominal):
            asset_max_size_var = f"{name}__max_size"
            self._asset_max_size_map[name] = asset_max_size_var
            self.__asset_max_size_var[asset_max_size_var] = ca.MX.sym(asset_max_size_var)
            self.__asset_max_size_bounds[asset_max_size_var] = (lb, ub)
            self.__asset_max_size_nominals[asset_max_size_var] = nominal

        for asset_name in self.heat_network_components.get("source", []):
            ub = bounds[f"{asset_name}.Heat_source"][1]
            lb = 0.0 if parameters[f"{asset_name}.state"] != 1 else ub
            _make_max_size_var(name=asset_name, lb=lb, ub=ub, nominal=ub / 2.0)

        for asset_name in self.heat_network_components.get("demand", []):
            ub = (
                bounds[f"{asset_name}.Heat_demand"][1]
                if not np.isinf(bounds[f"{asset_name}.Heat_demand"][1])
                else bounds[f"{asset_name}.HeatIn.Heat"][1]
            )
            # Note that we only enforce the upper bound in state enabled if it was explicitly
            # specified for the demand
            lb = 0.0 if np.isinf(bounds[f"{asset_name}.Heat_demand"][1]) else ub
            _make_max_size_var(name=asset_name, lb=lb, ub=ub, nominal=ub / 2.0)

        for asset_name in self.heat_network_components.get("ates", []):
            ub = bounds[f"{asset_name}.Heat_ates"][1]
            lb = 0.0 if parameters[f"{asset_name}.state"] != 1 else ub
            _make_max_size_var(name=asset_name, lb=lb, ub=ub, nominal=ub / 2.0)

        for asset_name in self.heat_network_components.get("buffer", []):
            ub = max(bounds[f"{asset_name}.Stored_heat"][1].values) if isinstance(bounds[f"{asset_name}.Stored_heat"][1], Timeseries) else bounds[f"{asset_name}.Stored_heat"][1]
            lb = 0.0 if parameters[f"{asset_name}.state"] != 1 else ub
            _make_max_size_var(
                name=asset_name,
                lb=lb,
                ub=ub,
                nominal=self.variable_nominal(f"{asset_name}.Stored_heat"),
            )

        for asset_name in [
            *self.heat_network_components.get("heat_exchanger", []),
            *self.heat_network_components.get("heat_pump", []),
            *self.heat_network_components.get("heat_pump_elec", []),
        ]:
            ub = bounds[f"{asset_name}.Secondary_heat"][1]
            lb = 0.0 if parameters[f"{asset_name}.state"] != 1 else ub
            _make_max_size_var(
                name=asset_name,
                lb=lb,
                ub=ub,
                nominal=self.variable_nominal(f"{asset_name}.Secondary_heat"),
            )

        # Making the __aggregation_count variable for each asset
        for asset_list in self.heat_network_components.values():
            for asset in asset_list:
                aggr_count_var = f"{asset}_aggregation_count"
                self._asset_aggregation_count_var_map[asset] = aggr_count_var
                self.__asset_aggregation_count_var[aggr_count_var] = ca.MX.sym(aggr_count_var)
                try:
                    aggr_count_max = parameters[f"{asset}.nr_of_doublets"]
                except KeyError:
                    aggr_count_max = 1.0
                if parameters[f"{asset}.state"] == 0:
                    aggr_count_max = 0.0
                self.__asset_aggregation_count_var_bounds[aggr_count_var] = (0.0, aggr_count_max)

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
        | ``include_asset_is_realized ``       | ``bool``  | ``False``                   |
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
        options["asset_sizing_option"] = True

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

    def get_unique_pipe_classes(self) -> Set[PipeClass]:
        """
        Method queries all hot pipes and returns the set of unique pipe classes defined
        for the network. This means the method assumes the possible pipe classes for each
        cold pipe match the possible pipe classes for the respective hot pipe.
        """
        unique_pipe_classes = set()
        for p in self.heat_network_components.get("pipe", []):
            unique_pipe_classes.update(self.pipe_classes(p))
        return unique_pipe_classes

    def get_optimized_pipe_class(self, pipe: str) -> PipeClass:
        """
        Return the optimized pipe class for a specific pipe. If no
        optimized pipe class is available (yet), a `KeyError` is returned.
        """
        return self.__pipe_topo_pipe_class_result[pipe]

    def get_optimized_deman_insulation_class(self, demand_insulation: str) -> DemandInsulationClass:
        """
        Return the optimized demand_insulation class for a specific pipe. If no
        optimized demand insulation class is available (yet), a `KeyError` is returned.
        """
        return self.__demand_insulation_class_result[demand_insulation]

    def pipe_diameter_symbol_name(self, pipe: str) -> str:
        """
        Return the symbol name for the pipe diameter
        """
        return self._pipe_topo_diameter_map[pipe]

    def pipe_cost_symbol_name(self, pipe: str) -> str:
        """
        Return the symbol name for the pipe investment cost per meter
        """
        return self._pipe_topo_cost_map[pipe]

    @property
    def extra_variables(self):
        """
        In this function we add all the variables defined in the HeatMixin to the optimization
        problem. Note that these are only the normal variables not path variables.
        """
        variables = super().extra_variables.copy()
        variables.extend(self.__pipe_topo_diameter_var.values())
        variables.extend(self.__pipe_topo_cost_var.values())
        variables.extend(self.__pipe_topo_pipe_class_var.values())
        variables.extend(self.__asset_max_size_var.values())
        variables.extend(self.__asset_aggregation_count_var.values())
        variables.extend(self.__pipe_topo_max_discharge_var.values())
        variables.extend(self.__pipe_topo_global_pipe_class_count_var.values())
        variables.extend(self.__pipe_topo_pipe_class_discharge_ordering_var.values())
        variables.extend(self.__pipe_topo_pipe_class_cost_ordering_var.values())
        variables.extend(self.__pipe_topo_pipe_class_heat_loss_ordering_var.values())
        variables.extend(self.__pipe_heat_loss_var.values())
        return variables

    @property
    def path_variables(self):
        """
        In this function we add all the path variables defined in the HeatMixin to the
        optimization problem. Note that path_variables are variables that are created for each
        time-step.
        """
        variables = super().path_variables.copy()
        variables.extend(self.__pipe_heat_loss_path_var.values())
        return variables

    def variable_is_discrete(self, variable):
        """
        All variables that only can take integer values should be added to this function.
        """
        if (
            variable in self.__pipe_topo_pipe_class_var
            or variable in self.__asset_aggregation_count_var
            or variable in self.__pipe_topo_pipe_class_discharge_ordering_var
            or variable in self.__pipe_topo_pipe_class_cost_ordering_var
            or variable in self.__pipe_topo_pipe_class_heat_loss_ordering_var
        ):
            return True
        else:
            return super().variable_is_discrete(variable)

    def variable_nominal(self, variable):
        """
        In this function we add all the nominals for the variables defined/added in the HeatMixin.
        """
        if variable in self.__pipe_topo_diameter_nominals:
            return self.__pipe_topo_diameter_nominals[variable]
        elif variable in self.__pipe_heat_loss_nominals:
            return self.__pipe_heat_loss_nominals[variable]
        elif variable in self.__pipe_topo_cost_nominals:
            return self.__pipe_topo_cost_nominals[variable]
        elif variable in self.__asset_max_size_nominals:
            return self.__asset_max_size_nominals[variable]
        elif variable in self.__pipe_topo_max_discharge_nominals:
            return self.__pipe_topo_max_discharge_nominals[variable]
        else:
            return super().variable_nominal(variable)

    def bounds(self):
        """
        In this function we add the bounds to the problem for all the variables defined/added in
        the HeatMixin.
        """
        bounds = super().bounds()
        bounds.update(self.__pipe_topo_pipe_class_var_bounds)
        bounds.update(self.__pipe_topo_diameter_var_bounds)
        bounds.update(self.__pipe_topo_cost_var_bounds)
        bounds.update(self.__pipe_heat_loss_var_bounds)
        bounds.update(self.__pipe_topo_heat_discharge_bounds)
        bounds.update(self.__asset_max_size_bounds)
        bounds.update(self.__asset_aggregation_count_var_bounds)
        bounds.update(self.__pipe_topo_max_discharge_var_bounds)
        bounds.update(self.__pipe_topo_global_pipe_class_count_var_bounds)
        bounds.update(self.__pipe_topo_pipe_class_discharge_ordering_var_bounds)
        bounds.update(self.__pipe_topo_pipe_class_cost_ordering_var_bounds)
        bounds.update(self.__pipe_topo_pipe_class_heat_loss_ordering_var_bounds)
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
                    pipe_classes = self._pipe_topo_pipe_class_map[pipe].keys()
                    head_loss += max(
                        self._head_loss_class._hn_pipe_head_loss(
                            pipe, self, options, parameters, pc.maximum_discharge, pipe_class=pc
                        )
                        for pc in pipe_classes
                        if pc.maximum_discharge > 0.0
                    )
                except KeyError:
                    area = parameters[f"{pipe}.area"]
                    max_discharge = options["maximum_velocity"] * area
                    head_loss += self._head_loss_class._hn_pipe_head_loss(pipe, self, options, parameters, max_discharge)

            head_loss += options["minimum_pressure_far_point"] * 10.2

            max_sum_dh_pipes = max(max_sum_dh_pipes, head_loss)

        # Maximum pressure difference allowed with user options
        # NOTE: Does not yet take elevation differences into acccount
        max_dh_network_options = (
            options["pipe_maximum_pressure"] - options["pipe_minimum_pressure"]
        ) * 10.2

        return min(max_sum_dh_pipes, max_dh_network_options)

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
                is_disconnected_var = self._pipe_disconnect_map.get(pipe)
                if is_disconnected_var is None:
                    is_disconnected = 0.0
                else:
                    is_disconnected = self.state(is_disconnected_var)

                flow_dir_var = self._pipe_to_flow_direct_map[pipe]
                flow_dir = self.state(flow_dir_var)  # 0/1: negative/positive flow direction

                if pipe in self._pipe_topo_pipe_class_map:
                    # Multiple diameter options for this pipe
                    pipe_classes = self._pipe_topo_pipe_class_map[pipe]
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
                            self._head_loss_class._hydraulic_power(
                                pipe,
                                self,
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
                        self._head_loss_class._hydraulic_power(
                            pipe,
                            self,
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
            pipe_classes = self._pipe_topo_pipe_class_map[pipe].keys()
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
            is_disconnected_var = self._pipe_disconnect_map.get(pipe)

            if is_disconnected_var is None:
                is_disconnected = 0.0
            else:
                is_disconnected = self.__state_vector_scaled(is_disconnected_var, ensemble_member)

            max_discharge = None
            max_head_loss = -np.inf

            if pipe in self._pipe_topo_pipe_class_map:
                # Multiple diameter options for this pipe
                pipe_classes = self._pipe_topo_pipe_class_map[pipe]
                max_discharge = max(c.maximum_discharge for c in pipe_classes)

                for pc, pc_var_name in pipe_classes.items():
                    if pc.inner_diameter == 0.0:
                        continue

                    head_loss_max_discharge = self._head_loss_class._hn_pipe_head_loss(
                        pipe, self, options, parameters, max_discharge, pipe_class=pc
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
                        self._head_loss_class._hn_pipe_head_loss(
                            pipe,
                            self,
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
                        self._head_loss_class._hn_pipe_head_loss(
                            pipe, self, options, parameters, pc.maximum_discharge, pipe_class=pc
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
                    self._head_loss_class._hn_pipe_head_loss(
                        pipe,
                        self,
                        options,
                        parameters,
                        discharge,
                        head_loss,
                        dh,
                        is_disconnected + is_topo_disconnected,
                        1.1 * self.__maximum_total_head_loss,
                    )
                )

                max_head_loss = self._head_loss_class._hn_pipe_head_loss(pipe, self, options, parameters, max_discharge)

            # Relate the head loss symbol to the pipe's dH symbol.

            # FIXME: Ugly hack. Cold pipes should be modelled completely with
            # their own integers as well.
            flow_dir = self.__state_vector_scaled(
                self._pipe_to_flow_direct_map[pipe], ensemble_member
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
            flow_dir_var = self._pipe_to_flow_direct_map[p]
            flow_dir = self.state(flow_dir_var)

            is_disconnected_var = self._pipe_disconnect_map.get(p)

            if is_disconnected_var is None:
                is_disconnected = 0.0
            else:
                is_disconnected = self.state(is_disconnected_var)

            q_pipe = self.state(f"{p}.Q")
            heat_in = self.state(f"{p}.HeatIn.Heat")
            heat_out = self.state(f"{p}.HeatOut.Heat")

            try:
                pipe_classes = self._pipe_topo_pipe_class_map[p].keys()
                maximum_discharge = max([c.maximum_discharge for c in pipe_classes])
                var_names = self._pipe_topo_pipe_class_map[p].values()
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

            base_flow_dir_var = self.state(self._pipe_to_flow_direct_map[pipes[0]])

            for p in pipes[1:]:
                flow_dir_var = self.state(self._pipe_to_flow_direct_map[p])
                constraints.append((base_flow_dir_var - flow_dir_var, 0.0, 0.0))

        return constraints

    def __pipe_topology_constraints(self, ensemble_member):
        """
        This function adds the constraints needed for the optimization of pipe classes (referred to
        as topology optimization). We ensure that only one pipe_class can be selected.
        Additionally,, we set the diameter and cost variable to those associated with the optimized
        pipe_class. Note that the cost symbol is the investment cost in EUR/meter, the actual
        investment cost of the pipe is set in the __investment_cost variable.

        Furthermore, ordering variables are set in this function. This is to give the optimization
        insight in the ordering of all the possible boolean choices in pipe classes and as such
        quicker find a feasible and optimal solution. The ordering are 0 or 1 depending on whether
        the variable is larger compared to the selected pipe-class. For example is the pipe class
        variable for DN200 is 1, then all discharge ordering variables for the pipe classes >=DN200
        are 1.
        """
        constraints = []

        # These are the constraints to count the amount of a certain pipe class
        unique_pipe_classes = self.get_unique_pipe_classes()
        pipe_class_count_sum = {pc.name: 0 for pc in unique_pipe_classes}

        for p in self.heat_network_components.get("pipe", []):
            try:
                pipe_classes = self._pipe_topo_pipe_class_map[p]
            except KeyError:
                pass
            else:
                for pc in pipe_classes:
                    neighbour = self.has_related_pipe(p)
                    if neighbour and p not in self.hot_pipes:
                        var_name = f"{self.cold_to_hot_pipe(p)}__hn_pipe_class_{pc.name}"
                    else:
                        var_name = f"{p}__hn_pipe_class_{pc.name}"
                    pipe_class_count_sum[pc.name] += self.extra_variable(var_name, ensemble_member)

        for pc in unique_pipe_classes:
            var = self.extra_variable(
                self.__pipe_topo_global_pipe_class_count_map[pc.name], ensemble_member
            )
            constraints.append(((pipe_class_count_sum[pc.name] - var), 0.0, 0.0))

        # These are the constraints to order the discharge capabilities of the pipe classes
        for p, pipe_classes in self.__pipe_topo_pipe_class_discharge_ordering_map.items():
            max_discharge = self.extra_variable(self._pipe_topo_max_discharge_map[p])
            max_discharges = {
                pc.name: pc.maximum_discharge for pc in self._pipe_topo_pipe_class_map[p]
            }
            median_discharge = np.median(list(max_discharges.values()))

            big_m = 2.0 * max(max_discharges.values())
            for pc, var_name in pipe_classes.items():
                pipe_class_discharge_ordering = self.extra_variable(var_name, ensemble_member)

                constraints.append(
                    (
                        (
                            max_discharge
                            - max_discharges[pc.name]
                            + pipe_class_discharge_ordering * big_m
                        )
                        / median_discharge,
                        0.0,
                        np.inf,
                    )
                )
                constraints.append(
                    (
                        (
                            max_discharge
                            - max_discharges[pc.name]
                            - (1.0 - pipe_class_discharge_ordering) * big_m
                        )
                        / median_discharge,
                        -np.inf,
                        0.0,
                    )
                )

        # These are the constraints to order the costs of the pipe classes
        for p, pipe_classes in self.__pipe_topo_pipe_class_cost_ordering_map.items():
            cost_sym_name = self._pipe_topo_cost_map[p]
            cost_sym = self.extra_variable(cost_sym_name, ensemble_member)
            costs = {pc.name: pc.investment_costs for pc in self._pipe_topo_pipe_class_map[p]}

            big_m = 2.0 * max(costs.values())
            for pc, var_name in pipe_classes.items():
                pipe_class_cost_ordering = self.extra_variable(var_name, ensemble_member)

                # should be one if >= than cost_symbol
                constraints.append(
                    (
                        (cost_sym - costs[pc.name] + pipe_class_cost_ordering * big_m)
                        / self.variable_nominal(cost_sym_name),
                        0.0,
                        np.inf,
                    )
                )
                constraints.append(
                    (
                        (cost_sym - costs[pc.name] - (1.0 - pipe_class_cost_ordering) * big_m)
                        / self.variable_nominal(cost_sym_name),
                        -np.inf,
                        0.0,
                    )
                )

        # These are the constraints to order the heat loss of the pipe classes.
        if not self.heat_network_options()["neglect_pipe_heat_losses"]:
            for pipe, pipe_classes in self.__pipe_topo_pipe_class_heat_loss_ordering_map.items():
                if pipe in self.hot_pipes and self.has_related_pipe(pipe):
                    heat_loss_sym_name = self._pipe_heat_loss_map[pipe]
                    heat_loss_sym = self.extra_variable(heat_loss_sym_name, ensemble_member)
                    cold_name = self._pipe_heat_loss_map[self.hot_to_cold_pipe(pipe)]
                    heat_loss_sym += self.extra_variable(cold_name, ensemble_member)
                    heat_losses = [
                        h1 + h2
                        for h1, h2 in zip(
                            self.__pipe_heat_losses[pipe],
                            self.__pipe_heat_losses[self.hot_to_cold_pipe(pipe)],
                        )
                    ]
                elif pipe in self.hot_pipes and not self.has_related_pipe(pipe):
                    heat_loss_sym_name = self._pipe_heat_loss_map[pipe]
                    heat_loss_sym = self.extra_variable(heat_loss_sym_name, ensemble_member)

                    heat_losses = self.__pipe_heat_losses[pipe]
                else:  # cold pipe
                    continue

                big_m = 2.0 * max(heat_losses)
                for var_name, heat_loss in zip(pipe_classes.values(), heat_losses):
                    pipe_class_heat_loss_ordering = self.extra_variable(var_name, ensemble_member)

                    # should be one if >= than heat_loss_symbol
                    constraints.append(
                        (
                            (heat_loss_sym - heat_loss + pipe_class_heat_loss_ordering * big_m)
                            / self.variable_nominal(heat_loss_sym_name),
                            0.0,
                            np.inf,
                        )
                    )
                    constraints.append(
                        (
                            (
                                heat_loss_sym
                                - heat_loss
                                - (1.0 - pipe_class_heat_loss_ordering) * big_m
                            )
                            / self.variable_nominal(heat_loss_sym_name),
                            -np.inf,
                            0.0,
                        )
                    )

        for p, pipe_classes in self._pipe_topo_pipe_class_map.items():
            variables = {
                pc.name: self.extra_variable(var_name, ensemble_member)
                for pc, var_name in pipe_classes.items()
            }

            # Make sure exactly one indicator is true
            constraints.append((sum(variables.values()), 1.0, 1.0))

            # set the max discharge
            max_discharge = self.extra_variable(self._pipe_topo_max_discharge_map[p])
            max_discharges = {pc.name: pc.maximum_discharge for pc in pipe_classes}
            max_discharge_expr = sum(
                variables[pc_name] * max_discharges[pc_name] for pc_name in variables
            )

            constraints.append(
                (
                    (max_discharge - max_discharge_expr)
                    / self.variable_nominal(self._pipe_topo_max_discharge_map[p]),
                    0.0,
                    0.0,
                )
            )

            # Match the indicators to the diameter symbol
            diam_sym_name = self._pipe_topo_diameter_map[p]
            diam_sym = self.extra_variable(diam_sym_name, ensemble_member)

            diameters = {pc.name: pc.inner_diameter for pc in pipe_classes}

            diam_expr = sum(variables[pc_name] * diameters[pc_name] for pc_name in variables)

            constraint_nominal = self.variable_nominal(diam_sym_name)
            constraints.append(((diam_sym - diam_expr) / constraint_nominal, 0.0, 0.0))

            # match the indicators to the cost symbol
            cost_sym_name = self._pipe_topo_cost_map[p]
            cost_sym = self.extra_variable(cost_sym_name, ensemble_member)

            investment_costs = {pc.name: pc.investment_costs for pc in pipe_classes}

            costs_expr = sum(
                variables[pc_name] * investment_costs[pc_name] for pc_name in variables
            )
            costs_constraint_nominal = self.variable_nominal(cost_sym_name)

            constraints.append(((cost_sym - costs_expr) / costs_constraint_nominal, 0.0, 0.0))

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

            heat_loss_sym_name = self._pipe_heat_loss_map[p]

            constraint_nominal = self.variable_nominal(heat_loss_sym_name)

            carrier = self.parameters(ensemble_member)[f"{p}.carrier_id"]
            temperatures = self.temperature_regimes(carrier)

            if len(temperatures) == 0:
                heat_loss_sym = self.extra_variable(heat_loss_sym_name, ensemble_member)
                try:
                    heat_losses = self.__pipe_heat_losses[p]
                    pipe_classes = self._pipe_topo_pipe_class_map[p]
                    variables = [
                        self.extra_variable(var_name, ensemble_member)
                        for pc, var_name in pipe_classes.items()
                    ]
                    heat_loss_expr = 0.0
                    for i in range(len(heat_losses)):
                        heat_loss_expr = heat_loss_expr + variables[i] * heat_losses[i]
                    constraints.append(
                        ((heat_loss_sym - heat_loss_expr) / constraint_nominal, 0.0, 0.0)
                    )
                except KeyError:
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
                    if len(pipe_classes) == 0:
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
                    else:
                        heat_losses = [
                            self._pipe_heat_loss(
                                self.heat_network_options(),
                                self.parameters(ensemble_member),
                                p,
                                u_values=c.u_values,
                                temp=temperature,
                            )
                            for c in pipe_classes
                        ]
                        count = 0
                        big_m = 2.0 * max(heat_losses)
                        for pc_var_name in pipe_classes.values():
                            pc = self.__pipe_topo_pipe_class_var[pc_var_name]
                            constraints.append(
                                (
                                    (
                                        heat_loss_sym
                                        - heat_losses[count] * np.ones(len(self.times()))
                                        + (1.0 - pc) * big_m * np.ones(len(self.times()))
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
                                        - heat_losses[count] * np.ones(len(self.times()))
                                        - (1.0 - pc) * big_m * np.ones(len(self.times()))
                                        - (1.0 - temperature_is_selected) * big_m
                                    )
                                    / constraint_nominal,
                                    -np.inf,
                                    0.0,
                                )
                            )
                            count += 1

        return constraints

    def __pipe_topology_path_constraints(self, ensemble_member):
        """
        This function adds constraints to limit the discharge that can flow through a pipe when the
        pipe class is being optimized. This is needed as the different pipe classes have different
        diameters and maximum velocities.
        """
        constraints = []

        # Clip discharge based on pipe class
        for p in self.heat_network_components.get("pipe", []):
            # Match the indicators to the discharge symbol(s)
            discharge_sym_hot = self.state(f"{p}.Q")
            nominal = self.variable_nominal(f"{p}.Q")

            max_discharge = self.__pipe_topo_max_discharge_var[
                self._pipe_topo_max_discharge_map[p]
            ]

            constraints.append(((max_discharge - discharge_sym_hot) / nominal, 0.0, np.inf))
            constraints.append(((-max_discharge - discharge_sym_hot) / nominal, -np.inf, 0.0))

        return constraints

    def __max_size_constraints(self, ensemble_member):
        """
        This function makes sure that the __max_size variable is at least as large as needed. For
        most assets the __max_size is related to the thermal Power it can produce or consume, there
        are a few exceptions like tank storage that sizes with volume.

        Since it are inequality constraints are inequality the __max_size variable can be larger
        than what is the actual needed size. In combination with the objectives, e.g. cost
        minimization, we can drag down the __max_size to the minimum required.
        """
        constraints = []
        bounds = self.bounds()

        for b in self.heat_network_components.get("buffer", []):
            max_var = self._asset_max_size_map[b]
            max_heat = self.extra_variable(max_var, ensemble_member)
            stored_heat = self.__state_vector_scaled(f"{b}.Stored_heat", ensemble_member)
            constraint_nominal = self.variable_nominal(max_var)

            constraints.append(
                (
                    (np.ones(len(self.times())) * max_heat - stored_heat) / constraint_nominal,
                    0.0,
                    np.inf,
                )
            )

            # Same as for the buffer but now for the source
        for s in self.heat_network_components.get("source", []):
            max_var = self._asset_max_size_map[s]
            max_heat = self.extra_variable(max_var, ensemble_member)
            heat_source = self.__state_vector_scaled(f"{s}.Heat_source", ensemble_member)
            constraint_nominal = self.variable_nominal(f"{s}.Heat_source")

            try:
                profile = self.get_timeseries(f"{s}.target_heat_source").values
                profile_scaled = profile / max(profile)
                for i in range(0, len(self.times())):
                    constraints.append(
                        (
                            (profile_scaled[i] * max_heat - heat_source[i]) / constraint_nominal,
                            0.0,
                            np.inf,
                        )
                    )
            except KeyError:
                constraints.append(
                    (
                        (np.ones(len(self.times())) * max_heat - heat_source) / constraint_nominal,
                        0.0,
                        np.inf,
                    )
                )

        for hx in [
            *self.heat_network_components.get("heat_exchanger", []),
            *self.heat_network_components.get("heat_pump", []),
            *self.heat_network_components.get("heat_pump_elec", []),
        ]:
            max_var = self._asset_max_size_map[hx]
            max_heat = self.extra_variable(max_var, ensemble_member)
            heat_secondary = self.__state_vector_scaled(f"{hx}.Secondary_heat", ensemble_member)
            constraint_nominal = self.variable_nominal(f"{hx}.Secondary_heat")

            constraints.append(
                (
                    (np.ones(len(self.times())) * max_heat - heat_secondary) / constraint_nominal,
                    0.0,
                    np.inf,
                )
            )

        for d in self.heat_network_components.get("demand", []):
            max_var = self._asset_max_size_map[d]
            max_heat = self.extra_variable(max_var, ensemble_member)
            heat_demand = self.__state_vector_scaled(f"{d}.Heat_demand", ensemble_member)
            constraint_nominal = max(
                self.variable_nominal(f"{d}.Heat_demand"), self.variable_nominal(f"{d}.HeatIn.Heat")
            )
            constraints.append(
                (
                    (np.ones(len(self.times())) * max_heat - heat_demand) / constraint_nominal,
                    0.0,
                    np.inf,
                )
            )

        for a in self.heat_network_components.get("ates", []):
            max_var = self._asset_max_size_map[a]
            max_heat = self.extra_variable(max_var, ensemble_member)
            heat_ates = self.__state_vector_scaled(f"{a}.Heat_ates", ensemble_member)
            constraint_nominal = bounds[f"{a}.Heat_ates"][1]

            constraints.append(
                (
                    (np.ones(len(self.times())) * max_heat - heat_ates) / constraint_nominal,
                    0.0,
                    np.inf,
                )
            )
            constraints.append(
                (
                    (np.ones(len(self.times())) * max_heat + heat_ates) / constraint_nominal,
                    0.0,
                    np.inf,
                )
            )

        return constraints

    def __optional_asset_path_constraints(self, ensemble_member):
        """
        This function adds constraints that set the _aggregation_count variable. This variable is
        used for most assets (except geo and ates) to turn on/off the asset. Which effectively mean
        that assets cannot exchange thermal power with the network when _aggregation_count == 0.

        Specifically for the geo and ATES we use the _aggregation_count for modelling the amount of
        doublets. Where the _aggregation_count allows increments in the upper limit for the thermal
        power that can be exchanged with the network.
        """
        constraints = []

        parameters = self.parameters(ensemble_member)
        bounds = self.bounds()

        for asset_name in [
            asset_name
            for asset_name_list in self.heat_network_components.values()
            for asset_name in asset_name_list
        ]:
            if parameters[f"{asset_name}.state"] == 0 or parameters[f"{asset_name}.state"] == 2:
                if asset_name in [
                    *self.heat_network_components.get("geothermal", []),
                    *self.heat_network_components.get("ates", []),
                ]:
                    state_var = self.state(f"{asset_name}.Heat_flow")
                    single_power = parameters[f"{asset_name}.single_doublet_power"]
                    nominal_value = 2.0 * bounds[f"{asset_name}.Heat_flow"][1]
                    nominal_var = self.variable_nominal(f"{asset_name}.Heat_flow")
                elif asset_name in [*self.heat_network_components.get("buffer", [])]:
                    state_var = self.state(f"{asset_name}.HeatIn.Q")
                    single_power = parameters[f"{asset_name}.volume"]
                    nominal_value = single_power
                    nominal_var = self.variable_nominal(f"{asset_name}.HeatIn.Q")
                elif asset_name in [*self.heat_network_components.get("node", [])]:
                    # TODO: can we generalize to all possible components to avoid having to skip
                    #  joints and other components in the future?
                    continue
                else:
                    state_var = self.state(f"{asset_name}.Heat_flow")
                    single_power = bounds[f"{asset_name}.Heat_flow"][1]
                    nominal_value = single_power
                    nominal_var = self.variable_nominal(f"{asset_name}.Heat_flow")
                aggregation_count = self.__asset_aggregation_count_var[
                    self._asset_aggregation_count_var_map[asset_name]
                ]
                constraint_nominal = (nominal_value * nominal_var) ** 0.5

                constraints.append(
                    (
                        (single_power * aggregation_count - state_var) / constraint_nominal,
                        0.0,
                        np.inf,
                    )
                )
                constraints.append(
                    (
                        (-single_power * aggregation_count - state_var) / constraint_nominal,
                        -np.inf,
                        0.0,
                    )
                )
            elif parameters[f"{asset_name}.state"] == 1:
                aggregation_count = self.__asset_aggregation_count_var[
                    self._asset_aggregation_count_var_map[asset_name]
                ]
                aggr_bound = self.__asset_aggregation_count_var_bounds[
                    asset_name + "_aggregation_count"
                ][1]
                constraints.append(((aggregation_count - aggr_bound), 0.0, 0.0))

        return constraints

    def path_constraints(self, ensemble_member):
        """
        Here we add all the path constraints to the optimization problem. Please note that the
        path constraints are the constraints that are applied to each time-step in the problem.
        """

        constraints = super().path_constraints(ensemble_member)

        constraints.extend(self.__flow_direction_path_constraints(ensemble_member))
        constraints.extend(self.__pipe_topology_path_constraints(ensemble_member))
        constraints.extend(self.__optional_asset_path_constraints(ensemble_member))
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

        if self.heat_network_options()["head_loss_option"] != HeadLossOption.NO_HEADLOSS:
            constraints.extend(self._hn_pipe_head_loss_constraints(ensemble_member))


        constraints.extend(self.__heat_loss_variable_constraints(ensemble_member))
        constraints.extend(self.__pipe_topology_constraints(ensemble_member))
        constraints.extend(self.__max_size_constraints(ensemble_member))

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

    def __pipe_class_to_results(self):
        """
        This functions writes all resulting pipe class results to a dict.
        """
        for ensemble_member in range(self.ensemble_size):
            results = self.extract_results(ensemble_member)

            for pipe in self.heat_network_components.get("pipe", []):
                pipe_classes = self.pipe_classes(pipe)

                if not pipe_classes:
                    continue
                elif len(pipe_classes) == 1:
                    pipe_class = pipe_classes[0]
                else:
                    pipe_class = next(
                        c
                        for c, s in self._pipe_topo_pipe_class_map[pipe].items()
                        if round(results[s][0]) == 1.0
                    )

                for p in [pipe, self.hot_to_cold_pipe(pipe)]:
                    self.__pipe_topo_pipe_class_result[p] = pipe_class

    def __pipe_diameter_to_parameters(self):
        """
        This function is used to update the parameters object with the results of the pipe class
        optimization
        """
        for ensemble_member in range(self.ensemble_size):
            d = self.__pipe_topo_diameter_area_parameters[ensemble_member]
            for pipe in self._pipe_topo_pipe_class_map:
                pipe_class = self.get_optimized_pipe_class(pipe)

                for p in [pipe, self.hot_to_cold_pipe(pipe)]:
                    d[f"{p}.diameter"] = pipe_class.inner_diameter
                    d[f"{p}.area"] = pipe_class.area

    def _pipe_heat_loss_to_parameters(self):
        """
        This function is used to set the optimized heat losses in the parameters object.
        """
        options = self.heat_network_options()

        for ensemble_member in range(self.ensemble_size):
            parameters = self.parameters(ensemble_member)

            h = self.__pipe_topo_heat_loss_parameters[ensemble_member]
            for pipe in self.__pipe_heat_losses:
                pipe_class = self.get_optimized_pipe_class(pipe)

                h[f"{pipe}.Heat_loss"] = self._pipe_heat_loss(
                    options, parameters, pipe, pipe_class.u_values
                )

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
            and priority == self._head_loss_class._hn_minimization_goal_class.priority
        ):
            self.__pipe_diameter_to_parameters()

        super().priority_completed(priority)
    def post(self):
        super().post()

        self.__pipe_class_to_results()
        self.__pipe_diameter_to_parameters()
        self._pipe_heat_loss_to_parameters()

