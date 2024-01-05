import logging

import casadi as ca

import numpy as np

from abc import abstractmethod

from rtctools.optimization.collocated_integrated_optimization_problem import (
    CollocatedIntegratedOptimizationProblem,
)
from rtctools.optimization.timeseries import Timeseries

from .base_component_type_mixin import BaseComponentTypeMixin
from .head_loss_mixin import HeadLossOption

logger = logging.getLogger("rtctools_heat_network")


class FinancialMixin(BaseComponentTypeMixin, CollocatedIntegratedOptimizationProblem):
    __allowed_head_loss_options = {
        HeadLossOption.NO_HEADLOSS,
        HeadLossOption.LINEAR,
        HeadLossOption.LINEARIZED_DW,
    }

    def __init__(self, *args, **kwargs):
        """
        In this __init__ we prepare the dicts for the variables added by the HeatMixin class
        """

        # Variable for fixed operational cost
        self._asset_fixed_operational_cost_map = {}
        self.__asset_fixed_operational_cost_var = {}
        self.__asset_fixed_operational_cost_nominals = {}
        self.__asset_fixed_operational_cost_bounds = {}

        # Variable for variable operational cost
        self._asset_variable_operational_cost_map = {}
        self.__asset_variable_operational_cost_var = {}
        self.__asset_variable_operational_cost_bounds = {}
        self.__asset_variable_operational_cost_nominals = {}

        # Variable for investment cost
        self._asset_investment_cost_map = {}
        self.__asset_investment_cost_var = {}
        self.__asset_investment_cost_nominals = {}
        self.__asset_investment_cost_bounds = {}

        # Variable for installation cost
        self._asset_installation_cost_map = {}
        self.__asset_installation_cost_var = {}
        self.__asset_installation_cost_bounds = {}
        self.__asset_installation_cost_nominals = {}

        # Variable for the cumulative investment and installation cost made per asset
        self.__cumulative_investments_made_in_eur_map = {}
        self.__cumulative_investments_made_in_eur_var = {}
        self.__cumulative_investments_made_in_eur_nominals = {}
        self.__cumulative_investments_made_in_eur_bounds = {}

        # Variable for when in time an asset is realized
        self.__asset_is_realized_map = {}
        self.__asset_is_realized_var = {}
        self.__asset_is_realized_bounds = {}

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

        # Making the cost variables; fixed_operational_cost, variable_operational_cost,
        # installation_cost and investment_cost
        for asset_name in [
            asset_name
            for asset_name_list in self.heat_network_components.values()
            for asset_name in asset_name_list
        ]:
            if asset_name in [
                *self.heat_network_components.get("node", []),
                *self.heat_network_components.get("pump", []),
                *self.heat_network_components.get("check_valve", []),
                *self.heat_network_components.get("control_valve", []),
                *self.heat_network_components.get("electricity_cable", []),
                *self.heat_network_components.get("electricity_source", []),
                *self.heat_network_components.get("electricity_demand", []),
                *self.heat_network_components.get("electricity_node", []),
                *self.heat_network_components.get("gas_node", []),
                *self.heat_network_components.get("gas_pipe", []),
                *self.heat_network_components.get("gas_source", []),
                *self.heat_network_components.get("gas_demand", []),
            ]:
                continue
            elif asset_name in [*self.heat_network_components.get("ates", [])]:
                nominal_fixed_operational = self.variable_nominal(f"{asset_name}.Heat_ates")
                nominal_variable_operational = nominal_fixed_operational
                nominal_investment = nominal_fixed_operational
            elif asset_name in [*self.heat_network_components.get("demand", [])]:
                nominal_fixed_operational = (
                    bounds[f"{asset_name}.Heat_demand"][1]
                    if not np.isinf(bounds[f"{asset_name}.Heat_demand"][1])
                    else bounds[f"{asset_name}.HeatIn.Heat"][1]
                )
                nominal_variable_operational = nominal_fixed_operational
                nominal_investment = nominal_fixed_operational
            elif asset_name in [*self.heat_network_components.get("source", [])]:
                nominal_fixed_operational = self.variable_nominal(f"{asset_name}.Heat_source")
                nominal_variable_operational = nominal_fixed_operational
                nominal_investment = nominal_fixed_operational
            elif asset_name in [*self.heat_network_components.get("pipe", [])]:
                nominal_fixed_operational = max(parameters[f"{asset_name}.length"], 1.0)
                nominal_variable_operational = nominal_fixed_operational
                nominal_investment = nominal_fixed_operational
            elif asset_name in [*self.heat_network_components.get("buffer", [])]:
                nominal_fixed_operational = self.variable_nominal(f"{asset_name}.Stored_heat")
                nominal_variable_operational = self.variable_nominal(f"{asset_name}.Heat_buffer")
                nominal_investment = nominal_fixed_operational
            elif asset_name in [
                *self.heat_network_components.get("heat_exchanger", []),
                *self.heat_network_components.get("heat_pump", []),
                *self.heat_network_components.get("heat_pump_elec", []),
            ]:
                nominal_fixed_operational = self.variable_nominal(f"{asset_name}.Secondary_heat")
                nominal_variable_operational = nominal_fixed_operational
                nominal_investment = nominal_fixed_operational
            else:
                logger.warning(
                    f"Asset {asset_name} has type for which "
                    f"we cannot determine bounds and nominals on the costs, "
                    f"skipping it."
                )
                nominal_fixed_operational = 1.0
                nominal_variable_operational = 1.0
                nominal_investment = 1.0

            # fixed operational cost
            asset_fixed_operational_cost_var = f"{asset_name}__fixed_operational_cost"
            self._asset_fixed_operational_cost_map[asset_name] = asset_fixed_operational_cost_var
            self.__asset_fixed_operational_cost_var[asset_fixed_operational_cost_var] = ca.MX.sym(
                asset_fixed_operational_cost_var
            )
            self.__asset_fixed_operational_cost_bounds[asset_fixed_operational_cost_var] = (
                0.0,
                np.inf,
            )
            self.__asset_fixed_operational_cost_nominals[asset_fixed_operational_cost_var] = (
                max(
                    parameters[f"{asset_name}.fixed_operational_cost_coefficient"]
                    * nominal_fixed_operational,
                    1.0e2,
                )
                if nominal_fixed_operational is not None
                else 1.0e2
            )

            # variable operational cost
            variable_operational_cost_var = f"{asset_name}__variable_operational_cost"
            self._asset_variable_operational_cost_map[asset_name] = variable_operational_cost_var
            self.__asset_variable_operational_cost_var[variable_operational_cost_var] = ca.MX.sym(
                variable_operational_cost_var
            )
            self.__asset_variable_operational_cost_bounds[variable_operational_cost_var] = (
                0.0,
                np.inf,
            )
            self.__asset_variable_operational_cost_nominals[variable_operational_cost_var] = (
                max(
                    parameters[f"{asset_name}.variable_operational_cost_coefficient"]
                    * nominal_variable_operational
                    * 24.0,
                    1.0e2,
                )
                if nominal_variable_operational is not None
                else 1.0e2
            )

            # installation cost
            asset_installation_cost_var = f"{asset_name}__installation_cost"
            self._asset_installation_cost_map[asset_name] = asset_installation_cost_var
            self.__asset_installation_cost_var[asset_installation_cost_var] = ca.MX.sym(
                asset_installation_cost_var
            )
            try:
                aggr_count_max = parameters[f"{asset_name}.nr_of_doublets"]
            except KeyError:
                aggr_count_max = 1.0
            if parameters[f"{asset_name}.state"] == 0:
                aggr_count_max = 0.0
            self.__asset_installation_cost_bounds[asset_installation_cost_var] = (
                0.0,
                parameters[f"{asset_name}.installation_cost"] * aggr_count_max,
            )
            self.__asset_installation_cost_nominals[asset_installation_cost_var] = (
                parameters[f"{asset_name}.installation_cost"]
                if parameters[f"{asset_name}.installation_cost"]
                else 1.0e2
            )

            # investment cost
            asset_investment_cost_var = f"{asset_name}__investment_cost"
            self._asset_investment_cost_map[asset_name] = asset_investment_cost_var
            self.__asset_investment_cost_var[asset_investment_cost_var] = ca.MX.sym(
                asset_investment_cost_var
            )

            if asset_name in self.heat_network_components.get("pipe", []):
                if asset_name in self.get_pipe_class_map().keys():
                    pipe_classes = self.get_pipe_class_map()[asset_name]
                    max_cost = (
                        2.0
                        * parameters[f"{asset_name}.length"]
                        * max([c.investment_costs for c in pipe_classes.keys()])
                    )
                else:
                    max_cost = (
                        2.0
                        * parameters[f"{asset_name}.length"]
                        * parameters[f"{asset_name}.investment_cost_coefficient"]
                    )
            else:
                max_cost = (
                    max(bounds[f"{asset_name}__max_size"][1].values)
                    * parameters[f"{asset_name}.investment_cost_coefficient"]
                    if isinstance(bounds[f"{asset_name}__max_size"][1], Timeseries)
                    else bounds[f"{asset_name}__max_size"][1]
                    * parameters[f"{asset_name}.investment_cost_coefficient"]
                )
            self.__asset_investment_cost_bounds[asset_investment_cost_var] = (0.0, max_cost)
            self.__asset_investment_cost_nominals[asset_investment_cost_var] = (
                max(
                    parameters[f"{asset_name}.investment_cost_coefficient"] * nominal_investment,
                    1.0e2,
                )
                if nominal_investment is not None
                else 1.0e2
            )

        if options["include_asset_is_realized"]:
            for asset in [
                *self.heat_network_components.get("source", []),
                *self.heat_network_components.get("demand", []),
                *self.heat_network_components.get("ates", []),
                *self.heat_network_components.get("buffer", []),
                *self.heat_network_components.get("heat_exchanger", []),
                *self.heat_network_components.get("heat_pump", []),
            ]:
                var_name = f"{asset}__cumulative_investments_made_in_eur"
                self.__cumulative_investments_made_in_eur_map[asset] = var_name
                self.__cumulative_investments_made_in_eur_var[var_name] = ca.MX.sym(var_name)
                self.__cumulative_investments_made_in_eur_nominals[
                    var_name
                ] = self.variable_nominal(f"{asset}__investment_cost") + self.variable_nominal(
                    f"{asset}__installation_cost"
                )
                self.__cumulative_investments_made_in_eur_bounds[var_name] = (0.0, np.inf)

                # This is an integer variable between [0, max_aggregation_count] that allows the
                # increments of the asset to become used by the optimizer. Meaning that when this
                # variable is zero not heat can be consumed or produced by this asset. When the
                # integer is >=1 the asset can consume and/or produce according to its increments.
                var_name = f"{asset}__asset_is_realized"
                self.__asset_is_realized_map[asset] = var_name
                self.__asset_is_realized_var[var_name] = ca.MX.sym(var_name)
                try:
                    aggr_count_max = parameters[f"{asset}.nr_of_doublets"]
                except KeyError:
                    aggr_count_max = 1.0
                if parameters[f"{asset}.state"] == 0:
                    aggr_count_max = 0.0
                self.__asset_is_realized_bounds[var_name] = (0.0, aggr_count_max)

    def heat_network_options(self):
        r"""
        Returns a dictionary of heat network specific options.
        """

        options = super().heat_network_options()

        options["include_asset_is_realized"] = False

        return options

    @abstractmethod
    def get_max_size_var(self, asset_name, ensemble_member):
        """
        This function should return the max size variable of an asset.

        Returns
        -------

        """
        raise NotImplementedError

    @abstractmethod
    def get_aggregation_count_var(self, asset_name, ensemble_member):
        """
        This function should return the aggregation count integer variable of an asset.

        Returns
        -------

        """
        raise NotImplementedError

    @abstractmethod
    def get_aggregation_count_max(self, asset_name):
        """
        This function should return the aggregation count upper bound.

        Returns
        -------

        """
        raise NotImplementedError

    @abstractmethod
    def get_pipe_investment_cost_coefficient(self, asset_name, ensemble_member):
        """
        This function should return the pipe investment cost coefficient variable.

        Returns
        -------

        """
        raise NotImplementedError

    @abstractmethod
    def get_pipe_class_map(self):
        """
        This function should return the mapping between the pipe and all the possible pipe classes
        available for that pipe.

        Returns
        -------

        """
        return NotImplementedError

    @property
    def extra_variables(self):
        """
        In this function we add all the variables defined in the HeatMixin to the optimization
        problem. Note that these are only the normal variables not path variables.
        """
        variables = super().extra_variables.copy()
        variables.extend(self.__asset_fixed_operational_cost_var.values())
        variables.extend(self.__asset_investment_cost_var.values())
        variables.extend(self.__asset_installation_cost_var.values())
        variables.extend(self.__asset_variable_operational_cost_var.values())
        return variables

    @property
    def path_variables(self):
        """
        In this function we add all the path variables defined in the HeatMixin to the
        optimization problem. Note that path_variables are variables that are created for each
        time-step.
        """
        variables = super().path_variables.copy()
        variables.extend(self.__cumulative_investments_made_in_eur_var.values())
        variables.extend(self.__asset_is_realized_var.values())
        return variables

    def variable_is_discrete(self, variable):
        """
        All variables that only can take integer values should be added to this function.
        """
        if variable in self.__asset_is_realized_var:
            return True
        else:
            return super().variable_is_discrete(variable)

    def variable_nominal(self, variable):
        """
        In this function we add all the nominals for the variables defined/added in the HeatMixin.
        """
        if variable in self.__asset_fixed_operational_cost_nominals:
            return self.__asset_fixed_operational_cost_nominals[variable]
        elif variable in self.__asset_investment_cost_nominals:
            return self.__asset_investment_cost_nominals[variable]
        elif variable in self.__asset_variable_operational_cost_nominals:
            return self.__asset_variable_operational_cost_nominals[variable]
        elif variable in self.__asset_installation_cost_nominals:
            return self.__asset_installation_cost_nominals[variable]
        elif variable in self.__cumulative_investments_made_in_eur_nominals:
            return self.__cumulative_investments_made_in_eur_nominals[variable]
        else:
            return super().variable_nominal(variable)

    def bounds(self):
        """
        In this function we add the bounds to the problem for all the variables defined/added in
        the HeatMixin.
        """
        bounds = super().bounds()
        bounds.update(self.__asset_fixed_operational_cost_bounds)
        bounds.update(self.__asset_investment_cost_bounds)
        bounds.update(self.__asset_installation_cost_bounds)
        bounds.update(self.__asset_variable_operational_cost_bounds)
        bounds.update(self.__asset_is_realized_bounds)
        bounds.update(self.__cumulative_investments_made_in_eur_bounds)
        return bounds

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

    def __state_vector_scaled(self, variable, ensemble_member):
        """
        This functions returns the casadi symbols scaled with their nominal for the entire time
        horizon.
        """
        canonical, sign = self.alias_relation.canonical_signed(variable)
        return (
            self.state_vector(canonical, ensemble_member) * self.variable_nominal(canonical) * sign
        )

    def __investment_cost_constraints(self, ensemble_member):
        """
        This function adds constraints to set the investment cost variable. The investment cost
        scales with the maximum size of the asset. This leaves two cases for the constraint. 1) The
        asset size is fixed (state==1): in this case the investment cost is set based on the upper
        bound of the size. 2) The asset size is optimized (state==2): in this case the investment
        cost is set based upon the __max_size variable.

        Specifically for demands we have a case where we set the investment cost based on the
        maximum demand as often the size of the demand is not seperately specified.

        For pipes the investment cost is set based on the pipe class and the length.
        """
        constraints = []

        parameters = self.parameters(ensemble_member)

        for asset_name in [
            asset_name
            for asset_name_list in self.heat_network_components.values()
            for asset_name in asset_name_list
        ]:
            if asset_name in [
                *self.heat_network_components.get("node", []),
                *self.heat_network_components.get("pump", []),
                *self.heat_network_components.get("check_valve", []),
                *self.heat_network_components.get("electricity_cable", []),
                *self.heat_network_components.get("electricity_node", []),
                *self.heat_network_components.get("electricity_source", []),
                *self.heat_network_components.get("electricity_demand", []),
                *self.heat_network_components.get("gas_pipe", []),
                *self.heat_network_components.get("gas_node", []),
                *self.heat_network_components.get("gas_source", []),
                *self.heat_network_components.get("gas_demand", []),
            ]:
                # TODO: add support for joints?
                continue

            investment_cost_var = self._asset_investment_cost_map[asset_name]
            investment_costs = self.extra_variable(investment_cost_var, ensemble_member)
            investment_cost_coefficient = parameters[f"{asset_name}.investment_cost_coefficient"]
            nominal = self.variable_nominal(investment_cost_var)

            if asset_name in [*self.heat_network_components.get("pipe", [])]:
                # We do the pipe seperately as their coefficients are specified per meter.
                investment_cost_coefficient = self.get_pipe_investment_cost_coefficient(asset_name, ensemble_member)
                asset_size = parameters[f"{asset_name}.length"]
            else:
                asset_size = self.get_max_size_var(asset_name, ensemble_member)

            constraints.append(
                (
                    (investment_costs - asset_size * investment_cost_coefficient) / nominal,
                    0.0,
                    0.0,
                )
            )

        return constraints

    def __fixed_operational_cost_constraints(self, ensemble_member):
        """
        This function adds the constraints to set the fixed operational cost. The fixed operational
        cost are the cost made independently of the operation of the asset. We assume that these
        cost scale with the maximum size of the asset.
        """
        constraints = []

        parameters = self.parameters(ensemble_member)

        for asset_name in [
            asset_name
            for asset_name_list in self.heat_network_components.values()
            for asset_name in asset_name_list
        ]:
            if asset_name in [
                *self.heat_network_components.get("node", []),
                *self.heat_network_components.get("pipe", []),
                *self.heat_network_components.get("electricity_cable", []),
                *self.heat_network_components.get("electricity_node", []),
                *self.heat_network_components.get("electricity_demand", []),
                *self.heat_network_components.get("electricity_source", []),
                *self.heat_network_components.get("gas_pipe", []),
                *self.heat_network_components.get("gas_node", []),
                *self.heat_network_components.get("gas_demand", []),
                *self.heat_network_components.get("gas_source", []),
                *self.heat_network_components.get("pump", []),
                *self.heat_network_components.get("check_valve", []),
            ]:
                # currently no support for joints
                continue
            fixed_operational_cost_var = self._asset_fixed_operational_cost_map[asset_name]
            fixed_operational_cost = self.extra_variable(
                fixed_operational_cost_var, ensemble_member
            )
            asset_size = self.get_max_size_var(asset_name, ensemble_member)
            fixed_operational_cost_coefficient = parameters[
                f"{asset_name}.fixed_operational_cost_coefficient"
            ]

            nominal = self.variable_nominal(fixed_operational_cost_var)

            constraints.append(
                (
                    (fixed_operational_cost - asset_size * fixed_operational_cost_coefficient)
                    / nominal,
                    0.0,
                    0.0,
                )
            )

        return constraints

    def __variable_operational_cost_constraints(self, ensemble_member):
        """
        This function adds the constraints for setting the variable operational cost. These are the
        cost that depend on the operation of the asset. At this moment we only support the variable
        operational cost for sources where they scale with the thermal energy production.
        """
        constraints = []

        parameters = self.parameters(ensemble_member)

        for s in self.heat_network_components.get("source", []):
            heat_source = self.__state_vector_scaled(f"{s}.Heat_source", ensemble_member)
            variable_operational_cost_var = self._asset_variable_operational_cost_map[s]
            variable_operational_cost = self.extra_variable(
                variable_operational_cost_var, ensemble_member
            )
            nominal = self.variable_nominal(variable_operational_cost_var)
            variable_operational_cost_coefficient = parameters[
                f"{s}.variable_operational_cost_coefficient"
            ]
            timesteps = np.diff(self.times()) / 3600.0

            sum = 0.0
            for i in range(1, len(self.times())):
                sum += variable_operational_cost_coefficient * heat_source[i] * timesteps[i - 1]
            constraints.append(((variable_operational_cost - sum) / (nominal), 0.0, 0.0))

        for _ in self.heat_network_components.get("buffer", []):
            pass

        # for a in self.heat_network_components.get("ates", []):
        # TODO: needs to be replaced with the positive or abs value of this, see varOPEX,
        #  then ates varopex also needs to be added to the mnimize_tco_goal
        # heat_ates = self.__state_vector_scaled(f"{a}.Heat_ates", ensemble_member)
        # variable_operational_cost_var = self._asset_variable_operational_cost_map[a]
        # variable_operational_cost = self.extra_variable(
        #     variable_operational_cost_var, ensemble_member
        # )
        # nominal = self.variable_nominal(variable_operational_cost_var)
        # variable_operational_cost_coefficient = parameters[
        #     f"{a}.variable_operational_cost_coefficient"
        # ]
        # timesteps = np.diff(self.times()) / 3600.0
        #
        # sum = 0.0
        #
        # for i in range(1, len(self.times())):
        #     varOPEX_dt = (variable_operational_cost_coefficient * heat_ates[i]
        #     * timesteps[i - 1])
        #     constraints.append(((varOPEX-varOPEX_dt)/nominal,0.0, np,inf))
        #     #varOPEX would be a variable>0 for everyt timestep
        #     sum += varOPEX
        # constraints.append(((variable_operational_cost - sum) / (nominal), 0.0, 0.0))

        return constraints

    def __installation_cost_constraints(self, ensemble_member):
        """
        This function adds the constraints for setting the installation cost variable. The
        installation cost is the cost element that comes with the placing of the asset
        independently of the size of the asset. Therefore, the installation cost is set with the
        _aggregation_count variable.
        """
        constraints = []

        parameters = self.parameters(ensemble_member)

        for asset_name in [
            asset_name
            for asset_name_list in self.heat_network_components.values()
            for asset_name in asset_name_list
        ]:
            if asset_name in [
                *self.heat_network_components.get("node", []),
                *self.heat_network_components.get("pump", []),
                *self.heat_network_components.get("check_valve", []),
                *self.heat_network_components.get("electricity_cable", []),
                *self.heat_network_components.get("electricity_node", []),
                *self.heat_network_components.get("electricity_source", []),
                *self.heat_network_components.get("electricity_demand", []),
                *self.heat_network_components.get("gas_pipe", []),
                *self.heat_network_components.get("gas_node", []),
                *self.heat_network_components.get("gas_source", []),
                *self.heat_network_components.get("gas_demand", []),
            ]:
                # no support for joints right now
                continue
            installation_cost_sym = self.extra_variable(
                self._asset_installation_cost_map[asset_name]
            )
            nominal = self.variable_nominal(self._asset_installation_cost_map[asset_name])
            installation_cost = parameters[f"{asset_name}.installation_cost"]
            aggregation_count_sym = self.get_aggregation_count_var(asset_name, ensemble_member)
            constraints.append(
                (
                    (installation_cost_sym - aggregation_count_sym * installation_cost) / nominal,
                    0.0,
                    0.0,
                )
            )

        return constraints

    def __cumulative_investments_made_in_eur_path_constraints(self, ensemble_member):
        r"""
        These constraints are linking the cummulitive investments made to the possiblity of
        utilizing an asset. The investments made are sufficient for that asset to be realized it
        becomes available.

        Meaning that an asset requires 1million euro investment to be realized
        and the investments made at timestep i are sufficient the asset also is realized (becomes
        available) in that same timestep.
        """
        constraints = []
        options = self.heat_network_options()
        if options["include_asset_is_realized"]:
            for asset in [
                *self.heat_network_components.get("demand", []),
                *self.heat_network_components.get("source", []),
                *self.heat_network_components.get("ates", []),
                *self.heat_network_components.get("buffer", []),
                *self.heat_network_components.get("heat_exchanger", []),
                *self.heat_network_components.get("heat_pump", []),
            ]:
                var_name = self.__cumulative_investments_made_in_eur_map[asset]
                cumulative_investments_made = self.state(var_name)
                nominal = self.variable_nominal(var_name)
                var_name = self.__asset_is_realized_map[asset]
                asset_is_realized = self.state(var_name)
                installation_cost_sym = self.__asset_installation_cost_var[
                    self._asset_installation_cost_map[asset]
                ]
                investment_cost_sym = self.__asset_investment_cost_var[
                    self._asset_investment_cost_map[asset]
                ]
                # TODO: add insulation class cost to the investments made.
                # if asset in self.heat_network_components.get("demand", []):
                #     for insulation_class in self.__get_insulation_classes(asset):
                #         insulation_class_active
                #         insulation_class_cost
                #         investment_cost_sym += insulation_class_active * insulation_class_cost
                big_m = (
                    1.5
                    * max(
                        self.bounds()[f"{asset}__investment_cost"][1]
                        + self.bounds()[f"{asset}__installation_cost"][1],
                        1.0,
                    )
                    / max(self.get_aggregation_count_max(asset), 1.0)
                )

                # Asset can be realized once the investments made equal the installation and
                # investment cost
                capex_sym = 0.0
                if self.variable_nominal(self._asset_installation_cost_map[asset]) > 1.0e2:
                    capex_sym = capex_sym + installation_cost_sym
                if self.variable_nominal(self._asset_investment_cost_map[asset]) > 1.0e2:
                    capex_sym = capex_sym + investment_cost_sym

                constraints.append(
                    (
                        (
                            cumulative_investments_made
                            - capex_sym
                            + (1.0 - asset_is_realized) * big_m
                        )
                        / nominal,
                        0.0,
                        np.inf,
                    )
                )

                # Once the asset is utilized the asset must be realized
                heat_flow = self.state(f"{asset}.Heat_flow")
                if not np.isinf(self.bounds()[f"{asset}.Heat_flow"][1]):
                    big_m = (
                        1.5
                        * self.bounds()[f"{asset}.Heat_flow"][1]
                        / max(self.get_aggregation_count_max(asset), 1.0)
                    )
                else:
                    try:
                        big_m = (
                            1.5
                            * max(
                                self.bounds()[f"{asset}.HeatOut.Heat"][1],
                                self.bounds()[f"{asset}.HeatIn.Heat"][1],
                            )
                            / max(self.get_aggregation_count_max(asset), 1.0)
                        )
                    except KeyError:
                        big_m = (
                            1.5
                            * max(
                                self.bounds()[f"{asset}.Primary.HeatOut.Heat"][1],
                                self.bounds()[f"{asset}.Primary.HeatIn.Heat"][1],
                            )
                            / max(self.get_aggregation_count_max(asset), 1.0)
                        )
                constraints.append(((heat_flow + asset_is_realized * big_m) / big_m, 0.0, np.inf))
                constraints.append(((heat_flow - asset_is_realized * big_m) / big_m, -np.inf, 0.0))

        return constraints

    def path_constraints(self, ensemble_member):
        """
        Here we add all the path constraints to the optimization problem. Please note that the
        path constraints are the constraints that are applied to each time-step in the problem.
        """

        constraints = super().path_constraints(ensemble_member)

        constraints.extend(
            self.__cumulative_investments_made_in_eur_path_constraints(ensemble_member)
        )

        return constraints

    def constraints(self, ensemble_member):
        """
        This function adds the normal constraints to the problem. Unlike the path constraints these
        are not applied to every time-step in the problem. Meaning that these constraints either
        consider global variables that are independent of time-step or that the relevant time-steps
        are indexed within the constraint formulation.
        """
        constraints = super().constraints(ensemble_member)

        constraints.extend(self.__variable_operational_cost_constraints(ensemble_member))
        constraints.extend(self.__fixed_operational_cost_constraints(ensemble_member))
        constraints.extend(self.__investment_cost_constraints(ensemble_member))
        constraints.extend(self.__installation_cost_constraints(ensemble_member))

        return constraints

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
