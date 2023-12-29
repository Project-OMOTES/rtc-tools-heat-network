import logging
from typing import List

import casadi as ca

import numpy as np

from rtctools.optimization.collocated_integrated_optimization_problem import (
    CollocatedIntegratedOptimizationProblem,
)
from rtctools.optimization.timeseries import Timeseries



from .base_component_type_mixin import BaseComponentTypeMixin
from .demand_insulation_class import DemandInsulationClass


logger = logging.getLogger("rtctools_heat_network")


class ElectricityPhysicsMixin(BaseComponentTypeMixin, CollocatedIntegratedOptimizationProblem):

    def __init__(self, *args, **kwargs):
        """
        In this __init__ we prepare the dicts for the variables added by the HeatMixin class
        """

        super().__init__(*args, **kwargs)


    def pre(self):
        """
        In this pre method we fill the dicts initiated in the __init__. This means that we create
        the Casadi variables and determine the bounds, nominals and create maps for easier
        retrieving of the variables.
        """
        super().pre()



    def heat_network_options(self):
        r"""
        Returns a dictionary of heat network specific options.
        """

        options = {}

        return options

    def demand_insulation_classes(self, demand_insulation: str) -> List[DemandInsulationClass]:
        """
        If the returned List is:
        - empty: use the demand insualtion properties from the model
        - len() == 1: use these demand insualtion properties to overrule that of the model
        - len() > 1: decide between the demand insualtion class options.

        """
        return []

    @property
    def extra_variables(self):
        """
        In this function we add all the variables defined in the HeatMixin to the optimization
        problem. Note that these are only the normal variables not path variables.
        """
        variables = super().extra_variables.copy()

        return variables

    @property
    def path_variables(self):
        """
        In this function we add all the path variables defined in the HeatMixin to the
        optimization problem. Note that path_variables are variables that are created for each
        time-step.
        """
        variables = super().path_variables.copy()

        return variables

    def variable_is_discrete(self, variable):
        """
        All variables that only can take integer values should be added to this function.
        """

        return super().variable_is_discrete(variable)

    def variable_nominal(self, variable):
        """
        In this function we add all the nominals for the variables defined/added in the HeatMixin.
        """

        return super().variable_nominal(variable)

    def bounds(self):
        """
        In this function we add the bounds to the problem for all the variables defined/added in
        the HeatMixin.
        """
        bounds = super().bounds()

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

    def __electricity_node_heat_mixing_path_constraints(self, ensemble_member):
        """
        This function adds constraints for power/energy and current conservation at nodes/busses.
        """
        constraints = []

        for bus, connected_cables in self.heat_network_topology.busses.items():
            power_sum = 0.0
            i_sum = 0.0
            power_nominal = []
            i_nominal = []

            for i_conn, (_cable, orientation) in connected_cables.items():
                heat_conn = f"{bus}.ElectricityConn[{i_conn + 1}].Power"
                i_port = f"{bus}.ElectricityConn[{i_conn + 1}].I"
                power_sum += orientation * self.state(heat_conn)
                i_sum += orientation * self.state(i_port)
                power_nominal.append(self.variable_nominal(heat_conn))
                i_nominal.append(self.variable_nominal(i_port))

            power_nominal = np.median(power_nominal)
            constraints.append((power_sum / power_nominal, 0.0, 0.0))

            i_nominal = np.median(i_nominal)
            constraints.append((i_sum / i_nominal, 0.0, 0.0))

        return constraints

    def __electricity_cable_heat_mixing_path_constraints(self, ensemble_member):
        """
        This function adds constraints relating the electrical power to the current flowing through
        the cable. The power through the cable is limited by the maximum voltage and the actual
        current variable with an inequality constraint. This is done to allow power losses through
        the network. As the current and power are related with an equality constraint at the
        demands exactly matching the P = U*I equation, we allow the inequalities for the lines. By
        overestimating the power losses and voltage drops, together we ensure that U*I>P.


        Furthermore, the power loss is estimated by linearizing with the maximum current, meaning
        that we are always overestimating the power loss in the cable.
        """
        constraints = []
        parameters = self.parameters(ensemble_member)

        for cable in self.heat_network_components.get("electricity_cable", []):
            current = self.state(f"{cable}.ElectricityIn.I")
            power_in = self.state(f"{cable}.ElectricityIn.Power")
            power_out = self.state(f"{cable}.ElectricityOut.Power")
            power_loss = self.state(f"{cable}.Power_loss")
            r = parameters[f"{cable}.r"]
            i_max = parameters[f"{cable}.max_current"]
            v_nom = parameters[f"{cable}.nominal_voltage"]
            v_max = parameters[f"{cable}.max_voltage"]

            # Ensure that the current is sufficient to transport the power
            constraints.append(((power_in - current * v_max) / (i_max * v_max), -np.inf, 0.0))
            constraints.append(((power_out - current * v_max) / (i_max * v_max), -np.inf, 0.0))

            # Power loss constraint
            constraints.append(((power_loss - current * r * i_max) / (i_max * v_nom * r), 0.0, 0.0))

        return constraints

    def __electricity_demand_path_constraints(self, ensemble_member):
        """
        This function adds the constraints for the electricity commodity at the demand assets. We
        enforce that a minimum voltage is exactly met together with the power that is carried by
        the current. By fixing the voltage at the demand we ensure that at the demands
        P = U * I is met exactly at this point in the network and the power is conservatively
        in the cables at all locations in the network.
        """
        constraints = []
        parameters = self.parameters(ensemble_member)

        for elec_demand in [
            *self.heat_network_components.get("electricity_demand", []),
            *self.heat_network_components.get("heat_pump_elec", []),
        ]:
            min_voltage = parameters[f"{elec_demand}.min_voltage"]
            voltage = self.state(f"{elec_demand}.ElectricityIn.V")
            # to ensure that voltage entering is equal or larger than the minimum voltage
            constraints.append(((voltage - min_voltage) / min_voltage, 0.0, np.inf))

            power_nom = self.variable_nominal(f"{elec_demand}.ElectricityIn.Power")
            curr_nom = self.variable_nominal(f"{elec_demand}.ElectricityIn.I")
            power_in = self.state(f"{elec_demand}.ElectricityIn.Power")
            current_in = self.state(f"{elec_demand}.ElectricityIn.I")
            constraints.append(
                (
                    (power_in - min_voltage * current_in)
                    / (power_nom * curr_nom * min_voltage) ** 0.5,
                    0,
                    0,
                )
            )

        return constraints

    def path_constraints(self, ensemble_member):
        """
        Here we add all the path constraints to the optimization problem. Please note that the
        path constraints are the constraints that are applied to each time-step in the problem.
        """

        constraints = super().path_constraints(ensemble_member)

        constraints.extend(self.__electricity_demand_path_constraints(ensemble_member))
        constraints.extend(self.__electricity_node_heat_mixing_path_constraints(ensemble_member))
        constraints.extend(self.__electricity_cable_heat_mixing_path_constraints(ensemble_member))

        return constraints

    def constraints(self, ensemble_member):
        """
        This function adds the normal constraints to the problem. Unlike the path constraints these
        are not applied to every time-step in the problem. Meaning that these constraints either
        consider global variables that are independent of time-step or that the relevant time-steps
        are indexed within the constraint formulation.
        """
        constraints = super().constraints(ensemble_member)

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

