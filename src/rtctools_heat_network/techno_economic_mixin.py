import logging

from rtctools.optimization.collocated_integrated_optimization_problem import (
    CollocatedIntegratedOptimizationProblem,
)

from .asset_sizing_mixin import AssetSizingMixin
from .base_component_type_mixin import BaseComponentTypeMixin
from .financial_mixin import FinancialMixin
from .physics_mixin import PhysicsMixin

logger = logging.getLogger("rtctools_heat_network")


class TechnoEconomicMixin(
    FinancialMixin,
    AssetSizingMixin,
    PhysicsMixin,
    BaseComponentTypeMixin,
    CollocatedIntegratedOptimizationProblem,
):
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

    def get_max_size_var(self, asset_name, ensemble_member):
        return self.extra_variable(self._asset_max_size_map[asset_name], ensemble_member)

    def get_aggregation_count_var(self, asset_name, ensemble_member):
        return self.extra_variable(
            self._asset_aggregation_count_var_map[asset_name], ensemble_member
        )

    def get_aggregation_count_max(self, asset_name):
        return self.bounds()[self._asset_aggregation_count_var_map[asset_name]][1]

    def get_pipe_class_map(self):
        return self._pipe_topo_pipe_class_map

    def get_pipe_investment_cost_coefficient(self, asset_name, ensemble_member):
        return self.extra_variable(self._pipe_topo_cost_map[asset_name], ensemble_member)

    def heat_network_options(self):
        r"""
        Returns a dictionary of heat network specific options.
        """

        options = super(PhysicsMixin, self).heat_network_options()
        # problem with abstractmethod
        options["include_asset_is_realized"] = False

        return options

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

    def path_constraints(self, ensemble_member):
        """
        Here we add all the path constraints to the optimization problem. Please note that the
        path constraints are the constraints that are applied to each time-step in the problem.
        """

        constraints = super().path_constraints(ensemble_member)

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
