from typing import Dict, Set

from rtctools.optimization.goal_programming_mixin_base import Goal

from rtctools_heat_network.techno_economic_mixin import TechnoEconomicMixin


class MinimizeTCO(Goal):
    """
    Minimize the Total Cost of Ownership (TCO) for a heat network.

    This goal aims to minimize the sum of operational, fixed operational,
    investment, and installation costs over a specified
    number of years.
    """

    order = 1
    NOMINAL = 1.0e6

    def __init__(self, priority: int = 2, number_of_years: float = 25.0):
        """
        Initialize the MinimizeTCO goal.

        Args:
            priority (int): The priority of this goal.
            number_of_years (float): The number of years over which to calculate the costs.
        """
        self.priority = priority
        self.number_of_years = number_of_years

    def _calculate_cost(
        self,
        optimization_problem: TechnoEconomicMixin,
        asset_types: Set[str],
        cost_map: Dict[str, float],
    ) -> float:
        """
        Calculate the cost for given asset types using a specified cost map.

        Args:
            optimization_problem (TechnoEconomicMixin): The optimization problem instance.
            asset_types (Set[str]): Set of asset types to consider for cost calculation.
            cost_map (Dict[str, float]): Mapping of assets to their respective costs.

        Returns:
            float: The total cost for the given asset types.
        """
        obj = 0.0
        for asset_type in asset_types:
            for asset in optimization_problem.heat_network_components.get(asset_type, []):
                obj += optimization_problem.extra_variable(cost_map[asset]) * self.number_of_years
        return obj

    def function(self, optimization_problem: TechnoEconomicMixin, ensemble_member) -> float:
        """
        Calculate the objective function value for the optimization problem.

        This method sums up the various costs associated with the heat network assets.

        Args:
            optimization_problem (TechnoEconomicMixin): The optimization problem instance.
            ensemble_member: The current ensemble member being considered in the optimization.

        Returns:
            float: The total cost objective function value in millions.
        """
        asset_types = {"source", "ates", "buffer", "demand", "heat_exchanger", "heat_pump", "pipe"}
        operational_cost_asset_types = {"source", "ates"}
        fixed_operational_cost_asset_types = {"source", "ates", "buffer"}
        investment_cost_asset_types = asset_types
        installation_cost_asset_types = asset_types

        cost_type_maps = {
            "operational": optimization_problem._asset_variable_operational_cost_map,
            "fixed_operational": optimization_problem._asset_fixed_operational_cost_map,
            "investment": optimization_problem._asset_investment_cost_map,
            "installation": optimization_problem._asset_installation_cost_map,
        }

        asset_type_maps = {
            "operational": operational_cost_asset_types,
            "fixed_operational": fixed_operational_cost_asset_types,
            "investment": investment_cost_asset_types,
            "installation": installation_cost_asset_types,
        }

        obj = 0.0
        for cost_type in asset_type_maps.keys():
            obj += self._calculate_cost(
                optimization_problem,
                asset_type_maps[cost_type],
                cost_type_maps[cost_type],
            )

        return obj / self.NOMINAL
