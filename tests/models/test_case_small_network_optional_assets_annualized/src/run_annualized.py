from typing import Any, Dict, List

import esdl

import numpy as np

from rtctools.data.storage import DataStore
from rtctools.optimization.collocated_integrated_optimization_problem import (
    CollocatedIntegratedOptimizationProblem,
)
from rtctools.optimization.goal_programming_mixin import Goal, GoalProgrammingMixin
from rtctools.optimization.linearized_order_goal_programming_mixin import (
    LinearizedOrderGoalProgrammingMixin,
)
from rtctools.util import run_optimization_problem

from rtctools_heat_network.esdl.esdl_mixin import ESDLMixin
from rtctools_heat_network.heat_mixin import HeatMixin


class TargetDemandGoal(Goal):
    priority = 1
    order = 2

    def __init__(self, state, target):
        self.state = state

        self.target_min = target
        self.target_max = target
        self.function_range = (-1.0, 2.0 * max(target.values))
        self.function_nominal = np.median(target.values)

    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.state(self.state)


class MinimizeDiscAnnualizedCostGoal(Goal):

    """
    This class represents an optimization goal that minimizes the
    discounted annualized cost of a heat network model.

    It takes the annualized cost calculated by the model and minimizes
    it by changing optimization variables such as asset sizes.

    Attributes:
        order (int): The order of the goal.
        priority (int): The priority of the goal.
        assets_and_costs_keys (list): List of tuples mapping asset categories to cost map keys.
    """

    order = 1
    priority = 2

    def __init__(self, assets_and_costs_keys=None):
        self.target_max = 0.0
        self.function_range = (0.0, 1.0e8)
        self.function_nominal = 1.0e7
        self.assets_and_costs_keys = (
            assets_and_costs_keys
            if assets_and_costs_keys is not None
            else [
                (["source", "ates"], ["_asset_variable_operational_cost_map"]),
                (["source", "ates", "buffer"], ["_asset_fixed_operational_cost_map"]),
                (
                    [
                        "source",
                        "ates",
                        "buffer",
                        "demand",
                        "heat_exchanger",
                        "heat_pump",
                        "pipe",
                    ],
                    ["_annualized_capex_var_map"],
                ),
            ]
        )

    def heat_network_options(self) -> Dict[str, Any]:
        options = super().heat_network_options()
        options["discounted_annualized_cost"] = True
        return options

    def function(self, optimization_problem: HeatMixin, ensemble_member):
        obj = 0.0
        """
        For the given optimization problem, this function 
        sums up the costs associated with specified assets in 
        given asset categories, using defined cost map keys.

        """

        for asset_categories, cost_map_keys in self.assets_and_costs_keys:
            for asset_category in asset_categories:
                for asset in optimization_problem.heat_network_components.get(asset_category, []):
                    for cost_map_key in cost_map_keys:
                        cost_map = getattr(optimization_problem, cost_map_key)
                        cost = cost_map.get(asset, 0)
                        obj += optimization_problem.extra_variable(cost)
        return obj


class _GoalsAndOptions:
    def path_goals(self) -> List[Goal]:
        goals = super().path_goals().copy()

        for demand in self.heat_network_components["demand"]:
            target = self.get_timeseries(f"{demand}.target_heat_demand")
            state = f"{demand}.Heat_demand"

            goals.append(TargetDemandGoal(state, target))

        return goals


class HeatProblem(
    _GoalsAndOptions,
    HeatMixin,
    LinearizedOrderGoalProgrammingMixin,
    GoalProgrammingMixin,
    ESDLMixin,
    CollocatedIntegratedOptimizationProblem,
):
    def path_goals(self) -> List[Goal]:
        goals = super().path_goals().copy()

        return goals

    def heat_network_options(self) -> Dict[str, Any]:
        options = super().heat_network_options()
        options["minimum_velocity"] = 0.0
        options["neglect_pipe_heat_losses"] = True
        options["heat_loss_disconnected_pipe"] = False
        return options

    def constraints(self, ensemble_member):
        constraints = super().constraints(ensemble_member)

        # By default we do not add any constraints on the cyclic behaviour of the ates, as we
        # might want to do optimization over shorter periods of time where this would lead to
        # infeasibility. In this case we do want the cyclic behaviour, therefore we add it to the
        # problem.
        for a in self.heat_network_components.get("ates", []):
            stored_heat = self.state_vector(f"{a}.Stored_heat")
            constraints.append(((stored_heat[0] - stored_heat[-1]), 0.0, 0.0))

        return constraints

    def solver_options(self) -> Dict[str, str]:
        options = super().solver_options()
        options["solver"] = "highs"

        return options

    def read(self) -> None:
        """
        Reads the yearly profile with hourly time steps and adapt to a daily averaged profile
        """
        super().read()

        demands = self.heat_network_components.get("demand", [])
        new_datastore = DataStore(self)
        new_datastore.reference_datetime = self.io.datetimes[0]

        for ensemble_member in range(self.ensemble_size):
            nr_of_days = 10
            new_date_times = list()
            for day in range(nr_of_days):
                new_date_times.append(self.io.datetimes[day * 24])
            new_date_times = np.asarray(new_date_times)

            for demand in demands:
                var_name = f"{demand}.target_heat_demand"
                data = self.get_timeseries(
                    variable=var_name, ensemble_member=ensemble_member
                ).values
                new_data = list()
                for day in range(nr_of_days):
                    data_for_day = data[day * 24 : (day + 1) * 24]
                    new_data.append(np.mean(data_for_day))
                new_datastore.set_timeseries(
                    variable=var_name,
                    datetimes=new_date_times,
                    values=np.asarray(new_data),
                    ensemble_member=ensemble_member,
                    check_duplicates=True,
                )

            self.io = new_datastore


class HeatProblemDiscAnnualizedCost(HeatProblem):
    def goals(self) -> List[Goal]:
        goals = super().goals().copy()

        goals.append(MinimizeDiscAnnualizedCostGoal())
        return goals

    def modify_discount_rate(self, assets):
        for asset in assets.values():
            # if asset.asset_type == "Pipe" or asset.asset_type == "HeatProducer":
            if asset.asset_type == "HeatProducer":
                if "costInformation" in asset.attributes and (
                    asset.attributes["costInformation"].discountRate is not None
                    and asset.attributes["costInformation"].discountRate.value is not None
                ):
                    asset.attributes["costInformation"].discountRate.value = 0.0
                else:
                    asset.attributes["costInformation"].discountRate = esdl.SingleValue(value=0.0)
        return assets


class HeatProblemDiscAnnualizedCostModifiedParam(HeatProblemDiscAnnualizedCost):
    @property
    def esdl_assets(self):
        assets = super().esdl_assets
        for asset in assets.values():
            # if asset.asset_type == "Pipe" or asset.asset_type == "HeatProducer":
            if asset.asset_type == "HeatProducer":
                asset.attributes["technicalLifetime"] = 1.0
        assets = self.modify_discount_rate(assets)
        return assets


class HeatProblemDiscAnnualizedCostModifiedDiscountRate(HeatProblemDiscAnnualizedCost):
    @property
    def esdl_assets(self):
        assets = super().esdl_assets
        assets = self.modify_discount_rate(assets)
        return assets


if __name__ == "__main__":
    from pathlib import Path

    base_folder = Path(__file__).resolve().parent.parent
    solution = run_optimization_problem(
        # HeatProblemDiscAnnualizedCost, base_folder=base_folder
        HeatProblemDiscAnnualizedCostModifiedParam,
        base_folder=base_folder,
    )
    results = solution.extract_results()
    print("\n HeatProblemAnnualized Completed \n \n")
