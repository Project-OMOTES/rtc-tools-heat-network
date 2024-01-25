from typing import Any, Dict, List

import esdl

import numpy as np


from rtctools.optimization.goal_programming_mixin import Goal
from rtctools.util import run_optimization_problem

from rtctools_heat_network.workflows.goals.minimize_tco_goal import MinimizeTCO


try:
    from models.test_case_small_network_optional_assets_annualized.src.run_ates import (
        HeatProblem,
    )
except ModuleNotFoundError:
    from run_ates import HeatProblem


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


class _GoalsAndOptions:
    def path_goals(self) -> List[Goal]:
        goals = super().path_goals().copy()

        for demand in self.heat_network_components["demand"]:
            target = self.get_timeseries(f"{demand}.target_heat_demand")
            state = f"{demand}.Heat_demand"

            goals.append(TargetDemandGoal(state, target))

        return goals


class HeatProblemDiscAnnualizedCost(HeatProblem):
    def heat_network_options(self) -> Dict[str, Any]:
        options = super().heat_network_options()
        options["discounted_annualized_cost"] = True
        return options

    def goals(self) -> List[Goal]:
        goals = super().goals().copy()

        custom_asset_type_maps = {
            "operational": {"source"},
            "fixed_operational": {"source"},
            "annualized": {"source"},
        }
        goals.append(MinimizeTCO(priority=2, custom_asset_type_maps=custom_asset_type_maps))
        return goals

    def modify_discount_rate(self, assets):
        for asset in assets.values():
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
        HeatProblemDiscAnnualizedCostModifiedParam,
        base_folder=base_folder,
    )
    results = solution.extract_results()
    print("\n HeatProblemAnnualized Completed \n \n")
