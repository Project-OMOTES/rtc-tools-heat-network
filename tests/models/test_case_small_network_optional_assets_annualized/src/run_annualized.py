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


class CommonCostGoal(Goal):
    order = 1

    def __init__(self, priority=2, number_of_years=30.0):
        self.priority = priority
        self.number_of_years = number_of_years

    def function(self, optimization_problem: HeatMixin, ensemble_member):
        obj = 0.0
        asset_categories = ["source", "ates"]
        cost_map_keys = ["_asset_variable_operational_cost_map"]
        obj += self.sum_cost(optimization_problem, asset_categories, cost_map_keys)

        asset_categories = ["source", "ates", "buffer"]
        cost_map_keys = ["_asset_fixed_operational_cost_map"]
        obj += self.sum_cost(optimization_problem, asset_categories, cost_map_keys)

        asset_categories = ["source", "ates", "buffer", "demand", "heat_exchanger", "heat_pump", "pipe"]
        obj += self.investment_cost(optimization_problem, asset_categories)
        return obj / 1.0e6

    def sum_cost(self, optimization_problem, asset_categories, cost_map_keys, divide_by_years=False):
        obj = 0.0
        for asset_category in asset_categories:
            for asset in optimization_problem.heat_network_components.get(asset_category, []):
                for cost_map_key in cost_map_keys:
                    cost_map = getattr(optimization_problem, cost_map_key)
                    cost = cost_map.get(asset, 0)
                    obj += self.calculate_cost(optimization_problem, cost, divide_by_years)
        return obj

    def investment_cost(self, optimization_problem, asset_categories):
        pass

    def calculate_cost(self, optimization_problem, cost, divide_by_years):
        if divide_by_years:
            return optimization_problem.extra_variable(cost) / self.number_of_years
        else:
            return optimization_problem.extra_variable(cost)


class MinimizeNoDiscountedCostGoal(CommonCostGoal):
    def investment_cost(self, optimization_problem, asset_categories):
        cost_map_keys = ["_asset_installation_cost_map", "_asset_investment_cost_map"]
        return self.sum_cost(optimization_problem, asset_categories, cost_map_keys, True)


class MinimizeDiscAnnualizedCostGoal(CommonCostGoal):
    def investment_cost(self, optimization_problem, asset_categories):
        cost_map_keys = ["_annualized_capex_var_map"]
        return self.sum_cost(optimization_problem, asset_categories, cost_map_keys)


# class MinimizeNoDiscountedCostGoal(Goal):
#     order = 1

#     def __init__(self, priority=2, number_of_years=30.0):
#         self.priority = priority
#         self.number_of_years = number_of_years

#     def function(self, optimization_problem: HeatMixin, ensemble_member):
#         obj = 0.0

#         asset_categories = ["source", "ates"]
#         cost_map_keys = ["_asset_variable_operational_cost_map"]
#         obj += self.sum_cost(optimization_problem, asset_categories, cost_map_keys)

#         asset_categories = ["source", "ates", "buffer"]
#         cost_map_keys = ["_asset_fixed_operational_cost_map"]
#         obj += self.sum_cost(optimization_problem, asset_categories, cost_map_keys)

#         asset_categories = ["source", "ates", "buffer", "demand", "heat_exchanger", "heat_pump", "pipe"]
#         cost_map_keys = ["_asset_installation_cost_map", "_asset_investment_cost_map"]
#         obj += self.sum_cost(optimization_problem, asset_categories, cost_map_keys, True)

#         return obj / 1.0e6

#     def sum_cost(self, optimization_problem, asset_categories, cost_map_keys, divide_by_years=False):
#         obj = 0.0
#         for asset_category in asset_categories:
#             for asset in optimization_problem.heat_network_components.get(asset_category, []):
#                 for cost_map_key in cost_map_keys:
#                     cost_map = getattr(optimization_problem, cost_map_key)
#                     cost = cost_map.get(asset, 0)
#                     if divide_by_years:
#                         obj += optimization_problem.extra_variable(cost) / self.number_of_years
#                     else:
#                         obj += optimization_problem.extra_variable(cost)
#         return obj
    

# class MinimizeAnnualizedCostGoal(Goal):
#     order = 1

#     def __init__(self, priority=2, number_of_years=30.0):
#         self.priority = priority
#         self.number_of_years = number_of_years

#     def function(self, optimization_problem: HeatMixin, ensemble_member):
#         obj = 0.0

#         asset_categories = ["source", "ates"]
#         cost_map_keys = ["_asset_variable_operational_cost_map"]
#         obj += self.sum_cost(optimization_problem, asset_categories, cost_map_keys)

#         asset_categories = ["source", "ates", "buffer"]
#         cost_map_keys = ["_asset_fixed_operational_cost_map"]
#         obj += self.sum_cost(optimization_problem, asset_categories, cost_map_keys)

#         asset_categories = ["source", "ates", "buffer", "demand", "heat_exchanger", "heat_pump", "pipe"]
#         cost_map_keys = ["_annualized_capex_var_map"]
#         obj += self.sum_cost(optimization_problem, asset_categories, cost_map_keys)

#         return obj / 1.0e6

#     def sum_cost(self, optimization_problem, asset_categories, cost_map_keys, divide_by_years=False):
#         obj = 0.0
#         for asset_category in asset_categories:
#             for asset in optimization_problem.heat_network_components.get(asset_category, []):
#                 for cost_map_key in cost_map_keys:
#                     cost_map = getattr(optimization_problem, cost_map_key)
#                     cost = cost_map.get(asset, 0)
#                     if divide_by_years:
#                         obj += optimization_problem.extra_variable(cost) / self.number_of_years
#                     else:
#                         obj += optimization_problem.extra_variable(cost)
#         return obj


class _GoalsAndOptions:
    def path_goals(self):
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
    def path_goals(self):
        goals = super().path_goals().copy()

        return goals

    def heat_network_options(self):
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

    def solver_options(self):
        options = super().solver_options()
        options["solver"] = "highs"

        return options

    def read(self):
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

    def goals(self):
        goals = super().goals().copy()

        goals.append(MinimizeDiscAnnualizedCostGoal())

        return goals


class HeatProblemNoDiscTotalCost(HeatProblem):

    def goals(self):
        goals = super().goals().copy()

        goals.append(MinimizeNoDiscountedCostGoal())

        return goals


if __name__ == "__main__":
    from pathlib import Path

    base_folder = Path(__file__).resolve().parent.parent
    # solution = run_optimization_problem(HeatProblemNonAnnulized, base_folder=base_folder)
    # results = solution.extract_results()
    # print(results)
    # print('\n HeatProblemNonAnnulized Completed \n \n')
    
    solution = run_optimization_problem(HeatProblemDiscAnnualizedCost, base_folder=base_folder)
    results = solution.extract_results()
    print('\n HeatProblemAnnualized Completed \n \n')

    # solution = run_optimization_problem(HeatProblemNoDiscTotalCost, base_folder=base_folder)
    # results = solution.extract_results()
    # print('\n HeatProblemAnnualized Completed \n \n')
