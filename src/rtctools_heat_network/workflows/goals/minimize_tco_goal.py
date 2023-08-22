from rtctools.optimization.goal_programming_mixin_base import Goal

from rtctools_heat_network.heat_mixin import HeatMixin


class MinimizeTCO(Goal):
    order = 1

    def __init__(
        self,
        priority=2,
        number_of_years=25.0,
    ):
        self.priority = priority
        self.number_of_years = number_of_years

    def function(self, optimization_problem: HeatMixin, ensemble_member):
        obj = 0.0

        for asset in [
            *optimization_problem.heat_network_components.get("source", []),
            *optimization_problem.heat_network_components.get("ates", []),
        ]:
            obj += (
                optimization_problem.extra_variable(
                    optimization_problem._asset_variable_operational_cost_map[asset]
                )
                * self.number_of_years
            )

        for asset in [
            *optimization_problem.heat_network_components.get("source", []),
            *optimization_problem.heat_network_components.get("ates", []),
            *optimization_problem.heat_network_components.get("buffer", []),
        ]:
            obj += (
                optimization_problem.extra_variable(
                    optimization_problem._asset_fixed_operational_cost_map[asset]
                )
                * self.number_of_years
            )

        for asset in [
            *optimization_problem.heat_network_components.get("source", []),
            *optimization_problem.heat_network_components.get("ates", []),
            *optimization_problem.heat_network_components.get("buffer", []),
            *optimization_problem.heat_network_components.get("demand", []),
            *optimization_problem.heat_network_components.get("heat_exchanger", []),
            *optimization_problem.heat_network_components.get("heat_pump", []),
            *optimization_problem.heat_network_components.get("pipe", []),
        ]:
            obj += optimization_problem.extra_variable(
                optimization_problem._asset_investment_cost_map[asset]
            )

        for asset in [
            *optimization_problem.heat_network_components.get("source", []),
            *optimization_problem.heat_network_components.get("ates", []),
            *optimization_problem.heat_network_components.get("buffer", []),
            *optimization_problem.heat_network_components.get("demand", []),
            *optimization_problem.heat_network_components.get("heat_exchanger", []),
            *optimization_problem.heat_network_components.get("heat_pump", []),
            *optimization_problem.heat_network_components.get("pipe", []),
        ]:
            obj += optimization_problem.extra_variable(
                optimization_problem._asset_installation_cost_map[asset]
            )

        return obj / 1.0e6
