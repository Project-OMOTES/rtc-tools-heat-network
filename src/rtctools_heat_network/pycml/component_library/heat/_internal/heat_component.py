from rtctools_heat_network.pycml import Component


class HeatComponent(Component):
    def __init__(self, name, **modifiers):
        super().__init__(name, **modifiers)


class BaseAsset(Component):
    def __init__(self, name, **modifiers):
        super().__init__(name, **modifiers)
        self.state = 1

        self.variable_operational_cost_coefficient = 0.0
        self.fixed_operational_cost_coefficient = 0.0
        self.investment_cost_coefficient = 0.0
        self.installation_cost = 0.0
