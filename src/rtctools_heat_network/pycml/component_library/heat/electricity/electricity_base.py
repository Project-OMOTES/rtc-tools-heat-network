from rtctools_heat_network.pycml import Connector, Variable

from .._internal.electricity_component import ElectricityComponent


class ElectricityPort(ElectricityComponent, Connector):
    def __init__(self, name, **modifiers):
        super().__init__(name, **modifiers)

        self.add_variable(Variable, "Power")
        self.add_variable(Variable, "V", min=0.0)
        self.add_variable(Variable, "I")


class ElectricityTwoPort(ElectricityComponent):
    def __init__(self, name, **modifiers):
        super().__init__(name, **modifiers)

        self.add_variable(ElectricityPort, "ElectricityIn")
        self.add_variable(ElectricityPort, "ElectricityOut")
