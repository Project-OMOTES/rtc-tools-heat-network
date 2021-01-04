from rtctools_heat_network.pycml import Connector, Variable

from ._internal import HeatComponent


class HeatPort(HeatComponent, Connector):
    def __init__(self, name, **modifiers):
        super().__init__(name, **modifiers)

        self.add_variable(Variable, "Heat")
        self.add_variable(Variable, "Q")
        self.add_variable(Variable, "H")
