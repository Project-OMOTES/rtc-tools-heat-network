from ._internal import HeatComponent
from .heat_port import HeatPort


class HeatTwoPort(HeatComponent):
    def __init__(self, name, **modifiers):
        super().__init__(name, **modifiers)

        self.add_variable(HeatPort, "HeatIn")
        self.add_variable(HeatPort, "HeatOut")
