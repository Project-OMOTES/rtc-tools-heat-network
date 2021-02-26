from rtctools_heat_network.pycml import Variable

from ._non_storage_component import _NonStorageComponent


class Pump(_NonStorageComponent):
    def __init__(self, name, **modifiers):
        super().__init__(name, **modifiers)

        self.component_type = "pump"

        self.add_variable(Variable, "dH", min=0.0)

        self.add_equation(self.dH - (self.HeatOut.H - self.HeatIn.H))

        self.add_equation((self.HeatIn.Heat - self.HeatOut.Heat) / self.Heat_nominal)
