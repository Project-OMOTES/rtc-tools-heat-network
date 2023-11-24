from numpy import nan

from rtctools_heat_network.pycml import Variable

from ._non_storage_component import _NonStorageComponent


class Pump(_NonStorageComponent):
    """
    The pump component is there to add head to the flow. We assume head can only be added for
    positive flow.
    """

    def __init__(self, name, **modifiers):
        super().__init__(name, **modifiers)

        self.temperature = nan
        self.carrier_id = -1

        self.component_type = "pump"

        self.add_variable(Variable, "dH", min=0.0)

        self.add_equation(self.dH - (self.HeatOut.H - self.HeatIn.H))

        self.add_equation((self.HeatIn.Heat - self.HeatOut.Heat) / self.Heat_nominal)
