from rtctools_heat_network.pycml import Variable

from .gas_base import GasPort
from .._internal import BaseAsset
from .._internal.gas_component import GasComponent


class GasSource(GasComponent, BaseAsset):
    def __init__(self, name, **modifiers):
        super().__init__(name, **modifiers)

        self.component_type = "gas_source"

        self.min_head = 30.0

        self.add_variable(GasPort, "GasOut")
        self.add_variable(Variable, "Gas_source_flow", min=0.0)

        self.add_equation((self.GasOut.Q - self.Gas_source_flow))
