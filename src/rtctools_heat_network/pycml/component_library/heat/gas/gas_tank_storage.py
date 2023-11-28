import math

from numpy import nan

from rtctools_heat_network.pycml import Variable
from rtctools_heat_network.pycml.component_library.heat._internal import BaseAsset

from .._internal.gas_component import GasComponent
from .gas_base import GasPort


class GasTankStorage(GasComponent, BaseAsset):
    """

    """

    def __init__(self, name, **modifiers):
        super().__init__(name, **modifiers)

        self.component_type = "gas_tank_storage"

        self.min_head = 30.0

        self.add_variable(GasPort, "GasIn")
        self.add_variable(Variable, "Gas_tank_flow")

        self.add_equation((self.GasIn.Q - self.Gas_tank_flow))
