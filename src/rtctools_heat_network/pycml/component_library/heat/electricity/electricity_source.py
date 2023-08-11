from numpy import nan

from rtctools_heat_network.pycml import Variable

from .electricity_base import ElectricityPort
from .._internal import BaseAsset
from .._internal.electricity_component import ElectricityComponent


class ElectricitySource(ElectricityComponent, BaseAsset):
    def __init__(self, name, **modifiers):
        super().__init__(name, **modifiers)

        self.component_type = "electricity_source"
        self.meret_place = 1

        self.price = nan

        self.add_variable(ElectricityPort, "ElectricityOut")
        self.add_variable(Variable, "Electricity_source", min=0.0)

        self.add_equation((self.ElectricityOut.Power - self.Electricity_source))
