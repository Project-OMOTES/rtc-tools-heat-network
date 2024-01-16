from numpy import nan

from rtctools_heat_network.pycml import Variable

from .electricity_base import ElectricityPort
from .. import ElectricitySource
from .._internal import BaseAsset
from .._internal.electricity_component import ElectricityComponent


class WindPark(ElectricitySource):

    def __init__(self, name, **modifiers):
        super().__init__(name, **modifiers)

        self.component_subtype = "wind_park"

        self.add_variable(Variable, "Set_point", min=0.0, max=1.0)

