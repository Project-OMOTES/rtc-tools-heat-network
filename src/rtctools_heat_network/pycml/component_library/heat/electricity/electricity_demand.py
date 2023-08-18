from rtctools_heat_network.pycml import Variable

from .electricity_base import ElectricityPort
from .._internal import BaseAsset
from .._internal.electricity_component import ElectricityComponent


class ElectricityDemand(ElectricityComponent, BaseAsset):
    def __init__(self, name, **modifiers):
        super().__init__(name, **modifiers)

        self.component_type = "electricity_demand"
        self.min_voltage = 230.0

        self.add_variable(ElectricityPort, "ElectricityIn")
        self.add_variable(Variable, "Electricity_demand", min=0.0)
        self.elec_power_nominal = self.Electricity_demand.nominal

        self.add_equation((self.ElectricityIn.Power - self.Electricity_demand))
