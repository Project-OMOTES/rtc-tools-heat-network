from rtctools_heat_network.pycml import Variable

from .electricity_base import ElectricityTwoPort
from .._internal import BaseAsset


class ElectricityCable(ElectricityTwoPort, BaseAsset):
    def __init__(self, name, **modifiers):
        super().__init__(name, **modifiers)

        self.component_type = "electricity_cable"
        self.disconnectable = False

        self.add_variable(Variable, "Power_loss")

        self.length = 1.0

        # Powerloss with inequality in the heat-mixin
        # values for NAYY 4x50 SE
        # from: https://pandapower.readthedocs.io/en/v2.6.0/std_types/basic.html
        self.max_current = 142.0
        self.max_voltage = 1000.0
        self.min_voltage = 230.0
        self.nominal_current = self.max_current / 2.0
        self.nominal_voltage = (self.max_voltage + self.min_voltage) / 2.0
        self.r = 1.0e-6 * self.length  # TODO: temporary value

        self.add_equation(
            (
                (self.ElectricityOut.V - (self.ElectricityIn.V - self.r * self.ElectricityIn.I))
                / ((self.nominal_voltage * self.r * self.nominal_current) ** 0.5)
            )
        )
        self.add_equation(((self.ElectricityIn.I - self.ElectricityOut.I) / self.nominal_current))
        self.add_equation(
            (
                (self.ElectricityOut.Power - (self.ElectricityIn.Power - self.Power_loss))
                / (self.nominal_voltage * self.nominal_current)
            )
        )
