from rtctools_heat_network.pycml import Variable

from numpy import nan

from .electricity_base import ElectricityTwoPort
from .._internal import BaseAsset


class ElectricityCable(ElectricityTwoPort, BaseAsset):
    """
    The electricity cable component is used to model voltage and power drops in the electricity
    lines. We model the power losses by over estimating them with the maximum current. We ensure
    that the power is always less than what the current is able to carry by an equality constraint
    at the demand where we enforce the minimum voltage.
    """

    def __init__(self, name, **modifiers):
        super().__init__(name, **modifiers)

        self.component_type = "electricity_cable"
        self.disconnectable = False

        self.length = 1.0

        # Powerloss with inequality in the heat-mixin
        # values for NAYY 4x50 SE
        # from: https://pandapower.readthedocs.io/en/v2.6.0/std_types/basic.html
        self.max_current = nan
        self.min_voltage = nan
        self.max_voltage = self.min_voltage * 2.
        self.nominal_current = nan
        self.nominal_voltage = nan
        self.r = 1.e-6 * self.length  # TODO: temporary value
        self.add_variable(Variable, "Power_loss", min=0.0, nominal=self.r * self.max_current**2)

        self.add_equation(
            (
                (self.ElectricityOut.V - (self.ElectricityIn.V - self.r * self.ElectricityIn.I))
                / ((self.nominal_current * self.r * self.nominal_current) ** 0.5)
            )
        )
        self.add_equation(((self.ElectricityIn.I - self.ElectricityOut.I) / self.nominal_current))
        self.add_equation(
            (
                (self.ElectricityOut.Power - (self.ElectricityIn.Power - self.Power_loss))
                / (self.nominal_voltage * self.nominal_current * self.r * self.max_current**2)
                ** 0.5
            )
        )
