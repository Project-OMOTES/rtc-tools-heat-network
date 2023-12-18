from numpy import nan

from rtctools_heat_network.pycml import Variable
from rtctools_heat_network.pycml.component_library.heat._internal import BaseAsset
from rtctools_heat_network.pycml.component_library.heat._internal.electricity_component import \
    ElectricityComponent
from rtctools_heat_network.pycml.component_library.heat.electricity.electricity_base import (
    ElectricityPort,
)
from rtctools_heat_network.pycml.component_library.heat.gas.gas_base import (
    GasPort,
)


class Electrolyzer(ElectricityComponent, BaseAsset):
    """
    An electrolyzer consumes electricity and produces hydrogen
    """
    def __init__(self, name, **modifiers):
        super().__init__(
            name,
            **self.merge_modifiers(
                dict(),
                modifiers,
            ),
        )

        self.component_type = "electrolyzer"

        self.a_eff_coefficient = nan
        self.b_eff_coefficient = nan
        self.c_eff_coefficient = nan

        self.minimum_load = nan

        self.density = 2.5  # H2 density [kg/m3] at 30bar

        self.Q_nominal = nan
        self.min_voltage = nan

        self.nominal_gass_mass_out = self.Q_nominal * self.density
        self.nominal_power_consumed = nan

        self.add_variable(ElectricityPort, "ElectricityIn")  # [W]
        self.add_variable(Variable, "Power_consumed", min=0.0, nominal=self.nominal_power_consumed)
        self.add_equation(
            (self.ElectricityIn.Power - self.Power_consumed) / self.nominal_power_consumed
        )  # [W]

        self.add_variable(GasPort, "GasOut")
        self.add_variable(
            Variable, "Gas_mass_flow_out", min=0.0, nominal=self.nominal_gass_mass_out
        )  # [kg/hr]
        self.add_equation(
            (self.GasOut.mass_flow - self.Gas_mass_flow_out) / self.nominal_gass_mass_out
        )
