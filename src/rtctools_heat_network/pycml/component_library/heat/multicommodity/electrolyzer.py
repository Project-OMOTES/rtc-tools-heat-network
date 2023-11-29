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


# TODO: for now in the electricity folder, but maybe we can make a multicommodity folder,
# where this is then placed.
class Electrolyzer(ElectricityComponent, BaseAsset):
    """
    ????
    """
    def __init__(self, name, **modifiers):
        super().__init__(
            name,
            **self.merge_modifiers(
                dict(),
                modifiers,
            ),
        )

        # ...
        self.component_type = "electrolyzer"
        self.nominal_gass_mass_out = 1.0
        self.nominal_power_consumed = 1.0
        self.Gass_mass_out_nominal = 1.0
        self.Power_consumed_nominal = 1.0

        # ...
        self.add_variable(ElectricityPort, "ElectricityIn")
        self.add_variable(Variable, "Power_consumed", min=0.0, nominal=self.nominal_power_consumed)
        # self.add_equation(
        #     (self.ElectricityIn.Power - self.Power_consumed) / self.Power_consumed_nominal
        # )

        # ... [kg ??]
        self.add_variable(GasPort, "GasOut")
        self.add_variable(Variable, "Gas_mass_out", min=0.0, nominal=self.nominal_gass_mass_out)
        # self.add_equation((self.GasOut - self.Gas_mass_out) / self.Gass_mass_out_nominal)