# from rtctools_heat_network.pycml import Variable
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

        # TODO: potentially we can keep the component type as heat_pump and set subcomponent to
        # heat_pump_elec, first need to check if there wouldn't be anything conflicting then.
        self.component_type = "electrolyzer"
        self.nominal_mass_source = 1.0
        self.nominal_power_consumed = 1.0

        # This or an electricity power variable?
        self.add_variable(ElectricityPort, "ElectricityIn")
        # self.add_variable(Variable, "Power_consumed", min=0.0, nominal=self.nominal_power_consumed)
        # self.add_equation(self.ElectricityIn.P - self.Power_consumed)


        # Use this ? or GasPort
        self.add_variable(GasPort, "GasOut")
        # self.add_variable(Variable, "Gas_mass_out")