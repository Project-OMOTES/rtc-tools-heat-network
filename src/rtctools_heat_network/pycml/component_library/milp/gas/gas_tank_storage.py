from numpy import nan

from rtctools_heat_network.pycml import Variable
from rtctools_heat_network.pycml.component_library.milp._internal import BaseAsset

from .gas_base import GasPort
from .._internal.gas_component import GasComponent


class GasTankStorage(GasComponent, BaseAsset):
    """
    ...
    """

    def __init__(self, name, **modifiers):
        super().__init__(name, **modifiers)

        self.component_type = "gas_tank_storage"

        self.min_head = 30.0
        self.density = 2.5  # H2 density [kg/m3] at 30bar
        self.density_max_storage = 23.715  # H2 density [kg/m3] at 350bar
        self.volume = nan
        self.Q_nominal = nan

        self.add_variable(GasPort, "GasIn")
        self.add_variable(Variable, "Gas_tank_flow", nominal=self.Q_nominal * self.density)

        self._typical_fill_time = 3600.0
        self._nominal_stored_gas = (
            self.Q_nominal * self.density_max_storage * self._typical_fill_time
        )
        self.add_variable(
            Variable,
            "Stored_gas_mass",
            min=0.0,
            max=self.density_max_storage * self.volume,
            nominal=self._nominal_stored_gas,
        )

        self.add_equation(
            ((self.GasIn.mass_flow - self.Gas_tank_flow) / (self.Q_nominal * self.density))
        )

        self.add_equation(
            (
                (self.der(self.Stored_gas_mass) - (self.Gas_tank_flow / 3600.0))
                / (self._nominal_stored_gas * self.Q_nominal * self.density / 3600.0) ** 0.5
            )
        )

        self.add_initial_equation((self.Stored_gas_mass / self._nominal_stored_gas))
