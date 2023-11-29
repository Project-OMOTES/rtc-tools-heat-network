import math

from numpy import nan, inf

from rtctools_heat_network.pycml import Variable
from rtctools_heat_network.pycml.component_library.heat._internal import BaseAsset

from .._internal.gas_component import GasComponent
from .gas_base import GasPort


class GasTankStorage(GasComponent, BaseAsset):
    """
    ...
    """

    def __init__(self, name, **modifiers):
        super().__init__(name, **modifiers)

        self.component_type = "gas_tank_storage"

        self.min_head = 30.0
        self.density = 23.715 # H2 density at 350bar
        self.volume = nan
        self.Q_nominal = nan

        self.add_variable(GasPort, "GasIn")
        self.add_variable(Variable, "Gas_tank_flow")

        self._typical_fill_time = 3600.0
        self._nominal_stored_gas = self.Q_nominal * self.density * self._typical_fill_time
        self.add_variable(
            Variable,
            "Stored_gas",
            min=0.,
            max=self.density*self.volume,
            nominal=self._nominal_stored_gas,
        )

        self.add_equation((self.GasIn.Q - self.Gas_tank_flow))

        self.add_equation((self.der(self.Stored_gas) - self.Gas_tank_flow))
