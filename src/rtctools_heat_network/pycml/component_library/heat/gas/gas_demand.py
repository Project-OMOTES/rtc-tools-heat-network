from numpy import nan

from rtctools_heat_network.pycml import Variable

from .gas_base import GasPort
from .._internal import BaseAsset
from .._internal.gas_component import GasComponent


class GasDemand(GasComponent, BaseAsset):
    """
    A gas demand consumes flow from the network.
    """

    def __init__(self, name, **modifiers):
        super().__init__(name, **modifiers)

        self.component_type = "gas_demand"
        self.min_head = 30.0

        self.Q_nominal =  nan

        self.density = 2.5  # H2 density [kg/m3] at 30bar

        self.add_variable(GasPort, "GasIn")
        self.add_variable(Variable, "Gas_demand_mass_flow", min=0.0, nominal=self.Q_nominal * self.density)

        self.add_equation(((self.GasIn.Q * self.density - self.Gas_demand_mass_flow) / (self.Q_nominal * self.density)))
