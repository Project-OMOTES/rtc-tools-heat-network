from rtctools_heat_network.pycml import Variable

from .electricity_base import ElectricityPort
from .._internal import BaseAsset
from .._internal.electricity_component import ElectricityComponent


class ElectricityNode(ElectricityComponent, BaseAsset):
    def __init__(self, name, **modifiers):
        super().__init__(name, **modifiers)

        self.component_type = "electricity_node"

        self.n = 2
        assert self.n >= 2

        self.add_variable(ElectricityPort, "ElectricityConn", self.n)
        self.add_variable(Variable, "V", min=0.0)

        for i in range(1, self.n + 1):
            self.add_equation(self.ElectricityConn[i].V - self.V)

        # Because the orientation of the connected cables are important to setup the energy
        # conservation, these constraints are added in the mixin.
