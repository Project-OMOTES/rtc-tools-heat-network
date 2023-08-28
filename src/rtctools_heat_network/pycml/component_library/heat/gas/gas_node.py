from rtctools_heat_network.pycml import Variable

from .gas_base import GasPort
from .._internal import BaseAsset
from .._internal.gas_component import GasComponent


class GasNode(GasComponent, BaseAsset):
    def __init__(self, name, **modifiers):
        super().__init__(name, **modifiers)

        self.component_type = "gas_node"

        self.n = 2
        assert self.n >= 2

        self.add_variable(GasPort, "GasConn", self.n)
        self.add_variable(Variable, "H", min=0.0)

        for i in range(1, self.n + 1):
            self.add_equation(self.GasConn[i].H - self.H)

        # Because the orientation of the connected pipes are important to setup the mass
        # conservation, these constraints are added in the mixin.
