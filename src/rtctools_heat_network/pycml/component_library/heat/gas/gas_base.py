from rtctools_heat_network.pycml import Connector, Variable

from .._internal.gas_component import GasComponent


class GasPort(GasComponent, Connector):
    def __init__(self, name, **modifiers):
        super().__init__(name, **modifiers)
        # TODO: think of more elegant approach for Q_shadow, currently required to ensure that
        #  every port has a unique variable to make the correct port mapping
        self.add_variable(Variable, "Q")
        self.add_variable(Variable, "Q_shadow")
        self.add_variable(Variable, "H")


class GasTwoPort(GasComponent):
    def __init__(self, name, **modifiers):
        super().__init__(name, **modifiers)

        self.add_variable(GasPort, "GasIn")
        self.add_variable(GasPort, "GasOut")
