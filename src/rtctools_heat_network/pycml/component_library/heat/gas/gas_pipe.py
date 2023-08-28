import math

from rtctools_heat_network.pycml import Variable

from .gas_base import GasTwoPort
from .._internal import BaseAsset


class GasPipe(GasTwoPort, BaseAsset):
    def __init__(self, name, **modifiers):
        super().__init__(name, **modifiers)

        self.component_type = "gas_pipe"
        self.disconnectable = False

        self.nominal_head = 30.0
        self.length = 1.0
        self.nominal_flow_velocity = 1.0
        self.diameter = 1.0
        self.r = 1.0e-6 * self.length  # TODO: temporary value
        self.nominal_flow = (self.diameter / 2) ** 2 * math.pi * self.nominal_flow_velocity
        self.nominal_head_loss = (self.nominal_flow * self.r * self.nominal_head) ** 0.5

        self.add_variable(Variable, "dH", nominal=self.nominal_flow * self.r)

        # head is lost over the pipe
        self.add_equation(((self.GasOut.H - (self.GasIn.H - self.dH)) / self.nominal_head))
        # for now simple linear head loss
        self.add_equation(((self.dH - self.GasIn.Q * self.r) / self.nominal_head_loss))
        # Flow should be preserved
        self.add_equation(((self.GasIn.Q - self.GasOut.Q) / self.nominal_flow))
        # shadow Q for aliases
        self.add_equation(
            ((self.GasOut.Q_shadow - (self.GasIn.Q_shadow - 1.0e-3)) / self.nominal_flow)
        )
