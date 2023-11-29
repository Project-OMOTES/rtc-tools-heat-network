import math

from numpy import pi, nan

from rtctools_heat_network.pycml import Variable

from .gas_base import GasTwoPort
from .._internal import BaseAsset


class GasPipe(GasTwoPort, BaseAsset):
    """
    The gas_pipe component is used to model head loss through the pipe. At the moment we only have
    a placeholder linear head loss formulation in place.
    """

    def __init__(self, name, **modifiers):
        super().__init__(name, **modifiers)

        self.component_type = "gas_pipe"
        self.disconnectable = False

        self.v_max = 15.
        self.diameter = nan
        self.area = pi * self.diameter ** 2
        self.Q_nominal = self.v_max * self.area

        self.nominal_head = 30.0
        self.length = nan
        self.nominal_flow_velocity = self.v_max / 2.
        self.r = 1.0e-6 * self.length  # TODO: temporary value
        self.nominal_head_loss = (self.Q_nominal * self.r * self.nominal_head) ** 0.5

        self.add_variable(Variable, "dH", nominal=self.Q_nominal * self.r)

        # head is lost over the pipe
        self.add_equation(((self.GasOut.H - (self.GasIn.H - self.dH)) / (self.Q_nominal*self.r)))
        # for now simple linear head loss
        self.add_equation(((self.dH - self.GasIn.Q * self.r) / (self.Q_nominal*self.r)))
        # Flow should be preserved
        self.add_equation(((self.GasIn.Q - self.GasOut.Q) / self.Q_nominal))
        # shadow Q for aliases
        self.add_equation(
            ((self.GasOut.Q_shadow - (self.GasIn.Q_shadow - 1.0e-3)) / self.Q_nominal)
        )
