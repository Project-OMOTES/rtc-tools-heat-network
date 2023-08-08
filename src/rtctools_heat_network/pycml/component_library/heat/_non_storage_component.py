from numpy import nan

from rtctools_heat_network.pycml import Variable

from ._internal.heat_component import BaseAsset
from .heat_two_port import HeatTwoPort


class _NonStorageComponent(HeatTwoPort, BaseAsset):
    def __init__(self, name, **modifiers):
        super().__init__(name, **modifiers)

        self.Q_nominal = 1.0
        self.T_supply = nan
        self.T_return = nan
        self.T_supply_id = -1
        self.T_return_id = -1
        self.dT = self.T_supply - self.T_return
        self.cp = 4200.0
        self.rho = 988.0

        # NOTE: We move a factor of 100.0 of the heat to the state entry, to
        # reduce the coefficient in front of the heat variables. This
        # particularly helps the scaling/range of the constraints that relate
        # the heat loss (if it is variable/optional) to the heat in- and out
        # of a component.
        self.Heat_nominal = self.cp * self.rho * self.dT * self.Q_nominal / 100.0

        self.add_variable(Variable, "Heat_in", nominal=self.Heat_nominal)
        self.add_variable(Variable, "Heat_out", nominal=self.Heat_nominal)

        self.add_variable(Variable, "Q", nominal=self.Q_nominal)

        self.add_variable(Variable, "H_in")
        self.add_variable(Variable, "H_out")

        self.add_variable(Variable, "Heat_flow", nominal=self.Heat_nominal)

        self.add_equation((self.Heat_out - self.HeatOut.Heat) / self.Heat_nominal)
        self.add_equation((self.Heat_in - self.HeatIn.Heat) / self.Heat_nominal)

        self.add_equation(self.HeatIn.Q - self.Q)
        self.add_equation(self.HeatIn.Q - self.HeatOut.Q)

        self.add_equation(self.HeatIn.H - self.H_in)
        self.add_equation(self.HeatOut.H - self.H_out)
