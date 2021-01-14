from numpy import nan

from rtctools_heat_network.pycml import SymbolicParameter, Variable

from .qth_two_port import QTHTwoPort


class Source(QTHTwoPort):
    def __init__(self, name, **modifiers):
        super().__init__(name, **modifiers)

        self.component_type = "source"

        self.Q_nominal = 1.0
        self.T_supply = nan
        self.T_return = nan
        self.dT = self.T_supply - self.T_return
        self.cp = 4200.0
        self.rho = 988.0
        self.head_loss = 0.0

        self.add_variable(SymbolicParameter, "theta")

        self.add_variable(
            Variable, "Heat_source", min=0.0, nominal=self.cp * self.rho * self.dT * self.Q_nominal
        )

        self.add_equation(self.QTHOut.Q - self.QTHIn.Q)
        self.add_equation(
            (
                self.Heat_source
                - self.cp
                * self.rho
                * self.QTHOut.Q
                * ((1 - self.theta) * self.dT + self.theta * (-self.QTHIn.T + self.QTHOut.T))
            )
            / (self.cp * self.rho * self.dT * self.Q_nominal)
        )
