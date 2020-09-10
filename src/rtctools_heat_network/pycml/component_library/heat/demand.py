from numpy import nan

from rtctools_heat_network.pycml import Variable

from .heat_two_port import HeatTwoPort


class Demand(HeatTwoPort):
    def __init__(self, name, **modifiers):
        super().__init__(name, **modifiers)

        self.component_type = "demand"

        self.T_supply = nan
        self.T_return = nan
        self.dT = self.T_supply - self.T_return
        self.cp = 4200.0
        self.rho = 988.0
        self.Heat_nominal = self.cp * self.rho * self.dT

        self.add_variable(Variable, "Heat_demand", min=0.0, nominal=self.Heat_nominal)
        self.add_variable(Variable, "Heat_in", min=0.0, nominal=self.Heat_nominal)
        self.add_variable(Variable, "Heat_out", min=0.0, max=0.0, nominal=self.Heat_nominal)

        self.add_equation(
            (self.HeatOut.Heat - (self.HeatIn.Heat - self.Heat_demand)) / self.Heat_nominal
        )
        self.add_equation((self.Heat_out - self.HeatOut.Heat) / self.Heat_nominal)
        self.add_equation((self.Heat_in - self.HeatIn.Heat) / self.Heat_nominal)
