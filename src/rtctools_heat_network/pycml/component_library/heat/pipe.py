from numpy import nan

from rtctools_heat_network.pycml import Variable

from .heat_two_port import HeatTwoPort


class Pipe(HeatTwoPort):
    def __init__(self, name, **modifiers):
        super().__init__(name, **modifiers)

        self.component_type = "pipe"
        self.disconnectable = False
        self.has_control_valve = False

        self.length = 1.0
        self.diameter = 1.0
        self.temperature = nan
        self.cp = 4200.0
        self.rho = 988.0

        # For a PUR-PE pipe estimated based on 1m deep 150 mm pipe with 75 mm
        # PUR and 15 mm PE and distance of 2x centre to centre
        self.U_1 = 0.397
        self.U_2 = 0.0185
        self.T_supply = nan
        self.T_return = nan
        self.dT = self.T_supply - self.T_return
        self.T_g = nan  # ground temperature

        self.Heat_nominal = self.cp * self.rho * self.dT
        self.Heat_loss = nan

        self.add_variable(Variable, "Heat_in", nominal=self.Heat_nominal)
        self.add_variable(Variable, "Heat_out", nominal=self.Heat_nominal)

        self.add_equation((self.Heat_out - self.HeatOut.Heat) / self.Heat_nominal)
        self.add_equation((self.Heat_in - self.HeatIn.Heat) / self.Heat_nominal)

        # Note: Heat loss is added in Python, because it depends on the flow direction
