from numpy import nan

from rtctools_heat_network.pycml import Variable

from .qth_two_port import QTHTwoPort


class Pipe(QTHTwoPort):
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

        self.sign_dT = 1.0  # 1.0 if supply pipe, else -1.0

        self.add_variable(Variable, "Q")
        self.add_variable(Variable, "dH", max=0.0)

        self.add_equation(self.QTHIn.Q - self.Q)
        self.add_equation(self.QTHOut.Q - self.QTHIn.Q)

        # Heat loss equation is added in the Python script to allow pipes to be disconnected.
        # It assumes constant ground temparature and constant dT at demand
        # positive negative dT depending on hot/cold pipe.
        # Roughly:
        #   cp * rho * Q * (Out.T - In.T)
        #   + length * (U_1-U_2) * avg_T
        #   - length * (U_1-U_2) * T_g
        #   + length * U_2 * sign_dT * dT = 0.0
