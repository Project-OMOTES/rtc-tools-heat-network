from numpy import nan

from rtctools_heat_network.pycml import Variable

from .heat_two_port import HeatTwoPort


class Pipe(HeatTwoPort):
    def __init__(self, name, **modifiers):
        super().__init__(name, **modifiers)

        self.component_type = "pipe"
        self.disconnectable = False
        self.has_control_valve = False

        self.Q_nominal = 1.0
        self.length = 1.0
        self.diameter = 1.0
        self.temperature = nan
        self.cp = 4200.0
        self.rho = 988.0

        # Parameters determining the heat loss
        # All of these have default values in the library function
        self.insulation_thickness = nan
        self.conductivity_insulation = nan
        self.conductivity_subsoil = nan
        self.depth = nan
        self.h_surface = nan
        self.pipe_pair_distance = nan

        self.T_supply = nan
        self.T_return = nan
        self.dT = self.T_supply - self.T_return
        self.T_ground = 10.0

        self.Heat_nominal = self.cp * self.rho * self.dT * self.Q_nominal
        self.Heat_loss = nan

        self.add_variable(Variable, "Heat_in", nominal=self.Heat_nominal)
        self.add_variable(Variable, "Heat_out", nominal=self.Heat_nominal)

        self.add_variable(Variable, "Q", nominal=self.Q_nominal)
        self.add_variable(Variable, "dH")

        self.add_equation(self.HeatIn.Q - self.Q)
        self.add_equation(self.HeatIn.Q - self.HeatOut.Q)

        self.add_equation((self.Heat_out - self.HeatOut.Heat) / self.Heat_nominal)
        self.add_equation((self.Heat_in - self.HeatIn.Heat) / self.Heat_nominal)

        # Note: Heat loss is added in Python, because it depends on the flow direction
