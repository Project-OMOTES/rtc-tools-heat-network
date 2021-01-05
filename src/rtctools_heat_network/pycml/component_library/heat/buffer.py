from numpy import nan

from rtctools_heat_network.pycml import Variable

from .heat_two_port import HeatTwoPort


class Buffer(HeatTwoPort):
    def __init__(self, name, **modifiers):
        super().__init__(name, **modifiers)

        self.component_type = "buffer"

        self.T_supply = nan
        self.T_return = nan
        self.dT = self.T_supply - self.T_return
        self.cp = 4200.0
        self.rho = 988.0
        self.Heat_nominal = self.cp * self.rho * self.dT

        self.heat_transfer_coeff = 1.0
        self.height = 5.0
        self.radius = 10.0
        self.heat_loss_coeff = 2 * self.heat_transfer_coeff / (self.radius * self.rho * self.cp)

        self.init_Heat = 0.0

        # Stored_heat is the heat that is contained in the buffer.
        # Heat_buffer is the amount of heat added to or extracted from the buffer
        # per timestep.
        # HeatHot (resp. HeatCold) is the amount of heat added or extracted from
        # the hot (resp. cold) line.
        # As by construction the cold line should have zero heat, we fix HeatCold to zero.
        # Thus Heat_buffer = HeatHot = der(Stored_heat).
        self.add_variable(Variable, "Heat_buffer", nominal=self.Heat_nominal)
        self.add_variable(Variable, "Stored_heat", min=0.0, nominal=self.Heat_nominal)
        self.add_variable(Variable, "Heat_loss", min=0.0, nominal=self.Heat_nominal)
        self.add_variable(Variable, "HeatHot", nominal=self.Heat_nominal)
        self.add_variable(Variable, "HeatCold", min=0.0, max=0.0, nominal=self.Heat_nominal)

        # Heat stored in the buffer
        self.add_equation(
            (self.der(self.Stored_heat) - self.Heat_buffer + self.Heat_loss) / self.Heat_nominal
        )
        self.add_equation(
            (self.Heat_loss - self.Stored_heat * self.heat_loss_coeff) / self.Heat_nominal
        )
        self.add_equation((self.Heat_buffer - (self.HeatHot - self.HeatCold)) / self.Heat_nominal)

        # Aliases
        # Set in Python script. We want HeatHot to be positive when the buffer is
        # charging, which means we need to know the orientation of the connected
        # pipe.
        # (HeatCold + cold_pipe_orientation * HeatOut.Heat) / Heat_nominal = 0.0;
        # (HeatHot - hot_pipe_orientation * HeatIn.Heat) / Heat_nominal = 0.0;

        self.add_initial_equation((self.Stored_heat - self.init_Heat) / self.Heat_nominal)
