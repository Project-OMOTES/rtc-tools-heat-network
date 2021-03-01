from numpy import nan

from rtctools_heat_network.pycml import Variable

from .qth_two_port import QTHTwoPort


class Pipe(QTHTwoPort):
    def __init__(self, name, **modifiers):
        super().__init__(name, **modifiers)

        self.temperature = nan

        super().__init__(
            name,
            **self.merge_modifiers(
                modifiers,
                dict(
                    QTHIn=dict(T=dict(nominal=self.temperature)),
                    QTHOut=dict(T=dict(nominal=self.temperature)),
                ),
            ),
        )

        self.component_type = "pipe"
        self.disconnectable = False
        self.has_control_valve = False

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

        self.add_variable(Variable, "Q")

        self.add_variable(Variable, "H_in")
        self.add_variable(Variable, "H_out")
        self.add_variable(Variable, "dH")

        self.add_equation(self.QTHIn.Q - self.Q)
        self.add_equation(self.QTHOut.Q - self.QTHIn.Q)

        self.add_equation(self.QTHIn.H - self.H_in)
        self.add_equation(self.QTHOut.H - self.H_out)

        # Heat loss equation is added in the Python script to allow pipes to be disconnected.
        # It assumes constant ground temparature and constant dT at demand
        # positive negative dT depending on hot/cold pipe.
        # Roughly:
        #   cp * rho * Q * (Out.T - In.T)
        #   + length * (U_1-U_2) * avg_T
        #   - length * (U_1-U_2) * T_ground
        #   + length * U_2 * dT = 0.0
