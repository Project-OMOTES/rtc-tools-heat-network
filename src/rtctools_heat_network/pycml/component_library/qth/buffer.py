from numpy import nan

from rtctools_heat_network.pycml import Variable

from .qth_two_port import QTHTwoPort


class Buffer(QTHTwoPort):
    def __init__(self, name, **modifiers):
        super().__init__(name, **modifiers)

        self.component_type = "buffer"

        self.cp = 4200.0
        self.rho = 988.0
        self.head_loss = 0.0

        # Source for U estimates:
        # https://www.spiraxsarco.com/learn-about-steam/steam-engineering-principles-and-heat-transfer/energy-consumption-of-tanks-and-vats
        self.heat_transfer_coeff = 1.0
        self.height = 5.0
        self.radius = 10.0
        self.volume = 3.14 * self.radius ** 2 * self.height

        # if the tank is plced on ground, ignore surface of bottom of tank
        self.surface = 2 * 3.14 * self.radius ** 2 + 2 * 3.14 * self.radius * self.height
        self.T_outside = 10.0

        # Nominals
        self.nom_tank_volume = self.volume / 2
        self.T_supply = nan
        self.T_return = nan
        self.dT = self.T_supply - self.T_return
        self.nom_heat = self.cp * self.rho * self.dT

        # Initial values
        self.init_V_hot_tank = 0.0
        self.init_T_hot_tank = self.T_supply
        self.init_T_cold_tank = self.T_return

        # Volume of the tanks
        self.add_variable(
            Variable, "V_hot_tank", min=0.0, max=self.volume, nominal=self.nom_tank_volume
        )
        self.add_variable(
            Variable, "V_cold_tank", min=0.0, max=self.volume, nominal=self.nom_tank_volume
        )

        # Temperature in the tanks
        self.add_variable(Variable, "T_hot_tank", min=self.T_outside, nominal=self.T_supply)
        self.add_variable(Variable, "T_cold_tank", min=self.T_outside, nominal=self.T_return)

        # Alias variables for the in/out pipes.
        self.add_variable(Variable, "Q_hot_pipe")
        self.add_variable(Variable, "Q_cold_pipe")
        self.add_variable(Variable, "T_hot_pipe", nominal=self.T_supply)
        self.add_variable(Variable, "T_cold_pipe", nominal=self.T_return)

        # Buffer is modelled with an hot and a cold tank.
        # The hot tank is connected to the supply line, while the cold one to the return line.
        # The total volume of the system is constant.
        # Assumption: constant density and heat capacity
        # Q_hot_pipe is positive when the buffer is charging, negative if discharching.
        # Note that, as volume is constant, the amount of hot and cold water discharged
        # must be equal.
        # For convention, we assume that Q_hot_pipe and Q_cold_pipe have the same sign, i.e.,
        # Q_hot_pipe = Q_cold_pipe.

        # Volume
        self.add_equation(((self.V_hot_tank + self.V_cold_tank) - self.volume) / self.volume)
        self.add_equation(self.der(self.V_hot_tank) - self.Q_hot_pipe)

        # The following relationships are set in the mixin:

        # * Temperatures of the in/out pipes which are direction dependend.
        # If Q_hot_pipe < 0.0, T_hot_pipe = T_hot_tank.
        # Else, T_cold_pipe = T_cold_tank.
        # T_hot_pipe and T_cold pipe must be the one coming from the network
        # in the other timesteps.

        # * Heat equations:
        # der(V_hot_tank * T_hot_tank) - (Q_hot_pipe * T_hot_pipe) = 0.0;
        # der(V_cold_tank * T_hot_tank) - (Q_cold_pipe * T_cold_pipe) = 0.0;

        # * Flows in/out:
        # recall that Q_hot_pipe and Q_cold_pipe have the same sign.
        # To compensate for pipe orientations, we have:
        # hot_pipe_orientation * QTHIn.Q = Q_hot_pipe;
        # -1 * cold_pipe_orientation * QTHOut.Q = Q_cold_pipe;

        # Aliases
        self.add_equation(self.QTHIn.T - self.T_hot_pipe)
        self.add_equation(self.QTHOut.T - self.T_cold_pipe)

        # Set volume and temperatures at t0
        self.add_initial_equation(self.V_hot_tank - self.init_V_hot_tank)
        self.add_initial_equation(self.T_hot_tank - self.init_T_hot_tank)
        self.add_initial_equation(self.T_cold_tank - self.init_T_cold_tank)
