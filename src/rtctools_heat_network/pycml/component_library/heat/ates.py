from numpy import nan


from rtctools_heat_network.pycml import Variable

from ._internal.heat_component import BaseAsset
from .heat_two_port import HeatTwoPort


class ATES(HeatTwoPort, BaseAsset):
    def __init__(self, name, **modifiers):
        super().__init__(name, **modifiers)

        self.component_type = "ates"

        self.Q_nominal = 1.0
        self.T_supply = nan
        self.T_return = nan
        self.T_supply_id = -1
        self.T_return_id = -1
        self.dT = self.T_supply - self.T_return
        self.cp = 4200.0
        self.rho = 988.0
        self.Heat_nominal = self.cp * self.rho * self.dT * self.Q_nominal

        self.heat_loss_coeff = 0.005 / (24.0 * 3600.0)
        self.single_doublet_power = nan
        self.nr_of_doublets = 1.0
        # The hot/cold tank can have a lower bound on its volume.
        # Meaning that they might always be, for e.g., 5% full.
        self.min_fraction_tank_volume = 0.05

        # Stored_heat is the heat that is contained in the ates.
        # Heat_ates is the amount of heat added to or extracted from the buffer
        # per timestep.
        # Thus Heat_buffer = HeatHot = der(Stored_heat).
        # We connect an ATES as an demand, meaning that flow and Heat_ates are positive under
        # charging and negative under discharge
        self.add_variable(Variable, "Heat_ates", nominal=self.Heat_nominal)
        self.add_variable(Variable, "Heat_flow", nominal=self.Heat_nominal)
        # Assume the storage fills in about 3 months at typical rate
        self._typical_fill_time = 3600.0 * 24.0 * 90.0
        self._nominal_stored_heat = self.Heat_nominal * self._typical_fill_time
        self.add_variable(
            Variable,
            "Stored_heat",
            min=0.0,
            nominal=self._nominal_stored_heat,
        )
        self.add_variable(Variable, "Q", nominal=self.Q_nominal)
        # For nicer constraint coefficient scaling, we shift a bit more error into
        # the state vector entry of `Heat_loss`. In other words, with a factor of
        # 10.0, we aim for a state vector entry of ~0.1 (instead of 1.0)
        self._heat_loss_error_to_state_factor = 10.0
        self._nominal_heat_loss = (
            self._nominal_stored_heat * self.heat_loss_coeff * self._heat_loss_error_to_state_factor
        )
        self.add_variable(Variable, "Heat_loss", min=0.0, nominal=self._nominal_heat_loss)

        self._heat_loss_eq_nominal_ates = (self.Heat_nominal * self._nominal_heat_loss) ** 0.5

        self.add_equation(self.HeatIn.Q - self.HeatOut.Q)
        self.add_equation(self.Q - self.HeatOut.Q)

        # Heat stored in the ates
        self.add_equation(
            (self.der(self.Stored_heat) - self.Heat_ates + self.Heat_loss)
            / self._heat_loss_eq_nominal_ates
        )
        self.add_equation(
            (self.Heat_loss - self.Stored_heat * self.heat_loss_coeff) / self._nominal_heat_loss
        )
        self.add_equation(
            (self.HeatIn.Heat - (self.HeatOut.Heat + self.Heat_ates)) / self.Heat_nominal
        )
        self.add_equation((self.Heat_flow - self.Heat_ates) / self.Heat_nominal)
