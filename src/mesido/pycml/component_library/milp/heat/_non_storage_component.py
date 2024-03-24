from mesido.pycml import Variable
from mesido.pycml.component_library.milp._internal.heat_component import BaseAsset

from numpy import nan

from .heat_two_port import HeatTwoPort


class _NonStorageComponent(HeatTwoPort, BaseAsset):
    """
    A non storage component is an asset type that does not have states that have interdependent
    time-steps, it does have time-varying states like Heat_flow. In the NonStorageComponent we
    define most of the properties being utilized by all components, like rho and cp.
    """

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
        self.nominal_pressure = 16.0e5

        # NOTE: We move a factor of 100.0 of the milp to the state entry, to
        # reduce the coefficient in front of the milp variables. This
        # particularly helps the scaling/range of the constraints that relate
        # the milp loss (if it is variable/optional) to the milp in- and out
        # of a component.
        self.Heat_nominal = self.cp * self.rho * self.Q_nominal * 100.0

        self.HeatIn.Heat.nominal = self.Heat_nominal
        self.HeatOut.Heat.nominal = self.Heat_nominal

        self.add_variable(Variable, "Q", nominal=self.Q_nominal)

        self.add_variable(Variable, "Heat_flow", nominal=self.Heat_nominal)

        # self.add_equation((self.Heat_out - self.HeatOut.Heat) / self.Heat_nominal)
        # self.add_equation((self.Heat_in - self.HeatIn.Heat) / self.Heat_nominal)

        self.add_equation(self.HeatIn.Q - self.Q)
        self.add_equation(self.HeatIn.Q - self.HeatOut.Q)

        # self.add_equation(self.HeatIn.H - self.H_in)
        # self.add_equation(self.HeatOut.H - self.H_out)
