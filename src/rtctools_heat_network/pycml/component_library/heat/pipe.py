from numpy import nan, pi

from rtctools_heat_network.pycml import Variable

from ._non_storage_component import _NonStorageComponent


class Pipe(_NonStorageComponent):
    """
    The pipe component is to model the pressure drop (and optionally hydraulic power) and
    heat losses over a pipe. Three options for head loss computation are available in the HeatMixin
    options: No_HeadLoss, Linear, DW_Linearized. The hydraulic power computation can only be done
    reasonably if DW_linearized is selected as otherwise head_losses are significantly
    over-estimated.

    The heat to discharge constraints are set in the HeatMixin. Where we ensure that the heat must
    be smaller that what the flow can carry. Meaning that the flow does lose energy but not
    temperature. In this manner the energy losses will always be overestimated as in reality the
    flow will also have a temperature drop.
    """

    def __init__(self, name, **modifiers):
        super().__init__(name, **modifiers)

        self.component_type = "pipe"
        self.disconnectable = False
        self.has_control_valve = False

        self.length = 1.0
        self.diameter = 1.0
        assert "area" not in modifiers, "modifying area directly is not allowed"
        self.area = 0.25 * pi * self.diameter**2
        self.temperature = nan

        # Parameters determining the heat loss
        # All of these have default values in the library function
        self.insulation_thickness = nan
        self.conductivity_insulation = nan
        self.conductivity_subsoil = nan
        self.depth = nan
        self.h_surface = nan
        self.pipe_pair_distance = nan

        self.T_ground = 10.0

        self.Heat_loss = nan

        self.add_variable(Variable, "dH")

        # rho * ff * length * area / 2 / diameter * velocity**3
        ff = 0.02  # Order of magnitude expected with 0.05-2.5m/s in 20mm-1200mm diameter pipe
        velo = self.Q_nominal / self.area
        self.Hydraulic_power_nominal = (
            self.rho * ff * self.length * pi * self.area / self.diameter / 2.0 * velo**3
        )
        self.add_variable(
            Variable, "Hydraulic_power", min=0.0, nominal=self.Hydraulic_power_nominal
        )  # [W]

        self.add_equation(((self.Heat_flow - self.HeatIn.Heat) / self.Heat_nominal))

        # Note: Heat loss is added in the mixin, because it depends on the flow direction
        # * heat loss equation: (HeatOut.Heat - (HeatIn.Heat +/- Heat_loss)) = 0.
