from numpy import nan, pi

from rtctools_heat_network.pycml import Variable

from ._non_storage_component import _NonStorageComponent


class Pipe(_NonStorageComponent):
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

        # Note: Heat loss is added in the mixin, because it depends on the flow direction
        # * heat loss equation: (HeatOut.Heat - (HeatIn.Heat +/- Heat_loss)) = 0.
