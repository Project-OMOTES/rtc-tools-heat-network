from rtctools_heat_network.pycml import Variable

from ._non_storage_component import _NonStorageComponent


class ControlValve(_NonStorageComponent):
    """
    The control valve is a component to create pressure drop. We allow the control valve to create
    pressure drop for flow in both directions. Note that we set the absolute head loss symbol in
    the HeatMixin.
    """

    def __init__(self, name, **modifiers):
        super().__init__(name, **modifiers)

        self.component_type = "control_valve"

        self.add_variable(Variable, "dH")

        self.add_equation(self.dH - (self.HeatOut.H - self.HeatIn.H))

        self.add_equation((self.HeatIn.Heat - self.HeatOut.Heat) / self.Heat_nominal)
