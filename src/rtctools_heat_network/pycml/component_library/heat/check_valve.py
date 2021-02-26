from rtctools_heat_network.pycml import Variable

from ._non_storage_component import _NonStorageComponent


class CheckValve(_NonStorageComponent):
    def __init__(self, name, **modifiers):
        super().__init__(
            name,
            **self.merge_modifiers(
                dict(
                    Q=dict(min=0.0),
                ),
                modifiers,
            ),
        )

        self.component_type = "check_valve"

        self.add_variable(Variable, "dH", min=0.0)

        self.add_equation(self.dH - (self.HeatOut.H - self.HeatIn.H))

        self.add_equation((self.HeatIn.Heat - self.HeatOut.Heat) / self.Heat_nominal)
