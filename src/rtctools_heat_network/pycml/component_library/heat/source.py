from numpy import nan

from rtctools_heat_network.pycml import Variable

from ._non_storage_component import _NonStorageComponent


class Source(_NonStorageComponent):
    def __init__(self, name, **modifiers):
        super().__init__(
            name,
            **self.merge_modifiers(
                dict(
                    Heat_in=dict(min=0.0, max=0.0),
                    Heat_out=dict(min=0.0),
                ),
                modifiers,
            ),
        )

        self.component_type = "source"

        self.price = nan

        # Assumption: heat in/out and added is nonnegative
        # Heat in the return (i.e. cold) line is zero
        self.add_variable(Variable, "Heat_source", min=0.0, nominal=self.Heat_nominal)
        self.add_variable(Variable, "dH", min=0.0)

        self.add_equation(self.dH - (self.HeatOut.H - self.HeatIn.H))

        self.add_equation(
            (self.HeatOut.Heat - (self.HeatIn.Heat + self.Heat_source)) / self.Heat_nominal
        )
