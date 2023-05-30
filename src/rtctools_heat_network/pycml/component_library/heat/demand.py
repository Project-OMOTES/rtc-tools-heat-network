from rtctools_heat_network.pycml import Variable

from ._non_storage_component import _NonStorageComponent


class Demand(_NonStorageComponent):
    def __init__(self, name, **modifiers):
        super().__init__(
            name,
            **self.merge_modifiers(
                dict(
                    Heat_in=dict(min=0.0),
                    Heat_out=dict(min=0.0),
                ),
                modifiers,
            ),
        )

        self.component_type = "demand"

        # Assumption: heat in/out and extracted is nonnegative
        # Heat in the return (i.e. cold) line is zero
        self.add_variable(Variable, "Heat_demand", min=0.0, nominal=self.Heat_nominal)

        self.add_equation(
            (self.HeatOut.Heat - (self.HeatIn.Heat - self.Heat_demand)) / self.Heat_nominal
        )

        self.add_equation((self.Heat_flow - self.Heat_demand) / self.Heat_nominal)
