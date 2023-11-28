from numpy import nan

from rtctools_heat_network.pycml import Variable

from ._non_storage_component import _NonStorageComponent


class Source(_NonStorageComponent):
    """
    The source component is there to insert thermal power (Heat) into the network.

    The heat to discharge constraints are set in the HeatMixin. We enforce that the outgoing
    temperature of the source matches the heat, similar as with the demands. This allows us to
    guarantee  that the flow can always carry the heat and that thermal losses are always
    estimated conservatively. It also implicitly assumes that the temperature drops in the
    network are small and thus satisfy minimum temperature requirements.
    """

    def __init__(self, name, **modifiers):
        super().__init__(
            name,
            **self.merge_modifiers(
                dict(
                    HeatIn=dict(Heat=dict(min=0.0)),  # ensures it is not cooled down to below 0C
                    HeatOut=dict(Heat=dict(min=0.0)),
                ),
                modifiers,
            ),
        )

        self.component_type = "source"

        self.price = nan  # TODO: delete not needed anymore
        self.co2_coeff = 1.0

        # Assumption: heat in/out and added is nonnegative
        # Heat in the return (i.e. cold) line is zero
        self.add_variable(Variable, "Heat_source", min=0.0, nominal=self.Heat_nominal)
        self.add_variable(Variable, "Emission", min=0.0, nominal=self.Heat_nominal)
        self.add_variable(Variable, "dH", min=0.0)

        self.add_equation(self.dH - (self.HeatOut.H - self.HeatIn.H))

        self.add_equation(
            (self.HeatOut.Heat - (self.HeatIn.Heat + self.Heat_source)) / self.Heat_nominal
        )
        # self.add_equation(
        #     (self.Emission - self.Heat_source * self.co2_coeff)
        #     / (self.Heat_nominal * self.co2_coeff)
        # )

        self.add_equation((self.Heat_flow - self.Heat_source) / self.Heat_nominal)
