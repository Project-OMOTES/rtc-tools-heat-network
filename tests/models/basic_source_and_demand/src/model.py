from rtctools_heat_network.pycml import ControlInput, Model as _Model
from rtctools_heat_network.pycml.component_library.heat import Demand, Pipe, Source


class Model(_Model):
    def __init__(self):
        super().__init__(None)

        self.T_supply = 75.0
        self.T_return = 45.0

        supply_return_modifiers = dict(T_supply=self.T_supply, T_return=self.T_return)

        self.add_variable(
            Source,
            "source",
            Heat_source=dict(min=0.75e5, max=1.25e5, nominal=1e5),
            Heat_out=dict(max=2e5),
            **supply_return_modifiers,
        )

        self.add_variable(Demand, "demand", Heat_in=dict(max=2e5), **supply_return_modifiers)

        self.add_variable(
            Pipe,
            "pipe_hot",
            length=1000.0,
            diameter=0.15,
            temperature=self.T_supply,
            HeatIn=dict(Heat=dict(min=-2e5, max=2e5, nominal=1e5)),
            HeatOut=dict(Heat=dict(nominal=1e5)),
            T_g=10.0,
            **supply_return_modifiers,
        )

        self.add_variable(
            Pipe,
            "pipe_cold",
            length=1000.0,
            diameter=0.15,
            temperature=self.T_return,
            HeatIn=dict(Heat=dict(min=-2e5, max=2e5, nominal=1e5)),
            HeatOut=dict(Heat=dict(nominal=1e5)),
            T_g=10.0,
            **supply_return_modifiers,
        )

        self.add_variable(ControlInput, "Heat_source", nominal=1e5, value=self.source.Heat_source)

        self.connect(self.source.HeatOut, self.pipe_hot.HeatIn)
        self.connect(self.pipe_hot.HeatOut, self.demand.HeatIn)
        self.connect(self.demand.HeatOut, self.pipe_cold.HeatIn)
        self.connect(self.pipe_cold.HeatOut, self.source.HeatIn)
