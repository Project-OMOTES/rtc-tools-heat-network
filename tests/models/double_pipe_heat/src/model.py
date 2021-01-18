from rtctools_heat_network.pycml import ControlInput, Model as _Model
from rtctools_heat_network.pycml.component_library.heat import Demand, Node, Pipe, Source


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
            **supply_return_modifiers,
        )

        self.add_variable(
            Pipe,
            "pipe_cold",
            length=1000.0,
            diameter=0.15,
            temperature=self.T_return,
            HeatIn=dict(Heat=dict(min=0.0, max=0.0, nominal=1e5)),
            HeatOut=dict(Heat=dict(nominal=1e5)),
            **supply_return_modifiers,
        )

        self.add_variable(
            Pipe,
            "pipe_2_hot",
            length=1000.0,
            diameter=0.15,
            temperature=self.T_supply,
            HeatIn=dict(Heat=dict(min=-2e5, max=2e5, nominal=1e5)),
            HeatOut=dict(Heat=dict(nominal=1e5)),
            **supply_return_modifiers,
        )

        self.add_variable(
            Pipe,
            "pipe_2_cold",
            length=1000.0,
            diameter=0.15,
            temperature=self.T_return,
            HeatIn=dict(Heat=dict(min=0.0, max=0.0, nominal=1e5)),
            HeatOut=dict(Heat=dict(nominal=1e5)),
            **supply_return_modifiers,
        )

        self.add_variable(
            Pipe,
            "pipe_demand_hot",
            length=0.0,
            diameter=0.15,
            temperature=self.T_supply,
            HeatIn=dict(Heat=dict(min=-2e5, max=2e5, nominal=1e5)),
            HeatOut=dict(Heat=dict(nominal=1e5)),
            **supply_return_modifiers,
        )

        self.add_variable(
            Pipe,
            "pipe_demand_cold",
            length=0.0,
            diameter=0.15,
            temperature=self.T_return,
            HeatIn=dict(Heat=dict(min=0.0, max=0.0, nominal=1e5)),
            HeatOut=dict(Heat=dict(nominal=1e5)),
            **supply_return_modifiers,
        )

        self.add_variable(
            Pipe,
            "pipe_source_hot",
            length=0.0,
            diameter=0.15,
            temperature=self.T_supply,
            HeatIn=dict(Heat=dict(min=-2e5, max=2e5, nominal=1e5)),
            HeatOut=dict(Heat=dict(nominal=1e5)),
            **supply_return_modifiers,
        )

        self.add_variable(
            Pipe,
            "pipe_source_cold",
            length=0.0,
            diameter=0.15,
            temperature=self.T_return,
            HeatIn=dict(Heat=dict(min=0.0, max=0.0, nominal=1e5)),
            HeatOut=dict(Heat=dict(nominal=1e5)),
            **supply_return_modifiers,
        )

        self.add_variable(
            Node,
            "node_source_hot",
            n=3,
        )

        self.add_variable(
            Node,
            "node_demand_hot",
            n=3,
        )

        self.add_variable(
            Node,
            "node_source_cold",
            n=3,
        )

        self.add_variable(
            Node,
            "node_demand_cold",
            n=3,
        )

        self.add_variable(ControlInput, "Heat_source", nominal=1e5, value=self.source.Heat_source)

        self.connect(self.source.HeatOut, self.pipe_source_hot.HeatIn)
        self.connect(self.pipe_source_hot.HeatOut, self.node_source_hot.HeatConn[1])
        self.connect(self.node_source_hot.HeatConn[2], self.pipe_hot.HeatIn)
        self.connect(self.node_source_hot.HeatConn[3], self.pipe_2_hot.HeatIn)

        self.connect(self.pipe_hot.HeatOut, self.node_demand_hot.HeatConn[2])
        self.connect(self.pipe_2_hot.HeatOut, self.node_demand_hot.HeatConn[3])
        self.connect(self.node_demand_hot.HeatConn[1], self.pipe_demand_hot.HeatIn)
        self.connect(self.pipe_demand_hot.HeatOut, self.demand.HeatIn)

        self.connect(self.demand.HeatOut, self.pipe_demand_cold.HeatIn)
        self.connect(self.pipe_demand_cold.HeatOut, self.node_demand_cold.HeatConn[1])
        self.connect(self.pipe_cold.HeatIn, self.node_demand_cold.HeatConn[2])
        self.connect(self.pipe_2_cold.HeatIn, self.node_demand_cold.HeatConn[3])

        self.connect(self.node_source_cold.HeatConn[2], self.pipe_cold.HeatOut)
        self.connect(self.node_source_cold.HeatConn[3], self.pipe_2_cold.HeatOut)
        self.connect(self.node_source_cold.HeatConn[1], self.pipe_source_cold.HeatIn)
        self.connect(self.pipe_source_cold.HeatOut, self.source.HeatIn)
