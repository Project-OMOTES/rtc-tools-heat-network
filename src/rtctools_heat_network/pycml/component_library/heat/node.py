from ._internal import HeatComponent
from .heat_port import HeatPort


class Node(HeatComponent):
    def __init__(self, name, **modifiers):
        super().__init__(name, **modifiers)

        self.component_type = "node"

        self.n = 2
        assert self.n >= 2

        self.add_variable(HeatPort, "HeatConn", self.n)

        # Because the orientation of the connected pipes are important to
        # setup the heat conservation, these constraints are added in the
        # mixin.
