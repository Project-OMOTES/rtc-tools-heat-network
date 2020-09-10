from rtctools_heat_network.pycml import Variable

from .qth_two_port import QTHTwoPort


class Pump(QTHTwoPort):
    def __init__(self, name, **modifiers):
        super().__init__(name, **modifiers)

        self.component_type = "pump"

        self.add_variable(Variable, "Q", min=0.0)
        self.add_variable(Variable, "H")
        self.add_variable(Variable, "dH", min=0.0)
        self.add_variable(Variable, "T")

        self.add_equation(self.QTHIn.Q - self.QTHOut.Q)
        self.add_equation(self.QTHIn.Q - self.Q)
        self.add_equation(self.QTHOut.H - self.H)
        self.add_equation(self.dH - (self.QTHOut.H - self.QTHIn.H))
        self.add_equation(self.QTHIn.T - self.QTHOut.T)
        self.add_equation(self.QTHIn.T - self.T)
