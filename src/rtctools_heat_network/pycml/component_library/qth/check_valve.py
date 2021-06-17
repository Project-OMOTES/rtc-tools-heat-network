from rtctools_heat_network.pycml import Variable

from .qth_two_port import QTHTwoPort


class CheckValve(QTHTwoPort):
    def __init__(self, name, **modifiers):
        super().__init__(name, **modifiers)

        self.component_type = "check_valve"

        self.add_variable(Variable, "Q", min=0.0)

        self.add_variable(Variable, "H_in")
        self.add_variable(Variable, "H_out")
        self.add_variable(Variable, "dH", min=0.0)

        self.add_equation(self.QTHIn.Q - self.QTHOut.Q)
        self.add_equation(self.QTHIn.Q - self.Q)

        self.add_equation(self.QTHIn.H - self.H_in)
        self.add_equation(self.QTHOut.H - self.H_out)
        self.add_equation(self.dH - (self.QTHOut.H - self.QTHIn.H))

        self.add_equation(self.QTHIn.T - self.QTHOut.T)
