block Pipe
  import SI = Modelica.SIunits;
  extends QTHTwoPort;
  parameter Real length = 1.0;
  parameter Real diameter = 1.0;
  parameter Real temperature = 50.0;
  Modelica.SIunits.VolumeFlowRate Q(min=0.0);
  Modelica.SIunits.Level dH(max=0.0);
equation
  QTHOut.Q = Q;
  QTHOut.Q = QTHIn.Q;
  QTHOut.T = QTHIn.T;
  dH = QTHOut.H - QTHIn.H;
 end Pipe;
