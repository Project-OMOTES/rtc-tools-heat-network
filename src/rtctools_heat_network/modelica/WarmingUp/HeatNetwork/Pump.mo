within WarmingUp.HeatNetwork;

model Pump "Pump"
  extends QTHTwoPort;
  input Modelica.SIunits.VolumeFlowRate Q(min=0.0);
  Modelica.SIunits.Level H;
  Modelica.SIunits.Level dH(min=0.0);
  Modelica.SIunits.Temperature T;
equation
  QTHIn.Q = QTHOut.Q;
  QTHIn.Q = Q;
  QTHOut.H = H;
  dH = QTHOut.H - QTHIn.H;
  QTHIn.T = QTHOut.T;
  QTHIn.T = T;
end Pump;
