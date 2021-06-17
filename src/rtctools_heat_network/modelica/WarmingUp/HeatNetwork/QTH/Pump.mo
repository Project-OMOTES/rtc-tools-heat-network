within WarmingUp.HeatNetwork.QTH;

model Pump "Pump"
  extends QTHTwoPort;
  parameter String component_type = "pump";

  input Modelica.SIunits.VolumeFlowRate Q(min=0.0);

  Modelica.SIunits.Level H_in;
  Modelica.SIunits.Level H_out;
  Modelica.SIunits.Level dH(min=0.0);
equation
  QTHIn.Q = QTHOut.Q;
  QTHIn.Q = Q;

  QTHIn.H = H_in;
  QTHOut.H = H_out;
  dH = QTHOut.H - QTHIn.H;

  QTHIn.T = QTHOut.T;
end Pump;
