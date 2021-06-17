within WarmingUp.HeatNetwork.QTH;

model ControlValve
  extends QTHTwoPort;
  parameter String component_type = "control_valve";

  input Modelica.SIunits.VolumeFlowRate Q;

  Modelica.SIunits.Level H_in;
  Modelica.SIunits.Level H_out;
  Modelica.SIunits.Level dH;
equation
  QTHIn.Q = QTHOut.Q;
  QTHIn.Q = Q;

  QTHIn.H = H_in;
  QTHOut.H = H_out;
  dH = QTHOut.H - QTHIn.H;

  QTHIn.T = QTHOut.T;
end ControlValve;
