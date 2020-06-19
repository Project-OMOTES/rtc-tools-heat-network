block NodeQTHPort "Three junction with 2 inflow pipes and 1 outflow pipe"
  import SI = Modelica.SIunits;
  extends QTHPort;
  replaceable parameter Integer nout(min = 1) = 1 "Number of outflows";
  parameter Integer nin(min = 1) = 1 "Number of inflows";
  // Homotopy parameter
  parameter Real theta;

  SI.VolumeFlowRate QInSum;
  SI.VolumeFlowRate QOutSum;
  SI.Heat HeatInSum;
  SI.Heat HeatOutSum;
  QInPort QIn[nin];
  QOutPort QOut[nout];
  input SI.VolumeFlowRate QOut_control[nout];
  output SI.Position H;

equation
  // Conservation of flow
  QInSum = sum(QIn.Q);
  QOutSum = sum(QOut_control);
  QInSum - QOutSum = 0.0;

  // Conservation of Heat
  HeatInSum = theta*sum(QIn.Q.*QIn.T);
  HeatOutSum = theta*sum(QOut.Q.*QOut.T);
  HeatInSum - HeatOutSum = 0.0;

  for i in 1:nout loop
    QOut[i].Q = QOut_control[i];
    QOut[i].H = H;
  end for;
  for i in 1:nin loop
    QIn[i].H = H;
  end for;
  QTHPort.H = H;
  QTHPort.T = T;

end NodeQTHPort;
