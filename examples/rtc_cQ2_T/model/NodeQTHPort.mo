block NodeQTHPort "Three junction with 2 inflow pipes and 1 outflow pipe"
  import SI = Modelica.SIunits;
  extends QTHPort;
  replaceable parameter Integer nout(min = 1) = 1 "Number of outflows";
  parameter Integer nin(min = 1) = 1 "Number of inflows";
  // Homotopy parameter
  parameter Real theta;
  parameter Real temperature;

  SI.VolumeFlowRate QInSum;
  SI.VolumeFlowRate QOutSum;
  SI.Heat HeatInSum;
  SI.Heat HeatOutSum;
  SI.Temperature Tnode;
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
  (1-theta)*(HeatInSum - temperature*QInSum) + theta*(HeatInSum - sum(QIn.Q.*QIn.T)) = 0.0;

  // Temperature mixing
  (1-theta)*(Tnode - temperature) + theta*(Tnode*QInSum - HeatInSum) = 0.0;

  for i in 1:nout loop
    QOut[i].Q = QOut_control[i];
    QOut[i].H = H;
    QOut[i].T = Tnode;
  end for;
  for i in 1:nin loop
    QIn[i].H = H;
  end for;
  QTHPort.H = H;

end NodeQTHPort;
