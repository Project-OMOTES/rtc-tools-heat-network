block NodeQTHPort "Three junction with 2 inflow pipes and 1 outflow pipe"
  import SI = Modelica.SIunits;
  extends QTHPort;
  replaceable parameter Integer nout(min = 1) = 1 "Number of outflows";
  parameter Integer nin(min = 1) = 1 "Number of inflows";
  SI.VolumeFlowRate QInSum;
  SI.VolumeFlowRate QOutSum;
  QInPort QIn[nin];
  QOutPort QOut[nout];
  input SI.VolumeFlowRate QOut_control[nout];
  output SI.Position H;
  output SI.Temperature T;
equation
  QInSum = sum(QIn.Q);
  QOutSum = sum(QOut_control);
  for i in 1:nout loop
    QOut[i].Q = QOut_control[i];
    QOut[i].T = T;
    QOut[i].H = H;
  end for;
  for i in 1:nin loop
    QIn[i].T = T;
    QIn[i].H = H;
  end for;
  QInSum - QOutSum = 0.0;
  QTHPort.H = H;
  QTHPort.T = T;
end NodeQTHPort;
