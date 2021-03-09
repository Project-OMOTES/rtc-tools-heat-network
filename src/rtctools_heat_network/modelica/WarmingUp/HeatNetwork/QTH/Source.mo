within WarmingUp.HeatNetwork.QTH;

block Source
  import SI = Modelica.SIunits;
  extends QTHTwoPort;
  parameter String component_type = "source";

  parameter Real Q_nominal = 1.0;
  parameter Real T_supply;
  parameter Real T_return;
  parameter Real dT = T_supply - T_return;
  parameter Real cp = 4200.0;
  parameter Real rho = 988.0;

  parameter Real theta;

  Modelica.SIunits.Heat Heat_source(nominal=cp * rho * dT * Q_nominal);

  Modelica.SIunits.VolumeFlowRate Q(nominal=Q_nominal);

  Modelica.SIunits.Level H_in;
  Modelica.SIunits.Level H_out;
  Modelica.SIunits.Level dH(min=0.0);
equation
  QTHIn.Q = Q;
  QTHOut.Q = QTHIn.Q;

  QTHIn.H = H_in;
  QTHOut.H = H_out;
  dH = QTHOut.H - QTHIn.H;

  (Heat_source - cp * rho * QTHOut.Q*((1-theta)*(dT) + (theta)*(-QTHIn.T + QTHOut.T)))/(cp* rho * dT * Q_nominal) = 0.0;

 end Source;
