block HeatSource
  import SI = Modelica.SIunits;
  extends QTHTwoPort;
  // TODO: remove parameters T_supply/T_return and corresponding equations
  parameter Real T_supply = 75.0;
  parameter Real T_return = 45.0;
  parameter Real dT = T_supply - T_return;
  parameter Real cp = 4200.0;
  parameter Real rho = 988.0;
  parameter Real head_loss = 0.0;
  Modelica.SIunits.Heat Heat(nominal=cp * rho * dT);
equation
  QTHOut.Q = QTHIn.Q;
  QTHOut.T = T_supply;
  QTHIn.T = T_return;
  (Heat - cp * rho * QTHOut.Q*(-QTHIn.T + QTHOut.T))/(cp * rho * dT) = 0.0;
 end HeatSource;
