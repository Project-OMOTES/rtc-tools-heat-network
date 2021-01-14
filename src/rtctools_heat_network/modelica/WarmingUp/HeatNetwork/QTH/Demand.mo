within WarmingUp.HeatNetwork.QTH;

block Demand
  import SI = Modelica.SIunits;
  extends QTHTwoPort;
  parameter String component_type = "demand";

  parameter Real Q_nominal = 1.0;
  parameter Real T_supply;
  parameter Real T_return;
  parameter Real dT = T_supply - T_return;
  parameter Real cp = 4200.0;
  parameter Real rho = 988.0;

  // Homotopic parameter
  parameter Real theta;

  Modelica.SIunits.Heat Heat_demand(nominal=cp * rho * dT * Q_nominal);

equation
  QTHOut.Q = QTHIn.Q;
  (Heat_demand - cp * rho * QTHOut.Q*((1-theta)*(dT) + (theta)*(QTHIn.T - QTHOut.T)))/(cp* rho * dT * Q_nominal) = 0.0;

 end Demand;