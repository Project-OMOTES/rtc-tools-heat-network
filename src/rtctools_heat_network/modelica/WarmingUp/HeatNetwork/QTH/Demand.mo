within WarmingUp.HeatNetwork.QTH;

block Demand
  import SI = Modelica.SIunits;
  extends QTHTwoPort;
  //component type
  parameter String component_type = "demand";

  parameter Real T_supply = 75.0;
  parameter Real T_return = 45.0;
  parameter Real dT = T_supply - T_return;
  parameter Real cp = 4200.0;
  parameter Real rho = 988.0;

  // Homotopic parameter
  parameter Real theta;

  Modelica.SIunits.Heat Heat(nominal=cp * rho * dT);

equation
  QTHOut.Q = QTHIn.Q;
  (Heat - cp * rho * QTHOut.Q*((1-theta)*(dT) + (theta)*(QTHIn.T - QTHOut.T)))/(cp* rho * dT) = 0.0;

 end Demand;