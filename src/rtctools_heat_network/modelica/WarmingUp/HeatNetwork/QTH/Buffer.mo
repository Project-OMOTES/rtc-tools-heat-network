within WarmingUp.HeatNetwork.QTH;

block Buffer
  import SI = Modelica.SIunits;
  extends QTHBufferPort;
  parameter String component_type = "buffer";

  parameter Real T_supply = 75.0;
  parameter Real T_return = 45.0;
  parameter Real dT = T_supply - T_return;
  parameter Real cp = 4200.0;
  parameter Real rho = 988.0;
  parameter Real head_loss = 0.0;

  // Homotopic parameter
  parameter Real theta;

  // Heat variables
  Modelica.SIunits.Heat Heat(nominal=cp * rho * dT);
  Modelica.SIunits.Heat Heat_extracted(max = 0.0, nominal=cp * rho * dT);
  Modelica.SIunits.Heat Heat_added(min = 0.0, nominal=cp * rho * dT);
  Real Stored_heat(min=0.0, nominal=cp * rho * dT * 3600.0);

equation
  // Heat balance
  // As bidirectional flow is modelled as two different pipes (having non-negative flow),
  // we need to account for heat extracted and added to the buffer.
  // Note that the heat added (same for extracted) comes from both the hot and cold pipes.

  // Heat equation is non-linear. The linearization is around the dT value.
  QTHHotIn.Q - QTHColdOut.Q = 0.0;
  QTHColdIn.Q - QTHHotOut.Q = 0.0;

  (Heat_added - ((1-theta)*rho*cp*QTHHotIn.Q*dT+(theta)*rho*cp*((QTHHotIn.Q*QTHHotIn.T)-(QTHColdOut.Q*QTHColdOut.T))))/(cp * rho * dT) = 0.0;
  (Heat_extracted-((1-theta)*rho*cp*QTHColdIn.Q*(-dT)+(theta)*rho*cp*((QTHColdIn.Q*QTHColdIn.T)-(QTHHotOut.Q*QTHHotOut.T))))/(cp * rho * dT) = 0.0;
  (Heat - (Heat_added+Heat_extracted))/(cp * rho * dT) = 0.0;

  // Heat stored in the buffer
  (der(Stored_heat) - Heat)/(cp * rho * dT) = 0.0;

initial equation
  // Initialize the derivative to zero
  der(Stored_heat)/(cp * rho * dT) = 0.0;
 end Buffer;
