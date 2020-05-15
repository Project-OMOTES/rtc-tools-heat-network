block HeatBuffer
  import SI = Modelica.SIunits;
  extends QTHBufferPort;
  parameter Real T_supply = 75.0;
  parameter Real T_return = 45.0;
  parameter Real dT = T_supply - T_return;
  parameter Real cp = 4200.0;
  parameter Real rho = 988.0;
  parameter Real head_loss = 0.0;
  Modelica.SIunits.VolumeFlowRate Q;
  Modelica.SIunits.Heat Heat(nominal=cp * rho * dT);
  Real Stored_heat(min=0.0, nominal=cp * rho * dT * 3600.0);
equation
  QTHHotIn.Q - QTHHotOut.Q = Q;
  QTHColdIn.Q - QTHColdOut.Q = -Q;
  QTHColdIn.T = T_return;
  QTHColdOut.T = T_return;
  QTHHotIn.T = T_supply;
  QTHHotOut.T = T_supply;
  (Heat - cp* rho * Q * (T_supply - T_return))/(cp * rho * dT) = 0.0;
  (der(Stored_heat) - Heat)/(cp * rho * dT) = 0.0;
initial equation
  der(Stored_heat)/(cp * rho * dT) = 0.0;
 end HeatBuffer;
