within WarmingUp.HeatNetwork.Heat;

model Pump
  extends HeatTwoPort;
  Modelica.SIunits.Level Heat;
  parameter String component_type = "pump";

  parameter Real T_supply = 75.0;
  parameter Real T_return = 45.0;
  parameter Real dT = T_supply - T_return;
  parameter Real cp = 4200.0;
  parameter Real rho = 988.0;
  parameter Real Heat_nominal = cp * rho * dT;
equation
// Heat is constant
  (HeatIn.Heat - HeatOut.Heat) / Heat_nominal = 0.0;
end Pump;
