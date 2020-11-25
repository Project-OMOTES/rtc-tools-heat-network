within WarmingUp.HeatNetwork.Heat;

model Pump
  extends HeatTwoPort;
  parameter String component_type = "pump";

  parameter Real T_supply;
  parameter Real T_return;
  parameter Real dT = T_supply - T_return;
  parameter Real cp = 4200.0;
  parameter Real rho = 988.0;
  parameter Real Heat_nominal = cp * rho * dT;

  Modelica.SIunits.Level Heat;
equation
// Heat is constant
  (HeatIn.Heat - HeatOut.Heat) / Heat_nominal = 0.0;
  Heat = HeatIn.Heat;
end Pump;
