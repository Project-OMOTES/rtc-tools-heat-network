within WarmingUp.HeatNetwork.Heat;

block Demand
  import SI = Modelica.SIunits;
  extends HeatTwoPort;
  parameter String component_type = "demand";

  parameter Real Q_nominal = 1.0;
  parameter Real T_supply;
  parameter Real T_return;
  parameter Real dT = T_supply - T_return;
  parameter Real cp = 4200.0;
  parameter Real rho = 988.0;
  parameter Real Heat_nominal = cp * rho * dT * Q_nominal;

  // Assumption: heat in/out and extracted is nonnegative
  // Heat in the return (i.e. cold) line is zero
  Modelica.SIunits.Heat Heat_demand(min=0.0, nominal=Heat_nominal);
  Modelica.SIunits.Heat Heat_in(min=0.0, nominal=Heat_nominal);
  Modelica.SIunits.Heat Heat_out(min=0.0, max=0.0, nominal=Heat_nominal);

  Modelica.SIunits.VolumeFlowRate Q(nominal=Q_nominal);
equation
  HeatIn.Q = Q;
  HeatIn.Q = HeatOut.Q;

  // Heat balance equation
  (HeatOut.Heat - (HeatIn.Heat - Heat_demand))/Heat_nominal = 0.0;

  // Heat in/out aliases
  (Heat_out - HeatOut.Heat)/Heat_nominal = 0.0;
  (Heat_in - HeatIn.Heat)/Heat_nominal = 0.0;
end Demand;