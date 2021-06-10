within WarmingUp.HeatNetwork.Heat;

model ControlValve
  extends HeatTwoPort;
  parameter String component_type = "control_valve";

  parameter Real Q_nominal = 1.0;
  parameter Real T_supply;
  parameter Real T_return;
  parameter Real dT = T_supply - T_return;
  parameter Real cp = 4200.0;
  parameter Real rho = 988.0;
  parameter Real Heat_nominal = cp * rho * dT * Q_nominal;

  Modelica.SIunits.VolumeFlowRate Q(nominal=Q_nominal);

  Modelica.SIunits.Level H_in;
  Modelica.SIunits.Level H_out;
  Modelica.SIunits.Level dH;
equation
  HeatIn.Q = Q;
  HeatIn.Q = HeatOut.Q;

  HeatIn.H = H_in;
  HeatOut.H = H_out;
  dH = HeatOut.H - HeatIn.H;

  (HeatIn.Heat - HeatOut.Heat) / Heat_nominal = 0.0;
end ControlValve;
