within WarmingUp.HeatNetwork.Heat;

block Source
  import SI = Modelica.SIunits;
  extends HeatTwoPort;
  parameter String component_type = "source";

  parameter Real Q_nominal = 1.0;
  parameter Real T_supply;
  parameter Real T_return;
  parameter Real dT = T_supply - T_return;
  parameter Real cp = 4200.0;
  parameter Real rho = 988.0;
  parameter Real Heat_nominal = cp * rho * dT * Q_nominal;
  parameter Real price;

  // Assumption: heat in/out and added is nonnegative
  // Heat in the return (i.e. cold) line is zero
  Modelica.SIunits.Heat Heat_source(min=0.0, nominal=Heat_nominal);
  Modelica.SIunits.Heat Heat_in(min=0.0, max=0.0, nominal=Heat_nominal);
  Modelica.SIunits.Heat Heat_out(min=0.0, nominal=Heat_nominal);

  Modelica.SIunits.VolumeFlowRate Q(nominal=Q_nominal);

  Modelica.SIunits.Level H_in;
  Modelica.SIunits.Level H_out;
  Modelica.SIunits.Level dH(min=0.0);
equation
  HeatIn.Q = Q;
  HeatIn.Q = HeatOut.Q;

  HeatIn.H = H_in;
  HeatOut.H = H_out;
  dH = HeatOut.H - HeatIn.H;

  (HeatOut.Heat - (HeatIn.Heat + Heat_source))/Heat_nominal = 0.0;

  (Heat_out - HeatOut.Heat) / Heat_nominal = 0.0;
  (Heat_in - HeatIn.Heat) / Heat_nominal = 0.0;
end Source;
