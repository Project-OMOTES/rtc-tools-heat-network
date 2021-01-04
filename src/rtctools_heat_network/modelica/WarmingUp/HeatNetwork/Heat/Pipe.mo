within WarmingUp.HeatNetwork.Heat;

block Pipe
  import SI = Modelica.SIunits;
  extends HeatTwoPort;
  parameter String component_type = "pipe";
  parameter Boolean disconnectable = false;
  parameter Boolean has_control_valve = false;

  parameter Real Q_nominal = 1.0;
  parameter Real length = 1.0;
  parameter Real diameter = 1.0;
  parameter Real temperature;
  parameter Real cp = 4200.0;
  parameter Real rho = 988.0;
  parameter Real Heat_nominal = cp * rho * dT * Q_nominal;

  // Parameters determining the heat loss
  // All of these have default values in the library function
  parameter SI.Thickness insulation_thickness;
  parameter SI.ThermalConductivity conductivity_insulation;
  parameter SI.ThermalConductivity conductivity_subsoil;
  parameter SI.Distance depth;
  parameter SI.CoefficientOfHeatTransfer h_surface;
  parameter SI.Distance pipe_pair_distance;

  parameter Real T_supply;
  parameter Real T_return;
  parameter Real dT = T_supply - T_return;
  parameter Real T_ground = 10.0;

  Modelica.SIunits.Heat Heat_in(nominal=Heat_nominal);
  Modelica.SIunits.Heat Heat_out(nominal=Heat_nominal);

  Modelica.SIunits.VolumeFlowRate Q(nominal=Q_nominal);
  Modelica.SIunits.Level dH;

  // Heat loss
  // To be calculated in RTC-Tools, or else the parameter would disappear
  parameter Real Heat_loss;
equation
  HeatIn.Q = Q;
  HeatIn.Q = HeatOut.Q;

  // Aliases
  (Heat_out - HeatOut.Heat)/Heat_nominal = 0.0;
  (Heat_in - HeatIn.Heat)/Heat_nominal = 0.0;

  // Note: Heat loss is added in Python, because it depends on the flow direction
  // * heat loss equation: (HeatOut.Heat - (HeatIn.Heat +/- Heat_loss)) = 0.0

end Pipe;
