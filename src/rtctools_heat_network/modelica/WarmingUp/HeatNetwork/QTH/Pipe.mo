within WarmingUp.HeatNetwork.QTH;

block Pipe
  import SI = Modelica.SIunits;

  parameter Real temperature;

  extends QTHTwoPort(QTHIn.T(nominal=temperature), QTHOut.T(nominal=temperature));

  parameter String component_type = "pipe";
  parameter Boolean disconnectable = false;
  parameter Boolean has_control_valve = false;

  parameter Real length = 1.0;
  parameter Real diameter = 1.0;
  parameter Real cp = 4200.0;
  parameter Real rho = 988.0;

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

  Modelica.SIunits.VolumeFlowRate Q;

  Modelica.SIunits.Level H_in;
  Modelica.SIunits.Level H_out;
  Modelica.SIunits.Level dH;
equation
  QTHIn.Q = Q;
  QTHOut.Q = QTHIn.Q;

  QTHIn.H = H_in;
  QTHOut.H = H_out;

  // Heat loss equation is added in the Python script to allow pipes to be disconnected.
  // It assumes constant ground temparature and constant dT at demand
  // positive negative dT depending on hot/cold pipe.
  // Roughly:
  // cp*rho*Q*(Out.T - In.T) + lenght*(U_1-U_2)*avg_T - lenght*(U_1-U_2)*T_ground + lenght*U_2*dT = 0.0

 end Pipe;
