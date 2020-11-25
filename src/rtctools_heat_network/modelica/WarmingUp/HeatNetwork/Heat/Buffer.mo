within WarmingUp.HeatNetwork.Heat;

block Buffer
  import SI = Modelica.SIunits;
  extends HeatTwoPort;
  parameter String component_type = "buffer";

  // Nominal
  parameter Real T_supply;
  parameter Real T_return;
  parameter Real dT = T_supply - T_return;
  parameter Real cp = 4200.0;
  parameter Real rho = 988.0;
  parameter Real Heat_nominal = cp * rho * dT;

  parameter Real heat_transfer_coeff = 1;
  parameter Real height = 5;
  parameter Real radius= 10;
  parameter Real heat_loss_coeff = 2 * heat_transfer_coeff / (radius * rho * cp);

  // Initial value
  parameter Real init_Heat = 0.0;

  // Stored_heat is the heat that is contained in the buffer.
  // Heat_buffer is the amount of heat added to or extracted from the buffer
  // per timestep.
  // HeatHot (resp. HeatCold) is the amount of heat added or extracted from
  // the hot (resp. cold) line. 
  // As by construction the cold line should have zero heat, we fix HeatCold to zero.
  // Thus Heat_buffer = HeatHot = der(Stored_heat).

  Modelica.SIunits.Heat Heat_buffer(nominal=cp * rho * dT);
  Modelica.SIunits.Heat Stored_heat(min = 0.0, nominal=cp * rho * dT);
  Modelica.SIunits.Heat Heat_loss(min = 0.0, nominal=cp * rho * dT);
  Modelica.SIunits.Heat HeatHot(nominal=cp * rho * dT);
  Modelica.SIunits.Heat HeatCold(min=0.0, max=0.0, nominal=cp * rho * dT);

equation

  // Heat stored in the buffer
  (der(Stored_heat) - Heat_buffer + Heat_loss)/(Heat_nominal) = 0.0;
  (Heat_loss - Stored_heat * heat_loss_coeff) / Heat_nominal = 0.0;
  (Heat_buffer - (HeatHot - HeatCold))/Heat_nominal = 0.0;

  // Aliases
  (HeatCold - HeatOut.Heat)/Heat_nominal = 0.0;
  (HeatHot - HeatIn.Heat)/Heat_nominal = 0.0;

initial equation
  (Stored_heat - init_Heat)/Heat_nominal = 0.0;
 end Buffer;
