within WarmingUp.HeatNetwork.Heat;

block Buffer
  import SI = Modelica.SIunits;
  extends HeatTwoPort;
  parameter String component_type = "buffer";

  parameter Real Q_nominal = 1.0;
  parameter Real T_supply;
  parameter Real T_return;
  parameter Real dT = T_supply - T_return;
  parameter Real cp = 4200.0;
  parameter Real rho = 988.0;
  constant Real pi = 3.141592653589793;
  parameter Real Heat_nominal = cp * rho * dT * Q_nominal;

  parameter Real heat_transfer_coeff = 1;
  parameter Real height = 5;
  parameter Real radius= 10;
  parameter Real volume = pi*radius^2 * height;
  parameter Real heat_loss_coeff = 2 * heat_transfer_coeff / (radius * rho * cp);
  // The hot/cold tank can have a lower bound on its volume. Meaning that they might always be, for e.g., 5% full.
  parameter Real min_fraction_tank_volume = 0.05;

  // Initial values
  parameter Real init_V_hot_tank;
  parameter Real init_Heat;

  // Minimum/maximum values
  parameter Real min_stored_heat = volume * min_fraction_tank_volume * dT * cp * rho;
  parameter Real max_stored_heat = volume * (1 - min_fraction_tank_volume) * dT * cp * rho;

  // Stored_heat is the heat that is contained in the buffer.
  // Heat_buffer is the amount of heat added to or extracted from the buffer
  // per timestep.
  // HeatHot (resp. HeatCold) is the amount of heat added or extracted from
  // the hot (resp. cold) line. 
  // As by construction the cold line should have zero heat, we fix HeatCold to zero.
  // Thus Heat_buffer = HeatHot = der(Stored_heat).
  Modelica.SIunits.Heat Heat_buffer(nominal=Heat_nominal);
  // Assume the storage fills in about an hour at typical rate
  parameter Modelica.SIUnits.Duration _typical_fill_time = 3600.0;
  parameter Modelica.SIunits.Heat _nominal_stored_heat = Heat_nominal * _typical_fill_time;
  Modelica.SIunits.Heat Stored_heat(min=min_stored_heat, max=max_stored_heat, nominal=_nominal_stored_heat);
  // For nicer constraint coefficient scaling, we shift a bit more error into
  // the state vector entry of `Heat_loss`. In other words, with a factor of
  // 10.0, we aim for a state vector entry of ~0.1 (instead of 1.0)
  parameter Real _heat_loss_error_to_state_factor = 10.0;
  parameter Modelica.SIunits.Heat _nominal_heat_loss = _nominal_stored_heat * heat_loss_coeff * _heat_loss_error_to_state_factor;
  Modelica.SIunits.Heat Heat_loss(min = 0.0, nominal=_nominal_heat_loss);
  Modelica.SIunits.Heat HeatHot(nominal=Heat_nominal);
  Modelica.SIunits.Heat HeatCold(min=0.0, max=0.0, nominal=Heat_nominal);

  parameter Real _heat_loss_eq_nominal_buf = sqrt(Heat_nominal * _nominal_heat_loss);
equation
  HeatIn.Q = HeatOut.Q;

  // Heat stored in the buffer
  (der(Stored_heat) - Heat_buffer + Heat_loss) / _heat_loss_eq_nominal_buf = 0.0;
  (Heat_loss - Stored_heat * heat_loss_coeff) / _nominal_heat_loss = 0.0;
  (Heat_buffer - (HeatHot - HeatCold))/Heat_nominal = 0.0;

  // Set in Mixin. We want HeatHot to be positive when the buffer is
  // charging, which means we need to know the orientation of the connected
  // pipe.
  // (HeatCold + cold_pipe_orientation * HeatOut.Heat) / Heat_nominal = 0.0;
  // (HeatHot - hot_pipe_orientation * HeatIn.Heat) / Heat_nominal = 0.0;
 end Buffer;
