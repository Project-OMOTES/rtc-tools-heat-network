within WarmingUp.HeatNetwork.QTH;

block Buffer
  import SI = Modelica.SIunits;
  extends QTHTwoPort;
  parameter String component_type = "buffer";

  parameter Real cp = 4200.0;
  parameter Real rho = 988.0;
  parameter Real head_loss = 0.0;
  //Source for U estimates: //https://www.spiraxsarco.com/learn-about-steam/steam-engineering-principles-and-heat-transfer/energy-consumption-of-tanks-and-vats
  parameter Real heat_transfer_coeff = 1;
  parameter Real height = 5;
  parameter Real radius= 10;
  parameter Real volume = 3.14*radius^2 * height ;
  //if the tank is plced on ground, ignore surface of bottom of tank
  parameter Real surface = 2 * 3.14 * radius^2 + 2 * 3.14 * radius * height;
  parameter Real T_outside = 10.0;

  // Nominals
  parameter Real nom_heat = cp * rho * dT;
  parameter Real nom_tank_volume = volume/2;
  parameter Real T_supply;
  parameter Real T_return;
  parameter Real dT = T_supply - T_return;

  // Initial values
  parameter Real init_V_hot_tank = 0.0;
  parameter Real init_T_hot_tank = T_supply;
  parameter Real init_T_cold_tank = T_return;

  // Volume of the tanks
  SI.Volume V_hot_tank(min=0.0, max=volume, nominal=nom_tank_volume);
  SI.Volume V_cold_tank(min=0.0, max=volume, nominal=nom_tank_volume);

  // Temperature in the tanks
  SI.Temperature T_hot_tank(min=T_outside, nominal=T_supply);
  SI.Temperature T_cold_tank(min=T_outside, nominal=T_return);

  // Alias variables for the in/out pipes.
  SI.VolumeFlowRate Q_hot_pipe;
  SI.VolumeFlowRate Q_cold_pipe;
  SI.Temperature T_hot_pipe(nominal=T_supply);
  SI.Temperature T_cold_pipe(nominal=T_return);

equation

  // Buffer is modelled with an hot and a cold tank.
  // The hot tank is connected to the supply line, while the cold one to the return line.
  // The total volume of the system is constant.
  // Assumption: constant density and heat capacity
  // Q_hot_pipe is positive when the buffer is charging, negative if discharching.
  // Note that, as volume is constant, the amount of hot and cold water discharged must be equal.
  // For convention, we assume that Q_hot_pipe and Q_cold_pipe have the same sign, i.e.,
  // Q_hot_pipe = Q_cold_pipe.

  // Volume
  ((V_hot_tank + V_cold_tank) - volume) / volume = 0.0;
  der(V_hot_tank) - Q_hot_pipe = 0.0;

  // The following relationships are set in the Python script:

  // * Temperatures of the in/out pipes which are direction dependend.
  // If Q_hot_pipe < 0.0, T_hot_pipe = T_hot_tank.
  // Else, T_cold_pipe = T_cold_tank.
  // T_hot_pipe and T_cold pipe must be the one coming from the network
  // in the other timesteps.

  // * Heat equations:
  // der(V_hot_tank * T_hot_tank) - (Q_hot_pipe * T_hot_pipe) = 0.0;
  // der(V_cold_tank * T_hot_tank) - (Q_cold_pipe * T_cold_pipe) = 0.0;

  // * Flows in/out:
  // recall that Q_hot_pipe and Q_cold_pipe have the same sign.
  // To compensate for pipe orientations, we have:
  // hot_pipe_orientation * QTHIn.Q = Q_hot_pipe;
  // -1 * cold_pipe_orientation * QTHOut.Q = Q_cold_pipe;

  // Aliases
  QTHIn.T = T_hot_pipe;
  QTHOut.T = T_cold_pipe;

initial equation
  // Set volume and temperatures at t0
  V_hot_tank - init_V_hot_tank= 0.0;
  T_hot_tank - init_T_hot_tank = 0.0;
  T_cold_tank - init_T_cold_tank = 0.0;

 end Buffer;
