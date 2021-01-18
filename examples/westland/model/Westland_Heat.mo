model Westland_Heat
  // Declare Model Elements

  parameter Real init_Heat = 0.0;

  constant Real heat_nominal_cold_pipes = 35.0*4200*988*0.01;
  constant Real heat_nominal_hot_pipes = 85.0*4200*988*0.01;
  constant Real heat_max_abs = 0.5*100.0*4200.0*988;

  // Set default temperatures
  parameter Real T_supply = 85.0;
  parameter Real T_return = 35.0;
  parameter Real T_ground = 10.0;

  //Heatsource min en max in [W]
  WarmingUp.HeatNetwork.Heat.Source Source1(Heat_source(min=0.0, max=12.5e6, nominal=1e7), Heat_in(nominal=heat_nominal_cold_pipes), Heat_out(nominal=heat_nominal_hot_pipes), T_supply=85.0, T_return=35.0);
  WarmingUp.HeatNetwork.Heat.Source Source2(Heat_source(min=0.0, max=17.5e6, nominal=1e7), Heat_in(nominal=heat_nominal_cold_pipes), Heat_out(nominal=heat_nominal_hot_pipes), T_supply=85.0, T_return=35.0);
  WarmingUp.HeatNetwork.Heat.Source Source3(Heat_source(min=0.0, max=15e6, nominal=1e7), Heat_in(nominal=heat_nominal_cold_pipes), Heat_out(nominal=heat_nominal_hot_pipes), T_supply=85.0, T_return=35.0);
  WarmingUp.HeatNetwork.Heat.Source Source12(Heat_source(min=0.0, max=40e6, nominal=1e7), Heat_in(nominal=heat_nominal_cold_pipes), Heat_out(nominal=heat_nominal_hot_pipes), T_supply=85.0, T_return=35.0);
  WarmingUp.HeatNetwork.Heat.Source Source4(Heat_source(min=0.0, max=17.5e6, nominal=1e7), Heat_in(nominal=heat_nominal_cold_pipes), Heat_out(nominal=heat_nominal_hot_pipes), T_supply=85.0, T_return=35.0);
  WarmingUp.HeatNetwork.Heat.Source Source6(Heat_source(min=0.0, max=40e6, nominal=1e7), Heat_in(nominal=heat_nominal_cold_pipes), Heat_out(nominal=heat_nominal_hot_pipes), T_supply=85.0, T_return=35.0);
  WarmingUp.HeatNetwork.Heat.Source Source5(Heat_source(min=0.0, max=12.5e6, nominal=1e7), Heat_in(nominal=heat_nominal_cold_pipes), Heat_out(nominal=heat_nominal_hot_pipes), T_supply=85.0, T_return=35.0);

  WarmingUp.HeatNetwork.Heat.Pipe pipeS1_cold(disconnectable=false, has_control_valve=false, length=5., diameter=0.2, temperature=35.0, T_supply=85.0, T_return=35.0, T_ground = 10.0, HeatOut.Heat(min=0.0, max=0.0, nominal=heat_nominal_cold_pipes), HeatIn.Heat(min=0.0, max=0.0, nominal=heat_nominal_cold_pipes));
  WarmingUp.HeatNetwork.Heat.Pipe pipeS2_cold(disconnectable=false, has_control_valve=false, length=5., diameter=0.2, temperature=35.0, T_supply=85.0, T_return=35.0, T_ground = 10.0, HeatOut.Heat(min=0.0, max=0.0, nominal=heat_nominal_cold_pipes), HeatIn.Heat(min=0.0, max=0.0, nominal=heat_nominal_cold_pipes));
  WarmingUp.HeatNetwork.Heat.Pipe pipeS1and2_cold(disconnectable=false, has_control_valve=false, length=5., diameter=0.2, temperature=35.0, T_supply=85.0, T_return=35.0, T_ground = 10.0, HeatOut.Heat(min=0.0, max=0.0, nominal=heat_nominal_cold_pipes), HeatIn.Heat(min=0.0, max=0.0, nominal=heat_nominal_cold_pipes));
  WarmingUp.HeatNetwork.Heat.Pipe pipeDH_cold(disconnectable=false, has_control_valve=false, length=5., diameter=0.2, temperature=35.0, T_supply=85.0, T_return=35.0, T_ground = 10.0, HeatOut.Heat(min=0.0, max=0.0, nominal=heat_nominal_cold_pipes), HeatIn.Heat(min=0.0, max=0.0, nominal=heat_nominal_cold_pipes));
  WarmingUp.HeatNetwork.Heat.Pipe pipeDG_cold(disconnectable=false, has_control_valve=false, length=5., diameter=0.2, temperature=35.0, T_supply=85.0, T_return=35.0, T_ground = 10.0, HeatOut.Heat(min=0.0, max=0.0, nominal=heat_nominal_cold_pipes), HeatIn.Heat(min=0.0, max=0.0, nominal=heat_nominal_cold_pipes));
  WarmingUp.HeatNetwork.Heat.Pipe pipeDF_cold(disconnectable=false, has_control_valve=false, length=5., diameter=0.2, temperature=35.0, T_supply=85.0, T_return=35.0, T_ground = 10.0, HeatOut.Heat(min=0.0, max=0.0, nominal=heat_nominal_cold_pipes), HeatIn.Heat(min=0.0, max=0.0, nominal=heat_nominal_cold_pipes));
  WarmingUp.HeatNetwork.Heat.Pipe pipeDI_cold(disconnectable=false, has_control_valve=false, length=5., diameter=0.2, temperature=35.0, T_supply=85.0, T_return=35.0, T_ground = 10.0, HeatOut.Heat(min=0.0, max=0.0, nominal=heat_nominal_cold_pipes), HeatIn.Heat(min=0.0, max=0.0, nominal=heat_nominal_cold_pipes));
  WarmingUp.HeatNetwork.Heat.Pipe pipeDE_cold(disconnectable=false, has_control_valve=false, length=5., diameter=0.2, temperature=35.0, T_supply=85.0, T_return=35.0, T_ground = 10.0, HeatOut.Heat(min=0.0, max=0.0, nominal=heat_nominal_cold_pipes), HeatIn.Heat(min=0.0, max=0.0, nominal=heat_nominal_cold_pipes));
  WarmingUp.HeatNetwork.Heat.Pipe pipeDB_cold(disconnectable=false, has_control_valve=false, length=5., diameter=0.2, temperature=35.0, T_supply=85.0, T_return=35.0, T_ground = 10.0, HeatOut.Heat(min=0.0, max=0.0, nominal=heat_nominal_cold_pipes), HeatIn.Heat(min=0.0, max=0.0, nominal=heat_nominal_cold_pipes));
  WarmingUp.HeatNetwork.Heat.Pipe pipeDA_cold(disconnectable=false, has_control_valve=false, length=5., diameter=0.2, temperature=35.0, T_supply=85.0, T_return=35.0, T_ground = 10.0, HeatOut.Heat(min=0.0, max=0.0, nominal=heat_nominal_cold_pipes), HeatIn.Heat(min=0.0, max=0.0, nominal=heat_nominal_cold_pipes));
  WarmingUp.HeatNetwork.Heat.Pipe pipeDC_cold(disconnectable=false, has_control_valve=false, length=5., diameter=0.2, temperature=35.0, T_supply=85.0, T_return=35.0, T_ground = 10.0, HeatOut.Heat(min=0.0, max=0.0, nominal=heat_nominal_cold_pipes), HeatIn.Heat(min=0.0, max=0.0, nominal=heat_nominal_cold_pipes));
  WarmingUp.HeatNetwork.Heat.Pipe pipeDD_cold(disconnectable=false, has_control_valve=false, length=5., diameter=0.2, temperature=35.0, T_supply=85.0, T_return=35.0, T_ground = 10.0, HeatOut.Heat(min=0.0, max=0.0, nominal=heat_nominal_cold_pipes), HeatIn.Heat(min=0.0, max=0.0, nominal=heat_nominal_cold_pipes));
  WarmingUp.HeatNetwork.Heat.Pipe pipeS5_cold(disconnectable=false, has_control_valve=false, length=5., diameter=0.2, temperature=35.0, T_supply=85.0, T_return=35.0, T_ground = 10.0, HeatOut.Heat(min=0.0, max=0.0, nominal=heat_nominal_cold_pipes), HeatIn.Heat(min=0.0, max=0.0, nominal=heat_nominal_cold_pipes));
  WarmingUp.HeatNetwork.Heat.Pipe pipeS3_cold(disconnectable=false, has_control_valve=false, length=5., diameter=0.2, temperature=35.0, T_supply=85.0, T_return=35.0, T_ground = 10.0, HeatOut.Heat(min=0.0, max=0.0, nominal=heat_nominal_cold_pipes), HeatIn.Heat(min=0.0, max=0.0, nominal=heat_nominal_cold_pipes));
  WarmingUp.HeatNetwork.Heat.Pipe pipeS4_cold(disconnectable=false, has_control_valve=false, length=5., diameter=0.2, temperature=35.0, T_supply=85.0, T_return=35.0, T_ground = 10.0, HeatOut.Heat(min=0.0, max=0.0, nominal=heat_nominal_cold_pipes), HeatIn.Heat(min=0.0, max=0.0, nominal=heat_nominal_cold_pipes));
  WarmingUp.HeatNetwork.Heat.Pipe pipeS6_cold(disconnectable=false, has_control_valve=false, length=5., diameter=0.2, temperature=35.0, T_supply=85.0, T_return=35.0, T_ground = 10.0, HeatOut.Heat(min=0.0, max=0.0, nominal=heat_nominal_cold_pipes), HeatIn.Heat(min=0.0, max=0.0, nominal=heat_nominal_cold_pipes));
  WarmingUp.HeatNetwork.Heat.Pipe pipe4_cold(disconnectable=false, has_control_valve=true, length=2000., diameter=0.25, temperature=35.0, T_supply=85.0, T_return=35.0, T_ground = 10.0, HeatOut.Heat(min=0.0, max=0.0, nominal=heat_nominal_cold_pipes), HeatIn.Heat(min=0.0, max=0.0, nominal=heat_nominal_cold_pipes));
  WarmingUp.HeatNetwork.Heat.Pipe pipe3_cold(disconnectable=false, has_control_valve=false, length=2500., diameter=0.3, temperature=35.0, T_supply=85.0, T_return=35.0, T_ground = 10.0, HeatOut.Heat(min=0.0, max=0.0, nominal=heat_nominal_cold_pipes), HeatIn.Heat(min=0.0, max=0.0, nominal=heat_nominal_cold_pipes));
  WarmingUp.HeatNetwork.Heat.Pipe pipe5_cold(disconnectable=false, has_control_valve=false, length=2900., diameter=0.5, temperature=35.0, T_supply=85.0, T_return=35.0, T_ground = 10.0, HeatOut.Heat(min=0.0, max=0.0, nominal=heat_nominal_cold_pipes), HeatIn.Heat(min=0.0, max=0.0, nominal=heat_nominal_cold_pipes));
  WarmingUp.HeatNetwork.Heat.Pipe pipe2_cold(disconnectable=false, has_control_valve=false, length=3900., diameter=0.4, temperature=35.0, T_supply=85.0, T_return=35.0, T_ground = 10.0, HeatOut.Heat(min=0.0, max=0.0, nominal=heat_nominal_cold_pipes), HeatIn.Heat(min=0.0, max=0.0, nominal=heat_nominal_cold_pipes));
  WarmingUp.HeatNetwork.Heat.Pipe pipe1_cold(disconnectable=false, has_control_valve=false, length=2700., diameter=0.25, temperature=35.0, T_supply=85.0, T_return=35.0, T_ground = 10.0, HeatOut.Heat(min=0.0, max=0.0, nominal=heat_nominal_cold_pipes), HeatIn.Heat(min=0.0, max=0.0, nominal=heat_nominal_cold_pipes));

  WarmingUp.HeatNetwork.Heat.Pipe pipepump1_cold(disconnectable=false, has_control_valve=false, length=1., diameter=0.25, temperature=35.0, T_supply=85.0, T_return=35.0, T_ground = 10.0, HeatOut.Heat(min=0.0, max=0.0, nominal=heat_nominal_cold_pipes), HeatIn.Heat(min=0.0, max=0.0, nominal=heat_nominal_cold_pipes));
  WarmingUp.HeatNetwork.Heat.Pipe pipepump2_cold(disconnectable=false, has_control_valve=false, length=1., diameter=0.25, temperature=35.0, T_supply=85.0, T_return=35.0, T_ground = 10.0, HeatOut.Heat(min=0.0, max=0.0, nominal=heat_nominal_cold_pipes), HeatIn.Heat(min=0.0, max=0.0, nominal=heat_nominal_cold_pipes));
  WarmingUp.HeatNetwork.Heat.Pipe pipepump3_cold(disconnectable=false, has_control_valve=false, length=1., diameter=0.25, temperature=35.0, T_supply=85.0, T_return=35.0, T_ground = 10.0, HeatOut.Heat(min=0.0, max=0.0, nominal=heat_nominal_cold_pipes), HeatIn.Heat(min=0.0, max=0.0, nominal=heat_nominal_cold_pipes));
  WarmingUp.HeatNetwork.Heat.Pipe pipepump4_cold(disconnectable=false, has_control_valve=false, length=1., diameter=0.25, temperature=35.0, T_supply=85.0, T_return=35.0, T_ground = 10.0, HeatOut.Heat(min=0.0, max=0.0, nominal=heat_nominal_cold_pipes), HeatIn.Heat(min=0.0, max=0.0, nominal=heat_nominal_cold_pipes));
  WarmingUp.HeatNetwork.Heat.Pipe pipepump5_cold(disconnectable=false, has_control_valve=false, length=1., diameter=0.25, temperature=35.0, T_supply=85.0, T_return=35.0, T_ground = 10.0, HeatOut.Heat(min=0.0, max=0.0, nominal=heat_nominal_cold_pipes), HeatIn.Heat(min=0.0, max=0.0, nominal=heat_nominal_cold_pipes));
  WarmingUp.HeatNetwork.Heat.Pipe pipepump6_cold(disconnectable=false, has_control_valve=false, length=1., diameter=0.25, temperature=35.0, T_supply=85.0, T_return=35.0, T_ground = 10.0, HeatOut.Heat(min=0.0, max=0.0, nominal=heat_nominal_cold_pipes), HeatIn.Heat(min=0.0, max=0.0, nominal=heat_nominal_cold_pipes));
  WarmingUp.HeatNetwork.Heat.Pipe pipepump12_cold(disconnectable=false, has_control_valve=false, length=1., diameter=0.25, temperature=35.0, T_supply=85.0, T_return=35.0, T_ground = 10.0, HeatOut.Heat(min=0.0, max=0.0, nominal=heat_nominal_cold_pipes), HeatIn.Heat(min=0.0, max=0.0, nominal=heat_nominal_cold_pipes));

  WarmingUp.HeatNetwork.Heat.Pipe pipe6a_hot(disconnectable=false, has_control_valve=true, length=2700., diameter=0.25, temperature=85.0, T_supply=85.0, T_return=35.0, T_ground = 10.0, HeatOut.Heat(min=0.0, max=heat_max_abs, nominal=heat_nominal_hot_pipes), HeatIn.Heat(min=0.0, max=heat_max_abs, nominal=heat_nominal_hot_pipes));
  WarmingUp.HeatNetwork.Heat.Pipe pipe6b_hot(disconnectable=false, has_control_valve=false, length=2700., diameter=0.25, temperature=85.0, T_supply=85.0, T_return=35.0, T_ground = 10.0, HeatOut.Heat(min=0.0, max=heat_max_abs, nominal=heat_nominal_hot_pipes), HeatIn.Heat(min=0.0, max=heat_max_abs, nominal=heat_nominal_hot_pipes));
  WarmingUp.HeatNetwork.Heat.Pipe pipe7a_hot(disconnectable=false, has_control_valve=true, length=2600., diameter=0.5, temperature=85.0, T_supply=85.0, T_return=35.0, T_ground = 10.0, HeatOut.Heat(min=0.0, max=heat_max_abs, nominal=heat_nominal_hot_pipes), HeatIn.Heat(min=0.0, max=heat_max_abs, nominal=heat_nominal_hot_pipes));
  WarmingUp.HeatNetwork.Heat.Pipe pipe7b_hot(disconnectable=false, has_control_valve=false, length=2600., diameter=0.5, temperature=85.0, T_supply=85.0, T_return=35.0, T_ground = 10.0, HeatOut.Heat(min=0.0, max=heat_max_abs, nominal=heat_nominal_hot_pipes), HeatIn.Heat(min=0.0, max=heat_max_abs, nominal=heat_nominal_hot_pipes));
  WarmingUp.HeatNetwork.Heat.Pipe pipe8a_hot(disconnectable=false, has_control_valve=false, length=3300., diameter=0.4, temperature=85.0, T_supply=85.0, T_return=35.0, T_ground = 10.0, HeatOut.Heat(min=0.0, max=heat_max_abs, nominal=heat_nominal_hot_pipes), HeatIn.Heat(min=0.0, max=heat_max_abs, nominal=heat_nominal_hot_pipes));
  WarmingUp.HeatNetwork.Heat.Pipe pipeR1_hot(disconnectable=false, has_control_valve=true, length=5200., diameter=0.3, temperature=85.0, T_supply=85.0, T_return=35.0, T_ground = 10.0, HeatOut.Heat(min=0.0, max=heat_max_abs, nominal=heat_nominal_hot_pipes), HeatIn.Heat(min=0.0, max=heat_max_abs, nominal=heat_nominal_hot_pipes));
  WarmingUp.HeatNetwork.Heat.Pipe pipe9a_hot(disconnectable=false, has_control_valve=false, length=3300., diameter=0.4, temperature=85.0, T_supply=85.0, T_return=35.0, T_ground = 10.0, HeatOut.Heat(min=0.0, max=heat_max_abs, nominal=heat_nominal_hot_pipes), HeatIn.Heat(min=0.0, max=heat_max_abs, nominal=heat_nominal_hot_pipes));
  WarmingUp.HeatNetwork.Heat.Pipe pipe9b_hot(disconnectable=false, has_control_valve=false, length=1600., diameter=0.4, temperature=85.0, T_supply=85.0, T_return=35.0, T_ground = 10.0, HeatOut.Heat(min=0.0, max=heat_max_abs, nominal=heat_nominal_hot_pipes), HeatIn.Heat(min=0.0, max=heat_max_abs, nominal=heat_nominal_hot_pipes));
  WarmingUp.HeatNetwork.Heat.Pipe pipeR2_hot(disconnectable=false, has_control_valve=true, length=4000., diameter=0.3, temperature=85.0, T_supply=85.0, T_return=35.0, T_ground = 10.0, HeatOut.Heat(min=-heat_max_abs, max=heat_max_abs, nominal=heat_nominal_hot_pipes), HeatIn.Heat(min=-heat_max_abs, max=heat_max_abs, nominal=heat_nominal_hot_pipes));

  WarmingUp.HeatNetwork.Heat.Pipe pipeS1_hot(disconnectable=false, has_control_valve=false, length=5., diameter=0.2, temperature=85.0, T_supply=85.0, T_return=35.0, T_ground = 10.0, HeatOut.Heat(min=0.0, max=heat_max_abs, nominal=heat_nominal_hot_pipes), HeatIn.Heat(min=0.0, max=heat_max_abs, nominal=heat_nominal_hot_pipes));
  WarmingUp.HeatNetwork.Heat.Pipe pipeS2_hot(disconnectable=false, has_control_valve=false, length=5., diameter=0.2, temperature=85.0, T_supply=85.0, T_return=35.0, T_ground = 10.0, HeatOut.Heat(min=0.0, max=heat_max_abs, nominal=heat_nominal_hot_pipes), HeatIn.Heat(min=0.0, max=heat_max_abs, nominal=heat_nominal_hot_pipes));
  WarmingUp.HeatNetwork.Heat.Pipe pipeS1and2_hot(disconnectable=false, has_control_valve=false, length=5., diameter=0.2, temperature=85.0, T_supply=85.0, T_return=35.0, T_ground = 10.0, HeatOut.Heat(min=0.0, max=heat_max_abs, nominal=heat_nominal_hot_pipes), HeatIn.Heat(min=0.0, max=heat_max_abs, nominal=heat_nominal_hot_pipes));
  WarmingUp.HeatNetwork.Heat.Pipe pipeDH_hot(disconnectable=false, has_control_valve=false, length=5., diameter=0.2, temperature=85.0, T_supply=85.0, T_return=35.0, T_ground = 10.0, HeatOut.Heat(min=0.0, max=heat_max_abs, nominal=heat_nominal_hot_pipes), HeatIn.Heat(min=0.0, max=heat_max_abs, nominal=heat_nominal_hot_pipes));
  WarmingUp.HeatNetwork.Heat.Pipe pipeDG_hot(disconnectable=false, has_control_valve=false, length=5., diameter=0.2, temperature=85.0, T_supply=85.0, T_return=35.0, T_ground = 10.0, HeatOut.Heat(min=0.0, max=heat_max_abs, nominal=heat_nominal_hot_pipes), HeatIn.Heat(min=0.0, max=heat_max_abs, nominal=heat_nominal_hot_pipes));
  WarmingUp.HeatNetwork.Heat.Pipe pipeDF_hot(disconnectable=false, has_control_valve=false, length=5., diameter=0.2, temperature=85.0, T_supply=85.0, T_return=35.0, T_ground = 10.0, HeatOut.Heat(min=0.0, max=heat_max_abs, nominal=heat_nominal_hot_pipes), HeatIn.Heat(min=0.0, max=heat_max_abs, nominal=heat_nominal_hot_pipes));
  WarmingUp.HeatNetwork.Heat.Pipe pipeDE_hot(disconnectable=false, has_control_valve=false, length=5., diameter=0.2, temperature=85.0, T_supply=85.0, T_return=35.0, T_ground = 10.0, HeatOut.Heat(min=0.0, max=heat_max_abs, nominal=heat_nominal_hot_pipes), HeatIn.Heat(min=0.0, max=heat_max_abs, nominal=heat_nominal_hot_pipes));
  WarmingUp.HeatNetwork.Heat.Pipe pipeDB_hot(disconnectable=false, has_control_valve=false, length=5., diameter=0.2, temperature=85.0, T_supply=85.0, T_return=35.0, T_ground = 10.0, HeatOut.Heat(min=0.0, max=heat_max_abs, nominal=heat_nominal_hot_pipes), HeatIn.Heat(min=0.0, max=heat_max_abs, nominal=heat_nominal_hot_pipes));
  WarmingUp.HeatNetwork.Heat.Pipe pipeDA_hot(disconnectable=false, has_control_valve=false, length=5., diameter=0.2, temperature=85.0, T_supply=85.0, T_return=35.0, T_ground = 10.0, HeatOut.Heat(min=0.0, max=heat_max_abs, nominal=heat_nominal_hot_pipes), HeatIn.Heat(min=0.0, max=heat_max_abs, nominal=heat_nominal_hot_pipes));
  WarmingUp.HeatNetwork.Heat.Pipe pipeDC_hot(disconnectable=false, has_control_valve=false, length=5., diameter=0.2, temperature=85.0, T_supply=85.0, T_return=35.0, T_ground = 10.0, HeatOut.Heat(min=0.0, max=heat_max_abs, nominal=heat_nominal_hot_pipes), HeatIn.Heat(min=0.0, max=heat_max_abs, nominal=heat_nominal_hot_pipes));
  WarmingUp.HeatNetwork.Heat.Pipe pipeDI_hot(disconnectable=false, has_control_valve=false, length=5., diameter=0.2, temperature=85.0, T_supply=85.0, T_return=35.0, T_ground = 10.0, HeatOut.Heat(min=0.0, max=heat_max_abs, nominal=heat_nominal_hot_pipes), HeatIn.Heat(min=0.0, max=heat_max_abs, nominal=heat_nominal_hot_pipes));
  WarmingUp.HeatNetwork.Heat.Pipe pipeDD_hot(disconnectable=false, has_control_valve=false, length=5., diameter=0.2, temperature=85.0, T_supply=85.0, T_return=35.0, T_ground = 10.0, HeatOut.Heat(min=0.0, max=heat_max_abs, nominal=heat_nominal_hot_pipes), HeatIn.Heat(min=0.0, max=heat_max_abs, nominal=heat_nominal_hot_pipes));
  WarmingUp.HeatNetwork.Heat.Pipe pipeS5_hot(disconnectable=false, has_control_valve=false, length=5., diameter=0.2, temperature=85.0, T_supply=85.0, T_return=35.0, T_ground = 10.0, HeatOut.Heat(min=0.0, max=heat_max_abs, nominal=heat_nominal_hot_pipes), HeatIn.Heat(min=0.0, max=heat_max_abs, nominal=heat_nominal_hot_pipes));
  WarmingUp.HeatNetwork.Heat.Pipe pipeS3_hot(disconnectable=false, has_control_valve=false, length=5., diameter=0.2, temperature=85.0, T_supply=85.0, T_return=35.0, T_ground = 10.0, HeatOut.Heat(min=0.0, max=heat_max_abs, nominal=heat_nominal_hot_pipes), HeatIn.Heat(min=0.0, max=heat_max_abs, nominal=heat_nominal_hot_pipes));
  WarmingUp.HeatNetwork.Heat.Pipe pipeS4_hot(disconnectable=false, has_control_valve=false, length=5., diameter=0.2, temperature=85.0, T_supply=85.0, T_return=35.0, T_ground = 10.0, HeatOut.Heat(min=0.0, max=heat_max_abs, nominal=heat_nominal_hot_pipes), HeatIn.Heat(min=0.0, max=heat_max_abs, nominal=heat_nominal_hot_pipes));
  WarmingUp.HeatNetwork.Heat.Pipe pipeS6_hot(disconnectable=false, has_control_valve=false, length=5., diameter=0.2, temperature=85.0, T_supply=85.0, T_return=35.0, T_ground = 10.0, HeatOut.Heat(min=0.0, max=heat_max_abs, nominal=heat_nominal_hot_pipes), HeatIn.Heat(min=0.0, max=heat_max_abs, nominal=heat_nominal_hot_pipes));
  WarmingUp.HeatNetwork.Heat.Pipe pipe4_hot(disconnectable=false, has_control_valve=true, length=2000., diameter=0.25, temperature=85.0, T_supply=85.0, T_return=35.0, T_ground = 10.0, HeatOut.Heat(min=-heat_max_abs, max=heat_max_abs, nominal=heat_nominal_hot_pipes), HeatIn.Heat(min=-heat_max_abs, max=heat_max_abs, nominal=heat_nominal_hot_pipes));
  WarmingUp.HeatNetwork.Heat.Pipe pipe3_hot(disconnectable=false, has_control_valve=false, length=2500., diameter=0.3, temperature=85.0, T_supply=85.0, T_return=35.0, T_ground = 10.0, HeatOut.Heat(min=0.0, max=heat_max_abs, nominal=heat_nominal_hot_pipes), HeatIn.Heat(min=0.0, max=heat_max_abs, nominal=heat_nominal_hot_pipes));
  WarmingUp.HeatNetwork.Heat.Pipe pipe5_hot(disconnectable=false, has_control_valve=false, length=2900., diameter=0.5, temperature=85.0, T_supply=85.0, T_return=35.0, T_ground = 10.0, HeatOut.Heat(min=0.0, max=heat_max_abs, nominal=heat_nominal_hot_pipes), HeatIn.Heat(min=0.0, max=heat_max_abs, nominal=heat_nominal_hot_pipes));
  WarmingUp.HeatNetwork.Heat.Pipe pipe2_hot(disconnectable=false, has_control_valve=false, length=2100., diameter=0.4, temperature=85.0, T_supply=85.0, T_return=35.0, T_ground = 10.0, HeatOut.Heat(min=0.0, max=heat_max_abs, nominal=heat_nominal_hot_pipes), HeatIn.Heat(min=0.0, max=heat_max_abs, nominal=heat_nominal_hot_pipes));
  WarmingUp.HeatNetwork.Heat.Pipe pipe1_hot(disconnectable=false, has_control_valve=false, length=1400., diameter=0.25, temperature=85.0, T_supply=85.0, T_return=35.0, T_ground = 10.0, HeatOut.Heat(min=0.0, max=heat_max_abs, nominal=heat_nominal_hot_pipes), HeatIn.Heat(min=0.0, max=heat_max_abs, nominal=heat_nominal_hot_pipes));

  WarmingUp.HeatNetwork.Heat.Pipe pipepump1_hot(disconnectable=false, has_control_valve=false, length=1., diameter=0.25, temperature=85.0, T_supply=85.0, T_return=35.0, T_ground = 10.0, HeatOut.Heat(min=0.0, max=heat_max_abs, nominal=heat_nominal_hot_pipes), HeatIn.Heat(min=0.0, max=heat_max_abs, nominal=heat_nominal_hot_pipes));
  WarmingUp.HeatNetwork.Heat.Pipe pipepump2_hot(disconnectable=false, has_control_valve=false, length=1., diameter=0.25, temperature=85.0, T_supply=85.0, T_return=35.0, T_ground = 10.0, HeatOut.Heat(min=0.0, max=heat_max_abs, nominal=heat_nominal_hot_pipes), HeatIn.Heat(min=0.0, max=heat_max_abs, nominal=heat_nominal_hot_pipes));
  WarmingUp.HeatNetwork.Heat.Pipe pipepump3_hot(disconnectable=false, has_control_valve=false, length=1., diameter=0.25, temperature=85.0, T_supply=85.0, T_return=35.0, T_ground = 10.0, HeatOut.Heat(min=0.0, max=heat_max_abs, nominal=heat_nominal_hot_pipes), HeatIn.Heat(min=0.0, max=heat_max_abs, nominal=heat_nominal_hot_pipes));
  WarmingUp.HeatNetwork.Heat.Pipe pipepump4_hot(disconnectable=false, has_control_valve=false, length=1., diameter=0.25, temperature=85.0, T_supply=85.0, T_return=35.0, T_ground = 10.0, HeatOut.Heat(min=0.0, max=heat_max_abs, nominal=heat_nominal_hot_pipes), HeatIn.Heat(min=0.0, max=heat_max_abs, nominal=heat_nominal_hot_pipes));
  WarmingUp.HeatNetwork.Heat.Pipe pipepump5_hot(disconnectable=false, has_control_valve=false, length=1., diameter=0.25, temperature=85.0, T_supply=85.0, T_return=35.0, T_ground = 10.0, HeatOut.Heat(min=0.0, max=heat_max_abs, nominal=heat_nominal_hot_pipes), HeatIn.Heat(min=0.0, max=heat_max_abs, nominal=heat_nominal_hot_pipes));
  WarmingUp.HeatNetwork.Heat.Pipe pipepump6_hot(disconnectable=false, has_control_valve=false, length=1., diameter=0.25, temperature=85.0, T_supply=85.0, T_return=35.0, T_ground = 10.0, HeatOut.Heat(min=0.0, max=heat_max_abs, nominal=heat_nominal_hot_pipes), HeatIn.Heat(min=0.0, max=heat_max_abs, nominal=heat_nominal_hot_pipes));
  WarmingUp.HeatNetwork.Heat.Pipe pipepump12_hot(disconnectable=false, has_control_valve=false, length=1., diameter=0.25, temperature=85.0, T_supply=85.0, T_return=35.0, T_ground = 10.0, HeatOut.Heat(min=0.0, max=heat_max_abs, nominal=heat_nominal_hot_pipes), HeatIn.Heat(min=0.0, max=heat_max_abs, nominal=heat_nominal_hot_pipes));

  WarmingUp.HeatNetwork.Heat.Pipe pipe6a_cold(disconnectable=false, has_control_valve=true, length=2700., diameter=0.25, temperature=35.0, T_supply=85.0, T_return=35.0, T_ground = 10.0, HeatOut.Heat(min=0.0, max=0.0, nominal=heat_nominal_cold_pipes), HeatIn.Heat(min=0.0, max=0.0, nominal=heat_nominal_cold_pipes));
  WarmingUp.HeatNetwork.Heat.Pipe pipe6b_cold(disconnectable=false, has_control_valve=false, length=2700., diameter=0.25, temperature=35.0, T_supply=85.0, T_return=35.0, T_ground = 10.0, HeatOut.Heat(min=0.0, max=0.0, nominal=heat_nominal_cold_pipes), HeatIn.Heat(min=0.0, max=0.0, nominal=heat_nominal_cold_pipes));
  WarmingUp.HeatNetwork.Heat.Pipe pipe7a_cold(disconnectable=false, has_control_valve=true, length=2600., diameter=0.5, temperature=35.0, T_supply=85.0, T_return=35.0, T_ground = 10.0, HeatOut.Heat(min=0.0, max=0.0, nominal=heat_nominal_cold_pipes), HeatIn.Heat(min=0.0, max=0.0, nominal=heat_nominal_cold_pipes));
  WarmingUp.HeatNetwork.Heat.Pipe pipe7b_cold(disconnectable=false, has_control_valve=false, length=2600., diameter=0.5, temperature=35.0, T_supply=85.0, T_return=35.0, T_ground = 10.0, HeatOut.Heat(min=0.0, max=0.0, nominal=heat_nominal_cold_pipes), HeatIn.Heat(min=0.0, max=0.0, nominal=heat_nominal_cold_pipes));
  WarmingUp.HeatNetwork.Heat.Pipe pipe8a_cold(disconnectable=false, has_control_valve=false, length=3300., diameter=0.4, temperature=35.0, T_supply=85.0, T_return=35.0, T_ground = 10.0, HeatOut.Heat(min=0.0, max=0.0, nominal=heat_nominal_cold_pipes), HeatIn.Heat(min=0.0, max=0.0, nominal=heat_nominal_cold_pipes));
  WarmingUp.HeatNetwork.Heat.Pipe pipeR1_cold(disconnectable=false, has_control_valve=true, length=5200., diameter=0.3, temperature=35.0, T_supply=85.0, T_return=35.0, T_ground = 10.0, HeatOut.Heat(min=0.0, max=0.0, nominal=heat_nominal_cold_pipes), HeatIn.Heat(min=0.0, max=0.0, nominal=heat_nominal_cold_pipes));
  WarmingUp.HeatNetwork.Heat.Pipe pipe9a_cold(disconnectable=false, has_control_valve=false, length=3300., diameter=0.4, temperature=35.0, T_supply=85.0, T_return=35.0, T_ground = 10.0, HeatOut.Heat(min=0.0, max=0.0, nominal=heat_nominal_cold_pipes), HeatIn.Heat(min=0.0, max=0.0, nominal=heat_nominal_cold_pipes));
  WarmingUp.HeatNetwork.Heat.Pipe pipe9b_cold(disconnectable=false, has_control_valve=false, length=1600., diameter=0.4, temperature=35.0, T_supply=85.0, T_return=35.0, T_ground = 10.0, HeatOut.Heat(min=0.0, max=0.0, nominal=heat_nominal_cold_pipes), HeatIn.Heat(min=0.0, max=0.0, nominal=heat_nominal_cold_pipes));
  WarmingUp.HeatNetwork.Heat.Pipe pipeR2_cold(disconnectable=false, has_control_valve=true, length=4000., diameter=0.3, temperature=35.0, T_supply=85.0, T_return=35.0, T_ground = 10.0, HeatOut.Heat(min=0.0, max=0.0, nominal=heat_nominal_cold_pipes), HeatIn.Heat(min=0.0, max=0.0, nominal=heat_nominal_cold_pipes));

  WarmingUp.HeatNetwork.Heat.Demand DemandH(Heat_demand(min=0.0, max=heat_max_abs, nominal=heat_max_abs), Heat_in(nominal=heat_nominal_hot_pipes), Heat_out(nominal=heat_nominal_cold_pipes), T_supply=85.0, T_return=35.0);
  WarmingUp.HeatNetwork.Heat.Demand DemandG(Heat_demand(min=0.0, max=heat_max_abs, nominal=heat_max_abs), Heat_in(nominal=heat_nominal_hot_pipes), Heat_out(nominal=heat_nominal_cold_pipes), T_supply=85.0, T_return=35.0);
  WarmingUp.HeatNetwork.Heat.Demand DemandF(Heat_demand(min=0.0, max=heat_max_abs, nominal=heat_max_abs), Heat_in(nominal=heat_nominal_hot_pipes), Heat_out(nominal=heat_nominal_cold_pipes), T_supply=85.0, T_return=35.0);
  WarmingUp.HeatNetwork.Heat.Demand DemandI(Heat_demand(min=0.0, max=heat_max_abs, nominal=heat_max_abs), Heat_in(nominal=heat_nominal_hot_pipes), Heat_out(nominal=heat_nominal_cold_pipes), T_supply=85.0, T_return=35.0);
  WarmingUp.HeatNetwork.Heat.Demand DemandJ(Heat_demand(min=0.0, max=heat_max_abs, nominal=heat_max_abs), Heat_in(nominal=heat_nominal_hot_pipes), Heat_out(nominal=heat_nominal_cold_pipes), T_supply=85.0, T_return=35.0);
  WarmingUp.HeatNetwork.Heat.Demand DemandE(Heat_demand(min=0.0, max=heat_max_abs, nominal=heat_max_abs), Heat_in(nominal=heat_nominal_hot_pipes), Heat_out(nominal=heat_nominal_cold_pipes), T_supply=85.0, T_return=35.0);
  WarmingUp.HeatNetwork.Heat.Demand DemandB(Heat_demand(min=0.0, max=heat_max_abs, nominal=heat_max_abs), Heat_in(nominal=heat_nominal_hot_pipes), Heat_out(nominal=heat_nominal_cold_pipes), T_supply=85.0, T_return=35.0);
  WarmingUp.HeatNetwork.Heat.Demand DemandC(Heat_demand(min=0.0, max=heat_max_abs, nominal=heat_max_abs), Heat_in(nominal=heat_nominal_hot_pipes), Heat_out(nominal=heat_nominal_cold_pipes), T_supply=85.0, T_return=35.0);
  WarmingUp.HeatNetwork.Heat.Demand DemandA(Heat_demand(min=0.0, max=heat_max_abs, nominal=heat_max_abs), Heat_in(nominal=heat_nominal_hot_pipes), Heat_out(nominal=heat_nominal_cold_pipes), T_supply=85.0, T_return=35.0);
  WarmingUp.HeatNetwork.Heat.Demand DemandD(Heat_demand(min=0.0, max=heat_max_abs, nominal=heat_max_abs), Heat_in(nominal=heat_nominal_hot_pipes), Heat_out(nominal=heat_nominal_cold_pipes), T_supply=85.0, T_return=35.0);

  WarmingUp.HeatNetwork.Heat.Node nodeS12_hot(n=3); 
  WarmingUp.HeatNetwork.Heat.Node nodeS12_cold(n=3);
  WarmingUp.HeatNetwork.Heat.Node nodeDH_hot(n=4); 
  WarmingUp.HeatNetwork.Heat.Node nodeDH_cold(n=4);
  WarmingUp.HeatNetwork.Heat.Node nodeDG_hot(n=3); 
  WarmingUp.HeatNetwork.Heat.Node nodeDG_cold(n=3);
  WarmingUp.HeatNetwork.Heat.Node nodeDF_hot(n=6); 
  WarmingUp.HeatNetwork.Heat.Node nodeDF_cold(n=6);
  WarmingUp.HeatNetwork.Heat.Node nodeDI_hot(n=4); 
  WarmingUp.HeatNetwork.Heat.Node nodeDI_cold(n=4);
  WarmingUp.HeatNetwork.Heat.Node node6_hot(n=3); 
  WarmingUp.HeatNetwork.Heat.Node node6_cold(n=3);
  WarmingUp.HeatNetwork.Heat.Node nodeDE_hot(n=3); 
  WarmingUp.HeatNetwork.Heat.Node nodeDE_cold(n=3);
  WarmingUp.HeatNetwork.Heat.Node nodeDB_hot(n=5); 
  WarmingUp.HeatNetwork.Heat.Node nodeDB_cold(n=5);
  WarmingUp.HeatNetwork.Heat.Node nodeS6_hot(n=3); 
  WarmingUp.HeatNetwork.Heat.Node nodeS6_cold(n=3);
  WarmingUp.HeatNetwork.Heat.Node node8_hot(n=3); 
  WarmingUp.HeatNetwork.Heat.Node node8_cold(n=3);
  WarmingUp.HeatNetwork.Heat.Node node9_hot(n=3); 
  WarmingUp.HeatNetwork.Heat.Node node9_cold(n=3);
  WarmingUp.HeatNetwork.Heat.Node nodeDD_hot(n=3); 
  WarmingUp.HeatNetwork.Heat.Node nodeDD_cold(n=3);

  WarmingUp.HeatNetwork.Heat.Pump pump1(T_supply=85.0, T_return=35.0);
  WarmingUp.HeatNetwork.Heat.Pump pump2(T_supply=85.0, T_return=35.0);
  WarmingUp.HeatNetwork.Heat.Pump pump3(T_supply=85.0, T_return=35.0);
  WarmingUp.HeatNetwork.Heat.Pump pump4(T_supply=85.0, T_return=35.0);
  WarmingUp.HeatNetwork.Heat.Pump pump5(T_supply=85.0, T_return=35.0);
  WarmingUp.HeatNetwork.Heat.Pump pump6(T_supply=85.0, T_return=35.0);
  WarmingUp.HeatNetwork.Heat.Pump pump12(T_supply=85.0, T_return=35.0);

equation
// Connect Model Elements

// Connect source 1 & 2 to their shared node
  connect(pipeS1_hot.HeatOut, nodeS12_hot.HeatConn[1]) ;
  connect(pipeS2_hot.HeatOut, nodeS12_hot.HeatConn[2]) ;
  connect(nodeS12_hot.HeatConn[3], pipeS1and2_hot.HeatIn) ;
  connect(pipeS1and2_cold.HeatOut, nodeS12_cold.HeatConn[1]) ;
  connect(nodeS12_cold.HeatConn[2], pipeS1_cold.HeatIn) ;
  connect(nodeS12_cold.HeatConn[3], pipeS2_cold.HeatIn) ;
// Connect Source 1
  connect(pipeS1_cold.HeatOut, pipepump1_cold.HeatIn) ;
  connect(pipepump1_cold.HeatOut, pump1.HeatIn) ;
  connect(pump1.HeatOut, Source1.HeatIn) ;
  connect(Source1.HeatOut, pipepump1_hot.HeatIn) ;
  connect(pipepump1_hot.HeatOut, pipeS1_hot.HeatIn) ;
// Connect source 2
  connect(pipeS2_cold.HeatOut, pipepump2_cold.HeatIn) ;
  connect(pipepump2_cold.HeatOut, pump2.HeatIn) ;
  connect(pump2.HeatOut, Source2.HeatIn) ;
  connect(Source2.HeatOut, pipepump2_hot.HeatIn) ;
  connect(pipepump2_hot.HeatOut, pipeS2_hot.HeatIn) ;
// Connect Node source 1&2 to demand H node
  connect(pipeS1and2_hot.HeatOut, nodeDH_hot.HeatConn[1]);
  connect(nodeDH_cold.HeatConn[3], pipeS1and2_cold.HeatIn);
// Connect Demand H to node
  connect(nodeDH_hot.HeatConn[3], pipeDH_hot.HeatIn);
  connect(pipeDH_hot.HeatOut, DemandH.HeatIn);
  connect(DemandH.HeatOut, pipeDH_cold.HeatIn);
  connect(pipeDH_cold.HeatOut, nodeDH_cold.HeatConn[1]);
// Connect pipe 4 in between demand H and DemandG
  connect(nodeDH_cold.HeatConn[4], pipe4_cold.HeatIn);
  connect(pipe4_cold.HeatOut, nodeDG_cold.HeatConn[1]);
  connect(nodeDG_hot.HeatConn[2], pipe4_hot.HeatIn);
  connect(pipe4_hot.HeatOut, nodeDH_hot.HeatConn[2]);
// Connect demand DemandG
  connect(nodeDG_hot.HeatConn[3], pipeDG_hot.HeatIn);
  connect(pipeDG_hot.HeatOut, DemandG.HeatIn);
  connect(DemandG.HeatOut, pipeDG_cold.HeatIn);
  connect(pipeDG_cold.HeatOut, nodeDG_cold.HeatConn[2]);
// Connect source 3
  connect(nodeDG_cold.HeatConn[3], pipeS3_cold.HeatIn);
  connect(pipeS3_cold.HeatOut, pipepump3_cold.HeatIn) ;
  connect(pipepump3_cold.HeatOut, pump3.HeatIn) ;
  connect(pump3.HeatOut, Source3.HeatIn) ;
  connect(Source3.HeatOut, pipepump3_hot.HeatIn) ;
  connect(pipepump3_hot.HeatOut, pipeS3_hot.HeatIn) ;
  connect(pipeS3_hot.HeatOut, nodeDG_hot.HeatConn[1]);
// connect pipe 3 in between DH and DF  
  connect(nodeDH_hot.HeatConn[4], pipe3_hot.HeatIn);
  connect(pipe3_hot.HeatOut, nodeDF_hot.HeatConn[1]);
  connect(nodeDF_cold.HeatConn[5], pipe3_cold.HeatIn);
  connect(pipe3_cold.HeatOut, nodeDH_cold.HeatConn[2]);
// Connect Demand F
  connect(nodeDF_hot.HeatConn[3], pipeDF_hot.HeatIn);
  connect(pipeDF_hot.HeatOut, DemandF.HeatIn);
  connect(DemandF.HeatOut, pipeDF_cold.HeatIn);
  connect(pipeDF_cold.HeatOut, nodeDF_cold.HeatConn[1]);
// Connect source 12
  connect(nodeDF_cold.HeatConn[6], pipe5_cold.HeatIn);
  connect(pipe5_cold.HeatOut, pipepump12_cold.HeatIn) ;
  connect(pipepump12_cold.HeatOut, pump12.HeatIn) ;
  connect(pump12.HeatOut, Source12.HeatIn) ;
  connect(Source12.HeatOut, pipepump12_hot.HeatIn) ;
  connect(pipepump12_hot.HeatOut, pipe5_hot.HeatIn) ;
  connect(pipe5_hot.HeatOut, nodeDF_hot.HeatConn[2]);
// Connect pipe 2
  connect(nodeDF_hot.HeatConn[4], pipe2_hot.HeatIn);
  connect(pipe2_hot.HeatOut, nodeDI_hot.HeatConn[1]);
  connect(nodeDI_cold.HeatConn[4], pipe2_cold.HeatIn);
  connect(pipe2_cold.HeatOut, nodeDF_cold.HeatConn[2]);
// Connect demand I
  connect(nodeDI_hot.HeatConn[2], pipeDI_hot.HeatIn);
  connect(pipeDI_hot.HeatOut, DemandI.HeatIn);
  connect(DemandI.HeatOut, pipeDI_cold.HeatIn);
  connect(pipeDI_cold.HeatOut, nodeDI_cold.HeatConn[1]);
// connect demand J
  connect(nodeDI_hot.HeatConn[3], pipe1_hot.HeatIn);
  connect(pipe1_hot.HeatOut, DemandJ.HeatIn);
  connect(DemandJ.HeatOut, pipe1_cold.HeatIn);
  connect(pipe1_cold.HeatOut, nodeDI_cold.HeatConn[2]);
// connect pipe 6a
  connect(nodeDF_hot.HeatConn[5], pipe6a_hot.HeatIn);
  connect(pipe6a_hot.HeatOut, node6_hot.HeatConn[1]);
  connect(node6_cold.HeatConn[2], pipe6a_cold.HeatIn);
  connect(pipe6a_cold.HeatOut, nodeDF_cold.HeatConn[3]);
//Connect pipe 6b
  connect(node6_cold.HeatConn[3], pipe6b_cold.HeatIn);
  connect(pipe6b_cold.HeatOut, nodeDE_cold.HeatConn[1]);
  connect(nodeDE_hot.HeatConn[2], pipe6b_hot.HeatIn);
  connect(pipe6b_hot.HeatOut, node6_hot.HeatConn[2]);
// Connect DemandE
  connect(nodeDE_hot.HeatConn[3], pipeDE_hot.HeatIn);
  connect(pipeDE_hot.HeatOut, DemandE.HeatIn);
  connect(DemandE.HeatOut, pipeDE_cold.HeatIn);
  connect(pipeDE_cold.HeatOut, nodeDE_cold.HeatConn[2]);
// Connect source 4
  connect(nodeDE_cold.HeatConn[3], pipeS4_cold.HeatIn);
  connect(pipeS4_cold.HeatOut, pipepump4_cold.HeatIn) ;
  connect(pipepump4_cold.HeatOut, pump4.HeatIn) ;
  connect(pump4.HeatOut, Source4.HeatIn) ;
  connect(Source4.HeatOut, pipepump4_hot.HeatIn) ;
  connect(pipepump4_hot.HeatOut, pipeS4_hot.HeatIn) ;
  connect(pipeS4_hot.HeatOut, nodeDE_hot.HeatConn[1]);
// connect pipe 7a
  connect(nodeDF_hot.HeatConn[6], pipe7a_hot.HeatIn);
  connect(pipe7a_hot.HeatOut, nodeS6_hot.HeatConn[1]);
  connect(nodeS6_cold.HeatConn[2], pipe7a_cold.HeatIn);
  connect(pipe7a_cold.HeatOut, nodeDF_cold.HeatConn[4]);
// connect source 6
  connect(nodeS6_cold.HeatConn[3], pipeS6_cold.HeatIn);
  connect(pipeS6_cold.HeatOut, pipepump6_cold.HeatIn) ;
  connect(pipepump6_cold.HeatOut, pump6.HeatIn) ;
  connect(pump6.HeatOut, Source6.HeatIn) ;
  connect(Source6.HeatOut, pipepump6_hot.HeatIn) ;
  connect(pipepump6_hot.HeatOut, pipeS6_hot.HeatIn) ;
  connect(pipeS6_hot.HeatOut, nodeS6_hot.HeatConn[2]);
// connect pipe 7b
  connect(nodeS6_hot.HeatConn[3], pipe7b_hot.HeatIn);
  connect(pipe7b_hot.HeatOut, nodeDB_hot.HeatConn[1]);
  connect(nodeDB_cold.HeatConn[4], pipe7b_cold.HeatIn);
  connect(pipe7b_cold.HeatOut, nodeS6_cold.HeatConn[1]);
// connect demand B
  connect(nodeDB_hot.HeatConn[3], pipeDB_hot.HeatIn);
  connect(pipeDB_hot.HeatOut, DemandB.HeatIn);
  connect(DemandB.HeatOut, pipeDB_cold.HeatIn);
  connect(pipeDB_cold.HeatOut, nodeDB_cold.HeatConn[1]);
// connect demand C
  connect(nodeDB_hot.HeatConn[4], pipeDC_hot.HeatIn);
  connect(pipeDC_hot.HeatOut, DemandC.HeatIn);
  connect(DemandC.HeatOut, pipeDC_cold.HeatIn);
  connect(pipeDC_cold.HeatOut, nodeDB_cold.HeatConn[2]);
// connect pipe 8a
  connect(nodeDB_hot.HeatConn[5], pipe8a_hot.HeatIn);
  connect(pipe8a_hot.HeatOut, node8_hot.HeatConn[1]);
  connect(node8_cold.HeatConn[2], pipe8a_cold.HeatIn);
  connect(pipe8a_cold.HeatOut, nodeDB_cold.HeatConn[3]);
// connect pipe R1
  connect(nodeDI_hot.HeatConn[4], pipeR1_hot.HeatIn);
  connect(pipeR1_hot.HeatOut, node8_hot.HeatConn[2]);
  connect(node8_cold.HeatConn[3], pipeR1_cold.HeatIn);
  connect(pipeR1_cold.HeatOut, nodeDI_cold.HeatConn[3]);
// connect Demand A
  connect(node8_hot.HeatConn[3], pipeDA_hot.HeatIn);
  connect(pipeDA_hot.HeatOut, DemandA.HeatIn);
  connect(DemandA.HeatOut, pipeDA_cold.HeatIn);
  connect(pipeDA_cold.HeatOut, node8_cold.HeatConn[1]);
// connect Demand 9a
  connect(nodeDB_cold.HeatConn[5], pipe9a_cold.HeatIn);
  connect(pipe9a_cold.HeatOut, node9_cold.HeatConn[1]);
  connect(node9_hot.HeatConn[3], pipe9a_hot.HeatIn);
  connect(pipe9a_hot.HeatOut, nodeDB_hot.HeatConn[2]);
// connect pipe R2
  connect(node6_hot.HeatConn[3], pipeR2_hot.HeatIn);
  connect(pipeR2_hot.HeatOut, node9_hot.HeatConn[1]);
  connect(node9_cold.HeatConn[2], pipeR2_cold.HeatIn);
  connect(pipeR2_cold.HeatOut, node6_cold.HeatConn[1]);
// connect pipe 9b
  connect(node9_cold.HeatConn[3], pipe9b_cold.HeatIn);
  connect(pipe9b_cold.HeatOut, nodeDD_cold.HeatConn[1]);
  connect(nodeDD_hot.HeatConn[2], pipe9b_hot.HeatIn);
  connect(pipe9b_hot.HeatOut, node9_hot.HeatConn[2]);
// connect DemandD
  connect(nodeDD_hot.HeatConn[3], pipeDD_hot.HeatIn);
  connect(pipeDD_hot.HeatOut, DemandD.HeatIn);
  connect(DemandD.HeatOut, pipeDD_cold.HeatIn);
  connect(pipeDD_cold.HeatOut, nodeDD_cold.HeatConn[2]);
// connect Source 5
  connect(nodeDD_cold.HeatConn[3], pipeS5_cold.HeatIn);
  connect(pipeS5_cold.HeatOut, pipepump5_cold.HeatIn) ;
  connect(pipepump5_cold.HeatOut, pump5.HeatIn) ;
  connect(pump5.HeatOut, Source5.HeatIn) ;
  connect(Source5.HeatOut, pipepump5_hot.HeatIn) ;
  connect(pipepump5_hot.HeatOut, pipeS5_hot.HeatIn) ;
  connect(pipeS5_hot.HeatOut, nodeDD_hot.HeatConn[1]);
end Westland_Heat;
