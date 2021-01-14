model Westland_QTH
  // Declare Model Elements

  parameter Real Q_nominal = 0.01;
  parameter Real Q_min = 0.0;
  parameter Real Q_max_long = 0.5;
  parameter Real Q_max = 0.25;

  parameter Real theta;
  parameter Real t_supply_max = 110.0;
  parameter Real t_supply_min = 10.0;
  parameter Real t_return_max = 110.0;
  parameter Real t_return_min = 10.0;

  parameter Real t_source1_min = 65.0;
  parameter Real t_source1_max = 85.0;

  parameter Real t_demand_min = 70;

  parameter Real t_supply_nom = 85.0;
  parameter Real t_return_nom = 35.0;

  //Heatsource min en max in [W]
  WarmingUp.HeatNetwork.QTH.Source Source1(Heat_source(min=0.0, max=12.5e6, nominal=1e7), theta = theta, QTHOut.T(min=t_source1_min, max=t_source1_max), Q_nominal=Q_nominal, T_supply=85.0, T_return=35.0);
  WarmingUp.HeatNetwork.QTH.Source Source2(Heat_source(min=0.0, max=17.5e6, nominal=1e7), theta = theta, QTHOut.T(min=t_source1_min, max=t_source1_max), Q_nominal=Q_nominal, T_supply=85.0, T_return=35.0);
  WarmingUp.HeatNetwork.QTH.Source Source3(Heat_source(min=0.0, max=15e6, nominal=1e7), theta = theta, QTHOut.T(min=t_source1_min, max=t_source1_max), Q_nominal=Q_nominal, T_supply=85.0, T_return=35.0);
  WarmingUp.HeatNetwork.QTH.Source Source12(Heat_source(min=0.0, max=40e6, nominal=1e7), theta = theta, QTHOut.T(min=t_source1_min, max=t_source1_max), Q_nominal=Q_nominal, T_supply=85.0, T_return=35.0);
  WarmingUp.HeatNetwork.QTH.Source Source4(Heat_source(min=0.0, max=17.5e6, nominal=1e7), theta = theta, QTHOut.T(min=t_source1_min, max=t_source1_max), Q_nominal=Q_nominal, T_supply=85.0, T_return=35.0);
  WarmingUp.HeatNetwork.QTH.Source Source6(Heat_source(min=0.0, max=40e6, nominal=1e7), theta = theta, QTHOut.T(min=t_source1_min, max=t_source1_max), Q_nominal=Q_nominal, T_supply=85.0, T_return=35.0);
  WarmingUp.HeatNetwork.QTH.Source Source5(Heat_source(min=0.0, max=12.5e6, nominal=1e7), theta = theta, QTHOut.T(min=t_source1_min, max=t_source1_max), Q_nominal=Q_nominal, T_supply=85.0, T_return=35.0);

  WarmingUp.HeatNetwork.QTH.Pipe pipeS1_cold(disconnectable=false, has_control_valve=false, length=5., diameter=0.2, temperature=35.0, T_supply=85.0, T_return=35.0, T_g=10.0, Q(min=Q_min, max=Q_max, nominal=Q_nominal), QTHIn.T(min=t_return_min, max=t_return_max), QTHOut.T(min=t_return_min, max=t_return_max), sign_dT=-1.0);
  WarmingUp.HeatNetwork.QTH.Pipe pipeS2_cold(disconnectable=false, has_control_valve=false, length=5., diameter=0.2, temperature=35.0, T_supply=85.0, T_return=35.0, T_g=10.0, Q(min=Q_min, max=Q_max, nominal=Q_nominal), QTHIn.T(min=t_return_min, max=t_return_max), QTHOut.T(min=t_return_min, max=t_return_max), sign_dT=-1.0);
  WarmingUp.HeatNetwork.QTH.Pipe pipeS1and2_cold(disconnectable=false, has_control_valve=false, length=5., diameter=0.2, temperature=35.0, T_supply=85.0, T_return=35.0, T_g=10.0, Q(min=Q_min, max=Q_max, nominal=Q_nominal), QTHIn.T(min=t_return_min, max=t_return_max), QTHOut.T(min=t_return_min, max=t_return_max), sign_dT=-1.0);
  WarmingUp.HeatNetwork.QTH.Pipe pipeDH_cold(disconnectable=false, has_control_valve=false, length=5., diameter=0.2, temperature=35.0, T_supply=85.0, T_return=35.0, T_g=10.0, Q(min=Q_min, max=Q_max, nominal=Q_nominal), QTHIn.T(min=t_return_min, max=t_return_max), QTHOut.T(min=t_return_min, max=t_return_max), sign_dT=-1.0);
  WarmingUp.HeatNetwork.QTH.Pipe pipeDG_cold(disconnectable=false, has_control_valve=false, length=5., diameter=0.2, temperature=35.0, T_supply=85.0, T_return=35.0, T_g=10.0, Q(min=Q_min, max=Q_max, nominal=Q_nominal), QTHIn.T(min=t_return_min, max=t_return_max), QTHOut.T(min=t_return_min, max=t_return_max), sign_dT=-1.0);
  WarmingUp.HeatNetwork.QTH.Pipe pipeDF_cold(disconnectable=false, has_control_valve=false, length=5., diameter=0.2, temperature=35.0, T_supply=85.0, T_return=35.0, T_g=10.0, Q(min=Q_min, max=Q_max, nominal=Q_nominal), QTHIn.T(min=t_return_min, max=t_return_max), QTHOut.T(min=t_return_min, max=t_return_max), sign_dT=-1.0);
  WarmingUp.HeatNetwork.QTH.Pipe pipeDI_cold(disconnectable=false, has_control_valve=false, length=5., diameter=0.2, temperature=35.0, T_supply=85.0, T_return=35.0, T_g=10.0, Q(min=Q_min, max=Q_max, nominal=Q_nominal), QTHIn.T(min=t_return_min, max=t_return_max), QTHOut.T(min=t_return_min, max=t_return_max), sign_dT=-1.0);
  WarmingUp.HeatNetwork.QTH.Pipe pipeDE_cold(disconnectable=false, has_control_valve=false, length=5., diameter=0.2, temperature=35.0, T_supply=85.0, T_return=35.0, T_g=10.0, Q(min=Q_min, max=Q_max, nominal=Q_nominal), QTHIn.T(min=t_return_min, max=t_return_max), QTHOut.T(min=t_return_min, max=t_return_max), sign_dT=-1.0);
  WarmingUp.HeatNetwork.QTH.Pipe pipeDB_cold(disconnectable=false, has_control_valve=false, length=5., diameter=0.2, temperature=35.0, T_supply=85.0, T_return=35.0, T_g=10.0, Q(min=Q_min, max=Q_max, nominal=Q_nominal), QTHIn.T(min=t_return_min, max=t_return_max), QTHOut.T(min=t_return_min, max=t_return_max), sign_dT=-1.0);
  WarmingUp.HeatNetwork.QTH.Pipe pipeDA_cold(disconnectable=false, has_control_valve=false, length=5., diameter=0.2, temperature=35.0, T_supply=85.0, T_return=35.0, T_g=10.0, Q(min=Q_min, max=Q_max, nominal=Q_nominal), QTHIn.T(min=t_return_min, max=t_return_max), QTHOut.T(min=t_return_min, max=t_return_max), sign_dT=-1.0);
  WarmingUp.HeatNetwork.QTH.Pipe pipeDC_cold(disconnectable=false, has_control_valve=false, length=5., diameter=0.2, temperature=35.0, T_supply=85.0, T_return=35.0, T_g=10.0, Q(min=Q_min, max=Q_max, nominal=Q_nominal), QTHIn.T(min=t_return_min, max=t_return_max), QTHOut.T(min=t_return_min, max=t_return_max), sign_dT=-1.0);
  WarmingUp.HeatNetwork.QTH.Pipe pipeDD_cold(disconnectable=false, has_control_valve=false, length=5., diameter=0.2, temperature=35.0, T_supply=85.0, T_return=35.0, T_g=10.0, Q(min=Q_min, max=Q_max, nominal=Q_nominal), QTHIn.T(min=t_return_min, max=t_return_max), QTHOut.T(min=t_return_min, max=t_return_max), sign_dT=-1.0);
  WarmingUp.HeatNetwork.QTH.Pipe pipeS5_cold(disconnectable=false, has_control_valve=false, length=5., diameter=0.2, temperature=35.0, T_supply=85.0, T_return=35.0, T_g=10.0, Q(min=Q_min, max=Q_max, nominal=Q_nominal), QTHIn.T(min=t_return_min, max=t_return_max), QTHOut.T(min=t_return_min, max=t_return_max), sign_dT=-1.0);
  WarmingUp.HeatNetwork.QTH.Pipe pipeS3_cold(disconnectable=false, has_control_valve=false, length=5., diameter=0.2, temperature=35.0, T_supply=85.0, T_return=35.0, T_g=10.0, Q(min=Q_min, max=Q_max, nominal=Q_nominal), QTHIn.T(min=t_return_min, max=t_return_max), QTHOut.T(min=t_return_min, max=t_return_max), sign_dT=-1.0);
  WarmingUp.HeatNetwork.QTH.Pipe pipeS4_cold(disconnectable=false, has_control_valve=false, length=5., diameter=0.2, temperature=35.0, T_supply=85.0, T_return=35.0, T_g=10.0, Q(min=Q_min, max=Q_max, nominal=Q_nominal), QTHIn.T(min=t_return_min, max=t_return_max), QTHOut.T(min=t_return_min, max=t_return_max), sign_dT=-1.0);
  WarmingUp.HeatNetwork.QTH.Pipe pipeS6_cold(disconnectable=false, has_control_valve=false, length=5., diameter=0.2, temperature=35.0, T_supply=85.0, T_return=35.0, T_g=10.0, Q(min=Q_min, max=Q_max, nominal=Q_nominal), QTHIn.T(min=t_return_min, max=t_return_max), QTHOut.T(min=t_return_min, max=t_return_max), sign_dT=-1.0);
  WarmingUp.HeatNetwork.QTH.Pipe pipe4_cold(disconnectable=false, has_control_valve=true, length=2000., diameter=0.25, temperature=35.0, T_supply=85.0, T_return=35.0, T_g=10.0, Q(min=-Q_max, max=Q_max, nominal=Q_nominal), QTHIn.T(min=t_return_min, max=t_return_max), QTHOut.T(min=t_return_min, max=t_return_max), sign_dT=-1.0);
  WarmingUp.HeatNetwork.QTH.Pipe pipe3_cold(disconnectable=false, has_control_valve=false, length=2500., diameter=0.3, temperature=35.0, T_supply=85.0, T_return=35.0, T_g=10.0, Q(min=Q_min, max=Q_max_long, nominal=Q_nominal), QTHIn.T(min=t_return_min, max=t_return_max), QTHOut.T(min=t_return_min, max=t_return_max), sign_dT=-1.0);
  WarmingUp.HeatNetwork.QTH.Pipe pipe5_cold(disconnectable=false, has_control_valve=false, length=2900., diameter=0.5, temperature=35.0, T_supply=85.0, T_return=35.0, T_g=10.0, Q(min=Q_min, max=Q_max_long, nominal=Q_nominal), QTHIn.T(min=t_return_min, max=t_return_max), QTHOut.T(min=t_return_min, max=t_return_max), sign_dT=-1.0);
  WarmingUp.HeatNetwork.QTH.Pipe pipe2_cold(disconnectable=false, has_control_valve=false, length=3900., diameter=0.4, temperature=35.0, T_supply=85.0, T_return=35.0, T_g=10.0, Q(min=Q_min, max=Q_max, nominal=Q_nominal), QTHIn.T(min=t_return_min, max=t_return_max), QTHOut.T(min=t_return_min, max=t_return_max), sign_dT=-1.0);
  WarmingUp.HeatNetwork.QTH.Pipe pipe1_cold(disconnectable=false, has_control_valve=false, length=2700., diameter=0.25, temperature=35.0, T_supply=85.0, T_return=35.0, T_g=10.0, Q(min=Q_min, max=Q_max, nominal=Q_nominal), QTHIn.T(min=t_return_min, max=t_return_max), QTHOut.T(min=t_return_min, max=t_return_max), sign_dT=-1.0);

  WarmingUp.HeatNetwork.QTH.Pipe pipepump1_cold(disconnectable=false, has_control_valve=false, length=1., diameter=0.25, temperature=35.0, T_supply=85.0, T_return=35.0, T_g = 10.0, Q(min=Q_min, max=Q_max, nominal=Q_nominal), QTHIn.T(min=t_return_min, max=t_return_max), QTHOut.T(min=t_return_min, max=t_return_max), sign_dT=-1.0);
  WarmingUp.HeatNetwork.QTH.Pipe pipepump2_cold(disconnectable=false, has_control_valve=false, length=1., diameter=0.25, temperature=35.0, T_supply=85.0, T_return=35.0, T_g = 10.0, Q(min=Q_min, max=Q_max, nominal=Q_nominal), QTHIn.T(min=t_return_min, max=t_return_max), QTHOut.T(min=t_return_min, max=t_return_max), sign_dT=-1.0);
  WarmingUp.HeatNetwork.QTH.Pipe pipepump3_cold(disconnectable=false, has_control_valve=false, length=1., diameter=0.25, temperature=35.0, T_supply=85.0, T_return=35.0, T_g = 10.0, Q(min=Q_min, max=Q_max, nominal=Q_nominal), QTHIn.T(min=t_return_min, max=t_return_max), QTHOut.T(min=t_return_min, max=t_return_max), sign_dT=-1.0);
  WarmingUp.HeatNetwork.QTH.Pipe pipepump4_cold(disconnectable=false, has_control_valve=false, length=1., diameter=0.25, temperature=35.0, T_supply=85.0, T_return=35.0, T_g = 10.0, Q(min=Q_min, max=Q_max, nominal=Q_nominal), QTHIn.T(min=t_return_min, max=t_return_max), QTHOut.T(min=t_return_min, max=t_return_max), sign_dT=-1.0);
  WarmingUp.HeatNetwork.QTH.Pipe pipepump5_cold(disconnectable=false, has_control_valve=false, length=1., diameter=0.25, temperature=35.0, T_supply=85.0, T_return=35.0, T_g = 10.0, Q(min=Q_min, max=Q_max, nominal=Q_nominal), QTHIn.T(min=t_return_min, max=t_return_max), QTHOut.T(min=t_return_min, max=t_return_max), sign_dT=-1.0);
  WarmingUp.HeatNetwork.QTH.Pipe pipepump6_cold(disconnectable=false, has_control_valve=false, length=1., diameter=0.25, temperature=35.0, T_supply=85.0, T_return=35.0, T_g = 10.0, Q(min=Q_min, max=Q_max, nominal=Q_nominal), QTHIn.T(min=t_return_min, max=t_return_max), QTHOut.T(min=t_return_min, max=t_return_max), sign_dT=-1.0);
  WarmingUp.HeatNetwork.QTH.Pipe pipepump12_cold(disconnectable=false, has_control_valve=false, length=1., diameter=0.25, temperature=35.0, T_supply=85.0, T_return=35.0, T_g = 10.0, Q(min=Q_min, max=Q_max, nominal=Q_nominal), QTHIn.T(min=t_return_min, max=t_return_max), QTHOut.T(min=t_return_min, max=t_return_max), sign_dT=-1.0);

  WarmingUp.HeatNetwork.QTH.Pipe pipe6a_hot(disconnectable=false, has_control_valve=true, length=2700., diameter=0.25, temperature=85.0, T_supply=85.0, T_return=35.0, T_g=10.0, Q(min=Q_min, max=Q_max, nominal=Q_nominal), QTHIn.T(min=t_supply_min, max=t_supply_max), QTHOut.T(min=t_supply_min, max=t_supply_max), sign_dT=1.0);
  WarmingUp.HeatNetwork.QTH.Pipe pipe6b_hot(disconnectable=false, has_control_valve=false, length=2700., diameter=0.25, temperature=85.0, T_supply=85.0, T_return=35.0, T_g=10.0, Q(min=Q_min, max=Q_max, nominal=Q_nominal), QTHIn.T(min=t_supply_min, max=t_supply_max), QTHOut.T(min=t_supply_min, max=t_supply_max), sign_dT=1.0);
  WarmingUp.HeatNetwork.QTH.Pipe pipe7a_hot(disconnectable=false, has_control_valve=true, length=2600., diameter=0.5, temperature=85.0, T_supply=85.0, T_return=35.0, T_g=10.0, Q(min=Q_min, max=Q_max_long, nominal=Q_nominal), QTHIn.T(min=t_supply_min, max=t_supply_max), QTHOut.T(min=t_supply_min, max=t_supply_max), sign_dT=1.0);
  WarmingUp.HeatNetwork.QTH.Pipe pipe7b_hot(disconnectable=false, has_control_valve=false, length=2600., diameter=0.5, temperature=85.0, T_supply=85.0, T_return=35.0, T_g=10.0, Q(min=Q_min, max=Q_max_long, nominal=Q_nominal), QTHIn.T(min=t_supply_min, max=t_supply_max), QTHOut.T(min=t_supply_min, max=t_supply_max), sign_dT=1.0);
  WarmingUp.HeatNetwork.QTH.Pipe pipe8a_hot(disconnectable=false, has_control_valve=false, length=3300., diameter=0.4, temperature=85.0, T_supply=85.0, T_return=35.0, T_g=10.0, Q(min=Q_min, max=Q_max, nominal=Q_nominal), QTHIn.T(min=t_supply_min, max=t_supply_max), QTHOut.T(min=t_supply_min, max=t_supply_max), sign_dT=1.0);
  WarmingUp.HeatNetwork.QTH.Pipe pipeR1_hot(disconnectable=false, has_control_valve=true, length=5200., diameter=0.3, temperature=85.0, T_supply=85.0, T_return=35.0, T_g=10.0, Q(min=Q_min, max=Q_max, nominal=Q_nominal), QTHIn.T(min=t_supply_min, max=t_supply_max), QTHOut.T(min=t_supply_min, max=t_supply_max), sign_dT=1.0);
  WarmingUp.HeatNetwork.QTH.Pipe pipe9a_hot(disconnectable=false, has_control_valve=false, length=3300., diameter=0.4, temperature=85.0, T_supply=85.0, T_return=35.0, T_g=10.0, Q(min=Q_min, max=Q_max, nominal=Q_nominal), QTHIn.T(min=t_supply_min, max=t_supply_max), QTHOut.T(min=t_supply_min, max=t_supply_max), sign_dT=1.0);
  WarmingUp.HeatNetwork.QTH.Pipe pipe9b_hot(disconnectable=false, has_control_valve=false, length=1600., diameter=0.4, temperature=85.0, T_supply=85.0, T_return=35.0, T_g=10.0, Q(min=Q_min, max=Q_max, nominal=Q_nominal), QTHIn.T(min=t_supply_min, max=t_supply_max), QTHOut.T(min=t_supply_min, max=t_supply_max), sign_dT=1.0);
  WarmingUp.HeatNetwork.QTH.Pipe pipeR2_hot(disconnectable=false, has_control_valve=true, length=4000., diameter=0.3, temperature=85.0, T_supply=85.0, T_return=35.0, T_g=10.0, Q(min=-Q_max, max=Q_max, nominal=Q_nominal), QTHIn.T(min=t_supply_min, max=t_supply_max), QTHOut.T(min=t_supply_min, max=t_supply_max), sign_dT=1.0);

  WarmingUp.HeatNetwork.QTH.Pipe pipeS1_hot(disconnectable=false, has_control_valve=false, length=5., diameter=0.2, temperature=85.0, T_supply=85.0, T_return=35.0, T_g=10.0, Q(min=Q_min, max=Q_max, nominal=Q_nominal), QTHIn.T(min=t_supply_min, max=t_supply_max), QTHOut.T(min=t_supply_min, max=t_supply_max), sign_dT=1.0);
  WarmingUp.HeatNetwork.QTH.Pipe pipeS2_hot(disconnectable=false, has_control_valve=false, length=5., diameter=0.2, temperature=85.0, T_supply=85.0, T_return=35.0, T_g=10.0, Q(min=Q_min, max=Q_max, nominal=Q_nominal), QTHIn.T(min=t_supply_min, max=t_supply_max), QTHOut.T(min=t_supply_min, max=t_supply_max), sign_dT=1.0);
  WarmingUp.HeatNetwork.QTH.Pipe pipeS1and2_hot(disconnectable=false, has_control_valve=false, length=5., diameter=0.2, temperature=85.0, T_supply=85.0, T_return=35.0, T_g=10.0, Q(min=Q_min, max=Q_max, nominal=Q_nominal), QTHIn.T(min=t_supply_min, max=t_supply_max), QTHOut.T(min=t_supply_min, max=t_supply_max), sign_dT=1.0);
  WarmingUp.HeatNetwork.QTH.Pipe pipeDH_hot(disconnectable=false, has_control_valve=false, length=5., diameter=0.2, temperature=85.0, T_supply=85.0, T_return=35.0, T_g=10.0, Q(min=Q_min, max=Q_max, nominal=Q_nominal), QTHIn.T(min=t_supply_min, max=t_supply_max), QTHOut.T(min=t_supply_min, max=t_supply_max), sign_dT=1.0);
  WarmingUp.HeatNetwork.QTH.Pipe pipeDG_hot(disconnectable=false, has_control_valve=false, length=5., diameter=0.2, temperature=85.0, T_supply=85.0, T_return=35.0, T_g=10.0, Q(min=Q_min, max=Q_max, nominal=Q_nominal), QTHIn.T(min=t_supply_min, max=t_supply_max), QTHOut.T(min=t_supply_min, max=t_supply_max), sign_dT=1.0);
  WarmingUp.HeatNetwork.QTH.Pipe pipeDF_hot(disconnectable=false, has_control_valve=false, length=5., diameter=0.2, temperature=85.0, T_supply=85.0, T_return=35.0, T_g=10.0, Q(min=Q_min, max=Q_max, nominal=Q_nominal), QTHIn.T(min=t_supply_min, max=t_supply_max), QTHOut.T(min=t_supply_min, max=t_supply_max), sign_dT=1.0);
  WarmingUp.HeatNetwork.QTH.Pipe pipeDE_hot(disconnectable=false, has_control_valve=false, length=5., diameter=0.2, temperature=85.0, T_supply=85.0, T_return=35.0, T_g=10.0, Q(min=Q_min, max=Q_max, nominal=Q_nominal), QTHIn.T(min=t_supply_min, max=t_supply_max), QTHOut.T(min=t_supply_min, max=t_supply_max), sign_dT=1.0);
  WarmingUp.HeatNetwork.QTH.Pipe pipeDB_hot(disconnectable=false, has_control_valve=false, length=5., diameter=0.2, temperature=85.0, T_supply=85.0, T_return=35.0, T_g=10.0, Q(min=Q_min, max=Q_max, nominal=Q_nominal), QTHIn.T(min=t_supply_min, max=t_supply_max), QTHOut.T(min=t_supply_min, max=t_supply_max), sign_dT=1.0);
  WarmingUp.HeatNetwork.QTH.Pipe pipeDA_hot(disconnectable=false, has_control_valve=false, length=5., diameter=0.2, temperature=85.0, T_supply=85.0, T_return=35.0, T_g=10.0, Q(min=Q_min, max=Q_max, nominal=Q_nominal), QTHIn.T(min=t_supply_min, max=t_supply_max), QTHOut.T(min=t_supply_min, max=t_supply_max), sign_dT=1.0);
  WarmingUp.HeatNetwork.QTH.Pipe pipeDC_hot(disconnectable=false, has_control_valve=false, length=5., diameter=0.2, temperature=85.0, T_supply=85.0, T_return=35.0, T_g=10.0, Q(min=Q_min, max=Q_max, nominal=Q_nominal), QTHIn.T(min=t_supply_min, max=t_supply_max), QTHOut.T(min=t_supply_min, max=t_supply_max), sign_dT=1.0);
  WarmingUp.HeatNetwork.QTH.Pipe pipeDI_hot(disconnectable=false, has_control_valve=false, length=5., diameter=0.2, temperature=85.0, T_supply=85.0, T_return=35.0, T_g=10.0, Q(min=Q_min, max=Q_max, nominal=Q_nominal), QTHIn.T(min=t_supply_min, max=t_supply_max), QTHOut.T(min=t_supply_min, max=t_supply_max), sign_dT=1.0);
  WarmingUp.HeatNetwork.QTH.Pipe pipeDD_hot(disconnectable=false, has_control_valve=false, length=5., diameter=0.2, temperature=85.0, T_supply=85.0, T_return=35.0, T_g=10.0, Q(min=Q_min, max=Q_max, nominal=Q_nominal), QTHIn.T(min=t_supply_min, max=t_supply_max), QTHOut.T(min=t_supply_min, max=t_supply_max), sign_dT=1.0);
  WarmingUp.HeatNetwork.QTH.Pipe pipeS5_hot(disconnectable=false, has_control_valve=false, length=5., diameter=0.2, temperature=85.0, T_supply=85.0, T_return=35.0, T_g=10.0, Q(min=Q_min, max=Q_max, nominal=Q_nominal), QTHIn.T(min=t_supply_min, max=t_supply_max), QTHOut.T(min=t_supply_min, max=t_supply_max), sign_dT=1.0);
  WarmingUp.HeatNetwork.QTH.Pipe pipeS3_hot(disconnectable=false, has_control_valve=false, length=5., diameter=0.2, temperature=85.0, T_supply=85.0, T_return=35.0, T_g=10.0, Q(min=Q_min, max=Q_max, nominal=Q_nominal), QTHIn.T(min=t_supply_min, max=t_supply_max), QTHOut.T(min=t_supply_min, max=t_supply_max), sign_dT=1.0);
  WarmingUp.HeatNetwork.QTH.Pipe pipeS4_hot(disconnectable=false, has_control_valve=false, length=5., diameter=0.2, temperature=85.0, T_supply=85.0, T_return=35.0, T_g=10.0, Q(min=Q_min, max=Q_max, nominal=Q_nominal), QTHIn.T(min=t_supply_min, max=t_supply_max), QTHOut.T(min=t_supply_min, max=t_supply_max), sign_dT=1.0);
  WarmingUp.HeatNetwork.QTH.Pipe pipeS6_hot(disconnectable=false, has_control_valve=false, length=5., diameter=0.2, temperature=85.0, T_supply=85.0, T_return=35.0, T_g=10.0, Q(min=Q_min, max=Q_max, nominal=Q_nominal), QTHIn.T(min=t_supply_min, max=t_supply_max), QTHOut.T(min=t_supply_min, max=t_supply_max), sign_dT=1.0);
  WarmingUp.HeatNetwork.QTH.Pipe pipe4_hot(disconnectable=false, has_control_valve=true, length=2000., diameter=0.25, temperature=85.0, T_supply=85.0, T_return=35.0, T_g=10.0, Q(min=-Q_max, max=Q_max, nominal=Q_nominal), QTHIn.T(min=t_supply_min, max=t_supply_max), QTHOut.T(min=t_supply_min, max=t_supply_max), sign_dT=1.0);
  WarmingUp.HeatNetwork.QTH.Pipe pipe3_hot(disconnectable=false, has_control_valve=false, length=2500., diameter=0.3, temperature=85.0, T_supply=85.0, T_return=35.0, T_g=10.0, Q(min=Q_min, max=Q_max_long, nominal=Q_nominal), QTHIn.T(min=t_supply_min, max=t_supply_max), QTHOut.T(min=t_supply_min, max=t_supply_max), sign_dT=1.0);
  WarmingUp.HeatNetwork.QTH.Pipe pipe5_hot(disconnectable=false, has_control_valve=false, length=2900., diameter=0.5, temperature=85.0, T_supply=85.0, T_return=35.0, T_g=10.0, Q(min=Q_min, max=Q_max_long, nominal=Q_nominal), QTHIn.T(min=t_supply_min, max=t_supply_max), QTHOut.T(min=t_supply_min, max=t_supply_max), sign_dT=1.0);
  WarmingUp.HeatNetwork.QTH.Pipe pipe2_hot(disconnectable=false, has_control_valve=false, length=2100., diameter=0.4, temperature=85.0, T_supply=85.0, T_return=35.0, T_g=10.0, Q(min=Q_min, max=Q_max, nominal=Q_nominal), QTHIn.T(min=t_supply_min, max=t_supply_max), QTHOut.T(min=t_supply_min, max=t_supply_max), sign_dT=1.0);
  WarmingUp.HeatNetwork.QTH.Pipe pipe1_hot(disconnectable=false, has_control_valve=false, length=1400., diameter=0.25, temperature=85.0, T_supply=85.0, T_return=35.0, T_g=10.0, Q(min=Q_min, max=Q_max, nominal=Q_nominal), QTHIn.T(min=t_supply_min, max=t_supply_max), QTHOut.T(min=t_supply_min, max=t_supply_max), sign_dT=1.0);

  WarmingUp.HeatNetwork.QTH.Pipe pipepump1_hot(disconnectable=false, has_control_valve=false, length=1., diameter=0.25, temperature=85.0, T_supply=85.0, T_return=35.0, T_g = 10.0, Q(min=Q_min, max=Q_max, nominal=Q_nominal), QTHIn.T(min=t_supply_min, max=t_supply_max), QTHOut.T(min=t_supply_min, max=t_supply_max), sign_dT=1.0);
  WarmingUp.HeatNetwork.QTH.Pipe pipepump2_hot(disconnectable=false, has_control_valve=false, length=1., diameter=0.25, temperature=85.0, T_supply=85.0, T_return=35.0, T_g = 10.0, Q(min=Q_min, max=Q_max, nominal=Q_nominal), QTHIn.T(min=t_supply_min, max=t_supply_max), QTHOut.T(min=t_supply_min, max=t_supply_max), sign_dT=1.0);
  WarmingUp.HeatNetwork.QTH.Pipe pipepump3_hot(disconnectable=false, has_control_valve=false, length=1., diameter=0.25, temperature=85.0, T_supply=85.0, T_return=35.0, T_g = 10.0, Q(min=Q_min, max=Q_max, nominal=Q_nominal), QTHIn.T(min=t_supply_min, max=t_supply_max), QTHOut.T(min=t_supply_min, max=t_supply_max), sign_dT=1.0);
  WarmingUp.HeatNetwork.QTH.Pipe pipepump4_hot(disconnectable=false, has_control_valve=false, length=1., diameter=0.25, temperature=85.0, T_supply=85.0, T_return=35.0, T_g = 10.0, Q(min=Q_min, max=Q_max, nominal=Q_nominal), QTHIn.T(min=t_supply_min, max=t_supply_max), QTHOut.T(min=t_supply_min, max=t_supply_max), sign_dT=1.0);
  WarmingUp.HeatNetwork.QTH.Pipe pipepump5_hot(disconnectable=false, has_control_valve=false, length=1., diameter=0.25, temperature=85.0, T_supply=85.0, T_return=35.0, T_g = 10.0, Q(min=Q_min, max=Q_max, nominal=Q_nominal), QTHIn.T(min=t_supply_min, max=t_supply_max), QTHOut.T(min=t_supply_min, max=t_supply_max), sign_dT=1.0);
  WarmingUp.HeatNetwork.QTH.Pipe pipepump6_hot(disconnectable=false, has_control_valve=false, length=1., diameter=0.25, temperature=85.0, T_supply=85.0, T_return=35.0, T_g = 10.0, Q(min=Q_min, max=Q_max, nominal=Q_nominal), QTHIn.T(min=t_supply_min, max=t_supply_max), QTHOut.T(min=t_supply_min, max=t_supply_max), sign_dT=1.0);
  WarmingUp.HeatNetwork.QTH.Pipe pipepump12_hot(disconnectable=false, has_control_valve=false, length=1., diameter=0.25, temperature=85.0, T_supply=85.0, T_return=35.0, T_g = 10.0, Q(min=Q_min, max=Q_max, nominal=Q_nominal), QTHIn.T(min=t_supply_min, max=t_supply_max), QTHOut.T(min=t_supply_min, max=t_supply_max), sign_dT=1.0);

  WarmingUp.HeatNetwork.QTH.Pipe pipe6a_cold(disconnectable=false, has_control_valve=true, length=2700., diameter=0.25, temperature=35.0, T_supply=85.0, T_return=35.0, T_g=10.0, Q(min=Q_min, max=Q_max, nominal=Q_nominal), QTHIn.T(min=t_return_min, max=t_return_max), QTHOut.T(min=t_return_min, max=t_return_max), sign_dT=-1.0);
  WarmingUp.HeatNetwork.QTH.Pipe pipe6b_cold(disconnectable=false, has_control_valve=false, length=2700., diameter=0.25, temperature=35.0, T_supply=85.0, T_return=35.0, T_g=10.0, Q(min=Q_min, max=Q_max, nominal=Q_nominal), QTHIn.T(min=t_return_min, max=t_return_max), QTHOut.T(min=t_return_min, max=t_return_max), sign_dT=-1.0);
  WarmingUp.HeatNetwork.QTH.Pipe pipe7a_cold(disconnectable=false, has_control_valve=true, length=2600., diameter=0.5, temperature=35.0, T_supply=85.0, T_return=35.0, T_g=10.0, Q(min=Q_min, max=Q_max_long, nominal=Q_nominal), QTHIn.T(min=t_return_min, max=t_return_max), QTHOut.T(min=t_return_min, max=t_return_max), sign_dT=-1.0);
  WarmingUp.HeatNetwork.QTH.Pipe pipe7b_cold(disconnectable=false, has_control_valve=false, length=2600., diameter=0.5, temperature=35.0, T_supply=85.0, T_return=35.0, T_g=10.0, Q(min=Q_min, max=Q_max_long, nominal=Q_nominal), QTHIn.T(min=t_return_min, max=t_return_max), QTHOut.T(min=t_return_min, max=t_return_max), sign_dT=-1.0);
  WarmingUp.HeatNetwork.QTH.Pipe pipe8a_cold(disconnectable=false, has_control_valve=false, length=3300., diameter=0.4, temperature=35.0, T_supply=85.0, T_return=35.0, T_g=10.0, Q(min=Q_min, max=Q_max, nominal=Q_nominal), QTHIn.T(min=t_return_min, max=t_return_max), QTHOut.T(min=t_return_min, max=t_return_max), sign_dT=-1.0);
  WarmingUp.HeatNetwork.QTH.Pipe pipeR1_cold(disconnectable=false, has_control_valve=true, length=5200., diameter=0.3, temperature=35.0, T_supply=85.0, T_return=35.0, T_g=10.0, Q(min=Q_min, max=Q_max, nominal=Q_nominal), QTHIn.T(min=t_return_min, max=t_return_max), QTHOut.T(min=t_return_min, max=t_return_max), sign_dT=-1.0);
  WarmingUp.HeatNetwork.QTH.Pipe pipe9a_cold(disconnectable=false, has_control_valve=false, length=3300., diameter=0.4, temperature=35.0, T_supply=85.0, T_return=35.0, T_g=10.0, Q(min=Q_min, max=Q_max, nominal=Q_nominal), QTHIn.T(min=t_return_min, max=t_return_max), QTHOut.T(min=t_return_min, max=t_return_max), sign_dT=-1.0);
  WarmingUp.HeatNetwork.QTH.Pipe pipe9b_cold(disconnectable=false, has_control_valve=false, length=1600., diameter=0.4, temperature=35.0, T_supply=85.0, T_return=35.0, T_g=10.0, Q(min=Q_min, max=Q_max, nominal=Q_nominal), QTHIn.T(min=t_return_min, max=t_return_max), QTHOut.T(min=t_return_min, max=t_return_max), sign_dT=-1.0);
  WarmingUp.HeatNetwork.QTH.Pipe pipeR2_cold(disconnectable=false, has_control_valve=true, length=4000., diameter=0.3, temperature=35.0, T_supply=85.0, T_return=35.0, T_g=10.0, Q(min=-Q_max, max=Q_max, nominal=Q_nominal), QTHIn.T(min=t_return_min, max=t_return_max), QTHOut.T(min=t_return_min, max=t_return_max), sign_dT=-1.0);

  WarmingUp.HeatNetwork.QTH.Demand DemandH(theta = theta, QTHIn.T(min=t_demand_min), Q_nominal=Q_nominal, T_supply=85.0, T_return=35.0);
  WarmingUp.HeatNetwork.QTH.Demand DemandG(theta = theta, QTHIn.T(min=t_demand_min), Q_nominal=Q_nominal, T_supply=85.0, T_return=35.0);
  WarmingUp.HeatNetwork.QTH.Demand DemandF(theta = theta, QTHIn.T(min=t_demand_min), Q_nominal=Q_nominal, T_supply=85.0, T_return=35.0);
  WarmingUp.HeatNetwork.QTH.Demand DemandI(theta = theta, QTHIn.T(min=t_demand_min), Q_nominal=Q_nominal, T_supply=85.0, T_return=35.0);
  WarmingUp.HeatNetwork.QTH.Demand DemandJ(theta = theta, QTHIn.T(min=t_demand_min), Q_nominal=Q_nominal, T_supply=85.0, T_return=35.0);
  WarmingUp.HeatNetwork.QTH.Demand DemandE(theta = theta, QTHIn.T(min=t_demand_min), Q_nominal=Q_nominal, T_supply=85.0, T_return=35.0);
  WarmingUp.HeatNetwork.QTH.Demand DemandB(theta = theta, QTHIn.T(min=t_demand_min), Q_nominal=Q_nominal, T_supply=85.0, T_return=35.0);
  WarmingUp.HeatNetwork.QTH.Demand DemandC(theta = theta, QTHIn.T(min=t_demand_min), Q_nominal=Q_nominal, T_supply=85.0, T_return=35.0);
  WarmingUp.HeatNetwork.QTH.Demand DemandA(theta = theta, QTHIn.T(min=t_demand_min), Q_nominal=Q_nominal, T_supply=85.0, T_return=35.0);
  WarmingUp.HeatNetwork.QTH.Demand DemandD(theta = theta, QTHIn.T(min=t_demand_min), Q_nominal=Q_nominal, T_supply=85.0, T_return=35.0);

  WarmingUp.HeatNetwork.QTH.Node nodeS12_hot(n=3, temperature=85.0); 
  WarmingUp.HeatNetwork.QTH.Node nodeS12_cold(n=3, temperature=35.0);
  WarmingUp.HeatNetwork.QTH.Node nodeDH_hot(n=4, temperature=85.0); 
  WarmingUp.HeatNetwork.QTH.Node nodeDH_cold(n=4, temperature=35.0);
  WarmingUp.HeatNetwork.QTH.Node nodeDG_hot(n=3, temperature=85.0); 
  WarmingUp.HeatNetwork.QTH.Node nodeDG_cold(n=3, temperature=35.0);
  WarmingUp.HeatNetwork.QTH.Node nodeDF_hot(n=6, temperature=85.0); 
  WarmingUp.HeatNetwork.QTH.Node nodeDF_cold(n=6, temperature=35.0);
  WarmingUp.HeatNetwork.QTH.Node nodeDI_hot(n=4, temperature=85.0); 
  WarmingUp.HeatNetwork.QTH.Node nodeDI_cold(n=4, temperature=35.0);
  WarmingUp.HeatNetwork.QTH.Node node6_hot(n=3, temperature=85.0); 
  WarmingUp.HeatNetwork.QTH.Node node6_cold(n=3, temperature=35.0);
  WarmingUp.HeatNetwork.QTH.Node nodeDE_hot(n=3, temperature=85.0); 
  WarmingUp.HeatNetwork.QTH.Node nodeDE_cold(n=3, temperature=35.0);
  WarmingUp.HeatNetwork.QTH.Node nodeDB_hot(n=5, temperature=85.0); 
  WarmingUp.HeatNetwork.QTH.Node nodeDB_cold(n=5, temperature=35.0);
  WarmingUp.HeatNetwork.QTH.Node nodeS6_hot(n=3, temperature=85.0); 
  WarmingUp.HeatNetwork.QTH.Node nodeS6_cold(n=3, temperature=35.0);
  WarmingUp.HeatNetwork.QTH.Node node8_hot(n=3, temperature=85.0); 
  WarmingUp.HeatNetwork.QTH.Node node8_cold(n=3, temperature=35.0);
  WarmingUp.HeatNetwork.QTH.Node node9_hot(n=3, temperature=85.0); 
  WarmingUp.HeatNetwork.QTH.Node node9_cold(n=3, temperature=35.0);
  WarmingUp.HeatNetwork.QTH.Node nodeDD_hot(n=3, temperature=85.0); 
  WarmingUp.HeatNetwork.QTH.Node nodeDD_cold(n=3, temperature=35.0);

  WarmingUp.HeatNetwork.QTH.Pump pump1(Q(min=0.0, max=0.5, nominal=Q_nominal), dH(min=0.0, max=1000.0), H(min=0.0, max=0.0));
  WarmingUp.HeatNetwork.QTH.Pump pump2(Q(min=0.0, max=0.5, nominal=Q_nominal), dH(min=0.0, max=1000.0));
  WarmingUp.HeatNetwork.QTH.Pump pump3(Q(min=0.0, max=0.5, nominal=Q_nominal), dH(min=0.0, max=1000.0));
  WarmingUp.HeatNetwork.QTH.Pump pump4(Q(min=0.0, max=0.5, nominal=Q_nominal), dH(min=0.0, max=1000.0));
  WarmingUp.HeatNetwork.QTH.Pump pump5(Q(min=0.0, max=0.5, nominal=Q_nominal), dH(min=0.0, max=1000.0));
  WarmingUp.HeatNetwork.QTH.Pump pump6(Q(min=0.0, max=0.5, nominal=Q_nominal), dH(min=0.0, max=1000.0));
  WarmingUp.HeatNetwork.QTH.Pump pump12(Q(min=0.0, max=0.5, nominal=Q_nominal), dH(min=0.0, max=1000.0));
equation
// Connect Model Elements

// Connect source 1 & 2 to their shared node
  connect(pipeS1_hot.QTHOut, nodeS12_hot.QTHConn[1]) ;
  connect(pipeS2_hot.QTHOut, nodeS12_hot.QTHConn[2]) ;
  connect(nodeS12_hot.QTHConn[3], pipeS1and2_hot.QTHIn) ;
  connect(pipeS1and2_cold.QTHOut, nodeS12_cold.QTHConn[1]) ;
  connect(nodeS12_cold.QTHConn[2], pipeS1_cold.QTHIn) ;
  connect(nodeS12_cold.QTHConn[3], pipeS2_cold.QTHIn) ;
// Connect Source 1
  connect(pipeS1_cold.QTHOut, pipepump1_cold.QTHIn) ;
  connect(pipepump1_cold.QTHOut, pump1.QTHIn) ;
  connect(pump1.QTHOut, Source1.QTHIn) ;
  connect(Source1.QTHOut, pipepump1_hot.QTHIn) ;
  connect(pipepump1_hot.QTHOut, pipeS1_hot.QTHIn) ;
// Connect source 2
  connect(pipeS2_cold.QTHOut, pipepump2_cold.QTHIn) ;
  connect(pipepump2_cold.QTHOut, pump2.QTHIn) ;
  connect(pump2.QTHOut, Source2.QTHIn) ;
  connect(Source2.QTHOut, pipepump2_hot.QTHIn) ;
  connect(pipepump2_hot.QTHOut, pipeS2_hot.QTHIn) ;
// Connect Node source 1&2 to demand H node
  connect(pipeS1and2_hot.QTHOut, nodeDH_hot.QTHConn[1]);
  connect(nodeDH_cold.QTHConn[3], pipeS1and2_cold.QTHIn);
// Connect Demand H to node
  connect(nodeDH_hot.QTHConn[3], pipeDH_hot.QTHIn);
  connect(pipeDH_hot.QTHOut, DemandH.QTHIn);
  connect(DemandH.QTHOut, pipeDH_cold.QTHIn);
  connect(pipeDH_cold.QTHOut, nodeDH_cold.QTHConn[1]);
// Connect pipe 4 in between demand H and DemandG
  connect(nodeDH_cold.QTHConn[4], pipe4_cold.QTHIn);
  connect(pipe4_cold.QTHOut, nodeDG_cold.QTHConn[1]);
  connect(nodeDG_hot.QTHConn[2], pipe4_hot.QTHIn);
  connect(pipe4_hot.QTHOut, nodeDH_hot.QTHConn[2]);
// Connect demand DemandG
  connect(nodeDG_hot.QTHConn[3], pipeDG_hot.QTHIn);
  connect(pipeDG_hot.QTHOut, DemandG.QTHIn);
  connect(DemandG.QTHOut, pipeDG_cold.QTHIn);
  connect(pipeDG_cold.QTHOut, nodeDG_cold.QTHConn[2]);
// Connect source 3
  connect(nodeDG_cold.QTHConn[3], pipeS3_cold.QTHIn);
  connect(pipeS3_cold.QTHOut, pipepump3_cold.QTHIn) ;
  connect(pipepump3_cold.QTHOut, pump3.QTHIn) ;
  connect(pump3.QTHOut, Source3.QTHIn) ;
  connect(Source3.QTHOut, pipepump3_hot.QTHIn) ;
  connect(pipepump3_hot.QTHOut, pipeS3_hot.QTHIn) ;
  connect(pipeS3_hot.QTHOut, nodeDG_hot.QTHConn[1]);
// connect pipe 3 in between DH and DF  
  connect(nodeDH_hot.QTHConn[4], pipe3_hot.QTHIn);
  connect(pipe3_hot.QTHOut, nodeDF_hot.QTHConn[1]);
  connect(nodeDF_cold.QTHConn[5], pipe3_cold.QTHIn);
  connect(pipe3_cold.QTHOut, nodeDH_cold.QTHConn[2]);
// Connect Demand F
  connect(nodeDF_hot.QTHConn[3], pipeDF_hot.QTHIn);
  connect(pipeDF_hot.QTHOut, DemandF.QTHIn);
  connect(DemandF.QTHOut, pipeDF_cold.QTHIn);
  connect(pipeDF_cold.QTHOut, nodeDF_cold.QTHConn[1]);
// Connect source 12
  connect(nodeDF_cold.QTHConn[6], pipe5_cold.QTHIn);
  connect(pipe5_cold.QTHOut, pipepump12_cold.QTHIn) ;
  connect(pipepump12_cold.QTHOut, pump12.QTHIn) ;
  connect(pump12.QTHOut, Source12.QTHIn) ;
  connect(Source12.QTHOut, pipepump12_hot.QTHIn) ;
  connect(pipepump12_hot.QTHOut, pipe5_hot.QTHIn) ;
  connect(pipe5_hot.QTHOut, nodeDF_hot.QTHConn[2]);
// Connect pipe 2
  connect(nodeDF_hot.QTHConn[4], pipe2_hot.QTHIn);
  connect(pipe2_hot.QTHOut, nodeDI_hot.QTHConn[1]);
  connect(nodeDI_cold.QTHConn[4], pipe2_cold.QTHIn);
  connect(pipe2_cold.QTHOut, nodeDF_cold.QTHConn[2]);
// Connect demand I
  connect(nodeDI_hot.QTHConn[2], pipeDI_hot.QTHIn);
  connect(pipeDI_hot.QTHOut, DemandI.QTHIn);
  connect(DemandI.QTHOut, pipeDI_cold.QTHIn);
  connect(pipeDI_cold.QTHOut, nodeDI_cold.QTHConn[1]);
// connect demand J
  connect(nodeDI_hot.QTHConn[3], pipe1_hot.QTHIn);
  connect(pipe1_hot.QTHOut, DemandJ.QTHIn);
  connect(DemandJ.QTHOut, pipe1_cold.QTHIn);
  connect(pipe1_cold.QTHOut, nodeDI_cold.QTHConn[2]);
// connect pipe 6a
  connect(nodeDF_hot.QTHConn[5], pipe6a_hot.QTHIn);
  connect(pipe6a_hot.QTHOut, node6_hot.QTHConn[1]);
  connect(node6_cold.QTHConn[2], pipe6a_cold.QTHIn);
  connect(pipe6a_cold.QTHOut, nodeDF_cold.QTHConn[3]);
//Connect pipe 6b
  connect(node6_cold.QTHConn[3], pipe6b_cold.QTHIn);
  connect(pipe6b_cold.QTHOut, nodeDE_cold.QTHConn[1]);
  connect(nodeDE_hot.QTHConn[2], pipe6b_hot.QTHIn);
  connect(pipe6b_hot.QTHOut, node6_hot.QTHConn[2]);
// Connect DemandE
  connect(nodeDE_hot.QTHConn[3], pipeDE_hot.QTHIn);
  connect(pipeDE_hot.QTHOut, DemandE.QTHIn);
  connect(DemandE.QTHOut, pipeDE_cold.QTHIn);
  connect(pipeDE_cold.QTHOut, nodeDE_cold.QTHConn[2]);
// Connect source 4
  connect(nodeDE_cold.QTHConn[3], pipeS4_cold.QTHIn);
  connect(pipeS4_cold.QTHOut, pipepump4_cold.QTHIn) ;
  connect(pipepump4_cold.QTHOut, pump4.QTHIn) ;
  connect(pump4.QTHOut, Source4.QTHIn) ;
  connect(Source4.QTHOut, pipepump4_hot.QTHIn) ;
  connect(pipepump4_hot.QTHOut, pipeS4_hot.QTHIn) ;
  connect(pipeS4_hot.QTHOut, nodeDE_hot.QTHConn[1]);
// connect pipe 7a
  connect(nodeDF_hot.QTHConn[6], pipe7a_hot.QTHIn);
  connect(pipe7a_hot.QTHOut, nodeS6_hot.QTHConn[1]);
  connect(nodeS6_cold.QTHConn[2], pipe7a_cold.QTHIn);
  connect(pipe7a_cold.QTHOut, nodeDF_cold.QTHConn[4]);
// connect source 6
  connect(nodeS6_cold.QTHConn[3], pipeS6_cold.QTHIn);
  connect(pipeS6_cold.QTHOut, pipepump6_cold.QTHIn) ;
  connect(pipepump6_cold.QTHOut, pump6.QTHIn) ;
  connect(pump6.QTHOut, Source6.QTHIn) ;
  connect(Source6.QTHOut, pipepump6_hot.QTHIn) ;
  connect(pipepump6_hot.QTHOut, pipeS6_hot.QTHIn) ;
  connect(pipeS6_hot.QTHOut, nodeS6_hot.QTHConn[2]);
// connect pipe 7b
  connect(nodeS6_hot.QTHConn[3], pipe7b_hot.QTHIn);
  connect(pipe7b_hot.QTHOut, nodeDB_hot.QTHConn[1]);
  connect(nodeDB_cold.QTHConn[4], pipe7b_cold.QTHIn);
  connect(pipe7b_cold.QTHOut, nodeS6_cold.QTHConn[1]);
// connect demand B
  connect(nodeDB_hot.QTHConn[3], pipeDB_hot.QTHIn);
  connect(pipeDB_hot.QTHOut, DemandB.QTHIn);
  connect(DemandB.QTHOut, pipeDB_cold.QTHIn);
  connect(pipeDB_cold.QTHOut, nodeDB_cold.QTHConn[1]);
// connect demand C
  connect(nodeDB_hot.QTHConn[4], pipeDC_hot.QTHIn);
  connect(pipeDC_hot.QTHOut, DemandC.QTHIn);
  connect(DemandC.QTHOut, pipeDC_cold.QTHIn);
  connect(pipeDC_cold.QTHOut, nodeDB_cold.QTHConn[2]);
// connect pipe 8a
  connect(nodeDB_hot.QTHConn[5], pipe8a_hot.QTHIn);
  connect(pipe8a_hot.QTHOut, node8_hot.QTHConn[1]);
  connect(node8_cold.QTHConn[2], pipe8a_cold.QTHIn);
  connect(pipe8a_cold.QTHOut, nodeDB_cold.QTHConn[3]);
// connect pipe R1
  connect(nodeDI_hot.QTHConn[4], pipeR1_hot.QTHIn);
  connect(pipeR1_hot.QTHOut, node8_hot.QTHConn[2]);
  connect(node8_cold.QTHConn[3], pipeR1_cold.QTHIn);
  connect(pipeR1_cold.QTHOut, nodeDI_cold.QTHConn[3]);
// connect Demand A
  connect(node8_hot.QTHConn[3], pipeDA_hot.QTHIn);
  connect(pipeDA_hot.QTHOut, DemandA.QTHIn);
  connect(DemandA.QTHOut, pipeDA_cold.QTHIn);
  connect(pipeDA_cold.QTHOut, node8_cold.QTHConn[1]);
// connect Demand 9a
  connect(nodeDB_cold.QTHConn[5], pipe9a_cold.QTHIn);
  connect(pipe9a_cold.QTHOut, node9_cold.QTHConn[1]);
  connect(node9_hot.QTHConn[3], pipe9a_hot.QTHIn);
  connect(pipe9a_hot.QTHOut, nodeDB_hot.QTHConn[2]);
// connect pipe R2
  connect(node6_hot.QTHConn[3], pipeR2_hot.QTHIn);
  connect(pipeR2_hot.QTHOut, node9_hot.QTHConn[1]);
  connect(node9_cold.QTHConn[2], pipeR2_cold.QTHIn);
  connect(pipeR2_cold.QTHOut, node6_cold.QTHConn[1]);
// connect pipe 9b
  connect(node9_cold.QTHConn[3], pipe9b_cold.QTHIn);
  connect(pipe9b_cold.QTHOut, nodeDD_cold.QTHConn[1]);
  connect(nodeDD_hot.QTHConn[2], pipe9b_hot.QTHIn);
  connect(pipe9b_hot.QTHOut, node9_hot.QTHConn[2]);
// connect DemandD
  connect(nodeDD_hot.QTHConn[3], pipeDD_hot.QTHIn);
  connect(pipeDD_hot.QTHOut, DemandD.QTHIn);
  connect(DemandD.QTHOut, pipeDD_cold.QTHIn);
  connect(pipeDD_cold.QTHOut, nodeDD_cold.QTHConn[2]);
// connect Source 5
  connect(nodeDD_cold.QTHConn[3], pipeS5_cold.QTHIn);
  connect(pipeS5_cold.QTHOut, pipepump5_cold.QTHIn) ;
  connect(pipepump5_cold.QTHOut, pump5.QTHIn) ;
  connect(pump5.QTHOut, Source5.QTHIn) ;
  connect(Source5.QTHOut, pipepump5_hot.QTHIn) ;
  connect(pipepump5_hot.QTHOut, pipeS5_hot.QTHIn) ;
  connect(pipeS5_hot.QTHOut, nodeDD_hot.QTHConn[1]);
end Westland_QTH;
