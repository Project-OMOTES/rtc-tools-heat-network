model Example_QTH
  // Declare Model Elements

  parameter Real Q_nominal = 0.001;
  parameter Real theta;
  parameter Real t_supply_max = 110.0;
  parameter Real t_supply_min = 10.0;
  parameter Real t_return_max = 110.0;
  parameter Real t_return_min = 10.0;

  parameter Real t_source1_min = 65.0;
  parameter Real t_source1_max = 85.0;
  parameter Real t_source2_min = 65.0;
  parameter Real t_source2_max = 90.0;

  parameter Real t_demand_min = 70;

  parameter Real t_supply_nom = 75.0;
  parameter Real t_return_nom = 45.0;

  parameter Real init_V_hot_tank = 0.0;

  //Heatsource min en max in [W]
  WarmingUp.HeatNetwork.QTH.Source source1(Heat_source(min=0.0, max=1.5e6, nominal=1e6), theta = theta, QTHOut.T(min=t_source1_min, max=t_source1_max), Q_nominal=Q_nominal, T_supply=75.0, T_return=45.0);
  WarmingUp.HeatNetwork.QTH.Source source2(Heat_source(min=0.0, max=1.5e7, nominal=1e6), theta = theta, QTHOut.T(min=t_source2_min, max=t_source2_max), Q_nominal=Q_nominal, T_supply=75.0, T_return=45.0);

  WarmingUp.HeatNetwork.QTH.Pipe pipe1a_cold(length = 170.365, diameter = 0.15, temperature=45.0, T_supply=75.0, T_return=45.0, Q(min=0.0, nominal=Q_nominal), QTHIn.T(min=t_return_min, max=t_return_max), QTHOut.T(min=t_return_min, max=t_return_max));
  WarmingUp.HeatNetwork.QTH.Pipe pipe1b_cold(length = 309.635, diameter = 0.15, temperature=45.0, T_supply=75.0, T_return=45.0, Q(min=0.0, nominal=Q_nominal), QTHIn.T(min=t_return_min, max=t_return_max), QTHOut.T(min=t_return_min, max=t_return_max));
  WarmingUp.HeatNetwork.QTH.Pipe pipe4a_cold(length = 5, diameter = 0.15, temperature=45.0, T_supply=75.0, T_return=45.0, Q(min=0.0, nominal=Q_nominal), QTHIn.T(min=t_return_min, max=t_return_max), QTHOut.T(min=t_return_min, max=t_return_max));
  WarmingUp.HeatNetwork.QTH.Pipe pipe4b_cold(length = 15, diameter = 0.15, temperature=45.0, T_supply=75.0, T_return=45.0, Q(min=0.0, nominal=Q_nominal), QTHIn.T(min=t_return_min, max=t_return_max), QTHOut.T(min=t_return_min, max=t_return_max));
  WarmingUp.HeatNetwork.QTH.Pipe pipe5_cold(length = 126, diameter = 0.15, temperature=45.0, T_supply=75.0, T_return=45.0, Q(min=0.0, nominal=Q_nominal), QTHIn.T(min=t_return_min, max=t_return_max), QTHOut.T(min=t_return_min, max=t_return_max));
  WarmingUp.HeatNetwork.QTH.Pipe pipe7_cold(length = 60, diameter = 0.15, temperature=45.0, T_supply=75.0, T_return=45.0, Q(min=0.0, nominal=Q_nominal), QTHIn.T(min=t_return_min, max=t_return_max), QTHOut.T(min=t_return_min, max=t_return_max));
  WarmingUp.HeatNetwork.QTH.Pipe pipe9_cold(length = 70, diameter = 0.15, temperature=45.0, T_supply=75.0, T_return=45.0, Q(min=0.0, nominal=Q_nominal), QTHIn.T(min=t_return_min, max=t_return_max), QTHOut.T(min=t_return_min, max=t_return_max));
  WarmingUp.HeatNetwork.QTH.Pipe pipe15_cold(length = 129, diameter = 0.15, temperature=45.0, T_supply=75.0, T_return=45.0, Q(min=0.0, nominal=Q_nominal), QTHIn.T(min=t_return_min, max=t_return_max), QTHOut.T(min=t_return_min, max=t_return_max));
  WarmingUp.HeatNetwork.QTH.Pipe pipe25_cold(length = 150, diameter = 0.15, temperature=45.0, T_supply=75.0, T_return=45.0, Q(min=0.0, nominal=Q_nominal), QTHIn.T(min=t_return_min, max=t_return_max), QTHOut.T(min=t_return_min, max=t_return_max));
  WarmingUp.HeatNetwork.QTH.Pipe pipe26_cold(length = 30, diameter = 0.1, temperature=45.0, T_supply=75.0, T_return=45.0, Q(min=0.0, nominal=Q_nominal), QTHIn.T(min=t_return_min, max=t_return_max), QTHOut.T(min=t_return_min, max=t_return_max));
  WarmingUp.HeatNetwork.QTH.Pipe pipe27_cold(length = 55, diameter = 0.15, temperature=45.0, T_supply=75.0, T_return=45.0, Q(min=0.0, nominal=Q_nominal), QTHIn.T(min=t_return_min, max=t_return_max), QTHOut.T(min=t_return_min, max=t_return_max));
  WarmingUp.HeatNetwork.QTH.Pipe pipe29_cold(length = 134, diameter = 0.15, temperature=45.0, T_supply=75.0, T_return=45.0, Q(min=0.0, nominal=Q_nominal), QTHIn.T(min=t_return_min, max=t_return_max), QTHOut.T(min=t_return_min, max=t_return_max));
  WarmingUp.HeatNetwork.QTH.Pipe pipe30_cold(length = 60, diameter = 0.1, temperature=45.0, T_supply=75.0, T_return=45.0, Q(min=0.0, nominal=Q_nominal), QTHIn.T(min=t_return_min, max=t_return_max), QTHOut.T(min=t_return_min, max=t_return_max));
  WarmingUp.HeatNetwork.QTH.Pipe pipe31_cold(length = 60, diameter = 0.1, temperature=45.0, T_supply=75.0, T_return=45.0, Q(min=0.0, nominal=Q_nominal), QTHIn.T(min=t_return_min, max=t_return_max), QTHOut.T(min=t_return_min, max=t_return_max));
  WarmingUp.HeatNetwork.QTH.Pipe pipe32_cold(length = 50, diameter = 0.1, temperature=45.0, T_supply=75.0, T_return=45.0, Q(min=0.0, nominal=Q_nominal), QTHIn.T(min=t_return_min, max=t_return_max), QTHOut.T(min=t_return_min, max=t_return_max));
  WarmingUp.HeatNetwork.QTH.Pipe pipe52_cold(disconnectable = true, length = 10, diameter = 0.164, temperature=45.0, T_supply=75.0, T_return=45.0, Q(nominal=Q_nominal), QTHIn.T(min=t_return_min, max=t_return_max), QTHOut.T(min=t_return_min, max=t_return_max));

  WarmingUp.HeatNetwork.QTH.Demand demand7(theta = theta, QTHIn.T(min=t_demand_min), Q_nominal=Q_nominal, T_supply=75.0, T_return=45.0);
  WarmingUp.HeatNetwork.QTH.Demand demand91(theta = theta, QTHIn.T(min=t_demand_min), Q_nominal=Q_nominal, T_supply=75.0, T_return=45.0);
  WarmingUp.HeatNetwork.QTH.Demand demand92(theta = theta, QTHIn.T(min=t_demand_min), Q_nominal=Q_nominal, T_supply=75.0, T_return=45.0);

  WarmingUp.HeatNetwork.QTH.Buffer buffer1(Q_nominal=Q_nominal, height = 10, radius = 5, heat_transfer_coeff = 1.0, init_V_hot_tank=init_V_hot_tank, init_T_hot_tank=t_supply_nom, init_T_cold_tank=t_return_nom, T_supply=75.0, T_return=45.0);

  WarmingUp.HeatNetwork.QTH.Pipe pipe1a_hot(length = 170.365, diameter = 0.15, temperature=75.0, T_supply=75.0, T_return=45.0, Q(min=0.0, nominal=Q_nominal), QTHIn.T(min=t_supply_min, max=t_supply_max), QTHOut.T(min=t_supply_min, max=t_supply_max));
  WarmingUp.HeatNetwork.QTH.Pipe pipe1b_hot(length = 309.635, diameter = 0.15, temperature=75.0, T_supply=75.0, T_return=45.0, Q(min=0.0, nominal=Q_nominal), QTHIn.T(min=t_supply_min, max=t_supply_max), QTHOut.T(min=t_supply_min, max=t_supply_max));
  WarmingUp.HeatNetwork.QTH.Pipe pipe4a_hot(length = 5, diameter = 0.15, temperature=75.0, T_supply=75.0, T_return=45.0, Q(min=0.0, nominal=Q_nominal), QTHIn.T(min=t_supply_min, max=t_supply_max), QTHOut.T(min=t_supply_min, max=t_supply_max));
  WarmingUp.HeatNetwork.QTH.Pipe pipe4b_hot(length = 15, diameter = 0.15, temperature=75.0, T_supply=75.0, T_return=45.0, Q(min=0.0, nominal=Q_nominal), QTHIn.T(min=t_supply_min, max=t_supply_max), QTHOut.T(min=t_supply_min, max=t_supply_max));
  WarmingUp.HeatNetwork.QTH.Pipe pipe5_hot(length = 126, diameter = 0.15, temperature=75.0, T_supply=75.0, T_return=45.0, Q(min=0.0, nominal=Q_nominal), QTHIn.T(min=t_supply_min, max=t_supply_max), QTHOut.T(min=t_supply_min, max=t_supply_max));
  WarmingUp.HeatNetwork.QTH.Pipe pipe7_hot(length = 60, diameter = 0.15, temperature=75.0, T_supply=75.0, T_return=45.0, Q(min=0.0, nominal=Q_nominal), QTHIn.T(min=t_supply_min, max=t_supply_max), QTHOut.T(min=t_supply_min, max=t_supply_max));
  WarmingUp.HeatNetwork.QTH.Pipe pipe9_hot(length = 70, diameter = 0.15, temperature=75.0, T_supply=75.0, T_return=45.0, Q(min=0.0, nominal=Q_nominal), QTHIn.T(min=t_supply_min, max=t_supply_max), QTHOut.T(min=t_supply_min, max=t_supply_max));
  WarmingUp.HeatNetwork.QTH.Pipe pipe15_hot(length = 129, diameter = 0.15, temperature=75.0, T_supply=75.0, T_return=45.0, Q(min=0.0, nominal=Q_nominal), QTHIn.T(min=t_supply_min, max=t_supply_max), QTHOut.T(min=t_supply_min, max=t_supply_max));
  WarmingUp.HeatNetwork.QTH.Pipe pipe25_hot(length = 150, diameter = 0.15, temperature=75.0, T_supply=75.0, T_return=45.0, Q(min=0.0, nominal=Q_nominal), QTHIn.T(min=t_supply_min, max=t_supply_max), QTHOut.T(min=t_supply_min, max=t_supply_max));
  WarmingUp.HeatNetwork.QTH.Pipe pipe26_hot(length = 30, diameter = 0.1, temperature=75.0, T_supply=75.0, T_return=45.0, Q(min=0.0, nominal=Q_nominal), QTHIn.T(min=t_supply_min, max=t_supply_max), QTHOut.T(min=t_supply_min, max=t_supply_max));
  WarmingUp.HeatNetwork.QTH.Pipe pipe27_hot(length = 55, diameter = 0.15, temperature=75.0, T_supply=75.0, T_return=45.0, Q(min=0.0, nominal=Q_nominal), QTHIn.T(min=t_supply_min, max=t_supply_max), QTHOut.T(min=t_supply_min, max=t_supply_max));
  WarmingUp.HeatNetwork.QTH.Pipe pipe29_hot(length = 134, diameter = 0.15, temperature=75.0, T_supply=75.0, T_return=45.0, Q(min=0.0, nominal=Q_nominal), QTHIn.T(min=t_supply_min, max=t_supply_max), QTHOut.T(min=t_supply_min, max=t_supply_max));
  WarmingUp.HeatNetwork.QTH.Pipe pipe30_hot(length = 60, diameter = 0.1, temperature=75.0, T_supply=75.0, T_return=45.0, Q(min=0.0, nominal=Q_nominal), QTHIn.T(min=t_supply_min, max=t_supply_max), QTHOut.T(min=t_supply_min, max=t_supply_max));
  WarmingUp.HeatNetwork.QTH.Pipe pipe31_hot(length = 60, diameter = 0.1, temperature=75.0, T_supply=75.0, T_return=45.0, Q(min=0.0, nominal=Q_nominal), QTHIn.T(min=t_supply_min, max=t_supply_max), QTHOut.T(min=t_supply_min, max=t_supply_max));
  WarmingUp.HeatNetwork.QTH.Pipe pipe32_hot(length = 50, diameter = 0.1, temperature=75.0, T_supply=75.0, T_return=45.0, Q(min=0.0, nominal=Q_nominal), QTHIn.T(min=t_supply_min, max=t_supply_max), QTHOut.T(min=t_supply_min, max=t_supply_max));
  WarmingUp.HeatNetwork.QTH.Pipe pipe52_hot(disconnectable = true, length = 10, diameter = 0.164, temperature=75.0, T_supply=75.0, T_return=45.0, Q(nominal=Q_nominal), QTHIn.T(min=t_supply_min, max=t_supply_max), QTHOut.T(min=t_supply_min, max=t_supply_max));

  WarmingUp.HeatNetwork.QTH.Node nodeS2_hot(n=3, temperature=75.0);
  WarmingUp.HeatNetwork.QTH.Node nodeD7_hot(n=3, temperature=75.0);
  WarmingUp.HeatNetwork.QTH.Node nodeD92_hot(n=3, temperature=75.0);
  WarmingUp.HeatNetwork.QTH.Node nodeB1_hot(n=3, temperature=75.0);

  WarmingUp.HeatNetwork.QTH.Node nodeS2_cold(n=3, temperature=45.0);
  WarmingUp.HeatNetwork.QTH.Node nodeD7_cold(n=3, temperature=45.0);
  WarmingUp.HeatNetwork.QTH.Node nodeD92_cold(n=3, temperature=45.0);
  WarmingUp.HeatNetwork.QTH.Node nodeB1_cold(n=3, temperature=45.0);

  //Q in [m^3/s] and H in [m]
  WarmingUp.HeatNetwork.QTH.Pump pump1(Q(min=0.00002778, max=0.01111, nominal=Q_nominal), dH(min=0.2, max=20.0), H(min=0.0, max=0.0));
  WarmingUp.HeatNetwork.QTH.Pump pump2(Q(min=0.00002778, max=0.01111, nominal=Q_nominal), dH(min=0.2, max=20.0));

  // Define Input/Output Variables and set them equal to model variables.
  //Heatdemand min en max in [W]
  input Modelica.SIunits.Heat Heat_source1(fixed=false) = source1.Heat_source;
  input Modelica.SIunits.Heat Heat_source2(fixed=false) = source2.Heat_source;

  output Modelica.SIunits.Heat Heat_demand7_opt = demand7.Heat_demand;
  output Modelica.SIunits.Heat Heat_demand91_opt = demand91.Heat_demand;
  output Modelica.SIunits.Heat Heat_demand92_opt = demand92.Heat_demand;

equation
// Connect Model Elements

//Hot lines from source1 to demand91
  connect(source1.QTHOut, pipe1a_hot.QTHIn) ;
  connect(pipe1a_hot.QTHOut, pump1.QTHIn) ;
  connect(pump1.QTHOut, pipe1b_hot.QTHIn) ;
  connect(pipe1b_hot.QTHOut, nodeS2_hot.QTHConn[1]) ;
  connect(nodeS2_hot.QTHConn[2], pipe5_hot.QTHIn) ;
  connect(pipe5_hot.QTHOut, pipe7_hot.QTHIn) ;
  connect(pipe7_hot.QTHOut, pipe9_hot.QTHIn) ;
  connect(pipe9_hot.QTHOut, nodeB1_hot.QTHConn[1]) ;
  connect(nodeB1_hot.QTHConn[2], pipe15_hot.QTHIn) ;
  connect(pipe15_hot.QTHOut, pipe25_hot.QTHIn) ;
  connect(pipe25_hot.QTHOut, nodeD7_hot.QTHConn[1]) ;
  connect(nodeD7_hot.QTHConn[2], pipe27_hot.QTHIn) ;
  connect(pipe27_hot.QTHOut, pipe29_hot.QTHIn) ;
  connect(pipe29_hot.QTHOut, nodeD92_hot.QTHConn[1]) ;
  connect(nodeD92_hot.QTHConn[2], pipe31_hot.QTHIn) ;
  connect(pipe31_hot.QTHOut, pipe32_hot.QTHIn) ;
  connect(pipe32_hot.QTHOut, demand91.QTHIn) ;


//Cold lines from demand91 to source 1
  connect(demand91.QTHOut, pipe32_cold.QTHIn) ;
  connect(pipe32_cold.QTHOut, pipe31_cold.QTHIn) ;
  connect(pipe31_cold.QTHOut, nodeD92_cold.QTHConn[1]) ;
  connect(nodeD92_cold.QTHConn[2], pipe29_cold.QTHIn) ;
  connect(pipe29_cold.QTHOut, pipe27_cold.QTHIn) ;
  connect(pipe27_cold.QTHOut, nodeD7_cold.QTHConn[1]) ;
  connect(nodeD7_cold.QTHConn[2], pipe25_cold.QTHIn) ;
  connect(pipe25_cold.QTHOut, pipe15_cold.QTHIn) ;
  connect(pipe15_cold.QTHOut, nodeB1_cold.QTHConn[1]) ;
  connect(nodeB1_cold.QTHConn[2], pipe9_cold.QTHIn) ;
  connect(pipe9_cold.QTHOut, pipe7_cold.QTHIn) ;
  connect(pipe7_cold.QTHOut, pipe5_cold.QTHIn) ;
  connect(pipe5_cold.QTHOut, nodeS2_cold.QTHConn[1]) ;
  connect(nodeS2_cold.QTHConn[2], pipe1b_cold.QTHIn) ;
  connect(pipe1b_cold.QTHOut, pipe1a_cold.QTHIn) ;
  connect(pipe1a_cold.QTHOut, source1.QTHIn) ;


//Source2
  connect(source2.QTHOut, pipe4a_hot.QTHIn) ;
  connect(pipe4a_hot.QTHOut, pump2.QTHIn) ;
  connect(pump2.QTHOut, pipe4b_hot.QTHIn) ;
  connect(pipe4b_hot.QTHOut, nodeS2_hot.QTHConn[3]) ;
  connect(nodeS2_cold.QTHConn[3], pipe4a_cold.QTHIn) ;
  connect(pipe4a_cold.QTHOut, pipe4b_cold.QTHIn) ;
  connect(pipe4b_cold.QTHOut, source2.QTHIn) ;

//Demand7
  connect(nodeD7_hot.QTHConn[3], pipe26_hot.QTHIn) ;
  connect(pipe26_hot.QTHOut, demand7.QTHIn) ;
  connect(demand7.QTHOut, pipe26_cold.QTHIn) ;
  connect(pipe26_cold.QTHOut, nodeD7_cold.QTHConn[3]) ;

//Demand92
  connect(nodeD92_hot.QTHConn[3], pipe30_hot.QTHIn) ;
  connect(pipe30_hot.QTHOut, demand92.QTHIn) ;
  connect(demand92.QTHOut, pipe30_cold.QTHIn) ;
  connect(pipe30_cold.QTHOut, nodeD92_cold.QTHConn[3]) ;

  //Buffer1
  // Hot
  connect(nodeB1_hot.QTHConn[3], pipe52_hot.QTHIn) ;
  connect(pipe52_hot.QTHOut, buffer1.QTHIn) ;
  // Cold
  connect(buffer1.QTHOut, pipe52_cold.QTHIn) ;
  connect(pipe52_cold.QTHOut, nodeB1_cold.QTHConn[3]) ;

end Example_QTH;
