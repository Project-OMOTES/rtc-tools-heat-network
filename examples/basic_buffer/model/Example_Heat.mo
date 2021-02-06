model Example_Heat
  // Declare Model Elements

  parameter Real init_Heat = 0.0;

  parameter Real Q_nominal = 0.001;

  constant Real heat_nominal_cold_pipes = 45.0*4200*988*0.005;
  constant Real heat_nominal_hot_pipes = 75.0*4200*988*0.005;
  constant Real heat_max_abs = 0.01111*100.0*4200.0*988;

  // Set default temperatures
  parameter Real T_supply = 75.0;
  parameter Real T_return = 45.0;

  //Heatsource min en max in [W]
  WarmingUp.HeatNetwork.Heat.Source source1(Heat_source(min=0.0, max=1.5e6, nominal=1e6), Heat_in(nominal=heat_nominal_cold_pipes), Heat_out(nominal=heat_nominal_hot_pipes), T_supply=75.0, T_return=45.0, Q_nominal=Q_nominal);
  WarmingUp.HeatNetwork.Heat.Source source2(Heat_source(min=0.0, max=1.5e7, nominal=1e6), Heat_in(nominal=heat_nominal_cold_pipes), Heat_out(nominal=heat_nominal_hot_pipes), T_supply=75.0, T_return=45.0, Q_nominal=Q_nominal);

  WarmingUp.HeatNetwork.Heat.Pipe pipe1a_cold(length = 170.365, diameter = 0.15, temperature=45.0, T_supply=75.0, T_return=45.0, HeatOut.Heat(nominal=heat_nominal_cold_pipes), HeatIn.Heat(max=heat_max_abs, nominal=heat_nominal_cold_pipes), Q_nominal=Q_nominal);
  WarmingUp.HeatNetwork.Heat.Pipe pipe1b_cold(length = 309.635, diameter = 0.15, temperature=45.0, T_supply=75.0, T_return=45.0, HeatOut.Heat(nominal=heat_nominal_cold_pipes), HeatIn.Heat(max=heat_max_abs, nominal=heat_nominal_cold_pipes), Q_nominal=Q_nominal);
  WarmingUp.HeatNetwork.Heat.Pipe pipe4a_cold(length = 5, diameter = 0.15, temperature=45.0, T_supply=75.0, T_return=45.0, HeatOut.Heat(nominal=heat_nominal_cold_pipes), HeatIn.Heat(max=heat_max_abs, nominal=heat_nominal_cold_pipes), Q_nominal=Q_nominal);
  WarmingUp.HeatNetwork.Heat.Pipe pipe4b_cold(length = 15, diameter = 0.15, temperature=45.0, T_supply=75.0, T_return=45.0, HeatOut.Heat(nominal=heat_nominal_cold_pipes), HeatIn.Heat(max=heat_max_abs, nominal=heat_nominal_cold_pipes), Q_nominal=Q_nominal);
  WarmingUp.HeatNetwork.Heat.Pipe pipe5_cold(length = 126, diameter = 0.15, temperature=45.0, T_supply=75.0, T_return=45.0, HeatOut.Heat(nominal=heat_nominal_cold_pipes), HeatIn.Heat(max=heat_max_abs, nominal=heat_nominal_cold_pipes), Q_nominal=Q_nominal);
  WarmingUp.HeatNetwork.Heat.Pipe pipe7_cold(length = 60, diameter = 0.15, temperature=45.0, T_supply=75.0, T_return=45.0, HeatOut.Heat(nominal=heat_nominal_cold_pipes), HeatIn.Heat(max=heat_max_abs, nominal=heat_nominal_cold_pipes), Q_nominal=Q_nominal);
  WarmingUp.HeatNetwork.Heat.Pipe pipe9_cold(length = 70, diameter = 0.15, temperature=45.0, T_supply=75.0, T_return=45.0, HeatOut.Heat(nominal=heat_nominal_cold_pipes), HeatIn.Heat(max=heat_max_abs, nominal=heat_nominal_cold_pipes), Q_nominal=Q_nominal);
  WarmingUp.HeatNetwork.Heat.Pipe pipe15_cold(length = 129, diameter = 0.15, temperature=45.0, T_supply=75.0, T_return=45.0, HeatOut.Heat(nominal=heat_nominal_cold_pipes), HeatIn.Heat(max=heat_max_abs, nominal=heat_nominal_cold_pipes), Q_nominal=Q_nominal);
  WarmingUp.HeatNetwork.Heat.Pipe pipe25_cold(length = 150, diameter = 0.15, temperature=45.0, T_supply=75.0, T_return=45.0, HeatOut.Heat(nominal=heat_nominal_cold_pipes), HeatIn.Heat(max=heat_max_abs, nominal=heat_nominal_cold_pipes), Q_nominal=Q_nominal);
  WarmingUp.HeatNetwork.Heat.Pipe pipe26_cold(length = 30, diameter = 0.1, temperature=45.0, T_supply=75.0, T_return=45.0, HeatOut.Heat(nominal=heat_nominal_cold_pipes), HeatIn.Heat(max=heat_max_abs, nominal=heat_nominal_cold_pipes), Q_nominal=Q_nominal);
  WarmingUp.HeatNetwork.Heat.Pipe pipe27_cold(length = 55, diameter = 0.15, temperature=45.0, T_supply=75.0, T_return=45.0, HeatOut.Heat(nominal=heat_nominal_cold_pipes), HeatIn.Heat(max=heat_max_abs, nominal=heat_nominal_cold_pipes), Q_nominal=Q_nominal);
  WarmingUp.HeatNetwork.Heat.Pipe pipe29_cold(length = 134, diameter = 0.15, temperature=45.0, T_supply=75.0, T_return=45.0, HeatOut.Heat(nominal=heat_nominal_cold_pipes), HeatIn.Heat(max=heat_max_abs, nominal=heat_nominal_cold_pipes), Q_nominal=Q_nominal);
  WarmingUp.HeatNetwork.Heat.Pipe pipe30_cold(length = 60, diameter = 0.1, temperature=45.0, T_supply=75.0, T_return=45.0, HeatOut.Heat(nominal=heat_nominal_cold_pipes), HeatIn.Heat(max=heat_max_abs, nominal=heat_nominal_cold_pipes), Q_nominal=Q_nominal);
  WarmingUp.HeatNetwork.Heat.Pipe pipe31_cold(length = 60, diameter = 0.1, temperature=45.0, T_supply=75.0, T_return=45.0, HeatOut.Heat(nominal=heat_nominal_cold_pipes), HeatIn.Heat(max=heat_max_abs, nominal=heat_nominal_cold_pipes), Q_nominal=Q_nominal);
  WarmingUp.HeatNetwork.Heat.Pipe pipe32_cold(length = 50, diameter = 0.1, temperature=45.0, T_supply=75.0, T_return=45.0, HeatOut.Heat(nominal=heat_nominal_cold_pipes), HeatIn.Heat(max=heat_max_abs, nominal=heat_nominal_cold_pipes), Q_nominal=Q_nominal);
  WarmingUp.HeatNetwork.Heat.Pipe pipe52_cold(disconnectable = true, length = 10, diameter = 0.164, temperature=45.0, T_supply=75.0, T_return=45.0, HeatOut.Heat(nominal=heat_nominal_cold_pipes), HeatIn.Heat(max=heat_max_abs, nominal=heat_nominal_cold_pipes), Q_nominal=Q_nominal);

  WarmingUp.HeatNetwork.Heat.Demand demand7(Heat_demand(min=0.0, max=heat_max_abs), T_supply=75.0, T_return=45.0, Q_nominal=Q_nominal);
  WarmingUp.HeatNetwork.Heat.Demand demand91(Heat_demand(min=0.0, max=heat_max_abs), T_supply=75.0, T_return=45.0, Q_nominal=Q_nominal);
  WarmingUp.HeatNetwork.Heat.Demand demand92(Heat_demand(min=0.0, max=heat_max_abs), T_supply=75.0, T_return=45.0, Q_nominal=Q_nominal);

  WarmingUp.HeatNetwork.Heat.Buffer buffer1(Stored_heat(min=0.0, max = 200*4200*988*30), init_Heat=init_Heat, T_supply=75.0, T_return=45.0, Q_nominal=Q_nominal);

  WarmingUp.HeatNetwork.Heat.Pipe pipe1a_hot(length = 170.365, diameter = 0.15, temperature=75.0, T_supply=75.0, T_return=45.0, HeatOut.Heat(min=-heat_max_abs, max=heat_max_abs, nominal=heat_nominal_hot_pipes), HeatIn.Heat(min=-heat_max_abs, max=heat_max_abs, nominal=heat_nominal_hot_pipes), Q_nominal=Q_nominal);
  WarmingUp.HeatNetwork.Heat.Pipe pipe1b_hot(length = 309.635, diameter = 0.15, temperature=75.0, T_supply=75.0, T_return=45.0, HeatOut.Heat(min=-heat_max_abs, max=heat_max_abs, nominal=heat_nominal_hot_pipes), HeatIn.Heat(min=-heat_max_abs, max=heat_max_abs, nominal=heat_nominal_hot_pipes), Q_nominal=Q_nominal);
  WarmingUp.HeatNetwork.Heat.Pipe pipe4a_hot(length = 5, diameter = 0.15, temperature=75.0, T_supply=75.0, T_return=45.0, HeatOut.Heat(min=-heat_max_abs, max=heat_max_abs, nominal=heat_nominal_hot_pipes), HeatIn.Heat(min=-heat_max_abs, max=heat_max_abs, nominal=heat_nominal_hot_pipes), Q_nominal=Q_nominal);
  WarmingUp.HeatNetwork.Heat.Pipe pipe4b_hot(length = 15, diameter = 0.15, temperature=75.0, T_supply=75.0, T_return=45.0, HeatOut.Heat(min=-heat_max_abs, max=heat_max_abs, nominal=heat_nominal_hot_pipes), HeatIn.Heat(min=-heat_max_abs, max=heat_max_abs, nominal=heat_nominal_hot_pipes), Q_nominal=Q_nominal);
  WarmingUp.HeatNetwork.Heat.Pipe pipe5_hot(length = 126, diameter = 0.15, temperature=75.0, T_supply=75.0, T_return=45.0, HeatOut.Heat(min=-heat_max_abs, max=heat_max_abs, nominal=heat_nominal_hot_pipes), HeatIn.Heat(min=-heat_max_abs, max=heat_max_abs, nominal=heat_nominal_hot_pipes), Q_nominal=Q_nominal);
  WarmingUp.HeatNetwork.Heat.Pipe pipe7_hot(length = 60, diameter = 0.15, temperature=75.0, T_supply=75.0, T_return=45.0, HeatOut.Heat(min=-heat_max_abs, max=heat_max_abs, nominal=heat_nominal_hot_pipes), HeatIn.Heat(min=-heat_max_abs, max=heat_max_abs, nominal=heat_nominal_hot_pipes), Q_nominal=Q_nominal);
  WarmingUp.HeatNetwork.Heat.Pipe pipe9_hot(length = 70, diameter = 0.15, temperature=75.0, T_supply=75.0, T_return=45.0, HeatOut.Heat(min=-heat_max_abs, max=heat_max_abs, nominal=heat_nominal_hot_pipes), HeatIn.Heat(min=-heat_max_abs, max=heat_max_abs, nominal=heat_nominal_hot_pipes), Q_nominal=Q_nominal);
  WarmingUp.HeatNetwork.Heat.Pipe pipe15_hot(length = 129, diameter = 0.15, temperature=75.0, T_supply=75.0, T_return=45.0, HeatOut.Heat(min=-heat_max_abs, max=heat_max_abs, nominal=heat_nominal_hot_pipes), HeatIn.Heat(min=-heat_max_abs, max=heat_max_abs, nominal=heat_nominal_hot_pipes), Q_nominal=Q_nominal);
  WarmingUp.HeatNetwork.Heat.Pipe pipe25_hot(length = 150, diameter = 0.15, temperature=75.0, T_supply=75.0, T_return=45.0, HeatOut.Heat(min=-heat_max_abs, max=heat_max_abs, nominal=heat_nominal_hot_pipes), HeatIn.Heat(min=-heat_max_abs, max=heat_max_abs, nominal=heat_nominal_hot_pipes), Q_nominal=Q_nominal);
  WarmingUp.HeatNetwork.Heat.Pipe pipe26_hot(length = 30, diameter = 0.1, temperature=75.0, T_supply=75.0, T_return=45.0, HeatOut.Heat(min=-heat_max_abs, max=heat_max_abs, nominal=heat_nominal_hot_pipes), HeatIn.Heat(min=-heat_max_abs, max=heat_max_abs, nominal=heat_nominal_hot_pipes), Q_nominal=Q_nominal);
  WarmingUp.HeatNetwork.Heat.Pipe pipe27_hot(length = 55, diameter = 0.15, temperature=75.0, T_supply=75.0, T_return=45.0, HeatOut.Heat(min=-heat_max_abs, max=heat_max_abs, nominal=heat_nominal_hot_pipes), HeatIn.Heat(min=-heat_max_abs, max=heat_max_abs, nominal=heat_nominal_hot_pipes), Q_nominal=Q_nominal);
  WarmingUp.HeatNetwork.Heat.Pipe pipe29_hot(length = 134, diameter = 0.15, temperature=75.0, T_supply=75.0, T_return=45.0, HeatOut.Heat(min=-heat_max_abs, max=heat_max_abs, nominal=heat_nominal_hot_pipes), HeatIn.Heat(min=-heat_max_abs, max=heat_max_abs, nominal=heat_nominal_hot_pipes), Q_nominal=Q_nominal);
  WarmingUp.HeatNetwork.Heat.Pipe pipe30_hot(length = 60, diameter = 0.1, temperature=75.0, T_supply=75.0, T_return=45.0, HeatOut.Heat(min=-heat_max_abs, max=heat_max_abs, nominal=heat_nominal_hot_pipes), HeatIn.Heat(min=-heat_max_abs, max=heat_max_abs, nominal=heat_nominal_hot_pipes), Q_nominal=Q_nominal);
  WarmingUp.HeatNetwork.Heat.Pipe pipe31_hot(length = 60, diameter = 0.1, temperature=75.0, T_supply=75.0, T_return=45.0, HeatOut.Heat(min=-heat_max_abs, max=heat_max_abs, nominal=heat_nominal_hot_pipes), HeatIn.Heat(min=-heat_max_abs, max=heat_max_abs, nominal=heat_nominal_hot_pipes), Q_nominal=Q_nominal);
  WarmingUp.HeatNetwork.Heat.Pipe pipe32_hot(length = 50, diameter = 0.1, temperature=75.0, T_supply=75.0, T_return=45.0, HeatOut.Heat(min=-heat_max_abs, max=heat_max_abs, nominal=heat_nominal_hot_pipes), HeatIn.Heat(min=-heat_max_abs, max=heat_max_abs, nominal=heat_nominal_hot_pipes), Q_nominal=Q_nominal);
  WarmingUp.HeatNetwork.Heat.Pipe pipe52_hot(disconnectable = true, length = 10, diameter = 0.164, temperature=75.0, T_supply=75.0, T_return=45.0, HeatOut.Heat(min=-heat_max_abs, max=heat_max_abs, nominal=heat_nominal_hot_pipes), HeatIn.Heat(min=-heat_max_abs, max=heat_max_abs, nominal=heat_nominal_hot_pipes), Q_nominal=Q_nominal);

  WarmingUp.HeatNetwork.Heat.Node nodeS2_hot(n=3);
  WarmingUp.HeatNetwork.Heat.Node nodeD7_hot(n=3);
  WarmingUp.HeatNetwork.Heat.Node nodeD92_hot(n=3);
  WarmingUp.HeatNetwork.Heat.Node nodeB1_hot(n=3);

  WarmingUp.HeatNetwork.Heat.Node nodeS2_cold(n=3);
  WarmingUp.HeatNetwork.Heat.Node nodeD7_cold(n=3);
  WarmingUp.HeatNetwork.Heat.Node nodeD92_cold(n=3);
  WarmingUp.HeatNetwork.Heat.Node nodeB1_cold(n=3);

  //Q in [m^3/s] and H in [m]
  WarmingUp.HeatNetwork.Heat.Pump pump1(T_supply=75.0, T_return=45.0, Q_nominal=Q_nominal);
  WarmingUp.HeatNetwork.Heat.Pump pump2(T_supply=75.0, T_return=45.0, Q_nominal=Q_nominal);

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
  connect(source1.HeatOut, pipe1a_hot.HeatIn) ;
  connect(pipe1a_hot.HeatOut, pump1.HeatIn) ;
  connect(pump1.HeatOut, pipe1b_hot.HeatIn) ;
  connect(pipe1b_hot.HeatOut, nodeS2_hot.HeatConn[1]) ;
  connect(nodeS2_hot.HeatConn[2], pipe5_hot.HeatIn) ;
  connect(pipe5_hot.HeatOut, pipe7_hot.HeatIn) ;
  connect(pipe7_hot.HeatOut, pipe9_hot.HeatIn) ;
  connect(pipe9_hot.HeatOut, nodeB1_hot.HeatConn[1]) ;
  connect(nodeB1_hot.HeatConn[2], pipe15_hot.HeatIn) ;
  connect(pipe15_hot.HeatOut, pipe25_hot.HeatIn) ;
  connect(pipe25_hot.HeatOut, nodeD7_hot.HeatConn[1]) ;
  connect(nodeD7_hot.HeatConn[2], pipe27_hot.HeatIn) ;
  connect(pipe27_hot.HeatOut, pipe29_hot.HeatIn) ;
  connect(pipe29_hot.HeatOut, nodeD92_hot.HeatConn[1]) ;
  connect(nodeD92_hot.HeatConn[2], pipe31_hot.HeatIn) ;
  connect(pipe31_hot.HeatOut, pipe32_hot.HeatIn) ;
  connect(pipe32_hot.HeatOut, demand91.HeatIn) ;


//Cold lines from demand91 to source 1
  connect(demand91.HeatOut, pipe32_cold.HeatIn) ;
  connect(pipe32_cold.HeatOut, pipe31_cold.HeatIn) ;
  connect(pipe31_cold.HeatOut, nodeD92_cold.HeatConn[1]) ;
  connect(nodeD92_cold.HeatConn[2], pipe29_cold.HeatIn) ;
  connect(pipe29_cold.HeatOut, pipe27_cold.HeatIn) ;
  connect(pipe27_cold.HeatOut, nodeD7_cold.HeatConn[1]) ;
  connect(nodeD7_cold.HeatConn[2], pipe25_cold.HeatIn) ;
  connect(pipe25_cold.HeatOut, pipe15_cold.HeatIn) ;
  connect(pipe15_cold.HeatOut, nodeB1_cold.HeatConn[1]) ;
  connect(nodeB1_cold.HeatConn[2], pipe9_cold.HeatIn) ;
  connect(pipe9_cold.HeatOut, pipe7_cold.HeatIn) ;
  connect(pipe7_cold.HeatOut, pipe5_cold.HeatIn) ;
  connect(pipe5_cold.HeatOut, nodeS2_cold.HeatConn[1]) ;
  connect(nodeS2_cold.HeatConn[2], pipe1b_cold.HeatIn) ;
  connect(pipe1b_cold.HeatOut, pipe1a_cold.HeatIn) ;
  connect(pipe1a_cold.HeatOut, source1.HeatIn) ;


//Source2
  connect(source2.HeatOut, pipe4a_hot.HeatIn) ;
  connect(pipe4a_hot.HeatOut, pump2.HeatIn) ;
  connect(pump2.HeatOut, pipe4b_hot.HeatIn) ;
  connect(pipe4b_hot.HeatOut, nodeS2_hot.HeatConn[3]) ;
  connect(nodeS2_cold.HeatConn[3], pipe4a_cold.HeatIn) ;
  connect(pipe4a_cold.HeatOut, pipe4b_cold.HeatIn) ;
  connect(pipe4b_cold.HeatOut, source2.HeatIn) ;

//Demand7
  connect(nodeD7_hot.HeatConn[3], pipe26_hot.HeatIn) ;
  connect(pipe26_hot.HeatOut, demand7.HeatIn) ;
  connect(demand7.HeatOut, pipe26_cold.HeatIn) ;
  connect(pipe26_cold.HeatOut, nodeD7_cold.HeatConn[3]) ;

//Demand92
  connect(nodeD92_hot.HeatConn[3], pipe30_hot.HeatIn) ;
  connect(pipe30_hot.HeatOut, demand92.HeatIn) ;
  connect(demand92.HeatOut, pipe30_cold.HeatIn) ;
  connect(pipe30_cold.HeatOut, nodeD92_cold.HeatConn[3]) ;

  //Buffer1
  // Hot
  connect(nodeB1_hot.HeatConn[3], pipe52_hot.HeatIn) ;
  connect(pipe52_hot.HeatOut, buffer1.HeatIn) ;
  // Cold
  connect(buffer1.HeatOut, pipe52_cold.HeatIn) ;
  connect(pipe52_cold.HeatOut, nodeB1_cold.HeatConn[3]) ;

end Example_Heat;
