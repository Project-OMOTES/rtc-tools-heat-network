model Example_Heat
  // Declare Model Elements

  constant Real heat_nominal_cold_pipes = 45.0*4200*988*0.005;
  constant Real heat_nominal_hot_pipes = 75.0*4200*988*0.005;
  constant Real heat_max_abs = 0.01111*100.0*4200.0*988;

  //Heatsource min en max in [W]
  WarmingUp.HeatNetwork.Heat.Source source1(Heat_source(min=0.0, max=1.5e6, nominal=1e6), Heat_out(max=heat_max_abs));
  WarmingUp.HeatNetwork.Heat.Source source2(Heat_source(min=0.0, max=1.5e7, nominal=1e6), Heat_out(max=heat_max_abs));

  WarmingUp.HeatNetwork.Heat.Pipe pipe1a_cold(length=170.365, diameter=0.15, HeatOut.Heat(nominal=heat_nominal_cold_pipes), HeatIn.Heat(max=0.01111*70.0*4200.0*988, nominal=heat_nominal_cold_pipes), temperature=45.0);
  WarmingUp.HeatNetwork.Heat.Pipe pipe1b_cold(length=309.635, diameter=0.15, HeatOut.Heat(nominal=heat_nominal_cold_pipes), HeatIn.Heat(max=0.01111*70.0*4200.0*988, nominal=heat_nominal_cold_pipes), temperature=45.0);
  WarmingUp.HeatNetwork.Heat.Pipe pipe4a_cold(length=5.0, diameter=0.15, HeatOut.Heat(nominal=heat_nominal_cold_pipes), HeatIn.Heat(max=0.01111*70.0*4200.0*988, nominal=heat_nominal_cold_pipes), temperature=45.0);
  WarmingUp.HeatNetwork.Heat.Pipe pipe4b_cold(length=15.0, diameter=0.15, HeatOut.Heat(nominal=heat_nominal_cold_pipes), HeatIn.Heat(max=0.01111*70.0*4200.0*988, nominal=heat_nominal_cold_pipes), temperature=45.0);
  WarmingUp.HeatNetwork.Heat.Pipe pipe579_cold(length=256.0, diameter=0.15, HeatOut.Heat(nominal=heat_nominal_cold_pipes), HeatIn.Heat(max=0.01111*70.0*4200.0*988, nominal=heat_nominal_cold_pipes), temperature=45.0);
  WarmingUp.HeatNetwork.Heat.Pipe pipe15_cold(length=129.0, diameter=0.15, HeatOut.Heat(nominal=heat_nominal_cold_pipes), HeatIn.Heat(max=0.01111*70.0*4200.0*988, nominal=heat_nominal_cold_pipes), temperature=45.0);
  WarmingUp.HeatNetwork.Heat.Pipe pipe25_cold(length=150.0, diameter=0.15, HeatOut.Heat(nominal=heat_nominal_cold_pipes), HeatIn.Heat(max=0.01111*70.0*4200.0*988, nominal=heat_nominal_cold_pipes), temperature=45.0);
  WarmingUp.HeatNetwork.Heat.Pipe pipe26_cold(length=30.0, diameter=0.1, HeatOut.Heat(nominal=heat_nominal_cold_pipes), HeatIn.Heat(max=0.01111*70.0*4200.0*988, nominal=heat_nominal_cold_pipes), temperature=45.0);
  WarmingUp.HeatNetwork.Heat.Pipe pipe27_cold(length=55.0, diameter=0.15, HeatOut.Heat(nominal=heat_nominal_cold_pipes), HeatIn.Heat(max=0.01111*70.0*4200.0*988, nominal=heat_nominal_cold_pipes), temperature=45.0);
  WarmingUp.HeatNetwork.Heat.Pipe pipe29_cold(length=134.0, diameter=0.15, HeatOut.Heat(nominal=heat_nominal_cold_pipes), HeatIn.Heat(max=0.01111*70.0*4200.0*988, nominal=heat_nominal_cold_pipes), temperature=45.0);
  WarmingUp.HeatNetwork.Heat.Pipe pipe30_cold(length=60.0, diameter=0.1, HeatOut.Heat(nominal=heat_nominal_cold_pipes), HeatIn.Heat(max=0.01111*70.0*4200.0*988, nominal=heat_nominal_cold_pipes), temperature=45.0);

  WarmingUp.HeatNetwork.Heat.Demand demand7(Heat_in(max=heat_max_abs));
  WarmingUp.HeatNetwork.Heat.Demand demand91(Heat_in(max=heat_max_abs));
  WarmingUp.HeatNetwork.Heat.Demand demand92(Heat_in(max=heat_max_abs));

  WarmingUp.HeatNetwork.Heat.Pipe pipe1a_hot(length=170.365, diameter=0.15, HeatIn.Heat(min=-heat_max_abs, max=heat_max_abs, nominal=heat_nominal_hot_pipes), HeatOut.Heat(nominal=heat_nominal_hot_pipes), temperature=75.0);
  WarmingUp.HeatNetwork.Heat.Pipe pipe1b_hot(length=309.635, diameter=0.15, HeatIn.Heat(min=-heat_max_abs, max=heat_max_abs, nominal=heat_nominal_hot_pipes), HeatOut.Heat(nominal=heat_nominal_hot_pipes), temperature=75.0);
  WarmingUp.HeatNetwork.Heat.Pipe pipe4a_hot(length=5.0, diameter=0.15, HeatIn.Heat(min=-heat_max_abs, max=heat_max_abs, nominal=heat_nominal_hot_pipes), HeatOut.Heat(nominal=heat_nominal_hot_pipes), temperature=75.0);
  WarmingUp.HeatNetwork.Heat.Pipe pipe4b_hot(length=15.0, diameter=0.15, HeatIn.Heat(min=-heat_max_abs, max=heat_max_abs, nominal=heat_nominal_hot_pipes), HeatOut.Heat(nominal=heat_nominal_hot_pipes), temperature=75.0);
  WarmingUp.HeatNetwork.Heat.Pipe pipe579_hot(length=256.0, diameter=0.15, HeatIn.Heat(min=-heat_max_abs, max=heat_max_abs, nominal=heat_nominal_hot_pipes), HeatOut.Heat(nominal=heat_nominal_hot_pipes), temperature=75.0);
  WarmingUp.HeatNetwork.Heat.Pipe pipe15_hot(length=129.0, diameter=0.15, HeatIn.Heat(min=-heat_max_abs, max=heat_max_abs, nominal=heat_nominal_hot_pipes), HeatOut.Heat(nominal=heat_nominal_hot_pipes), temperature=75.0);
  WarmingUp.HeatNetwork.Heat.Pipe pipe25_hot(length=150.0, diameter=0.15, HeatIn.Heat(min=-heat_max_abs, max=heat_max_abs, nominal=heat_nominal_hot_pipes), HeatOut.Heat(nominal=heat_nominal_hot_pipes), temperature=75.0);
  WarmingUp.HeatNetwork.Heat.Pipe pipe26_hot(length=30.0, diameter=0.1, HeatIn.Heat(min=-heat_max_abs, max=heat_max_abs, nominal=heat_nominal_hot_pipes), HeatOut.Heat(nominal=heat_nominal_hot_pipes), temperature=75.0);
  WarmingUp.HeatNetwork.Heat.Pipe pipe27_hot(length=55.0, diameter=0.15, HeatIn.Heat(min=-heat_max_abs, max=heat_max_abs, nominal=heat_nominal_hot_pipes), HeatOut.Heat(nominal=heat_nominal_hot_pipes), temperature=75.0);
  WarmingUp.HeatNetwork.Heat.Pipe pipe29_hot(length=134.0, diameter=0.15, HeatIn.Heat(min=-heat_max_abs, max=heat_max_abs, nominal=heat_nominal_hot_pipes), HeatOut.Heat(nominal=heat_nominal_hot_pipes), temperature=75.0);
  WarmingUp.HeatNetwork.Heat.Pipe pipe30_hot(length=60.0, diameter=0.1, HeatIn.Heat(min=-heat_max_abs, max=heat_max_abs, nominal=heat_nominal_hot_pipes), HeatOut.Heat(nominal=heat_nominal_hot_pipes), temperature=75.0);

  WarmingUp.HeatNetwork.Heat.Node nodeS2_hot(n=3);
  WarmingUp.HeatNetwork.Heat.Node nodeD7_hot(n=3);
  WarmingUp.HeatNetwork.Heat.Node nodeD92_hot(n=3);

  WarmingUp.HeatNetwork.Heat.Node nodeS2_cold(n=3);
  WarmingUp.HeatNetwork.Heat.Node nodeD7_cold(n=3);
  WarmingUp.HeatNetwork.Heat.Node nodeD92_cold(n=3);

  //Q in [m^3/s] and H in [m]
  WarmingUp.HeatNetwork.Heat.Pump pump1;
  WarmingUp.HeatNetwork.Heat.Pump pump2;

  // Define Input/Output Variables and set them equal to model variables.
  //Heatdemand min en max in [W]
  input Modelica.SIunits.Heat Heat_source1(fixed=false, nominal=heat_nominal_hot_pipes) = source1.Heat_source;
  input Modelica.SIunits.Heat Heat_source2(fixed=false, nominal=heat_nominal_hot_pipes) = source2.Heat_source;

  output Modelica.SIunits.Heat Heat_demand7_opt = demand7.Heat_demand;
  output Modelica.SIunits.Heat Heat_demand91_opt = demand91.Heat_demand;
  output Modelica.SIunits.Heat Heat_demand92_opt = demand92.Heat_demand;

equation
// Connect Model Elements

  // S1 -> D7 -> S2 -> D92 -> D91

  //Hot lines from source1 to demand91
  connect(source1.HeatOut, pipe1a_hot.HeatIn) ;
  connect(pipe1a_hot.HeatOut, pump1.HeatIn) ;
  connect(pump1.HeatOut, pipe1b_hot.HeatIn) ;
  connect(pipe1b_hot.HeatOut, nodeD7_hot.HeatConn[1]) ;
  connect(nodeD7_hot.HeatConn[2], pipe579_hot.HeatIn) ;

  connect(pipe579_hot.HeatOut, nodeS2_hot.HeatConn[1]) ;
  connect(nodeS2_hot.HeatConn[2], pipe15_hot.HeatIn) ;
  connect(pipe15_hot.HeatOut, pipe25_hot.HeatIn) ;
  connect(pipe25_hot.HeatOut, nodeD92_hot.HeatConn[1]) ;
  connect(nodeD92_hot.HeatConn[2], pipe27_hot.HeatIn) ;
  connect(pipe27_hot.HeatOut, pipe29_hot.HeatIn) ;
  connect(pipe29_hot.HeatOut, demand91.HeatIn) ;

  //Cold lines from demand91 to source 1
  connect(demand91.HeatOut, pipe29_cold.HeatIn) ;
  connect(pipe29_cold.HeatOut, pipe27_cold.HeatIn) ;
  connect(pipe27_cold.HeatOut, nodeD92_cold.HeatConn[1]) ;
  connect(nodeD92_cold.HeatConn[2], pipe25_cold.HeatIn) ;
  connect(pipe25_cold.HeatOut, pipe15_cold.HeatIn) ;
  connect(pipe15_cold.HeatOut, nodeS2_cold.HeatConn[1]) ;
  connect(nodeS2_cold.HeatConn[2], pipe579_cold.HeatIn) ;
  connect(pipe579_cold.HeatOut, nodeD7_cold.HeatConn[1]) ;
  connect(nodeD7_cold.HeatConn[2], pipe1b_cold.HeatIn) ;
  connect(pipe1b_cold.HeatOut, pipe1a_cold.HeatIn) ;
  connect(pipe1a_cold.HeatOut, source1.HeatIn) ;

  //Demand7
  connect(nodeD7_hot.HeatConn[3], pipe26_hot.HeatIn) ;
  connect(pipe26_hot.HeatOut, demand7.HeatIn) ;
  connect(demand7.HeatOut, pipe26_cold.HeatIn) ;
  connect(pipe26_cold.HeatOut, nodeD7_cold.HeatConn[3]) ;

  //Source2
  connect(source2.HeatOut, pipe4a_hot.HeatIn) ;
  connect(pipe4a_hot.HeatOut, pump2.HeatIn) ;
  connect(pump2.HeatOut, pipe4b_hot.HeatIn) ;
  connect(pipe4b_hot.HeatOut, nodeS2_hot.HeatConn[3]) ;

  connect(nodeS2_cold.HeatConn[3], pipe4b_cold.HeatIn) ;
  connect(pipe4b_cold.HeatOut, pipe4a_cold.HeatIn) ;
  connect(pipe4a_cold.HeatOut, source2.HeatIn) ;

  //Demand92
  connect(nodeD92_hot.HeatConn[3], pipe30_hot.HeatIn) ;
  connect(pipe30_hot.HeatOut, demand92.HeatIn) ;
  connect(demand92.HeatOut, pipe30_cold.HeatIn) ;
  connect(pipe30_cold.HeatOut, nodeD92_cold.HeatConn[3]) ;
end Example_Heat;
