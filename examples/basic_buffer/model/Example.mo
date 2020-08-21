model Example
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

  //Heatsource min en max in [W]
  WarmingUp.HeatNetwork.HeatSource source1(Heat(min=0.0, max=1.5e6, nominal=1e6), theta = theta, QTHOut.T(min=t_source1_min, max=t_source1_max));
  WarmingUp.HeatNetwork.HeatSource source2(Heat(min=0.0, max=1.5e7, nominal=1e6), theta = theta, QTHOut.T(min=t_source2_min, max=t_source2_max));

  WarmingUp.HeatNetwork.Pipe pipe1aC(length = 170.365, diameter = 0.15, temperature=45.0, Q(nominal=Q_nominal), QTHIn.T(min=t_return_min, max=t_return_max), QTHOut.T(min=t_return_min, max=t_return_max), theta = theta, sign_dT=-1.0);
  WarmingUp.HeatNetwork.Pipe pipe1bC(length = 309.635, diameter = 0.15, temperature=45.0, Q(nominal=Q_nominal), QTHIn.T(min=t_return_min, max=t_return_max), QTHOut.T(min=t_return_min, max=t_return_max), theta = theta, sign_dT=-1.0);
  WarmingUp.HeatNetwork.Pipe pipe4aC(length = 5, diameter = 0.15, temperature=45.0, Q(nominal=Q_nominal), QTHIn.T(min=t_return_min, max=t_return_max), QTHOut.T(min=t_return_min, max=t_return_max), theta = theta, sign_dT=-1.0);
  WarmingUp.HeatNetwork.Pipe pipe4bC(length = 15, diameter = 0.15, temperature=45.0, Q(nominal=Q_nominal), QTHIn.T(min=t_return_min, max=t_return_max), QTHOut.T(min=t_return_min, max=t_return_max), theta = theta, sign_dT=-1.0);
  WarmingUp.HeatNetwork.Pipe pipe5C(length = 126, diameter = 0.15, temperature=45.0, Q(nominal=Q_nominal), QTHIn.T(min=t_return_min, max=t_return_max), QTHOut.T(min=t_return_min, max=t_return_max), theta = theta, sign_dT=-1.0);
  WarmingUp.HeatNetwork.Pipe pipe7C(length = 60, diameter = 0.15, temperature=45.0, Q(nominal=Q_nominal), QTHIn.T(min=t_return_min, max=t_return_max), QTHOut.T(min=t_return_min, max=t_return_max), theta = theta, sign_dT=-1.0);
  WarmingUp.HeatNetwork.Pipe pipe9C(length = 70, diameter = 0.15, temperature=45.0, Q(nominal=Q_nominal), QTHIn.T(min=t_return_min, max=t_return_max), QTHOut.T(min=t_return_min, max=t_return_max), theta = theta, sign_dT=-1.0);
  WarmingUp.HeatNetwork.Pipe pipe15C(length = 129, diameter = 0.15, temperature=45.0, Q(nominal=Q_nominal), QTHIn.T(min=t_return_min, max=t_return_max), QTHOut.T(min=t_return_min, max=t_return_max), theta = theta, sign_dT=-1.0);
  WarmingUp.HeatNetwork.Pipe pipe25C(length = 150, diameter = 0.15, temperature=45.0, Q(nominal=Q_nominal), QTHIn.T(min=t_return_min, max=t_return_max), QTHOut.T(min=t_return_min, max=t_return_max), theta = theta, sign_dT=-1.0);
  WarmingUp.HeatNetwork.Pipe pipe26C(length = 30, diameter = 0.1, temperature=45.0, Q(nominal=Q_nominal), QTHIn.T(min=t_return_min, max=t_return_max), QTHOut.T(min=t_return_min, max=t_return_max), theta = theta, sign_dT=-1.0);
  WarmingUp.HeatNetwork.Pipe pipe27C(length = 55, diameter = 0.15, temperature=45.0, Q(nominal=Q_nominal), QTHIn.T(min=t_return_min, max=t_return_max), QTHOut.T(min=t_return_min, max=t_return_max), theta = theta, sign_dT=-1.0);
  WarmingUp.HeatNetwork.Pipe pipe29C(length = 134, diameter = 0.15, temperature=45.0, Q(nominal=Q_nominal), QTHIn.T(min=t_return_min, max=t_return_max), QTHOut.T(min=t_return_min, max=t_return_max), theta = theta, sign_dT=-1.0);
  WarmingUp.HeatNetwork.Pipe pipe30C(length = 60, diameter = 0.1, temperature=45.0, Q(nominal=Q_nominal), QTHIn.T(min=t_return_min, max=t_return_max), QTHOut.T(min=t_return_min, max=t_return_max), theta = theta, sign_dT=-1.0);
  WarmingUp.HeatNetwork.Pipe pipe31C(length = 60, diameter = 0.1, temperature=45.0, Q(nominal=Q_nominal), QTHIn.T(min=t_return_min, max=t_return_max), QTHOut.T(min=t_return_min, max=t_return_max), theta = theta, sign_dT=-1.0);
  WarmingUp.HeatNetwork.Pipe pipe32C(length = 50, diameter = 0.1, temperature=45.0, Q(nominal=Q_nominal), QTHIn.T(min=t_return_min, max=t_return_max), QTHOut.T(min=t_return_min, max=t_return_max), theta = theta, sign_dT=-1.0);
  WarmingUp.HeatNetwork.Pipe pipe52_inC(length = 10, diameter = 0.164, temperature=45.0, Q(max=0.023, nominal=Q_nominal), QTHIn.T(min=t_return_min, max=t_return_max), QTHOut.T(min=t_return_min, max=t_return_max), dH(min=-1.0), theta = 0.0, sign_dT=-1.0);
  WarmingUp.HeatNetwork.Pipe pipe52_outC(length = 10, diameter = 0.164, temperature=45.0, Q(max=0.023, nominal=Q_nominal), QTHIn.T(min=t_return_min, max=t_return_max), QTHOut.T(min=t_return_min, max=t_return_max), dH(min=-1.0), theta = 0.0, sign_dT=-1.0);

  WarmingUp.HeatNetwork.HeatDemand demand7(theta = theta, QTHIn.T(min=t_demand_min));
  WarmingUp.HeatNetwork.HeatDemand demand91(theta = theta, QTHIn.T(min=t_demand_min));
  WarmingUp.HeatNetwork.HeatDemand demand92(theta = theta, QTHIn.T(min=t_demand_min));

  WarmingUp.HeatNetwork.HeatBuffer buffer1(theta = theta);

  WarmingUp.HeatNetwork.Pipe pipe1aH(length = 170.365, diameter = 0.15, temperature=75.0, Q(nominal=Q_nominal), QTHIn.T(min=t_supply_min, max=t_supply_max), QTHOut.T(min=t_supply_min, max=t_supply_max), theta = theta, sign_dT=1.0);
  WarmingUp.HeatNetwork.Pipe pipe1bH(length = 309.635, diameter = 0.15, temperature=75.0, Q(nominal=Q_nominal), QTHIn.T(min=t_supply_min, max=t_supply_max), QTHOut.T(min=t_supply_min, max=t_supply_max), theta = theta, sign_dT=1.0);
  WarmingUp.HeatNetwork.Pipe pipe4aH(length = 5, diameter = 0.15, temperature=75.0, Q(nominal=Q_nominal), QTHIn.T(min=t_supply_min, max=t_supply_max), QTHOut.T(min=t_supply_min, max=t_supply_max), theta = theta, sign_dT=1.0);
  WarmingUp.HeatNetwork.Pipe pipe4bH(length = 15, diameter = 0.15, temperature=75.0, Q(nominal=Q_nominal), QTHIn.T(min=t_supply_min, max=t_supply_max), QTHOut.T(min=t_supply_min, max=t_supply_max), theta = theta, sign_dT=1.0);
  WarmingUp.HeatNetwork.Pipe pipe5H(length = 126, diameter = 0.15, temperature=75.0, Q(nominal=Q_nominal), QTHIn.T(min=t_supply_min, max=t_supply_max), QTHOut.T(min=t_supply_min, max=t_supply_max), theta = theta, sign_dT=1.0);
  WarmingUp.HeatNetwork.Pipe pipe7H(length = 60, diameter = 0.15, temperature=75.0, Q(nominal=Q_nominal), QTHIn.T(min=t_supply_min, max=t_supply_max), QTHOut.T(min=t_supply_min, max=t_supply_max), theta = theta, sign_dT=1.0);
  WarmingUp.HeatNetwork.Pipe pipe9H(length = 70, diameter = 0.15, temperature=75.0, Q(nominal=Q_nominal), QTHIn.T(min=t_supply_min, max=t_supply_max), QTHOut.T(min=t_supply_min, max=t_supply_max), theta = theta, sign_dT=1.0);
  WarmingUp.HeatNetwork.Pipe pipe15H(length = 129, diameter = 0.15, temperature=75.0, Q(nominal=Q_nominal), QTHIn.T(min=t_supply_min, max=t_supply_max), QTHOut.T(min=t_supply_min, max=t_supply_max), theta = theta, sign_dT=1.0);
  WarmingUp.HeatNetwork.Pipe pipe25H(length = 150, diameter = 0.15, temperature=75.0, Q(nominal=Q_nominal), QTHIn.T(min=t_supply_min, max=t_supply_max), QTHOut.T(min=t_supply_min, max=t_supply_max), theta = theta, sign_dT=1.0);
  WarmingUp.HeatNetwork.Pipe pipe26H(length = 30, diameter = 0.1, temperature=75.0, Q(nominal=Q_nominal), QTHIn.T(min=t_supply_min, max=t_supply_max), QTHOut.T(min=t_supply_min, max=t_supply_max), theta = theta, sign_dT=1.0);
  WarmingUp.HeatNetwork.Pipe pipe27H(length = 55, diameter = 0.15, temperature=75.0, Q(nominal=Q_nominal), QTHIn.T(min=t_supply_min, max=t_supply_max), QTHOut.T(min=t_supply_min, max=t_supply_max), theta = theta, sign_dT=1.0);
  WarmingUp.HeatNetwork.Pipe pipe29H(length = 134, diameter = 0.15, temperature=75.0, Q(nominal=Q_nominal), QTHIn.T(min=t_supply_min, max=t_supply_max), QTHOut.T(min=t_supply_min, max=t_supply_max), theta = theta, sign_dT=1.0);
  WarmingUp.HeatNetwork.Pipe pipe30H(length = 60, diameter = 0.1, temperature=75.0, Q(nominal=Q_nominal), QTHIn.T(min=t_supply_min, max=t_supply_max), QTHOut.T(min=t_supply_min, max=t_supply_max), theta = theta, sign_dT=1.0);
  WarmingUp.HeatNetwork.Pipe pipe31H(length = 60, diameter = 0.1, temperature=75.0, Q(nominal=Q_nominal), QTHIn.T(min=t_supply_min, max=t_supply_max), QTHOut.T(min=t_supply_min, max=t_supply_max), theta = theta, sign_dT=1.0);
  WarmingUp.HeatNetwork.Pipe pipe32H(length = 50, diameter = 0.1, temperature=75.0, Q(nominal=Q_nominal), QTHIn.T(min=t_supply_min, max=t_supply_max), QTHOut.T(min=t_supply_min, max=t_supply_max), theta = theta, sign_dT=1.0);
  WarmingUp.HeatNetwork.Pipe pipe52_inH(length = 10, diameter = 0.164, temperature=75.0, Q(max=0.023, nominal=Q_nominal), QTHIn.T(min=t_supply_min, max=t_supply_max), QTHOut.T(min=t_supply_min, max=t_supply_max), dH(min=-1.0), theta = 0.0, sign_dT=1.0);
  WarmingUp.HeatNetwork.Pipe pipe52_outH(length = 10, diameter = 0.164, temperature=75.0, Q(max=0.023, nominal=Q_nominal), QTHIn.T(min=t_supply_min, max=t_supply_max), QTHOut.T(min=t_supply_min, max=t_supply_max), dH(min=-1.0), theta = 0.0, sign_dT=1.0);


  WarmingUp.HeatNetwork.NodeQTHPort nodeS2H(nin=2, nout=1, theta = theta, temperature=75.0);
  WarmingUp.HeatNetwork.NodeQTHPort nodeD7H(nin=1, nout=2, theta = theta, temperature=75.0);
  WarmingUp.HeatNetwork.NodeQTHPort nodeD92H(nin=1, nout=2, theta = theta, temperature=75.0);
  WarmingUp.HeatNetwork.NodeQTHPort nodeB1H(nin=2, nout=2, theta = theta, temperature=75.0);

  WarmingUp.HeatNetwork.NodeQTHPort nodeS2C(nin=1, nout=2, theta = theta, temperature=45.0);
  WarmingUp.HeatNetwork.NodeQTHPort nodeD7C(nin=2, nout=1, theta = theta, temperature=45.0);
  WarmingUp.HeatNetwork.NodeQTHPort nodeD92C(nin=2, nout=1, theta = theta, temperature=45.0);
  WarmingUp.HeatNetwork.NodeQTHPort nodeB1C(nin=2, nout=2, theta = theta, temperature=45.0);

  //Q in [m^3/s] and H in [m]
  WarmingUp.HeatNetwork.Pump pump1(Q(min=0.00002778, max=0.01111, nominal=Q_nominal), dH(min=0.2, max=20.0), H(min=0.0, max=0.0));
  WarmingUp.HeatNetwork.Pump pump2(Q(min=0.00002778, max=0.01111, nominal=Q_nominal), dH(min=0.2, max=20.0));

  // Define Input/Output Variables and set them equal to model variables.
  //Heatdemand min en max in [W]
  input Modelica.SIunits.Heat Heat_source1(fixed=false, nominal=4200*988*30) = source1.Heat;
  input Modelica.SIunits.Heat Heat_source2(fixed=false, nominal=4200*988*30) = source2.Heat;

  output Modelica.SIunits.Heat Heat_demand7_opt = demand7.Heat;
  output Modelica.SIunits.Heat Heat_demand91_opt = demand91.Heat;
  output Modelica.SIunits.Heat Heat_demand92_opt = demand92.Heat;
  output Modelica.SIunits.Heat StoredHeat_buffer = buffer1.Stored_heat;

equation
// Connect Model Elements

//Hot lines from source1 to demand91
  connect(source1.QTHOut, pipe1aH.QTHIn) ;
  connect(pipe1aH.QTHOut, pump1.QTHIn) ;
  connect(pump1.QTHOut, pipe1bH.QTHIn) ;
  connect(pipe1bH.QTHOut, nodeS2H.QIn[1]) ;
  connect(nodeS2H.QOut[1], pipe5H.QTHIn) ;
  connect(pipe5H.QTHOut, pipe7H.QTHIn) ;
  connect(pipe7H.QTHOut, pipe9H.QTHIn) ;
  connect(pipe9H.QTHOut, nodeB1H.QIn[1]) ;
  connect(nodeB1H.QOut[1], pipe15H.QTHIn) ;
  connect(pipe15H.QTHOut, pipe25H.QTHIn) ;
  connect(pipe25H.QTHOut, nodeD7H.QIn[1]) ;
  connect(nodeD7H.QOut[1], pipe27H.QTHIn) ;
  connect(pipe27H.QTHOut, pipe29H.QTHIn) ;
  connect(pipe29H.QTHOut, nodeD92H.QIn[1]) ;
  connect(nodeD92H.QOut[1], pipe31H.QTHIn) ;
  connect(pipe31H.QTHOut, pipe32H.QTHIn) ;
  connect(pipe32H.QTHOut, demand91.QTHIn) ;


//Cold lines from demand91 to source 1
  connect(demand91.QTHOut, pipe32C.QTHIn) ;
  connect(pipe32C.QTHOut, pipe31C.QTHIn) ;
  connect(pipe31C.QTHOut, nodeD92C.QIn[1]) ;
  connect(nodeD92C.QOut[1], pipe29C.QTHIn) ;
  connect(pipe29C.QTHOut, pipe27C.QTHIn) ;
  connect(pipe27C.QTHOut, nodeD7C.QIn[1]) ;
  connect(nodeD7C.QOut[1], pipe25C.QTHIn) ;
  connect(pipe25C.QTHOut, pipe15C.QTHIn) ;
  connect(pipe15C.QTHOut, nodeB1C.QIn[1]) ;
  connect(nodeB1C.QOut[1], pipe9C.QTHIn) ;
  connect(pipe9C.QTHOut, pipe7C.QTHIn) ;
  connect(pipe7C.QTHOut, pipe5C.QTHIn) ;
  connect(pipe5C.QTHOut, nodeS2C.QIn[1]) ;
  connect(nodeS2C.QOut[1], pipe1bC.QTHIn) ;
  connect(pipe1bC.QTHOut, pipe1aC.QTHIn) ;
  connect(pipe1aC.QTHOut, source1.QTHIn) ;


//Source2
  connect(source2.QTHOut, pipe4aH.QTHIn) ;
  connect(pipe4aH.QTHOut, pump2.QTHIn) ;
  connect(pump2.QTHOut, pipe4bH.QTHIn) ;
  connect(pipe4bH.QTHOut, nodeS2H.QIn[2]) ;
  connect(nodeS2C.QOut[2], pipe4aC.QTHIn) ;
  connect(pipe4aC.QTHOut, pipe4bC.QTHIn) ;
  connect(pipe4bC.QTHOut, source2.QTHIn) ;

//Demand7
  connect(nodeD7H.QOut[2], pipe26H.QTHIn) ;
  connect(pipe26H.QTHOut, demand7.QTHIn) ;
  connect(demand7.QTHOut, pipe26C.QTHIn) ;
  connect(pipe26C.QTHOut, nodeD7C.QIn[2]) ;

//Demand92
  connect(nodeD92H.QOut[2], pipe30H.QTHIn) ;
  connect(pipe30H.QTHOut, demand92.QTHIn) ;
  connect(demand92.QTHOut, pipe30C.QTHIn) ;
  connect(pipe30C.QTHOut, nodeD92C.QIn[2]) ;

//Buffer1
  //Hot
  connect(nodeB1H.QOut[2], pipe52_inH.QTHIn) ;
  connect(pipe52_inH.QTHOut, buffer1.QTHHotIn) ;
  connect(buffer1.QTHHotOut, pipe52_outH.QTHIn) ;
  connect(pipe52_outH.QTHOut, nodeB1H.QIn[2]) ;
  //Cold
  connect(nodeB1C.QOut[2], pipe52_inC.QTHIn) ;
  connect(pipe52_inC.QTHOut, buffer1.QTHColdIn) ;
  connect(buffer1.QTHColdOut, pipe52_outC.QTHIn) ;
  connect(pipe52_outC.QTHOut, nodeB1C.QIn[2]) ;

end Example;
