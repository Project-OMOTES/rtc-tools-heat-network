block Pipe
  import SI = Modelica.SIunits;
  extends QTHTwoPort;
  parameter Real length = 1.0;
  parameter Real diameter = 1.0;
  parameter Real temperature = 50.0;
  parameter Real cp = 4200.0;

  // TODO: add the parameters R_g/R_iso etc (or U1/U2). plus cp and rho

  // e.g. parameter Real dT = 30.0;
  // TODO: in the example.mo file we can modify cp with the value that depends on the nominal temperature. 
  // I.e., No assumption that is always 4200 anymore. This can be done also for the other components.
  Modelica.SIunits.VolumeFlowRate Q(min=0.0);
  Modelica.SIunits.Level dH(max=0.0);
  // TODO: add U1, U2 (they are variables depending on the Rs)
  // e.g.
  // Real U_1;

  // Homotopic parameter
  parameter Real theta;

equation
  QTHOut.Q = Q;
  QTHOut.Q = QTHIn.Q;
  dH = QTHOut.H - QTHIn.H;

  // TODO: remove the following once the heat loss equation is implemented
  QTHOut.T = QTHIn.T;


  // Heat Loss equation
  // TODO: vary cp depending whether is hot/cold pipe?

  // High level idea
  // HeatLoss = l*(U1-U2)*(T_h - T_g)+ L*(U2)*(T_h - T_c)
  // Assumption: dT = T_h - T_c

  // HeatLoss = length*(U_1-U_2)*(QTHIn.T - QTHOut.T)/2 + constant (TODO:specify constant through the correct dependencies)
  // Heat_out = Heat_in - HeatLoss
  // Heat = T * Q * cp * rho
  // (T_out-T_in)*cp*rho*Q + length*(U_1-U_2)*(T_in - T_out)/2 + constant = 0

  // Equations to be added:
  // U_1 = dependency on R...
  // U_2 = 
  // Homotopy HeatLoss equation: (1- theta)*Linear_approximation + theta*(Nonlinear_function) = 0.0
  // (1-theta)*(QTHOut.T - QTHIn.T) + theta*(T_out*cp*rho*Q - T_in*cp*rho*Q + length*(U_1*U_2)*(T_in - T_out)/2 + constant)= 0.0

  // TODO: Teresa add if else statement to flip the sign of dT

 end Pipe;
