within WarmingUp.HeatNetwork;

block Pipe
  import SI = Modelica.SIunits;
  extends QTHTwoPort;
  parameter Real length = 1.0;
  parameter Real diameter = 1.0;
  parameter Real temperature = 50.0;
  parameter Real cp = 4200.0;
  parameter Real rho = 988.0;

  //For a PUR-PE pipe estimated based on 1m deep 150mm pipe with 75mm PUR and 15mm PE and distance of 2x centre to centre
  parameter Real U_1 = 0.397;
  parameter Real U_2 = 0.0185;
  parameter Real T_supply = 75.0;
  parameter Real T_return = 45.0;
  parameter Real dT = T_supply - T_return;
  // T_g = ground temperature
  parameter Real T_g = 10.0;
  // sign_dT = 1.0 if supply pipe, else sign_dT = -1.0
  parameter Real sign_dT = 1.0;

  // Homotopic parameter
  parameter Real theta;

  Modelica.SIunits.VolumeFlowRate Q(min=0.0);
  Modelica.SIunits.Level dH(max=0.0);

equation
  // Flow is constant
  QTHOut.Q = Q;
  QTHOut.Q = QTHIn.Q;
  dH = QTHOut.H - QTHIn.H;

  // Heat loss estimate assuming constant ground temparature and constant dT at demand
  //positive negative dT depending on hot/coldpipe
  (1-theta)*(QTHOut.T - QTHIn.T) + theta*(QTHOut.T*cp*rho*Q - QTHIn.T*cp*rho*Q + length*(U_1-U_2)*(QTHIn.T + QTHOut.T)/2 -(length*(U_1-U_2)*T_g)+(length*U_2*(sign_dT*dT))) = 0.0;

 end Pipe;
