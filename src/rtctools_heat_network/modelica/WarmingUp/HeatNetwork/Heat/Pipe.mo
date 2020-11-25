within WarmingUp.HeatNetwork.Heat;

block Pipe
  import SI = Modelica.SIunits;
  extends HeatTwoPort;
  parameter String component_type = "pipe";
  parameter Boolean disconnectable = false;

  parameter Real length = 1.0;
  parameter Real diameter = 1.0;
  parameter Real temperature;
  parameter Real cp = 4200.0;
  parameter Real rho = 988.0;
  parameter Real Heat_nominal = cp * rho * dT;

  //For a PUR-PE pipe estimated based on 1m deep 150mm pipe with 75mm PUR and 15mm PE and distance of 2x centre to centre
  parameter Real U_1 = 0.397;
  parameter Real U_2 = 0.0185;
  parameter Real T_supply = 75.0;
  parameter Real T_return = 45.0;
  parameter Real dT = T_supply - T_return;
  // T_g = ground temperature
  parameter Real T_g = 10.0;

  Modelica.SIunits.Heat Heat_in(nominal=cp * rho * dT);
  Modelica.SIunits.Heat Heat_out(nominal=cp * rho * dT);

  // Heat loss
  // To be calculated in RTC-Tools, or else the parameter would disappear
  parameter Real Heat_loss;
equation
  // Aliases
  (Heat_out - HeatOut.Heat)/Heat_nominal = 0.0;
  (Heat_in - HeatIn.Heat)/Heat_nominal = 0.0;

  // Note: Heat loss is added in Python, because it depends on the flow direction
  // * heat loss equation: (HeatOut.Heat - (HeatIn.Heat +/- Heat_loss)) = 0.0

end Pipe;
