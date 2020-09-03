within WarmingUp.HeatNetwork.Heat;

block Node
  import SI = Modelica.SIunits;
  parameter String component_type = "node";

  replaceable parameter Integer n(min = 2) = 2 "Number of flows";

  HeatPort HeatConn[n];
equation
  // Because the orientation of the connected pipes are important to setup the
  // heat conservation, these constraints are added in Python.
end Node;
