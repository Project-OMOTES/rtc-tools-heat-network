within WarmingUp.HeatNetwork.Heat;

model _NonStorageComponent
  extends HeatTwoPort;

  parameter Real Q_nominal = 1.0;
  parameter Real T_supply;
  parameter Real T_return;
  parameter Real dT = T_supply - T_return;
  parameter Real cp = 4200.0;
  parameter Real rho = 988.0;

  // NOTE: We move a factor of 100.0 of the heat to the state entry, to
  // reduce the coefficient in front of the heat variables. This
  // particularly helps the scaling/range of the constraints that relate
  // the heat loss (if it is variable/optional) to the heat in- and out
  // of a component.
  parameter Real Heat_nominal = cp * rho * dT * Q_nominal / 100.0;

  Modelica.SIunits.Heat Heat_in(nominal=Heat_nominal);
  Modelica.SIunits.Heat Heat_out(nominal=Heat_nominal);

  Modelica.SIunits.VolumeFlowRate Q(nominal=Q_nominal);

  Modelica.SIunits.Level H_in;
  Modelica.SIunits.Level H_out;
equation
  (Heat_out - HeatOut.Heat)/Heat_nominal = 0.0;
  (Heat_in - HeatIn.Heat)/Heat_nominal = 0.0;

  HeatIn.Q = Q;
  HeatIn.Q = HeatOut.Q;

  HeatIn.H = H_in;
  HeatOut.H = H_out;
end _NonStorageComponent;
