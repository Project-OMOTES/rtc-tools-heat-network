model SinglePipeQTH

  parameter Real Q_nominal = 0.001;
  parameter Real theta;

  parameter Real t_demand_min = 70.0;

  WarmingUp.HeatNetwork.QTH.Source source(
    theta=theta,
    Heat_source(min=0.75e6, max=1.25e6, nominal=1e6),
    Q_nominal=Q_nominal,
    T_supply=75.0,
    T_return=45.0,
    QTHOut.T(min=65.0, max=90.0)
  );

  WarmingUp.HeatNetwork.QTH.Pipe pipe_hot(
    length=1000.0,
    diameter=0.15,
    temperature=75.0,
    T_supply=75.0,
    T_return=45.0,
    Q(min=0.0, nominal=Q_nominal),
    QTHIn.T(min=10.0, max=90.0),
    QTHOut.T(min=10.0, max=90.0)
  );

  WarmingUp.HeatNetwork.QTH.Demand demand(
    theta=theta,
    Q_nominal=Q_nominal,
    T_supply=75.0,
    T_return=45.0,
    QTHIn.T(min=t_demand_min)
  );


  WarmingUp.HeatNetwork.QTH.Pipe pipe_cold(
    length=1000.0,
    diameter=0.15,
    temperature=45.0,
    T_supply=75.0,
    T_return=45.0,
    Q(min=0.0, nominal=Q_nominal),
    QTHIn.T(min=10.0, max=90.0),
    QTHOut.T(min=10.0, max=90.0)
  );

  WarmingUp.HeatNetwork.QTH.Pump pump(
    Q(nominal=Q_nominal),
    dH(min=0.0, max=50.0),
    QTHIn.H(min=0.0, max=0.0)
  );

   input Modelica.SIunits.Heat Heat_source(fixed=false, nominal=1e6) = source.Heat_source;
equation
  connect(source.QTHOut, pipe_hot.QTHIn);
  connect(pipe_hot.QTHOut, demand.QTHIn);
  connect(demand.QTHOut, pipe_cold.QTHIn);
  connect(pipe_cold.QTHOut, pump.QTHIn);
  connect(pump.QTHOut, source.QTHIn);
end SinglePipeQTH;
