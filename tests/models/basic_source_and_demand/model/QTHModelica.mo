model QTHModelica
  parameter Real Q_nominal = 0.001;
  parameter Real theta;

  parameter Real t_supply_max = 110.0;
  parameter Real t_supply_min = 10.0;
  parameter Real t_return_max = 110.0;
  parameter Real t_return_min = 10.0;

  parameter Real t_source_min = 65.0;
  parameter Real t_source_max = 85.0;
  parameter Real t_demand_min = 70.0;

  WarmingUp.HeatNetwork.QTH.Source source(
    Heat_source(min=0.75e5, max=1.25e5, nominal=1e5),
    theta=theta,
    QTHOut.T(min=t_source_min, max=t_source_max),
    Q_nominal=Q_nominal,
    T_supply=75.0,
    T_return=45.0
  );
  WarmingUp.HeatNetwork.QTH.Demand demand(
    theta=theta,
    QTHIn.T(min=t_demand_min),
    Q_nominal=Q_nominal,
    T_supply=75.0,
    T_return=45.0
  );
  WarmingUp.HeatNetwork.QTH.Pipe pipe_hot(
    length=1000.0,
    diameter=0.15,
    temperature=75.0,
    Q(min=0.0, nominal=Q_nominal),
    QTHIn.T(min=t_supply_min, max=t_supply_max),
    QTHOut.T(min=t_supply_min, max=t_supply_max),
    T_supply=75.0,
    T_return=45.0
  );
  WarmingUp.HeatNetwork.QTH.Pipe pipe_cold(
    length=1000.0,
    diameter=0.15,
    temperature=45.0,
    Q(min=0.0, nominal=Q_nominal),
    QTHIn.T(min=t_return_min, max=t_return_max),
    QTHOut.T(min=t_return_min, max=t_return_max),
    T_supply=75.0,
    T_return=45.0
  );
  WarmingUp.HeatNetwork.QTH.Pump pump(
    Q(min=0.0, nominal=Q_nominal),
    QTHIn.H(min=0.0, max=0.0)
  );

   input Modelica.SIunits.Heat Heat_source(fixed=false) = source.Heat_source;
equation
  connect(source.QTHOut, pipe_hot.QTHIn);
  connect(pipe_hot.QTHOut, demand.QTHIn);
  connect(demand.QTHOut, pipe_cold.QTHIn);
  connect(pipe_cold.QTHOut, pump.QTHIn);
  connect(pump.QTHOut, source.QTHIn);
end QTHModelica;
