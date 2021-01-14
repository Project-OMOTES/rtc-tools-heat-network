model DoublePipeBase

  parameter Real Q_nominal = 0.001;
  parameter Real theta;

  parameter Real t_demand_min = 70.0;

  WarmingUp.HeatNetwork.QTH.Source source(
    theta=theta,
    Heat_source(min=0.0, max=1.25e6, nominal=1e6),
    Q_nominal=Q_nominal,
    T_supply=75.0,
    T_return=45.0,
    QTHOut.T(min=65.0, max=90.0)
  );

  WarmingUp.HeatNetwork.QTH.Pipe pipe_source_hot(
    length=100.0,
    diameter=0.15,
    temperature=75.0,
    T_supply=75.0,
    T_return=45.0,
    T_g=10.0,
    Q(min=0.0, nominal=Q_nominal),
    QTHIn.T(min=10.0, max=90.0),
    QTHOut.T(min=10.0, max=90.0),
    sign_dT=1.0
  );

  WarmingUp.HeatNetwork.QTH.Node node_source_hot(
    n=3,
    temperature=75.0
  );

  WarmingUp.HeatNetwork.QTH.Pipe pipe_1_hot(
    length=1000.0,
    diameter=0.15,
    temperature=75.0,
    T_supply=75.0,
    T_return=45.0,
    T_g=10.0,
    Q(min=0.0, nominal=Q_nominal),
    QTHIn.T(min=10.0, max=90.0),
    QTHOut.T(min=10.0, max=90.0),
    sign_dT=1.0
  );

  WarmingUp.HeatNetwork.QTH.Pipe pipe_2_hot(
    length=1000.0,
    diameter=0.15,
    temperature=75.0,
    T_supply=75.0,
    T_return=45.0,
    T_g=10.0,
    Q(min=0.0, nominal=Q_nominal),
    QTHIn.T(min=10.0, max=90.0),
    QTHOut.T(min=10.0, max=90.0),
    sign_dT=1.0
  );

  WarmingUp.HeatNetwork.QTH.Node node_demand_hot(
    n=3,
    temperature=75.0
  );

  WarmingUp.HeatNetwork.QTH.Pipe pipe_demand_hot(
    length=100.0,
    diameter=0.15,
    temperature=75.0,
    T_supply=75.0,
    T_return=45.0,
    T_g=10.0,
    Q(min=0.0, nominal=Q_nominal),
    QTHIn.T(min=10.0, max=90.0),
    QTHOut.T(min=10.0, max=90.0),
    sign_dT=1.0
  );

  WarmingUp.HeatNetwork.QTH.Demand demand(
    theta=theta,
    Q_nominal=Q_nominal,
    T_supply=75.0,
    T_return=45.0,
    QTHIn.T(min=t_demand_min)
  );

  WarmingUp.HeatNetwork.QTH.Pipe pipe_demand_cold(
    length=100.0,
    diameter=0.15,
    temperature=45.0,
    T_supply=75.0,
    T_return=45.0,
    T_g=10.0,
    Q(min=0.0, nominal=Q_nominal),
    QTHIn.T(min=10.0, max=90.0),
    QTHOut.T(min=10.0, max=90.0),
    sign_dT=-1.0
  );

  WarmingUp.HeatNetwork.QTH.Node node_demand_cold(
    n=3,
    temperature=45.0
  );

  WarmingUp.HeatNetwork.QTH.Pipe pipe_1_cold(
    length=1000.0,
    diameter=0.15,
    temperature=45.0,
    T_supply=75.0,
    T_return=45.0,
    T_g=10.0,
    Q(min=0.0, nominal=Q_nominal),
    QTHIn.T(min=10.0, max=90.0),
    QTHOut.T(min=10.0, max=90.0),
    sign_dT=-1.0
  );

  WarmingUp.HeatNetwork.QTH.Pipe pipe_2_cold(
    length=1000.0,
    diameter=0.15,
    temperature=45.0,
    T_supply=75.0,
    T_return=45.0,
    T_g=10.0,
    Q(min=0.0, nominal=Q_nominal),
    QTHIn.T(min=10.0, max=90.0),
    QTHOut.T(min=10.0, max=90.0),
    sign_dT=-1.0
  );

  WarmingUp.HeatNetwork.QTH.Node node_source_cold(
    n=3,
    temperature=45.0
  );

  WarmingUp.HeatNetwork.QTH.Pipe pipe_source_cold(
    length=100.0,
    diameter=0.15,
    temperature=45.0,
    T_supply=75.0,
    T_return=45.0,
    T_g=10.0,
    Q(min=0.0, nominal=Q_nominal),
    QTHIn.T(min=10.0, max=90.0),
    QTHOut.T(min=10.0, max=90.0),
    sign_dT=-1.0
  );

  WarmingUp.HeatNetwork.QTH.Pump pump(
    Q(nominal=Q_nominal),
    dH(min=0.0, max=50.0),
    QTHIn.H(min=0.0, max=0.0)
  );

   input Modelica.SIunits.Heat Heat_source(fixed=false, nominal=1e6) = source.Heat_source;
equation
  connect(source.QTHOut, pipe_source_hot.QTHIn);
  connect(pipe_source_hot.QTHOut, node_source_hot.QTHConn[1]);
  connect(node_source_hot.QTHConn[2], pipe_1_hot.QTHIn);
  connect(node_source_hot.QTHConn[3], pipe_2_hot.QTHIn);

  connect(pipe_1_hot.QTHOut, node_demand_hot.QTHConn[2]);
  connect(pipe_2_hot.QTHOut, node_demand_hot.QTHConn[3]);
  connect(node_demand_hot.QTHConn[1], pipe_demand_hot.QTHIn);
  connect(pipe_demand_hot.QTHOut, demand.QTHIn);

  connect(demand.QTHOut, pipe_demand_cold.QTHIn);
  connect(pipe_demand_cold.QTHOut, node_demand_cold.QTHConn[1]);
  connect(node_demand_cold.QTHConn[2], pipe_1_cold.QTHIn);
  connect(node_demand_cold.QTHConn[3], pipe_2_cold.QTHIn);

  connect(pipe_1_cold.QTHOut, node_source_cold.QTHConn[2]);
  connect(pipe_2_cold.QTHOut, node_source_cold.QTHConn[3]);
  connect(node_source_cold.QTHConn[1], pipe_source_cold.QTHIn);
  connect(pipe_source_cold.QTHOut, pump.QTHIn);
  connect(pump.QTHOut, source.QTHIn);
end DoublePipeBase;
