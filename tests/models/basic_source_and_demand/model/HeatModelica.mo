model HeatModelica
  WarmingUp.HeatNetwork.Heat.Source source(
    Heat_source(min=0.75e5, max=1.25e5, nominal=1e5),
    Heat_out(max=2e5),
    T_supply=75.0,
    T_return=45.0
  );
  WarmingUp.HeatNetwork.Heat.Demand demand(
    Heat_in(max=2e5),
    T_supply=75.0,
    T_return=45.0
  );
  WarmingUp.HeatNetwork.Heat.Pipe pipe_hot(
    length=1000.0,
    diameter=0.15,
    temperature=75.0,
    HeatIn.Heat(min=-2e5, max=2e5, nominal=1e5),
    HeatOut.Heat(nominal=1e5),
    T_supply=75.0,
    T_return=45.0
  );
  WarmingUp.HeatNetwork.Heat.Pipe pipe_cold(
    length=1000.0,
    diameter=0.15,
    temperature=45.0,
    HeatOut.Heat(nominal=1e5),
    HeatIn.Heat(max=2e5, nominal=1e5),
    T_supply=75.0,
    T_return=45.0
  );

   input Modelica.SIunits.Heat Heat_source(fixed=false, nominal=1e5) = source.Heat_source;
equation
  connect(source.HeatOut, pipe_hot.HeatIn);
  connect(pipe_hot.HeatOut, demand.HeatIn);
  connect(demand.HeatOut, pipe_cold.HeatIn);
  connect(pipe_cold.HeatOut, source.HeatIn);
end HeatModelica;
