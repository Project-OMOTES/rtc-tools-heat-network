within WarmingUp.HeatNetwork.Heat;

block GeothermalSource
  extends Source;

  parameter String component_subtype = "geothermal";

  parameter Real target_flow_rate;
end GeothermalSource;
