model DoublePipeUnequalWithValveQTH
  extends DoublePipeUnequalQTH(
    pipe_2_hot.has_control_valve=true,
    pipe_2_cold.has_control_valve=true
  );
end DoublePipeUnequalWithValveQTH;
