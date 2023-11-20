from rtctools.util import run_optimization_problem

from rtctools_heat_network.workflows import EndScenarioSizingStaged
if __name__ == "__main__":
    import time

    start_time = time.time()

    solution = run_optimization_problem(EndScenarioSizingStaged, stage=1)
    results = solution.extract_results()
    boolean_bounds = {}
    # We give bounds for stage 2 by allowing up to 2 DN sizes larger than what was found in the
    # stage 1 optimization.
    pc_map = solution._HeatMixin__pipe_topo_pipe_class_map
    for pipe_classes in pc_map.values():
        v_prev = 0.0
        for var_name in pipe_classes.values():
            v = results[var_name][0]
            boolean_bounds[var_name] = (v, v)
            if v_prev == 1.0:
                boolean_bounds[var_name] = (0.0, 1.0)
            v_prev = v

    # Run a full horizon optimization to find the optimal sizes and placement variables
    problem_s2 = run_optimization_problem(
        EndScenarioSizingStaged,
        stage=2,
        boolean_bounds=boolean_bounds,
    )

    print("Execution time: " + time.strftime("%M:%S", time.gmtime(time.time() - start_time)))
