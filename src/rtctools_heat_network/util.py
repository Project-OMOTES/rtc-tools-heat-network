from rtctools.optimization.timeseries import Timeseries
from rtctools.util import run_optimization_problem


def run_heat_network_optimization(heat_class, qht_class, *args, **kwargs):
    heat_problem = run_optimization_problem(heat_class, *args, **kwargs)
    results = heat_problem.extract_results()
    times = heat_problem.times()

    directions = {}

    hot_pipes = [p for p in heat_problem.heat_network_components["pipe"] if p.endswith("_hot")]

    for p in hot_pipes:
        heat_in = results[p + ".HeatIn.Heat"]
        heat_out = results[p + ".HeatOut.Heat"]
        directions[p] = Timeseries(
            times, ((heat_in >= 0.0) & (heat_out >= 0.0)).astype(int) * 2 - 1
        )

        # NOTE: The assumption is that the orientation of the cold pipes is such that the flow
        # is always in the same direction as its "hot" pipe companion.
        cold_pipe = f"{p[:-4]}_cold"
        directions[cold_pipe] = directions[p]

    qth_problem = run_optimization_problem(qht_class, *args, flow_directions=directions, **kwargs)

    return heat_problem, qth_problem
