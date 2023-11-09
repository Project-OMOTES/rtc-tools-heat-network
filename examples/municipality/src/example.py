from rtctools.util import run_optimization_problem

from rtctools_heat_network.workflows import EndScenarioSizing

if __name__ == "__main__":
    import time

    start_time = time.time()

    heat_problem = run_optimization_problem(EndScenarioSizing)
    results = heat_problem.extract_results()

    print("Execution time: " + time.strftime("%M:%S", time.gmtime(time.time() - start_time)))
