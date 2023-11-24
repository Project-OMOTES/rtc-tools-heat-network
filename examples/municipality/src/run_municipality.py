from rtctools_heat_network.workflows import EndScenarioSizingStaged, run_end_scenario_sizing

if __name__ == "__main__":
    import time

    start_time = time.time()

    solution = run_end_scenario_sizing(EndScenarioSizingStaged)

    print("Execution time: " + time.strftime("%M:%S", time.gmtime(time.time() - start_time)))
