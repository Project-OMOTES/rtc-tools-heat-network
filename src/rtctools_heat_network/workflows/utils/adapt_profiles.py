import datetime
import logging

import numpy as np

from rtctools.data.storage import DataStore

logger = logging.getLogger("WarmingUP-MPC")
logger.setLevel(logging.INFO)


def adapt_hourly_profile_to_common(problem, problem_day_steps):
    """
    Adapt yearly porifle with hourly time steps to a common profile (daily averaged profile except
    for the day with the peak demand).

    Return the following:
        - problem_indx_max_peak: index of the maximum of the peak values
        - heat_demand_nominal: max demand value found for a specific heating demand
    """

    demands = problem.heat_network_components.get("demand", [])
    new_datastore = DataStore(problem)
    new_datastore.reference_datetime = problem.io.datetimes[0]

    for ensemble_member in range(problem.ensemble_size):
        problem_indx_max_peak = -1.0
        day_steps = -1.0
        parameters = problem.parameters(ensemble_member)
        total_demand = None
        heat_demand_nominal = dict()

        for demand in demands:
            try:
                demand_values = problem.get_timeseries(
                    f"{demand}.target_heat_demand", ensemble_member
                ).values
            except KeyError:
                continue
            if total_demand is None:
                total_demand = demand_values
            else:
                total_demand += demand_values
            heat_demand_nominal[f"{demand}.Heat_demand"] = max(demand_values)

        # TODO: the approach of picking one peak day was introduced for a network with a tree
        #  layout and all big sources situated at the root of the tree. It is not guaranteed
        #  that an optimal solution is reached in different network topologies.
        idx_max = int(np.argmax(total_demand))
        max_day = idx_max // 24
        nr_of_days = len(total_demand) // 24
        new_date_times = list()
        day_steps = problem_day_steps

        problem_indx_max_peak = max_day // day_steps
        if max_day % day_steps > 0:
            problem_indx_max_peak += 1.0

        for day in range(0, nr_of_days, day_steps):
            if day == max_day // day_steps * day_steps:
                if max_day > day:
                    new_date_times.append(problem.io.datetimes[day * 24])
                new_date_times.extend(problem.io.datetimes[max_day * 24 : max_day * 24 + 24])
                if (day + day_steps - 1) > max_day:
                    new_date_times.append(problem.io.datetimes[max_day * 24 + 24])
            else:
                new_date_times.append(problem.io.datetimes[day * 24])
        new_date_times.append(problem.io.datetimes[-1] + datetime.timedelta(hours=1))

        new_date_times = np.asarray(new_date_times)
        parameters["times"] = [x.timestamp() for x in new_date_times]

        for demand in demands:
            var_name = f"{demand}.target_heat_demand"
            problem._set_data_with_averages_and_peak_day(
                datastore=new_datastore,
                variable_name=var_name,
                ensemble_member=ensemble_member,
                new_date_times=new_date_times,
            )

        # TODO: this has not been tested but is required if a production profile is included
        #  in the data
        for source in problem.heat_network_components.get("source", []):
            try:
                problem.get_timeseries(f"{source}.target_heat_source", ensemble_member)
            except KeyError:
                logger.debug(
                    f"{source} has no production profile, skipping setting the "
                    f"production profile"
                )
                continue
            var_name = f"{source}.target_heat_source"
            problem._set_data_with_averages_and_peak_day(
                datastore=new_datastore,
                variable_name=var_name,
                ensemble_member=ensemble_member,
                new_date_times=new_date_times,
            )

    problem.io = new_datastore

    logger.info("Profile data has been adapted to a common format")

    return problem_indx_max_peak, heat_demand_nominal
