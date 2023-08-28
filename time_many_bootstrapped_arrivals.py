import datetime
from argparse import ArgumentParser
from random import choice
from time import time

from compare_matching_algorithms import setup
from optimization.capacities import ConstantCapacities
from optimization.matching import KindOfDual, get_duals, DynamicDualPriceAlgorithm, ConstantCasePool, \
    TrustCapacityTrajectoryLengths


def get_command_line_args():
    parser = ArgumentParser()
    parser.add_argument("fy", type=int, help="Fiscal year to evaluate on.")
    parser.add_argument("potential", type=int, help="Which potential to choose (1 or 2).")
    parser.add_argument("iterations", type=int, help="How many iterations to average runtime over.")
    parser.add_argument("--numarrivalslist", type=int, nargs="+", help="Number of arriving refugees.")
    args = parser.parse_args()
    fiscal_year = args.fy
    potential = args.potential
    iterations = args.iterations
    num_arrivals_list = args.numarrivalslist

    min_arrival_date = datetime.date(fiscal_year - 1, 10, 1)
    max_arrival_date = datetime.date(fiscal_year, 10, 1)

    assert potential in (1, 2)
    if potential == 1:
        which_dual = KindOfDual.maximal
    else:
        which_dual = KindOfDual.minimal

    return min_arrival_date, max_arrival_date, iterations, which_dual, num_arrivals_list


def time_algorithm(min_arrival_date, max_arrival_date, iterations, which_dual, num_arrivals_list):
    experiment_setup = setup(min_arrival_date, max_arrival_date, caps_mode=0, reverse_time=False, unit_cases=False,
                             batching=True)
    capacities: ConstantCapacities
    localities, _, _, range_dates, range_batches, capacities, reference_avg_case_size = experiment_setup
    case_pool = ConstantCasePool([case for batch in range_batches for case in batch])
    num_historic_arrivals = sum(case.size for case in case_pool.pool)

    timings = {}
    for trajectory_length in num_arrivals_list:
        timings[trajectory_length] = []
        print(trajectory_length)
        scaled_capacities = ConstantCapacities({loc: round(trajectory_length / num_historic_arrivals * cap)
                                                for loc, cap in capacities.capacities.items()})
        for i in range(iterations):
            traj_handler = TrustCapacityTrajectoryLengths(reference_avg_case_size)
            alg = DynamicDualPriceAlgorithm(localities, case_pool, which_dual == KindOfDual.minimal, which_dual,
                                            traj_handler, scaled_capacities, 5, 1.)

            batch_size = 0
            test_batch = []
            while batch_size < trajectory_length / 52:
                new_case = choice(case_pool.pool)
                test_batch.append(new_case)
                batch_size += new_case.size

            start = time()
            alg.allocate_batch(test_batch, scaled_capacities.capacities, range_dates[0])
            duration = time() - start
            print("\t", duration)
            timings[trajectory_length].append(duration)

    average_timings = {traj_length: sum(durations) / len(durations) for traj_length, durations in timings.items()}
    return average_timings


if __name__ == "__main__":
    min_arrival_date, max_arrival_date, iterations, which_dual, num_arrivals_list = get_command_line_args()
    average_timings = time_algorithm(min_arrival_date, max_arrival_date, iterations, which_dual, num_arrivals_list)
    print(average_timings)