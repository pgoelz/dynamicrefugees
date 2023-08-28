import os
from argparse import ArgumentParser
from pickle import dump
from random import choice
import re
from typing import Tuple

import numpy as np
import pandas as pd

from annie_ml.dataclean import get_df, historic_data_to_agents
from optimization.capacities import Capacities, get_fy_caps, ConstantCapacities, CapacitiesDynamic
from optimization.cases import real_year_cases
from optimization.matching import *


def trace_dynamic_algorithm(locality_names: List[str], dates: List[datetime.date], batches: List[List[Case]],
                            capacities: Capacities, algorithm: PriceAlgorithm, ignore_checks=False) \
        -> Tuple[float, int, int, Dict[str, List]]:

    employment = 0.
    count_matched = 0
    count_unmatched = 0
    count_refugees = 0
    capacities_used = {loc: 0 for loc in locality_names}
    num_remaining_cases = sum(len(batch) for batch in batches)

    trace = {}
    final_capacities = capacities.get_capacities_at_date(dates[-1])
    hindsight_prices = get_duals(locality_names, [case for batch in batches for case in batch], final_capacities,
                                 KindOfDual.minimal)

    for date, batch in zip(dates, batches):
        num_remaining_cases -= len(batch)
        count_refugees += sum(case.size for case in batch)

        current_capacities = capacities.get_capacities_at_date(date)
        remaining_capacities = {loc: max(0, current_capacities[loc] - capacities_used[loc]) for loc in locality_names}

        allocation = algorithm.allocate_batch(batch, remaining_capacities, date)
        assert len(allocation) == len(batch)

        for case, alloc in zip(batch, allocation):
            compat = [loc for loc in locality_names if case.is_compatible(loc)]
            cango = [loc for loc in compat if remaining_capacities[loc] >= case.size]
            trace_entry = {"case size": case.size, "date": date, "allocated where": alloc,
                           "% capacity free": sum(remaining_capacities.values()) / sum(current_capacities.values()),
                           "num compatible affiliates": len(compat),
                           "num compatible affilitates with space": len(cango),
                           "priced capacity": sum(num * hindsight_prices[loc] for loc, num in remaining_capacities.items())}

            if alloc is not None:
                if not ignore_checks:
                    assert case.is_compatible(alloc)
                    assert remaining_capacities[alloc] >= case.size
                capacities_used[alloc] += case.size
                remaining_capacities[alloc] += case.size  # only used for stats of later cases in the batch
                employment += case.expected_employment[alloc]
                count_matched += case.size
                trace_entry["employment"] = case.expected_employment[alloc]
                # numpy just turns division by zero into nan
                trace_entry["employment % of max"] = case.expected_employment[alloc] / max(case.expected_employment.values())
            else:
                count_unmatched += case.size
                trace_entry["employment"] = 0.
                trace_entry["employment % of max"] = 0.

            for te_key, te_value in trace_entry.items():
                if te_key not in trace:
                    trace[te_key] = []
                trace[te_key].append(te_value)

    return employment, count_matched, count_unmatched, trace


def get_stats(min_arrival_date: datetime.date, max_arrival_date: datetime.date,
              min_allocation_date: Optional[datetime.date], max_allocation_date: Optional[datetime.date],
              only_free: bool) -> Tuple[Dict[str, int], float]:
    df = historic_data_to_agents("~/HIAS_FY06-FY20.csv", min_allocation_date, max_allocation_date,
                                 min_arrival_date=min_arrival_date - (max_arrival_date - min_arrival_date),
                                 max_arrival_date=max_arrival_date, only_free=only_free)
    df_current = df[(min_arrival_date <= df["arrival date"].dt.date) & (df["arrival date"].dt.date < max_arrival_date)]
    arrivals_by_affiliate = df_current.groupby('historic affiliate', sort=True)['seq'].count().to_dict()
    assert len(arrivals_by_affiliate) >= 1
    df_previous = df[(min_arrival_date - (max_arrival_date - min_arrival_date) <= df["arrival date"].dt.date) & (df["arrival date"].dt.date < min_arrival_date)]
    previous_avg_case_size = len(df_previous) / df_previous["case number"].nunique()

    return arrivals_by_affiliate, previous_avg_case_size


def split_cases_into_individuals(batches: List[List[CaseWithArrival]]) -> List[List[CaseWithArrival]]:
    new_batches = []
    for old_batch in batches:
        new_batch = []
        for case in old_batch:
            size = case.size
            mini_case = CaseWithArrival(1, {loc: emp / size for loc, emp in case.expected_employment.items()},
                                        case.ties_to, case.nationality_code, case.single_parent,
                                        case.language_codes, case.arrival_date, case.historic_placement)
            for _ in range(size):
                new_batch.append(mini_case)
        new_batches.append(new_batch)
    return new_batches


def split_batches_into_cases(dates: List[datetime.date], batches: List[List[CaseWithArrival]]) -> \
        Tuple[List[datetime.date], List[List[CaseWithArrival]]]:
    new_dates = []
    new_batches = []
    for date, old_batch in zip(dates, batches):
        for case in old_batch:
            new_dates.append(date)
            new_batches.append([case])
    return new_dates, new_batches


def get_command_line_args():
    parser = ArgumentParser()
    parser.add_argument("iterations", type=int, help="How many iterations to run of each randomized algorithm.")
    parser.add_argument("--fy", type=int, help="Fiscal year to evaluate on.")
    parser.add_argument("--capsmode", type=int, default=0,
                        help="Where to take capacities from. 0: capacities are 110%% of arrivals, 1: capacities are the "
                             "official capacities announced at the start of the fiscal year, 2: same as 1 but the "
                             "algorithm does not know the number of official arrivals and must rely on the official "
                             "capacity, 3: same as 2 but incorporating historic revisions to official capacities")
    parser.add_argument("--customrange", type=str, help="Alternatively to providing a fiscal year, a custom range of "
                                                        "dates in format YYYYMMDD-YYYYMMDD (end day exclusive). Only "
                                                        "compatible with capsmode 0.")
    parser.add_argument("--reverse", action="store_true", help="Reverse the allocation order of refugee batches "
                                                               "(allocation dates will be fictitious).")
    parser.add_argument("--bootstrap", type=float, help="Don't use real arrivals but bootstrap over specified arrival "
                                                        "interval. Argument says what fraction of sum of caps arrives "
                                                        "(counted in refugees).")
    parser.add_argument("trace", type=str, help="Save traces of allocations to file.")
    parser.add_argument("--nobatching", action="store_true", help="Treat cases as arriving one-by-one.")
    parser.add_argument("--split", action="store_true", help="Split cases into single-refugee cases.")
    parser.add_argument("--lookback", type=int, help="The length of the period of past arrivals from which dual "
                                                     "algorithms sample cases, in days (default is 183).")
    parser.add_argument("-k", type=int, default=5, help="How many traces to compute the duals?")

    args = parser.parse_args()

    fiscal_year = args.fy
    custom_range = args.customrange
    if fiscal_year is None and custom_range is None:
        raise ValueError("Must specify either --fy or --customrange.")
    elif fiscal_year is not None and custom_range is not None:
        raise ValueError("Cannot specify both --fy and --customrange.")
    elif custom_range is not None:
        if args.capsmode != 0:
            raise ValueError("Custom date ranges are only compatible with caps mode 0.")
        if args.capsmode >= 2 and args.bootstrap is not None:
            raise ValueError("Bootstrapping is only compatible with caps modes 0 and 1.")
        range_match = re.match(r"(\d\d\d\d)(\d\d)(\d\d)-(\d\d\d\d)(\d\d)(\d\d)", custom_range)
        if range_match is None:
            raise ValueError("Custom range must have format 'YYYYMMDD-YYYYMMDD'")
        from_year, from_month, from_day, to_year, to_month, to_day = (int(group) for group in range_match.groups())
        assert 2000 <= from_year <= to_year <= 2100
        assert 1 <= from_month <= 12 and 1 <= to_month <= 12
        assert 1 <= from_day <= 31 and 1 <= to_day <= 31
        min_arrival_date = datetime.date(from_year, from_month, from_day)
        max_arrival_date = datetime.date(to_year, to_month, to_day)
    else:
        min_arrival_date = datetime.date(fiscal_year - 1, 10, 1)
        max_arrival_date = datetime.date(fiscal_year, 10, 1)

    batching = not args.nobatching
    unit_cases = args.split
    iterations = args.iterations
    reverse_time = args.reverse
    bootstrap = args.bootstrap
    if bootstrap is not None:
        assert bootstrap > 0.
    trace_file = args.trace
    caps_mode = args.capsmode
    assert 0 <= caps_mode <= 3
    delta_days = args.lookback
    if delta_days is not None:
        assert delta_days >= 0
        assert bootstrap is None, "When bootstrapping, lookback doesn't happen, so cannot specify both parameters."
    elif bootstrap is None:
        delta_days = 183
    k = args.k
    assert k >= 1

    params = vars(args)
    print(params)

    return {"min_arrival_date": min_arrival_date, "max_arrival_date": max_arrival_date, "caps_mode": caps_mode,
            "reverse_time": reverse_time, "unit_cases": unit_cases, "batching": batching, "iterations": iterations,
            "trace_file": trace_file, "delta_days": delta_days, "bootstrap": bootstrap, "k": k}, params


def setup(min_arrival_date: datetime.date, max_arrival_date: datetime.date, caps_mode: int, reverse_time: bool,
          unit_cases: bool, batching: bool):
    min_allocation_date = datetime.date(2011, 3, 1)  # no unemployment rates before
    max_allocation_date = datetime.date(2020, 2, 1)  # newer unemployment data not available in some counties
    only_free = False

    arrivals_by_affiliate, previous_avg_case_size = get_stats(min_arrival_date, max_arrival_date, min_allocation_date,
                                                              max_allocation_date, only_free)
    localities = list(arrivals_by_affiliate.keys())
    # on purpose include arrival times outside the window, which are still useful in sampling trajectories
    all_dates, all_batches = real_year_cases(localities, "~/HIAS_FY06-FY20.csv", min_allocation_date,
                                             max_allocation_date, only_free=only_free)

    if reverse_time:
        all_batches.reverse()
        first_allocation_date, last_allocation_date = all_dates[0], all_dates[-1]
        all_dates = [last_allocation_date - (date - first_allocation_date) for date in reversed(all_dates)]

    if unit_cases:
        all_batches = split_cases_into_individuals(all_batches)

    range_dates = []
    range_batches = []
    for date, old_batch in zip(all_dates, all_batches):
        new_batch = []
        for case in old_batch:
            if only_free and case.ties_to is not None:
                continue
            if min_arrival_date <= case.arrival_date < max_arrival_date:
                new_batch.append(case)
        if len(new_batch) > 0:
            range_dates.append(date)
            range_batches.append(new_batch)

    if not batching:
        range_dates, range_batches = split_batches_into_cases(range_dates, range_batches)

    print("number of range batches", len(range_batches))
    print("number of range cases", sum(len(batch) for batch in range_batches))
    print("number of range refugees", sum(case.size for batch in range_batches for case in batch))
    print("first range date", range_dates[0])
    print("last range date", range_dates[-1])

    if caps_mode == 0:
        caps_dict = {}
        for loc in localities:
            caps_dict[loc] = round(1.1 * arrivals_by_affiliate[loc])
        capacities = ConstantCapacities(caps_dict)
    else:
        fiscal_year = max_arrival_date.year
        assert max_arrival_date == datetime.date(fiscal_year, 10, 1)
        assert min_arrival_date == datetime.date(fiscal_year - 1, 10, 1)
        if caps_mode == 1:
            capacities = get_fy_caps(fiscal_year, localities, hundredten_percent=True,
                                     dynamic_mode=CapacitiesDynamic.static_end_of_FY,
                                     end_of_year_minimums=arrivals_by_affiliate)
        elif caps_mode == 2:
            capacities = get_fy_caps(fiscal_year, localities, hundredten_percent=True,
                                     dynamic_mode=CapacitiesDynamic.static_start_of_FY)
        else:
            assert caps_mode == 3
            capacities = get_fy_caps(fiscal_year, localities, hundredten_percent=True,
                                     dynamic_mode=CapacitiesDynamic.dynamic)

    return localities, all_dates, all_batches, range_dates, range_batches, capacities, previous_avg_case_size


def main():
    parsed_params, params = get_command_line_args()
    min_arrival_date = parsed_params["min_arrival_date"]
    max_arrival_date = parsed_params["max_arrival_date"]
    caps_mode = parsed_params["caps_mode"]
    reverse_time = parsed_params["reverse_time"]
    bootstrap = parsed_params["bootstrap"]
    unit_cases = parsed_params["unit_cases"]
    batching = parsed_params["batching"]
    iterations = parsed_params["iterations"]
    trace_file = parsed_params["trace_file"]
    k = parsed_params["k"]

    experiment_setup = setup(min_arrival_date, max_arrival_date, caps_mode, reverse_time, unit_cases, batching)
    if bootstrap is None:
        localities, all_dates, all_batches, range_dates, range_batches, capacities, reference_avg_case_size = experiment_setup
        sampling_delta = datetime.timedelta(days=parsed_params["delta_days"])
        case_pool = LookBackCasePool(all_dates, all_batches, sampling_delta)
        if caps_mode <= 1:
            trajectory_lengths = KnownCaseNumberTrajectoryLengths(sum(len(batch) for batch in range_batches))
        else:
            trajectory_lengths = TrustCapacityTrajectoryLengths(reference_avg_case_size)
    else:
        localities, _, _, reference_dates, reference_batches, capacities, reference_avg_case_size = experiment_setup
        assert isinstance(capacities, ConstantCapacities)
        all_cases = [case for batch in reference_batches for case in batch]
        reference_avg_case_size = sum(case.size for case in all_cases) / len(all_cases)
        case_pool = ConstantCasePool(all_cases)
        simulated_arrival_sequences = []
        expected_arrivals = sum(capacities.capacities.values()) / 1.1
        how_many_arrivals_to_simulate = round(bootstrap * expected_arrivals)
        for _ in range(iterations):
            next_date = datetime.date(1990, 1, 1)
            iteration_dates = []
            iteration_cases = []
            prefix_optimal_employments = []
            arrival_count = 0
            while arrival_count < how_many_arrivals_to_simulate:
                new_case = choice(all_cases)
                iteration_dates.append(next_date)
                iteration_cases.append(new_case)
                arrival_count += new_case.size
                next_date += datetime.timedelta(days=1)

                greedy_algorithm = GreedyAlgorithm(localities)
                optimal_assignment = greedy_algorithm.allocate_batch(iteration_cases, capacities.capacities,
                                                                     datetime.date(1990, 1, 1))
                optimal_employment = sum(case.expected_employment[assignment] for case, assignment
                                         in zip(iteration_cases, optimal_assignment) if assignment is not None)
                prefix_optimal_employments.append(optimal_employment)
            iteration_batches = [[case] for case in iteration_cases]
            simulated_arrival_sequences.append({"batches": iteration_batches, "dates": iteration_dates,
                                                "optimal prefix employments": prefix_optimal_employments})
        assert caps_mode <= 1
        trajectory_lengths = TrustCapacityTrajectoryLengths(reference_avg_case_size)
        trajectory_lengths_poisson = PoissonTrustCapacityTrajectoryLengths(reference_avg_case_size)
        trajectory_lengths_nbinom = NegBinomialTrustCapacityTrajectoryLengths(reference_avg_case_size)

    algorithms = {
        "optimum": (OptimumHindsightAlgorithm, None),
        #"historic": (HistoricMatching, (localities,)),
        "greedy": (GreedyAlgorithm, (localities,)),
        #f"price1 k={k}": (DynamicDualPriceAlgorithm, (localities, case_pool, False, KindOfDual.maximal,
        #                                           trajectory_lengths, capacities, k, 1.)),
        f"price2 k={k}": (DynamicDualPriceAlgorithm, (localities, case_pool, True, KindOfDual.minimal,
                                                   trajectory_lengths, capacities, k, 1.))
    }

    if bootstrap is not None:
        #algorithms[f"price1 Poisson k={k}"] = (DynamicDualPriceAlgorithm, (localities, case_pool, False, KindOfDual.maximal,
        #                                                                trajectory_lengths_poisson, capacities, k, 1.))
        algorithms[f"price2 Poisson k={k}"] = (DynamicDualPriceAlgorithm, (localities, case_pool, True, KindOfDual.minimal,
                                                                        trajectory_lengths_poisson, capacities, k, 1.))
        #algorithms[f"price1 NegBinom k={k}"] = (DynamicDualPriceAlgorithm, (localities, case_pool, False, KindOfDual.maximal,
        #                                                                trajectory_lengths_nbinom, capacities, k, 1.))
        algorithms[f"price2 NegBinom k={k}"] = (DynamicDualPriceAlgorithm, (localities, case_pool, True, KindOfDual.minimal,
                                                                        trajectory_lengths_nbinom, capacities, k, 1.))

    employments = {}
    ratio_matched = {}

    traces = {}
    for alg_name, (alg_class, alg_init_args) in algorithms.items():
        print(alg_name)
        employments[alg_name] = []
        ratio_matched[alg_name] = []
        traces[alg_name] = []
        ignore_compatibility = alg_name == "historic" or alg_name == "optimal"
        if bootstrap is None and alg_class.deterministic:
            effective_iterations = 1
        else:
            effective_iterations = iterations
        for iteration_number in range(effective_iterations):
            if bootstrap is not None:
                range_dates = simulated_arrival_sequences[iteration_number]["dates"]
                range_batches = simulated_arrival_sequences[iteration_number]["batches"]

            if alg_init_args is not None:
                alg = alg_class(*alg_init_args)
            else:
                assert alg_class is OptimumHindsightAlgorithm
                alg = OptimumHindsightAlgorithm(localities, range_dates, range_batches,
                                                capacities.get_capacities_at_date(range_dates[-1]))

            emp, num_matched, num_unmatched, trace = trace_dynamic_algorithm(localities, range_dates, range_batches,
                                                                             capacities, alg, ignore_compatibility)

            if bootstrap is not None:
                cumulative_emp = 0.
                prefix_employment_perc = []
                arrival_count = 0
                percentage_arrival = []
                for case_emp, case_size, optimal_prefix_emp in zip(trace["employment"], trace["case size"],
                                           simulated_arrival_sequences[iteration_number]["optimal prefix employments"]):
                    cumulative_emp += case_emp
                    arrival_count += case_size
                    if optimal_prefix_emp > 0:
                        prefix_employment_perc.append(cumulative_emp / optimal_prefix_emp)
                    else:
                        prefix_employment_perc.append(None)
                    percentage_arrival.append(arrival_count / expected_arrivals)
                trace["prefix employment % of optimal"] = prefix_employment_perc
                trace["fraction of expected arrivals"] = percentage_arrival

            traces[alg_name].append(trace)
            employments[alg_name].append(emp)
            print(emp, num_matched / (num_matched + num_unmatched))
            ratio_matched[alg_name].append(num_matched / (num_matched + num_unmatched))
        if effective_iterations >= 2:
            print(np.mean(employments[alg_name]), np.std(employments[alg_name], ddof=1))

    with open(trace_file, "wb") as file:
        dump((params, traces, employments, ratio_matched), file)

    print(employments)
    print(ratio_matched)


if __name__ == "__main__":
    main()
