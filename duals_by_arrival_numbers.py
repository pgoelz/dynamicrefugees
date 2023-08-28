import datetime
from argparse import ArgumentParser
from random import choice

import pandas as pd
import seaborn as sns
from matplotlib import ticker

from compare_matching_algorithms import setup
from optimization.capacities import ConstantCapacities
from optimization.matching import KindOfDual, get_duals


def get_command_line_args():
    parser = ArgumentParser()
    parser.add_argument("fy", type=int, help="Fiscal year to evaluate on.")
    parser.add_argument("potential", type=int, help="Which potential to choose (1 or 2).")
    parser.add_argument("iterations", type=int)
    parser.add_argument("datapath", type=str, help="Where to write the csv.")
    parser.add_argument("plotpath", type=str, help="Where to write the PDF plot.")
    args = parser.parse_args()
    fiscal_year = args.fy
    potential = args.potential
    iterations = args.iterations
    data_path = args.datapath
    plot_path = args.plotpath

    min_arrival_date = datetime.date(fiscal_year - 1, 10, 1)
    max_arrival_date = datetime.date(fiscal_year, 10, 1)

    assert potential in (1, 2)
    if potential == 1:
        which_dual = KindOfDual.maximal
    else:
        which_dual = KindOfDual.minimal

    return min_arrival_date, max_arrival_date, iterations, which_dual, data_path, plot_path


def locality_statistics(localities, case_pool, capacities):
    employment_sum = {loc: 0. for loc in localities}
    compatibility_case_num = {loc: 0 for loc in localities}
    compatibility_ref_num = {loc: 0 for loc in localities}
    single_option_count = {loc: 0 for loc in localities}
    employment_sum_single = {loc: 0. for loc in localities}
    max_count = {loc: 0 for loc in localities}

    for case in case_pool:
        compats = []
        best_loc = None
        best_emp = 0.
        for loc in localities:
            if case.is_compatible(loc):
                compats.append(loc)
                employment_sum[loc] += case.expected_employment[loc]
                compatibility_case_num[loc] += 1
                compatibility_ref_num[loc] += case.size
                if case.expected_employment[loc] > best_emp:
                    best_emp = case.expected_employment[loc]
                    best_loc = loc
        if len(compats) == 1:
            single_option_count[compats[0]] += case.size
            employment_sum_single[compats[0]] += case.expected_employment[compats[0]]
        if best_loc is not None:
            max_count[best_loc] += 1

    avg_employment_per_person = {}
    avg_employment_per_single = {}
    for loc in localities:
        if compatibility_ref_num[loc] > 0:
            avg_employment_per_person[loc] = employment_sum[loc] / compatibility_ref_num[loc]
        if single_option_count[loc] > 0:
            avg_employment_per_single[loc] = employment_sum_single[loc] / single_option_count[loc]

    percent_single = {loc: single_option_count[loc] / capacities.capacities[loc] for loc in localities}

    data = {"avg emp": avg_employment_per_person, "comp refugees": compatibility_ref_num,
            "single option": single_option_count, "best option": max_count, "avg single": avg_employment_per_single,
            "percent single": percent_single, "capacity": capacities.capacities}
    for name, dic in data.items():
        print("\n\n")
        print(name)
        as_list = [(val, key) for key, val in dic.items()]
        as_list.sort()
        for val, key in as_list:
            print("\t", key, ":", val)


def compute_duals_per_arrival_numbers(min_arrival_date, max_arrival_date, iterations, which_dual):
    experiment_setup = setup(min_arrival_date, max_arrival_date, caps_mode=0, reverse_time=False, unit_cases=False,
                             batching=True)
    capacities: ConstantCapacities
    localities, _, _, _, range_batches, capacities, _ = experiment_setup
    case_pool = [case for batch in range_batches for case in batch]
    num_cases = len(case_pool)

    locality_statistics(localities, case_pool, capacities)

    data = []
    for trajectory_length in range(1, 2 * num_cases + 1, 100):
        print(trajectory_length, 2 * num_cases)
        for i in range(iterations):
            trajectory = [choice(case_pool) for _ in range(trajectory_length)]
            new_prices = get_duals(localities, trajectory, capacities.capacities, which_dual)
            for loc, price in new_prices.items():
                if capacities.capacities[loc] < 20:
                    continue
                data.append({"affiliate": loc, "price": price, "trajectory length": trajectory_length,
                             "relative trajectory length": trajectory_length / num_cases, "iteration": i})

    return pd.DataFrame(data)


if __name__ == "__main__":
    min_arrival_date, max_arrival_date, iterations, which_dual, data_path, plot_path = get_command_line_args()
    df = compute_duals_per_arrival_numbers(min_arrival_date, max_arrival_date, iterations, which_dual)
    df.to_csv(data_path)

    average_prices = df.groupby("affiliate")["price"].mean().to_dict()
    localities = list(average_prices.keys())
    localities.sort(key=lambda x: -average_prices[x])
    locality_aliases = []
    colors = []
    dashes = []
    dash_patterns = [(1, 0), (3, 1.25, 1.5, 1.25), (1, 1, .1, 1)]
    i = 0
    while len(colors) < len(localities):
        colors = colors + sns.color_palette()
        dashes = dashes + [dash_patterns[i] for _ in sns.color_palette()]
        i += 1
    colors = colors[:len(localities)]
    dashes = dashes[:len(localities)]
    print(len(localities))
    print(len(colors))
    df["relative trajectory length"] = 100 * df["relative trajectory length"]

    fig = sns.relplot(data=df, x="relative trajectory length", y="price", kind="line", hue="affiliate", hue_order=localities, style="affiliate", style_order=localities, palette=colors, dashes=dashes, aspect=2, ci=None, legend=False)
    fig.ax.set_xlim((0, 175))
    fig.ax.xaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))
    fig.tight_layout()
    fig.fig.savefig(plot_path)
