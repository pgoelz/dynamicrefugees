from math import floor, ceil
from typing import List, Optional, Tuple

import plotly.express as px
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import datetime
import numpy as np

pd.set_option('display.max_rows', None)

from pathlib import Path
from pickle import load
from scipy import signal


def alg_pretty(alg: str, batching: bool) -> Tuple[str, str, Optional[str]]:
    # return algorithm name, string description of k, possibly a string description of the dual used
    if alg == "historic":
        return "historical", "", None
    if "price" in alg:
        which_pot = None
        if "price1" in alg:
            which_pot = "Pot1"
        if "price2" in alg:
            assert which_pot is None
            which_pot = "Pot2"
        _, end = alg.split("k=")
        k = int(end)
        res = "PM"
        if batching:
            res += "B"
        if "poisson" in alg.lower():
            res += "Poisson"
        if "negbinom" in alg.lower():
            res += "NegBinom"
        res += f"({which_pot}(k={k}))"
        return res, str(k), which_pot
    return alg, "", None


def trace_files_to_dataframe(directories: List[str]):
    inputs = []

    for directory in directories:
        for dir_entry in sorted(Path(directory).iterdir()):
            if dir_entry.is_file() and (dir_entry.name.endswith(".txt") or dir_entry.name.endswith(".pickle")):
                print(dir_entry.name)
                with open(dir_entry, "rb") as file:
                    params, alg_traces, alg_emps, alg_ratio_matcheds = load(file)
                inputs.append((params, alg_traces, alg_emps, alg_ratio_matcheds, str(dir_entry.resolve())))

    data = []
    data_trace = []
    employment_series = {}
    for params, alg_traces, alg_emps, alg_ratio_matcheds, input_path in inputs:
        for alg in alg_emps:
            emps = alg_emps[alg]
            ratio_matched = alg_ratio_matcheds[alg]
            for i, (emp, unmatched) in enumerate(zip(emps, ratio_matched)):
                datum = params.copy()
                datum["input path"] = input_path
                datum["trace"] = i
                datum["algorithm"], datum["k"], datum["potential"] = alg_pretty(alg, not datum["nobatching"])
                datum["employment"] = emp
                datum["relative employment"] = emp / alg_emps["optimum"][0]
                datum["experiment"] = str(params)
                datum["percentage matched"] = unmatched
                if params["fy"] is not None:
                    datum["time range"] = f"FY {params['fy']}"
                else:
                    assert params["customrange"] is not None
                    datum["time range"] = params["customrange"]
                data.append(datum)

                if alg_traces is None:
                    continue
                traces = alg_traces[alg]
                trace = traces[i]
                cum_emp = 0
                ref_count = 0
                emp_ser = []

                if (datum["experiment"], datum["algorithm"], datum["trace"]) not in employment_series:
                    for j in range(len(trace["case size"])):
                        entry = {key: trace[key][j] for key in trace}
                        entry["time range"] = datum["time range"]
                        entry["algorithm"] = datum["algorithm"]
                        entry["experiment"] = datum["experiment"]
                        entry["input path"] = datum["input path"]
                        entry["trace"] = i
                        entry["case"] = j
                        entry["refugee"] = ref_count
                        cum_emp += entry["employment"]
                        entry["cum_emp"] = cum_emp
                        data_trace.append(entry)
                        ref_count += entry["case size"]
                        emp_ser.extend([entry["employment"] / entry["case size"] for _ in range(entry["case size"])])
                    employment_series[(datum["experiment"], datum["algorithm"])] = emp_ser

    df = pd.DataFrame(data)
    df_trace = pd.DataFrame(data_trace)

    return df, df_trace


def plot_employment_bars(df: pd.DataFrame, which_algorithms: List[str], output_path: str):
    assert "ABSREL" in output_path
    colors = dict(zip(
        ["optimum", "greedy", "historical", "PM(Pot1(k=1))", "PM(Pot1(k=3))", "PM(Pot1(k=5))", "PM(Pot2(k=1))",
         "PM(Pot2(k=3))", "PM(Pot2(k=5))"],
        [sns.color_palette("tab10")[0], sns.color_palette("tab10")[1], sns.color_palette("tab10")[2],
         sns.color_palette("pastel")[3], sns.color_palette("tab10")[3], sns.color_palette("tab10")[3],
         sns.color_palette("pastel")[4], sns.color_palette("tab10")[4], sns.color_palette("tab10")[4]]))
    for alg, col in list(colors.items()):
        if "PM(" in alg:
            colors[alg.replace("PM(", "PMB(")] = col
            colors[alg.replace("PM(", "PMBPoisson(").replace("k=5", "k=7")] = col

    df = df[df["algorithm"].isin(which_algorithms)]

    # algs = ["optimum", "greedy", "historical", "PM(Pot1(k=5))", "PM(Pot2(k=5))"]
    #algs = ["optimum", "PMBPoisson(Pot1(k=7))", "PMBPoisson(Pot2(k=7))"]
    sns.set_style("whitegrid")

    fig = sns.catplot(data=df, x="time range", y="employment", hue="algorithm", kind="bar", hue_order=which_algorithms,
                      aspect=2, height=3.5, palette=[colors[alg] for alg in which_algorithms])
    ax = fig.ax
    ax.set_xlabel("fiscal year")
    ax.set_ylabel("total employment")

    ax.yaxis.set_major_locator(ticker.MultipleLocator(100))
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

    fig.savefig(output_path.replace("ABSREL", "absolute"))

    fig = sns.catplot(data=df, x="time range", y="relative employment", hue="algorithm", kind="bar",
                      hue_order=which_algorithms, aspect=3, height=2.5,
                      palette=[colors[alg] for alg in which_algorithms], ci=None)
    # print(df_p.groupby(by=["experiment","fy","algorithm"])["employment"].count())
    ax = fig.ax
    ax.set_ylim(.85, 1)
    ax.set_xlabel("fiscal year")
    ax.set_ylabel("total employment as fraction of optimum")

    abs_employments = df.groupby(by=["time range", "algorithm"])["employment"].mean().to_dict()
    time_ranges = df["time range"].unique()
    for alg, cont in zip(which_algorithms, ax.containers):
        for fy, patch in zip(time_ranges, cont.patches):
            ax.text(patch.get_x() + .5 * patch.get_width(), patch.get_height(),
                    str(round(abs_employments[(fy, alg)])) + " ", verticalalignment='top', horizontalalignment='center',
                    color="white", rotation=90)

    ax.minorticks_on()
    ax.yaxis.set_major_locator(ticker.MultipleLocator(.05))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(.01))
    plt.grid(which='minor', linestyle='-', alpha=.4, axis="y")
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1., decimals=0))

    fig.savefig(output_path.replace("ABSREL", "relative"))
    plt.close()


def plot_lookback(df: pd.DataFrame, which_algorithm: str, output_path: str):
    df = df[df["algorithm"] == which_algorithm]
    df["total employment as fraction of optimum"] = df["relative employment"] * 100
    fig = sns.relplot(data=df, x="lookback", y="total employment as fraction of optimum", kind="line", hue="time range", ci=None, aspect=2.)
    fig.ax.set_xlabel("sampling window (days)")
    fig.ax.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))
    fig.savefig(output_path)
    plt.close()


def plot_bootstrap(df: pd.DataFrame, output_path: str):
    print(list(df.columns))
    df = df[(~df["algorithm"].isin(["historical", "optimum"]))&(~df["algorithm"].str.contains("k=1"))]

    df = df.sort_values(by=["fraction of expected arrivals"])

    step = .01
    maxi = float(df["fraction of expected arrivals"].max())
    max_x = floor(maxi / step) + 1
    algs = sorted(list(df["algorithm"].unique()))
    time_ranges = list(df["time range"].unique())

    trajectory_sums = {(alg, tr): [0. for x in range(max_x)] for alg in algs for tr in time_ranges}
    trajectory_denoms = {(alg, tr): 0 for alg in algs for tr in time_ranges}

    for (alg, time, _, _), values in df.groupby(["algorithm", "time range", "input path", "trace"]):
        trajectory_denoms[(alg, time)] += 1
        last_x = 0
        last_y = 0
        old_entry = None
        i = 0
        for entry in values.to_dict("records"):
            new_x = entry["fraction of expected arrivals"] / step
            assert new_x > last_x, f"Somehow, no move forward {repr(old_entry)}, old: {repr(entry)}"
            new_y = entry["prefix employment % of optimal"]
            while i <= new_x and i < max_x:
                phi = (i - last_x) / (new_x - last_x)
                value = (1 - phi) * last_y + phi * new_y
                trajectory_sums[(alg, time)][i] += value
                i += 1
            last_x = new_x
            last_y = new_y
            old_entry = entry

    data = []
    for alg in algs:
        for tr in time_ranges:
            tr = str(tr)
            if tr.startswith("FY "):
                fy = int(tr[3:])
            else:
                fy = None
            for x in range(max_x):
                if trajectory_denoms[(alg, tr)] > 0:
                    data.append({"algorithm": alg, "fraction of expected arrivals": x * step * 100,
                                 "prefix employment % of optimal": trajectory_sums[(alg, tr)][x] / trajectory_denoms[(alg, tr)] * 100,
                                 "time range": tr, "fiscal year": fy})

    """
    step = .01
    maxi = float(df["fraction of expected arrivals"].max())
    print(maxi)
    triangle_width = 2
    max_x = floor(maxi / step) + 1
    algs = sorted(list(df["algorithm"].unique()))
    time_ranges = list(df["time range"].unique())

    weighted_sums = {(alg, tr): [0. for x in range(max_x)] for alg in algs for tr in time_ranges}
    weight_sums = {(alg, tr): [0. for x in range(max_x)] for alg in algs for tr in time_ranges}

    for entry in df.to_dict("records"):
        alg = entry["algorithm"]
        tr = entry["time range"]
        xo = entry["fraction of expected arrivals"] / step
        xo_rounded = round(xo)
        for offset in range(-triangle_width - 2, triangle_width + 3):  # this overshoots the real interval to avoid off-by-ones
            x = xo_rounded + offset
            if x < 0 or x > floor(maxi / step):
                continue
            if abs(x - xo) < triangle_width:
                weight = 1. - abs(x - xo) / triangle_width
                weight_sums[(alg, tr)][x] += weight
                weighted_sums[(alg, tr)][x] += weight * entry["prefix employment % of optimal"]

    data = []
    for alg in algs:
        for tr in time_ranges:
            for x in range(max_x):
                if weight_sums[(alg, tr)][x] > 0:
                    data.append({"algorithm": alg, "fraction of expected arrivals": x * step * 100,
                                 "prefix employment % of optimal": weighted_sums[(alg, tr)][x] / weight_sums[(alg, tr)][x] * 100,
                                 "time range": tr})
                                 """

    colors = dict(zip(
        ["optimum", "greedy", "historical", "PM(Pot1(k=1))", "PM(Pot1(k=3))", "PM(Pot1(k=5))", "PM(Pot2(k=1))",
         "PM(Pot2(k=3))", "PM(Pot2(k=5))"],
        [sns.color_palette("tab10")[0], sns.color_palette("tab10")[1], sns.color_palette("tab10")[2],
         sns.color_palette("pastel")[3], sns.color_palette("tab10")[3], sns.color_palette("tab10")[3],
         sns.color_palette("pastel")[4], sns.color_palette("tab10")[4], sns.color_palette("tab10")[4]]))
    for alg in algs:
        basic_alg = alg.replace("PMBPoisson", "PM").replace("PMBNegBinom", "PM").replace("PMB(", "PM(")
        colors[alg] = colors[basic_alg]
    dashes = {}
    for alg in algs:
        if "Poisson" in alg:
            dashes[alg] = (1, 1)
        elif "NegBinom" in alg:
            dashes[alg] = (2, 2)
        else:
            dashes[alg] = ""

    df_smoothed = pd.DataFrame(data)
    df_smoothed = df_smoothed[df_smoothed["fraction of expected arrivals"] >= 20]

    df_smoothed = df_smoothed.rename(columns={"fraction of expected arrivals": "arriving refugees as % of expected arrivals",
                                              "prefix employment % of optimal": "total employment as % of optimum"})
    alg_order = sorted(df_smoothed["algorithm"].unique(), key=lambda x: (x=="greedy", "Pot1" in x, "NegBinom" in x, "Poisson" in x))
    print(alg_order)

    with sns.axes_style("whitegrid"), sns.plotting_context("talk"):
        #fig = sns.relplot(data=df_smoothed, x="fraction of expected arrivals", y="prefix employment % of optimal", kind="line",
        #                  hue="algorithm", palette=colors, style="algorithm", dashes=dashes, col="fiscal year", col_wrap=2, ci=None, aspect=2., height=3.5)
        fig = sns.relplot(data=df_smoothed, x="arriving refugees as % of expected arrivals", y="total employment as % of optimum", kind="line",
                          hue="algorithm", hue_order=alg_order, palette=colors, style="algorithm", dashes=dashes, col="fiscal year", col_wrap=2, ci=None)

        #plt.subplots_adjust(hspace=1)

    for count, ax in enumerate(fig.axes.flat):
        ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
        ax.xaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))
        ax.set(ylim=(90, 100))
        ax.set(xlim=(20, 110))
        ax.grid(visible=True, which='minor', color='gainsboro', linewidth=0.5)
        #if count == 5:
        #    plt.setp(ax.get_xticklabels()[0], visible=False)

    sns.move_legend(fig, "center left", bbox_to_anchor=(1, 0.5), prop={'size': 18})
    plt.tight_layout()

    fig.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    df, df_trace = trace_files_to_dataframe(["../revision_results/traces/", "../revision_results/traces_bootstrap/"])
    print(df["algorithm"].unique())
    print(df["time range"].unique())
    print("read in")

    plot_bootstrap(df_trace[(df_trace["time range"].isin(["FY 2016", "FY 2019"])) & (df_trace["input path"].str.contains("negbin", regex=False, na=0))], "../revision_results/plots/negbinom_16_19.pdf")
    plot_bootstrap(df_trace[(~df_trace["time range"].isin(["FY 2016", "FY 2019"])) & (df_trace["input path"].str.contains("negbin", regex=False, na=0))], "../revision_results/plots/negbinom_others.pdf")

    plot_bootstrap(df_trace[(df_trace["time range"].isin(["FY 2016", "FY 2019"])) & (df_trace["input path"].str.contains("bootstrap", regex=False, na=0))], "../revision_results/plots/bootstrap_16_19.pdf")
    plot_bootstrap(df_trace[(~df_trace["time range"].isin(["FY 2016", "FY 2019"])) & (df_trace["input path"].str.contains("bootstrap", regex=False, na=0))], "../revision_results/plots/bootstrap_others.pdf")

    plot_lookback(df[df["input path"].str.contains("lookback", regex=False, na=0)], "PMB(Pot2(k=5))",
                  "../revision_results/plots/lookback_pot2k5.pdf")

    plot_employment_bars(df[df["input path"].str.contains("lookback517", regex=False, na=0)],
                         ["optimum", "greedy", "historical", "PMB(Pot1(k=5))", "PMB(Pot2(k=5))"],
                         "../revision_results/plots/517lookback_ABSREL.pdf")

    plot_employment_bars(df[df["input path"].str.contains("lookback183", regex=False, na=0)],
                         ["optimum", "greedy", "historical", "PMB(Pot1(k=5))", "PMB(Pot2(k=5))"],
                         "../revision_results/plots/183lookback_ABSREL.pdf")

    plot_employment_bars(df[df["input path"].str.contains("fwd", regex=False, na=0)],
                         ["optimum", "greedy", "historical", "PMB(Pot1(k=5))", "PMB(Pot2(k=5))"],
                         "../revision_results/plots/forward_ABSREL.pdf")

    plot_employment_bars(df[df["input path"].str.contains("[0-9]rev[0-9]", regex=True, na=0)],
                         ["optimum", "greedy", "historical", "PMB(Pot1(k=5))", "PMB(Pot2(k=5))"],
                         "../revision_results/plots/reversed_ABSREL.pdf")

    plot_employment_bars(df[df["input path"].str.contains("arrivalcaps", regex=False, na=0)],
                         ["optimum", "greedy", "historical", "PMB(Pot1(k=5))", "PMB(Pot2(k=5))"],
                         "../revision_results/plots/arrivalcaps_ABSREL.pdf")

    plot_employment_bars(df[df["input path"].str.contains("april", regex=False, na=0)],
                         ["optimum", "greedy", "historical", "PMB(Pot1(k=5))", "PMB(Pot2(k=5))"],
                         "../revision_results/plots/apriltoapril_ABSREL.pdf")

    plot_employment_bars(df[df["input path"].str.contains("july", regex=False, na=0)],
                         ["optimum", "greedy", "historical", "PMB(Pot1(k=5))", "PMB(Pot2(k=5))"],
                         "../revision_results/plots/julytojuly_ABSREL.pdf")

    plot_employment_bars(df[df["input path"].str.contains("january", regex=False, na=0)],
                         ["optimum", "greedy", "historical", "PMB(Pot1(k=5))", "PMB(Pot2(k=5))"],
                         "../revision_results/plots/januarytojanuary_ABSREL.pdf")

    plot_employment_bars(df[df["input path"].str.contains("firsthalf", regex=False, na=0)],
                         ["optimum", "greedy", "historical", "PMB(Pot1(k=5))", "PMB(Pot2(k=5))"],
                         "../revision_results/plots/firsthalf_ABSREL.pdf")

    plot_employment_bars(df[df["input path"].str.contains("secondhalf", regex=False, na=0)],
                         ["optimum", "greedy", "historical", "PMB(Pot1(k=5))", "PMB(Pot2(k=5))"],
                         "../revision_results/plots/secondhalf_ABSREL.pdf")

    plot_employment_bars(df[df["input path"].str.contains("twofys", regex=False, na=0)],
                         ["optimum", "greedy", "historical", "PMB(Pot1(k=5))", "PMB(Pot2(k=5))"],
                         "../revision_results/plots/twoyears_ABSREL.pdf")
