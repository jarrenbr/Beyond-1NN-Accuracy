from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikit_posthocs as sp
import seaborn as sns
from scipy.stats import hmean
from statds import no_parametrics as npnp

import common as cmn
from config import ExpCol


FRIEDMAN_ALPHA = 0.1
STABLE_MIN_PVAL = 1e-12

import warnings
warnings.filterwarnings("ignore")

def nemenyi_post_hoc(rank_matrix:pd.DataFrame, title:str=None, results_dir:Path = cmn.DIRTS_DIR):
    if title is None:
        title = f"Critical Difference Diagram"
    # Post-hoc Nemenyi test
    nemenyi_pvals = sp.posthoc_nemenyi_friedman(rank_matrix)

    avg_ranks = rank_matrix.mean(axis=0).rename("Rank")
    plt.figure(figsize=(10, 6))
    plt.title(title)
    sp.critical_difference_diagram(
        ranks=avg_ranks,
        sig_matrix=nemenyi_pvals,
        # color_palette=cmap
    )

    n_measures = avg_ranks.shape[0]
    ax = plt.gca()
    ax.set_xlim(0.9, n_measures + 0.1)  # show ranks from 1 to k
    ax.set_xticks(range(1, n_measures + 1))  # integer ticks only
    ax.set_xticklabels(range(1, n_measures + 1))
    plt.tight_layout()
    file_stem = title.replace(" ", "_").replace(":", "_").replace("\n", "_")
    plt.savefig(results_dir / f"{file_stem}.png")

def iman_friedman(rank_matrix:pd.DataFrame):
    avg_rankings, stat, p_value, crit_val_or_none, hypothesis_str = npnp.iman_davenport(rank_matrix, alpha=FRIEDMAN_ALPHA)
    return stat, p_value

def get_rank_matrix(df, algorithm:str):
    dataset_df = df[df[ExpCol.lens] == algorithm].copy(deep=True)

    dataset_df['rank'] = (
        dataset_df.groupby(ExpCol.dataset)[ExpCol.score] # Rank each dataset by score
                          .rank(ascending=False, method="average")
                          )
    rank_matrix = dataset_df.pivot(index=ExpCol.dataset, columns=ExpCol.measure, values=ExpCol.ranking)

    return rank_matrix


def get_rank_matrix_hmp_over_dataset(df, algorithm:str, hmp_tau:float = 0.05)->tuple[pd.DataFrame, pd.Series]:
    dataset_df = df[df[ExpCol.lens] == algorithm].copy(deep=True)
    minimum_pval = STABLE_MIN_PVAL
    dataset_df[ExpCol.p_value] = np.clip(dataset_df[ExpCol.p_value], minimum_pval, None)
    dataset_hmp: dict[str,float] = {}
    for dataset_name, dataset_group in dataset_df.groupby(ExpCol.dataset):
        hmp = hmean(dataset_group[ExpCol.p_value])
        dataset_hmp[dataset_name] = hmp

    dataset_hmp = pd.Series(dataset_hmp).rename("HMP")
    dataset_fits = dataset_hmp[dataset_hmp <= hmp_tau]
    dataset_df = dataset_df[dataset_df[ExpCol.dataset].isin(dataset_fits.index)]

    dataset_df['rank'] = (
        dataset_df.groupby(ExpCol.dataset)[ExpCol.score] # Rank each dataset by score
                          .rank(ascending=False, method="average")
                          )

    rank_matrix = dataset_df.pivot(index=ExpCol.dataset, columns=ExpCol.measure, values=ExpCol.ranking)

    return rank_matrix, dataset_fits

def get_ts_df():
    df = pd.read_csv(cmn.RESULT_DIR / "all_time_series_scores.csv")
    df = df.sort_values(["domain", "dataset", "lens", "measure"])

    df = df.drop_duplicates(
        subset=["dataset", "lens", "measure"],
        keep="first"
    ).reset_index(drop=True)
    df.drop(columns=['domain'], inplace=True)

    return df

def do_analyses():
    presented_results = ["1-NN Acc", "1-NN MCC", "SVC-LOO", "K-Medoids", "Aggregate Clustering",
                         "Spectral Clustering", "RNLDD", "Weighted n-1-NN AUROC"]
    # df = pd.read_csv(cmn.RESULT_DIR / "all_scores.csv")
    df = get_ts_df()

    friedman_analyses : [tuple[str, str], dict] = {} # [algorithm name, filter name], data_dict
    for algorithm_name in df[ExpCol.lens].unique():
        try:
            rm = get_rank_matrix(df, algorithm_name)
            stat, pval = iman_friedman(rm)
            friedman_analyses[(algorithm_name, "No Filter")] = {"Stat": stat, "Pval": pval, "DatasetCount" : rm.shape[0], "MeasureCount" : rm.shape[1]}
            if algorithm_name in presented_results:
                nemenyi_post_hoc(rm, title=f"Critical Difference Diagram for {algorithm_name}")

            hmp_tau = 0.05
            rm, hmp = get_rank_matrix_hmp_over_dataset(df, algorithm_name, hmp_tau=hmp_tau)
            filter_name = f"HMP={hmp_tau} Filtering Datasets"
            stat, pval = iman_friedman(rm)
            friedman_analyses[(algorithm_name, filter_name)] = {"Stat": stat, "Pval": pval, "DatasetCount" : rm.shape[0], "MeasureCount" : rm.shape[1]}
            # nemenyi_post_hoc(rm, title=f"Critical Difference Diagram for {algorithm_name} with {filter_name}")
        except Exception as e:
            continue

    friedman_analyses_df = pd.DataFrame.from_dict(friedman_analyses, orient="index")
    friedman_analyses_df.index = pd.MultiIndex.from_tuples(friedman_analyses_df.index, names=[ExpCol.lens, "Filter"])
    friedman_analyses_df.reset_index(inplace=True)
    friedman_analyses_df.to_csv(cmn.RESULT_DIR / "friedman_analyses.csv", index=False)

    return friedman_analyses_df


def plot_dataset_count(df_in):
    from common import Lens as L
    df = df_in[
            df_in['lens'].isin([L.spectral, L.rndldd_auroc, L.svm, L.k_medoids, L.agg_cluster, "1-NN MCC", "Weighted n-1-NN AUROC", "1-NN Acc"])
        ].copy()
    df['lens'] = df['lens'].replace({"Weighted n-1-NN AUROC" : "Weighted (n-1)-NN"})


    ds_filter_df = df[df['Filter'] == "HMP=0.05 Filtering Datasets"].copy()
    ds_filter_df.sort_values(by=['DatasetCount'], inplace=True)
    dataset_count_name = "Count of Datasets Meeting HMP ≤ 0.05"
    ds_filter_df.rename(columns={"lens": "Algorithm", "DatasetCount" : dataset_count_name}, inplace=True)

    ds_filter_df["color_group"] = np.where(ds_filter_df["Algorithm"] == "1-NN Acc", "1-NN Acc", "Other")
    palette = {"1-NN Acc": "C1", "Other": "C0"}  # change C1/C0 if you want

    order = ds_filter_df["Algorithm"].tolist()

    plt.figure(figsize = (8,5))
    ax = sns.barplot(y="Algorithm", x=dataset_count_name,
                     hue='color_group', data=ds_filter_df, orient="h",
                     palette=palette, dodge=False, order=order)
    ax.set(ylabel="Diagnostic")
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend_.remove()
    # ax.xaxis.grid(True)

    plt.tight_layout()
    plt.savefig(cmn.DIRTS_DIR / "dataset_count_plot.png")
    return

def plot_stat_power_no_filter(df):
    plot_df = df[df['Filter'] == "No Filter"].copy()

    stat_name = "Friedman-Imam-Davenport Statistic"

    plot_df['lens'] = plot_df['lens'].replace({"SVC-LOO" : "SVC", "Weighted n-1-NN AUROC" : "Weighted (n-1)-NN"})

    lens_to_category = {
        "SVC": "Supervised",
        "1-NN MCC": "Supervised",
        "1-NN Acc": "Supervised",
        "Weighted (n-1)-NN": "Supervised",

        "Spectral Clustering": "Unsupervised",
        "Aggregate Clustering": "Unsupervised",
        "K-Medoids": "Unsupervised",

        "RNLDD": "Structural",
    }

    plot_df["category"] = plot_df["lens"].map(lens_to_category)



    plot_df.sort_values(by=['Stat'], inplace=True)
    plot_df.rename(columns = {"Stat" : stat_name, "lens" : "Algorithm"}, inplace=True)

    plot_df["color_group"] = np.where(plot_df["Algorithm"] == "1-NN Acc", "1-NN Acc", "Other")
    palette = {"1-NN Acc": "C1", "Other": "C0"}  # change C1/C0 if you want

    plt.figure(figsize=(8, 6))
    ax = sns.barplot(
        y="Algorithm",
        x=stat_name,
        data=plot_df,
        orient="h",
        order=plot_df["Algorithm"].unique().tolist(),
        hue="color_group",
        palette=palette,
        dodge=False,
    )
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_ylabel("Diagnostic")

    ax.legend_.remove()
    plt.tight_layout()
    plt.savefig(cmn.DIRTS_DIR / "friedman_stats_no_filter.png")

    return


def knn_stat_power(df:pd.DataFrame):
    # Keep only "No Filter" rows and -NN algorithms
    df = df[df["Filter"] == "No Filter"].copy()
    df = df[df["lens"].str.contains("-NN")].copy()

    # Parse lens into (Group, Variant)
    def parse_lens(lens: str) -> tuple[str, str]:
        lens = str(lens)

        # Group = base rule up through "-NN"
        idx = lens.find("-NN")
        if idx != -1:
            group = lens[:idx + 3]  # include "-NN"
            remainder = lens[idx + 3:].strip()
        else:
            group = lens
            remainder = ""

        # Variant = metric / variant name
        if remainder == "":
            variant = "Default"
        else:
            remainder = remainder.lstrip("- ").strip()
            up = remainder.upper()
            if "AUROC" in up:
                variant = "AUROC"
            elif "ACC" in up:
                variant = "Accuracy"
            elif "MCC" in up:
                variant = "MCC"
            else:
                variant = remainder

        return group, variant

    df[["Group", "Performance Metric"]] = df["lens"].apply(
        lambda x: pd.Series(parse_lens(x))
    )

    df["IsWeighted"] = df["lens"].str.startswith("Weighted")

    base_order = [
        "1-NN",
        "3-NN",
        "5-NN",
        "7-NN",
        "2%-NN",
        "5%-NN",
        "n-1-NN",
        "n-NN",
        "√n-NN",
    ]

    def plot_grouped_bars(pivot: pd.DataFrame, title: str, filename: str) -> None:
        """Grouped bar plot of Stat for each Group x Performance Metric."""
        if pivot.empty:
            return

        groups = list(pivot.index)
        variants = list(pivot.columns)

        x = np.arange(len(groups))
        width = 0.8 / max(len(variants), 1)

        fig, ax = plt.subplots(figsize=(12, 6))

        for i, variant in enumerate(variants):
            heights = pivot[variant].values
            # Center the groups
            x_offsets = x + (i - (len(variants) - 1) / 2) * width
            ax.bar(x_offsets, heights, width, label=variant)

        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=45, ha="right")
        ax.set_ylabel("Friedman-Imam-Davenport Statistic")
        ax.set_title(title)
        ax.legend(title="Performance Metric")
        fig.tight_layout()

        plt.savefig(cmn.DIRTS_DIR / filename)

    unweighted_df = df[~df["IsWeighted"]].copy()
    unweighted_df = unweighted_df[~unweighted_df['lens'].isin(
        ["n-1-NN Acc", "n-1-NN MCC", "n-NN Acc", "n-NN MCC"]
    )]
    unweighted_pivot = unweighted_df.pivot_table(
        index="Group", columns="Performance Metric", values="Stat"
    )

    # Order: √n-NN, 1-NN, 3-NN, 5-NN, 7-NN, 2%-NN, 5%-NN, n-1-NN, n-NN
    unweighted_pivot = unweighted_pivot.reindex(base_order).dropna(how="all")

    plot_grouped_bars(
        unweighted_pivot,
        title="Friedman Statistic for k-NN Algorithms",
        filename="friedman_unweighted_nn_grouped.png",
    )
    return

def figure9():
    def sort_dict_for_keys(d):
        return [k for k, _ in sorted(d.items(), key=lambda kv: kv[1])]

    df = get_ts_df()
    average_rankings = df.groupby(["lens", "measure"])["ranking"].mean()

    one_nn_acc_ranks = average_rankings["1-NN Acc"].to_dict()
    one_nn_mcc_ranks = average_rankings["1-NN MCC"].to_dict()
    weighted_n1_nn_auroc_ranks = average_rankings["Weighted n-1-NN AUROC"].to_dict()
    spectral_clustering_ranks = average_rankings['Spectral Clustering'].to_dict()
    rnldd_ranks = average_rankings['RNLDD'].to_dict()
    aggregate_clustering_ranks = average_rankings['Aggregate Clustering'].to_dict()
    svc_loo_ranks = average_rankings['SVC-LOO'].to_dict()
    kmedoids_ranks = average_rankings["K-Medoids"].to_dict()

    baseline_top = set(sort_dict_for_keys(one_nn_acc_ranks)[:5])
    n_measures = len(one_nn_acc_ranks.keys())

    series = {
        "K-Medoids": kmedoids_ranks,
        "1-NN MCC": one_nn_mcc_ranks,
        "Weighted (n-1)-NN": weighted_n1_nn_auroc_ranks,
        "Spectral Clustering": spectral_clustering_ranks,
        "RNLDD": rnldd_ranks,
        "Aggregate Clustering": aggregate_clustering_ranks,
        "SVC": svc_loo_ranks,
    }

    perf_data = defaultdict(list)
    for name, ranks in series.items():
        num_hits = 0
        ordered_ranks = sort_dict_for_keys(ranks)
        for measure in ordered_ranks:
            num_hits += int(measure in baseline_top)
            perf_data[name].append(num_hits)

    df = pd.DataFrame(perf_data)
    df['Top X'] = np.arange(len(df)) + 1

    df_long = df.melt(
        id_vars="Top X",
        var_name="Diagnostic",
        value_name="Hits"
    )

    df_long["PercentBaseline"] = df_long['Hits'] / len(baseline_top) * 100

    plt.figure(figsize=(8, 5))
    ax = sns.lineplot(
        data=df_long,
        x="Top X",
        y="PercentBaseline",
        hue="Diagnostic",
        style='Diagnostic',
        drawstyle="steps-post"
    )

    ax.set_xlabel("Top-X Cutoff")
    ax.set_ylabel("Percent of 1-NN Accuracy's Top 5 Seen")

    ax.legend(title=None, frameon=False)
    plt.tight_layout()
    plt.savefig(cmn.DIRTS_DIR / "performance_profile.png")
    plt.show()

if __name__ == "__main__":
    df = do_analyses() # Figures 5-10
    # exit()
    # df = pd.read_csv(cmn.RESULT_DIR / "friedman_analyses.csv")
    knn_stat_power(df) # Figure 2

    from common import Lens as L

    order = list(L.all_names)
    order = [c for c in order if c != "RNLDD"] + ["RNLDD"]

    order = [c.replace("-acc", " Acc") for c in order]
    df['lens'] = df['lens'].str.replace('-acc', ' Acc', regex=False)

    df['lens'] = pd.Categorical(df['lens'], categories=order, ordered=True)
    df = df.sort_values('lens').reset_index(drop=True)

    df["Pval"] = np.clip(df["Pval"], 1e-16, None)


    algorithms = ["1-NN Acc", "1-NN MCC", "RNLDD", "SVC", "Aggregate Clustering", "K-Medoids",
                  "Weighted n-1-NN AUROC", "Spectral Clustering"]
    filtered_df = df[df['lens'].isin(algorithms)]
    filtered_df['lens'] = filtered_df['lens'].cat.remove_unused_categories()
    plot_dataset_count(filtered_df) # Figure 1
    plot_stat_power_no_filter(filtered_df) # Figure 3

    figure9()
    plt.show()
    exit()