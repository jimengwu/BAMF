import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pandas.plotting import register_matplotlib_converters
from scipy.stats import ks_2samp, binom_test, kruskal, pearsonr, spearmanr
from sklearn.metrics import mean_squared_error

register_matplotlib_converters()

plt.rcParams["figure.figsize"] = (40, 14)
plt.rcParams["font.size"] = 40
plt.tight_layout()

sns.set_style("whitegrid")
sns.set_context("poster", font_scale=2.0)
sns.set_palette(sns.color_palette("colorblind"))


def cal_metrics(data1, data2, metric):
    coeffs = []
    try:
        if len(data1.shape) == 1:
            data1 = data1.reshape(1, -1)
    except:
        raise ("input data is empty.")
    for i in range(data1.shape[0]):
        if metric == mean_squared_error:
            coeff = metric(data2, data1[i, :], squared=True)
        else:
            coeff, _ = metric(data2, data1[i, :])
        coeffs.append(coeff)
    return coeffs


def plot_benchmark(data, palette=None, label="G"):

    if label == "F":
        idx_factor = 0
        factor_position = data.index
        x_position = data.columns
    else:
        idx_factor = 1
        factor_position = data.columns
        x_position = data.index

    fig, axes = plt.subplots(data.shape[idx_factor], sharex=True)
    if palette is None:
        palette = sns.color_palette(n_colors=len(axes))

    for i, x in enumerate(axes):
        ylabel = np.array(factor_position)[i]
        if label == "F":
            x.bar(
                np.array(x_position),
                np.array(data[data.index == ylabel]).ravel(),
                label=np.array(factor_position)[i],
                color=palette[i],
            )

        else:
            x.plot(data.index, data[ylabel], "--", label=ylabel, color=palette[i])

        twin = axes[i].twinx()

        twin.get_yaxis().set_ticks([])  # hiding the ticks

        x.set_ylim([0, x.get_ylim()[1]])  # set the y value axis range

        axes[i].tick_params(axis="y", labelsize=22)
        axes[i].legend(loc="upper right", fontsize=18)
        for tick in x.xaxis.get_major_ticks():
            tick.label.set_fontsize(20)


#     return fig


def plot_diagnostic(di, name, thresh=0.05):

    n = int(di.shape[0] / 2)
    half = n + int(n / 2)

    # first half part of sampling draws
    first = di[n:half, :].ravel()
    second = di[half:, :].ravel()

    ks, p = ks_2samp(first, second)

    fig = plt.figure()

    ax = sns.kdeplot(first, shade=True, alpha=0.5, color="r", label="first half")
    ax = sns.kdeplot(second, shade=True, alpha=0.5, color="b", label="second half")

    if p < thresh:
        textcol = "r"
    else:
        textcol = "g"

    ax.set_title(f"{name}, p-value:{p:1.2f}", c=textcol)
    ax.legend()
    return fig, p


def errorbar_batched(
    data, labels, title, n_rows=5, xlabel="", palette=None, fontsize=22, label="factor"
):

    fig, axes = plt.subplots(n_rows, sharex=True)

    if palette is None:
        palette = sns.color_palette(n_colors=len(axes))

    axes[0].set_title(title)

    if label == "factor":
        quants = np.quantile(data["F"], q=[0.25, 0.5, 0.75], axis=0)
        for i, x in enumerate(axes):

            bp = axes[i].errorbar(
                data["variable"],
                quants[1, i, :],
                np.abs(quants[[0, 2], i, :] - quants[1, i, :]),
                color=palette[i],
                marker="o",
                markeredgecolor=palette[i],
                capthick=1,
                alpha=0.75,
                linewidth=0,
                elinewidth=1,
                ecolor=palette[i],
                capsize=3,
                markersize=5,
            )
            plt.xlabel(xlabel)
            twin = axes[i].twinx()
            twin.get_yaxis().set_ticks([])

            axes[i].legend([labels[i]], loc="upper right", fontsize=fontsize)
            axes[i].tick_params(axis="y", labelsize=fontsize)

        plt.xticks(rotation=90)
    elif label == "time":
        quants = np.quantile(data["G"], q=[0.25, 0.5, 0.75], axis=0)
        for i, x in enumerate(axes):
            # plot model results
            x.plot(
                data["timestep"],
                quants[1, :, i],
                "x",
                color=palette[i],
                markersize=(150 / fig.dpi) ** 2,
            )
            x.fill_between(
                np.array(data["timestep"]),
                quants[0, :, i],
                quants[-1, :, i],
                alpha=0.5,
                color=palette[i],
            )
    return fig, axes


def plot_F(data, benchmark=False, extra_data=None):

    fig, axes = errorbar_batched(
        data,
        labels=np.array(data["labels"]),
        title="F",
        n_rows=data["G"].shape[-1],
        label="factor",
    )

    for i, x in enumerate(axes):
        ylabel = str(np.array(data["labels"])[i])
        if benchmark and "F_orig" in data.keys():
            if np.array(data["labels"])[i] in np.array(data["orig_labels"]):
                orig_index = np.where(
                    np.array(data["orig_labels"]) == np.array(data["labels"])[i]
                )[0]

                x.bar(
                    np.array(data["variable"]),
                    np.array(data["F_orig"])[orig_index, :].ravel(),
                    width=0.5,
                    color="k",
                    alpha=0.6,
                    label="reference",
                )

                coeff_spear = cal_metrics(
                    np.array(data["F"])[:, i, :],
                    np.array(data["F_orig"])[orig_index, :].ravel(),
                    spearmanr,
                )

                coeff_rmse = cal_metrics(
                    np.array(data["F"])[:, i, :],
                    np.array(data["F_orig"])[orig_index, :].ravel(),
                    mean_squared_error,
                )

                ylabel = (
                    str(np.array(data["labels"])[i])
                    + "\nSpearman's ρ:\n"
                    + "{:.4f}".format(np.mean(coeff_spear))
                    + " +/- "
                    + "{:.4f}".format(np.std(coeff_spear))
                    + "\nRMSE:\n"
                    + "{:.4f}".format(np.mean(coeff_rmse))
                    + " +/- "
                    + "{:.4f}".format(np.std(coeff_rmse))
                )
        # set the subplot's y label
        twin = axes[i].twinx()
        twin.set_ylabel(
            # ylabel,
            # labelpad=1,
            # fontsize=18,
            # rotation="horizontal",
            # horizontalalignment="left",
            # wrap=True,
            # loc="top",
            ylabel,
            #             labelpad=1,
            fontsize=18,
            rotation="horizontal",
            horizontalalignment="left",
            wrap=True,
            verticalalignment="center_baseline",
        )
        twin.get_yaxis().set_ticks([])  # hiding the ticks
        x.set_xticks(data["variable"], minor=True)

        for tick in x.xaxis.get_major_ticks():
            tick.label.set_fontsize(18)
            tick.label.set_rotation(90)

    return fig


def plot_individual_F(data, compound_idx, title="F", benchmark=False, sample_size=7):

    fig, axes = plt.subplots(sample_size, 1, sharex=True)

    compound = data["labels"][compound_idx]

    if benchmark:
        orig_index = np.where(np.array(data["orig_labels"]) == compound)[0]
        title = title + ", " + str(np.array(data["orig_labels"])[orig_index])
    else:
        title = title + f", {compound}"

    colors = sns.color_palette(n_colors=len(axes))

    for i, x in enumerate(axes):
        ylabel = "sample " + str(i)
        x.plot(
            np.array(data["variable"]),
            np.array(data["F"])[-i, compound_idx, :],
            linestyle="",
            marker="o",
            color=colors[i],
            alpha=0.5,
            label=f"{i}",
        )

        if benchmark and compound in np.array(data["orig_labels"]):
            orig_index = np.where(np.array(data["orig_labels"]) == compound)[0]

            x.bar(
                np.array(data["variable"]),
                np.array(data["F_orig"])[orig_index, :].ravel(),
                width=0.5,
                color="k",
                alpha=0.25,
            )

            coeff_spear = cal_metrics(
                np.array(data["F"])[-i, compound_idx, :],
                np.array(data["F_orig"])[orig_index, :].ravel(),
                spearmanr,
            )

            coeff_rmse = cal_metrics(
                np.array(data["F"])[-i, compound_idx, :],
                np.array(data["F_orig"])[orig_index, :].ravel(),
                mean_squared_error,
            )

            ylabel = (
                "sample "
                + str(i)
                + "\nSpearman's ρ:"
                + "{:.4f}".format(np.mean(coeff_spear))
                + "\nRMSE:"
                + "{:.4f}".format(np.mean(coeff_rmse))
            )
        twin = axes[i].twinx()
        twin.set_ylabel(
            # ylabel,
            # labelpad=len("sample " + str(i)),
            # fontsize=18,
            # rotation="horizontal",
            # horizontalalignment="left",
            # wrap=True,
            # loc="top",
            ylabel,
            #             labelpad=1,
            fontsize=18,
            rotation="horizontal",
            horizontalalignment="left",
            wrap=True,
            verticalalignment="center_baseline",
        )
        twin.get_yaxis().set_ticks([])

        x.set_xticks(data["variable"], minor=True)
        axes[i].tick_params(axis="y", labelsize=22)
        axes[i].legend(loc="upper right", fontsize=18)

        for tick in x.xaxis.get_major_ticks():
            tick.label.set_fontsize(18)
            tick.label.set_rotation(90)

    return fig


def plot_G(data, title="G", benchmark=False, quantiles=[0.25, 0.5, 0.75]):

    fig, axes = errorbar_batched(
        data,
        labels=np.array(data["labels"]),
        title=title,
        n_rows=data["G"].shape[-1],
        label="time",
    )
    # plot benchmark

    for i, x in enumerate(axes):
        ylabel = str(np.array(data["labels"])[i])
        if benchmark and np.array(data["labels"])[i] in np.array(data["orig_labels"]):

            orig_index = np.where(
                np.array(data["orig_labels"]) == np.array(data["labels"])[i]
            )[0]
            x.plot(
                data["timestep"],
                data["G_orig"][:, orig_index],
                "k--",
                label="Truth",
            )

            # pearson coeff notation

            coeff_pear = cal_metrics(
                np.array(data["G"])[:, :, i],
                np.array(data["G_orig"])[:, orig_index].ravel(),
                pearsonr,
            )

            coeff_rmse = cal_metrics(
                np.array(data["G"])[:, :, i],
                np.array(data["G_orig"])[:, orig_index].ravel(),
                mean_squared_error,
            )

            ylabel = (
                str(np.array(data["labels"])[i])
                + "\nPearson's r:\n"
                + "{:.4f}".format(np.mean(coeff_pear))
                + " +/- "
                + "{:.4f}".format(np.std(coeff_pear))
                + "\nRMSE:\n"
                + "{:.4f}".format(np.mean(coeff_rmse))
                + " +/- "
                + "{:.4f}".format(np.std(coeff_rmse))
            )

            # set the subplot's y label
        twin = axes[i].twinx()
        twin.set_ylabel(
            # data["labels"][i] + "\nPearson's r: " + "{:.4f}".format(coeff),
            # ylabel,
            # labelpad=1,
            # fontsize=18,
            # rotation="horizontal",
            # horizontalalignment="left",
            # wrap=True,
            # loc="top",
            ylabel,
            #             labelpad=1,
            fontsize=18,
            rotation="horizontal",
            horizontalalignment="left",
            wrap=True,
            verticalalignment="center_baseline",
        )
        twin.get_yaxis().set_ticks([])  # hiding the ticks

        x.set_ylim([0, x.get_ylim()[1]])  # set the y value axis range

        axes[i].tick_params(axis="y", labelsize=22)
        axes[i].legend(loc="upper right", fontsize=18)
        for tick in x.xaxis.get_major_ticks():
            tick.label.set_fontsize(20)
    fig.autofmt_xdate()
    return fig


def plot_individual_G(data, title="G", benchmark=False, sample_size=7):
    """
    plot the multiple G results on each sub plot.
    """

    fig, axes = plt.subplots(nrows=data["G"].shape[-1], sharex=True)

    axes[0].set_title(title)

    colors = sns.color_palette(n_colors=sample_size)

    for i, x in enumerate(axes):
        orig_index = np.where(
            np.array(data["orig_labels"]) == np.array(data["labels"])[i]
        )[0]

        for j in range(sample_size):

            x.plot(
                data["timestep"],
                data["G"][-j, :, i],
                "x--",
                color=colors[j],
                alpha=0.5,
                label=f"{j}",
            )

        if benchmark:
            x.plot(
                data["timestep"], data["G_orig"][:, orig_index], "k--", label="Truth"
            )
            # pearson coeff notation

        twin = axes[i].twinx()
        twin.set_ylabel(
            data["labels"][i],
            # labelpad=len(data["labels"][i]),
            # # rotation=90,
            # fontsize=18,
            # rotation="horizontal",
            # horizontalalignment="left",
            # wrap=True,
            # loc="top",
            # ylabel,
            #             labelpad=1,
            fontsize=18,
            rotation="horizontal",
            horizontalalignment="left",
            wrap=True,
            verticalalignment="center_baseline",
        )
        twin.get_yaxis().set_ticks([])

        x.set_ylim([0, x.get_ylim()[1]])
        axes[i].tick_params(axis="y", labelsize=22)
        axes[i].legend(loc="upper right", fontsize=18)
    for tick in x.xaxis.get_major_ticks():
        tick.label.set_fontsize(18)
        tick.label.set_rotation(90)

    fig.autofmt_xdate()

    return fig


def plot_median_G(data, title="G_median", benchmark=False, quantiles=[0.5]):

    q = np.quantile(data["G"], quantiles, axis=0)

    fig, axes = plt.subplots(nrows=data["G"].shape[-1], sharex=True)

    axes[0].set_title(title)

    colors = sns.color_palette(n_colors=len(axes))

    for i, x in enumerate(axes):
        ylabel = str(np.array(data["labels"])[i])
        x.plot(
            data["timestep"],
            q[0, :, i],
            "--",
            markersize=(150 / fig.dpi) ** 2,
            color=colors[i],
        )

        if benchmark:
            if np.array(data["labels"])[i] in np.array(data["orig_labels"]):
                orig_index = np.where(
                    np.array(data["orig_labels"]) == np.array(data["labels"])[i]
                )[0]
                x.plot(
                    data["timestep"],
                    data["G_orig"][:, orig_index],
                    "k--",
                    label="Truth",
                )

                coeff_pear = cal_metrics(
                    np.array(data["G_orig"])[:, orig_index].ravel(),
                    q[0, :, i],
                    pearsonr,
                )
                coeff_mse = cal_metrics(
                    np.array(data["G_orig"])[:, orig_index].ravel(),
                    q[0, :, i],
                    mean_squared_error,
                )
                ylabel = (
                    str(np.array(data["labels"])[i])
                    + "\nPearson's r:"
                    + "{:.4f}".format(coeff_pear[0])
                    + "\nRMSE:"
                    + "{:.4f}".format(coeff_mse[0])
                )

        # set the subplot's y label
        twin = axes[i].twinx()
        twin.set_ylabel(
            # ylabel,
            # labelpad=1,
            # fontsize=18,
            # rotation="horizontal",
            # horizontalalignment="left",
            # loc="top",
            ylabel,
            #             labelpad=1,
            fontsize=18,
            rotation="horizontal",
            horizontalalignment="left",
            wrap=True,
            verticalalignment="center_baseline",
        )
        twin.get_yaxis().set_ticks([])  # hiding the ticks

        x.set_ylim([0, x.get_ylim()[1]])  # set the y value axis range

        axes[i].tick_params(axis="y", labelsize=22)
        axes[i].legend(loc="upper right", fontsize=18)
    for tick in x.xaxis.get_major_ticks():
        tick.label.set_fontsize(20)

    fig.autofmt_xdate()

    return fig


def mass_conservation(data, sample=-1):

    total_orig = data["X"].sum(axis=-1)
    if isinstance(sample, int):
        # total = data["G"].sum(axis=-1)[sample, ...]
        total = data["G"][sample, :].sum(axis=-1)
        title = "Mass conservation from last sample"
    elif sample == "median":
        total = np.median(data["G"], axis=0).sum(axis=-1)
        # np.median(data["G"].sum(axis=-1), axis=0)
        title = "Mass conservation from median G value"

    fig, axes = plt.subplots(3, 1)

    axes[0].set_title(title)

    axes[0].scatter(total_orig, total, s=20)
    axes[1].scatter(total_orig, total / total_orig, s=20)
    axes[2].scatter(total_orig, total / total_orig, s=20)
    axes[1].axhline(y=1, color="r", linestyle="--", alpha=0.8)
    axes[2].axhline(y=1, color="r", linestyle="--", alpha=0.8)

    axes[2].set_xscale("log")
    axes[0].set_ylabel("Sum(G)", fontsize=20)
    axes[1].set_ylabel("Sum(G)/\nsum(Input)", fontsize=20)
    axes[2].set_ylabel("Sum(G)/\nsum(Input)", fontsize=20)
    axes[2].set_xlabel("Sum(Input)", fontsize=20)
    for i in range(3):
        axes[i].tick_params(axis="y", labelsize=18)
        axes[i].tick_params(axis="x", labelsize=18)

    return fig


def cal_errorbar(data, q=[0.25, 0.5, 0.75]):

    quants = np.quantile(data, q, axis=0)
    return quants


def mass_conservation_comparison(data1, data2, sample):

    total_orig = data1["X"].sum(axis=-1)
    if isinstance(sample, int):
        # total = data["G"].sum(axis=-1)[sample, ...]
        total = data1["G"][sample, :].sum(axis=-1)
        total2 = data2["G"][sample, :].sum(axis=-1)
        title = "Mass conservation from the Nr. " + str(sample) + " sample"
    elif sample == "median":
        total = np.median(data1["G"], axis=0).sum(axis=-1)
        total2 = np.median(data2["G"], axis=0).sum(axis=-1)
        # np.median(data["G"].sum(axis=-1), axis=0)
        title = "Mass conservation from median G value"

    fig, axes = plt.subplots(3, 1)

    axes[0].set_title(title)

    axes[0].scatter(total_orig, total, s=20, color="blue", label="BAMF")
    axes[1].scatter(total_orig, total / total_orig, s=20, color="blue", label="BAMF")
    axes[2].scatter(total_orig, total / total_orig, s=20, color="blue", label="BAMF")

    axes[0].scatter(total_orig, total2, s=20, color="green", label="PMF")
    axes[1].scatter(total_orig, total2 / total_orig, s=20, color="green", label="PMF")
    axes[2].scatter(total_orig, total2 / total_orig, s=20, color="green", label="PMF")

    axes[1].axhline(y=1, color="r", linestyle="--", alpha=0.8)
    axes[2].axhline(y=1, color="r", linestyle="--", alpha=0.8)

    axes[2].set_xscale("log")
    axes[0].set_ylabel("Sum(G)", fontsize=20)
    axes[1].set_ylabel("Sum(G)/\nsum(Input)", fontsize=20)
    axes[2].set_ylabel("Sum(G)/\nsum(Input)", fontsize=20)
    axes[2].set_xlabel("Sum(Input)", fontsize=20)
    for i in range(3):
        axes[i].legend(loc="upper right", fontsize=18)
        axes[i].tick_params(axis="y", labelsize=18)
        axes[i].tick_params(axis="x", labelsize=18)

    return fig


def errorbar_batched_comparison(
    data_bamf,
    data_pmf,
    labels,
    title,
    n_rows=5,
    xlabel="",
    palette=None,
    fontsize=22,
    label="factor",
):

    fig, axes = plt.subplots(n_rows, sharex=True)

    if palette is None:
        palette = sns.color_palette(n_colors=len(axes) * 2)

    axes[0].set_title(title)

    if label == "factor":
        quants_bamf = np.quantile(data_bamf["F"], q=[0.25, 0.5, 0.75], axis=0)
        quants_pmf = np.quantile(data_pmf["F"], q=[0.25, 0.5, 0.75], axis=0)
        for i, x in enumerate(axes):

            bp = axes[i].errorbar(
                data_bamf["variable"],
                quants_bamf[1, i, :],
                np.abs(quants_bamf[[0, 2], i, :] - quants_bamf[1, i, :]),
                color=palette[i],
                marker="o",
                markeredgecolor=palette[i],
                label="BAMF",
                capthick=1,
                alpha=0.75,
                linewidth=0,
                elinewidth=1,
                ecolor=palette[i],
                capsize=3,
                markersize=5,
            )
            bp = axes[i].errorbar(
                data_pmf["variable"],
                quants_pmf[1, i, :],
                np.abs(quants_pmf[[0, 2], i, :] - quants_pmf[1, i, :]),
                color=palette[i + len(axes)],
                marker="o",
                markeredgecolor=palette[i + len(axes)],
                capthick=1,
                alpha=0.75,
                linewidth=0,
                label="PMF",
                elinewidth=1,
                ecolor=palette[i + len(axes)],
                capsize=3,
                markersize=5,
            )
            #             plt.xlabel(xlabel)
            twin = axes[i].twinx()
            twin.get_yaxis().set_ticks([])

            axes[i].legend(loc="upper right", fontsize=fontsize)
            axes[i].tick_params(axis="y", labelsize=fontsize)

        plt.xticks(rotation=90)
    elif label == "time":
        quants_bamf = np.quantile(data_bamf["G"], q=[0.25, 0.5, 0.75], axis=0)
        quants_pmf = np.quantile(data_pmf["G"], q=[0.25, 0.5, 0.75], axis=0)
        for i, x in enumerate(axes):
            # plot model results
            x.plot(
                data_bamf["timestep"],
                quants_bamf[1, :, i],
                "x",
                color=palette[i],
                label="BAMF",
                markersize=(150 / fig.dpi) ** 2,
            )
            x.fill_between(
                np.array(data_bamf["timestep"]),
                quants_bamf[0, :, i],
                quants_bamf[-1, :, i],
                alpha=0.5,
                color=palette[i],
            )

            x.plot(
                data_pmf["timestep"],
                quants_pmf[1, :, i],
                "x",
                color=palette[i + len(axes)],
                label="PMF",
                markersize=(150 / fig.dpi) ** 2,
            )
            x.fill_between(
                np.array(data_pmf["timestep"]),
                quants_pmf[0, :, i],
                quants_pmf[-1, :, i],
                alpha=0.5,
                color=palette[i + len(axes)],
            )

            twin = axes[i].twinx()
            twin.get_yaxis().set_ticks([])

            axes[i].legend(loc="upper right", fontsize=fontsize)
            axes[i].tick_params(axis="y", labelsize=fontsize)
    return fig, axes


def plot_F_comparison(data_bamf, data_pmf, benchmark=False, extra_data=None):

    fig, axes = errorbar_batched_comparison(
        data_bamf,
        data_pmf,
        labels=np.array(data_bamf["labels"]),
        title="F",
        n_rows=data_bamf["G"].shape[-1],
        label="factor",
    )

    for i, x in enumerate(axes):
        ylabel = str(np.array(data_pmf["labels"])[i])
        if benchmark and "F_orig" in data_bamf.keys():
            if np.array(data_bamf["labels"])[i] in np.array(data_bamf["orig_labels"]):
                orig_index = np.where(
                    np.array(data_bamf["orig_labels"])
                    == np.array(data_bamf["labels"])[i]
                )[0]

                x.bar(
                    data_bamf["variable"],
                    np.array(data_bamf["F_orig"])[orig_index, :].ravel(),
                    width=0.5,
                    color="k",
                    alpha=0.6,
                    label="reference",
                )

                coeffs_spear_pmf = cal_metrics(
                    np.array(data_pmf["F"])[:, i, :],
                    np.array(data_pmf["F_orig"])[orig_index, :].ravel(),
                    spearmanr,
                )

                coeffs_spear_bamf = cal_metrics(
                    np.array(data_bamf["F"])[:, i, :],
                    np.array(data_bamf["F_orig"])[orig_index, :].ravel(),
                    spearmanr,
                )

                coeff_rmse_pmf = cal_metrics(
                    np.array(data_pmf["F"])[:, i, :],
                    np.array(data_pmf["F_orig"])[orig_index, :].ravel(),
                    mean_squared_error,
                )

                coeff_rmse_bamf = cal_metrics(
                    np.array(data_bamf["F"])[:, i, :],
                    np.array(data_bamf["F_orig"])[orig_index, :].ravel(),
                    mean_squared_error,
                )

                ylabel = (
                    str(np.array(data_pmf["labels"])[i])
                    + "\nBAMF_Spearman's ρ:"
                    + "{:.4f}".format(np.mean(coeffs_spear_bamf))
                    + " +/- "
                    + "{:.4f}".format(np.std(coeffs_spear_bamf))
                    + "\nBAMF_RMSE:"
                    + "{:.4f}".format(np.mean(coeff_rmse_bamf))
                    + " +/- "
                    + "{:.4f}".format(np.std(coeff_rmse_bamf))
                    + "\nPMF_Spearman's ρ:"
                    + "{:.4f}".format(np.mean(coeffs_spear_pmf))
                    + " +/- "
                    + "{:.4f}".format(np.std(coeffs_spear_pmf))
                    + "\nPMF_RMSE:"
                    + "{:.4f}".format(np.mean(coeff_rmse_pmf))
                    + " +/- "
                    + "{:.4f}".format(np.std(coeff_rmse_pmf))
                )
        # set the subplot's y label
        twin = axes[i].twinx()

        twin.set_ylabel(
            # ylabel,
            # labelpad=1,
            # fontsize=18,
            # rotation="horizontal",
            # horizontalalignment="left",
            # wrap=True,
            # loc="top",
            ylabel,
            #             labelpad=1,
            fontsize=18,
            rotation="horizontal",
            horizontalalignment="left",
            wrap=True,
            verticalalignment="center_baseline",
        )

        twin.get_yaxis().set_ticks([])  # hiding the ticks
        x.set_xticks(data_bamf["variable"], minor=True)

        for tick in x.xaxis.get_major_ticks():
            tick.label.set_fontsize(18)
            tick.label.set_rotation(90)
    return fig


def plot_G_comparison(
    data_bamf, data_pmf, title="G", benchmark=False, quantiles=[0.25, 0.5, 0.75]
):

    fig, axes = errorbar_batched_comparison(
        data_bamf,
        data_pmf,
        labels=np.array(data_bamf["labels"]),
        title=title,
        n_rows=data_bamf["G"].shape[-1],
        label="time",
    )
    # plot benchmark

    for i, x in enumerate(axes):
        ylabel = str(np.array(data_pmf["labels"])[i])
        if benchmark and np.array(data_pmf["labels"])[i] in np.array(
            data_pmf["orig_labels"]
        ):

            orig_index = np.where(
                np.array(data_bamf["orig_labels"]) == np.array(data_bamf["labels"])[i]
            )[0]

            x.plot(
                data_bamf["timestep"],
                data_bamf["G_orig"][:, orig_index],
                "k--",
                label="Truth",
            )

            # pearson coeff notation

            coeffs_pear_pmf = cal_metrics(
                np.array(data_pmf["G"])[:, :, i],
                np.array(data_pmf["G_orig"])[:, orig_index].ravel(),
                pearsonr,
            )

            coeffs_pear_bamf = cal_metrics(
                np.array(data_bamf["G"])[:, :, i],
                np.array(data_bamf["G_orig"])[:, orig_index].ravel(),
                pearsonr,
            )

            coeff_rmse_pmf = cal_metrics(
                np.array(data_pmf["G"])[:, :, i],
                np.array(data_pmf["G_orig"])[:, orig_index].ravel(),
                mean_squared_error,
            )

            coeff_rmse_bamf = cal_metrics(
                np.array(data_pmf["G"])[:, :, i],
                np.array(data_pmf["G_orig"])[:, orig_index].ravel(),
                mean_squared_error,
            )

            ylabel = (
                str(np.array(data_pmf["labels"])[i])
                + "\nBAMF_Spearman's ρ:"
                + "{:.4f}".format(np.mean(coeffs_pear_bamf))
                + " +/- "
                + "{:.4f}".format(np.std(coeffs_pear_bamf))
                + "\nBAMF_RMSE:"
                + "{:.4f}".format(np.mean(coeff_rmse_bamf))
                + " +/- "
                + "{:.4f}".format(np.std(coeff_rmse_bamf))
                + "\nPMF_Spearman's ρ:"
                + "{:.4f}".format(np.mean(coeffs_pear_pmf))
                + " +/- "
                + "{:.4f}".format(np.std(coeffs_pear_pmf))
                + "\nPMF_RMSE:"
                + "{:.4f}".format(np.mean(coeff_rmse_pmf))
                + " +/- "
                + "{:.4f}".format(np.std(coeff_rmse_pmf))
            )

        # set the subplot's y label
        twin = axes[i].twinx()
        twin.set_ylabel(
            # data["labels"][i] + "\nPearson's r: " + "{:.4f}".format(coeff),
            # ylabel,
            # labelpad=len(np.array(data_pmf["labels"])[i]),
            # fontsize=18,
            # rotation="horizontal",
            # horizontalalignment="left",
            # wrap=True,
            # loc="top",
            ylabel,
            #             labelpad=1,
            fontsize=18,
            rotation="horizontal",
            horizontalalignment="left",
            wrap=True,
            verticalalignment="center_baseline",
        )
        twin.get_yaxis().set_ticks([])  # hiding the ticks

        x.set_ylim([0, x.get_ylim()[1]])  # set the y value axis range

        axes[i].tick_params(axis="y", labelsize=22)
        axes[i].legend(loc="upper right", fontsize=18)
        for tick in x.xaxis.get_major_ticks():
            tick.label.set_fontsize(20)
    fig.autofmt_xdate()
    return fig


def plot_median_G_comparison(
    data_bamf, data_pmf, title="G_median", benchmark=False, quantiles=[0.5]
):

    q_bamf = np.quantile(data_bamf["G"], quantiles, axis=0)
    q_pmf = np.quantile(data_pmf["G"], quantiles, axis=0)

    fig, axes = plt.subplots(nrows=data_bamf["G"].shape[-1], sharex=True)

    axes[0].set_title(title)

    colors = sns.color_palette(n_colors=len(axes) * 2)

    for i, x in enumerate(axes):
        ylabel = str(np.array(data_bamf["labels"])[i])
        x.plot(
            data_bamf["timestep"],
            q_bamf[0, :, i],
            "o",
            markersize=(150 / fig.dpi) ** 2,
            color=colors[i],
            label="BAMF",
        )
        x.plot(
            data_pmf["timestep"],
            q_pmf[0, :, i],
            "x",
            markersize=(150 / fig.dpi) ** 2,
            color=colors[i + len(axes)],
            label="PMF",
        )

        if benchmark:
            if np.array(data_bamf["labels"])[i] in np.array(data_bamf["orig_labels"]):
                orig_index = np.where(
                    np.array(data_bamf["orig_labels"])
                    == np.array(data_bamf["labels"])[i]
                )[0]
                x.plot(
                    data_bamf["timestep"],
                    data_bamf["G_orig"][:, orig_index],
                    "k--",
                    label="Truth",
                )

                coeff_pearson_bamf = cal_metrics(
                    np.array(data_bamf["G_orig"])[:, orig_index].ravel(),
                    q_bamf[0, :, i],
                    pearsonr,
                )
                coeff_pearson_pmf = cal_metrics(
                    np.array(data_pmf["G_orig"])[:, orig_index].ravel(),
                    q_pmf[0, :, i],
                    pearsonr,
                )

                coeff_mse_bamf = cal_metrics(
                    q_bamf[0, :, i],
                    np.array(data_bamf["G_orig"])[:, orig_index].ravel(),
                    mean_squared_error,
                )
                coeff_mse_pmf = cal_metrics(
                    q_pmf[0, :, i],
                    np.array(data_pmf["G_orig"])[:, orig_index].ravel(),
                    mean_squared_error,
                )
                ylabel = (
                    str(np.array(data_bamf["labels"])[i])
                    + "\nBAMF_Pearson's r: "
                    + "{:.4f}".format(np.mean(coeff_pearson_bamf))
                    + "\nBAMF_RMSE: "
                    + "{:.4f}".format(np.mean(coeff_mse_bamf))
                    + "\nPMF_Pearson's r: "
                    + "{:.4f}".format(np.mean(coeff_pearson_pmf))
                    + "\nPMF_RMSE: "
                    + "{:.4f}".format(np.mean(coeff_mse_pmf))
                )

        # set the subplot's y label
        twin = axes[i].twinx()
        twin.set_ylabel(
            # ylabel,
            # labelpad=1,
            # fontsize=18,
            # rotation="horizontal",
            # horizontalalignment="left",
            # loc="top",
            ylabel,
            #             labelpad=1,
            fontsize=18,
            rotation="horizontal",
            horizontalalignment="left",
            wrap=True,
            verticalalignment="center_baseline",
        )
        twin.get_yaxis().set_ticks([])  # hiding the ticks

        x.set_ylim([0, x.get_ylim()[1]])  # set the y value axis range

        axes[i].tick_params(axis="y", labelsize=22)
        axes[i].legend(loc="upper right", fontsize=18)
    for tick in x.xaxis.get_major_ticks():
        tick.label.set_fontsize(20)

    fig.autofmt_xdate()

    return fig


# ---------------------------old version-----------------


def errorbar_batched_factors_xarray_old(
    mz, sample1, labels, title, n_factors=5, xlabel="", palette=None, fontsize=12
):

    fig, axes = plt.subplots(nrows=n_factors, sharex=True)

    if palette is None:
        palette = sns.color_palette(n_colors=len(axes))

    axes[0].set_title(title)
    for i in range(n_factors):

        c = palette[i]

        data = sample1[:, i, :]
        # .squeeze()
        quants = np.quantile(data, q=[0.25, 0.5, 0.75], axis=0)
        # quants = data.quantile(dim="sample", q=[0.25, 0.5, 0.75]).to_numpy()
        # print(quants.shape)

        bp = axes[i].errorbar(
            mz,
            quants[1, :],
            np.abs(quants[[0, 2], :] - quants[1, :]),
            color=c,
            marker="o",
            markeredgecolor=c,
            capthick=1,
            alpha=0.75,
            linestyle="",
            linewidth=0,
            elinewidth=0.2,
            ecolor=c,
            capsize=5,
        )

        # plt.xticks(mz, fontsize=fontsize)
        plt.xlabel(xlabel)
        twin = axes[i].twinx()
        # twin.set_ylabel(labels[i], labelpad=len(labels[i]), rotation=90)
        twin.get_yaxis().set_ticks([])

        axes[i].legend([labels[i]], loc="upper right", fontsize=22)
        axes[i].tick_params(axis="y", labelsize=22)

        # axes[i].legend([labels[i]], loc="upper right")

    plt.xticks(rotation=90)
    return fig, axes


def F_xarray_old(data, title="F", benchmark=True):
    mz = data["F"]["variable"].to_numpy()
    y_value = data["F"].to_numpy()

    labels = data.labels.to_numpy()

    fig, axes = errorbar_batched_factors_xarray_old(
        mz,
        y_value,
        labels=labels,
        title="F",
        n_factors=len(data.labels),
    )

    for i, x in enumerate(axes):
        ylabel = data["labels"].values[i]
        # if "F_orig" in data.keys():
        if benchmark and data["labels"][i].item() in data["orig_labels"]:
            # if data["F"]["component"][i] in data["F_orig"].index:
            orig_index = np.where(data["orig_labels"] == data["labels"].values[i])[0]
            x.bar(
                data["F"]["variable"],
                data["F_orig"].values[orig_index, :].ravel(),
                width=0.5,
                color="k",
                alpha=0.25,
                label="reference",
            )

            x.set_xticks(data["F"]["variable"], minor=True)
            coeffs_spear = cal_metrics(
                np.array(data["F"])[:, i, :],
                np.array(data["F_orig"])[orig_index, :].ravel(),
                spearmanr,
            )

            coeff_rmse = cal_metrics(
                np.array(data["F"])[:, i, :],
                np.array(data["F_orig"])[orig_index, :].ravel(),
                mean_squared_error,
            )

            ylabel = (
                data["labels"].values[i]
                + "\nSpearman's ρ:\n"
                + "{:.4f}".format(np.mean(coeffs_spear))
                + " +/- "
                + "{:.4f}".format(np.std(coeffs_spear))
                + "\nRMSE:\n"
                + "{:.4f}".format(np.mean(coeff_rmse))
                + " +/- "
                + "{:.4f}".format(np.std(coeff_rmse))
            )

        twin = axes[i].twinx()
        # print(data["labels"].values[i])
        twin.set_ylabel(
            # ylabel,
            # labelpad=len(data["labels"].values[i]),
            # fontsize=18,
            # rotation="horizontal",
            # horizontalalignment="left",
            # wrap=True,
            # loc="top",
            ylabel,
            #             labelpad=1,
            fontsize=18,
            rotation="horizontal",
            horizontalalignment="left",
            wrap=True,
            verticalalignment="center_baseline",
        )
        twin.get_yaxis().set_ticks([])  # hiding the ticks
        for tick in x.xaxis.get_major_ticks():
            tick.label.set_fontsize(16)
            tick.label.set_rotation(90)

    return fig


def G_xarray_old(data, title="G", benchmark=False, quantiles=[0.025, 0.5, 0.975]):

    q = data["G"].quantile(dim="samples", q=quantiles).to_numpy()

    fig, axes = plt.subplots(nrows=len(data.labels), sharex=True)

    axes[0].set_title(title)

    colors = sns.color_palette(n_colors=len(axes))

    for i, x in enumerate(axes):
        ylabel = data["labels"].values[i]
        x.plot(
            data["timestep"],
            q[1, :, i],
            "x",
            markersize=(150 / fig.dpi) ** 2,
            color=colors[i],
            label=data["labels"][i].item()
            #             label="run"
        )
        x.fill_between(
            data["timestep"].to_numpy(),
            q[0, :, i],
            q[-1, :, i],
            alpha=0.5,
            color=colors[i],
        )

        if benchmark:
            if data["labels"][i].item() in data["orig_labels"]:

                x.plot(
                    data["timestep"],
                    data["G_orig"].loc[:, data["labels"][i].item()],
                    "k--",
                    label="Truth",
                )
                coeffs_pear = cal_metrics(
                    #                     q[1, :, i],
                    np.array(data["G"])[:, :, i],
                    np.array(data["G_orig"])[:, data["labels"][i].item()],
                    pearsonr,
                )

                coeff_rmse = cal_metrics(
                    np.array(data["G"])[:, :, i],
                    np.array(data["G_orig"])[:, data["labels"][i].item()],
                    mean_squared_error,
                )

                ylabel = (
                    data["labels"].values[i]
                    + "\nSpearman's ρ:\n"
                    + "{:.4f}".format(np.mean(coeffs_pear))
                    + " +/- "
                    + "{:.4f}".format(np.std(coeffs_pear))
                    + "\nRMSE:\n"
                    + "{:.4f}".format(np.mean(coeff_rmse))
                    + " +/- "
                    + "{:.4f}".format(np.std(coeff_rmse))
                )

                twin = axes[i].twinx()
                twin.set_ylabel(
                    # ylabel,
                    # labelpad=len(data["labels"][i].item()),
                    # rotation="horizontal",
                    # fontsize=18,
                    # horizontalalignment="left",
                    # loc="top",
                    ylabel,
                    #             labelpad=1,
                    fontsize=18,
                    rotation="horizontal",
                    horizontalalignment="left",
                    wrap=True,
                    verticalalignment="center_baseline",
                )
                twin.get_yaxis().set_ticks([])

                x.set_ylim([0, x.get_ylim()[1]])

                axes[i].tick_params(axis="y", labelsize=22)
                axes[i].legend(loc="upper right", fontsize=18)
            for tick in x.xaxis.get_major_ticks():
                tick.label.set_fontsize(20)

    fig.autofmt_xdate()
    return fig


def median_G_xarray_old(
    data, title="G", benchmark=False, quantiles=[0.025, 0.5, 0.975]
):

    q = data["G"].quantile(dim="samples", q=quantiles).to_numpy()

    fig, axes = plt.subplots(nrows=data["G"].shape[-1], sharex=True)

    axes[0].set_title(title)

    colors = sns.color_palette(n_colors=len(axes))

    for i, x in enumerate(axes):

        x.plot(
            data["timestep"],
            q[1, :, i],
            "--",
            markersize=(150 / fig.dpi) ** 2,
            color=colors[i],
            # alpha=0.5,
            label="model",
        )

        if benchmark and data["labels"][i].item() in data["orig_labels"]:
            x.plot(
                data["timestep"],
                data["G_orig"].loc[:, data["labels"][i].item()],
                "k--",
                label="Truth",
            )

            coeff, _ = pearsonr(
                data["G_orig"].loc[:, data["labels"][i].item()], q[1, :, i]
            )
            label = (
                str(data["labels"][i].item())
                + "\nPearson's r: "
                + "{:.4f}".format(coeff)
            )

        twin = axes[i].twinx()
        twin.set_ylabel(
            # label,
            # labelpad=len(data["labels"][i].item()),
            # rotation="horizontal",
            # fontsize=18,
            # horizontalalignment="left",
            # loc="top",
            ylabel,
            #             labelpad=1,
            fontsize=18,
            rotation="horizontal",
            horizontalalignment="left",
            wrap=True,
            verticalalignment="center_baseline",
        )
        twin.get_yaxis().set_ticks([])

        x.set_ylim([0, x.get_ylim()[1]])

        axes[i].tick_params(axis="y", labelsize=22)
        axes[i].legend(loc="upper right", fontsize=18)
    for tick in x.xaxis.get_major_ticks():
        tick.label.set_fontsize(20)

    fig.autofmt_xdate()
    return fig


def plot_median_G_old(data, title="G_median", benchmark=False, quantiles=[0.5]):

    q = np.quantile(data["G"], quantiles, axis=0)

    fig, axes = plt.subplots(nrows=data["G"].shape[-1], sharex=True)

    axes[0].set_title(title)

    colors = sns.color_palette(n_colors=len(axes))

    for i, x in enumerate(axes):
        label = data["labels"][i]
        x.plot(
            data["timestep"],
            q[0, :, i],
            "--",
            markersize=(150 / fig.dpi) ** 2,
            color=colors[i],
        )

        if benchmark:
            if label in data["orig_labels"]:
                orig_index = data["orig_labels"].index(label)
                x.plot(
                    data["timestep"],
                    data["G_orig"][:, orig_index],
                    "k--",
                    label="Truth",
                )

                coeff, _ = pearsonr(
                    data["G_orig"][:, orig_index],
                    q[0, :, i],
                )
                label = label + "\nPearson's r: " + "{:.4f}".format(coeff)

        # set the subplot's y label
        twin = axes[i].twinx()
        twin.set_ylabel(
            # label,
            # labelpad=len(data["labels"][i]),
            # fontsize=18,
            # rotation="horizontal",
            # horizontalalignment="left",
            # loc="top",
            ylabel,
            #             labelpad=1,
            fontsize=18,
            rotation="horizontal",
            horizontalalignment="left",
            wrap=True,
            verticalalignment="center_baseline",
        )
        twin.get_yaxis().set_ticks([])  # hiding the ticks

        x.set_ylim([0, x.get_ylim()[1]])  # set the y value axis range

        axes[i].tick_params(axis="y", labelsize=22)
        axes[i].legend(loc="upper right", fontsize=18)
    for tick in x.xaxis.get_major_ticks():
        tick.label.set_fontsize(20)

    fig.autofmt_xdate()

    return fig


def errorbar_batched_old(
    data, labels, title, n_rows=5, xlabel="", palette=None, fontsize=22, label="factor"
):
    """plot the factir profiles or time series with quantiles value

    Args:
        mz (_type_): mz labels
        sample1 (_type_): draw's F matrix
        labels (_type_): component labels
        title (_type_): plot title
        n_factors (int, optional): number of factor (component). Defaults to 5.
        xlabel (str, optional): x axis label. Defaults to "".
        palette (_type_, optional): colors used for every component. Defaults to None.
        fontsize (int, optional): _description_. Defaults to 22.

    Returns:
        _type_: _description_
    """
    fig, axes = plt.subplots(n_rows, sharex=True)

    if palette is None:
        palette = sns.color_palette(n_colors=len(axes))

    axes[0].set_title(title)

    if label == "factor":
        quants = np.quantile(data["F"], q=[0.25, 0.5, 0.75], axis=0)
        for i, x in enumerate(axes):

            bp = axes[i].errorbar(
                data["labels"],
                quants[1, i, :],
                np.abs(quants[[0, 2], i, :] - quants[1, i, :]),
                color=palette[i],
                marker="o",
                markeredgecolor=palette[i],
                capthick=1,
                alpha=0.75,
                linewidth=0,
                elinewidth=1,
                ecolor=palette[i],
                capsize=3,
                markersize=5,
            )
            plt.xlabel(xlabel)
            twin = axes[i].twinx()
            twin.get_yaxis().set_ticks([])

            axes[i].legend([labels[i]], loc="upper right", fontsize=fontsize)
            axes[i].tick_params(axis="y", labelsize=fontsize)

        plt.xticks(rotation=90)
    elif label == "time":
        quants = np.quantile(data["G"], q=[0.25, 0.5, 0.75], axis=0)
        for i, x in enumerate(axes):
            # plot model results
            x.plot(
                data["timestep"],
                quants[1, :, i],
                "x",
                color=palette[i],
                markersize=(150 / fig.dpi) ** 2,
            )
            x.fill_between(
                data["timestep"],
                quants[0, :, i],
                quants[-1, :, i],
                alpha=0.5,
                color=palette[i],
            )
    return fig, axes


def plot_F_old(data, extra_data=None):

    fig, axes = errorbar_batched(
        data,
        labels=data["labels"],
        title="F",
        n_rows=data["G"].shape[-1],
        label="factor",
    )

    for i, x in enumerate(axes):
        if "F_orig" in data.keys():
            # if data["labels"][i] in data["F_orig"].index:
            if data["labels"][i] in data["orig_labels"]:
                orig_index = data["orig_labels"].index(data["labels"][i])

                # x.plot(
                #     data["timestep"], data["G_orig"][:, orig_index], "k--", label="Truth"
                # )

                x.bar(
                    data["variable"],
                    # data["F_orig"][data["labels"][i], :],
                    data["F_orig"][orig_index, :],
                    width=0.5,
                    color="k",
                    alpha=0.6,
                    label="reference",
                )

                # spearmann coeff notation
                ####### TODO: only use the median G value to calculate the coeffcient########
                # quants = np.quantile(data["F"][:, i, :].squeeze(), q=[0.5], axis=0)
                # coeff, _ = spearmanr(
                #     data["F_orig"][data["labels"][i], :],
                #     quants,
                #     axis=None,
                # )
                ####### only use the median G value to calculate the coeffcient########
                coeffs = cal_metrics(
                    data["F"][:, i, :],
                    # data["F_orig"][data["labels"][i], :],
                    data["F_orig"][orig_index, :],
                    spearmanr,
                )

                # set the subplot's y label
                twin = axes[i].twinx()
                twin.set_ylabel(
                    data["labels"][i]
                    + "\nSpearman's ρ:\n"
                    + "{:.4f}".format(np.mean(coeffs))
                    + " +/- "
                    + "{:.4f}".format(np.std(coeffs)),
                    # labelpad=len(data["labels"][i]),
                    # fontsize=18,
                    # rotation="horizontal",
                    # horizontalalignment="left",
                    # wrap=True,
                    # loc="top",
                    # ylabel,
                    #             labelpad=1,
                    fontsize=18,
                    rotation="horizontal",
                    horizontalalignment="left",
                    wrap=True,
                    verticalalignment="center_baseline",
                )
                twin.get_yaxis().set_ticks([])  # hiding the ticks
        x.set_xticks(data["variable"], minor=True)

        for tick in x.xaxis.get_major_ticks():
            tick.label.set_fontsize(18)
            tick.label.set_rotation(90)

    return fig
