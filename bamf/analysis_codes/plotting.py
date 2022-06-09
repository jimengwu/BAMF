import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pandas.plotting import register_matplotlib_converters
from scipy.stats import ks_2samp, binom_test, kruskal, pearsonr, spearmanr

register_matplotlib_converters()

plt.rcParams["figure.figsize"] = (30, 14)
plt.rcParams["font.size"] = 40

sns.set_style("whitegrid")
sns.set_context("poster", font_scale=2.0)
sns.set_palette(sns.color_palette("colorblind"))


def cal_metrics(data1, data2, metric):
    coeffs = []

    for i in range(data1.shape[0]):
        coeff, _ = metric(
            data2,
            data1[i, :],
        )
        coeffs.append(coeff)
    return coeffs


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


def plot_F(data, extra_data=None):

    fig, axes = errorbar_batched(
        data, labels=data["labels"], title="F", n_rows=data["p"], label="factor"
    )

    for i, x in enumerate(axes):
        if "reference_F" in data.keys():
            if data["labels"][i] in data["reference_F"].index:

                x.bar(
                    data["columns"],
                    data["reference_F"].loc[data["labels"][i], :],
                    width=0.5,
                    color="k",
                    alpha=0.6,
                    label="reference",
                )

                # spearmann coeff notation
                ####### TODO: only use the median G value to calculate the coeffcient########
                # quants = np.quantile(data["F"][:, i, :].squeeze(), q=[0.5], axis=0)
                # coeff, _ = spearmanr(
                #     data["reference_F"].loc[data["labels"][i], :],
                #     quants,
                #     axis=None,
                # )
                ####### only use the median G value to calculate the coeffcient########
                coeffs = cal_metrics(
                    data["F"][:, i, :],
                    data["reference_F"].loc[data["labels"][i], :],
                    pearsonr,
                )
                coeff_avg = np.mean(coeffs)
                coeff_std = np.std(coeffs)
                # set the subplot's y label
                twin = axes[i].twinx()
                twin.set_ylabel(
                    data["labels"][i]
                    + "\nSpearman's ρ:"
                    + "{:.4f}".format(coeff_avg)
                    + "+/-"
                    + "{:.4f}".format(coeff_std),
                    labelpad=len(data["labels"][i]),
                    fontsize=18,
                    rotation="horizontal",
                    horizontalalignment="left",
                    wrap=True,
                )
                twin.get_yaxis().set_ticks([])  # hiding the ticks
        x.set_xticks(data["columns"], minor=True)

        for tick in x.xaxis.get_major_ticks():
            tick.label.set_fontsize(18)
            tick.label.set_rotation(90)

    return fig


def plot_individual_F(data, compound, title="F", benchmark=False):

    fig, ax = plt.subplots(data["samples"], 1, sharex=True)

    title = "F"

    if benchmark:
        title = title + "," + data["orig_labels"][compound]
    else:
        title = title + f", {compound}"

    colors = sns.color_palette()

    for i, x in enumerate(ax):

        x.plot(
            data["columns"],
            data["F"][i, compound, :],
            linestyle="",
            marker="o",
            color=colors[i],
            alpha=0.5,
        )
        if benchmark:
            x.bar(
                data["columns"],
                data["reference_F"].iloc[compound, :],
                width=0.5,
                color="k",
                alpha=0.25,
            )

        x.set_xticks(data["columns"], minor=True)

        for tick in x.xaxis.get_major_ticks():
            tick.label.set_fontsize(16)
            tick.label.set_rotation(90)

    return fig


def plot_G(data, title="G", benchmark=False, quantiles=[0.25, 0.5, 0.75]):

    fig, axes = errorbar_batched(
        data, labels=data["labels"], title=title, n_rows=data["p"], label="time"
    )
    # plot benchmark
    if benchmark:
        for i, x in enumerate(axes):
            if data["labels"][i] in data["orig_labels"]:
                orig_index = data["orig_labels"].index(data["labels"][i])
                x.plot(
                    data["times"], data["G_orig"][:, orig_index], "k--", label="Truth"
                )

                # pearson coeff notation

                coeffs = cal_metrics(
                    data["G"][:, :, i], data["G_orig"][:, orig_index], pearsonr
                )
                coeff_avg = np.mean(coeffs)
                coeff_std = np.std(coeffs)
                # set the subplot's y label
                twin = axes[i].twinx()
                twin.set_ylabel(
                    # data["labels"][i] + "\nPearson's r: " + "{:.4f}".format(coeff),
                    data["labels"][i]
                    + "\nPearson's r: "
                    + "{:.4f}".format(coeff_avg)
                    + "+/-"
                    + "{:.4f}".format(coeff_std),
                    labelpad=len(data["labels"][i]),
                    fontsize=18,
                    rotation="horizontal",
                    horizontalalignment="left",
                    wrap=True,
                )
                twin.get_yaxis().set_ticks([])  # hiding the ticks

                x.set_ylim([0, x.get_ylim()[1]])  # set the y value axis range

                axes[i].tick_params(axis="y", labelsize=22)
                axes[i].legend(loc="upper right", fontsize=18)
            for tick in x.xaxis.get_major_ticks():
                tick.label.set_fontsize(20)
    fig.autofmt_xdate()
    return fig


def plot_individual_G(data, title="G", benchmark=False):

    fig, axes = plt.subplots(nrows=data["G"].shape[-1], sharex=True)

    axes[0].set_title(title)

    colors = sns.color_palette()

    for i, x in enumerate(axes):

        for j in range(data["samples"]):
            x.plot(
                data["times"],
                data["G"][j, :, i],
                "x--",
                color=colors[j],
                alpha=0.5,
                label=f"{j}",
            )

        if benchmark:
            x.plot(data["times"], data["G_orig"][:, i], "k--", label="Truth")

        twin = axes[i].twinx()
        twin.set_ylabel(data["labels"][i], labelpad=len(data["labels"][i]), rotation=90)
        twin.get_yaxis().set_ticks([])

        x.set_ylim([0, x.get_ylim()[1]])

    for tick in x.xaxis.get_major_ticks():
        tick.label.set_fontsize(16)

    fig.autofmt_xdate()

    return fig


def plot_median_G(data, title="G", benchmark=False, quantiles=[0.5]):

    q = np.quantile(data["G"], quantiles, axis=0)

    fig, axes = plt.subplots(nrows=data["p"], sharex=True)

    axes[0].set_title(title)

    colors = sns.color_palette()

    for i, x in enumerate(axes):
        label = data["labels"][i]
        x.plot(
            data["times"],
            q[0, :, i],
            "--",
            markersize=(150 / fig.dpi) ** 2,
            color=colors[i],
        )

        if benchmark:
            if label in data["orig_labels"]:
                orig_index = data["orig_labels"].index(label)
                x.plot(
                    data["times"], data["G_orig"][:, orig_index], "k--", label="Truth"
                )

                coeff, _ = pearsonr(
                    data["G_orig"][:, orig_index],
                    q[0, :, i],
                )
                label = label + "\nPearson's r: " + "{:.4f}".format(coeff)

        # set the subplot's y label
        twin = axes[i].twinx()
        twin.set_ylabel(
            label,
            labelpad=len(data["labels"][i]),
            fontsize=18,
            rotation="horizontal",
            horizontalalignment="left",
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

    print(data["X"].dtype, data["X"].shape, flush=True)
    data["X"].shape, data["G"].shape
    total_orig = data["X"].sum(axis=-1)
    total = data["G"].sum(axis=-1)[sample, ...]

    fig, axes = plt.subplots(3, 1)

    axes[0].set_title("Mass conservation from one sample")

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


def errorbar_batched(
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
        palette = sns.color_palette()

    axes[0].set_title(title)

    if label == "factor":
        quants = np.quantile(data["F"], q=[0.25, 0.5, 0.75], axis=0)
        for i, x in enumerate(axes):

            bp = axes[i].errorbar(
                data["columns"],
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
                data["times"],
                quants[1, :, i],
                "x",
                color=palette[i],
                markersize=(150 / fig.dpi) ** 2,
            )
            x.fill_between(
                data["times"],
                quants[0, :, i],
                quants[-1, :, i],
                alpha=0.5,
                color=palette[i],
            )
    return fig, axes


# --------------------------drawing PMF results--------------------


def mass_conservation_xarray(data):

    total_orig = data["data"].sum(axis=-1)
    total = data["G"].sum("component").median(dim="sample")

    fig, axes = plt.subplots(3, 1)

    axes[0].set_title("Mass conservation for median")

    axes[0].scatter(total_orig, total)
    axes[1].scatter(total_orig, total / total_orig)
    axes[2].scatter(total_orig, total / total_orig)
    axes[1].axhline(y=1, color="r", linestyle="--", alpha=0.8)
    axes[2].axhline(y=1, color="r", linestyle="--", alpha=0.8)

    axes[2].set_xscale("log")
    axes[0].set_ylabel("Sum(G)")
    axes[1].set_ylabel("Sum(G)/\nsum(Input)")
    axes[2].set_ylabel("Sum(G)/\nsum(Input)")
    axes[2].set_xlabel("Sum(Input)")

    return fig


def errorbar_batched_factors_xarray(
    mz, sample1, labels, title, n_factors=5, xlabel="", palette=None, fontsize=12
):

    fig, axes = plt.subplots(nrows=n_factors, sharex=True)

    if palette is None:
        palette = sns.color_palette()

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


def F_xarray(data, title="F", benchmark=True):
    mz = data["F"]["variable"].to_numpy()
    y_value = data["F"].to_numpy()

    components = data.component.to_numpy()

    fig, ax = errorbar_batched_factors_xarray(
        mz,
        y_value,
        labels=components,
        title="F",
        n_factors=len(data.component),
    )

    for i, x in enumerate(ax):
        # if "reference_F" in data.keys():
        if benchmark:
            if data["component"][i].item() in data["original_component"]:
                # if data["F"]["component"][i] in data["reference_F"].index:

                x.bar(
                    data["F"]["variable"],
                    data["F_orig"].loc[components[i], :],
                    width=0.5,
                    color="k",
                    alpha=0.25,
                    label="reference",
                )

        x.set_xticks(data["F"]["variable"], minor=True)

        for tick in x.xaxis.get_major_ticks():
            tick.label.set_fontsize(16)
            tick.label.set_rotation(90)

    return fig


def G_xarray(data, title="G", benchmark=False, quantiles=[0.025, 0.5, 0.975]):

    q = data["G"].quantile(dim="sample", q=quantiles).to_numpy()

    fig, axes = plt.subplots(nrows=len(data.component), sharex=True)

    axes[0].set_title(title)

    colors = sns.color_palette()

    for i, x in enumerate(axes):

        x.plot(
            data["timestep"],
            q[1, :, i],
            "x",
            markersize=(150 / fig.dpi) ** 2,
            color=colors[i],
            # alpha=0.7,
            # label="model",
        )
        x.fill_between(
            data["timestep"].to_numpy(),
            q[0, :, i],
            q[-1, :, i],
            alpha=0.5,
            color=colors[i],
        )

        if benchmark:
            if data["component"][i].item() in data["original_component"]:
                x.plot(
                    data["timestep"],
                    data["G_orig"].loc[:, data["component"][i].item()],
                    "k--",
                    label="Truth",
                )

        twin = axes[i].twinx()
        twin.set_ylabel(
            data["component"][i].item(),
            labelpad=len(data["component"][i].item()),
            rotation="horizontal",
            fontsize=18,
            horizontalalignment="left",
        )
        twin.get_yaxis().set_ticks([])

        x.set_ylim([0, x.get_ylim()[1]])

        axes[i].tick_params(axis="y", labelsize=22)
        axes[i].legend(loc="upper right", fontsize=18)
    for tick in x.xaxis.get_major_ticks():
        tick.label.set_fontsize(20)

    fig.autofmt_xdate()
    return fig


def median_G_xarray(data, title="G", benchmark=False, quantiles=[0.025, 0.5, 0.975]):

    q = data["G"].quantile(dim="sample", q=quantiles).to_numpy()

    fig, axes = plt.subplots(nrows=len(data.component), sharex=True)

    axes[0].set_title(title)

    colors = sns.color_palette()

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

        if benchmark:
            if data["component"][i].item() in data["original_component"]:
                x.plot(
                    data["timestep"],
                    data["G_orig"].loc[:, data["component"][i].item()],
                    "k--",
                    label="Truth",
                )

        twin = axes[i].twinx()
        twin.set_ylabel(
            data["component"][i].item(),
            labelpad=len(data["component"][i].item()),
            rotation=90,
            fontsize=18,
            horizontalalignment="left",
        )
        twin.get_yaxis().set_ticks([])

        x.set_ylim([0, x.get_ylim()[1]])

        axes[i].tick_params(axis="y", labelsize=22)
        axes[i].legend(loc="upper right", fontsize=18)
    for tick in x.xaxis.get_major_ticks():
        tick.label.set_fontsize(20)

    fig.autofmt_xdate()
    return fig
