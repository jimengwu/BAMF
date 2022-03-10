
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pandas.plotting import register_matplotlib_converters
from scipy.stats import ks_2samp

register_matplotlib_converters()

plt.rcParams["figure.figsize"] = (30,14)
plt.rcParams["font.size"] = 40

sns.set_style("whitegrid")
sns.set_context("poster", font_scale=2.0)
sns.set_palette(sns.color_palette("colorblind"))

def plot_diagnostic(di, name, thresh=0.05):
    
    n = int(di.shape[0]/2)
    half = n + int(n/2)
    
    first = di[n:half,:].ravel()
    second = di[half:,:].ravel()
    
    ks, p = ks_2samp(first, second)
    
    fig = plt.figure()
    
    ax = sns.kdeplot(first, shade=True, alpha=0.5, color="r", label="first half")
    ax = sns.kdeplot(second, shade=True, alpha=0.5, color="b", label="second")
    
    if p < thresh:
        textcol = "r"
    else:
        textcol = "g"
        
    ax.set_title(f"{name}, p-value:{p:1.2f}", c=textcol)
    
    return fig, p

def plot_F(data, extra_data=None):
    
    fig, ax = errorbar_batched_factors(data["columns"], data["F"], labels=data["labels"], title="F",
                                             n_factors=data["p"])
    
    
    for i,x in enumerate(ax):
        if "reference_F" in data.keys():
            if data["labels"][i] in data["reference_F"].index:
                
                x.bar(data["columns"], data["reference_F"].loc[data["labels"][i],:], width=0.5, color='k', alpha=0.25)
        
        x.set_xticks(data["columns"], minor=True)
                
        for tick in x.xaxis.get_major_ticks():
            tick.label.set_fontsize(16)
            tick.label.set_rotation(90)
        
    return fig

def plot_individual_F(data, compound, title="F", benchmark=False):

    fig, ax = plt.subplots(data["samples"], 1, sharex=True)

    title="F"

    if benchmark:
        title = title + "," + data["orig_labels"][compound]
    else:
        title = title + f", {compound}"

    colors = sns.color_palette()

    for i,x in enumerate(ax):
        
        x.plot(data["columns"], data["F"][i,compound,:], linestyle="", marker="o", color=colors[i], alpha=0.5)
        if benchmark:
            x.bar(data["columns"], data["reference_F"].iloc[compound,:], width=0.5, color='k', alpha=0.25)
        
        x.set_xticks(data["columns"], minor=True)
                
        for tick in x.xaxis.get_major_ticks():
            tick.label.set_fontsize(16)
            tick.label.set_rotation(90)
        
    return fig

def plot_G(data, title="G", benchmark=False, quantiles=[0.25, 0.5, 0.75]):
    
    q = np.quantile(data["G"], quantiles, axis=0)
    
    fig, axes = plt.subplots(nrows=data["p"], sharex=True)
    
    axes[0].set_title(title)

    colors = sns.color_palette()
    
    for i,x in enumerate(axes):
        
        x.plot(data["times"], q[1,:,i] , "x--", color=colors[i], alpha=0.5, label="model")
        x.fill_between(data["times"], q[0,:,i], q[-1,:,i], alpha=0.5, color=colors[i])
        
        if benchmark:
            if data["labels"][i] in data["orig_labels"]:
                orig_index = data["orig_labels"].index(data["labels"][i])
                x.plot(data["times"], data["G_orig"][:,orig_index], 'k--', label="Truth")

        twin = axes[i].twinx()
        twin.set_ylabel(data["labels"][i], labelpad=len(data["labels"][i]), rotation=90)
        twin.get_yaxis().set_ticks([])
                                                             
        x.set_ylim([0, x.get_ylim()[1]])

    for tick in x.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
                                                             
    fig.autofmt_xdate()

    return fig

def plot_individual_G(data, title="G", benchmark=False):

    fig, axes = plt.subplots(nrows=data["G"].shape[-1], sharex=True)
    
    axes[0].set_title(title)

    colors = sns.color_palette()
    
    for i,x in enumerate(axes):
        
        for j in range(data["samples"]):
            x.plot(data["times"], data["G"][j,:,i] , "x--", color=colors[j], alpha=0.5, label=f"{j}")

        if benchmark:
            x.plot(data["times"], data["G_orig"][:,i], 'k--', label="Truth")

        twin = axes[i].twinx()
        twin.set_ylabel(data["labels"][i], labelpad=len(data["labels"][i]), rotation=90)
        twin.get_yaxis().set_ticks([])
                                                             
        x.set_ylim([0, x.get_ylim()[1]])

    for tick in x.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
                                                             
    fig.autofmt_xdate()

    return fig

def plot_median_G(data, title="G", benchmark=False, quantiles=[0.25, 0.5, 0.75]):

    
    q = np.quantile(data["G"], quantiles, axis=0)
    
    fig, axes = plt.subplots(nrows=data["p"], sharex=True)
    
    axes[0].set_title(title)

    colors = sns.color_palette()
    
    for i,x in enumerate(axes):
        
        x.plot(data["times"], q[1,:,i] , "x--", color=colors[i], alpha=0.5, label="model")
        #x.fill_between(data["times"], q[0,:,i], q[-1,:,i], alpha=0.5, color=colors[i])
        
        if benchmark:
            if data["labels"][i] in data["orig_labels"]:
                orig_index = data["orig_labels"].index(data["labels"][i])
                x.plot(data["times"], data["G_orig"][:,orig_index], 'k--', label="Truth")
        
        twin = axes[i].twinx()
        twin.set_ylabel(data["labels"][i], labelpad=len(data["labels"][i]), rotation=90)
        twin.get_yaxis().set_ticks([])
                                                             
        x.set_ylim([0, x.get_ylim()[1]])

    for tick in x.xaxis.get_major_ticks():
        tick.label.set_fontsize(16) 
                                                             
    fig.autofmt_xdate()

    return fig

def mass_conservation(data, sample=-1):

    print(data["X"].dtype, data["X"].shape)
    data["X"].shape, data["G"].shape
    total_orig = data["X"].sum(axis=-1)
    total = data["G"].sum(axis=-1)[sample, ...]

    fig, axes = plt.subplots(3,1)

    axes[0].set_title("Mass conservation from one sample")

    axes[0].scatter(total_orig, total)
    axes[1].scatter(total_orig, total/total_orig)
    axes[2].scatter(total_orig, total/total_orig)

    axes[2].set_xscale('log')
    axes[0].set_ylabel("Sum(G)")
    axes[1].set_ylabel("Sum(G)/\nsum(Input)")
    axes[2].set_ylabel("Sum(G)/\nsum(Input)")
    axes[2].set_xlabel("Sum(Input)")

    return fig

def errorbar_batched_factors(mz, sample1, labels, title, n_factors=5, xlabel="", palette=None, fontsize=12):

    fig, axes = plt.subplots(nrows=n_factors, sharex=True)
    
    if palette is None:
        palette = sns.color_palette()

    axes[0].set_title(title)
    for i in range(n_factors):


        c = palette[i]
        data = sample1[:,i,:].squeeze()
        quants = np.quantile(data, q=[0.25, 0.5, 0.75], axis=0)
        #print(quants.shape)

        bp = axes[i].errorbar(mz, quants[1,:] , np.abs(quants[[0,2], :] - quants[1, :]), color = c, marker='o',
        markeredgecolor=c, capthick=1, alpha=0.75, linestyle='', linewidth=0, elinewidth=0.2, ecolor=c,
        capsize=5)
        
        #plt.xticks(mz, fontsize=fontsize)
        plt.xlabel(xlabel)
        twin = axes[i].twinx()
        twin.set_ylabel(labels[i], labelpad=len(labels[i]), rotation=90)
        twin.get_yaxis().set_ticks([])
        
        #axes[i].legend([labels[i]], loc="upper right")

    plt.xticks(rotation=90)
    return fig, axes


