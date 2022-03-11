from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.stats import (binom_test, kruskal, ks_2samp, pearsonr, spearmanr)

from . import plotting
from .helpers import (Collated_output, calculate_Z_contribution,
                      distance_label_batch, distance_label_batch_by_Z,
                      dump_hdf5, get_plot_filename,
                      normalize, normalize_chains,
                      sort_by_batch_label)


def get_generic_data(filename, labels=["F", "G", "X_orig", "error"]):
    result = {}
    
    with h5py.File(filename, "r") as result_file:
        for label in labels:
            result[label] = result_file[label][()]

    return result

def get_axis(shape):
    if len(shape) == 1:
        return 0
    for i,s in enumerate(shape):
        if s == 1:
            return i
    return 0

def get_generic_data_from_multiple_files(filenames, labels=["F", "G", "X_orig", "error", "alpha_n", "alpha_r", "alpha_s", "norm_const"], singles=["X_orig", "X", "n", "p", "m", "error", "sigma", "norm_const"]):
    
    combined = {}

    results = [get_generic_data(f, labels) for f in filenames]
    for item in results:
        norm_and_scale(item)


    for key in results[0].keys():
        
        if key in singles:
            combined[key] = results[0][key]
            
            c = {}
            for j,f in enumerate(filenames):
                if j==0:
                    continue
                c[f] = np.allclose(results[0][key], results[j][key])
            if np.any([not j for j in c.items()]):
                print(key, c)
            else:
                print(key, "The files have matching variables.")
            continue

        combined[key] = np.stack([r[key].squeeze() for r in results], axis=get_axis(results[0][key].shape))

    return combined

def norm_and_scale(data):

    if "X" in data.keys():
        data["X"] = data["X"] * data["norm_const"]
    else:
        data["X_orig"] = data["X_orig"] * data["norm_const"]
    if "sigma" in data.keys():
        data["sigma"] = data["sigma"] * data["norm_const"]
    else:
        data["error"] = data["error"] * data["norm_const"]

    data["G"] = data["G"]*data["norm_const"]

def reshaper(x, n=1000):

    return np.reshape(x[n:], [-1,x.shape[-2], x.shape[-1]])

def reshaper_1d(x, n=1000):

    return np.reshape(x[n:], [-1, x.shape[-1]])

def merge_chains(data, n=1000, partial=False):

    autocorrelation_parameters = [i for i in data.keys() if i.startswith("alpha")]

    #Sometimes all you have is one chain...
    if data["F"].shape[1] == 1:
        data["F"] = data["F"].squeeze()
        data["G"] = data["G"].squeeze()
        for item in autocorrelation_parameters:            
                data[item] = data[item].squeeze()

        if partial:
            data["height"] = data["height"].squeeze()
        
    else:

        data["F"] = reshaper(data["F"], n)
        data["G"] = reshaper(data["G"], n)

        for item in autocorrelation_parameters:
                data[item] = reshaper_1d(data[item], n)

        if partial:
                data["height"] = reshaper_1d(data["height"],n)

def spear(u,v):
    coeff, _ = spearmanr(u,v,axis=None)
    return coeff

def scale_times(times):

    timediff = times - times[0]

    medtime = np.median(np.diff(timediff.total_seconds().values))

    scaletime = timediff.total_seconds()/medtime

    return scaletime, medtime

def sort_by_input_F(data, F_orig=None, metric="cosine"):
    
    if F_orig is None:
        indices = distance_label_batch(data["F"], data["F_orig"], metric=metric)
    else:
        indices = distance_label_batch(data["F"], F_orig, metric=metric)

    data["F"] = sort_by_batch_label(data["F"], indices)    
    data["G"] = sort_by_batch_label(data["G"], indices, axis=3)
    data["G"] = np.swapaxes(data["G"], -1, -2)

    autocorrelation_parameters = ["alpha_n", "alpha_r", "alpha_s"]
    for item in autocorrelation_parameters:
        if item in data.keys():
            data[item] = sort_by_batch_label(data[item], indices, axis='1d')

def sort_by_Z_contribution(data, take=slice(None), use_exemplars=None, metric="correlation"):

    if use_exemplars is None:
        subset_F = data["F"][take,:,:] 
        subset_G = data["G"][take,:,:]

        print("Sorting sample sizes:", subset_F.shape, subset_G.shape)

        n = subset_F.shape[0]

        exemplars = [calculate_Z_contribution(subset_F[i,:,:], subset_G[i,:,:]) for i in range(n)]
    else:
        exemplars = use_exemplars
        n = 1

    orders = np.stack([distance_label_batch_by_Z(data["F"], data["G"], exemplars[i], metric=metric).flatten() for i in range(n)])

    ind, count = np.unique(orders, axis=0, return_counts=True)

    print("There are {} possible orderings".format(len(count)))
    print(count)

    indices = ind[np.argmax(count)].reshape(data["F"].shape[0:2])

    print("Sorting")
    data["F"] = sort_by_batch_label(data["F"], indices)

    data["G"] = sort_by_batch_label(data["G"], indices, axis=3)

    data["G"] = np.swapaxes(data["G"], -1, -2)

    autocorrelation_parameters = [i for i in data.keys() if i.startswith("alpha")]
    for item in autocorrelation_parameters:
        data[item] = sort_by_batch_label(data[item], indices, axis="1d")

    return data

def z_scale(residuals):

    return (residuals - residuals.mean(axis=-2, keepdims=True))/residuals.std(axis=-2, keepdims=True)

def reconstruct(F,G):
    return np.einsum("aij, ajk -> aik", G, F)

def match_labels(values, exemplar_frame, cutoff=0.1, distance_metric='cityblock', unknown_label="Unknown"):
    
    distances = cdist(exemplar_frame, values, distance_metric)
    
    row_ind, selections = linear_sum_assignment(distances)
    
    accepted =  distances[row_ind, selections] < cutoff
    
    rejected = selections[~accepted]
    row_ind = row_ind[accepted]
    selections = selections[accepted]
    
    #This is mainly to give them in the same order as the exemplars
    ordering = np.argsort(row_ind)
    
    selections = selections[ordering]
    labels = exemplar_frame.index.values[row_ind[ordering]]
    
    #Rest we couldn't identify
    selections = np.append(selections, rejected)
    labels = np.append(labels, np.repeat(unknown_label, len(rejected)))
    
    return selections, labels

def summarize_frame(data, benchmark=False):

    numbers = {}

    print("median solution")

    median_F = np.median(data["F"], axis=0)
    median_G = np.median(data["G"], axis=0)

    residuals = data["X"] - data["Z"]
    respererr = residuals/data["sigma"]

    print("descriptive")

    numbers["Mean(abs(residual))"] = np.mean(np.abs(residuals))
    numbers["Median(abs(residual))"] = np.median(np.abs(residuals))
    numbers["Max(abs(residual))"] = np.max(np.abs(residuals))

    numbers["Mean(abs(residual/error))"] = np.mean(np.abs(respererr))
    numbers["Median(abs(residual/error))"] = np.median(np.abs(respererr))
    numbers["Max(abs(residual/error))"] = np.max(np.abs(respererr))

    print("z-data")

    d_means = data["Z"].mean(axis=(0,1))
    d_stds = data["Z"].std(axis=(0,1))

    z_data = (data["X"] - d_means)/d_stds

    #

    print("Q statistics")
    numbers["Qm"] = np.sum(z_data**2)
    numbers["Qexp"] = np.prod(data["X"].shape) # - p*(np.sum(data["X_orig"].shape)) 

    print("Z statistics")
    z_mean = np.mean(z_data)
    
    z_std = np.std(z_data)

    numbers["Z scaled mean"] = z_mean
    numbers["Z scaled std"] = z_std

    print("Alpha statistics")

    alphas = [i for i in data.keys() if i.startswith("alpha")]
    
    for item in alphas:

        print(f"{item} mean")
        means = np.mean(data[item], axis=0)
        print(f"{item} median")
        medians = np.median(data[item], axis=0)
        print(f"{item} std")
        stds = np.std(data[item], axis=0)

        for i in range(means.shape[0]):
            numbers[f"{item}{i} mean"] = means[i]
            numbers[f"{item}{i} median"] = medians[i]
            numbers[f"{item}{i} std"] = stds[i]

    if benchmark:
        #Only for matching labels to external reference

        ids = [(i,l) for i,l in enumerate(data["labels"]) if l in data["orig_labels"]]

        numbers["identified components"] = len(ids)

        for index,l in ids:

            print(f"{l} statistics")

            data_F = median_F[index,:]
            data_G = median_G[:, index]

            orig_index = data["orig_labels"].index(l)

            orig_F = data["F_orig"][orig_index,:]
            orig_G = data["G_orig"][:, orig_index]

            mean_label = f"Mean $G/G_o$ {l}"
            median_label = f"Median $G-G_o$ {l}"
            maximum_label = f"Maximum $G-G_o$ {l}"
            spear_label = f"G $ρ_{{{l}}}$"
            pearson_label = f"G $r_{{{l}}}$"

            #Add pearson
            numbers[spear_label] = spear(data_G, orig_G)
            numbers[pearson_label], _ = pearsonr(data_G, orig_G) 

            div = np.mean(data_G) / np.mean(orig_G)
            diff = np.abs(data_G - orig_G)
        
            numbers[mean_label] = div
            numbers[median_label] = np.nanmedian(diff)
            numbers[maximum_label] = np.nanmax(diff)

            mean_label = f"Mean $F-F_o$ {l}"
            median_label = f"Median $F-F_o$ {l}"
            maximum_label = f"Maximum $F-F_o$ {l}"
            spear_label = f"F $ρ_{{{l}}}$"
            pearson_label = f"F $r_{{{l}}}$"

            numbers[spear_label] = spear(data_F, orig_F)
            numbers[pearson_label], _ = pearsonr(data_F, orig_F) 

            div = np.abs(data_F - orig_F)

            numbers[mean_label] = np.mean(div)
            numbers[median_label] = np.median(div)
            numbers[maximum_label] = np.max(div)


    return numbers

def summarize_diagnostic(di, name):
    
    result = {}
    
    n = int(di.shape[0]/2)
    
    raveled = di[n:,...].ravel()
    
    result[f"mean {name}"] = np.mean(raveled)
    result[f"median {name}"] = np.median(raveled)
    result[f"std {name}"] = np.std(raveled)
    
    return result

def chain_diagnostics(data, ks=25):

    results = {}

    residuals = []

    samples = data["samples"]

    for n in range(data["chains"]):

        Z = np.stack([data["G"][i,n,:,:]@data["F"][i,n,:,:] for i in range(data["G"].shape[0])])

        residuals =  Z - data["X"]

        samp1 = residuals[samples:samples+ks,...].ravel()
        samp2 = residuals[-ks:,...].ravel()

        #take last 25 and first 25 samples and compare distributions, if the result is notably different at the end, then chain has wandered onto another minima
        _, p = ks_2samp(samp1, samp2)

        results[f"Chain {n} residual ks, p"] = p

    
    #check chain median divergence from max 500 last samples
    x = -500
    if samples < 500:
        x = samples

    if n > 1:
        s, results["log prob kruskall chains p"] = kruskal(*[data["lp__"][x:, n].ravel() for n in range(data["chains"])])
    
        s, results["residuals kruskall chains p"] = kruskal(*[data["lp__"][x:, n].ravel() for n in range(data["chains"])])

    return results


def sign_diagnostic(di, name):
    
    result = {}
    
    n = int(di.shape[0]/2)
    
    samples = di[n:,...]

    signs = np.sign(samples[1:, :] - samples[:-1, :]).ravel()
    plus_minus = [np.count_nonzero(signs > 0), np.count_nonzero(signs < 0)]

    result[f"mean sign {name}"] = np.mean(signs)
    result[f"sign test {name} p"] = binom_test(plus_minus)
    
    return result

default_model_labels = {
    "bamf":["F" , "G", "X", "m", "n", "p", "lp__", "sigma", "norm_const", "alpha_a", "alpha_b"],
}

def analyze_frame(dataset, benchmark=False, match_file=None, distance_metric="cityblock", cutoff=0.1, model_data=None, get_tags=None, basepath=Path("./plots"), save_sorted=None):

    metrics = {}

    if model_data is not None:
        model_labels = model_data
    else:
        model_labels = default_model_labels

    if get_tags is None:
        model = Path(dataset.iloc[0,:]["model"]).stem
        if model not in model_labels.keys():
            raise ValueError(f"Tags to extract not specified and model: {model} unknown.")
        else:
            tags = model_labels[model]
    else:
        tags = get_tags

    action = dataset.iloc[0,:]["action"]

    first_ids = dataset.iloc[0,:]

    if action == "map" and "lp__" in tags:
        t = tags.copy()
        t.remove("lp__")
        tags = t

    data = get_generic_data_from_multiple_files(dataset["filename"], labels=tags)
    first_file = dataset.iloc[0, :]["filename"]
    datafile = dataset.iloc[0,:]["datafile"]
    data["chains"] = len(dataset.filename)

    if action == "sampling":
        #Choose the most popular order of the last samples

        data["samples"] = int(data["G"].shape[0]/2)

        metrics.update(chain_diagnostics(data))

        merge_chains(data, n=data["samples"])

        slicy = slice(-len(dataset["filename"]), None)

        data = sort_by_Z_contribution(data, take=slicy, metric=distance_metric)

        data["Z"] = np.stack([data["G"][i,:,:]@data["F"][i,:,:] for i in range(data["G"].shape[0])])


        #Take metadata from first file (should match across files)
        with h5py.File(first_file, 'r') as infile:
            time_str = infile["time_index"][()].astype("str")
            data["times"] = pd.to_datetime(time_str, format="%Y-%m-%d %H:%M:%S")
            data["columns"] = infile["columns"][()].astype("str")

    elif action == "map":

        data["samples"] = min(data["chains"], 7)
        
        data = sort_by_Z_contribution(data, take=slice(0,data["samples"]), metric=distance_metric)

        data["Z"] = np.stack([data["G"][i,:,:]@data["F"][i,:,:] for i in range(data["G"].shape[0])])

        #Take metadata from first file (should match across files)
        with h5py.File(first_file, 'r') as infile:
            time_str = infile["time_index"][()].astype("str")
            data["times"] = pd.to_datetime(time_str, format="%Y-%m-%d %H:%M:%S")
            data["columns"] = infile["columns"][()].astype("str")

    else:
        raise ValueError(f"Unknown action:{action} performed by model")

    if benchmark:
        first_file = dataset.iloc[0, :]["filename"]
        #Assuming these are normalized correctly and the right size.
        with h5py.File(first_file, 'r') as infile:
            data["F_orig"] = infile["F_orig"][()]
            data["G_orig"] = infile["G_orig"][()]

    

    #Sort and label with reference F if available.
    if match_file is not None:

        if match_file == "datafile":
            match_profiles = pd.read_hdf(datafile, "F")
        else:
            match_profiles = pd.read_hdf(match_file, "F")

        if benchmark:
            data["orig_labels"] = list(match_profiles.index.values)

        if action == "sampling" or action == "map":
            
            median_F = np.median(data["F"], axis=0)

            indices, labels = match_labels(median_F, match_profiles, cutoff=cutoff, distance_metric=distance_metric)

            all_indices = set(np.arange(data["F"].shape[1]))
            unresolved = list(set(all_indices) - set(indices))

            indices = [*indices, *unresolved]

            data["F"] = data["F"][:, indices, :]
            data["G"] = data["G"][:,:, indices]

            autocorr = [i for i in data.keys() if i.startswith("alpha")]

            for item in autocorr:
                data[item] = data[item][:, indices]

            data["labels"] = labels

            if len(unresolved) > 0:
                for kk in range(len(unresolved)):
                    data["labels"] = np.append(data["labels"], "?"+str(kk))

            data["reference_F"] = match_profiles
            
    else:
        data["labels"] = np.arange(data["p"]).astype(str)

    records = dataset.to_dict('records')

    outfilename = get_plot_filename(basepath, first_file)
    outfilename.parent.mkdir(parents=True, exist_ok=True)
    
    sorted_filename = outfilename.with_suffix(".nc4")
    print(outfilename)

    with Collated_output(outfilename) as outfile:

        if "lp__" in data:
            f,p = plotting.plot_diagnostic(data["lp__"], "log prob")
            outfile.savefig(f, "log prob")
            plt.close()
            metrics["log prob p"] = p

            lp_m = summarize_diagnostic(data["lp__"], "log prob")    

            metrics.update(lp_m)

            f = plotting.mass_conservation(data)

            outfile.savefig(f, "conservation")

            plt.close()

    #plot standard G,F plots, with truth if benchmark

        f = plotting.plot_F(data)
        outfile.savefig(f, "F")
        plt.close()

        f = plotting.plot_G(data, benchmark=benchmark)
        outfile.savefig(f, "G")
        plt.close()

        f = plotting.plot_median_G(data, benchmark=benchmark)
        outfile.savefig(f, "median_G")
        plt.close()

        if action =="map":

            for i in range(data["G"].shape[-1]):
                f = plotting.plot_individual_F(data, i, benchmark=benchmark)
                outfile.savefig(f, f"F_map_{i}")
                plt.close()

            f = plotting.plot_individual_G(data, benchmark=benchmark)
            outfile.savefig(f, "G_map")
            plt.close()

            pass

    #Calculate metrics

    summary = summarize_frame(data, benchmark=benchmark)

    metrics.update(summary)

    if "lp__" in data:
        metrics.update(sign_diagnostic(data["lp__"], "log prob"))

    #Save intermediary file containing sorted data
    samples =  xr.Dataset({"F":(["sample", "component", "variable"], data["F"]),
                           "G":(["sample", "timestep", "component"], data["G"]),
                           "alpha_a":(["sample", "component"], data["alpha_a"]),
                           "alpha_b":(["sample", "component"], data["alpha_b"])},
                           coords= {"sample": np.arange(data["F"].shape[0]), 
                                    "component": data["labels"],
                                    "variable": data["columns"],
                                    "timestep": data["times"]    
                                    })

    samples.to_netcdf(sorted_filename)

    return data, metrics
