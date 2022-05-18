import io
import json
import re
import zipfile
from itertools import product
from pathlib import Path

import h5py
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


class Collated_output:
    def __init__(self, filename, save_types=["pdf"], compression=zipfile.lzma):

        self.output = zipfile.ZipFile(filename, "w")
        self.save_types = save_types

    def __enter__(self):

        return self

    def savefig(self, figure, name):

        for suffix in self.save_types:
            out = io.BytesIO()
            savename = f"{suffix}/{name}.{suffix}"
            figure.savefig(out, format=suffix)
            self.output.writestr(savename, out.getvalue())

    def __exit__(self, type=None, value=None, traceback=None):

        self.output.close()


def get_plot_filename(basepath, filename, removal=re.compile("_seed-[0-9]+")):
    # Removes the seed (since that is merged by analysis code) and provides a zipfilename to save plots

    removals = removal.findall(filename)

    endfilename = filename

    for item in removals:
        endfilename = endfilename.replace(item, "")

    endfilename = Path(endfilename).with_suffix(".zip").name

    return basepath / endfilename


def find_lists(x):

    iterables = []

    for key, item in x.items():
        if type(item) is list:
            iterables.append(key)

    return iterables


def generate_options(filepath):
    # There is probably a smarter way to do this, but this has already taken long enough.

    possible_paths = {"datafile", "model", "restriction", "outputbase"}
    unhashable = {"initoptions"}

    forbidden = possible_paths | unhashable

    with filepath.open() as infile:
        opts = json.load(infile)

    experiments = list(opts.keys())

    # Options pertaining to all
    experiments.remove("general")

    general = opts["general"]

    output_base = Path(general["outputbase"])

    for ex in experiments:

        ex_opts = general.copy()

        output = output_base / ex

        ex_opts.update(opts[ex])

        lists = find_lists(ex_opts)

        # It makes no sense to loop over extractions.
        # And it must be a list due to the format
        if "extract" in lists:
            lists.remove("extract")

        if not lists:
            yield ex_opts

        else:
            iterables = []

            for item in lists:
                iterables.append(ex_opts[item])

            for values in product(*iterables):
                specifics = dict(zip(lists, values))

                spec_name = "_".join(
                    [
                        str(key) + "-" + str(item)
                        for key, item in specifics.items()
                        if key not in forbidden
                    ]
                )

                spec_name = ex + "_" + spec_name

                # TODO? if you name all your different datafiles with the same name, you are going to have a bad time
                for item in possible_paths:
                    if item in specifics.keys():
                        spec_name = (
                            spec_name
                            + "_"
                            + str(item)
                            + "-"
                            + Path(specifics[item]).stem
                        )

                # Maybe adding everything to filename can only go so far.
                # Some other naming scheme could work with metadata inside the files
                # This does have human readability, but
                if "modelopts" in specifics.keys():
                    spec_name = (
                        spec_name
                        + "_"
                        + "initoptions"
                        + "-"
                        + str(ex_opts["initoptions"].index(specifics["initoptions"]))
                    )

                spec_name = spec_name + ".h5"

                specifics["filename"] = output / spec_name

                # Of course we need another copy, why not
                spec_opts = ex_opts.copy()
                spec_opts.update(specifics)

                # This has now served its purpose
                if "outputbase" in spec_opts:
                    del spec_opts["outputbase"]

                yield spec_opts


def dump_hdf5(filename, **data):

    # for i, k in data.items():
    #    print(i, type(k))

    with h5py.File(filename, "w") as f:
        for name, d in data.items():
            #    print("Saving:", name)

            if type(d) is list:
                # So apparently hdf5 library doesn't like unicode.
                if type(d[0]) is str:
                    d = np.array(d, dtype="S")
                else:
                    d = np.array(d)

            # Can't have nice things
            if type(d) is dict:
                d = str(d)

            if isinstance(d, Path):
                d = str(d)

            f.create_dataset(name, data=d)


def maximize():
    plt.get_current_fig_manager().window.showMaximized()


def normalize(F, G):
    return F / np.sum(F, axis=-1, keepdims=True), G * np.sum(F, axis=-1)[:, None, :]


def normalize_chains(F, G, factor):
    return {
        "F": F / np.sum(F, axis=-1, keepdims=True),
        "G": factor * G * np.sum(F, axis=-1)[:, :, None, :],
    }


def labels(odict):
    return list(odict.keys())


def sort_by_batch_label(data, labels, axis=2):
    # Isin't really the most efficient method ever, but ¯\_(ツ)_/¯
    # Im sure there is a nicer way to do this,
    # but all the obvious ones return memoryerror on 16 Gb machine

    if axis == 3:
        return np.stack([data[n, :, labels[n, :]] for n in range(data.shape[0])])
    elif axis == 2:
        return np.stack([data[n, labels[n, :], :] for n in range(data.shape[0])])
    elif axis == "1d":
        return np.stack([data[n, labels[n, :]] for n in range(data.shape[0])])

    raise ValueError("Axis must be 2, 3, or '1d'")


def select_hungarian(values, exemplars, distance_metric="correlation"):

    distances = cdist(exemplars, values, distance_metric)

    row_ind, selections = linear_sum_assignment(distances)

    selections = selections[np.argsort(row_ind)]

    return selections


def calculate_Z_contribution(F, G):

    components = F.shape[0]

    Z = []

    for c in range(components):
        Z.append(
            (((G[:, c][:, None]) @ (F[c, :][None, :])) / np.mean(G[:, c])).flatten()
        )

    return np.stack(Z)


def distance_label_batch_by_Z(F, G, exemplars, metric="euclidean"):

    samples = F.shape[0]

    lists = [
        select_hungarian(
            calculate_Z_contribution(F[n, ...].squeeze(), G[n, ...].squeeze()),
            exemplars,
            metric,
        )
        for n in range(samples)
    ]

    return np.vstack(lists)


def distance_label_batch(F, exemplars, metric="correlation"):

    samples = F.shape[0]

    lists = [
        select_hungarian(F[n, :, :].squeeze(), exemplars, metric)
        for n in range(samples)
    ]

    return np.vstack(lists)


def distance_label_batch_chains(F, exemplars, metric="correlation"):

    samples = F.shape[0]
    chains = F.shape[1]

    lists = []
    for n in range(samples):
        for i in range(chains):
            lists.append(select_hungarian(F[n, i, :, :].squeeze(), exemplars, metric))

    return np.vstack(lists)


def label_and_sort_data(
    F, G, data, errors, by="Hungarian", distance_metric="correlation"
):

    if by == "Hungarian":
        # Take a "random" (1000 isin't special afaik) one as a starting point, this could be changed
        indices = distance_label_batch(F, F[1000, :, :], metric=distance_metric)
    else:
        ValueError("Unknown labeling techique {}".format(by))

    sorted_F = sort_by_batch_label(F, indices)
    sorted_G = sort_by_batch_label(G, indices, axis=3)

    return {"F": sorted_F, "G": sorted_G}


def get_quantiles_and_median(x, axis=0, quantiles=[0.025, 0.25, 0.75, 0.975]):
    """
    Just getting the default values in a convenient package.
    Yes technically quantile 0 would be median, but I want it in another variable
    """

    median = np.median(x, axis=axis)
    quantiles = np.quantile(x, quantiles, axis=axis)

    return median, quantiles


def quantile_data(F, G):

    result = {}

    result["F"] = np.quantile(F, [0.025, 0.5, 0.975], axis=0)
    result["G"] = np.quantile(G, [0.025, 0.5, 0.975], axis=0)

    return result
