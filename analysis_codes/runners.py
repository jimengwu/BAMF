import h5py
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from operator import itemgetter
import multiprocessing as mp
import multiprocessing.connection as connection
import pystan
from functools import partial
import time
from contextlib import redirect_stderr, redirect_stdout
import logging
import sys
import os

from io import StringIO

from . import helpers, analysis

checked = set()

killwait = 5

class Silencer():
    """
    C++/Fortran subroutines don't care if you redirected via usual means
    This is to stop them from printing everything to stdout/stderr
    Because that is a mess when there are 2+ processes on.
    Pystans logger https://pystan.readthedocs.io/en/latest/logging.html didin't help.
    """

    def __init__(self, filepath):

        self.f = filepath.open('w')
        self.f_no = self.f.fileno()

        self.stdout_actual = os.dup(sys.stdout.fileno())
        self.stderr_actual = os.dup(sys.stderr.fileno())

        os.dup2(self.f_no, sys.stdout.fileno())
        os.dup2(self.f_no, sys.stderr.fileno())

    def __enter__(self):
        return self

    def __exit__(self, type=None, value=None, traceback=None):

        os.dup2(self.stdout_actual, sys.stdout.fileno())
        os.dup2(self.stderr_actual, sys.stderr.fileno())

        os.close(self.stdout_actual)
        os.close(self.stderr_actual)

        self.f.close()

class Timer_pool():

    active_processes = {}
    start_times = {}
    end_times = {}
    running_id = 0

    table = None

    diagnostics = {"start", "stop", "duration", "filename", "status", "result", "seed"}

    def __init__(self, nprocs=2, wait_n_times=2 , logfile=None):

        self.max_processes = nprocs
        self.n = wait_n_times
        self.logfile = logfile

    def execute(self, jobs, absolute_timeout=None, check_timeout=None):

        #Using pystans own thing would be much easier.
        #But the optimization never converges for some starting points...
        #and then you get no data even from the converged runs.
        #(Well it will eventually exhaust the treedepth, but that is a long time to wait)
        #So here we are.

        #Note: None of the processes should return anything

        #Data used for bookkeeping

        self.table = pd.DataFrame(j for i,j in jobs)

        #compile all used models, once before multithreading
        for item in self.table["model"].unique():
            compile_model(Path(item))
            premodels = self.table[np.logical_or(self.table["init"] == "model_init",self.table["init"] == "multimodel")]
            if len(premodels) > 0:
                for item in premodels["initoptions"]:
                    if type(item["premodel"]) is list:
                        for k in item["premodel"]:
                            compile_model(Path(k))
                    else:
                        mod = item["premodel"]
                        compile_model(Path(mod))

        self.table["function"] = [i.__name__ for i,j in jobs]
        self.table["status"] = "undefined"
        self.table["start"] = pd.NaT
        self.table["stop"] = pd.NaT
        self.table["duration"] = np.nan

        #Hmmm, maybe there is a pattern here.
        if "initoptions" in self.table.columns:
            self.table.loc[:,"initoptions"] = self.table["initoptions"].apply(str)

        if "extract" in self.table.columns:
            self.table.loc[:,"extract"] = self.table["extract"].apply(str)

        #Path doesn't like to go quietly in to hdf5
        if "filename" in self.table.columns:
            self.table.loc[:,"filename"] = self.table["filename"].apply(str)

        self.groupers = [item for item in self.table.columns if item not in self.diagnostics]

        self.selector = itemgetter(*self.groupers)

        self.absolute_limit = absolute_timeout

        #list of function, keyword pairs
        for i, job in enumerate(jobs):

            self.running_id = i

            pipeline_full = len(self.active_processes) >= self.max_processes

            while pipeline_full:
                self.wait(check_timeout)
                pipeline_full = len(self.active_processes) >= self.max_processes
            
            
            self.spawn(job)

        while len(self.active_processes) > 0:
            self.wait(check_timeout)


    def spawn(self, job):

        p = mp.Process(target=job[0], kwargs=job[1])
        p.start()

        self.active_processes[self.running_id] = p
        self.table.loc[self.running_id, "start"] = pd.Timestamp("now")
        self.table.loc[self.running_id, "status"] = "started"

    def wait(self, timeout):
        #Well waiting like this "wastes" a core but idk what to do about it

        print(pd.Timestamp("now"), "running:", self.active_processes.keys())

        sentinels = [p.sentinel for k,p in self.active_processes.items()]

        ready = connection.wait(sentinels, timeout=timeout)

        done = None

        #I don't get why they tell me they're ready, then complain.
        #Sentinel reports ready, then close raises error.
        #So that's why we wait few seconds to let the process get its business in order
        time.sleep(killwait)

        if ready is not None:
            done = {}
            for identifier, proc in self.active_processes.items():
                    if proc.sentinel in ready:
                        done[identifier] = proc.exitcode        
                        proc.close()      

        
        self.reap(done)

        if self.logfile is not None:
            self.log(self.logfile)

    def reap(self, done):

        now = pd.Timestamp("now")

        removed = []

        #Remove those that are done
        for tag,code in done.items():
            self.table.loc[tag, "stop"] = now
            if code != 0:
                self.table.loc[tag, "status"] = str(code)
            else:
                self.table.loc[tag, "status"] = "finished"
                self.table.loc[tag, "duration"] = (now-self.table.loc[tag]["start"]).total_seconds()

            removed.append(tag)

        for tag in removed:
            self.active_processes.pop(tag)

        removed = []

        #kill and remove those that overstayed their welcome
        for tag, proc in self.active_processes.items():
            if (now - self.table.loc[tag]["start"]).total_seconds() >= self.absolute_limit:
                proc.kill()
                self.table.loc[tag, "stop"] = now
                self.table.loc[tag, "status"] = "timeout"
                removed.append(tag)

        time.sleep(killwait)

        for tag in removed:
            self.active_processes[tag].close()
            self.active_processes.pop(tag)

        if len(self.active_processes) == 0:
            return

        removed = []

        #stuck chain detection

        means = self.table.groupby(self.groupers)["duration"].mean()
        for tag, proc in self.active_processes.items():
            group = self.selector(self.table.loc[tag,:])

            if ((self.n * means[group]) < (now - self.table.loc[tag]["start"]).total_seconds()):
                proc.kill()
                self.table.loc[tag, "stop"] = now
                self.table.loc[tag, "status"] = "chain timeout"
                removed.append(tag)

        time.sleep(killwait)

        for tag in removed:
            self.active_processes[tag].close()
            self.active_processes.pop(tag) 

        return

    def log(self, filepath):
        if self.table is not None:
            self.table.to_hdf(filepath, "runs")

    def __enter__(self):
        return self

    def __exit__(self, type=None, value=None, traceback=None):

        for tag,proc in self.active_processes.items():
            proc.kill()

        time.sleep(killwait)

        #Give a sec for kill to take effect.
        for tag,proc in self.active_processes.items():
            proc.close()

        print(pd.Categorical(self.table["status"]).describe())

        if self.logfile is not None:
            self.log(self.logfile)



def compile_model(modelpath, force=False):

    #Compiling them in the worker processes would mess with timing
    #Also making sure it only gets compiled once

    picklename = modelpath.with_suffix(".pkl")

    if picklename in checked:
        return

    use_pickle = False
    if picklename.exists():
        sp = picklename.stat()
        sm = modelpath.stat()
        if sp.st_mtime >= sm.st_mtime:
            use_pickle = True

    if use_pickle and (not force):
        checked.add(picklename)
        print(f"Using precompiled model: {picklename}")
        return

    with modelpath.open("r") as infile:
      model_definition = infile.read()
    
    model = pystan.StanModel(model_code=model_definition)
    with picklename.open("wb") as f:
        pickle.dump(model, f)
        print("Compiled model:",picklename)
        checked.add(picklename)

def do_nothing(data, meta, extra):
    return data, meta, extra

def basic_normalization(data, meta, extra):

    const = np.nanmean(data["X"])

    data["X"] = data["X"]/const
    data["sigma"] = data["sigma"]/const

    meta["norm_const"] = const

    return data, meta, extra

def read_benchmark(filepath):

    data, meta, extra = read_input(filepath)

    #The original data is stored in the result file, so the postprocessing doesn't have to mix and match
    original_data = {}

    original_data["F_orig"] = pd.read_hdf(filepath, "F")
    original_data["G_orig"] = pd.read_hdf(filepath, "G")
    meta["components"] = original_data["F_orig"].index.to_numpy().astype("S")
    original_data["F_orig"] = original_data["F_orig"].values
    original_data["G_orig"] = original_data["G_orig"].values
    meta.update(original_data)

    return data, meta, extra

def read_input(filepath):

    data = pd.read_hdf(filepath, "data")
    error = pd.read_hdf(filepath, "error")

    actual_times = data.index
    scaled_time, _ = analysis.scale_times(actual_times)
    times = data.index.to_series().dt.strftime("%Y-%m-%d %H:%M:%S").values.astype("S20")
    columns = data.columns.values.astype("S")

    timedeltas = data.index.to_series().diff().dt.total_seconds()[1:].values

    #Stuff that goes to model
    data_out = {"X":data.values, "m":data.shape[1], "n":data.shape[0], "sigma":error.values, "timesteps":timedeltas, "beta_tau":np.log(np.prod(data.shape))}

    #A lower bound on a variable in the model
    data_out["minerr"] = data_out["sigma"].min() / 10

    #Stuff that goes to output file
    meta = {"columns":columns, "time_index":times}

    #Stuff that goes to preprocessing (if applicable)
    extra = {"time_actual":actual_times, "time_delta":timedeltas}

    return data_out, meta, extra

preprocessing_functions = {"nothing":do_nothing, "basic":basic_normalization}
loaders = {"basic":read_input ,"benchmark":read_benchmark}

def run_model(action, datafile, description, init, loader, model, iterations, components, preprocessing, filename, initoptions, delta, treedepth, seed, extract=["tau", "F", "G", "alpha_n", "alpha_r", "alpha_s", "lp__"]):
    """
    Should extract be explicitly always given?
    You can now still override it.
    """
    
    outpath = Path(filename)
    logpath = outpath.with_suffix(".log")

    #Make sure output directory exists
    outpath.parent.mkdir(parents=True, exist_ok=True)

    with Silencer(logpath):

        picklename = Path(model).with_suffix(".pkl")

        loaded_model = pickle.load(picklename.open('rb'))

        premodel = None
        if "premodel" in initoptions:
            if type(initoptions["premodel"]) is list:
                premodel = []
                for k in initoptions["premodel"]:
                    prepicklename = Path(k).with_suffix(".pkl")
                    premodel.append(pickle.load(prepicklename.open('rb')))      
            else:
                prepicklename = Path(initoptions["premodel"]).with_suffix(".pkl")
                premodel = pickle.load(prepicklename.open('rb'))

        #Metadata is added to output
        data, meta, extra = loaders[loader](Path(datafile))

        if preprocessing not in preprocessing_functions:
            raise ValueError(f"Unknown preprocesser {preprocessing}")

        data, meta, extra = preprocessing_functions[preprocessing](data, meta, extra)
        
        if "samples" in initoptions:
            extra["samples"] = initoptions["samples"]

        extra["seed"] = seed

        data["p"] = components

        args = ["action", "datafile", "init", "loader", "model", "iterations", "components", "preprocessing", "filename", "initoptions", "delta", "treedepth", "seed", "extract"]

        for key in args:
            locs = locals()
            put = locs[key]
            if put is not dict:
                meta["arg_"+key] = put
            else:
                meta["arg_"+key] = str(put)

        #Pystans runner really doesn't like partial functions which is the more elegant way to do this

        def model_init():

            preoptions = {}

            if "premodel_options" in initoptions:
                preoptions.update(initoptions["premodel_options"])

            inits = premodel.optimizing(data, **preoptions, as_vector=True)

            print(type(inits))
            print(inits)

            return inits


        def init_nothing():
            return {}


        init_functions = {"model_init":model_init, "nothing":init_nothing}
    
        if init not in init_functions:
            raise ValueError(f"Unknown init {init}")

        init_function = init_functions[init]
        wall_start = pd.Timestamp("now")
  

        if action == "sampling":
            fitted = loaded_model.sampling(data=data, iter=iterations, init=init_function, chains=1, n_jobs=1, seed=seed, control={"adapt_delta":delta, "max_treedepth":treedepth})
            results = fitted.extract(pars=extract, permuted=False, inc_warmup=True)
        elif action == "map":
            #We want random init here
            results =  loaded_model.optimizing(data=data, iter=iterations, as_vector=True, seed=seed)
        else:
            raise ValueError(f"Unknown action {action}")

        wall_end = pd.Timestamp("now")
        wall_duration = (wall_end - wall_start).total_seconds()
        
        helpers.dump_hdf5(outpath, **results, description=description, wall_duration=wall_duration, **data, **meta)