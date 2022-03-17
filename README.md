# BAMF

Bayesian Autocorrelated Matrix Factorization

## Installation

Install the conda packages: (jupyter, matplotlib and seaborn are for demonstration, not required by library):

    conda create --name bamfdev python=3.9 cython pip numpy scipy pandas pytables xarray h5py matplotlib seaborn jupyterlab

then in the enviroment:

    pip install pystan==2.19.1.1

It might also work if you install everything from channel conda-forge in anaconda.

## Usage

For the example we are going to use a datafile, where the truth is known, this is normally not the case.
This is why at the end there is explanation of how you would run this normally, but that takes ~1 hour even on my laptop so we are not going to do that today.

### Example

To run the example sampling do

    python experiment_runner.py --procs 2 --input json/bamf_tutorial.json --log results/run_info.h5

procs is the number of processors, so leave enough room that zoom won't freeze.
Log file is where it collects all the run information, we will need it for the next step. It's a pandas dataframe stored as the variable "runs"
The json file lists all options relating to the actual runs themselves. We'll go through this while it runs.

You will get unsorted data and STAN log files as results. One should usually look at the STAN logs if there are big problems.

After the runs are complete we can run the sorting + some basic analysis

    python experiment_analyzer.py --input ./results/run_info.h5 --output ./results/analysis/metrics.h5 --results ./plots/benchmark --benchmark True --reference_library datafile

This will run the analysis in benchmark mode (Where we know the true G), using the original input file as a source for names/references to F (you could have a separate file and give that). You can also omit it, but the components are then named 0,1,etc.

As a result, for each separate run, you will get some diagnostic plots (.zip) and an xarray dataset containing the sorted results (.nc4).

Just for reference, we can do this without the benchmark + reference information:

    python experiment_analyzer.py --input ./results/run_info.h5 --output ./results/analysis/metrics_normal.h5 --results ./plots/normal

The output files metrics*.h5 contain some diagnostics like reconstruction, they are pandas dataframes, stored as the variable "metrics".

For a more representative example you can run:

    python experiment_runner.py --procs 2 --input json/bamf_tutorial_normal_mode.json --log results/run_info_normal.h5

Note, in this case the benchmark info is omitted, so you use the last example of how to run the analysis.

Some notes on the normal example:
* Running with fewer components than actual usually works and gives a solution
* Running with more takes longer and usually gives a poor result for at least one component
* Because you usually don't know the exact amount of components, it is mostly a matter of an educated guess, or running with more components until you get poor ones. Some tests with real datasets indicate that there could be 20+ components in mass spectrometry data, so that might be not really practical.
* Giving good names to your runs is usually necessary to keep track of things in the long term
* There are few sanity checks, so doing something like running with 0 components just crashes ungracefully.


# Model initialization

Sometimes with longer timeseries a random starting point leads to especially slow convergence (it can also fail by being too improbable). In these cases we want to supply a closer guess to initialize the model. One way to do this would be to run NMF or similar before and feed that answer to the model, but that would require running another model(NMF). Since we already have a model, we can just run it in the maximum a priori (MAP) mode to get a point estimate of a maximum in the probability. Then we proceed to sample from it. Note that we don't actually need the esimate to converge, it just needs to be a plausible solution to the factorization.

With very small datasets the MAP estimate fails, even though sampling with random initialization works. We are not completely sure of the cause, but it should be noted that the model shouldn't be used for very small datasets anyways since we estimate a lot of variables compared to the data size there.

There is another json script showing how to set up model initialization. Basically the only arguments are how many iterations should be run before feeding the result into the sample.

    python experiment_runner.py --procs 2 --input json/bamf_tutorial_model_init.json --log results/run_init.h5

    python experiment_analyzer.py --input ./results/run_init.h5 --output ./results/analysis/metrics_initialized.h5 --results ./plots/initialized --benchmark True --reference_library datafile

Again running without the benchmark data works as well

    python experiment_analyzer.py --input ./results/run_init.h5 --output ./results/analysis/metrics_initialized_normal.h5 --results ./plots/initialized_normal

The initial model results are fed directly to the sampler, so the output data has the exact same format. In this case we can note that the results are also a bit better than the shorter dataset (which was this one cut in half in both dimensions), but that could also be a result of having more data.