import argparse
import pandas as pd
import sys
from pathlib import Path
from analysis_codes import helpers, runners

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Parser")
    parser.add_argument('--procs', type=int, required=True)
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--wait_n', default=3, required=False)
    parser.add_argument('--max_time_per_job', default=pd.Timedelta("24 hours"), type=pd.Timedelta, required=False)
    parser.add_argument('--check_interval', default=pd.Timedelta("10 min"), type=pd.Timedelta, required=False)
    parser.add_argument('--function', default="run", type=str, required=False)
    parser.add_argument('--log', type=str, required=True)

    arguments = vars(parser.parse_args(sys.argv[1:]))

    functions = {"run":runners.run_model}

    arguments["max_time_per_job"] = arguments["max_time_per_job"].total_seconds()

    arguments["check_interval"] = arguments["check_interval"].total_seconds()

    output = Path(arguments["log"])

    print(arguments)
    
    jobs = [(functions[arguments["function"]],opt) for opt in helpers.generate_options(Path(arguments["input"]))]

    with runners.Timer_pool(nprocs=arguments["procs"], wait_n_times=arguments["wait_n"], logfile = output) as tp:
        tp.execute(jobs, absolute_timeout=arguments["max_time_per_job"], check_timeout=arguments["check_interval"])


    