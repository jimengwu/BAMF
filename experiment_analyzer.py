import argparse
from email.policy import default
import pandas as pd
import sys
from pathlib import Path
from analysis_codes import analysis
from time import ctime


def bool_validator(value):
    if value == "True":
        return True
    else:
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Parser")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--results", type=str, required=True)
    parser.add_argument(
        "--benchmark", default=False, type=bool_validator, required=False
    )
    parser.add_argument("--reference_library", default=None, type=str, required=False)
    parser.add_argument(
        "--distance_metric", default="cityblock", type=str, required=False
    )
    parser.add_argument("--cutoff", default=5, type=float, required=False)

    # changed: arguments = vars(parser.parse_args(sys.argv[1:]))
    arguments = vars(parser.parse_args())

    runfile = pd.read_hdf(arguments["input"], "runs")

    complete = runfile[runfile["status"] == "finished"]

    diagnostics = {"start", "stop", "duration", "filename", "status", "result", "seed"}
    groupers = [h for h in complete.columns if h not in diagnostics]
    gb = complete.groupby(groupers)

    print("Groups to analyze:", len(gb), ".", ctime())

    rows = []

    for group in gb:

        full_row = {g: v for g, v in zip(groupers, group[0])}
        frame = group[1]

        sorted_data, metrics, indices_chains = analysis.analyze_frame(
            frame,
            benchmark=arguments["benchmark"],
            basepath=Path(arguments["results"]),
            distance_metric=arguments["distance_metric"],
            match_file=arguments["reference_library"],
            cutoff=arguments["cutoff"],
        )

        full_row.update(metrics)

        rows.append(full_row)

    print("plots saved", ctime())
    outpath = Path(arguments["output"])
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outframe = pd.DataFrame(rows)
    outframe.to_hdf(arguments["output"], "metrics")
