# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import collections
 

import argparse
import functools
import glob
import pickle
import itertools
import json
import os
import random
import sys

import numpy as np
import tqdm

from domainbed import datasets
from domainbed import algorithms
from domainbed.lib import misc, reporting
from domainbed import model_selection
from domainbed.lib.query import Q
import warnings
import operator

def remove_key(d,key):
    new_d = d.copy()
    new_d.pop(key)
    return new_d

def recursive_freeze(obj):
    if isinstance(obj, dict):
        return frozenset((key, recursive_freeze(val)) for key, val in obj.items())
    elif isinstance(obj, list):
        return tuple(recursive_freeze(item) for item in obj)
    elif isinstance(obj, set):
        return frozenset(recursive_freeze(item) for item in obj)
    elif isinstance(obj, tuple):
        return tuple(recursive_freeze(item) for item in obj)
    else:
        return obj

def merge_records(records):
    merged_records = []
    args_set = set()  # Store unique args dictionaries

    # Group records by unique 'args' dictionaries
    for record in records:
        args = record['args'].copy()
        args.pop('holdout_fraction', None)  # Remove 'holdout_fraction' from comparison
        args_key = recursive_freeze(args)
        args_set.add(args_key)

    # Merge records with the same 'args' except for 'holdout_fraction'
    for args_key in args_set:
        args_dict = dict(args_key)
        filtered_records = [record for record in records if dict(recursive_freeze(remove_key(record['args'],'holdout_fraction'))) == args_dict]
        merged_record = {}
        for record in filtered_records:
            merged_record.update(record)
        merged_records.append(merged_record)
    return Q(merged_records)

def format_mean(data, latex, sweep_point=None):
    """Given a list of datapoints, return a string describing their mean and
    standard error"""
    if len(data) == 0:
        return None, None, "X"
    mean = 100 * np.mean(list(data))
    err = 100 * np.std(list(data) / np.sqrt(len(data)))
    if latex:
        if sweep_point is None:
            return mean, err, "{:.1f} $\\pm$ {:.1f}".format(mean, err)
        else:
            return mean, err, "{:.1f} $\\pm$ {:.1f} @ {}/{}".format(mean, err, sweep_point[0].seed, sweep_point[0].step)
    else:
        if sweep_point is None:
            return mean, err, "{:.1f} +/- {:.1f}".format(mean, err)
        else:
            return mean, err, "{:.1f} +/- {:.1f} @ {}/{}".format(mean, err, sweep_point[0].seed, sweep_point[0].step)
        
def print_table(table, header_text, row_labels, col_labels, colwidth=10,
    latex=True):
    """Pretty-print a 2D array of data, optionally with row/col labels"""
    print("")

    if latex:
        num_cols = len(table[0])
        print("\\begin{center}")
        print("\\adjustbox{max width=\\textwidth}{%")
        print("\\begin{tabular}{l" + "c" * num_cols + "}")
        print("\\toprule")
    else:
        print("--------", header_text)

    for row, label in zip(table, row_labels):
        row.insert(0, label)

    if latex:
        col_labels = ["\\textbf{" + str(col_label).replace("%", "\\%") + "}"
            for col_label in col_labels]
    table.insert(0, col_labels)

    for r, row in enumerate(table):
        misc.print_row(row, colwidth=colwidth, latex=latex)
        if latex and r == 0:
            print("\\midrule")
    if latex:
        print("\\bottomrule")
        print("\\end{tabular}}")
        print("\\end{center}")

def print_results_tables(records, selection_method, latex, start_step=0, end_step=None):
    """Given all records, print a results table for each dataset."""

    if start_step > 0:
        records = records.filter_lop("step", start_step, operator.ge)

    if end_step is not None:
        records = records.filter_lop('step', end_step, operator.lt)

    grouped_records = reporting.get_grouped_records(records)

    if selection_method == model_selection.IIDAutoLRAccuracySelectionMethod:
        for r in grouped_records:
            r['records'] = merge_records(r['records'])
       
    # Must pass in test_env for leave-one-out selection method so that it knows what test env of
    # those in test_envs to look at and hence what the val_env is (the other env in the pair of envs
    # in test_envs.
    grouped_records = grouped_records.map(lambda group:
        { **group, 
          **dict(zip(["sweep_acc", "sweep_point"], selection_method.sweep_acc(group["test_env"], group["records"]))) }
    ).filter(lambda g: g["sweep_acc"] is not None)
    """grouped records is a Q (list?) of dictionaries with entries from grouped_records above
    with added sweep_acc key. Note that sweep_acc is calculated by selection_method when it's
    been given ONLY the records of a group. This means that the test_envs it sees are the ones
    in args.test_envs.
    Then filter out ONLY groups which sweep_acc is not None."""

    # read algorithm names and sort (predefined order)
    alg_names = Q(records).select("args.algorithm").unique()
    alg_names = ([n for n in algorithms.ALGORITHMS if n in alg_names] +
        [n for n in alg_names if n not in algorithms.ALGORITHMS])

    # read dataset names and sort (lexicographic order)
    dataset_names = Q(records).select("args.dataset").unique().sorted()
    dataset_names = [d for d in datasets.DATASETS if d in dataset_names]

    for dataset in dataset_names:
        if latex:
            print()
            print("\\subsubsection{{{}}}".format(dataset))
        test_envs = range(datasets.num_environments(dataset))

        table = [[None for _ in [*test_envs, "Avg"]] for _ in alg_names]
        for i, algorithm in enumerate(alg_names):
            means = []
            for j, test_env in enumerate(test_envs):
                trial_rec = (grouped_records
                                    .filter_equals(
                                        "dataset, algorithm, test_env",
                                        (dataset, algorithm, test_env)
                    ))
                trial_accs = trial_rec.select("sweep_acc")
                sweep_point = trial_rec.select("sweep_point")
                mean, err, table[i][j] = format_mean(trial_accs, latex, sweep_point)
                means.append(mean)
            if None in means:
                table[i][-1] = "X"
            else:
                table[i][-1] = "{:.1f}".format(sum(means) / len(means))

        col_labels = [
            "Algorithm",
            *datasets.get_dataset_class(dataset).ENVIRONMENTS,
            "Avg"
        ]
        header_text = (f"Dataset: {dataset}, "
            f"model selection method: {selection_method.name}")
        # make col_width minimum of 25 and maximum of 20 and length of all entries in the table
        col_width = min(max(max(len(s) for sublist in table for s in sublist),20),25)
        print_table(table, header_text, alg_names, list(col_labels),
            colwidth=col_width, latex=latex)

    # Print an "averages" table
    if latex:
        print()
        print("\\subsubsection{Averages}")

    table = [[None for _ in [*dataset_names, "Avg"]] for _ in alg_names]
    for i, algorithm in enumerate(alg_names):
        means = []
        for j, dataset in enumerate(dataset_names):
            trial_averages = (grouped_records
                .filter_equals("algorithm, dataset", (algorithm, dataset))
                .group("trial_seed")
                .map(lambda trial_seed, group:
                    group.select("sweep_acc").mean()
                )
            )
            mean, err, table[i][j] = format_mean(trial_averages, latex)
            means.append(mean)
        if None in means:
            table[i][-1] = "X"
        else:
            table[i][-1] = "{:.1f}".format(sum(means) / len(means))

    col_labels = ["Algorithm", *dataset_names, "Avg"]
    header_text = f"Averages, model selection method: {selection_method.name}"
    print_table(table, header_text, alg_names, col_labels, colwidth=25,
        latex=latex)

if __name__ == "__main__":
    np.set_printoptions(suppress=True)

    parser = argparse.ArgumentParser(
        description="Domain generalization testbed")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--latex", action="store_true")
    parser.add_argument("--auto_lr", action="store_true")
    parser.add_argument("--start_step", type=int, default=0, help="Start step (inclusive) to begin analysis at.")
    parser.add_argument("--end_step", type=int, default=None, help="End step (exclusive) to end analysis at.")
    args = parser.parse_args()
    if args.end_step is not None and args.start_step >= args.end_step:
        raise ValueError("start step must be smaller than end step")

    results_file = "results.tex" if args.latex else "results.txt"

    sys.stdout = misc.Tee(os.path.join(args.input_dir, results_file), "w")

    records = reporting.load_records(args.input_dir)

    if args.latex:
        print("\\documentclass{article}")
        print("\\usepackage{booktabs}")
        print("\\usepackage{adjustbox}")
        print("\\begin{document}")
        print("\\section{Full DomainBed results}")
        print("% Total records:", len(records))
    else:
        print("Total records:", len(records))

    if args.auto_lr:
        SELECTION_METHODS = [model_selection.IIDAutoLRAccuracySelectionMethod]
    else:
        SELECTION_METHODS = [
            model_selection.IIDAccuracySelectionMethod,
            model_selection.LeaveOneOutSelectionMethod,
            model_selection.OracleSelectionMethod,
        ]

    for selection_method in SELECTION_METHODS:
        if args.latex:
            print()
            print("\\subsection{{Model selection: {}}}".format(
                selection_method.name))
        print_results_tables(records, selection_method, args.latex, start_step=args.start_step, end_step=args.end_step)

    if args.latex:
        print("\\end{document}")
