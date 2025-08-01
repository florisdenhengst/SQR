import numpy as np
import scipy
import sympy as sp
import torch
import lightgbm as lgb
import optuna
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_pinball_loss
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import friedmanchisquare, wilcoxon
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
import pmlb
from pmlb import fetch_data, regression_dataset_names
import random
import pandas as pd
import json
import sys
import glob
import argparse
# Set up argument parser
# parser = argparse.ArgumentParser(description="Process files matching a glob pattern.")
# parser.add_argument("pattern", help="Glob pattern to match files (e.g., '*.txt')")

# # Parse arguments
# args = parser.parse_args()

# # Use glob to find matching files
# result_files = glob.glob(args.pattern)
def filename_to_dsname(filename):
    start = len('results50_')
    stop = len(filename) - len('.json')
    return filename[start:stop]

result_files = sys.argv[1:]
results_dsname = filename_to_dsname(result_files[0])

merged_result = {}
result_sets = []
for file in result_files:
    with open(file, 'r') as f:
        result = json.load(f)
    result_sets.append(result)

models = list(result_sets[0].keys())
# Extract all available metrics dynamically
available_metrics = set()
for model, metrics in result_sets[0].items():
    available_metrics.update(metrics)
valid_metrics = list(available_metrics - {'tau',}) # Ensures only existing metrics are used


results90 = {
        "SQR": {"losses": [],
                "coverage": [],
                "complexity": [],
                },
        "LightGBM": {"losses": [],
                     "coverage": [],
                     },
        "DecisionTree": {"losses": [],
                         "coverage": [],
                         "complexity": [],
                         },
        "LinearQuantile": {"losses": [],
                           "coverage": [],
                           "complexity": [],
                           }
}

results50 = {
        "SQR": {"losses": [],
                "coverage": [],
                "complexity": [],
                },
        "LightGBM": {"losses": [],
                     "coverage": [],
                     },
        "DecisionTree": {"losses": [],
                         "coverage": [],
                         "complexity": [],
                         },
        "LinearQuantile": {"losses": [],
                           "coverage": [],
                           "complexity": [],
                           }
}

# merge the result sets
for result_set in result_sets:
    for model, metrics in sorted(result_set.items()):
        for metric, model_scores in sorted(metrics.items()):
            if metric in valid_metrics:
                for _, scores in sorted(model_scores.items()):
                    if metrics['tau'] == 0.5:
                        target = results50
                    elif metrics['tau'] == 0.9:
                        target = results90
                    else:
                        raise ValueError(f"Unknown tau: {metrics['tau']}")
                    target[model][metric].extend(scores)

# Significance level for Bonferroni correction
alpha = 0.05

# Store results
friedman_results = {}
wilcoxon_results = {}

# Perform Friedman test for each available metric
for metric in metrics:
    # Filter models that contain this metric
    valid_models = [model for model in models if metric in results90[model]]

    # Gather data for each valid model
    data = np.array([results90[model][metric] for model in valid_models])
    # Perform Friedman test (only if multiple models have the metric)
    for i, model_1 in enumerate(valid_models):
        if len(data[i,:]) > 0:
            print(','.join(map(str, [results_dsname, '.9', model_1, metric, data[i,:].mean(),
                                     data[i,:].std()])))

# Extract all available metrics dynamically
available_metrics = set()
for model in results50:
    available_metrics.update(results50[model].keys())
metrics = list(available_metrics)  # Ensures only existing metrics are used

models = list(results50.keys())  # ["SQR", "LightGBM", "DecisionTree", "LinearQuantile"]

# Significance level for Bonferroni correction
alpha = 0.05

# Store results
friedman_results = {}
wilcoxon_results = {}

# Perform Friedman test for each available metric
for metric in metrics:
    # Filter models that contain this metric
    valid_models = [model for model in models if metric in results50[model]]

    # Gather data for each valid model
    data = np.array([results50[model][metric] for model in valid_models])

    for i, model_1 in enumerate(valid_models):
        if len(data[i,:]) > 0:
            print(','.join(map(str, [results_dsname, '.9', model_1, metric, data[i,:].mean(),
                           data[i,:].std()])))

