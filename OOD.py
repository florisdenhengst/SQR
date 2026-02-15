#!/usr/bin/env python
# coding: utf-8

# Imports and configuration
# - Core ML libraries, data fetching and utilities
# - SHAP is used for OOD feature selection when applicable
import numpy as np
import scipy
import sympy as sp

import lightgbm as lgb
import optuna
import copy
from sklearn.model_selection import KFold
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
import time

# SHAP voor OOD-featureselectie (gebruikt bij het OOD-pad)
import shap

# Import PySR (zorg dat Julia-dependencies geconfigureerd zijn voordat PySR wordt gebruikt)
from pysr import PySRRegressor
import torch

N_SPLITS = 5
N_ITERS = 900 #<--- TESTRUN - MOVE UP TO 500(?) FOR FULL EXPERIMENT
DATASET_ID = int(sys.argv[1])

# Parse command line arguments with defaults
tau_argv = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
sample_size = int(sys.argv[3]) if len(sys.argv) > 3 else 100

def get_feature_type(data_column):
    unique_values = np.unique(data_column)
    num_unique_values = len(unique_values)
    if num_unique_values <= 10:
        return "categorical"
    if num_unique_values == 2:
        return "binary"
    return "numerical"

def get_categorical_features(X):
    categorical_features = []
    for i, col in enumerate(X.T):
        unique_values = np.unique(col)
        num_unique_values = len(unique_values)
        if num_unique_values <= 10:
            categorical_features.append(i)
    return categorical_features

def create_dummy_variables(X, categorical_features):
    if categorical_features == []:
        return X
    else:
        X_categorical = X[:, categorical_features]
        encoder = OneHotEncoder(sparse_output=False)
        encoder.fit(X_categorical)
        X_dummy_categorical = encoder.transform(X_categorical)
        X_numerical = np.delete(X, categorical_features, axis=1)
        X_with_dummies = np.concatenate([X_numerical, X_dummy_categorical], axis=1)
        return X_with_dummies

def process_fold_scores(model_name, ds_name, fold_scores, resultsdict):
    for metric, scores in fold_scores.items():
        resultsdict[model_name][metric][ds_name].extend(scores)

# datasets - ALL regression datasets
regression_dataset_namestry = [regression_dataset_names[DATASET_ID]]  # ONLY FIRST 4 DATASETS FOR TESTING
print(f"Processing {len(regression_dataset_namestry)} datasets...")

# Helper-functies gebruikt in beide paden
def pinball_loss(y_true, y_pred, tau):
    residuals = y_true - y_pred
    loss = np.maximum(tau * residuals, (tau - 1) * residuals)
    return np.mean(loss)

def normalized_pinball_loss(y_true, y_pred, global_min, global_max, tau):
    range_y = global_max - global_min
    loss = pinball_loss(y_true, y_pred, tau)
    if range_y == 0:
        raise ValueError('Y range is 0!')
    return loss / range_y

def absolute_coverage_error(y_true, y_pred, tau):
    coverage = np.mean(y_true <= y_pred)
    return np.abs(coverage - tau)

def calculate_expression_complexity(expression, complexity_of_operators):
    try:
        expr = sp.sympify(expression)
    except sp.SympifyError:
        raise ValueError("Invalid expression")
    complexity = 0
    for atom in sp.preorder_traversal(expr):
        if isinstance(atom, sp.Symbol):
            complexity += 1
        elif isinstance(atom, (int, float, sp.Integer, sp.Float)):
            complexity += 1
        elif atom in complexity_of_operators:
            complexity += complexity_of_operators[atom]
    return complexity

def objective_sqr(trial, train_X, train_y, val_X, val_y, params, tau):
    params = copy.deepcopy(params)
    params.update({'parsimony': trial.suggest_float('parsimony', 0.0, 0.0)})
    modelq = PySRRegressor(**params)
    modelq.fit(train_X, train_y)
    y_pred = modelq.predict(val_X)
    return pinball_loss(val_y, y_pred, tau)

def objective_lgb(trial, train_X, train_y, val_X, val_y, tau, alpha=None):
    params = {
        'objective': 'quantile',
        'alpha': tau if alpha is None else alpha,
        'num_leaves': trial.suggest_int('num_leaves', 2, 100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5, log=True),
        'max_depth': trial.suggest_int('max_depth', 1, 20),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'random_state': 42,
        'bagging_seed': 42,
        'feature_fraction_seed': 42,
        'data_random_seed': 42,
        'verbose': -1,
    }
    if params['min_child_samples'] >= params['num_leaves']:
        raise optuna.exceptions.TrialPruned()
    model = lgb.LGBMRegressor(**params)
    model.fit(train_X, train_y)
    y_pred = model.predict(val_X)
    return pinball_loss(val_y, y_pred, tau=tau)

def objective_linear(trial, train_X, train_y, val_X, val_y, tau):
    max_iter = trial.suggest_int('max_iter', 1000, 5000)
    model = QuantReg(train_y, train_X)
    results = model.fit(q=tau, max_iter=max_iter)
    y_pred = results.predict(val_X)
    return pinball_loss(val_y, y_pred, tau)

class QuantileDecisionTreeRegressor:
    def __init__(self, quantile=0.9, min_samples_leaf=5, random_state=42):
        self.quantile = quantile
        self.min_samples_leaf = min_samples_leaf
        self.tree = DecisionTreeRegressor(min_samples_leaf=min_samples_leaf, random_state=random_state)
    def fit(self, X, y):
        self.tree.fit(X, y)
        self._add_quantile_info(X, y)
    def _add_quantile_info(self, X, y):
        leaf_indices = self.tree.apply(X)
        unique_leaves = np.unique(leaf_indices)
        self.quantile_values = {}
        for leaf in unique_leaves:
            leaf_y = y[leaf_indices == leaf]
            self.quantile_values[leaf] = np.percentile(leaf_y, self.quantile * 100)
    def predict(self, X):
        leaf_indices = self.tree.apply(X)
        predictions = np.array([self.quantile_values[leaf] for leaf in leaf_indices])
        return predictions

def objective_tree(trial, train_X, train_y, val_X, val_y, tau):
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 2, 50)
    model = QuantileDecisionTreeRegressor(quantile=tau, min_samples_leaf=min_samples_leaf)
    model.fit(train_X, train_y)
    y_pred = model.predict(val_X)
    return pinball_loss(val_y, y_pred, tau=tau)

# Hoofdlogica
SEED = 42
np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)
optuna.logging.set_verbosity(optuna.logging.WARNING)

binary_operators = ["+", "*", "/", "-"]
unary_operators = ["exp", "sin", "cos", "log", "square"]
complexity_of_operators = {"+": 1, "-": 1, "*": 1, "/": 2, "exp": 4, "sin": 3, "cos": 3, "log": 3, "square": 2}

# -------------------- PAD MET CLI-SAMPLING + OOD-ONLY --------------------
# OOD-selectie wordt alleen toegepast wanneer alle inputfeatures continu zijn.
# Code runt ENKEL op OOD data (Out-Of-Distribution split)
# Verwijder datasets met categorische features.

if True:
    SEED = 42
    np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)
    QUANTILE = tau_argv

    results = {
        "SQR": {"losses": {ds_name: [] for ds_name in regression_dataset_namestry},
                "coverage": {ds_name: [] for ds_name in regression_dataset_namestry},
                "complexity": {ds_name: [] for ds_name in regression_dataset_namestry},
                "time_all": {ds_name: [] for ds_name in regression_dataset_namestry},
                "time_fit": {ds_name: [] for ds_name in regression_dataset_namestry},
                "tau": QUANTILE,},
        "LightGBM": {"losses": {ds_name: [] for ds_name in regression_dataset_namestry},
                     "coverage": {ds_name: [] for ds_name in regression_dataset_namestry},
                     "time_all": {ds_name: [] for ds_name in regression_dataset_namestry},
                     "time_fit": {ds_name: [] for ds_name in regression_dataset_namestry},
                     "tau": QUANTILE,},
        "DecisionTree": {"losses": {ds_name: [] for ds_name in regression_dataset_namestry},
                         "coverage": {ds_name: [] for ds_name in regression_dataset_namestry},
                         "complexity": {ds_name: [] for ds_name in regression_dataset_namestry},
                         "time_all": {ds_name: [] for ds_name in regression_dataset_namestry},
                         "time_fit": {ds_name: [] for ds_name in regression_dataset_namestry},
                         "tau": QUANTILE,},
        "LinearQuantile": {"losses": {ds_name: [] for ds_name in regression_dataset_namestry},
                           "coverage": {ds_name: [] for ds_name in regression_dataset_namestry},
                           "complexity": {ds_name: [] for ds_name in regression_dataset_namestry},
                           "time_all": {ds_name: [] for ds_name in regression_dataset_namestry},
                           "time_fit": {ds_name: [] for ds_name in regression_dataset_namestry},
                           "tau": QUANTILE,}
    }

    def make_json_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_json_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_json_serializable(v) for v in obj]
        if isinstance(obj, (np.floating,)):
            if np.isnan(obj):
                return None
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return make_json_serializable(obj.tolist())
        return obj

    for regression_dataset in regression_dataset_namestry:
        try:
            print(regression_dataset, QUANTILE)
            X1, y = fetch_data(regression_dataset, return_X_y=True)

            # Verwerk ENKEL datasets met ALLE continuous features (geen categoricals)
            cat_feats = get_categorical_features(X1)
            if len(cat_feats) > 0:
                print(f"  ⊘ SKIPPED: categorische features aanwezig")
                continue
            
            print(f"  ✓ Processing OOD (all continuous)")
            X = X1.copy()

            global_min, global_max = np.min(y), np.max(y)
            kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

            fold_scores_sqr = {"losses": [], "coverage": [], "complexity": [], 'time_all': [], 'time_fit': []}
            fold_scores_lgb = {"losses": [], "coverage": [], 'time_all': [], 'time_fit': []}
            fold_scores_tree = {"losses": [], "coverage": [], "complexity": [], 'time_all': [], 'time_fit': []}
            fold_scores_linear = {"losses": [], "coverage": [], "complexity": [], 'time_all': [], 'time_fit': []}

            for train_index, test_index in kf.split(X):
                train_X, test_X = X[train_index], X[test_index]
                train_y, test_y = y[train_index], y[test_index]

                # --- SHAP-gedreven OOD-selectie op train-fold ---
                shap_lgb = lgb.LGBMRegressor(
                    objective='quantile', alpha=QUANTILE,
                    num_leaves=31, learning_rate=0.1, n_estimators=200,
                    random_state=SEED, verbose=-1
                )
                shap_lgb.fit(train_X, train_y)

                explainer = shap.TreeExplainer(shap_lgb)
                shap_vals = explainer.shap_values(train_X)  # (n_samples, n_features)
                mean_abs = np.mean(np.abs(shap_vals), axis=0)
                top_feat_idx = int(np.argmax(mean_abs))

                p90 = np.quantile(train_X[:, top_feat_idx], 0.90)

                # ID data: training data (normal distribution)
                mask_train_id = train_X[:, top_feat_idx] <= p90
                X_train_id, y_train_id = train_X[mask_train_id], train_y[mask_train_id]

                # OOD data: test data (out-of-distribution split)
                mask_test_ood = test_X[:, top_feat_idx] > p90
                X_test_ood, y_test_ood = test_X[mask_test_ood], test_y[mask_test_ood]

                # Skip fold als ID train/OOD test set leeg is
                if X_train_id.shape[0] == 0 or X_test_ood.shape[0] == 0:
                    print(f"    Fold skipped: ID train size={X_train_id.shape[0]}, OOD test size={X_test_ood.shape[0]}")
                    continue

                # Sample ID training data
                sqr_sample_size = min(X_train_id.shape[0], sample_size)
                sqr_sample_index = np.random.choice(np.arange(X_train_id.shape[0]), sqr_sample_size, replace=False)
                sqr_train_X, sqr_train_y = X_train_id[sqr_sample_index], y_train_id[sqr_sample_index]

                # SQR (PySR) - ENKEL op OOD test data
                params_sqr = {
                    "niterations": N_ITERS,
                    "binary_operators": binary_operators,
                    "unary_operators": unary_operators,
                    "complexity_of_operators": complexity_of_operators,
                    "elementwise_loss": f"QuantileLoss({QUANTILE})",
                    "deterministic": False,
                    "parallelism": "multithreading",
                    "temp_equation_file": True,
                    "parsimony": 0.0,
                    "progress": False,
                    "verbosity": 0,
                    "batch_size": 5000,
                    "random_state": SEED,
                }
                modelq = PySRRegressor(**params_sqr)
                t1 = time.time(); modelq.fit(sqr_train_X, sqr_train_y); t2 = time.time()
                y_pred_symbolic_ood = modelq.predict(X_test_ood)
                fold_scores_sqr["losses"].append(
                    normalized_pinball_loss(y_test_ood, y_pred_symbolic_ood, global_min, global_max, tau=QUANTILE)
                )
                fold_scores_sqr["coverage"].append(
                    absolute_coverage_error(y_test_ood, y_pred_symbolic_ood, tau=QUANTILE)
                )
                try:
                    fold_scores_sqr["complexity"].append(calculate_expression_complexity(modelq.sympy(), complexity_of_operators))
                except Exception:
                    fold_scores_sqr["complexity"].append(np.nan)
                fold_scores_sqr['time_all'].append(t2 - t1)
                fold_scores_sqr['time_fit'].append(t2 - t1)

                # LightGBM - train on ID data, test on OOD data
                t1 = time.time()
                study_lgb = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED))
                study_lgb.optimize(lambda trial: objective_lgb(trial, X_train_id, y_train_id, X_test_ood, y_test_ood, tau=QUANTILE), n_trials=10)
                best_params_lgb = study_lgb.best_params
                model_lgb = lgb.LGBMRegressor(objective='quantile', alpha=QUANTILE, **best_params_lgb)
                t2 = time.time(); model_lgb.fit(X_train_id, y_train_id); t3 = time.time()
                y_pred_lgb_ood = model_lgb.predict(X_test_ood)
                fold_scores_lgb["losses"].append(normalized_pinball_loss(y_test_ood, y_pred_lgb_ood, global_min, global_max, tau=QUANTILE))
                fold_scores_lgb["coverage"].append(absolute_coverage_error(y_test_ood, y_pred_lgb_ood, tau=QUANTILE))
                fold_scores_lgb['time_all'].append(t3-t1)
                fold_scores_lgb['time_fit'].append(t3-t2)

                # DecisionTree - train on ID data, test on OOD data
                t1 = time.time()
                study_tree = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED))
                study_tree.optimize(lambda trial: objective_tree(trial, X_train_id, y_train_id, X_test_ood, y_test_ood, tau=QUANTILE), n_trials=10)
                best_params_tree = study_tree.best_params
                model_tree = QuantileDecisionTreeRegressor(quantile=QUANTILE, min_samples_leaf=best_params_tree['min_samples_leaf'])
                t2 = time.time(); model_tree.fit(X_train_id, y_train_id); t3 = time.time()
                y_pred_tree_ood = model_tree.predict(X_test_ood)
                fold_scores_tree["losses"].append(normalized_pinball_loss(y_test_ood, y_pred_tree_ood, global_min, global_max, tau=QUANTILE))
                fold_scores_tree["coverage"].append(absolute_coverage_error(y_test_ood, y_pred_tree_ood, tau=QUANTILE))
                fold_scores_tree["complexity"].append(model_tree.tree.tree_.node_count)
                fold_scores_tree['time_all'].append(max(t3-t1, 0.0))
                fold_scores_tree['time_fit'].append(max(t3-t2, 0.0))

                # LinearQuantile - train on ID data, test on OOD data
                t1 = time.time()
                study_linear = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED))
                study_linear.optimize(
                    lambda trial: objective_linear(trial, X_train_id, y_train_id, X_test_ood, y_test_ood, tau=QUANTILE),
                    n_trials=10
                )
                best_params_linear = study_linear.best_params
                t2 = time.time()
                model_linear = QuantReg(y_train_id, X_train_id).fit(q=QUANTILE, max_iter=best_params_linear['max_iter'])
                t3 = time.time()
                y_pred_linear_ood = model_linear.predict(X_test_ood)
                fold_scores_linear["losses"].append(
                    normalized_pinball_loss(y_test_ood, y_pred_linear_ood, global_min, global_max, tau=QUANTILE)
                )
                fold_scores_linear["coverage"].append(
                    absolute_coverage_error(y_test_ood, y_pred_linear_ood, tau=QUANTILE)
                )
                fold_scores_linear["complexity"].append(X_train_id.shape[1])
                fold_scores_linear['time_all'].append(t3 - t1)
                fold_scores_linear['time_fit'].append(t3 - t2)

            process_fold_scores("SQR", regression_dataset, fold_scores_sqr, results)
            process_fold_scores("LightGBM", regression_dataset, fold_scores_lgb, results)
            process_fold_scores("DecisionTree", regression_dataset, fold_scores_tree, results)
            process_fold_scores("LinearQuantile", regression_dataset, fold_scores_linear, results)
            
            num_folds = len(fold_scores_sqr['losses'])
            if num_folds > 0:
                print(f"  ✓ Completed: {num_folds} folds processed\n")
            else:
                print(f"  ⊘ No valid folds (all OOD splits were empty)\n")

        except Exception as e:
            print(f"  ✗ Error: {e}\n")

    # Verwijder datasets met geen resultaten uit de dictionary
    datasets_to_remove = []
    for model in results.keys():
        for ds_name in results[model]["losses"].keys():
            if len(results[model]["losses"][ds_name]) == 0:
                datasets_to_remove.append(ds_name)
    
    datasets_to_remove = list(set(datasets_to_remove))  # Unieke waarden
    for ds_name in datasets_to_remove:
        for model in results.keys():
            for metric in results[model].keys():
                if metric != "tau" and ds_name in results[model][metric]:
                    del results[model][metric][ds_name]

    print("\n" + "="*60)
    print("SUMMARY: Processing complete")
    print("="*60)
    print(f"Datasets processed: {len(regression_dataset_namestry) - len(datasets_to_remove)}")
    print(f"Datasets skipped: {len(datasets_to_remove)}")
    print("="*60)
    
    # Sla resultaten op als JSON
    output_file = f"results_{QUANTILE}_{regression_dataset_namestry[0]}_{sample_size}OOD.json"
    with open(output_file, 'w') as f:
        json.dump(make_json_serializable(results), f, indent=2)
    print(f"✓ Results saved to: {output_file}\n")


#IN TERMINAL: python try.py 0.9 100
#- 0.9 = tau-QUANTILE
#- 100 = sample size