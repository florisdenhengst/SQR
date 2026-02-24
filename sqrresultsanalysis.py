import os
import json
import glob
import math
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shutil


METRICS_OF_INTEREST = ["losses", "coverage", "complexity", "time_all", "time_fit"]
METRIC_LABELS = {
	"losses": "Loss",
	"coverage": "Coverage",
	"complexity": "Complexity",
	"time_all": "Total Time",
	"time_fit": "Fit Time"
}


def read_summary_stats(tsv_path):
	df = pd.read_csv(tsv_path, sep="\t")
	df = df.rename(columns={"dataset": "dataset"})
	return df.set_index("dataset")


def collect_results(results_dir, summary_df):
	rows = []
	files = glob.glob(os.path.join(results_dir, "*.json"))
	for f in files:
		with open(f, "r") as fh:
			data = json.load(fh)

		# data: top-level keys are models
		for model_name, model_dict in data.items():
			tau = model_dict.get("tau", None)
			for metric in METRICS_OF_INTEREST:
				if metric not in model_dict:
					continue
				metric_map = model_dict[metric]
				for ds_name, values in metric_map.items():
					# skip if dataset not in summary
					if ds_name not in summary_df.index:
						continue
					# values is usually a list (multiple runs/folds)
					if isinstance(values, list) and len(values) > 0:
						# take mean (ignore None)
						vals = [v for v in values if v is not None]
						if len(vals) == 0:
							continue
						v = float(np.mean(vals))
					else:
						try:
							v = float(values)
						except Exception:
							continue

					n_instances = int(summary_df.loc[ds_name, "n_instances"]) if "n_instances" in summary_df.columns else None
					rows.append({
						"dataset": ds_name,
						"n_instances": n_instances,
						"model": model_name,
						"metric": metric,
						"tau": tau,
						"value": v,
						"source_file": os.path.basename(f),
					})

	df = pd.DataFrame(rows)
	return df


def plot_metric_by_instances(df, out_dir, logx=True, kind="line"):
	sns.set(style="whitegrid")

	# similar to ``plot_metric_by_features`` but for dataset size.  use
	# quartile-based bins (xs/s/l/xl) with equal numbers of points and
	# display ranges on the x-axis rather than drawing individual
	# trajectories which clutter the view.
	for metric, metric_df in df.groupby("metric"):
		for tau, tau_df in metric_df.groupby("tau"):
			sub = tau_df.dropna(subset=["n_instances"])
			if sub.empty:
				continue

			edges, labels = _quantile_bins(sub["n_instances"], n_bins=4)
			if edges is None:
				continue
			sub = sub.copy()
			sub["inst_bin"] = pd.cut(
				sub["n_instances"], bins=edges, labels=labels, include_lowest=True
			)

			plt.figure(figsize=(10, 6))
			sns.boxplot(
				x="inst_bin",
				y="value",
				hue="model",
				data=sub,
				showcaps=True,
				showfliers=False,
				whiskerprops={"linewidth": 0.5},
			)
			plt.xlabel("Number of Instances (bin label / range)")
			# annotate xticks with ranges
			ranges = [f"{labels[i]}\n{int(edges[i])}-{int(edges[i+1])}"
				 for i in range(len(labels))]
			plt.xticks(range(len(ranges)), ranges)
			metric_label = METRIC_LABELS.get(metric, metric)
			plt.ylabel(metric_label)
			title_tau = f" tau={tau}" if tau is not None else ""
			plt.title(f"Instance distribution — {metric_label}{title_tau}")

			safe_tau = str(tau).replace(".", "_") if tau is not None else "none"
			out_subdir = os.path.join(out_dir, metric, "by_instances", "distribution")
			os.makedirs(out_subdir, exist_ok=True)
			out_path = os.path.join(out_subdir, f"all_models_tau_{safe_tau}.png")
			plt.tight_layout()
			plt.savefig(out_path, dpi=150)
			plt.close()


def plot_all_models_together(df, out_dir, metric, tau=None, logx=True):
	# Produce an aggregated comparison line using instance bins.
	sel = df[df["metric"] == metric]
	if tau is not None:
		sel = sel[sel["tau"] == tau]
	if sel.empty:
		return None
	# drop missing sizes
	sel = sel.dropna(subset=["n_instances"])
	edges, labels = _quantile_bins(sel["n_instances"], n_bins=4)
	if edges is None:
		return None
	sel = sel.copy()
	sel["inst_bin"] = pd.cut(
		sel["n_instances"], bins=edges, labels=labels, include_lowest=True
	)
	agg = sel.groupby(["model", "inst_bin"]).agg(
		value=("value", "median"),
		n_instances=("n_instances", "median"),
	).reset_index()
	agg["bin_center"] = agg["n_instances"]

	plt.figure(figsize=(10, 6))
	sns.set(style="whitegrid")
	for model, mdf in agg.groupby("model"):
		plt.plot(mdf["bin_center"], mdf["value"], marker="o", label=model)

	plt.xlabel("Number of Instances (bin center)")
	metric_label = METRIC_LABELS.get(metric, metric)
	plt.ylabel(metric_label)
	title_tau = f" tau={tau}" if tau is not None else ""
	plt.title(f"Model Comparison — {metric_label}{title_tau}")
	if logx:
		plt.xscale("log")
	plt.legend()
	plt.tight_layout()
	safe_tau = str(tau).replace(".", "_") if tau is not None else "none"
	out_subdir = os.path.join(out_dir, metric, "by_instances", "comparison_binned")
	os.makedirs(out_subdir, exist_ok=True)
	out_path = os.path.join(out_subdir, f"all_models_tau_{safe_tau}.png")
	plt.savefig(out_path, dpi=150)
	plt.close()
	return out_path


def _quantile_bins(series, n_bins=4):
	"""Compute ``n_bins`` quantile-based intervals for ``series``.

	Returns a tuple ``(edges, labels)`` where
	* ``edges`` is a list of ``n_bins+1`` boundary values (monotonic),
	* ``labels`` are the corresponding textual labels ``['xs','s','l','xl']``
	  (trimmed if fewer bins are available).

	Bins contain roughly equal numbers of points.  The caller can display
	the numeric ranges alongside the labels; for example the "xs" bin
	may be rendered as ``'xs\n5–20 features'``.  If the series is constant
	or empty the function returns ``(None, None)`` to signal the caller
	should skip plotting.
	"""
	vals = series.dropna()
	if vals.empty:
		return None, None
	min_val = vals.min()
	max_val = vals.max()
	if min_val == max_val:
		return None, None
	quantiles = vals.quantile(np.linspace(0, 1, n_bins + 1)).tolist()
	# ensure the edges are strictly increasing
	edges = sorted(set(quantiles))
	if len(edges) - 1 < n_bins:
		# fall back to simple linear spacing if some quantiles collapsed
		edges = list(np.linspace(min_val, max_val, n_bins + 1))
	labels = ["xs", "s", "l", "xl"][: len(edges) - 1]
	return edges, labels


def _categorical_bins(series):
	"""Compute bin boundaries for positive categorical feature counts.

	Zeros are handled separately by callers.  This routine divides the
	positive values into three groups with roughly equal numbers of

datapoints, returning the corresponding edge values and labels
	(``xs``, ``s``, ``l``).  When there are too few positive samples the
	function falls back to the quantile-based helper used elsewhere.

	If the input is empty or there are no positive values the function
	returns ``(None, None)`` so that the caller can treat the zero-case on
	their own.
	"""
	vals = series.dropna()
	if vals.empty:
		return None, None
	# positive counts only
	pos = vals[vals > 0]
	if pos.empty:
		return None, None

	sorted_pos = pos.sort_values()
	n = len(sorted_pos)
	if n < 3:
		# too few points to split into three, let quantile helper do its work
		edges, labels = _quantile_bins(pos, n_bins=3)
		return edges, labels

	# compute split indices for roughly equal-sized buckets
	indices = [0, int(n / 3), int(2 * n / 3), n - 1]
	edges = [sorted_pos.iloc[i] for i in indices]
	# enforce monotonicity and drop duplicates that might collapse bins
	edges = sorted(set(edges))
	labels = ["xs", "s", "l"][: len(edges) - 1]
	return edges, labels

def plot_metric_by_features(df, out_dir, logx=True, kind="line"):
	sns.set(style="whitegrid")

	# show distribution of metric values across quartile-based bins of
	# feature count.  each bin contains roughly the same number of
	# datasets and is labeled ``xs``, ``s``, ``l``, ``xl``; the numeric
	# range of the smallest bin is shown on the x-axis.
	for metric, metric_df in df.groupby("metric"):
		for tau, tau_df in metric_df.groupby("tau"):
			sub = tau_df.dropna(subset=["n_features"])
			if sub.empty:
				continue

			# compute quartile edges and text labels
			edges, labels = _quantile_bins(sub["n_features"], n_bins=4)
			if edges is None:
				continue
			# bin the data with the fixed labels
			sub = sub.copy()
			sub["feat_bin"] = pd.cut(
				sub["n_features"], bins=edges, labels=labels, include_lowest=True
			)

			plt.figure(figsize=(10, 6))
			sns.boxplot(
				x="feat_bin",
				y="value",
				hue="model",
				data=sub,
				showcaps=True,
				showfliers=False,
				whiskerprops={"linewidth": 0.5},
			)
			plt.xlabel("Number of Features (bin label / range)")
			# annotate xticks with ranges
			ranges = [f"{labels[i]}\n{int(edges[i])}-{int(edges[i+1])}"
				 for i in range(len(labels))]
			plt.xticks(range(len(ranges)), ranges)
			metric_label = METRIC_LABELS.get(metric, metric)
			plt.ylabel(metric_label)
			title_tau = f" tau={tau}" if tau is not None else ""
			plt.title(f"Feature distribution — {metric_label}{title_tau}")

			safe_tau = str(tau).replace(".", "_") if tau is not None else "none"
			out_subdir = os.path.join(out_dir, metric, "by_features", "distribution")
			os.makedirs(out_subdir, exist_ok=True)
			out_path = os.path.join(out_subdir, f"all_models_tau_{safe_tau}.png")
			plt.tight_layout()
			plt.savefig(out_path, dpi=150)
			plt.close()


def plot_all_models_together_features(df, out_dir, metric, tau=None, logx=True):
	sel = df[df["metric"] == metric]
	if tau is not None:
		sel = sel[sel["tau"] == tau]
	# drop rows without feature counts
	sel = sel.dropna(subset=["n_features"])
	if sel.empty:
		return None

	# bin features into equal-count quartiles and compute median per bin
	edges, labels = _quantile_bins(sel["n_features"], n_bins=4)
	if edges is None:
		return None
	sel = sel.copy()
	sel["feat_bin"] = pd.cut(
		sel["n_features"], bins=edges, labels=labels, include_lowest=True
	)
	# aggregate both metric value and feature count
	agg = sel.groupby(["model", "feat_bin"]).agg(
	value=("value", "median"), n_features=("n_features", "median")
	).reset_index()
	# use the median feature count as the plotting coordinate
	agg["bin_center"] = agg["n_features"]

	plt.figure(figsize=(10, 6))
	sns.set(style="whitegrid")
	for model, mdf in agg.groupby("model"):
		plt.plot(mdf["bin_center"], mdf["value"], marker="o", label=model)

	plt.xlabel("Number of Features (bin center)")
	metric_label = METRIC_LABELS.get(metric, metric)
	plt.ylabel(metric_label)
	title_tau = f" tau={tau}" if tau is not None else ""
	plt.title(f"Model Comparison — {metric_label}{title_tau}")
	if logx:
		plt.xscale("log")
	plt.legend()
	plt.tight_layout()
	safe_tau = str(tau).replace(".", "_") if tau is not None else "none"
	out_subdir = os.path.join(out_dir, metric, "by_features", "comparison_binned")
	os.makedirs(out_subdir, exist_ok=True)
	out_path = os.path.join(out_subdir, f"all_models_tau_{safe_tau}.png")
	plt.savefig(out_path, dpi=150)
	plt.close()
	return out_path


def plot_metric_by_categorical_features(df, out_dir, logx=False, kind="line"):
	sns.set(style="whitegrid")

	# analogous to instances/features plots but reserve a zero bin
	for metric, metric_df in df.groupby("metric"):
		for tau, tau_df in metric_df.groupby("tau"):
			sub = tau_df.dropna(subset=["n_categorical_features"])
			if sub.empty:
				continue

			edges, labels = _categorical_bins(sub["n_categorical_features"])
			if edges is None:
				# either constant or only zeros; we'll still plot zeros if present
				# create a dummy bin
				sub = sub.copy()
				sub["cat_bin"] = sub["n_categorical_features"].apply(lambda x: "0" if x == 0 else "")
				plt.figure(figsize=(10, 6))
				sns.boxplot(
					x="cat_bin",
					y="value",
					data=sub[sub["cat_bin"] == "0"],
					showcaps=True,
					showfliers=False,
					whiskerprops={"linewidth": 0.5},
				)
				plt.xlabel("Number of Categorical Features")
				metric_label = METRIC_LABELS.get(metric, metric)
				plt.ylabel(metric_label)
				title_tau = f" tau={tau}" if tau is not None else ""
				plt.title(f"Categorical feature distribution — {metric_label}{title_tau}")
				safe_tau = str(tau).replace(".", "_") if tau is not None else "none"
				out_subdir = os.path.join(out_dir, metric, "by_categorical_features", "distribution")
				os.makedirs(out_subdir, exist_ok=True)
				out_path = os.path.join(out_subdir, f"all_models_tau_{safe_tau}.png")
				plt.tight_layout()
				plt.savefig(out_path, dpi=150)
				plt.close()
				continue

			sub = sub.copy()
			# assign bins: zeros separately
			pos_mask = sub["n_categorical_features"] > 0
			sub.loc[~pos_mask, "cat_bin"] = "0"
			sub.loc[pos_mask, "cat_bin"] = pd.cut(
				sub.loc[pos_mask, "n_categorical_features"],
				bins=edges,
				labels=labels,
				include_lowest=True,
			)

			plt.figure(figsize=(10, 6))
			sns.boxplot(
				x="cat_bin",
				y="value",
				hue="model",
				data=sub,
				showcaps=True,
				showfliers=False,
				whiskerprops={"linewidth": 0.5},
			)
			plt.xlabel("Number of Categorical Features (bin label / range)")
			# annotate xticks with ranges; zero has no numeric range
			ranges = ["0"]
			for i, lab in enumerate(labels):
				ranges.append(f"{lab}\n{int(edges[i])}-{int(edges[i+1])}")
			plt.xticks(range(len(ranges)), ranges)
			metric_label = METRIC_LABELS.get(metric, metric)
			plt.ylabel(metric_label)
			title_tau = f" tau={tau}" if tau is not None else ""
			plt.title(f"Categorical feature distribution — {metric_label}{title_tau}")

			safe_tau = str(tau).replace(".", "_") if tau is not None else "none"
			out_subdir = os.path.join(out_dir, metric, "by_categorical_features", "distribution")
			os.makedirs(out_subdir, exist_ok=True)
			out_path = os.path.join(out_subdir, f"all_models_tau_{safe_tau}.png")
			plt.tight_layout()
			plt.savefig(out_path, dpi=150)
			plt.close()



def plot_all_models_together_categorical_features(df, out_dir, metric, tau=None, logx=False):
	sel = df[df["metric"] == metric]
	if tau is not None:
		sel = sel[sel["tau"] == tau]
	if sel.empty:
		return None

	# drop NaNs then allocate zeros plus three equal bins
	sel = sel.dropna(subset=["n_categorical_features"])
	edges, labels = _categorical_bins(sel["n_categorical_features"])
	if edges is None:
		# only zeros or insufficient variation: handle separately
		sel = sel.copy()
		sel["cat_bin"] = sel["n_categorical_features"].apply(lambda x: "0" if x == 0 else "")
		agg = sel[sel["cat_bin"] == "0"].groupby(["model", "cat_bin"]).agg(
			value=("value", "median"),
			n_cat=("n_categorical_features", "median"),
		).reset_index()
		agg["bin_center"] = agg["n_cat"]
	else:
		sel = sel.copy()
		pos_mask = sel["n_categorical_features"] > 0
		sel.loc[~pos_mask, "cat_bin"] = "0"
		sel.loc[pos_mask, "cat_bin"] = pd.cut(
			sel.loc[pos_mask, "n_categorical_features"],
			bins=edges,
			labels=labels,
			include_lowest=True,
		)
		agg = sel.groupby(["model", "cat_bin"]).agg(
			value=("value", "median"),
			n_cat=("n_categorical_features", "median"),
		).reset_index()
		agg["bin_center"] = agg["n_cat"]

	plt.figure(figsize=(10, 6))
	sns.set(style="whitegrid")
	for model, mdf in agg.groupby("model"):
		plt.plot(mdf["bin_center"], mdf["value"], marker="o", label=model)

	plt.xlabel("Number of Categorical Features (bin center)")
	metric_label = METRIC_LABELS.get(metric, metric)
	plt.ylabel(metric_label)
	title_tau = f" tau={tau}" if tau is not None else ""
	plt.title(f"Model Comparison — {metric_label}{title_tau}")
	if logx:
		plt.xscale("log")
	plt.legend()
	plt.tight_layout()
	safe_tau = str(tau).replace(".", "_") if tau is not None else "none"
	out_subdir = os.path.join(out_dir, metric, "by_categorical_features", "comparison_binned")
	os.makedirs(out_subdir, exist_ok=True)
	out_path = os.path.join(out_subdir, f"all_models_tau_{safe_tau}.png")
	plt.savefig(out_path, dpi=150)
	plt.close()
	return out_path


def generate_html_index(out_dir, metrics):
	"""Generate HTML index for easy navigation through plots"""
	html_content = """<!DOCTYPE html>
<html lang="en">
<head>
	<meta charset="UTF-8">
	<meta name="viewport" content="width=device-width, initial-scale=1.0">
	<title>SQR Results — Plot Overview</title>
	<style>
		body {
			font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
			line-height: 1.6;
			color: #333;
			max-width: 1200px;
			margin: 0 auto;
			padding: 20px;
			background-color: #f5f5f5;
		}
		h1 {
			color: #2c3e50;
			border-bottom: 3px solid #3498db;
			padding-bottom: 10px;
		}
		h2 {
			color: #34495e;
			margin-top: 30px;
			border-left: 4px solid #3498db;
			padding-left: 10px;
		}
		h3 {
			color: #7f8c8d;
			font-size: 1.1em;
		}
		.metric-section {
			background: white;
			padding: 20px;
			margin: 20px 0;
			border-radius: 5px;
			box-shadow: 0 2px 5px rgba(0,0,0,0.1);
		}
		.plot-grid {
			display: grid;
			grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
			gap: 20px;
			margin: 20px 0;
		}
		.plot-item {
			background: #f9f9f9;
			padding: 15px;
			border-radius: 5px;
			border: 1px solid #e0e0e0;
		}
		.plot-item h4 {
			margin-top: 0;
			color: #2c3e50;
		}
		.plot-item a {
			display: inline-block;
			margin: 5px 5px 5px 0;
			padding: 8px 12px;
			background: #3498db;
			color: white;
			text-decoration: none;
			border-radius: 3px;
			font-size: 0.9em;
			transition: background 0.3s;
		}
		.plot-item a:hover {
			background: #2980b9;
		}
		.comparison {
			background-color: #ecf0f1;
		}
		.individual {
			background-color: #fff9e6;
		}
		.nav {
			background: white;
			padding: 15px;
			border-radius: 5px;
			margin-bottom: 20px;
			box-shadow: 0 2px 5px rgba(0,0,0,0.1);
		}
		.nav a {
			display: inline-block;
			margin-right: 15px;
			color: #3498db;
			text-decoration: none;
			font-weight: 500;
		}
		.nav a:hover {
			text-decoration: underline;
		}
	</style>
</head>
<body>
	<h1>📊 SQR Analysis — Plot Overview</h1>
	<p>All generated plots are organized below by metric and type.</p>
	
	<div class="nav">
		<strong>Quick Navigation:</strong>
"""
	
	for metric in metrics:
		metric_label = METRIC_LABELS.get(metric, metric)
		html_content += f'\t\t<a href="#{metric}">{metric_label}</a>\n'
	
	html_content += """	</div>
"""
	
	for metric in metrics:
		metric_label = METRIC_LABELS.get(metric, metric)
		metric_dir = os.path.join(out_dir, metric)
		
		if not os.path.exists(metric_dir):
			continue
		
		html_content += f"""	<div class="metric-section">
		<h2 id="{metric}">📈 {metric_label}</h2>
"""
		
		# By instances section
		by_inst_dir = os.path.join(metric_dir, "by_instances")
		if os.path.exists(by_inst_dir):
			html_content += """\t\t<h3>By Number of Instances</h3>
"""
			# Individual plots
			individual_dir = os.path.join(by_inst_dir, "individual")
			if os.path.exists(individual_dir):
				html_content += """\t\t<h4>📌 Individual Models</h4>
		<div class="plot-grid">
"""
				for png_file in sorted(os.listdir(individual_dir)):
					if png_file.endswith(".png"):
						rel_path = os.path.relpath(os.path.join(individual_dir, png_file), out_dir)
						html_content += f"""			<div class="plot-item individual">
				<h4>{png_file.replace('.png', '').replace('_', ' ')}</h4>
						<a href="{rel_path}">📸 View Plot</a>
			</div>
"""
				html_content += """		</div>
"""
			
			# Comparison plots
			comparison_dir = os.path.join(by_inst_dir, "comparison")
			if os.path.exists(comparison_dir):
				html_content += """		<h4>🔄 Model Comparison</h4>
		<div class="plot-grid">
"""
				for png_file in sorted(os.listdir(comparison_dir)):
					if png_file.endswith(".png"):
						rel_path = os.path.relpath(os.path.join(comparison_dir, png_file), out_dir)
						html_content += f"""			<div class="plot-item comparison">
				<h4>{png_file.replace('.png', '').replace('_', ' ')}</h4>
				<a href="{rel_path}">📸 View Plot</a>
			</div>
"""
				html_content += """		</div>
"""				# additional categorical bins
				dist_dir = os.path.join(by_cat_feat_dir, "distribution")
				if os.path.exists(dist_dir):
					html_content += """\t\t<h4>📊 Distribution</h4>
			<div class=\"plot-grid\">
"""
					for png_file in sorted(os.listdir(dist_dir)):
						if png_file.endswith(".png"):
							rel_path = os.path.relpath(os.path.join(dist_dir, png_file), out_dir)
							html_content += f"""\t\t	<div class=\"plot-item comparison\">\n\t\t\t\t<h4>{png_file.replace('.png','').replace('_',' ')}</h4>\n\t\t\t\t<a href=\"{rel_path}\">📸 View Plot</a>\n\t\t\t</div>\n"""
					html_content += """\t\t</div>
"""
				compb_dir = os.path.join(by_cat_feat_dir, "comparison_binned")
				if os.path.exists(compb_dir):
					html_content += """\t\t<h4>🔄 Model Comparison (binned)</h4>
			<div class=\"plot-grid\">
"""
					for png_file in sorted(os.listdir(compb_dir)):
						if png_file.endswith(".png"):
							rel_path = os.path.relpath(os.path.join(compb_dir, png_file), out_dir)
							html_content += f"""\t\t	<div class=\"plot-item comparison\">\n\t\t\t\t<h4>{png_file.replace('.png','').replace('_',' ')}</h4>\n\t\t\t\t<a href=\"{rel_path}\">📸 View Plot</a>\n\t\t\t</div>\n"""
					html_content += """\t\t</div>
"""		
		# additional bins for instances
			dist_dir = os.path.join(by_inst_dir, "distribution")
			if os.path.exists(dist_dir):
				html_content += """\t\t<h4>📊 Distribution</h4>
        <div class=\"plot-grid\">
"""
				for png_file in sorted(os.listdir(dist_dir)):
					if png_file.endswith(".png"):
						rel_path = os.path.relpath(os.path.join(dist_dir, png_file), out_dir)
						html_content += f"""            <div class=\"plot-item comparison\">\n                <h4>{png_file.replace('.png','').replace('_',' ')}</h4>\n                <a href=\"{rel_path}\">📸 View Plot</a>\n            </div>\n"""
					html_content += """        </div>
"""
			compb_dir = os.path.join(by_inst_dir, "comparison_binned")
			if os.path.exists(compb_dir):
				html_content += """\t\t<h4>🔄 Model Comparison (binned)</h4>
        <div class=\"plot-grid\">
"""
				for png_file in sorted(os.listdir(compb_dir)):
					if png_file.endswith(".png"):
						rel_path = os.path.relpath(os.path.join(compb_dir, png_file), out_dir)
						html_content += f"""            <div class=\"plot-item comparison\">\n                <h4>{png_file.replace('.png','').replace('_',' ')}</h4>\n                <a href=\"{rel_path}\">📸 View Plot</a>\n            </div>\n"""
					html_content += """        </div>
"""
		# By features section
		by_feat_dir = os.path.join(metric_dir, "by_features")
		if os.path.exists(by_feat_dir):
			html_content += """\t\t<h3>By Number of Features</h3>
"""
			# Individual plots
			individual_dir = os.path.join(by_feat_dir, "individual")
			if os.path.exists(individual_dir):
				html_content += """\t\t<h4>📌 Individual Models</h4>
		<div class="plot-grid">
"""
				for png_file in sorted(os.listdir(individual_dir)):
					if png_file.endswith(".png"):
						rel_path = os.path.relpath(os.path.join(individual_dir, png_file), out_dir)
						html_content += f"""			<div class="plot-item individual">
				<h4>{png_file.replace('.png', '').replace('_', ' ')}</h4>
				<a href="{rel_path}">📸 View Plot</a>
			</div>
"""
				html_content += """		</div>
"""
			
			# Comparison plots
			comparison_dir = os.path.join(by_feat_dir, "comparison")
			if os.path.exists(comparison_dir):
				html_content += """		<h4>🔄 Model Comparison</h4>
		<div class="plot-grid">
"""
				for png_file in sorted(os.listdir(comparison_dir)):
					if png_file.endswith(".png"):
						rel_path = os.path.relpath(os.path.join(comparison_dir, png_file), out_dir)
						html_content += f"""			<div class="plot-item comparison">
				<h4>{png_file.replace('.png', '').replace('_', ' ')}</h4>
				<a href="{rel_path}">📸 View Plot</a>
			</div>
"""
				html_content += """		</div>
"""
		
		# By categorical features section
		by_cat_feat_dir = os.path.join(metric_dir, "by_categorical_features")
		if os.path.exists(by_cat_feat_dir):
			html_content += """\t\t<h3>By Number of Categorical Features</h3>
"""
			# Individual plots
			individual_dir = os.path.join(by_cat_feat_dir, "individual")
			if os.path.exists(individual_dir):
				html_content += """\t\t<h4>📌 Individual Models</h4>
		<div class="plot-grid">
"""
				for png_file in sorted(os.listdir(individual_dir)):
					if png_file.endswith(".png"):
						rel_path = os.path.relpath(os.path.join(individual_dir, png_file), out_dir)
						html_content += f"""			<div class="plot-item individual">
				<h4>{png_file.replace('.png', '').replace('_', ' ')}</h4>
				<a href="{rel_path}">📸 View Plot</a>
			</div>
"""
				html_content += """		</div>
"""
			
			# Comparison plots
			comparison_dir = os.path.join(by_cat_feat_dir, "comparison")
			if os.path.exists(comparison_dir):
				html_content += """		<h4>🔄 Model Comparison</h4>
		<div class="plot-grid">
"""
				for png_file in sorted(os.listdir(comparison_dir)):
					if png_file.endswith(".png"):
						rel_path = os.path.relpath(os.path.join(comparison_dir, png_file), out_dir)
						html_content += f"""			<div class="plot-item comparison">
				<h4>{png_file.replace('.png', '').replace('_', ' ')}</h4>
				<a href="{rel_path}">📸 View Plot</a>
			</div>
"""
				html_content += """		</div>
"""
		
		html_content += """	</div>
"""
	
	html_content += """</body>
</html>
"""
	
	index_path = os.path.join(out_dir, "index.html")
	with open(index_path, "w", encoding="utf-8") as f:
		f.write(html_content)
	
	return index_path


def main(results_dir="results", summary_tsv="all_summary_stats.tsv", out_dir="plots", metrics=None, models=None):
	# Remove old plots so we overwrite them automatically
	if os.path.exists(out_dir):
		shutil.rmtree(out_dir)
	os.makedirs(out_dir, exist_ok=True)

	summary_df = read_summary_stats(summary_tsv)
	df = collect_results(results_dir, summary_df)

	if df.empty:
		print("No results found to plot.")
		return

	if metrics is None:
		metrics = sorted(df["metric"].unique())

	# filter models if provided
	if models is not None:
		df = df[df["model"].isin(models)]

	# produce per-metric per-tau per-model plots
	plot_metric_by_instances(df[df["metric"].isin(metrics)], out_dir)
	
	# generate plots by number of features
	# add n_features from summary_df
	if "n_features" in summary_df.columns:
		df = df.merge(summary_df[["n_features"]], left_on="dataset", right_index=True, how="left")
		plot_metric_by_features(df[df["metric"].isin(metrics)], out_dir)
		# combined plots by features
		for metric in metrics:
			for tau in df[df["metric"] == metric]["tau"].dropna().unique():
				plot_all_models_together_features(df, out_dir, metric, tau=tau)
	
	# generate plots by number of categorical features
	if "n_categorical_features" in summary_df.columns:
		df = df.merge(summary_df[["n_categorical_features"]], left_on="dataset", right_index=True, how="left")
		plot_metric_by_categorical_features(df[df["metric"].isin(metrics)], out_dir)
		# combined plots by categorical features
		for metric in metrics:
			for tau in df[df["metric"] == metric]["tau"].dropna().unique():
				plot_all_models_together_categorical_features(df, out_dir, metric, tau=tau)

	# also produce combined plots per metric/tau
	for metric in metrics:
		for tau in df[df["metric"] == metric]["tau"].dropna().unique():
			plot_all_models_together(df, out_dir, metric, tau=tau)

	# Generate HTML index
	index_path = generate_html_index(out_dir, metrics)
	print(f"Plots saved to: {os.path.abspath(out_dir)}")
	print(f"HTML index generated: {os.path.abspath(index_path)}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Plot results per metric/model/tau against dataset size.")
	parser.add_argument("--results_dir", default="results", help="Directory with result JSON files")
	parser.add_argument("--summary_tsv", default="all_summary_stats.tsv", help="TSV with dataset metadata (n_instances)")
	parser.add_argument("--out_dir", default="plots", help="Output directory for plots")
	parser.add_argument("--metrics", nargs="*", default=None, help="Specific metrics to plot")
	parser.add_argument("--models", nargs="*", default=None, help="Specific models to plot")
	args = parser.parse_args()

	main(results_dir=args.results_dir, summary_tsv=args.summary_tsv, out_dir=args.out_dir, metrics=args.metrics, models=args.models)

