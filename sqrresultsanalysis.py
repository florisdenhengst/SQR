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

	# group by metric -> tau -> model
	for metric, metric_df in df.groupby("metric"):
		for tau, tau_df in metric_df.groupby("tau"):
			for model, model_df in tau_df.groupby("model"):
				if model_df["n_instances"].isnull().all():
					continue
				# sort by instances
				model_df = model_df.sort_values("n_instances")

				plt.figure(figsize=(8, 5))
				if kind == "line":
					sns.lineplot(x="n_instances", y="value", data=model_df, marker="o")
				else:
					sns.scatterplot(x="n_instances", y="value", data=model_df)

				plt.xlabel("Number of Instances")
				plt.ylabel(metric)
				metric_label = METRIC_LABELS.get(metric, metric)
				plt.title(f"{model} — {metric_label} — tau={tau}")
				if logx:
					plt.xscale("log")

				safe_tau = str(tau).replace(".", "_") if tau is not None else "none"
				out_subdir = os.path.join(out_dir, metric, "by_instances", "individual")
				os.makedirs(out_subdir, exist_ok=True)
				out_path = os.path.join(out_subdir, f"{model}_tau_{safe_tau}.png")
				plt.tight_layout()
				plt.savefig(out_path, dpi=150)
				plt.close()


def plot_all_models_together(df, out_dir, metric, tau=None, logx=True):
	# Create a single plot with one line per model for a given metric and tau
	sel = df[df["metric"] == metric]
	if tau is not None:
		sel = sel[sel["tau"] == tau]
	if sel.empty:
		return None

	plt.figure(figsize=(10, 6))
	sns.set(style="whitegrid")
	for model, model_df in sel.groupby("model"):
		model_df = model_df.sort_values("n_instances")
		plt.plot(model_df["n_instances"], model_df["value"], marker="o", label=model)

	plt.xlabel("Number of Instances")
	metric_label = METRIC_LABELS.get(metric, metric)
	plt.ylabel(metric_label)
	title_tau = f" tau={tau}" if tau is not None else ""
	plt.title(f"Model Comparison — {metric_label}{title_tau}")
	if logx:
		plt.xscale("log")
	plt.legend()
	plt.tight_layout()
	safe_tau = str(tau).replace(".", "_") if tau is not None else "none"
	out_subdir = os.path.join(out_dir, metric, "by_instances", "comparison")
	os.makedirs(out_subdir, exist_ok=True)
	out_path = os.path.join(out_subdir, f"all_models_tau_{safe_tau}.png")
	plt.savefig(out_path, dpi=150)
	plt.close()
	return out_path


def plot_metric_by_features(df, out_dir, logx=True, kind="line"):
	sns.set(style="whitegrid")

	# group by metric -> tau -> model
	for metric, metric_df in df.groupby("metric"):
		for tau, tau_df in metric_df.groupby("tau"):
			for model, model_df in tau_df.groupby("model"):
				if model_df["n_features"].isnull().all():
					continue
				# sort by features
				model_df = model_df.sort_values("n_features")

				plt.figure(figsize=(8, 5))
				if kind == "line":
					sns.lineplot(x="n_features", y="value", data=model_df, marker="o")
				else:
					sns.scatterplot(x="n_features", y="value", data=model_df)

				plt.xlabel("Number of Features")
				metric_label = METRIC_LABELS.get(metric, metric)
				plt.ylabel(metric_label)
				plt.title(f"{model} — {metric_label} — tau={tau}")
				if logx:
					plt.xscale("log")

				safe_tau = str(tau).replace(".", "_") if tau is not None else "none"
				out_subdir = os.path.join(out_dir, metric, "by_features", "individual")
				os.makedirs(out_subdir, exist_ok=True)
				out_path = os.path.join(out_subdir, f"{model}_tau_{safe_tau}.png")
				plt.tight_layout()
				plt.savefig(out_path, dpi=150)
				plt.close()


def plot_all_models_together_features(df, out_dir, metric, tau=None, logx=True):
	sel = df[df["metric"] == metric]
	if tau is not None:
		sel = sel[sel["tau"] == tau]
	if sel.empty:
		return None

	plt.figure(figsize=(10, 6))
	sns.set(style="whitegrid")
	for model, model_df in sel.groupby("model"):
		model_df = model_df.sort_values("n_features")
		plt.plot(model_df["n_features"], model_df["value"], marker="o", label=model)

	plt.xlabel("Number of Features")
	metric_label = METRIC_LABELS.get(metric, metric)
	plt.ylabel(metric_label)
	title_tau = f" tau={tau}" if tau is not None else ""
	plt.title(f"Model Comparison — {metric_label}{title_tau}")
	if logx:
		plt.xscale("log")
	plt.legend()
	plt.tight_layout()
	safe_tau = str(tau).replace(".", "_") if tau is not None else "none"
	out_subdir = os.path.join(out_dir, metric, "by_features", "comparison")
	os.makedirs(out_subdir, exist_ok=True)
	out_path = os.path.join(out_subdir, f"all_models_tau_{safe_tau}.png")
	plt.savefig(out_path, dpi=150)
	plt.close()
	return out_path


def plot_metric_by_categorical_features(df, out_dir, logx=False, kind="line"):
	sns.set(style="whitegrid")

	# group by metric -> tau -> model
	for metric, metric_df in df.groupby("metric"):
		for tau, tau_df in metric_df.groupby("tau"):
			for model, model_df in tau_df.groupby("model"):
				if model_df["n_categorical_features"].isnull().all():
					continue
				# sort by categorical features
				model_df = model_df.sort_values("n_categorical_features")

				plt.figure(figsize=(8, 5))
				if kind == "line":
					sns.lineplot(x="n_categorical_features", y="value", data=model_df, marker="o")
				else:
					sns.scatterplot(x="n_categorical_features", y="value", data=model_df)

				plt.xlabel("Number of Categorical Features")
				metric_label = METRIC_LABELS.get(metric, metric)
				plt.ylabel(metric_label)
				plt.title(f"{model} — {metric_label} — tau={tau}")
				if logx:
					plt.xscale("log")

				safe_tau = str(tau).replace(".", "_") if tau is not None else "none"
				out_subdir = os.path.join(out_dir, metric, "by_categorical_features", "individual")
				os.makedirs(out_subdir, exist_ok=True)
				out_path = os.path.join(out_subdir, f"{model}_tau_{safe_tau}.png")
				plt.tight_layout()
				plt.savefig(out_path, dpi=150)
				plt.close()



def plot_all_models_together_categorical_features(df, out_dir, metric, tau=None, logx=False):
	sel = df[df["metric"] == metric]
	if tau is not None:
		sel = sel[sel["tau"] == tau]
	if sel.empty:
		return None

	plt.figure(figsize=(10, 6))
	sns.set(style="whitegrid")
	for model, model_df in sel.groupby("model"):
		model_df = model_df.sort_values("n_categorical_features")
		plt.plot(model_df["n_categorical_features"], model_df["value"], marker="o", label=model)

	plt.xlabel("Number of Categorical Features")
	metric_label = METRIC_LABELS.get(metric, metric)
	plt.ylabel(metric_label)
	title_tau = f" tau={tau}" if tau is not None else ""
	plt.title(f"Model Comparison — {metric_label}{title_tau}")
	if logx:
		plt.xscale("log")
	plt.legend()
	plt.tight_layout()
	safe_tau = str(tau).replace(".", "_") if tau is not None else "none"
	out_subdir = os.path.join(out_dir, metric, "by_categorical_features", "comparison")
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

