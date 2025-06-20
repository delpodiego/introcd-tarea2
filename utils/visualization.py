import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV

# Letra para que coincida con LaTex
plt.rcParams.update(
    {
        "font.family": "Latin Modern Roman",
        "mathtext.fontset": "cm",
        "figure.titlesize": 18,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.title_fontsize": 16,
        "legend.fontsize": 14,
    }
)


def _extract_cv_metrics(
    grid_search: GridSearchCV, include_scorings: list[str] | None = None
) -> pd.DataFrame:
    """Extract cross-validation metrics from a GridSearchCV object into a DataFrame.

    Args:
        grid_search: The fitted GridSearchCV object containing cross-validation results
        include_scorings: List of scoring metrics to include. If None, uses all metrics from grid_search.

    Returns:
        DataFrame with columns: Metric, Score, Params
    """
    if include_scorings is None:
        include_scorings = list(grid_search.scoring)

    records = []
    for i, params in enumerate(grid_search.cv_results_["params"]):
        for metric in include_scorings:
            for split in range(5):
                score = grid_search.cv_results_[f"split{split}_test_{metric}"][i]
                records.append(
                    {"Metric": metric, "Score": score, "Params": str(params)}
                )

    return pd.DataFrame(records)


def violin_plot_cv_experiments(
    grid_search: GridSearchCV,
    filename: str,
    *,
    include_scorings: list[str] | None = None,
):
    scores_df = _extract_cv_metrics(grid_search, include_scorings)

    plt.figure(figsize=(12, 6))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel("Métrica", fontsize=14)
    plt.ylabel("Puntaje", fontsize=14)
    sns.violinplot(data=scores_df, x="Metric", y="Score", hue="Params", split=True)
    plt.title("Distribución de métricas por modelo y fold (cross-validation)")
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.legend(title="Parámetros")
    plt.tight_layout()
    plt.savefig(fname=filename, bbox_inches="tight")
    plt.close()


def strip_plot_cv_experiments(
    grid_search: GridSearchCV,
    filename: str,
    *,
    include_scorings: list[str] | None = None,
):
    scores_df = _extract_cv_metrics(grid_search, include_scorings)

    plt.figure(figsize=(12, 6))
    sns.stripplot(
        data=scores_df,
        x="Metric",
        y="Score",
        hue="Params",
        dodge=True,
        jitter=True,
        size=10,
    )
    plt.xlabel("Métrica", fontsize=14)
    plt.ylabel("Puntaje", fontsize=14)
    plt.title("Distribución de métricas por modelo y fold (cross-validation)")
    plt.legend(title="Parámetros")
    plt.grid(which="both")
    plt.tight_layout()
    plt.savefig(fname=filename, bbox_inches="tight")
    plt.close()


def _format_param_label(params: dict | str) -> str:
    """Format parameters into a readable multi-line label.

    Args:
        params: Either a dictionary of parameters or their string representation
               (e.g. {'alpha': 0.1, 'fit_prior': False} or "{'alpha': 0.1, 'fit_prior': False}")

    Returns:
        Formatted string with one parameter per line
    """
    # If input is a string, convert to dict using json
    if isinstance(params, str):
        # Replace Python bool/None with JSON equivalents
        json_str = (
            params.replace("True", "true")
            .replace("False", "false")
            .replace("None", "null")
            .replace("'", '"')
        )
        try:
            params = json.loads(json_str)
        except json.JSONDecodeError as e:
            # Fallback to original string if JSON parsing fails
            print(f"Error parsing JSON: {json_str}, {e}")
            return params

    # Format each parameter as key: value
    formatted_params = [f"{key}: {value}" for key, value in sorted(params.items())]

    # Join with newlines
    return "\n".join(formatted_params)


def bar_plot_cv_experiments(
    grid_search: GridSearchCV,
    filename: str,
    *,
    include_scorings: list[str] | None = None,
):
    """Create a grouped bar plot showing mean and standard deviation for each metric and parameter combination.

    Args:
        grid_search: The fitted GridSearchCV object containing cross-validation results
        filename: Path where to save the plot
        include_scorings: List of scoring metrics to include in the plot. If None, uses all metrics from grid_search.
    """
    scores_df = _extract_cv_metrics(grid_search, include_scorings)

    # Calculate mean and std for each parameter-metric combination
    summary_stats = (
        scores_df.groupby(["Params", "Metric"])["Score"]
        .agg(["mean", "std"])
        .reset_index()
    )

    # Set up the plot
    plt.figure(figsize=(12, 6))

    # Calculate bar positions
    n_metrics = (
        len(include_scorings)
        if include_scorings is not None
        else len(scores_df["Metric"].unique())
    )
    param_labels = sorted(scores_df["Params"].unique())
    formatted_labels = [_format_param_label(label) for label in param_labels]
    n_params = len(param_labels)
    width = 0.8 / n_metrics  # Width of each bar

    # Create bars for each metric
    metrics_to_plot = (
        include_scorings
        if include_scorings is not None
        else sorted(scores_df["Metric"].unique())
    )
    for i, metric in enumerate(metrics_to_plot):
        metric_data = summary_stats[summary_stats["Metric"] == metric]
        x = np.arange(n_params) + i * width - (n_metrics - 1) * width / 2
        means = np.array(metric_data["mean"])
        stds = np.array(metric_data["std"])
        plt.bar(x, means, width, label=metric, yerr=stds, capsize=5)

    # Customize the plot
    plt.xlabel("Parámetros", fontsize=14)
    plt.ylabel("Puntaje", fontsize=14)
    plt.title("Métricas promedio por experimento con desviación estándar")
    plt.xticks(np.arange(n_params), formatted_labels, ha="center")
    plt.ylim(0, 1.5)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend(title="Métricas")
    plt.gca().set_axisbelow(True)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(fname=filename, bbox_inches="tight")
    plt.close()


def report_metrics(
    y_true: list[str], y_pred: list[str], verbose: bool = True
) -> dict[str, float]:
    metrics: dict[str, float] = {}

    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro")
    metrics["precision_macro"] = precision_score(y_true, y_pred, average="macro")
    metrics["recall_macro"] = recall_score(y_true, y_pred, average="macro")

    if verbose:
        print(metrics)

    return metrics


def confusion_matrix(y_true: list[str], y_pred: list[str], title: str, filename: str):
    plt.figure(figsize=(14, 14))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap="Blues")
    plt.title(label=title, pad=20)
    plt.xticks(rotation=15)
    plt.xlabel(xlabel="Predicción", labelpad=15, fontsize=14)
    plt.ylabel(ylabel="Observación", labelpad=15)
    plt.savefig(fname=filename, bbox_inches="tight")
    plt.close()


def pie_chart(y_train: list, y_test: list):
    # %% Compare counts
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))

    # Colores por candidato
    speaker_colors = {
        "Donald Trump": "#e79c9c",
        "Joe Biden": "#a3b6d7",
        "Mike Pence": "#e0e0e0",
    }

    # Plot train set
    train_counts = pd.Series(y_train).value_counts()
    train_colors = [speaker_colors[speaker] for speaker in train_counts.index]
    ax1.pie(
        train_counts.values,
        labels=train_counts.index,
        colors=train_colors,
        autopct="%1.1f%%",
        startangle=90,
        counterclock=False,
    )
    ax1.set_title("Set de datos de entrenamiento")

    # Plot test set
    test_counts = pd.Series(y_test).value_counts()
    test_colors = [speaker_colors[speaker] for speaker in test_counts.index]
    ax2.pie(
        test_counts.values,
        labels=test_counts.index,
        colors=test_colors,
        autopct="%1.1f%%",
        startangle=90,
        counterclock=False,
    )
    ax2.set_title("Set de datos de prueba")

    fig.tight_layout()
    fig.savefig(fname="img/train_test_balance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def pca_plot(X_train_pca, y_train, filename):
    plt.figure(figsize=(8, 6))

    speaker_colors = {
        "Donald Trump": "#e79c9c",
        "Joe Biden": "#a3b6d7",
        "Mike Pence": "#e0e0e0",
    }
    colors = [speaker_colors[speaker] for speaker in y_train]

    legend_elements = [
        Line2D(
            xdata=[0],
            ydata=[0],
            marker="o",
            color="w",
            markerfacecolor="#a3b6d7",
            label="Joe Biden",
            markersize=10,
        ),
        Line2D(
            xdata=[0],
            ydata=[0],
            marker="o",
            color="w",
            markerfacecolor="#e79c9c",
            label="Donald Trump",
            markersize=10,
        ),
        Line2D(
            xdata=[0],
            ydata=[0],
            marker="o",
            color="w",
            markerfacecolor="#e0e0e0",
            label="Mike Pence",
            markersize=10,
        ),
    ]

    plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=colors, s=80, edgecolors="k")
    plt.title(
        label="Proyección PCA de vectores TF-IDF sobre conjunto de entrenamiento",
        pad=20,
    )
    plt.xlabel(xlabel="PC1", labelpad=15)
    plt.ylabel(ylabel="PC2", labelpad=15)
    plt.legend(handles=legend_elements)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(fname=f"img/{filename}", dpi=300, bbox_inches="tight")
    plt.close()


def pca_line_plot(n_components, cumulative_variance):
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, n_components + 1), cumulative_variance, marker="o", linestyle="-")
    plt.title(label="Varianza explicada acumulada por N° de componentes PCA", pad=20)
    plt.xlabel(xlabel="N° de componentes principales", labelpad=15)
    plt.ylabel(ylabel="Varianza explicada acumulada", labelpad=15)
    plt.xticks(range(1, n_components + 1))
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fname="img/pca_variance.png", dpi=300, bbox_inches="tight")
    plt.close()
