import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV


def violin_plot_cv_experiments(
    grid_search: GridSearchCV, *, include_scorings: list[str] | None = None
):
    if include_scorings is None:
        include_scorings = grid_search.scoring

    records = []
    for i, params in enumerate(grid_search.cv_results_["params"]):
        for metric in include_scorings:
            for split in range(5):
                score = grid_search.cv_results_[f"split{split}_test_{metric}"][i]
                records.append(
                    {"Metric": metric, "Score": score, "Params": str(params)}
                )

    scores_df = pd.DataFrame(records)

    plt.figure(figsize=(12, 6))
    sns.violinplot(data=scores_df, x="Metric", y="Score", hue="Params", split=True)
    plt.title("Distribución de métricas por modelo y fold (cross-validation)")
    plt.xticks(rotation=45)
    plt.tight_layout()


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


def confusion_matrix(y_true: list[str], y_pred: list[str], title: str):
    conf_matrix = ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    plt.figure(figsize=(10, 10))
    conf_matrix.plot()
    plt.title(title)
