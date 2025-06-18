import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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


def violin_plot_cv_experiments(
    grid_search: GridSearchCV, *, include_scorings: list[str] | None = None, filename: str
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
    conf_matrix = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap='RdYlGn')
    plt.figure(figsize=(14, 14))
    conf_matrix.plot()
    plt.title(label=title, pad=20)
    plt.xlabel(xlabel="Predicción", labelpad=15)
    plt.ylabel(ylabel="Observación", labelpad=15)
    plt.savefig(fname=filename, bbox_inches="tight")
    plt.close()


def pie_chart(y_train: list, y_test: list):
    # %% Compare counts
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    # Colores por candidato
    speaker_colors = {
        "Donald Trump": "#e79c9c",
        "Joe Biden": "#a3b6d7",
        "Mike Pence": "#e0e0e0"
    }

    # Plot train set
    train_counts = pd.Series(y_train).value_counts()
    train_colors = [speaker_colors[speaker] for speaker in train_counts.index]
    ax1.pie(
        train_counts.values,
        labels=train_counts.index,
        colors=train_colors,
        autopct='%1.1f%%',
        startangle=90,
        counterclock=False
    )
    ax1.set_title("Set de datos de entrenamiento")

    # Plot test set
    test_counts = pd.Series(y_test).value_counts()
    test_colors = [speaker_colors[speaker] for speaker in test_counts.index]
    ax2.pie(
        test_counts.values,
        labels=test_counts.index,
        colors=test_colors,
        autopct='%1.1f%%',
        startangle=90,
        counterclock=False
    )
    ax2.set_title("Set de datos de prueba")

    fig.tight_layout()
    fig.savefig(fname='img/train_test_balance.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def pca_plot(X_train_pca, y_train, filename):
    plt.figure(figsize=(10, 7))

    speaker_colors = {
        "Donald Trump": "#e79c9c",
        "Joe Biden": "#a3b6d7",
        "Mike Pence": "#e0e0e0"
    }
    colors = [speaker_colors[speaker] for speaker in y_train]

    legend_elements = [
        Line2D(
            xdata=[0],
            ydata=[0],
            marker='o',
            color='w',
            markerfacecolor='#a3b6d7',
            label='Joe Biden',
            markersize=10
        ),
        Line2D(
            xdata=[0],
            ydata=[0],
            marker='o',
            color='w',
            markerfacecolor='#e79c9c',
            label='Donald Trump',
            markersize=10
        ),
        Line2D(
            xdata=[0],
            ydata=[0],
            marker='o',
            color='w',
            markerfacecolor='#e0e0e0',
            label='Mike Pence',
            markersize=10
        )
    ]

    plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=colors, s=80, edgecolors='k')
    plt.title(label="Proyección PCA de vectores TF-IDF sobre conjunto de entrenamiento", pad=20)
    plt.xlabel(xlabel="PCA 1", labelpad=15)
    plt.ylabel(ylabel="PCA 2", labelpad=15)
    plt.legend(handles=legend_elements)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(fname=f"img/{filename}", dpi=300, bbox_inches="tight")
    plt.close()


def pca_line_plot(n_components, cumulative_variance):
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, n_components + 1), cumulative_variance, marker='o', linestyle='-')
    plt.title(label="Varianza explicada acumulada por N° de componentes PCA", pad=20)
    plt.xlabel(xlabel="N° de componentes principales", labelpad=15)
    plt.ylabel(ylabel="Varianza explicada acumulada", labelpad=15)
    plt.xticks(range(1, n_components + 1))
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(fname='img/pca_variance.png', dpi=300, bbox_inches='tight')
    plt.close()
