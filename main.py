# Importo los módulos a utilizar
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import typer
import yaml
from sklearn.model_selection import GridSearchCV
from typer import Typer

from research.visualizaciones_train_test import pca_line_plot, pca_plot, pie_chart
from utils.data_clean import clean_text
from utils.data_splitting import get_train_test
from utils.data_transformation import (
    bag_of_words,
    pca_explained_variance_ratio,
    principal_component_analysis,
    tf_idf,
    transform_speeches,
)
from utils.models import get_model
from utils.visualization import (
    confusion_matrix,
    report_metrics,
    violin_plot_cv_experiments,
)

app = Typer(no_args_is_help=True)


@app.command("pca")
def pca_analysis():
    # %% Se pide 1.1
    df = transform_speeches(csv_path=r"data/us_2020_election_speeches.csv")
    df = df[df["speaker"].isin(["Donald Trump", "Joe Biden", "Mike Pence"])]
    X_train, X_test, y_train, y_test = get_train_test(df=df)

    # %% Se pide 1.2
    pie_chart(y_train=y_train, y_test=y_test)

    # %% Se pide 1.3
    X_train_bow = bag_of_words(corpus=X_train)
    print(X_train_bow.shape)

    # %% Se pide 1.4
    _, X_train_tf_idf = tf_idf(corpus=X_train)

    # %% Se pide 1.5
    X_train_pca = principal_component_analysis(matrix=X_train_tf_idf, n_components=2)
    pca_plot(X_train_pca=X_train_pca, y_train=y_train, filename="pca.png")

    # TF-IDF con parámetros
    _, X_train_tf_idf = tf_idf(
        corpus=X_train, stop_words="english", use_idf=True, ngram_range=(1, 2)
    )
    X_train_pca = principal_component_analysis(matrix=X_train_tf_idf, n_components=2)
    pca_plot(X_train_pca=X_train_pca, y_train=y_train, filename="pca_con_param_1.png")

    # TF-IDF con parámetros 2
    _, X_train_tf_idf = tf_idf(corpus=X_train, stop_words="english", ngram_range=(1, 2))
    X_train_pca = principal_component_analysis(matrix=X_train_tf_idf, n_components=2)
    pca_plot(X_train_pca=X_train_pca, y_train=y_train, filename="pca_con_param_2.png")

    # Varianza a medida que agregamos los 10 componentes principales
    explained_variance_ratio = pca_explained_variance_ratio(
        matrix=X_train_tf_idf, n_components=10
    )
    cumulative_variance = explained_variance_ratio.cumsum()
    pca_line_plot(n_components=10, cumulative_variance=cumulative_variance)


@app.command()
def run_experiment(
    experiment_path: str = typer.Argument(
        ..., help="Path to the experiment file", exists=True
    ),
):
    with open(experiment_path, "r") as file:
        experiment = yaml.safe_load(file)

    if experiment["data"] == "full":
        df = pd.read_csv(r"data/us_2020_election_speeches.csv")
        df["text"] = clean_text(text=df["text"])
    else:
        df = transform_speeches(csv_path="data/us_2020_election_speeches.csv")
        if experiment["data"] == "filtered":
            df = df.groupby(df.index).agg(
                text=("text", " ".join), speaker=("speaker", "first")
            )

    df = df[df["speaker"].isin(experiment["speakers"])]

    X_train, X_test, y_train, y_test = get_train_test(df=df)
    vectorizer, X_train_tf_idf = tf_idf(
        corpus=X_train, stop_words="english", use_idf=True, ngram_range=(1, 2)
    )

    model = get_model(model_name=experiment["model"])
    grid_search = GridSearchCV(
        model,
        param_grid=experiment["grid_search"],
        cv=5,
        scoring=experiment["scoring"],
        refit=experiment["refit"],
    )
    grid_search.fit(X_train_tf_idf, y_train)
    print("Best params: ", grid_search.best_params_)

    # Violin plots
    output_path = Path("results") / f"{experiment['experiment_name']}"
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "best_params.yaml", "w") as file:
        yaml.dump(grid_search.best_params_, file)

    violin_plot_cv_experiments(grid_search=grid_search)
    plt.savefig(output_path / "violin_plot.png", bbox_inches="tight")

    model = get_model(model_name=experiment["model"], params=grid_search.best_params_)
    model.fit(X_train_tf_idf, y_train)
    X_test_tf_idf = vectorizer.transform(X_test)

    y_train_pred = model.predict(X_train_tf_idf)
    y_test_pred = model.predict(X_test_tf_idf)

    metrics = {}
    print("Train metrics:")
    metrics["train"] = report_metrics(
        y_true=y_train,
        y_pred=y_train_pred,
        verbose=True,
    )
    print("Test metrics:")
    metrics["test"] = report_metrics(
        y_true=y_test,
        y_pred=y_test_pred,
        verbose=True,
    )
    with open(output_path / "metrics.yaml", "w") as file:
        yaml.dump(metrics, file)

    confusion_matrix(
        y_true=y_train, y_pred=y_train_pred, title="Confusion matrix train"
    )
    plt.savefig(output_path / "conf_matrix_train.png")

    confusion_matrix(y_true=y_test, y_pred=y_test_pred, title="Confusion matrix test")
    plt.savefig(output_path / "conf_matrix_test.png")


if __name__ == "__main__":
    app()
