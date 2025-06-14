# Importo los módulos a utilizar
from utils.data_transformation import transform_speeches, bag_of_words, tf_idf, principal_component_analysis, pca_explained_variance_ratio
from utils.data_splitting import get_train_test
from research.visualizaciones_train_test import pie_chart, pca_plot, pca_line_plot


if __name__ == "__main__":
    # %% Se pide 1.1
    df = transform_speeches(csv_path=r"data/us_2020_election_speeches.csv")
    X_train, X_test, y_train, y_test = get_train_test(df=df)

    # %% Se pide 1.2
    pie_chart(y_train=y_train, y_test=y_test)

    # %% Se pide 1.3
    X_train_bow = bag_of_words(corpus=X_train)
    print(X_train_bow.shape)

    # %% Se pide 1.4
    X_train_tf_idf = tf_idf(corpus=X_train)

    # %% Se pide 1.5
    X_train_pca = principal_component_analysis(matrix=X_train_tf_idf, n_components=2)
    pca_plot(X_train_pca=X_train_pca, y_train=y_train, filename="pca.png")

    # TF-IDF con parámetros
    X_train_tf_idf = tf_idf(corpus=X_train, stop_words="english", use_idf=True, ngram_range=(1, 2))
    X_train_pca = principal_component_analysis(matrix=X_train_tf_idf, n_components=2)
    pca_plot(X_train_pca=X_train_pca, y_train=y_train, filename='pca_con_param_1.png')

    # TF-IDF con parámetros 2
    X_train_tf_idf = tf_idf(corpus=X_train, stop_words="english", ngram_range=(1, 2))
    X_train_pca = principal_component_analysis(matrix=X_train_tf_idf, n_components=2)
    pca_plot(X_train_pca=X_train_pca, y_train=y_train, filename='pca_con_param_2.png')

    # Varianza a medida que agregamos los 10 componentes principales
    explained_variance_ratio = pca_explained_variance_ratio(matrix=X_train_tf_idf, n_components=10)
    cumulative_variance = explained_variance_ratio.cumsum()
    pca_line_plot(n_components=10, cumulative_variance=cumulative_variance)
