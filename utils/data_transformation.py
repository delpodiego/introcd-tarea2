# Importo los módulos a utilizar
import pandas as pd
import sys
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA
from utils.data_clean import clean_text, list_of_tuples


def transform_speeches(csv_path: str) -> pd.DataFrame:
    # %% Leemos el CSV
    df = pd.read_csv(
        filepath_or_buffer=csv_path,
        sep=","
    )

    # %% Convierto tipo de dato de "date"
    df["date"] = pd.to_datetime(df["date"], format="%b %d, %Y")

    # %% Elimino lo que está entre [] como [crosstalk...], [inaudible...], etc.
    df["text"] = df["text"].str.replace(r"\[.*?\]", "", regex=True)

    # %% Este comercial en particular me rompe el index 79 donde quiero separar las intervenciones
    df["text"] = df["text"].str.replace("Commercial: (48:14)\r\n", "")

    # %% Elimino el patrón ': (mm:ss)'
    df["text"] = df["text"].str.replace(r": \(\d{2}:\d{2}\)", "", regex=True)

    # %% Elimino el patrón ': (hh:mm:ss)'
    df["text"] = df["text"].str.replace(r": \(\d{2}:\d{2}\:\d{2}\)", "", regex=True)

    # %% Convierto la columna en una lista donde cada elemento es una intervención de un orador
    # Hay que aplicar un regex si es Mac y otro si es Windows
    if sys.platform == "win32":
        df["text"] = df["text"].str.split(r"\n(?:\xa0\n)?", regex=True)
    else:
        df["text"] = df["text"].str.split(r"[(\r\n)\n](?:\xa0[(\r\n)\n])?", regex=True)

    # %% Convierto la lista en una lista de tuplas donde cada tupla tiene el par orador-discurso
    df["text"] = df["text"].apply(list_of_tuples)

    # %% Con explode hago que cada elemento de la lista (cada tupla) sea una fila
    df = df.explode("text")

    # %% Sobreescribo los valores de speaker y text con los valores de cada tupla
    df[["speaker", "text"]] = pd.DataFrame(df["text"].tolist(), index=df.index)
    # df.to_csv(path_or_buf='data/speeches.csv', index=False)

    # %% Homogeneizar nombres
    names = {
        "President Trump": "Donald Trump",
        "President Donald J. Trump": "Donald Trump",
        "President Donald Trump": "Donald Trump",
        "Donald J. Trump": "Donald Trump",
        "Trump": "Donald Trump",
        "Vice President Joe Biden": "Joe Biden",
        "VIce President Biden": "Joe Biden",
        "Joe Biden ": "Joe Biden",
        "Vice President Mike Pence": "Mike Pence",
        "Vice President Mike Pence ": "Mike Pence",
        "Kamala Harris ": "Kamala Harris",
        "Senator Kamala Harris": "Kamala Harris",
        "Senator Harris": "Kamala Harris",
        "Senator Bernie Sanders": "Bernie Sanders",
        "Sanders": "Bernie Sanders",
    }
    df["speaker"] = (df["speaker"].map(names).fillna(df["speaker"]))

    # %% Limpio los discursos
    df["text"] = clean_text(text=df["text"])

    # %% Objeto devuelto por la función
    return df[["speaker", "text"]]


def bag_of_words(corpus: list[str] | pd.Series) -> csr_matrix:
    # %% Inicio el vectorizador
    vectorizer = CountVectorizer()

    # %% Ajustar y transformar el corpus
    matrix = vectorizer.fit_transform(corpus)

    # %% Objeto devuelto por la función
    return matrix


def tf_idf(corpus: list[str] | pd.Series, stop_words: str = None, use_idf: bool = False, ngram_range: tuple = (1, 1)) -> csr_matrix:
    # %% Inicio el vectorizador
    vectorizer = TfidfVectorizer(stop_words=stop_words, use_idf=use_idf, ngram_range=ngram_range)

    # %% Ajustar y transformar el corpus
    matrix = vectorizer.fit_transform(corpus)

    # %% Objeto devuelto por la función
    return matrix


def principal_component_analysis(matrix: csr_matrix, n_components: int = 2):
    # %% Aplicamos PCA a la matriz
    return PCA(n_components=n_components).fit_transform(matrix.toarray())

def pca_explained_variance_ratio(matrix: csr_matrix, n_components: int = 2):
    # %% Aplicamos PCA a la matriz
    pca = PCA(n_components=n_components)
    pca.fit(matrix.toarray())

    # %% Objeto devuelto por la función
    return pca.explained_variance_ratio_

