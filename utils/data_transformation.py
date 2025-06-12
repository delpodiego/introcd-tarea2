# Importo los módulos a utilizar
import pandas as pd
import sys
from utils.data_clean import list_of_tuples


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

    # %% Objeto devuelto por la función
    return df
