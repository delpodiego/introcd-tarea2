# Importo los módulos a utilizar
import pandas as pd
from sklearn.model_selection import train_test_split


def get_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # %% Cantidad de discursos por candidato
    n_speeches = df.groupby("speaker").size().sort_values(ascending=False)

    # %% Top 3 candidatos con más discursos
    top_3 = list(n_speeches.head(3).index)
    df = df[df["speaker"].isin(top_3)]

    # %% Separamos en 70% train y 30% test
    train, test = train_test_split(
        df,
        test_size=0.3,
        # Para reproducibilidad
        random_state=42,
        stratify=df["speaker"]
    )

    # %% Objetos devueltos por la función
    return train, test