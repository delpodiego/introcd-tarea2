# Importo los módulos a utilizar
import pandas as pd
from sklearn.model_selection import train_test_split


def get_train_test(df: pd.DataFrame) -> tuple[list, list, list, list]:
    y = list(df["speaker"])
    X = list(df["text"])

    # %% Separamos en 70% train y 30% test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        # Para reproducibilidad
        random_state=103,
        stratify=y,
    )

    # %% Objetos devueltos por la función
    return X_train, X_test, y_train, y_test
