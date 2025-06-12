# Importo los módulos a utilizar
from utils.data_tranformation import transform_speeches
from utils.data_splitting import get_train_test
from research.visualizaciones_train_test import pie_chart, histogram


if __name__ == "__main__":
    # %% Se pide 1.1
    df = transform_speeches(csv_path=r"data/us_2020_election_speeches.csv")
    train, test = get_train_test(df=df)

    # %% Se pide 1.2
    pie_chart(train=train, test=test)
    histogram(train=train, test=test)
