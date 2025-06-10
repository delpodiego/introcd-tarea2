# Importo los módulos a utilizar
from utils.data_splitting import get_train_test


if __name__ == "__main__":
    # %% Se pide 1
    train, test = get_train_test(csv_path=r"data/us_2020_election_speeches.csv")