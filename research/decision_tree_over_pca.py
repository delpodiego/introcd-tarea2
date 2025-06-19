# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import start

from utils.data_splitting import get_train_test
from utils.data_transformation import (
    principal_component_analysis,
    tf_idf,
    transform_speeches,
)
from utils.models import get_model

# %%
df = pd.read_csv(r"data/us_2020_election_speeches.csv")
df = transform_speeches(
    csv_path=r"data/us_2020_election_speeches.csv",
    speakers=["Joe Biden", "Donald Trump", "Mike Pence"],
    per_sentence=True,
)
df.head()
# %%
X_train, X_test, y_train, y_test = get_train_test(df=df)
# %%
vectorizer, X_train_vec = tf_idf(
    corpus=X_train, stop_words="english", use_idf=True, ngram_range=(1, 2)
)
# %%
pca, X_train_pca = principal_component_analysis(matrix=X_train_vec, n_components=10)
# %%
model = get_model(model_name="decision-tree")

# %%
from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(
    estimator=model,
    param_grid={"max_depth": [10, 25, 50, 100, None]},
    scoring=["f1_macro", "accuracy", "recall_macro", "precision_macro"],
    refit="f1_macro",
    verbose=2,
    n_jobs=-1,
)
# %%
grid_search.fit(X_train_pca, y_train)
# %%
grid_search.best_score_

# %%
