# %%
import start
from utils.data_transformation import transform_speeches, tf_idf
from utils.models import get_model
from utils.data_splitting import get_train_test

# %%
df = transform_speeches("data/us_2020_election_speeches.csv")
X_train, X_test, y_train, y_test = get_train_test(df)
vectorizer, X_train_tf_idf = tf_idf(
    corpus=X_train, ngram_range=(1, 2), stop_words="english", use_idf=True
)
# %%
model = get_model(model_name="naive-bayes")

# %%
from sklearn.model_selection import GridSearchCV

metrics = ["f1_macro", "accuracy", "recall_macro", "precision_macro"]
grid_search = GridSearchCV(
    model,
    param_grid={"alpha": [0.01, 0.1, 0.5, 1.0], "fit_prior": [True, False]},
    cv=5,
    scoring=metrics,
    refit=False,
)
# %%
grid_search.fit(X_train_tf_idf, y_train)

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

records = []
for i, params in enumerate(grid_search.cv_results_["params"]):
    for metric in metrics:
        for split in range(5):
            score = grid_search.cv_results_[f"split{split}_test_{metric}"][i]
            records.append({"Metric": metric, "Score": score, "Params": str(params)})

scores_df = pd.DataFrame(records)

plt.figure(figsize=(12, 6))
sns.violinplot(data=scores_df, x="Metric", y="Score", hue="Params", split=True)
plt.title("Distribución de métricas por modelo y fold (cross-validation)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%
