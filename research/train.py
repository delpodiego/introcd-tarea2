# %%
import start

from utils.data_transformation import transform_speeches
from utils.data_splitting import get_train_test
from utils.data_transformation import tf_idf

# %%
df = transform_speeches("data/us_2020_election_speeches.csv")
# %%
speakers = ["Donald Trump", "Joe Biden", "Mike Pence"]
df = df[df["speaker"].isin(speakers)]
df = df.groupby(df.index).agg(text=("text", " ".join), speaker=("speaker", "first"))
# %%
X_train, X_test, y_train, y_test = get_train_test(df)
# %%
import numpy as np
import matplotlib.pyplot as plt

plt.hist(y_train)
# %%
len(X_train), len(X_test)
# %%
vectorizer, X_train_tf_idf = tf_idf(
    corpus=X_train, ngram_range=(1, 2), stop_words="english", use_idf=True
)
# %%
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca.fit(X_train_tf_idf)
# %%
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(X_train_tf_idf, y_train)
# %%
y_train_pred = model.predict(X_train_tf_idf)
# %%
from sklearn.metrics import accuracy_score

accuracy_score(y_train, y_train_pred)
# %%
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred)
# %%
X_test_tf_idf = vectorizer.transform(X_test)
y_test_pred = model.predict(X_test_tf_idf)
# %%
accuracy_score(y_test, y_test_pred)
# %%
ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred)
# %%
from sklearn.metrics import recall_score, precision_score
from sklearn.model_selection import cross_val_score

model = MultinomialNB()

f1_scores = cross_val_score(model, X_train_tf_idf, y_train, cv=5, scoring="f1_macro")
f1_scores.mean()
# %%
from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(
    model,
    param_grid={"alpha": [0.1, 0.5, 1.0], "fit_prior": [True, False]},
    cv=5,
    scoring=["f1_macro", "accuracy", "recall_macro", "precision_macro"],
    refit="f1_macro",
)
grid_search.fit(X_train_tf_idf, y_train)
# %%
grid_search.best_params_
# %%
grid_search.best_score_
# %%
best_model = grid_search.best_estimator_
best_model
# %%
best_model.fit(X_train_tf_idf, y_train)
# %%
y_train_pred = best_model.predict(X_train_tf_idf)
# %%
accuracy_score(y_train, y_train_pred)
# %%
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred)
# %%
y_test_pred = best_model.predict(X_test_tf_idf)
print(accuracy_score(y_test, y_test_pred))
ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred)
# %%
