# %%
import start
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from utils.data_splitting import get_train_test
from utils.data_transformation import transform_speeches    

# %%
df = transform_speeches("data/us_2020_election_speeches.csv")
# %%

X_train, X_test, y_train, y_test = get_train_test(df)
# %%
X_train
# %%
train_clean = clean_text(train, "text")
test_clean = clean_text(test, "text")
# %%
count_vectorizer = CountVectorizer(max_features=10000)
# %%
train_count_vectorized = count_vectorizer.fit_transform(train_clean)
# %%
train_count_vectorized.shape
# %%
count_vectorizer.get_feature_names_out(train_count_vectorized[0])
# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.hist(train_count_vectorized.data, bins=100)
plt.title('Distribution of Word Counts')
plt.xlabel('Count')
plt.ylabel('Frequency')
plt.yscale('log')
plt.grid("both")
plt.show()
# %%

# %%
count_vectorizer.transform(test["text"])
# %%
tfidf_vectorizer = TfidfVectorizer(ngram_range=(2, 5), max_features=100_000)
# %%
train_tfidf_vectorized = tfidf_vectorizer.fit_transform(train_clean)
# %%
train_tfidf_vectorized.shape
# %%
tfidf_vectorizer.get_feature_names_out(train_tfidf_vectorized[0])
# %%
plt.figure(figsize=(10, 6))
plt.hist(train_tfidf_vectorized.data, bins=100)
plt.title('Distribution of Word Counts')
plt.xlabel('Count')
plt.ylabel('Frequency')
plt.yscale('log')
plt.grid("both")
plt.show()
# %%
from sklearn.decomposition import PCA
pca = PCA(n_components=20)
pca.fit(train_tfidf_vectorized)

# %%
pca_2d = PCA(n_components=2)
train_pca_2d = pca_2d.fit_transform(train_tfidf_vectorized.toarray())

# Create scatter plot
plt.figure(figsize=(10, 6))
for speaker in train['speaker'].unique():
    mask = train['speaker'] == speaker
    plt.scatter(
        train_pca_2d[mask, 0],
        train_pca_2d[mask, 1],
        label=speaker,
        alpha=0.7
    )

plt.title('PCA of Speeches (First 2 Components)')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.legend()
plt.grid(True)
plt.show()
# %%


# %%
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=100_000, stop_words="english", use_idf=False)
train_tfidf_vectorized = tfidf_vectorizer.fit_transform(train_clean)
pca_2d = PCA(n_components=2)
train_pca_2d = pca_2d.fit_transform(train_tfidf_vectorized.toarray())

# Create scatter plot
plt.figure(figsize=(10, 6))
for speaker in train['speaker'].unique():
    mask = train['speaker'] == speaker
    plt.scatter(
        train_pca_2d[mask, 0],
        train_pca_2d[mask, 1],
        label=speaker,
        alpha=0.7
    )

plt.title('PCA of Speeches (First 2 Components)')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.legend()
plt.grid(True)
plt.show()
# %% Sobre el texto crudo
fidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), use_idf=False)
train_tfidf_vectorized = fidf_vectorizer.fit_transform(train['text'])
pca_2d = PCA(n_components=2)
train_pca_2d = pca_2d.fit_transform(train_tfidf_vectorized.toarray())

# Create scatter plot
plt.figure(figsize=(10, 6))
for speaker in train['speaker'].unique():
    mask = train['speaker'] == speaker
    plt.scatter(
        train_pca_2d[mask, 0],
        train_pca_2d[mask, 1],
        label=speaker,
        alpha=0.7
    )

plt.title('PCA of Speeches (First 2 Components)')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.legend()
plt.grid(True)
plt.show()

# %%
pca_10 = PCA(n_components=10)
pca_10.fit(train_tfidf_vectorized.toarray())

explained_var = pca_10.explained_variance_ratio_
cumulative_var = explained_var.cumsum()

plt.figure(figsize=(10, 6))
plt.bar(range(1, 11), explained_var, alpha=0.7, label='Individual Explained Variance')
plt.plot(range(1, 11), cumulative_var, marker='o', color='red', label='Cumulative Explained Variance')
plt.xlabel('Number of Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance by Principal Components')
plt.xticks(range(1, 11))
plt.legend()
plt.grid(True)
plt.show()

# %%
