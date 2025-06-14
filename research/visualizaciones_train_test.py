# %%
# import start
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
# from utils.data_clean import clean_text


def pie_chart(y_train: list, y_test: list):
    # %% Compare counts
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    # Colores por candidato
    speaker_colors = {
        "Donald Trump": "#e79c9c",
        "Joe Biden": "#a3b6d7",
        "Mike Pence": "#e0e0e0"
    }

    # Plot train set
    train_counts = pd.Series(y_train).value_counts()
    train_colors = [speaker_colors[speaker] for speaker in train_counts.index]
    ax1.pie(
        train_counts.values,
        labels=train_counts.index,
        colors=train_colors,
        autopct='%1.1f%%',
        startangle=90,
        counterclock=False
    )
    ax1.set_title("Set de datos de entrenamiento")

    # Plot test set
    test_counts = pd.Series(y_test).value_counts()
    test_colors = [speaker_colors[speaker] for speaker in test_counts.index]
    ax2.pie(
        test_counts.values,
        labels=test_counts.index,
        colors=test_colors,
        autopct='%1.1f%%',
        startangle=90,
        counterclock=False
    )
    ax2.set_title("Set de datos de prueba")

    fig.tight_layout()
    fig.savefig(fname='img/piechart.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

# %%
# train_clean = clean_text(train, column_name="text")
# test_clean = clean_text(test, column_name="text")
#
# # Compare speech lengths between train and test sets
# fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
#
# # Plot train set speech lengths
# train_lengths = train_clean.str.len()
# ax1.hist(train_lengths, bins=30)
# ax1.set_title("Train Set Speech Lengths")
# ax1.set_xlabel("Number of Characters")
# ax1.set_ylabel("Frequency")
#
# # Plot test set speech lengths
# test_lengths = test_clean.str.len()
# ax2.hist(test_lengths, bins=30)
# ax2.set_title("Test Set Speech Lengths")
# ax2.set_xlabel("Number of Characters")
# ax2.set_ylabel("Frequency")
#
# fig.tight_layout()
# fig.savefig(fname='img/hist.png', dpi=300, bbox_inches='tight')
# plt.close(fig)
#
# # %% Print summary statistics
# print("\nTrain set speech length statistics:")
# print(train_lengths.describe())
# print("\nTest set speech length statistics:")
# print(test_lengths.describe())

# %% Create a figure with subplots for each speaker
# def histogram(train: pd.DataFrame, test: pd.DataFrame):
#     train_clean = clean_text(train, column_name="text")
#     test_clean = clean_text(test, column_name="text")
#     fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
#
#     # Get unique speakers
#     speakers = train["speaker"].unique()
#
#     # Plot speech lengths for each speaker in train set
#     for i, speaker in enumerate(speakers):
#         speaker_train = train_clean[train["speaker"] == speaker]
#         speaker_test = test_clean[test["speaker"] == speaker]
#
#         # Plot histograms
#         axes[i].hist(speaker_train.str.len(), bins=30, alpha=0.6, label='Train')
#         axes[i].hist(speaker_test.str.len(), bins=30, alpha=0.6, label='Test')
#
#         axes[i].set_title(f"{speaker} Speech Lengths")
#         axes[i].set_xlabel("Number of Characters")
#         axes[i].set_ylabel("Frequency")
#         axes[i].legend()
#
#     fig.tight_layout()
#     fig.savefig(fname='img/histogram.png', dpi=300, bbox_inches='tight')
#     plt.close(fig)

# %% Print summary statistics for each speaker
# for speaker in speakers:
#     print(f"\n{speaker} speech length statistics:")
#     print("\nTrain set:")
#     print(train_clean[train["speaker"] == speaker].str.len().describe())
#     print("\nTest set:")
#     print(test_clean[test["speaker"] == speaker].str.len().describe())


def pca_plot(X_train_pca, y_train, filename):
    plt.figure(figsize=(10, 7))

    speaker_colors = {
        "Donald Trump": "#e79c9c",
        "Joe Biden": "#a3b6d7",
        "Mike Pence": "#e0e0e0"
    }
    colors = [speaker_colors[speaker] for speaker in y_train]

    legend_elements = [
        Line2D(
            xdata=[0],
            ydata=[0],
            marker='o',
            color='w',
            markerfacecolor='#a3b6d7',
            label='Joe Biden',
            markersize=10
        ),
        Line2D(
            xdata=[0],
            ydata=[0],
            marker='o',
            color='w',
            markerfacecolor='#e79c9c',
            label='Donald Trump',
            markersize=10
        ),
        Line2D(
            xdata=[0],
            ydata=[0],
            marker='o',
            color='w',
            markerfacecolor='#e0e0e0',
            label='Mike Pence',
            markersize=10
        )
    ]

    plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=colors, s=80, edgecolors='k')
    plt.title("Proyección PCA de vectores TF-IDF sobre conjunto de entrenamiento")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend(handles=legend_elements)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.savefig(fname=f"img/{filename}", dpi=300, bbox_inches="tight")
    plt.close()


def pca_line_plot(n_components, cumulative_variance):
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, n_components + 1), cumulative_variance, marker='o', linestyle='-')
    plt.title("Varianza acumulada por número de componentes PCA")
    plt.xlabel("Número de componentes principales")
    plt.ylabel("Varianza acumulada")
    plt.xticks(range(1, n_components + 1))
    plt.ylim(0, 1.05)
    plt.savefig(fname='img/pca_variance.png', dpi=300, bbox_inches='tight')
    plt.close()
