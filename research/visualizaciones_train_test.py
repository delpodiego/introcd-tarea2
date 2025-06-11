# %%
import start
import matplotlib.pyplot as plt

from utils.data_splitting import get_train_test

# %%
train, test = get_train_test(csv_path=r"data/us_2020_election_speeches.csv")

# %% Compare counts
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Plot train set
train_counts = train["speaker"].value_counts()
ax1.bar(train_counts.index, train_counts.values)
ax1.set_title("Train Set")
ax1.set_ylabel("Number of Speeches")
ax1.tick_params(axis="x")

# Plot test set
test_counts = test["speaker"].value_counts()
ax2.bar(test_counts.index, test_counts.values)
ax2.set_title("Test Set")
ax2.set_ylabel("Number of Speeches")
ax2.tick_params(axis="x")

plt.tight_layout()
plt.show()
# %%
from utils.clean_data import clean_text

train_clean = clean_text(train, "text")
test_clean = clean_text(test, "text")

# %%
# Compare speech lengths between train and test sets
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Plot train set speech lengths
train_lengths = train_clean.str.len()
ax1.hist(train_lengths, bins=30)
ax1.set_title("Train Set Speech Lengths")
ax1.set_xlabel("Number of Characters")
ax1.set_ylabel("Frequency")

# Plot test set speech lengths
test_lengths = test_clean.str.len()
ax2.hist(test_lengths, bins=30)
ax2.set_title("Test Set Speech Lengths")
ax2.set_xlabel("Number of Characters")
ax2.set_ylabel("Frequency")

plt.tight_layout()
plt.show()

# Print summary statistics
print("\nTrain set speech length statistics:")
print(train_lengths.describe())
print("\nTest set speech length statistics:")
print(test_lengths.describe())
# %%
# Create a figure with subplots for each speaker
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Get unique speakers
speakers = train["speaker"].unique()

# Plot speech lengths for each speaker in train set
for i, speaker in enumerate(speakers):
    speaker_train = train_clean[train["speaker"] == speaker]
    speaker_test = test_clean[test["speaker"] == speaker]
    
    # Plot histograms
    axes[i].hist(speaker_train.str.len(), bins=30, alpha=0.6, label='Train')
    axes[i].hist(speaker_test.str.len(), bins=30, alpha=0.6, label='Test')
    
    axes[i].set_title(f"{speaker} Speech Lengths")
    axes[i].set_xlabel("Number of Characters")
    axes[i].set_ylabel("Frequency")
    axes[i].legend()

plt.tight_layout()
plt.show()

# Print summary statistics for each speaker
for speaker in speakers:
    print(f"\n{speaker} speech length statistics:")
    print("\nTrain set:")
    print(train_clean[train["speaker"] == speaker].str.len().describe())
    print("\nTest set:")
    print(test_clean[test["speaker"] == speaker].str.len().describe())
