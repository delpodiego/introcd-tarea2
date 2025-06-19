# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import start

from utils.data_splitting import get_train_test
from utils.data_transformation import tf_idf, transform_speeches
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
model = get_model(model_name="naive-bayes", params={"alpha": 0.1, "fit_prior": False})

# %%
X_train, X_test, y_train, y_test = get_train_test(df=df)
# %%
vectorizer, X_train_tf_idf = tf_idf(
    corpus=X_train, stop_words="english", use_idf=True, ngram_range=(1, 2)
)
# %%
model.fit(X_train_tf_idf, y_train)

# %%
log_probs = model.feature_log_prob_
classes = model.classes_
print(classes)
# %%
feature_importance_df = pd.DataFrame(
    log_probs, index=classes, columns=vectorizer.get_feature_names_out()
)

# %%
feature_importance_df.head()


# %%
def get_distinguishing_phrases_per_candidate(feature_importance_df, n_top=10):
    """
    Find the top phrases that distinguish each candidate from the rest.
    Uses the difference between the candidate's log probability and the average of others.
    """
    distinguishing_phrases = {}

    for candidate in feature_importance_df.index:
        # Get log probabilities for this candidate
        candidate_probs = feature_importance_df.loc[candidate]

        # Get log probabilities for all other candidates
        other_candidates = feature_importance_df.drop(candidate)
        other_avg_probs = other_candidates.mean(axis=0)

        # Calculate how much more likely this phrase is for the candidate vs others
        # Positive values mean the phrase is more characteristic of this candidate
        distinguishing_score = candidate_probs - other_avg_probs

        # Sort by distinguishing score (highest first)
        top_distinguishing = distinguishing_score.sort_values(ascending=False).head(
            n_top
        )

        distinguishing_phrases[candidate] = top_distinguishing

    return distinguishing_phrases


distinguishing_phrases = get_distinguishing_phrases_per_candidate(
    feature_importance_df, n_top=10
)

# Display results
for candidate, phrases in distinguishing_phrases.items():
    print(f"\n{'=' * 60}")
    print(f"TOP 10 DISTINGUISHING PHRASES FOR {candidate.upper()}")
    print(f"{'=' * 60}")
    print(f"{'Phrase':<30} {'Distinguishing Score':<20}")
    print(f"{'-' * 30} {'-' * 20}")

    for phrase, score in phrases.items():
        print(f"{phrase:<30} {score:>8.4f}")

    print()


# %%
def plot_distinguishing_phrases(distinguishing_phrases, n_top=10):
    """Create a bar plot showing the most distinguishing phrases for each candidate."""
    fig, axes = plt.subplots(1, len(distinguishing_phrases), figsize=(20, 8))

    if len(distinguishing_phrases) == 1:
        axes = [axes]

    for i, (candidate, phrases) in enumerate(distinguishing_phrases.items()):
        top_n = phrases.head(n_top)

        # Create horizontal bar plot
        axes[i].barh(range(len(top_n)), top_n.values, color="skyblue", alpha=0.7)
        axes[i].set_yticks(range(len(top_n)))
        axes[i].set_yticklabels(top_n.index, fontsize=9)
        axes[i].set_xlabel("Distinguishing Score\n(Candidate vs Others)")
        axes[i].set_title(f"Top {n_top} Distinguishing Phrases\nfor {candidate}")
        axes[i].invert_yaxis()  # Show highest values at the top

        # Add grid for better readability
        axes[i].grid(axis="x", alpha=0.3)

        # Add value labels on bars
        for j, (phrase, score) in enumerate(top_n.items()):
            axes[i].text(score + 0.01, j, f"{score:.3f}", va="center", fontsize=8)

    plt.tight_layout()
    plt.show()


plot_distinguishing_phrases(distinguishing_phrases, n_top=10)


# %%
# Alternative approach: Using log odds ratio
def get_distinguishing_phrases_log_odds(feature_importance_df, n_top=10):
    """
    Find distinguishing phrases using log odds ratio approach.
    This shows how much more likely a phrase is for one candidate vs the average of others.
    """
    distinguishing_phrases = {}

    for candidate in feature_importance_df.index:
        # Get log probabilities for this candidate
        candidate_probs = feature_importance_df.loc[candidate]

        # Get log probabilities for all other candidates
        other_candidates = feature_importance_df.drop(candidate)
        other_avg_probs = other_candidates.mean(axis=0)

        # Calculate log odds ratio: log(P(candidate) / P(others))
        # This is equivalent to: log(P(candidate)) - log(P(others))
        log_odds_ratio = candidate_probs - other_avg_probs

        # Sort by log odds ratio (highest first)
        top_distinguishing = log_odds_ratio.sort_values(ascending=False).head(n_top)

        distinguishing_phrases[candidate] = top_distinguishing

    return distinguishing_phrases


# %%
# Get results using log odds ratio approach
distinguishing_phrases_log_odds = get_distinguishing_phrases_log_odds(
    feature_importance_df, n_top=10
)

# Display results with interpretation
for candidate, phrases in distinguishing_phrases_log_odds.items():
    print(f"\n{'=' * 70}")
    print(f"TOP 10 DISTINGUISHING PHRASES FOR {candidate.upper()}")
    print(f"Log Odds Ratio: log(P({candidate}) / P(others))")
    print(f"{'=' * 70}")
    print(f"{'Phrase':<35} {'Log Odds Ratio':<15} {'Interpretation'}")
    print(f"{'-' * 35} {'-' * 15} {'-' * 20}")

    for phrase, score in phrases.items():
        # Interpret the log odds ratio
        if score > 2:
            interpretation = "Very distinctive"
        elif score > 1:
            interpretation = "Distinctive"
        elif score > 0.5:
            interpretation = "Somewhat distinctive"
        else:
            interpretation = "Slightly distinctive"

        print(f"{phrase:<35} {score:>8.4f}     {interpretation}")

    print()

# %%
# Save the distinguishing phrases to CSV
distinguishing_df = pd.DataFrame(distinguishing_phrases)
distinguishing_df.to_csv("distinguishing_phrases_per_candidate.csv")
print("Distinguishing phrases saved to 'distinguishing_phrases_per_candidate.csv'")

# %%
# Summary statistics
print("SUMMARY OF DISTINGUISHING PHRASES:")
print("=" * 50)
for candidate, phrases in distinguishing_phrases.items():
    avg_score = phrases.mean()
    max_score = phrases.max()
    min_score = phrases.min()

    print(f"\n{candidate}:")
    print(f"  Average distinguishing score: {avg_score:.4f}")
    print(f"  Most distinctive phrase: '{phrases.idxmax()}' (score: {max_score:.4f})")
    print(f"  Least distinctive phrase: '{phrases.idxmin()}' (score: {min_score:.4f})")

# %%
# Transform test data using the same vectorizer
X_test_tf_idf = vectorizer.transform(X_test)


# %%
def analyze_distinguishing_features_in_test_data(
    distinguishing_phrases, X_test_tf_idf, y_test, vectorizer
):
    """
    Analyze how often distinguishing features appear in test data and for which candidate.
    """
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()

    # Create a DataFrame to store the analysis
    analysis_results = {}

    for candidate, phrases in distinguishing_phrases.items():
        candidate_results = {}

        for phrase, distinguishing_score in phrases.items():
            # Find the index of this phrase in the feature names
            if phrase in feature_names:
                phrase_idx = list(feature_names).index(phrase)

                # Get the TF-IDF values for this phrase across all test documents
                phrase_values = X_test_tf_idf[:, phrase_idx].toarray().flatten()

                # Count how many times this phrase appears (non-zero values)
                total_occurrences = np.sum(phrase_values > 0)

                # Analyze by actual speaker
                phrase_by_speaker = {}
                for i, speaker in enumerate(y_test):
                    if phrase_values[i] > 0:  # If phrase appears in this document
                        if speaker not in phrase_by_speaker:
                            phrase_by_speaker[speaker] = 0
                        phrase_by_speaker[speaker] += 1

                # Calculate average TF-IDF value when phrase appears
                avg_tfidf = (
                    np.mean(phrase_values[phrase_values > 0])
                    if total_occurrences > 0
                    else 0
                )

                candidate_results[phrase] = {
                    "distinguishing_score": distinguishing_score,
                    "total_occurrences": total_occurrences,
                    "occurrences_by_speaker": phrase_by_speaker,
                    "avg_tfidf": avg_tfidf,
                    "feature_index": phrase_idx,
                }

        analysis_results[candidate] = candidate_results

    return analysis_results


# %%
# Analyze distinguishing features in test data
test_analysis = analyze_distinguishing_features_in_test_data(
    distinguishing_phrases, X_test_tf_idf, y_test, vectorizer
)


# %%
def create_feature_occurrence_heatmap(test_analysis, distinguishing_phrases):
    """
    Create a heatmap showing how often each distinguishing feature appears for each candidate in test data.
    """
    # Prepare data for heatmap
    all_candidates = list(distinguishing_phrases.keys())
    all_phrases = []

    for candidate, phrases in distinguishing_phrases.items():
        all_phrases.extend(phrases.index.tolist())

    # Remove duplicates while preserving order
    unique_phrases = list(dict.fromkeys(all_phrases))

    # Create matrix for heatmap
    heatmap_data = []
    phrase_labels = []

    for phrase in unique_phrases:
        row = []
        for candidate in all_candidates:
            if phrase in test_analysis[candidate]:
                # Get occurrences for this candidate specifically
                occurrences = test_analysis[candidate][phrase][
                    "occurrences_by_speaker"
                ].get(candidate, 0)
                row.append(occurrences)
            else:
                row.append(0)

        heatmap_data.append(row)
        phrase_labels.append(phrase)

    heatmap_df = pd.DataFrame(heatmap_data, index=phrase_labels, columns=all_candidates)

    # Create the heatmap
    plt.figure(figsize=(12, max(8, len(unique_phrases) * 0.4)))
    sns.heatmap(
        heatmap_df,
        annot=True,
        cmap="Blues",
        fmt="d",
        cbar_kws={"label": "Number of Occurrences"},
    )
    plt.title(
        "Occurrence of Distinguishing Features in Test Data\n(Actual Speaker vs Feature Owner)"
    )
    plt.xlabel("Actual Speaker in Test Data")
    plt.ylabel("Distinguishing Features")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    return heatmap_df


# %%
# Create the occurrence heatmap
occurrence_heatmap = create_feature_occurrence_heatmap(
    test_analysis, distinguishing_phrases
)


# %%
def create_feature_effectiveness_plot(test_analysis, distinguishing_phrases):
    """
    Create a plot showing the effectiveness of distinguishing features.
    """
    effectiveness_data = []

    for candidate, phrases in distinguishing_phrases.items():
        for phrase, distinguishing_score in phrases.items():
            if phrase in test_analysis[candidate]:
                analysis = test_analysis[candidate][phrase]

                # Calculate effectiveness: how often the feature appears for the correct candidate
                correct_occurrences = analysis["occurrences_by_speaker"].get(
                    candidate, 0
                )
                total_occurrences = analysis["total_occurrences"]
                effectiveness = (
                    correct_occurrences / total_occurrences
                    if total_occurrences > 0
                    else 0
                )

                effectiveness_data.append(
                    {
                        "candidate": candidate,
                        "phrase": phrase,
                        "distinguishing_score": distinguishing_score,
                        "total_occurrences": total_occurrences,
                        "correct_occurrences": correct_occurrences,
                        "effectiveness": effectiveness,
                    }
                )

    effectiveness_df = pd.DataFrame(effectiveness_data)

    # Create scatter plot
    plt.figure(figsize=(12, 8))

    colors = ["red", "blue", "green"]
    for i, candidate in enumerate(distinguishing_phrases.keys()):
        candidate_data = effectiveness_df[effectiveness_df["candidate"] == candidate]

        plt.scatter(
            candidate_data["distinguishing_score"],
            candidate_data["effectiveness"],
            s=candidate_data["total_occurrences"] * 10,  # Size based on occurrences
            alpha=0.7,
            color=colors[i],
            label=candidate,
        )

        # Add text labels for top features
        top_features = candidate_data.nlargest(3, "distinguishing_score")
        for _, row in top_features.iterrows():
            plt.annotate(
                row["phrase"],
                (row["distinguishing_score"], row["effectiveness"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

    plt.xlabel("Distinguishing Score")
    plt.ylabel("Effectiveness (Correct Occurrences / Total Occurrences)")
    plt.title("Feature Effectiveness: Distinguishing Score vs Test Performance")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return effectiveness_df


# %%
# Create effectiveness plot
effectiveness_df = create_feature_effectiveness_plot(
    test_analysis, distinguishing_phrases
)


# %%
def create_feature_usage_comparison(test_analysis, distinguishing_phrases):
    """
    Create a comparison plot showing feature usage patterns.
    """
    # Prepare data for comparison
    comparison_data = []

    for candidate, phrases in distinguishing_phrases.items():
        for phrase, distinguishing_score in phrases.items():
            if phrase in test_analysis[candidate]:
                analysis = test_analysis[candidate][phrase]

                for speaker, occurrences in analysis["occurrences_by_speaker"].items():
                    comparison_data.append(
                        {
                            "feature_owner": candidate,
                            "actual_speaker": speaker,
                            "phrase": phrase,
                            "occurrences": occurrences,
                            "distinguishing_score": distinguishing_score,
                        }
                    )

    comparison_df = pd.DataFrame(comparison_data)

    # Create grouped bar plot
    plt.figure(figsize=(15, 10))

    # Group by feature owner and actual speaker
    pivot_data = comparison_df.pivot_table(
        values="occurrences",
        index="feature_owner",
        columns="actual_speaker",
        aggfunc="sum",
    ).fillna(0)

    # Create stacked bar plot
    ax = pivot_data.plot(kind="bar", stacked=True, figsize=(12, 8))
    plt.title("Usage of Distinguishing Features by Actual Speakers in Test Data")
    plt.xlabel("Feature Owner (Candidate)")
    plt.ylabel("Total Occurrences")
    plt.legend(title="Actual Speaker")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return comparison_df


# %%
# Create usage comparison
usage_comparison = create_feature_usage_comparison(
    test_analysis, distinguishing_phrases
)


# %%
def create_detailed_feature_analysis(test_analysis, distinguishing_phrases):
    """
    Create a detailed analysis table and visualization.
    """
    detailed_results = []

    for candidate, phrases in distinguishing_phrases.items():
        for phrase, distinguishing_score in phrases.items():
            if phrase in test_analysis[candidate]:
                analysis = test_analysis[candidate][phrase]

                # Calculate metrics
                correct_occurrences = analysis["occurrences_by_speaker"].get(
                    candidate, 0
                )
                total_occurrences = analysis["total_occurrences"]
                precision = (
                    correct_occurrences / total_occurrences
                    if total_occurrences > 0
                    else 0
                )

                # Get all speakers who used this phrase
                speakers_used = list(analysis["occurrences_by_speaker"].keys())

                detailed_results.append(
                    {
                        "Candidate": candidate,
                        "Phrase": phrase,
                        "Distinguishing_Score": distinguishing_score,
                        "Total_Occurrences": total_occurrences,
                        "Correct_Occurrences": correct_occurrences,
                        "Precision": precision,
                        "Speakers_Used": ", ".join(speakers_used),
                        "Avg_TFIDF": analysis["avg_tfidf"],
                    }
                )

    detailed_df = pd.DataFrame(detailed_results)

    # Display summary statistics
    print("DETAILED FEATURE ANALYSIS SUMMARY:")
    print("=" * 60)

    for candidate in distinguishing_phrases.keys():
        candidate_data = detailed_df[detailed_df["Candidate"] == candidate]
        print(f"\n{candidate}:")
        print(f"  Average precision: {candidate_data['Precision'].mean():.3f}")
        print(
            f"  Total feature occurrences: {candidate_data['Total_Occurrences'].sum()}"
        )
        print(
            f"  Most effective feature: {candidate_data.loc[candidate_data['Precision'].idxmax(), 'Phrase']}"
        )
        print(
            f"  Least effective feature: {candidate_data.loc[candidate_data['Precision'].idxmin(), 'Phrase']}"
        )

    return detailed_df


# %%
# Create detailed analysis
detailed_analysis = create_detailed_feature_analysis(
    test_analysis, distinguishing_phrases
)

# %%
