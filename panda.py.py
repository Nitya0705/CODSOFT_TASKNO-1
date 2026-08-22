"""
Task 1 - Movie Genre Classification
Prodigy InfoTech Internship

Predicts a movie's genre from its plot summary using TF-IDF text
features and compares three classifiers: Multinomial Naive Bayes,
Logistic Regression, and Linear SVM.

Dataset: hijest/genre-classification-dataset-imdb (Kaggle)
Format (::: delimited):
  train_data.txt : ID ::: TITLE ::: GENRE ::: DESCRIPTION
  test_data.txt  : ID ::: TITLE ::: DESCRIPTION
  test_data_solution.txt : ID ::: TITLE ::: GENRE ::: DESCRIPTION
"""

import os
import re
import string

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# --------------------------------------------------------------------------
# 1. Download the dataset from Kaggle
# --------------------------------------------------------------------------
import kagglehub

DATA_DIR = kagglehub.dataset_download("hijest/genre-classification-dataset-imdb")
print("Path to dataset files:", DATA_DIR)

# The dataset ships its .txt files inside a subfolder whose exact name can
# vary between versions, so search for them instead of hardcoding a path.
def find_file(root, filename):
    for dirpath, _, filenames in os.walk(root):
        if filename in filenames:
            return os.path.join(dirpath, filename)
    raise FileNotFoundError(f"Could not find {filename} under {root}")

TRAIN_PATH = find_file(DATA_DIR, "train_data.txt")
TEST_PATH = find_file(DATA_DIR, "test_data.txt")
TEST_SOLUTION_PATH = find_file(DATA_DIR, "test_data_solution.txt")

# --------------------------------------------------------------------------
# 2. Load the ::: delimited text files into DataFrames
# --------------------------------------------------------------------------
def load_labeled(path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split(" ::: ")
            if len(parts) == 4:
                rows.append(parts)
    return pd.DataFrame(rows, columns=["id", "title", "genre", "description"])


def load_unlabeled(path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split(" ::: ")
            if len(parts) == 3:
                rows.append(parts)
    return pd.DataFrame(rows, columns=["id", "title", "description"])


train_df = load_labeled(TRAIN_PATH)
test_df = load_unlabeled(TEST_PATH)
test_solution_df = load_labeled(TEST_SOLUTION_PATH)

train_df["genre"] = train_df["genre"].str.strip().str.lower()
test_solution_df["genre"] = test_solution_df["genre"].str.strip().str.lower()

print(f"Train rows: {len(train_df)} | Test rows: {len(test_df)}")
print(train_df["genre"].value_counts())

# --------------------------------------------------------------------------
# 3. Clean the plot summaries
# --------------------------------------------------------------------------
try:
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    STOPWORDS = set(stopwords.words("english"))

_punct_table = str.maketrans("", "", string.punctuation)


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)      # urls
    text = re.sub(r"\d+", " ", text)                  # digits
    text = text.translate(_punct_table)               # punctuation
    tokens = [w for w in text.split() if w not in STOPWORDS and len(w) > 2]
    return " ".join(tokens)


train_df["clean_description"] = train_df["description"].apply(clean_text)
test_df["clean_description"] = test_df["description"].apply(clean_text)
test_solution_df["clean_description"] = test_solution_df["description"].apply(clean_text)

# --------------------------------------------------------------------------
# 4. Split the (labeled) training data into train / validation
# --------------------------------------------------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    train_df["clean_description"],
    train_df["genre"],
    test_size=0.2,
    random_state=42,
    stratify=train_df["genre"],
)

# --------------------------------------------------------------------------
# 5. TF-IDF vectorization
# --------------------------------------------------------------------------
vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=2)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)

# --------------------------------------------------------------------------
# 6. Train and compare classifiers
# --------------------------------------------------------------------------
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Linear SVM": LinearSVC(),
}

results = {}
for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    preds = model.predict(X_val_tfidf)
    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds, average="weighted")
    results[name] = {"model": model, "accuracy": acc, "f1": f1, "val_preds": preds}
    print(f"\n{name}  ->  accuracy: {acc:.4f} | weighted F1: {f1:.4f}")
    print(classification_report(y_val, preds, zero_division=0))

best_name = max(results, key=lambda k: results[k]["f1"])
best_model = results[best_name]["model"]
print(f"\nBest model on validation split: {best_name}")

# --------------------------------------------------------------------------
# 6a. Model comparison table + bar chart
# --------------------------------------------------------------------------
comparison_df = pd.DataFrame(
    {name: {"Accuracy": res["accuracy"], "Weighted F1": res["f1"]} for name, res in results.items()}
).T.sort_values("Weighted F1", ascending=False)

print("\nModel comparison (validation set):")
print(comparison_df.round(4))

fig, ax = plt.subplots(figsize=(8, 5))
comparison_df.plot(kind="bar", ax=ax, rot=0, color=["#4C72B0", "#DD8452"])
ax.set_title("Model Comparison - Validation Accuracy & Weighted F1")
ax.set_ylabel("Score")
ax.set_ylim(0, 1.12)
ax.legend(loc="lower right")
for container in ax.containers:
    ax.bar_label(container, fmt="%.3f", padding=2, fontsize=8)
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150)
plt.show()

# --------------------------------------------------------------------------
# 6b. Confusion matrix for every model (validation set)
# --------------------------------------------------------------------------
labels = sorted(train_df["genre"].unique())
annotate = len(labels) <= 15  # keep cells readable; too many genres gets cluttered

fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 6))
if len(models) == 1:
    axes = [axes]

for ax, (name, res) in zip(axes, results.items()):
    cm = confusion_matrix(y_val, res["val_preds"], labels=labels)
    sns.heatmap(
        cm, annot=annotate, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax, cbar=False,
    )
    ax.set_title(f"{name}\n(accuracy: {res['accuracy']:.3f})")
    ax.set_xlabel("Predicted genre")
    ax.set_ylabel("Actual genre")
    ax.tick_params(axis="x", rotation=90)
    ax.tick_params(axis="y", rotation=0)

plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150)
plt.show()

# --------------------------------------------------------------------------
# 7. Final check against the official labeled test set, then predict
# --------------------------------------------------------------------------
X_test_solution_tfidf = vectorizer.transform(test_solution_df["clean_description"])
final_preds = best_model.predict(X_test_solution_tfidf)
final_acc = accuracy_score(test_solution_df["genre"], final_preds)
print(f"\n{best_name} accuracy on official test_data_solution.txt: {final_acc:.4f}")

test_labels = sorted(test_solution_df["genre"].unique())
cm_final = confusion_matrix(test_solution_df["genre"], final_preds, labels=test_labels)
plt.figure(figsize=(9, 8))
sns.heatmap(
    cm_final, annot=len(test_labels) <= 15, fmt="d", cmap="Blues",
    xticklabels=test_labels, yticklabels=test_labels, cbar=True,
)
plt.title(f"{best_name} - Confusion Matrix on Official Test Set (accuracy: {final_acc:.3f})")
plt.xlabel("Predicted genre")
plt.ylabel("Actual genre")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("confusion_matrix_final_test.png", dpi=150)
plt.show()

X_test_tfidf = vectorizer.transform(test_df["clean_description"])
test_df["predicted_genre"] = best_model.predict(X_test_tfidf)

output_path = "test_predictions.csv"
test_df[["id", "title", "predicted_genre"]].to_csv(output_path, index=False)
print(f"\nSaved predictions for the unlabeled test set to {output_path}")
print(test_df[["title", "predicted_genre"]].head(10))
