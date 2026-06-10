
"""
evaluate.py

Detailed Evaluation Module
Author: Samir Azam
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    learning_curve
)

from sentiment_classifier import (
    load_and_prepare,
    TFIDFClassifier
)

# --------------------------------------------------
# CREATE SCREENSHOTS FOLDER
# --------------------------------------------------

os.makedirs(
    "screenshots",
    exist_ok=True
)

# --------------------------------------------------
# ALL METRICS
# --------------------------------------------------

def print_metrics(
        y_test,
        y_pred,
        y_prob
):

    print("\n" + "=" * 50)
    print("MODEL PERFORMANCE")
    print("=" * 50)

    print(
        f"Accuracy  : "
        f"{accuracy_score(y_test, y_pred):.4f}"
    )

    print(
        f"Precision : "
        f"{precision_score(y_test, y_pred):.4f}"
    )

    print(
        f"Recall    : "
        f"{recall_score(y_test, y_pred):.4f}"
    )

    print(
        f"F1 Score  : "
        f"{f1_score(y_test, y_pred):.4f}"
    )

    print(
        f"ROC-AUC   : "
        f"{roc_auc_score(y_test, y_prob):.4f}"
    )

# --------------------------------------------------
# CROSS VALIDATION
# --------------------------------------------------

def perform_cross_validation(
        model,
        X,
        y,
        cv=5
):

    scores = cross_val_score(
        model.pipeline,
        X,
        y,
        cv=cv,
        scoring="accuracy",
        n_jobs=1
    )

    print("\n" + "=" * 50)
    print("CROSS VALIDATION")
    print("=" * 50)

    print(
        "Scores:",
        [round(x, 4) for x in scores]
    )

    print(
        f"Mean Accuracy: "
        f"{scores.mean():.4f}"
    )

    print(
        f"Std Dev: "
        f"{scores.std():.4f}"
    )

    return scores

# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------

def plot_confusion_matrix(
        y_test,
        y_pred
):

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    plt.figure(
        figsize=(6, 5)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.title(
        "Confusion Matrix"
    )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Actual"
    )

    plt.tight_layout()

    plt.savefig(
        "screenshots/confusion_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

# --------------------------------------------------
# ROC CURVE
# --------------------------------------------------

def plot_roc_curve(
        y_test,
        y_prob
):

    fpr, tpr, _ = roc_curve(
        y_test,
        y_prob
    )

    auc = roc_auc_score(
        y_test,
        y_prob
    )

    plt.figure(
        figsize=(7, 5)
    )

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"AUC = {auc:.4f}"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.title(
        "ROC Curve"
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "screenshots/roc_curve.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

# --------------------------------------------------
# LEARNING CURVE
# --------------------------------------------------

def plot_learning_curve(
        model,
        X,
        y
):

    train_sizes, train_scores, val_scores = learning_curve(
        model.pipeline,
        X,
        y,
        cv=5,
        scoring="accuracy",
        n_jobs=1
    )

    train_mean = train_scores.mean(
        axis=1
    )

    val_mean = val_scores.mean(
        axis=1
    )

    plt.figure(
        figsize=(7, 5)
    )

    plt.plot(
        train_sizes,
        train_mean,
        linewidth=2,
        label="Training Accuracy"
    )

    plt.plot(
        train_sizes,
        val_mean,
        linewidth=2,
        label="Validation Accuracy"
    )

    plt.xlabel(
        "Training Examples"
    )

    plt.ylabel(
        "Accuracy"
    )

    plt.title(
        "Learning Curve"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "screenshots/learning_curve.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

# --------------------------------------------------
# ERROR ANALYSIS
# --------------------------------------------------

def show_errors(
        X_test,
        y_test,
        y_pred,
        n=5
):

    mistakes = []

    for text, actual, pred in zip(
            X_test,
            y_test,
            y_pred
    ):

        if actual != pred:

            mistakes.append(
                (
                    text,
                    actual,
                    pred
                )
            )

    print("\n" + "=" * 50)
    print("ERROR ANALYSIS")
    print("=" * 50)

    for idx, (
            text,
            actual,
            pred
    ) in enumerate(
        mistakes[:n]
    ):

        print(
            f"\nExample {idx + 1}"
        )

        print("-" * 50)

        print(
            text[:250]
        )

        print(
            f"\nActual : "
            f"{'Positive' if actual else 'Negative'}"
        )

        print(
            f"Predicted : "
            f"{'Positive' if pred else 'Negative'}"
        )

# --------------------------------------------------
# NGRAM COMPARISON
# --------------------------------------------------

def compare_ngrams(
        X_train,
        y_train,
        X_test,
        y_test
):

    print("\n" + "=" * 50)
    print("NGRAM COMPARISON")
    print("=" * 50)

    configs = [

        ("Unigrams", (1, 1)),
        ("Uni + Bigrams", (1, 2)),
        ("Up To Trigrams", (1, 3))

    ]

    for name, ngram in configs:

        model = TFIDFClassifier()

        model.pipeline.set_params(
            tfidf__ngram_range=ngram
        )

        model.fit(
            X_train,
            y_train
        )

        preds = model.pipeline.predict(
            X_test
        )

        acc = accuracy_score(
            y_test,
            preds
        )

        print(
            f"{name:<20} "
            f"Accuracy = "
            f"{acc:.4f}"
        )

# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    df = load_and_prepare(
        "Reviews.csv",
        sample_size=50000
    )

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )

    model = TFIDFClassifier()

    model.fit(
        X_train,
        y_train
    )

    y_pred = model.pipeline.predict(
        X_test
    )

    y_prob = model.pipeline.predict_proba(
        X_test
    )[:, 1]

    print_metrics(
        y_test,
        y_pred,
        y_prob
    )

    perform_cross_validation(
        model,
        df["clean_text"],
        df["label"]
    )

    compare_ngrams(
        X_train,
        y_train,
        X_test,
        y_test
    )

    show_errors(
        X_test.tolist(),
        y_test.tolist(),
        y_pred.tolist()
    )

    model.get_top_features()

    plot_confusion_matrix(
        y_test,
        y_pred
    )

    plot_roc_curve(
        y_test,
        y_prob
    )

    plot_learning_curve(
        model,
        df["clean_text"],
        df["label"]
    )

    print(
        "\nEvaluation Complete."
    )

    print(
        "\nScreenshots saved to screenshots/ folder."
    )

if __name__ == "__main__":
    main()
