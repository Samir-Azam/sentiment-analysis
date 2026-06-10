
"""
Amazon Product Review Sentiment Classifier
Author: Samir Azam

Features:
- Text Cleaning
- TF-IDF Vectorization
- Logistic Regression
- GridSearchCV Hyperparameter Tuning
- Feature Importance Analysis
"""

import re
import joblib
import warnings
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")


# --------------------------------------------------
# TEXT CLEANING
# --------------------------------------------------

def clean_text(text):

    if not isinstance(text, str):
        return ""

    text = text.lower()

    text = re.sub(
        r"<[^>]+>",
        " ",
        text
    )

    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# --------------------------------------------------
# DATA LOADING
# --------------------------------------------------

def load_and_prepare(
        filepath="Reviews.csv",
        sample_size=50000
):

    df = pd.read_csv(filepath)

    df = df[df["Score"] != 3]

    df["label"] = (
        df["Score"] >= 4
    ).astype(int)

    df["clean_text"] = (
        df["Text"]
        .astype(str)
        .apply(clean_text)
    )

    if len(df) > sample_size:

        df = df.sample(
            sample_size,
            random_state=42
        )

    pos = df[
        df["label"] == 1
    ]

    neg = df[
        df["label"] == 0
    ]

    min_count = min(
        len(pos),
        len(neg)
    )

    pos = pos.sample(
        min_count,
        random_state=42
    )

    neg = neg.sample(
        min_count,
        random_state=42
    )

    df = pd.concat(
        [pos, neg]
    )

    df = df.sample(
        frac=1,
        random_state=42
    )

    print(
        f"[DATA] Samples: {len(df)}"
    )

    return df


# --------------------------------------------------
# CLASSIFIER
# --------------------------------------------------

class TFIDFClassifier:

    def __init__(self):

        self.pipeline = Pipeline([

            (
                "tfidf",
                TfidfVectorizer(
                    stop_words="english",
                    sublinear_tf=True
                )
            ),

            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42
                )
            )

        ])

    def fit(
            self,
            X_train,
            y_train
    ):

        self.pipeline.fit(
            X_train,
            y_train
        )

    def predict(
            self,
            text
    ):

        cleaned = clean_text(text)

        pred = self.pipeline.predict(
            [cleaned]
        )[0]

        prob = self.pipeline.predict_proba(
            [cleaned]
        )[0]

        return pred, prob

    def save_model(
            self,
            path="model.pkl"
    ):

        joblib.dump(
            self.pipeline,
            path
        )

    def load_model(
            self,
            path="model.pkl"
    ):

        self.pipeline = joblib.load(
            path
        )

    def get_top_features(
            self,
            n=15
    ):

        tfidf = self.pipeline.named_steps[
            "tfidf"
        ]

        clf = self.pipeline.named_steps[
            "clf"
        ]

        feature_names = np.array(
            tfidf.get_feature_names_out()
        )

        coef = clf.coef_[0]

        top_pos = feature_names[
            np.argsort(coef)[-n:]
        ][::-1]

        top_neg = feature_names[
            np.argsort(coef)[:n]
        ]

        print(
            "\nTop Positive Features:"
        )

        print(
            ", ".join(top_pos)
        )

        print(
            "\nTop Negative Features:"
        )

        print(
            ", ".join(top_neg)
        )


# --------------------------------------------------
# HYPERPARAMETER TUNING
# --------------------------------------------------

def tune_model(
        X_train,
        y_train
):

    pipeline = Pipeline([

        (
            "tfidf",
            TfidfVectorizer(
                stop_words="english"
            )
        ),

        (
            "clf",
            LogisticRegression(
                max_iter=1000
            )
        )

    ])

    param_grid = {

        "tfidf__max_features":
            [5000, 10000],

        "tfidf__ngram_range":
            [
                (1, 1),
                (1, 2)
            ],

        "clf__C":
            [
                0.1,
                1,
                10
            ]

    }

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    grid.fit(
        X_train,
        y_train
    )

    print(
        "\nBest Parameters:"
    )

    print(
        grid.best_params_
    )

    print(
        f"Best CV Accuracy: "
        f"{grid.best_score_:.4f}"
    )

    return grid.best_estimator_


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    df = load_and_prepare(
        "Reviews.csv",
        sample_size=50000
    )

    print(df.head())

