
import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split

from sentiment_classifier import (
    load_and_prepare,
    TFIDFClassifier
)

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Amazon Review Sentiment Analysis",
    page_icon="🛒",
    layout="wide"
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():

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

    return model


model = load_model()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.title("📊 Project Info")

    st.markdown("""
    ### Model

    - TF-IDF Vectorization
    - Logistic Regression
    - 50K Amazon Reviews
    - Binary Sentiment Classification

    ### Features

    - Real-time Prediction
    - Confidence Score
    - Probability Analysis
    - Batch Predictions
    """)

    st.success(
        "Built by Samir Azam"
    )

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title(
    "🛒 Amazon Review Sentiment Analyzer"
)

st.markdown(
    """
Analyze customer reviews and predict whether
the sentiment is Positive or Negative.
"""
)

# --------------------------------------------------
# EXAMPLES
# --------------------------------------------------

with st.expander(
    "💡 Example Reviews"
):

    st.write(
        "Positive Example:"
    )

    st.code(
        "This product is amazing. Great quality and fast delivery."
    )

    st.write(
        "Negative Example:"
    )

    st.code(
        "Terrible purchase. Completely useless and broke in two days."
    )

# --------------------------------------------------
# SINGLE REVIEW
# --------------------------------------------------

st.subheader(
    "🔍 Analyze Single Review"
)

review = st.text_area(
    "Enter Review Text",
    height=180
)

if st.button(
    "Analyze Sentiment"
):

    if review.strip():

        pred, prob = model.predict(
            review
        )

        confidence = max(prob)

        positive_prob = prob[1]
        negative_prob = prob[0]

        st.markdown("---")

        if pred == 1:

            st.success(
                "✅ Positive Sentiment"
            )

        else:

            st.error(
                "❌ Negative Sentiment"
            )

        st.metric(
            "Confidence Score",
            f"{confidence:.2%}"
        )

        st.subheader(
            "Probability Distribution"
        )

        probs_df = pd.DataFrame({

            "Class": [
                "Negative",
                "Positive"
            ],

            "Probability": [
                negative_prob,
                positive_prob
            ]

        })

        st.bar_chart(
            probs_df.set_index(
                "Class"
            )
        )

    else:

        st.warning(
            "Please enter a review."
        )

# --------------------------------------------------
# BATCH PREDICTION
# --------------------------------------------------

st.markdown("---")

st.subheader(
    "📂 Batch Review Analysis"
)

uploaded_file = st.file_uploader(
    "Upload CSV with a column named 'review'",
    type=["csv"]
)

if uploaded_file:

    try:

        df = pd.read_csv(
            uploaded_file
        )

        if "review" not in df.columns:

            st.error(
                "CSV must contain a 'review' column."
            )

        else:

            predictions = []
            confidences = []

            for text in df["review"]:

                pred, prob = model.predict(
                    str(text)
                )

                predictions.append(
                    "Positive"
                    if pred == 1
                    else "Negative"
                )

                confidences.append(
                    max(prob)
                )

            df["prediction"] = predictions

            df["confidence"] = confidences

            st.success(
                "Predictions Complete"
            )

            st.dataframe(
                df.head(20)
            )

            csv = df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="Download Results",
                data=csv,
                file_name="sentiment_predictions.csv",
                mime="text/csv"
            )

    except Exception as e:

        st.error(
            str(e)
        )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.caption(
    "Amazon Product Review Sentiment Classification using TF-IDF + Logistic Regression"
)

