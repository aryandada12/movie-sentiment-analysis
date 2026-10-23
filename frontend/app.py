import streamlit as st
import requests


st.set_page_config(
    page_title="Movie Sentiment AI",
    page_icon="🎬",
    layout="centered"
)


st.title("🎬 Movie Sentiment Analysis")

st.write(
    "Enter a movie review and  we "
    "predict whether the sentiment is positive or negative."
)


review = st.text_area(
    "Movie Review",
    placeholder="Example: This movie was amazing and I really enjoyed it!"
)


if st.button("Analyze Sentiment"):

    if not review.strip():

        st.warning(
            "Please enter a movie review."
        )

    else:

        with st.spinner("Analyzing review..."):

            try:

                response = requests.post(
                    "http://127.0.0.1:8000/predict",
                    json={
                        "review": review
                    }
                )

                if response.status_code == 200:

                    result = response.json()

                    sentiment = result["sentiment"]
                    confidence = result["confidence"]

                    st.divider()

                    if sentiment == "positive":

                        st.success(
                            f"Positive Sentiment"
                        )

                    else:

                        st.error(
                            f"Negative Sentiment"
                        )

                    st.metric(
                        "Confidence",
                        f"{confidence * 100:.2f}%"
                    )

                    col1, col2 = st.columns(2)

                    with col1:

                        st.metric(
                            "Positive Probability",
                            f"{result['positive_probability'] * 100:.2f}%"
                        )

                    with col2:

                        st.metric(
                            "Negative Probability",
                            f"{result['negative_probability'] * 100:.2f}%"
                        )

                else:

                    st.error(
                        f"API Error: {response.status_code}"
                    )

            except requests.exceptions.ConnectionError:

                st.error(
                    "Cannot connect to FastAPI. "
                    "Make sure the backend is running."
                )