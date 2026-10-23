from fastapi import FastAPI
from pydantic import BaseModel

from services.predictor import predict_sentiment


app = FastAPI(
    title="Movie Sentiment Analysis API",
    description="RNN-based movie review sentiment analysis API",
    version="1.0.0"
)


class ReviewRequest(BaseModel):

    review: str


@app.get("/")
def home():

    return {
        "message": "Movie Sentiment Analysis API is running"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(request: ReviewRequest):

    result = predict_sentiment(
        request.review
    )

    return {
        "review": request.review,
        **result
    }