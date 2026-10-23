# 🎬 Movie Sentiment Analysis

An end-to-end machine learning application that predicts whether a movie review expresses **positive or negative sentiment** using a trained PyTorch RNN model and TF-IDF text features.

## 🚀 Features

- Text-based movie review sentiment prediction
- Text preprocessing and TF-IDF feature extraction
- Trained PyTorch RNN model
- FastAPI backend for prediction requests
- Streamlit frontend for interactive predictions
- REST API endpoint for integrating the model with other applications

## 🧠 Model Architecture

1. **Input:** Movie review text
2. **Preprocessing:** Clean and transform the review text
3. **Feature extraction:** Convert text into numerical TF-IDF features
4. **Prediction:** Pass the features to the trained PyTorch RNN model
5. **Output:** Positive or negative sentiment

## 🛠️ Tech Stack

- Python
- PyTorch
- Scikit-learn
- TF-IDF
- FastAPI
- Streamlit
- Joblib

## 📁 Project Structure

```text
movie-sentiment-analysis/
├── backend/
│   ├── main.py
│   ├── model/
│   │   ├── sentiment_model.pth
│   │   └── tfidf.pkl
│   ├── services/
│   │   └── predictor.py
│   └── requirements.txt
├── frontend/
│   ├── app.py
│   └── requirements.txt
└── README.md
```

Adjust the structure above if your actual folders or filenames differ.

##  Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/aryandada12/movie-sentiment-analysis.git
cd movie-sentiment-analysis
```

### 2. Install backend dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 3. Start the FastAPI server

```bash
python -m uvicorn main:app --reload
```

Backend URL: `http://127.0.0.1:8000`

Interactive API documentation: `http://127.0.0.1:8000/docs`

### 4. Start the Streamlit frontend

Open a second terminal from the project root:

```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL shown by Streamlit, usually `http://localhost:8501`.

**Note:** Keep the backend running while using the frontend. The frontend's API URL must point to the running backend.

## 🔌 API Usage

**Endpoint:** `POST /predict`

Example request:

```json
{
  "review": "This movie was fantastic. I really enjoyed it!"
}
```

The API returns the prediction in JSON format. Check the response schema at `/docs`.

## 📊 Model Performance

The model achieved approximately **85.16% test accuracy** in the original notebook experiment.

Performance may vary depending on the environment and preprocessing.

## 🔮 Future Improvements

- Experiment with LSTM and GRU architectures
- Improve text preprocessing and model evaluation
- Add confidence scores to predictions
- Deploy the backend and frontend publicly
- Add automated tests and continuous integration

## 👨‍💻 Author

**Aryan Maurya**

- GitHub: [@aryandada12](https://github.com/aryandada12)

---

*Built as an end-to-end machine learning project to learn NLP, model serving, and application development.*
