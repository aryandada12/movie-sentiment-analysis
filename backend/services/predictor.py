import re
import joblib
import torch
import torch.nn as nn



class RNN(nn.Module):

    def __init__(self, input_size, hidden_size=128, num_layers=1):

        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):

        h0 = torch.zeros(
            self.num_layers,
            x.size(0),
            self.hidden_size
        )

        out, _ = self.rnn(x, h0)

        out = self.fc(out[:, -1, :])

        return out



tfidf = joblib.load("model/tfidf.pkl")

model = RNN(
    input_size=5000
)

model.load_state_dict(
    torch.load(
        "model/sentiment_model.pth",
        map_location=torch.device("cpu")
    )
)

model.eval()


def preprocess_text(text):

    text = text.lower()

    text = re.sub(
        r"http\S+",
        "",
        text
    )

    text = re.sub(
        r"<.*?>",
        "",
        text
    )

    text = re.sub(
        r"[^A-Za-z0-9\s]",
        "",
        text
    )

    return text


def predict_sentiment(text):

    cleaned_text = preprocess_text(text)

    vector = tfidf.transform([cleaned_text])

    vector = vector.toarray()

    tensor = torch.from_numpy(vector).float()

    tensor = tensor.unsqueeze(1)

    with torch.no_grad():

        output = model(tensor)

        probability = torch.sigmoid(
            output.squeeze()
        ).item()

    if probability >= 0.5:

        sentiment = "positive"
        confidence = probability

    else:

        sentiment = "negative"
        confidence = 1 - probability

    return {
        "sentiment": sentiment,
        "confidence": round(confidence, 4),
        "positive_probability": round(probability, 4),
        "negative_probability": round(1 - probability, 4)
    }