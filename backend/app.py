from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import joblib
import torch
import torch.nn as nn
import pandas as pd
from typing import Dict, List

# -----------------------
# App & Config
# -----------------------
app = FastAPI(title="NEPSE Stock Predictor API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # frontend local dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = "trained_models"
DATA_DIR = "fetchStockData"
WINDOW = 30
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Dynamic Model Classes
# -----------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.linear(out[:, -1, :])


class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.rnn = nn.GRU(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.linear(out[:, -1, :])


# -----------------------
# Utilities
# -----------------------
def get_symbols() -> List[str]:
    if not os.path.exists(MODELS_DIR):
        return []
    symbols = set()
    for f in os.listdir(MODELS_DIR):
        if f.endswith("_LSTM_model_state_dict.pth") or f.endswith("_GRU_model_state_dict.pth"):
            symbols.add(f.split("_")[0])  # e.g. ADBL_LSTM... -> ADBL
    return sorted(symbols)


def load_close_window(symbol: str) -> torch.Tensor:
    csv_path = os.path.join(DATA_DIR, f"{symbol}.csv")
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail=f"No CSV found for {symbol}")

    df = pd.read_csv(csv_path)
    if "Close" not in df.columns:
        raise HTTPException(status_code=400, detail=f"CSV for {symbol} missing 'Close' column")

    closing = df["Close"].dropna().values[-WINDOW:]
    if len(closing) < WINDOW:
        raise HTTPException(status_code=400, detail=f"Not enough data (need {WINDOW} days) for {symbol}")

    return torch.from_numpy(closing.reshape(1, WINDOW, 1)).float().to(device)


def infer_hidden_size(state_dict: Dict[str, torch.Tensor], rnn_type="LSTM") -> int:
    """Infer hidden size from checkpoint weight shape."""
    key = "rnn.weight_hh_l0"
    if key not in state_dict:
        raise RuntimeError(f"State dict missing key {key} for {rnn_type}")
    return state_dict[key].shape[1]  # hidden_size is the second dim


def predict_model(symbol: str, model_type: str, seq_raw: torch.Tensor) -> float:
    model_path = os.path.join(MODELS_DIR, f"{symbol}_{model_type}_model_state_dict.pth")
    scaler_path = os.path.join(MODELS_DIR, f"{symbol}_{model_type}_scaler.pkl")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(f"{model_type} artifacts missing")

    # load scaler
    scaler = joblib.load(scaler_path)

    # scale input
    seq_scaled = torch.from_numpy(
        scaler.transform(seq_raw.detach().cpu().numpy().reshape(-1, 1))
    ).float().view(1, WINDOW, 1).to(device)

    # load checkpoint to infer hidden size
    checkpoint = torch.load(model_path, map_location=device)
    hidden_size = infer_hidden_size(checkpoint, model_type)

    # create model with correct hidden size
    if model_type == "LSTM":
        model = LSTMModel(hidden_layer_size=hidden_size).to(device)
    else:
        model = GRUModel(hidden_layer_size=hidden_size).to(device)

    model.load_state_dict(checkpoint)
    model.eval()

    with torch.no_grad():
        pred_scaled = model(seq_scaled).cpu().numpy()  # shape [1,1]
    return float(scaler.inverse_transform(pred_scaled).item())


# -----------------------
# Routes
# -----------------------
@app.get("/symbols")
def list_symbols() -> Dict[str, List[str]]:
    return {"symbols": get_symbols()}


class PredictRequest(BaseModel):
    symbol: str


@app.post("/predict")
def predict(req: PredictRequest):
    symbol = req.symbol.upper()
    if symbol not in get_symbols():
        raise HTTPException(status_code=400, detail="Invalid symbol")

    try:
        seq_raw = load_close_window(symbol)
        results: Dict[str, float] = {}

        # try LSTM
        try:
            results["LSTM"] = predict_model(symbol, "LSTM", seq_raw)
        except FileNotFoundError:
            pass

        # try GRU
        try:
            results["GRU"] = predict_model(symbol, "GRU", seq_raw)
        except FileNotFoundError:
            pass

        if not results:
            raise HTTPException(status_code=404, detail=f"No trained models found for {symbol}")

        return {"symbol": symbol, "predictions": results}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {e}")
