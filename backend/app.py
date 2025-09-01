import os
import pandas as pd
import torch
import torch.nn as nn
import joblib
from typing import List, Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# -----------------------------
# Global Config
# -----------------------------
MODELS_DIR = "trained_models"
DATA_DIR = "fetchStockData"
WINDOW = 30
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# RNN Models
# -----------------------------
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


# -----------------------------
# Classes from Diagram
# -----------------------------
class User:
    def select_stock(self, stock_name: str):
        return Stock(stock_name)


class Stock:
    def __init__(self, stock_name: str):
        self.stock_name = stock_name

    def fetch_data(self) -> pd.DataFrame:
        csv_path = os.path.join(DATA_DIR, f"{self.stock_name}.csv")
        if not os.path.exists(csv_path):
            raise HTTPException(status_code=404, detail=f"No CSV found for {self.stock_name}")
        return pd.read_csv(csv_path)


class ClosePriceProcessor:
    def __init__(self, stock: Stock):
        self.stock = stock

    def fetch_close_prices(self) -> pd.DataFrame:
        df = self.stock.fetch_data()
        if "Close" not in df.columns:
            raise HTTPException(status_code=400, detail=f"CSV missing 'Close' column")
        return df[["Close"]]  # only return Close column


class FeatureSelector:
    def __init__(self, model_type: str):
        self.model_type = model_type
        self.model = None

    def select_features(self, df: pd.DataFrame) -> torch.Tensor:
        closing = df["Close"].dropna().values[-WINDOW:]
        if len(closing) < WINDOW:
            raise HTTPException(status_code=400, detail=f"Not enough data for {self.model_type}")
        return torch.from_numpy(closing.reshape(1, WINDOW, 1)).float().to(device)


class Predictor:
    def __init__(self, model_type: str):
        self.model_type = model_type

    def _infer_hidden_size(self, state_dict: Dict[str, torch.Tensor]) -> int:
        return state_dict["rnn.weight_hh_l0"].shape[1]

    def predict(self, symbol: str, seq_raw: torch.Tensor) -> float:
        model_path = os.path.join(MODELS_DIR, f"{symbol}_{self.model_type}_model_state_dict.pth")
        scaler_path = os.path.join(MODELS_DIR, f"{symbol}_{self.model_type}_scaler.pkl")

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError(f"{self.model_type} artifacts missing")

        scaler = joblib.load(scaler_path)
        seq_scaled = torch.from_numpy(
            scaler.transform(seq_raw.cpu().numpy().reshape(-1, 1))
        ).float().view(1, WINDOW, 1).to(device)

        checkpoint = torch.load(model_path, map_location=device)
        hidden_size = self._infer_hidden_size(checkpoint)

        if self.model_type == "LSTM":
            model = LSTMModel(hidden_layer_size=hidden_size).to(device)
        else:
            model = GRUModel(hidden_layer_size=hidden_size).to(device)

        model.load_state_dict(checkpoint)
        model.eval()

        with torch.no_grad():
            pred_scaled = model(seq_scaled).cpu().numpy()
        return float(scaler.inverse_transform(pred_scaled).item())


# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(title="NEPSE Stock Predictor API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/symbols")
def list_symbols() -> Dict[str, List[str]]:
    if not os.path.exists(MODELS_DIR):
        return {"symbols": []}
    symbols = set()
    for f in os.listdir(MODELS_DIR):
        if f.endswith("_LSTM_model_state_dict.pth") or f.endswith("_GRU_model_state_dict.pth"):
            symbols.add(f.split("_")[0])
    return {"symbols": sorted(symbols)}


class PredictRequest(BaseModel):
    symbol: str


@app.post("/predict")
def predict(req: PredictRequest):
    symbol = req.symbol.upper()
    stock = Stock(symbol)
    processor = ClosePriceProcessor(stock)
    df = processor.fetch_close_prices()
    # print("PC",df)

    results: Dict[str, float] = {}
    for model_type in ["LSTM", "GRU"]:
        try:
            selector = FeatureSelector(model_type)
            seq_raw = selector.select_features(df)
            predictor = Predictor(model_type)
            results[model_type] = predictor.predict(symbol, seq_raw)
        except FileNotFoundError:
            pass

    if not results:
        raise HTTPException(status_code=404, detail=f"No trained models found for {symbol}")

    return {"symbol": symbol, "predictions": results}
