import { useEffect, useState } from "react";
import { useParams, Link } from "react-router-dom";

const API = "http://localhost:8000";

export default function Prediction() {
  const { symbol } = useParams(); // from URL
  const [prediction, setPrediction] = useState(null);
  const [status, setStatus] = useState("⏳ Loading...");

  useEffect(() => {
    async function fetchPrediction() {
      try {
        const res = await fetch(`${API}/predict`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ symbol }),
        });
        const data = await res.json();

        if (res.ok && data?.predictions) {
          setPrediction(data.predictions);
          setStatus("");
        } else if (data?.detail) {
          setStatus(`❌ Error: ${data.detail}`);
        } else {
          setStatus("❌ Unknown error occurred");
        }
      } catch (e) {
        setStatus("❌ Failed to connect to server");
      }
    }
    fetchPrediction();
  }, [symbol]);

  return (
    <div className="card">
      <h1>Prediction for {symbol.toUpperCase()}</h1>
      {status && <p>{status}</p>}

      {prediction && (
        <p>
          {symbol.toUpperCase()} →{" "}
          {prediction.LSTM !== undefined && (
            <span>LSTM: {Number(prediction.LSTM).toFixed(2)} </span>
          )}
          {prediction.GRU !== undefined && (
            <span>
              {prediction.LSTM !== undefined ? " | " : ""}
              GRU: {Number(prediction.GRU).toFixed(2)}
            </span>
          )}
        </p>
      )}

      <Link to="/">Back</Link>
    </div>
  );
}
