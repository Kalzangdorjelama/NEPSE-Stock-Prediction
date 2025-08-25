import "../styles.css";
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
    <div className="card-prediction">
      <h1 className="preheading">Prediction for {symbol.toUpperCase()}</h1>
      {status && <p>{status}</p>}

      {prediction && (
        <p
          style={{
            display: "flex",
            alignItems: "center",
            gap: "60px",
            fontSize: "2rem",
          }}
        >
          {/* {symbol.toUpperCase()} →{" "} */}
          {prediction.LSTM !== undefined && (
            <span
              style={{
                backgroundColor: "green",
                padding: "20px",
                borderRadius: "10px",
                border: "2px solid skyblue",
              }}
            >
              <span style={{ marginBottom: "20px", fontWeight: "bold" }}>
                LSTM
              </span>{" "}
              <div>{Number(prediction.LSTM).toFixed(2)}</div>{" "}
            </span>
          )}
          {prediction.GRU !== undefined && (
            <span
              style={{
                backgroundColor: "#4A3699",
                padding: "20px",
                borderRadius: "10px",
                border: "2px solid skyblue",
              }}
            >
              {prediction.LSTM !== undefined ? "  " : ""}
              <span>
                <span
                  style={{
                    marginBottom: "20px",
                    color: "white",
                    fontWeight: "bold",
                  }}
                >
                  GRU
                </span>
              </span>{" "}
              <div style={{ color: "white" }}>
                {Number(prediction.GRU).toFixed(2)}
              </div>
            </span>
          )}
        </p>
      )}

      <Link
        to="/"
        style={{ color: "white", textDecoration: "none" }}
        className="back-page"
      >
        back
      </Link>
    </div>
  );
}
