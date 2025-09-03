import "../styles.css";
import { useEffect, useState } from "react";
import { useParams, Link } from "react-router-dom";
import LoadingSpinner from "../loadingSpinner/LoadingSpinner.jsx";

const API = "http://localhost:8000";
const today = new Date();

// make a copy of today
const tomorrow = new Date(today);
tomorrow.setDate(today.getDate() + 1);
const options = {
  weekday: "long",
  year: "numeric",
  month: "long",
  day: "numeric",
};

export default function Prediction() {
  const { symbol } = useParams(); // from URL
  const [prediction, setPrediction] = useState(null);
  const [averagePrediction, setAveragePrediction] = useState(null);
  const [status, setStatus] = useState(LoadingSpinner());
  console.log("SYMBOL: ", symbol);
  console.log("PREDICTION :", prediction);

  useEffect(() => {
    async function fetchPrediction() {
      try {
        const res = await fetch(`${API}/predict`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ symbol }),
        });
        const data = await res.json();
        console.log("DATA :", data);
        const averagePredictedData =
          (data.predictions.LSTM + data.predictions.GRU) / 2;
        setAveragePrediction(averagePredictedData);

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
    <div className="card-prediction" style={{ position: "relative" }}>
      <h1 className="preheading" style={{ textAlign: "center" }}>
        Prediction of the next day’s closing price for {" "}
        <span
          style={{
            backgroundColor: "black",
            padding: "10px",
            borderRadius: "8px",
          }}
        >
          {symbol.toUpperCase()}
        </span>
      </h1>
      <h3 style={{ marginBottom: "80px" }}>
        {tomorrow.toLocaleDateString("en-US", options)}
      </h3>
      {status && <p>{status}</p>}

      {prediction && (
        <span
          style={{
            display: "flex",
            gap: "50px",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          <p
            style={{
              display: "flex",
              alignItems: "center",
              gap: "60px",
              fontSize: "2rem",
              margin: "10px",
              // flexDirection:""
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
                <span style={{fontWeight: "bold" }}>
                  LSTM
                </span>{" "}
                <div
                  style={{
                    backgroundColor: "white",
                    fontWeight: "normal",
                    padding: "10px",
                    color: "black",
                    borderRadius: "8px",
                    marginTop:"10px"
                  }}
                >
                  {Number(prediction.LSTM).toFixed(2)}
                </div>{" "}
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
                <span style={{ marginBottom: "600px" }}>
                  <span
                    style={{
                      color: "white",
                      fontWeight: "bold",
                    }}
                  >
                    GRU
                  </span>
                </span>{" "}
                <div
                  style={{
                    color: "black",
                    backgroundColor: "white",
                    fontWeight: "normal",
                    padding: "10px",
                    borderRadius: "8px",
                     marginTop:"10px"
                  }}
                >
                  {Number(prediction.GRU).toFixed(2)}
                </div>
              </span>
            )}
          </p>
          <span
            style={{
              // marginBottom: "20px",
              color: "white",
              fontWeight: "bold",
              backgroundColor: "#4A3699",
              padding: "15px",
              borderRadius: "10px",
              border: "2px solid skyblue",
              textAlign: "center",
              // marginBottom: "20px",
              color: "white",
              fontSize: "2rem",
              // margintop: "20px"
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
            }}
          >
            <span style={{ fontWeight: "bold", marginBottom: "10px" }}>
              Average price
            </span>{" "}
            <span
              style={{
                color: "black",
                backgroundColor: "white",
                fontWeight: "normal",
                padding: "15px",
                borderRadius: "8px",
                marginTop:"4px"
              }}
            >
              {Number(averagePrediction).toFixed(2)}
            </span>
          </span>
        </span>
      )}
      <Link
        to="/"
        style={{
          color: "white",
          textDecoration: "none",
          position: "absolute",
          top: "25px",
          left: "10px",
          padding: "20px",
        }}
        className="back-page"
      >
        Back
      </Link>
    </div>
  );
}
