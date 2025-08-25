import "../styles.css";
import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

const API = "http://localhost:8000";

export default function Home() {
  const [symbols, setSymbols] = useState([]);
  const [loading, setLoading] = useState(true);
  const [symbol, setSymbol] = useState("");
  const navigate = useNavigate();

  useEffect(() => {
    async function loadSymbols() {
      try {
        const res = await fetch(`${API}/symbols`);
        const data = await res.json();
        setSymbols(data.symbols || []);
        if (data.symbols?.length) setSymbol(data.symbols[0]);
      } finally {
        setLoading(false);
      }
    }
    loadSymbols();
  }, []);

  function handlePredict() {
    if (symbol) {
      navigate(`/stock/${symbol.toUpperCase()}`);
    }
  }

  return (
    <div className="card-home">
      <h1 style={{ margin: "20px", fontSize: "4rem" }}>
        NEPSE Stock Predictor
      </h1>
      <label style={{ fontSize: "2rem" }}>Select Stock Symbol:</label>
      <select
        disabled={loading || !symbols.length}
        value={symbol}
        onChange={(e) => setSymbol(e.target.value)}
      >
        {loading && <option>Loading symbols…</option>}
        {!loading &&
          symbols.map((s) => (
            <option key={s} value={s} style={{ height: "20px" }}>
              {s}
            </option>
          ))}
      </select>
      <button onClick={handlePredict} disabled={!symbol}>
        Predict Next Day
      </button>
    </div>
  );
}
