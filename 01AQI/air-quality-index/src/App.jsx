import { useState } from 'react';
import Axios from 'axios';
import './App.css';

function App() {
  const [data, setData] = useState(null);
  const [search, setSearch] = useState("");
  const [loading, setLoading] = useState(false);

  const getIndex = async () => {
    if (!search) return alert("Please enter a city");

    setLoading(true);
    setData(null);

    try {
      const params = {
        apikey: import.meta.env.VITE_AQI_KEY,
        q: search
      };

      const queryString = new URLSearchParams(params).toString();
      const url = `/aqi-api/aqi/v1/city?${queryString}`;

      const response = await Axios.get(url);
      setData(response.data.data);
      
    } catch (error) {
      console.error("Connection Error:", error);
      alert("Failed to fetch data.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className='Main-Container'>
      <div className='Search'>
        <input 
          value={search} 
          type='text' 
          placeholder="Enter city (e.g., kathmandu)"
          onChange={(e) => setSearch(e.target.value)} 
        />
        <button onClick={getIndex} disabled={loading}>
          {loading ? "Searching..." : "GET API"}
        </button>
      </div>

      <div className='Result'>
        {loading && <div className="loader"></div>}

        {!loading && data ? (
          <div className="aqi-card">
            <h2>{data.city}</h2>
            <p className="aqi-value">AQI: <strong>{data.aqi}</strong></p>
            <div className="details">
              <span>PM2.5: {data.pm25}</span> | <span>PM10: {data.pm10}</span>
            </div>
          </div>
        ) : !loading && (
          <p>Enter a city to see Air Quality Index</p>
        )}
      </div>
    </div>
  );
}

export default App;