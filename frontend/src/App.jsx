import { useEffect, useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { getHistoricalPrices } from './api';
import './App.css';

function App() {
  const [prices, setPrices] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    getHistoricalPrices()
      .then((data) => {
        setPrices(data);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  if (loading) return <div style={{ padding: 40 }}>Loading price data...</div>;
  if (error) return <div style={{ padding: 40, color: 'red' }}>Error: {error}</div>;

  return (
    <div style={{ padding: 40 }}>
      <h1>Brent Oil Price History</h1>
      <p>{prices.length} observations loaded from the backend</p>
      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={prices}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="date" tick={false} />
          <YAxis domain={['auto', 'auto']} />
          <Tooltip />
          <Line type="monotone" dataKey="price" stroke="#1f4e79" dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export default App;
