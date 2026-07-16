import { useEffect, useState } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine,
} from 'recharts';
import { getHistoricalPrices, getEvents } from './api';
import './App.css';

const CATEGORY_COLORS = {
  'OPEC policy': '#e07b39',
  'Conflict': '#c0392b',
  'Sanctions': '#8e44ad',
  'Demand shock': '#2980b9',
};

function App() {
  const [prices, setPrices] = useState([]);
  const [events, setEvents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [hoveredEvent, setHoveredEvent] = useState(null);

  useEffect(() => {
    Promise.all([getHistoricalPrices(), getEvents()])
      .then(([priceData, eventData]) => {
        setPrices(priceData);
        setEvents(eventData);
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
      <p>{prices.length} price observations · {events.length} curated events overlaid</p>

      <ResponsiveContainer width="100%" height={450}>
        <LineChart data={prices}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="date" tick={false} />
          <YAxis domain={['auto', 'auto']} label={{ value: 'USD/barrel', angle: -90, position: 'insideLeft' }} />
          <Tooltip />
          <Line type="monotone" dataKey="price" stroke="#1f4e79" dot={false} strokeWidth={1.2} />

          {events.map((ev) => (
            <ReferenceLine
              key={ev.date}
              x={ev.date}
              stroke={CATEGORY_COLORS[ev.category] || '#888'}
              strokeOpacity={0.6}
              onMouseEnter={() => setHoveredEvent(ev)}
              onMouseLeave={() => setHoveredEvent(null)}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>

      <div style={{ marginTop: 16, minHeight: 60 }}>
        {hoveredEvent ? (
          <div style={{
            padding: 12, border: `2px solid ${CATEGORY_COLORS[hoveredEvent.category] || '#888'}`,
            borderRadius: 6, maxWidth: 600,
          }}>
            <strong>{hoveredEvent.date}</strong> — {hoveredEvent.category}
            <div>{hoveredEvent.event}</div>
          </div>
        ) : (
          <div style={{ color: '#888' }}>Hover over a colored vertical line to see the event.</div>
        )}
      </div>

      <div style={{ marginTop: 16, display: 'flex', gap: 16, flexWrap: 'wrap' }}>
        {Object.entries(CATEGORY_COLORS).map(([category, color]) => (
          <div key={category} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <div style={{ width: 14, height: 14, backgroundColor: color, borderRadius: 3 }} />
            <span style={{ fontSize: 14 }}>{category}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default App;