import axios from 'axios';

const API_BASE = 'http://localhost:5000/api';

export async function getHistoricalPrices(startDate, endDate) {
  const params = {};
  if (startDate) params.start_date = startDate;
  if (endDate) params.end_date = endDate;

  const response = await axios.get(`${API_BASE}/historical`, { params });
  return response.data.data; // array of {date, price}
}
export async function getEvents(category = null) {
  const params = {};
  if (category) params.category = category;

  const response = await axios.get(`${API_BASE}/events`, { params });
  return response.data.data; // array of {date, event, category, expected_direction}
}