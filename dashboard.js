const zoneSelect = document.getElementById("zoneSelect");
const anomalyBadge = document.getElementById("anomalyBadge");

let tempChart, humidityWindChart, forecastChart;

async function fetchJSON(url) {
  const res = await fetch(url);
  return res.json();
}

function makeLineChart(ctx, labels, datasets) {
  return new Chart(ctx, {
    type: "line",
    data: { labels, datasets },
    options: {
      responsive: true,
      interaction: { mode: "index", intersect: false },
      scales: {
        x: { ticks: { color: "#94a3b8", maxTicksLimit: 8 }, grid: { color: "#334155" } },
        y: { ticks: { color: "#94a3b8" }, grid: { color: "#334155" } },
      },
      plugins: { legend: { labels: { color: "#e2e8f0" } } },
    },
  });
}

async function loadZone(zone) {
  const ts = await fetchJSON(`/api/timeseries?zone=${encodeURIComponent(zone)}`);
  const anomalyCount = ts.is_anomaly.filter(Boolean).length;
  anomalyBadge.textContent = `${anomalyCount} anomalies in view`;
  anomalyBadge.className = anomalyCount > 0 ? "badge alert" : "badge";

  if (tempChart) tempChart.destroy();
  tempChart = makeLineChart(document.getElementById("tempChart"), ts.timestamps, [
    {
      label: "Temperature (°C)",
      data: ts.temperature_c,
      borderColor: "#38bdf8",
      backgroundColor: "rgba(56,189,248,0.15)",
      pointRadius: ts.is_anomaly.map(a => (a ? 4 : 0)),
      pointBackgroundColor: "#f87171",
      tension: 0.3,
      fill: true,
    },
  ]);

  if (humidityWindChart) humidityWindChart.destroy();
  humidityWindChart = makeLineChart(document.getElementById("humidityWindChart"), ts.timestamps, [
    { label: "Humidity (%)", data: ts.humidity_pct, borderColor: "#a78bfa", tension: 0.3 },
    { label: "Wind (km/h)", data: ts.wind_speed_kmh, borderColor: "#34d399", tension: 0.3 },
  ]);

  const forecast = await fetchJSON(`/api/forecast?zone=${encodeURIComponent(zone)}`);
  if (forecastChart) forecastChart.destroy();
  if (forecast.forecast_c) {
    const hourLabels = forecast.forecast_c.map((_, i) => `+${i + 1}h`);
    forecastChart = makeLineChart(document.getElementById("forecastChart"), hourLabels, [
      { label: "Forecast (°C)", data: forecast.forecast_c, borderColor: "#fbbf24", tension: 0.3 },
    ]);
  }

  const anomalies = await fetchJSON(`/api/anomalies?zone=${encodeURIComponent(zone)}`);
  const tbody = document.querySelector("#anomalyTable tbody");
  tbody.innerHTML = "";
  anomalies.forEach(a => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${a.timestamp}</td><td>${a.temperature_c.toFixed(1)}</td>
                    <td>${a.humidity_pct.toFixed(0)}</td><td>${a.anomaly_score.toFixed(3)}</td>`;
    tbody.appendChild(tr);
  });
}

zoneSelect.addEventListener("change", () => loadZone(zoneSelect.value));
loadZone(zoneSelect.value);
