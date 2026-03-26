// src/components/ROITrendChart.jsx
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart } from 'recharts';

export default function ROITrendChart({ roiData, selectedWindow }) {
  if (!roiData) return null;

  const chartData = Object.entries(roiData).map(([window, roi]) => ({
    window: `${window} days`,
    roi: roi,
    isSelected: parseInt(window) === selectedWindow
  }));

  return (
    <div style={{ marginTop: "24px", padding: "20px", background: "#f9fafb", borderRadius: "12px" }}>
      <p style={{ fontSize: "14px", color: "#6b7280", marginBottom: "16px", fontWeight: "500" }}>
        📈 ROI Trend Across Time Horizons
      </p>
      <div style={{ height: "250px" }}>
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={chartData}>
            <defs>
              <linearGradient id="roiGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#667eea" stopOpacity={0.8}/>
                <stop offset="95%" stopColor="#764ba2" stopOpacity={0.2}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis dataKey="window" stroke="#6b7280" />
            <YAxis stroke="#6b7280" label={{ value: 'ROI (%)', angle: -90, position: 'insideLeft' }} />
            <Tooltip 
              contentStyle={{ 
                background: "rgba(255,255,255,0.95)", 
                border: "none", 
                borderRadius: "8px", 
                boxShadow: "0 4px 12px rgba(0,0,0,0.1)"
              }}
              formatter={(value) => [`${value}%`, 'ROI']}
            />
            <Area 
              type="monotone" 
              dataKey="roi" 
              stroke="#667eea" 
              strokeWidth={3}
              fill="url(#roiGradient)" 
              dot={{ r: 6, fill: "#667eea", strokeWidth: 2, stroke: "#fff" }}
              activeDot={{ r: 8, fill: "#764ba2" }}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
      <div style={{ marginTop: "16px", textAlign: "center", fontSize: "12px", color: "#6b7280" }}>
        {selectedWindow === 30 && "⚠️ Short-term ROI may be lower due to front-loaded costs"}
        {selectedWindow === 60 && "📊 Mid-term ROI shows early benefits from avoided events"}
        {selectedWindow === 90 && "🚀 Long-term ROI maximizes value from compounding savings"}
      </div>
    </div>
  );
}
