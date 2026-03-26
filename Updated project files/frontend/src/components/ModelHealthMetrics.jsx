// src/components/ModelHealthMetrics.jsx
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';

export default function ModelHealthMetrics({ mlMetrics, trainingAuc }) {
  if (!mlMetrics) return null;

  const riskData = [
    { name: 'Very Low', value: mlMetrics.high_risk_fraction ? (1 - mlMetrics.high_risk_fraction) * 100 : 0 },
    { name: 'High Risk', value: mlMetrics.high_risk_fraction ? mlMetrics.high_risk_fraction * 100 : 0 },
  ];

  const COLORS = ['#10b981', '#ef4444'];

  const metricsData = [
    { name: 'Mean Risk', value: (mlMetrics.mean_predicted_risk * 100).toFixed(1), color: '#3b82f6' },
    { name: 'Std Deviation', value: (mlMetrics.risk_std_dev * 100).toFixed(1), color: '#8b5cf6' },
    { name: 'Top Decile Risk', value: (mlMetrics.top_decile_avg_risk * 100).toFixed(1), color: '#ec489a' },
  ];

  return (
    <div style={{
      background: "rgba(255, 255, 255, 0.95)",
      backdropFilter: "blur(10px)",
      borderRadius: "20px",
      padding: "24px",
      boxShadow: "0 8px 32px rgba(0,0,0,0.1)",
      transition: "transform 0.3s ease",
      cursor: "pointer"
    }}
    onMouseEnter={(e) => e.currentTarget.style.transform = "translateY(-5px)"}
    onMouseLeave={(e) => e.currentTarget.style.transform = "translateY(0)"}>
      <h3 style={{ marginBottom: "20px", fontSize: "18px", fontWeight: "600", color: "#1f2937", display: "flex", alignItems: "center", gap: "8px" }}>
        <span>🤖</span> Model Health Metrics
        {trainingAuc && (
          <span style={{
            marginLeft: "auto",
            fontSize: "14px",
            background: "#10b981",
            color: "white",
            padding: "4px 12px",
            borderRadius: "20px"
          }}>
            AUC: {(trainingAuc * 100).toFixed(1)}%
          </span>
        )}
      </h3>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "16px", marginBottom: "24px" }}>
        {metricsData.map((metric, idx) => (
          <div key={idx} style={{
            textAlign: "center",
            padding: "16px",
            background: "linear-gradient(135deg, #f9fafb 0%, #ffffff 100%)",
            borderRadius: "12px",
            border: "1px solid #e5e7eb"
          }}>
            <p style={{ fontSize: "12px", color: "#6b7280", marginBottom: "8px" }}>{metric.name}</p>
            <p style={{ fontSize: "28px", fontWeight: "bold", color: metric.color }}>{metric.value}%</p>
          </div>
        ))}
      </div>

      <div style={{ height: "200px", marginTop: "16px" }}>
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={riskData}
              cx="50%"
              cy="50%"
              innerRadius={60}
              outerRadius={80}
              paddingAngle={5}
              dataKey="value"
              label={({ name, value }) => `${name}: ${value.toFixed(1)}%`}
            >
              {riskData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </div>

      <div style={{ marginTop: "16px", paddingTop: "16px", borderTop: "1px solid #e5e7eb", textAlign: "center" }}>
        <p style={{ fontSize: "14px", color: "#6b7280" }}>
          Model predicts {metricsData[0].value}% average readmission risk
        </p>
      </div>
    </div>
  );
}
