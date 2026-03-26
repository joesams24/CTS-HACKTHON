// src/components/RiskTierChart.jsx
import { useState, useEffect } from 'react';
import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip, BarChart, Bar, XAxis, YAxis, CartesianGrid } from 'recharts';

export default function RiskTierChart({ tierDistribution }) {
  const [selectedTier, setSelectedTier] = useState(null);
  const [animated, setAnimated] = useState(false);

  useEffect(() => {
    setAnimated(true);
    const timer = setTimeout(() => setAnimated(false), 500);
    return () => clearTimeout(timer);
  }, [tierDistribution]);

  if (!tierDistribution) return null;

  const tiers = [
    { name: "Very Low", color: "#10b981", icon: "🟢", gradient: "linear-gradient(135deg, #10b981 0%, #34d399 100%)" },
    { name: "Low", color: "#34d399", icon: "🔵", gradient: "linear-gradient(135deg, #34d399 0%, #6ee7b7 100%)" },
    { name: "Medium", color: "#f59e0b", icon: "🟡", gradient: "linear-gradient(135deg, #f59e0b 0%, #fbbf24 100%)" },
    { name: "High", color: "#ef4444", icon: "🟠", gradient: "linear-gradient(135deg, #ef4444 0%, #f87171 100%)" },
    { name: "Very High", color: "#dc2626", icon: "🔴", gradient: "linear-gradient(135deg, #dc2626 0%, #ef4444 100%)" }
  ];

  const total = Object.values(tierDistribution).reduce((a, b) => a + b, 0);
  
  const pieData = tiers.map(tier => ({
    name: tier.name,
    value: tierDistribution[tier.name] || 0,
    percentage: total > 0 ? ((tierDistribution[tier.name] || 0) / total * 100).toFixed(1) : 0,
    color: tier.color
  })).filter(d => d.value > 0);

  const barData = tiers.map(tier => ({
    name: tier.name,
    count: tierDistribution[tier.name] || 0,
    percentage: total > 0 ? ((tierDistribution[tier.name] || 0) / total * 100) : 0,
    color: tier.color
  }));

  return (
    <div style={{
      background: "rgba(255, 255, 255, 0.95)",
      backdropFilter: "blur(10px)",
      borderRadius: "20px",
      padding: "24px",
      boxShadow: "0 8px 32px rgba(0,0,0,0.1)",
      transition: "transform 0.3s ease, box-shadow 0.3s ease",
      animation: animated ? "scaleIn 0.5s ease-out" : "none"
    }}
    onMouseEnter={(e) => {
      e.currentTarget.style.transform = "translateY(-5px)";
      e.currentTarget.style.boxShadow = "0 12px 48px rgba(0,0,0,0.15)";
    }}
    onMouseLeave={(e) => {
      e.currentTarget.style.transform = "translateY(0)";
      e.currentTarget.style.boxShadow = "0 8px 32px rgba(0,0,0,0.1)";
    }}>
      <style>{`
        @keyframes scaleIn {
          from {
            opacity: 0;
            transform: scale(0.95);
          }
          to {
            opacity: 1;
            transform: scale(1);
          }
        }
      `}</style>
      
      <h3 style={{ marginBottom: "20px", fontSize: "20px", fontWeight: "600", color: "#1f2937", display: "flex", alignItems: "center", gap: "8px" }}>
        <span>📊</span> Risk Tier Distribution
        <span style={{
          marginLeft: "auto",
          fontSize: "14px",
          background: "#f3f4f6",
          padding: "4px 12px",
          borderRadius: "20px",
          color: "#6b7280"
        }}>
          Total: {total} members
        </span>
      </h3>

      {/* Chart Type Selector */}
      <div style={{ display: "flex", gap: "8px", marginBottom: "20px" }}>
        <button style={{
          padding: "6px 16px",
          background: "#667eea",
          color: "white",
          border: "none",
          borderRadius: "8px",
          fontSize: "12px",
          cursor: "pointer"
        }}>
          Pie Chart
        </button>
        <button style={{
          padding: "6px 16px",
          background: "#f3f4f6",
          color: "#374151",
          border: "none",
          borderRadius: "8px",
          fontSize: "12px",
          cursor: "pointer"
        }}>
          Bar Chart
        </button>
      </div>

      {/* Pie Chart */}
      <div style={{ height: "300px", marginBottom: "24px" }}>
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={pieData}
              cx="50%"
              cy="50%"
              innerRadius={60}
              outerRadius={100}
              paddingAngle={3}
              dataKey="value"
              onMouseEnter={(data) => setSelectedTier(data.name)}
              onMouseLeave={() => setSelectedTier(null)}
            >
              {pieData.map((entry, index) => (
                <Cell 
                  key={`cell-${index}`} 
                  fill={entry.color}
                  stroke="white"
                  strokeWidth={2}
                  style={{
                    cursor: "pointer",
                    transition: "transform 0.3s ease",
                    filter: selectedTier === entry.name ? "brightness(1.1)" : "brightness(1)",
                    transform: selectedTier === entry.name ? "scale(1.05)" : "scale(1)",
                    transformOrigin: "center"
                  }}
                />
              ))}
            </Pie>
            <Tooltip 
              contentStyle={{ 
                background: "rgba(255,255,255,0.95)", 
                border: "none", 
                borderRadius: "8px", 
                boxShadow: "0 4px 12px rgba(0,0,0,0.1)"
              }}
              formatter={(value, name, props) => [`${value} members (${props.payload.percentage}%)`, name]}
            />
            <Legend />
          </PieChart>
        </ResponsiveContainer>
      </div>

      {/* Detailed Stats with Progress Bars */}
      <div style={{ marginTop: "24px" }}>
        {barData.map((tier, idx) => (
          <div 
            key={tier.name} 
            style={{
              marginBottom: "16px",
              animation: `slideInRight 0.5s ease-out ${idx * 0.1}s both`
            }}
            onMouseEnter={() => setSelectedTier(tier.name)}
            onMouseLeave={() => setSelectedTier(null)}
          >
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "8px" }}>
              <span style={{ fontWeight: "500", fontSize: "14px" }}>
                {tiers.find(t => t.name === tier.name)?.icon} {tier.name}
              </span>
              <span style={{ color: "#6b7280", fontSize: "14px" }}>
                {tier.count} members ({tier.percentage.toFixed(1)}%)
              </span>
            </div>
            <div style={{ 
              background: "#e5e7eb", 
              borderRadius: "12px", 
              overflow: "hidden",
              height: "40px",
              position: "relative"
            }}>
              <div
                style={{
                  width: `${tier.percentage}%`,
                  background: tiers.find(t => t.name === tier.name)?.gradient,
                  height: "100%",
                  transition: "width 1s ease-out",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "flex-end",
                  paddingRight: tier.percentage > 15 ? "12px" : "0",
                  color: "white",
                  fontSize: "14px",
                  fontWeight: "600",
                  borderRadius: "12px",
                  position: "relative",
                  overflow: "hidden"
                }}
              >
                {tier.percentage > 15 && `${tier.percentage.toFixed(0)}%`}
                <div style={{
                  position: "absolute",
                  top: 0,
                  left: 0,
                  right: 0,
                  bottom: 0,
                  background: "linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent)",
                  animation: "shimmer 2s infinite"
                }} />
              </div>
            </div>
          </div>
        ))}
      </div>

      <style>{`
        @keyframes slideInRight {
          from {
            opacity: 0;
            transform: translateX(-20px);
          }
          to {
            opacity: 1;
            transform: translateX(0);
          }
        }
        
        @keyframes shimmer {
          0% {
            transform: translateX(-100%);
          }
          100% {
            transform: translateX(100%);
          }
        }
      `}</style>
    </div>
  );
}
