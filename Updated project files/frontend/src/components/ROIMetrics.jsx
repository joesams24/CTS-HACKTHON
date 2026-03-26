// src/components/ROIMetrics.jsx
import { useState, useEffect } from 'react';

export default function ROIMetrics({ interventionMetrics }) {
  const [animated, setAnimated] = useState(false);

  useEffect(() => {
    setAnimated(true);
    const timer = setTimeout(() => setAnimated(false), 1000);
    return () => clearTimeout(timer);
  }, [interventionMetrics]);

  if (!interventionMetrics) return null;

  const isPositive = interventionMetrics.net_benefit > 0;
  const roiColor = interventionMetrics.roi_percent >= 0 ? "#10b981" : "#ef4444";

  const metrics = [
    {
      title: "💰 Intervention Cost",
      value: interventionMetrics.total_intervention_cost,
      format: "currency",
      icon: "💰",
      color: "#3b82f6"
    },
    {
      title: "💾 Expected Savings",
      value: interventionMetrics.total_expected_savings,
      format: "currency",
      icon: "💾",
      color: "#10b981"
    },
    {
      title: "📈 Net Benefit",
      value: interventionMetrics.net_benefit,
      format: "currency",
      icon: "📈",
      color: isPositive ? "#10b981" : "#ef4444"
    },
    {
      title: "📊 ROI",
      value: interventionMetrics.roi_percent,
      format: "percentage",
      icon: "📊",
      color: roiColor
    }
  ];

  return (
    <div style={{
      display: "grid",
      gridTemplateColumns: "repeat(auto-fit, minmax(250px, 1fr))",
      gap: "20px",
      marginBottom: "32px"
    }}>
      {metrics.map((metric, idx) => (
        <div
          key={idx}
          style={{
            background: "rgba(255, 255, 255, 0.95)",
            backdropFilter: "blur(10px)",
            borderRadius: "20px",
            padding: "24px",
            boxShadow: "0 8px 32px rgba(0,0,0,0.1)",
            transition: "all 0.3s ease",
            animation: animated ? `fadeInUp 0.5s ease-out ${idx * 0.1}s both` : "none",
            cursor: "pointer",
            border: "1px solid rgba(255,255,255,0.2)"
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.transform = "translateY(-8px) scale(1.02)";
            e.currentTarget.style.boxShadow = "0 12px 48px rgba(0,0,0,0.15)";
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.transform = "translateY(0) scale(1)";
            e.currentTarget.style.boxShadow = "0 8px 32px rgba(0,0,0,0.1)";
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: "12px", marginBottom: "16px" }}>
            <span style={{ fontSize: "32px" }}>{metric.icon}</span>
            <p style={{ fontSize: "14px", color: "#6b7280", margin: 0 }}>{metric.title}</p>
          </div>
          <p style={{ 
            fontSize: "32px", 
            fontWeight: "bold", 
            color: metric.color,
            marginBottom: "8px"
          }}>
            {metric.format === "currency" 
              ? `₹${metric.value?.toLocaleString("en-IN") || 0}`
              : `${metric.value || 0}%`
            }
          </p>
          <div style={{
            width: "100%",
            height: "4px",
            background: "#e5e7eb",
            borderRadius: "2px",
            overflow: "hidden",
            marginTop: "16px"
          }}>
            <div style={{
              width: animated ? "100%" : "0%",
              height: "100%",
              background: metric.color,
              transition: "width 1s ease-out",
              borderRadius: "2px"
            }} />
          </div>
        </div>
      ))}
      <style>{`
        @keyframes fadeInUp {
          from {
            opacity: 0;
            transform: translateY(20px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
      `}</style>
    </div>
  );
}
