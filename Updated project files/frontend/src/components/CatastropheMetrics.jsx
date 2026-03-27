// src/components/CatastropheMetrics.jsx
import React, { useState } from 'react';

const CatastropheMetrics = React.memo(({ catastropheMetrics }) => {
  const [hoveredCard, setHoveredCard] = useState(null);
  const [animated, setAnimated] = useState(false);

  if (!catastropheMetrics) return null;

  React.useEffect(() => {
    setAnimated(true);
    const timer = setTimeout(() => setAnimated(false), 1000);
    return () => clearTimeout(timer);
  }, [catastropheMetrics]);

  const formatCurrency = (value) => {
    if (!value && value !== 0) return "0";
    const absValue = Math.abs(value);
    if (absValue >= 10000000) {
      return `${(absValue / 10000000).toFixed(2)} Cr`;
    } else if (absValue >= 100000) {
      return `${(absValue / 100000).toFixed(2)} L`;
    } else {
      return absValue.toLocaleString("en-IN");
    }
  };

  const formatNumber = (value) => {
    if (!value && value !== 0) return "0";
    return value.toLocaleString("en-IN");
  };

  const metrics = [
    {
      title: "Baseline Events",
      value: catastropheMetrics.baseline_events,
      displayValue: formatNumber(catastropheMetrics.baseline_events),
      icon: "⚠️",
      color: "#ef4444",
      bgLight: "#fee2e2",
      description: "Predicted catastrophic events without intervention"
    },
    {
      title: "Treated Events",
      value: catastropheMetrics.treated_events,
      displayValue: formatNumber(catastropheMetrics.treated_events),
      icon: "🛡️",
      color: "#f59e0b",
      bgLight: "#fed7aa",
      description: "Events after intervention program"
    },
    {
      title: "Avoided Events",
      value: catastropheMetrics.avoided_events,
      displayValue: formatNumber(catastropheMetrics.avoided_events),
      icon: "✅",
      color: "#10b981",
      bgLight: "#d1fae5",
      description: "Catastrophic events prevented"
    },
    {
      title: "Acute Savings",
      value: catastropheMetrics.acute_savings,
      displayValue: `₹${formatCurrency(catastropheMetrics.acute_savings)}`,
      icon: "💰",
      color: "#059669",
      bgLight: "#d1fae5",
      description: "Total savings from avoided events"
    }
  ];

  const getImpactLevel = () => {
    const avoidanceRate = catastropheMetrics.avoided_events / catastropheMetrics.baseline_events;
    if (avoidanceRate >= 0.7) return { level: "Excellent", color: "#10b981", icon: "🎉" };
    if (avoidanceRate >= 0.4) return { level: "Good", color: "#f59e0b", icon: "👍" };
    return { level: "Moderate", color: "#ef4444", icon: "⚠️" };
  };

  const impact = getImpactLevel();
  const avoidanceRate = ((catastropheMetrics.avoided_events / catastropheMetrics.baseline_events) * 100).toFixed(1);

  return (
    <div style={{
      background: "white",
      borderRadius: "20px",
      padding: "20px",
      boxShadow: "0 2px 8px rgba(0,0,0,0.05)",
      transition: "all 0.3s ease",
      marginBottom: "24px"
    }}>
      <style>{`
        @keyframes slideUp {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes shimmer {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(100%); }
        }
      `}</style>

      {/* Header - Consistent with RiskTierChart */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "16px", flexWrap: "wrap", gap: "12px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          <div style={{
            width: "40px",
            height: "40px",
            background: "linear-gradient(135deg, #ef4444 0%, #f97316 100%)",
            borderRadius: "12px",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: "20px"
          }}>
            🚨
          </div>
          <div>
            <h3 style={{ fontSize: "18px", fontWeight: "700", color: "#1f2937", margin: 0 }}>
              Catastrophic Event Impact
            </h3>
            <p style={{ fontSize: "11px", color: "#6b7280", margin: "2px 0 0 0" }}>
              Prevention of acute medical events and hospitalizations
            </p>
          </div>
        </div>
        
        {/* Impact Badge - Compact */}
        <div style={{
          background: impact.color,
          padding: "4px 12px",
          borderRadius: "30px",
          display: "flex",
          alignItems: "center",
          gap: "6px",
          boxShadow: `0 2px 8px ${impact.color}40`
        }}>
          <span style={{ fontSize: "14px" }}>{impact.icon}</span>
          <span style={{ fontSize: "11px", fontWeight: "600", color: "white" }}>
            {impact.level} ({avoidanceRate}% avoided)
          </span>
        </div>
      </div>

      {/* Metrics Grid - 2x2 Grid like RiskTierChart */}
      <div style={{
        display: "grid",
        gridTemplateColumns: "repeat(2, 1fr)",
        gap: "12px",
        marginBottom: "20px"
      }}>
        {metrics.map((metric, idx) => (
          <div
            key={idx}
            style={{
              background: metric.bgLight,
              borderRadius: "16px",
              padding: "14px",
              transition: "all 0.2s ease",
              cursor: "pointer",
              transform: hoveredCard === idx ? "translateY(-2px)" : "translateY(0)",
              boxShadow: hoveredCard === idx ? "0 4px 12px rgba(0,0,0,0.1)" : "none",
              border: `1px solid ${metric.color}20`
            }}
            onMouseEnter={() => setHoveredCard(idx)}
            onMouseLeave={() => setHoveredCard(null)}
          >
            <div style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              marginBottom: "10px"
            }}>
              <span style={{ fontSize: "24px" }}>{metric.icon}</span>
              <p style={{
                fontSize: "11px",
                fontWeight: "500",
                color: metric.color,
                background: "rgba(255,255,255,0.8)",
                padding: "2px 8px",
                borderRadius: "20px",
                margin: 0
              }}>
                {metric.title}
              </p>
            </div>
            
            <p style={{
              fontSize: "24px",
              fontWeight: "bold",
              color: metric.color,
              margin: "0 0 4px 0",
              letterSpacing: "-0.5px"
            }}>
              {metric.displayValue}
            </p>
            
            <p style={{
              fontSize: "10px",
              color: "#6b7280",
              margin: 0,
              lineHeight: "1.3"
            }}>
              {metric.description}
            </p>
          </div>
        ))}
      </div>

      {/* Progress Bar Section - Compact */}
      <div style={{
        background: "#f9fafb",
        borderRadius: "12px",
        padding: "14px",
        marginBottom: "16px"
      }}>
        <div style={{ marginBottom: "10px" }}>
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "6px" }}>
            <span style={{ fontSize: "11px", fontWeight: "500", color: "#374151" }}>
              Event Prevention Rate
            </span>
            <span style={{ fontSize: "11px", fontWeight: "700", color: "#10b981" }}>
              {avoidanceRate}%
            </span>
          </div>
          <div style={{
            width: "100%",
            height: "8px",
            background: "#e5e7eb",
            borderRadius: "10px",
            overflow: "hidden"
          }}>
            <div style={{
              width: `${avoidanceRate}%`,
              height: "100%",
              background: "linear-gradient(90deg, #10b981 0%, #34d399 100%)",
              borderRadius: "10px",
              transition: "width 0.8s ease-out"
            }} />
          </div>
        </div>

        <div style={{
          display: "flex",
          justifyContent: "space-between",
          gap: "12px",
          flexWrap: "wrap"
        }}>
          <div style={{ flex: 1, textAlign: "center" }}>
            <p style={{ fontSize: "9px", color: "#6b7280", marginBottom: "2px" }}>
              Cost Per Avoided
            </p>
            <p style={{ fontSize: "12px", fontWeight: "700", color: "#1f2937", margin: 0 }}>
              ₹{formatCurrency(catastropheMetrics.acute_savings / catastropheMetrics.avoided_events || 0)}
            </p>
          </div>
          <div style={{ flex: 1, textAlign: "center" }}>
            <p style={{ fontSize: "9px", color: "#6b7280", marginBottom: "2px" }}>
              Efficiency
            </p>
            <p style={{ fontSize: "12px", fontWeight: "700", color: "#1f2937", margin: 0 }}>
              {((catastropheMetrics.avoided_events / catastropheMetrics.baseline_events) * 100).toFixed(1)}%
            </p>
          </div>
          <div style={{ flex: 1, textAlign: "center" }}>
            <p style={{ fontSize: "9px", color: "#6b7280", marginBottom: "2px" }}>
              Savings/Member
            </p>
            <p style={{ fontSize: "12px", fontWeight: "700", color: "#10b981", margin: 0 }}>
              ₹{formatCurrency(catastropheMetrics.acute_savings / (catastropheMetrics.baseline_events || 1))}
            </p>
          </div>
        </div>
      </div>

      {/* Insight Message - Compact */}
      <div style={{
        background: "#eff6ff",
        borderRadius: "12px",
        padding: "12px 14px",
        borderLeft: "3px solid #3b82f6",
        transition: "transform 0.2s ease",
        cursor: "pointer"
      }}
      onMouseEnter={(e) => e.currentTarget.style.transform = "translateX(3px)"}
      onMouseLeave={(e) => e.currentTarget.style.transform = "translateX(0)"}>
        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          <span style={{ fontSize: "18px" }}>💡</span>
          <p style={{ fontSize: "11px", color: "#1e40af", margin: 0, lineHeight: "1.4", flex: 1 }}>
            {catastropheMetrics.avoided_events > 50 
              ? `Preventing ${catastropheMetrics.avoided_events} events saves ₹${formatCurrency(catastropheMetrics.acute_savings)} in acute care costs.`
              : `${catastropheMetrics.avoided_events} events avoided saves ₹${formatCurrency(catastropheMetrics.acute_savings)} in acute care costs.`
            }
          </p>
        </div>
      </div>
    </div>
  );
});

export default CatastropheMetrics;
