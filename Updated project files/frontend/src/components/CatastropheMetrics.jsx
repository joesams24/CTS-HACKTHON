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

  const metrics = [
    {
      title: "Baseline Events",
      value: catastropheMetrics.baseline_events,
      icon: "⚠️",
      color: "#ef4444",
      bgGradient: "linear-gradient(135deg, #fee2e2 0%, #fecaca 100%)",
      description: "Predicted catastrophic events without intervention",
      trend: "negative"
    },
    {
      title: "Treated Events",
      value: catastropheMetrics.treated_events,
      icon: "🛡️",
      color: "#f59e0b",
      bgGradient: "linear-gradient(135deg, #fed7aa 0%, #fdba74 100%)",
      description: "Events after intervention program",
      trend: "neutral"
    },
    {
      title: "Avoided Events",
      value: catastropheMetrics.avoided_events,
      icon: "✅",
      color: "#10b981",
      bgGradient: "linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%)",
      description: "Catastrophic events prevented",
      trend: "positive"
    },
    {
      title: "Acute Savings",
      value: catastropheMetrics.acute_savings,
      icon: "💰",
      color: "#059669",
      bgGradient: "linear-gradient(135deg, #d1fae5 0%, #6ee7b7 100%)",
      description: "Total savings from avoided events",
      trend: "positive",
      isCurrency: true
    }
  ];

  const formatCurrency = (value) => {
    if (!value && value !== 0) return "0";
    return new Intl.NumberFormat('en-IN', { 
      maximumFractionDigits: 0,
      minimumFractionDigits: 0
    }).format(value);
  };

  const formatNumber = (value) => {
    if (!value && value !== 0) return "0";
    return new Intl.NumberFormat('en-IN').format(value);
  };

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
      borderRadius: "24px",
      padding: "28px",
      boxShadow: "0 4px 20px rgba(0,0,0,0.08)",
      marginBottom: "24px",
      transition: "all 0.3s ease",
      animation: animated ? "fadeInUp 0.6s ease-out" : "none"
    }}>
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
        
        @keyframes pulse {
          0%, 100% {
            transform: scale(1);
          }
          50% {
            transform: scale(1.05);
          }
        }
        
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
        
        @keyframes countUp {
          from {
            opacity: 0;
            transform: translateY(10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
      `}</style>

      {/* Header with Impact Badge */}
      <div style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        marginBottom: "24px",
        flexWrap: "wrap",
        gap: "16px"
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
          <div style={{
            width: "48px",
            height: "48px",
            background: "linear-gradient(135deg, #ef4444 0%, #f97316 100%)",
            borderRadius: "16px",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: "24px",
            animation: "pulse 2s infinite"
          }}>
            🚨
          </div>
          <div>
            <h3 style={{ fontSize: "20px", fontWeight: "700", color: "#1f2937", margin: 0 }}>
              Catastrophic Event Impact
            </h3>
            <p style={{ fontSize: "13px", color: "#6b7280", margin: "4px 0 0 0" }}>
              Prevention of acute medical events and hospitalizations
            </p>
          </div>
        </div>
        
        {/* Impact Badge */}
        <div style={{
          background: impact.color,
          padding: "8px 20px",
          borderRadius: "40px",
          display: "flex",
          alignItems: "center",
          gap: "8px",
          boxShadow: `0 4px 12px ${impact.color}40`,
          animation: "slideInRight 0.5s ease-out"
        }}>
          <span style={{ fontSize: "20px" }}>{impact.icon}</span>
          <div>
            <p style={{ fontSize: "11px", color: "white", opacity: 0.9, margin: 0 }}>
              Intervention Impact
            </p>
            <p style={{ fontSize: "16px", fontWeight: "bold", color: "white", margin: 0 }}>
              {impact.level} ({avoidanceRate}% avoided)
            </p>
          </div>
        </div>
      </div>

      {/* Metrics Grid */}
      <div style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(250px, 1fr))",
        gap: "20px",
        marginBottom: "28px"
      }}>
        {metrics.map((metric, idx) => {
          const displayValue = metric.isCurrency 
            ? `₹${formatCurrency(metric.value)}`
            : formatNumber(metric.value);
          
          const isPositive = metric.trend === "positive";
          const isNegative = metric.trend === "negative";
          
          return (
            <div
              key={idx}
              style={{
                background: metric.bgGradient,
                borderRadius: "20px",
                padding: "20px",
                transition: "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
                cursor: "pointer",
                animation: `countUp 0.5s ease-out ${idx * 0.1}s both`,
                transform: hoveredCard === idx ? "translateY(-8px) scale(1.02)" : "translateY(0)",
                boxShadow: hoveredCard === idx 
                  ? "0 12px 24px rgba(0,0,0,0.15)" 
                  : "0 2px 8px rgba(0,0,0,0.05)"
              }}
              onMouseEnter={() => setHoveredCard(idx)}
              onMouseLeave={() => setHoveredCard(null)}
            >
              <div style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                marginBottom: "16px"
              }}>
                <span style={{ fontSize: "32px" }}>{metric.icon}</span>
                <div style={{
                  width: "40px",
                  height: "40px",
                  borderRadius: "12px",
                  background: "rgba(255,255,255,0.5)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  transition: "transform 0.3s ease",
                  transform: hoveredCard === idx ? "rotate(10deg)" : "rotate(0)"
                }}>
                  {isPositive && <span style={{ fontSize: "20px" }}>📈</span>}
                  {isNegative && <span style={{ fontSize: "20px" }}>📉</span>}
                  {!isPositive && !isNegative && <span style={{ fontSize: "20px" }}>📊</span>}
                </div>
              </div>
              
              <p style={{
                fontSize: "32px",
                fontWeight: "800",
                color: metric.color,
                margin: "0 0 8px 0",
                letterSpacing: "-0.5px"
              }}>
                {displayValue}
              </p>
              
              <p style={{
                fontSize: "14px",
                fontWeight: "600",
                color: "#374151",
                margin: "0 0 4px 0"
              }}>
                {metric.title}
              </p>
              
              <p style={{
                fontSize: "12px",
                color: "#6b7280",
                margin: 0,
                lineHeight: "1.4"
              }}>
                {metric.description}
              </p>
            </div>
          );
        })}
      </div>

      {/* Progress Bar & Additional Stats */}
      <div style={{
        background: "linear-gradient(135deg, #f3f4f6 0%, #ffffff 100%)",
        borderRadius: "16px",
        padding: "20px",
        marginBottom: "20px"
      }}>
        <div style={{ marginBottom: "16px" }}>
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "8px" }}>
            <span style={{ fontSize: "13px", fontWeight: "500", color: "#374151" }}>
              Event Prevention Rate
            </span>
            <span style={{ fontSize: "13px", fontWeight: "700", color: "#10b981" }}>
              {avoidanceRate}%
            </span>
          </div>
          <div style={{
            width: "100%",
            height: "12px",
            background: "#e5e7eb",
            borderRadius: "20px",
            overflow: "hidden"
          }}>
            <div style={{
              width: `${avoidanceRate}%`,
              height: "100%",
              background: "linear-gradient(90deg, #10b981 0%, #34d399 100%)",
              borderRadius: "20px",
              transition: "width 1s ease-out",
              position: "relative",
              animation: animated ? "slideInRight 1s ease-out" : "none"
            }}>
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

        <div style={{
          display: "flex",
          justifyContent: "space-between",
          gap: "20px",
          flexWrap: "wrap"
        }}>
          <div style={{ flex: 1 }}>
            <p style={{ fontSize: "11px", color: "#6b7280", marginBottom: "4px" }}>
              Cost Per Avoided Event
            </p>
            <p style={{ fontSize: "16px", fontWeight: "700", color: "#1f2937" }}>
              ₹{formatCurrency(catastropheMetrics.acute_savings / catastropheMetrics.avoided_events || 0)}
            </p>
          </div>
          <div style={{ flex: 1 }}>
            <p style={{ fontSize: "11px", color: "#6b7280", marginBottom: "4px" }}>
              Intervention Efficiency
            </p>
            <p style={{ fontSize: "16px", fontWeight: "700", color: "#1f2937" }}>
              {((catastropheMetrics.avoided_events / catastropheMetrics.baseline_events) * 100).toFixed(1)}% reduction
            </p>
          </div>
          <div style={{ flex: 1 }}>
            <p style={{ fontSize: "11px", color: "#6b7280", marginBottom: "4px" }}>
              Savings per Member
            </p>
            <p style={{ fontSize: "16px", fontWeight: "700", color: "#10b981" }}>
              ₹{formatCurrency(catastropheMetrics.acute_savings / (catastropheMetrics.baseline_events || 1))}
            </p>
          </div>
        </div>
      </div>

      {/* Insight Message */}
      <div style={{
        background: "linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%)",
        borderRadius: "16px",
        padding: "16px 20px",
        borderLeft: "4px solid #3b82f6",
        transition: "transform 0.3s ease",
        cursor: "pointer"
      }}
      onMouseEnter={(e) => e.currentTarget.style.transform = "translateX(5px)"}
      onMouseLeave={(e) => e.currentTarget.style.transform = "translateX(0)"}>
        <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
          <span style={{ fontSize: "24px" }}>💡</span>
          <div>
            <p style={{ fontSize: "13px", fontWeight: "600", color: "#1e40af", margin: 0 }}>
              Key Insight
            </p>
            <p style={{ fontSize: "13px", color: "#1e3a8a", margin: "4px 0 0 0", lineHeight: "1.5" }}>
              {catastropheMetrics.avoided_events > 50 
                ? `Excellent results! Preventing ${catastropheMetrics.avoided_events} catastrophic events saves ₹${formatCurrency(catastropheMetrics.acute_savings)} in acute care costs. This intervention shows strong ROI potential.`
                : `By preventing ${catastropheMetrics.avoided_events} catastrophic events, the intervention saves ₹${formatCurrency(catastropheMetrics.acute_savings)} in acute care costs. Continue monitoring high-risk members for maximum impact.`
              }
            </p>
          </div>
        </div>
      </div>

      <style>{`
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
});

export default CatastropheMetrics;
