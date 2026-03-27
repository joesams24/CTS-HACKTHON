// src/components/ROIMetrics.jsx
import React, { useState, useEffect } from 'react';

const ROIMetrics = React.memo(({ interventionMetrics }) => {
  const [animated, setAnimated] = useState(false);

  useEffect(() => {
    setAnimated(true);
    const timer = setTimeout(() => setAnimated(false), 1000);
    return () => clearTimeout(timer);
  }, [interventionMetrics]);

  if (!interventionMetrics) return null;

  const isPositive = interventionMetrics.net_benefit > 0;
  const roiColor = interventionMetrics.roi_percent >= 0 ? "#10b981" : "#ef4444";

  const formatCurrency = (value) => {
    if (!value && value !== 0) return "0";
    
    // Convert to absolute value for formatting, then add sign back
    const absValue = Math.abs(value);
    let formattedValue = '';
    
    if (absValue >= 10000000) { // 1 Crore = 10,000,000
      formattedValue = `${(absValue / 10000000).toFixed(2)} Cr`;
    } else if (absValue >= 100000) { // 1 Lakh = 100,000
      formattedValue = `${(absValue / 100000).toFixed(2)} L`;
    } else {
      formattedValue = absValue.toLocaleString("en-IN");
    }
    
    // Add negative sign if value is negative
    const sign = value < 0 ? '-' : '';
    return `₹${sign}${formattedValue}`;
  };

  const metrics = [
    {
      title: "Intervention Cost",
      value: interventionMetrics.total_intervention_cost,
      displayValue: formatCurrency(interventionMetrics.total_intervention_cost),
      icon: "💰",
      color: "#3b82f6",
      bgGradient: "linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%)"
    },
    {
      title: "Expected Savings",
      value: interventionMetrics.total_expected_savings,
      displayValue: formatCurrency(interventionMetrics.total_expected_savings),
      icon: "💾",
      color: "#10b981",
      bgGradient: "linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%)"
    },
    {
      title: "Net Benefit",
      value: interventionMetrics.net_benefit,
      displayValue: formatCurrency(interventionMetrics.net_benefit),
      icon: "📈",
      color: isPositive ? "#10b981" : "#ef4444",
      bgGradient: isPositive 
        ? "linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%)"
        : "linear-gradient(135deg, #fee2e2 0%, #fecaca 100%)"
    },
    {
      title: "ROI",
      value: interventionMetrics.roi_percent,
      displayValue: `${interventionMetrics.roi_percent || 0}%`,
      icon: "📊",
      color: roiColor,
      bgGradient: "linear-gradient(135deg, #fef3c7 0%, #fde68a 100%)"
    }
  ];

  return (
    <div style={{
      display: "grid",
      gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))",
      gap: "20px",
      marginBottom: "32px"
    }}>
      {metrics.map((metric, idx) => (
        <div
          key={idx}
          style={{
            background: metric.bgGradient,
            borderRadius: "20px",
            padding: "20px",
            boxShadow: "0 2px 8px rgba(0,0,0,0.05)",
            transition: "all 0.3s ease",
            animation: animated ? `fadeInUp 0.5s ease-out ${idx * 0.1}s both` : "none",
            cursor: "pointer",
            border: "1px solid rgba(255,255,255,0.5)",
            display: "flex",
            flexDirection: "column",
            minWidth: 0,
            overflow: "hidden"
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.transform = "translateY(-4px)";
            e.currentTarget.style.boxShadow = "0 8px 20px rgba(0,0,0,0.1)";
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.transform = "translateY(0)";
            e.currentTarget.style.boxShadow = "0 2px 8px rgba(0,0,0,0.05)";
          }}
        >
          {/* Icon and Title Row */}
          <div style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            marginBottom: "16px",
            gap: "12px"
          }}>
            <div style={{
              width: "40px",
              height: "40px",
              borderRadius: "12px",
              background: "rgba(255,255,255,0.8)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: "24px",
              flexShrink: 0
            }}>
              {metric.icon}
            </div>
            <p style={{
              fontSize: "13px",
              fontWeight: "500",
              color: "#4b5563",
              margin: 0,
              textAlign: "right",
              wordBreak: "break-word"
            }}>
              {metric.title}
            </p>
          </div>
          
          {/* Value Display */}
          <div style={{
            marginBottom: "12px",
            textAlign: "center",
            padding: "8px 0"
          }}>
            <p style={{
              fontSize: "clamp(20px, 5vw, 32px)",
              fontWeight: "bold",
              color: metric.color,
              margin: 0,
              lineHeight: "1.2",
              wordBreak: "break-word",
              whiteSpace: "normal"
            }}>
              {metric.displayValue}
            </p>
          </div>
          
          {/* Progress Bar */}
          <div style={{
            width: "100%",
            height: "4px",
            background: "rgba(0,0,0,0.1)",
            borderRadius: "2px",
            overflow: "hidden",
            marginTop: "auto"
          }}>
            <div style={{
              width: animated ? "100%" : "0%",
              height: "100%",
              background: metric.color,
              transition: "width 1s ease-out",
              borderRadius: "2px"
            }} />
          </div>
          
          {/* Helper Text */}
          <p style={{
            fontSize: "10px",
            color: "#6b7280",
            margin: "12px 0 0 0",
            textAlign: "center",
            wordBreak: "break-word"
          }}>
            {metric.title === "ROI" && "Return on Investment"}
            {metric.title === "Net Benefit" && "Savings - Cost"}
            {metric.title === "Expected Savings" && "Projected cost reduction"}
            {metric.title === "Intervention Cost" && "Total program spend"}
          </p>
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
});

export default ROIMetrics;
