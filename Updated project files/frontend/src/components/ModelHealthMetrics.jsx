// src/components/ModelHealthMetrics.jsx
import React from 'react';

const ModelHealthMetrics = React.memo(({ mlMetrics, trainingAuc }) => {
  if (!mlMetrics) return null;

  // Format metrics properly based on type
  const metricsData = [
    { 
      name: 'Mean Risk', 
      value: mlMetrics.mean_predicted_risk,
      displayValue: mlMetrics.mean_predicted_risk.toFixed(4),
      color: '#3b82f6',
      description: 'Average predicted readmission risk',
      unit: ''
    },
    { 
      name: 'Std Deviation', 
      value: mlMetrics.risk_std_dev,
      displayValue: mlMetrics.risk_std_dev.toFixed(4),
      color: '#8b5cf6',
      description: 'Risk variability across population',
      unit: ''
    },
    { 
      name: 'High Risk %', 
      value: mlMetrics.high_risk_fraction * 100,
      displayValue: (mlMetrics.high_risk_fraction * 100).toFixed(1),
      color: '#ef4444',
      description: 'Members with elevated risk',
      unit: '%'
    },
    { 
      name: 'Top Decile Risk', 
      value: mlMetrics.top_decile_avg_risk,
      displayValue: mlMetrics.top_decile_avg_risk.toFixed(4),
      color: '#ec489a',
      description: 'Average risk of highest 10%',
      unit: ''
    },
  ];

  return (
    <div style={{
      background: "white",
      borderRadius: "20px",
      padding: "28px",
      boxShadow: "0 2px 8px rgba(0,0,0,0.05)",
      transition: "transform 0.2s ease"
    }}>
      <div style={{ 
        display: "flex", 
        alignItems: "center", 
        justifyContent: "space-between", 
        marginBottom: "24px", 
        flexWrap: "wrap", 
        gap: "12px" 
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
          <span style={{ fontSize: "28px" }}></span>
          <h2 style={{ fontSize: "20px", fontWeight: "600", color: "#1f2937", margin: 0 }}>
            Model Health Metrics
          </h2>
        </div>
        {trainingAuc && (
          <div style={{
            background: "linear-gradient(135deg, #10b981 0%, #059669 100%)",
            padding: "8px 20px",
            borderRadius: "30px",
            color: "white",
            fontSize: "14px",
            fontWeight: "600"
          }}>
            AUC Score: {(trainingAuc * 100).toFixed(1)}%
          </div>
        )}
      </div>

      {/* Metrics Cards Grid */}
      <div style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))",
        gap: "20px",
        marginBottom: "24px"
      }}>
        {metricsData.map((metric, idx) => (
          <div
            key={idx}
            style={{
              padding: "24px",
              background: "linear-gradient(135deg, #f9fafb 0%, #ffffff 100%)",
              borderRadius: "16px",
              border: "1px solid #e5e7eb",
              textAlign: "center",
              transition: "all 0.2s ease",
              cursor: "pointer"
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.transform = "translateY(-2px)";
              e.currentTarget.style.boxShadow = "0 4px 12px rgba(0,0,0,0.1)";
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.transform = "translateY(0)";
              e.currentTarget.style.boxShadow = "none";
            }}
          >
            <p style={{ fontSize: "13px", color: "#6b7280", marginBottom: "12px", fontWeight: "500" }}>
              {metric.name}
            </p>
            <p style={{ 
              fontSize: "36px", 
              fontWeight: "bold", 
              color: metric.color, 
              marginBottom: "12px",
              letterSpacing: "-0.5px"
            }}>
              {metric.displayValue}{metric.unit}
            </p>
            <p style={{ fontSize: "12px", color: "#9ca3af", lineHeight: "1.4" }}>
              {metric.description}
            </p>
          </div>
        ))}
      </div>

      {/* Interpretation */}
      <div style={{
        marginTop: "8px",
        padding: "20px",
        background: "#f0f9ff",
        borderRadius: "12px",
        borderLeft: "4px solid #3b82f6"
      }}>
        <p style={{ fontSize: "14px", color: "#1e40af", margin: 0, lineHeight: "1.6" }}>
          📊 <strong>Model Interpretation:</strong> The model shows average risk of <strong>{(mlMetrics.mean_predicted_risk * 100).toFixed(2)}%</strong> 
          with a standard deviation of <strong>{mlMetrics.risk_std_dev.toFixed(4)}</strong> ({ (mlMetrics.risk_std_dev * 100).toFixed(2)}%). 
          The high-risk population (top 40%) represents <strong>{(mlMetrics.high_risk_fraction * 100).toFixed(1)}%</strong> of members,
          with the top decile showing risk of <strong>{(mlMetrics.top_decile_avg_risk * 100).toFixed(2)}%</strong>.
        </p>
      </div>

      {/* Additional Insight */}
      <div style={{
        marginTop: "16px",
        padding: "16px",
        background: "#fef3c7",
        borderRadius: "12px",
        borderLeft: "4px solid #f59e0b"
      }}>
        <p style={{ fontSize: "13px", color: "#92400e", margin: 0 }}>
          💡 <strong>Insight:</strong> {
            mlMetrics.risk_std_dev > 0.15 
              ? "High risk variability suggests diverse member needs. Consider targeted interventions for different risk segments."
              : "Moderate risk variability indicates consistent risk patterns across the population. Standardized interventions may be effective."
          }
        </p>
      </div>
    </div>
  );
});

export default ModelHealthMetrics;
