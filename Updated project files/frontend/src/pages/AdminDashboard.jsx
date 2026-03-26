// src/pages/AdminDashboard.jsx
import { useState, useEffect } from "react";
import FileUploader from "../components/FileUploader";
import RiskTierChart from "../components/RiskTierChart";
import ROIMetrics from "../components/ROIMetrics";
import CatastropheMetrics from "../components/CatastropheMetrics";
import RecommendationCard from "../components/RecommendationCard";
import ModelHealthMetrics from "../components/ModelHealthMetrics";
import ROITrendChart from "../components/ROITrendChart";

export default function AdminDashboard() {
  const [dashboard, setDashboard] = useState(null);
  const [selectedWindow, setSelectedWindow] = useState(30);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [animated, setAnimated] = useState(false);

  const windows = [30, 60, 90];

  useEffect(() => {
    if (dashboard) {
      setAnimated(true);
      const timer = setTimeout(() => setAnimated(false), 500);
      return () => clearTimeout(timer);
    }
  }, [selectedWindow, dashboard]);

  const getWindowData = () => {
    if (!dashboard || !dashboard.windows) return null;
    return dashboard.windows[selectedWindow];
  };

  const windowData = getWindowData();

  const formatCurrency = (value) => {
    if (!value && value !== 0) return "0";
    return value.toLocaleString("en-IN", { maximumFractionDigits: 0 });
  };

  const handleWindowChange = (window) => {
    setSelectedWindow(window);
  };

  const handleDataReceived = (data) => {
    console.log("Dashboard data received:", data);
    setDashboard(data);
    setError(null);
    setSelectedWindow(30);
  };

  return (
    <div style={{ 
      minHeight: "100vh", 
      background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
      fontFamily: "'Inter', system-ui, -apple-system, sans-serif"
    }}>
      {/* Animated Background */}
      <div style={{
        position: "fixed",
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        background: "radial-gradient(circle at 20% 50%, rgba(102, 126, 234, 0.1) 0%, transparent 50%)",
        pointerEvents: "none",
        zIndex: 0
      }} />

      {/* Header with Glassmorphism */}
      <div style={{
        backgroundColor: "rgba(255, 255, 255, 0.95)",
        backdropFilter: "blur(10px)",
        borderBottom: "1px solid rgba(255, 255, 255, 0.2)",
        padding: "32px 48px",
        marginBottom: "32px",
        boxShadow: "0 4px 20px rgba(0,0,0,0.1)",
        position: "relative",
        zIndex: 1
      }}>
        <div style={{ maxWidth: "1400px", margin: "0 auto" }}>
          <h1 style={{ 
            fontSize: "36px", 
            fontWeight: "700",
            background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
            marginBottom: "8px"
          }}>
            🏥 Member Risk Stratification Dashboard
          </h1>
          <p style={{ color: "#6b7280", fontSize: "16px" }}>
            AI-powered risk prediction and ROI simulation for proactive care management
          </p>
        </div>
      </div>

      <div style={{ maxWidth: "1400px", margin: "0 auto", padding: "0 32px 32px", position: "relative", zIndex: 1 }}>
        {/* Glassmorphism Upload Section */}
        <div style={{
          background: "rgba(255, 255, 255, 0.95)",
          backdropFilter: "blur(10px)",
          borderRadius: "20px",
          padding: "32px",
          marginBottom: "32px",
          boxShadow: "0 8px 32px rgba(0,0,0,0.1)",
          border: "1px solid rgba(255,255,255,0.2)",
          transition: "transform 0.3s ease, box-shadow 0.3s ease"
        }}>
          <h3 style={{ marginBottom: "20px", fontSize: "20px", fontWeight: "600", color: "#1f2937" }}>
            📁 Upload Member Data
          </h3>
          <FileUploader 
            onResult={handleDataReceived}
            onLoading={setLoading}
            onError={setError}
          />
          {error && (
            <div style={{ 
              marginTop: "16px", 
              padding: "12px", 
              background: "linear-gradient(135deg, #fee2e2 0%, #fecaca 100%)", 
              borderRadius: "12px", 
              color: "#dc2626",
              border: "1px solid #fecaca"
            }}>
              ⚠️ Error: {error}
            </div>
          )}
        </div>

        {/* Loading Animation */}
        {loading && (
          <div style={{
            textAlign: "center",
            padding: "80px",
            background: "rgba(255, 255, 255, 0.95)",
            backdropFilter: "blur(10px)",
            borderRadius: "20px",
            boxShadow: "0 8px 32px rgba(0,0,0,0.1)"
          }}>
            <div style={{
              width: "60px",
              height: "60px",
              margin: "0 auto 20px",
              border: "4px solid #e5e7eb",
              borderTopColor: "#667eea",
              borderRadius: "50%",
              animation: "spin 1s linear infinite"
            }} />
            <p style={{ fontSize: "18px", color: "#4b5563" }}>Processing your data...</p>
            <p style={{ fontSize: "14px", color: "#9ca3af", marginTop: "8px" }}>Training ML model and simulating ROI</p>
            <style>{`
              @keyframes spin {
                to { transform: rotate(360deg); }
              }
            `}</style>
          </div>
        )}

        {/* Dashboard Content with Animation */}
        {dashboard && !loading && (
          <div style={{
            animation: animated ? "fadeInUp 0.5s ease-out" : "none"
          }}>
            <style>{`
              @keyframes fadeInUp {
                from {
                  opacity: 0;
                  transform: translateY(30px);
                }
                to {
                  opacity: 1;
                  transform: translateY(0);
                }
              }
            `}</style>

            {/* Executive Summary with Gradient */}
            <div style={{
              background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
              borderRadius: "20px",
              padding: "28px 32px",
              marginBottom: "32px",
              boxShadow: "0 8px 32px rgba(102, 126, 234, 0.3)",
              transition: "transform 0.3s ease",
              cursor: "pointer"
            }}
            onMouseEnter={(e) => e.currentTarget.style.transform = "translateY(-5px)"}
            onMouseLeave={(e) => e.currentTarget.style.transform = "translateY(0)"}>
              <p style={{ fontSize: "14px", fontWeight: "500", color: "rgba(255,255,255,0.9)", marginBottom: "12px" }}>
                📋 Executive Summary
              </p>
              <p style={{ fontSize: "16px", color: "white", lineHeight: "1.6", marginBottom: "20px" }}>
                {dashboard.executive_summary}
              </p>
              <div style={{ display: "flex", gap: "48px", flexWrap: "wrap" }}>
                <div>
                  <p style={{ fontSize: "12px", color: "rgba(255,255,255,0.8)" }}>Total Members</p>
                  <p style={{ fontSize: "32px", fontWeight: "bold", color: "white" }}>
                    {dashboard.population_size?.toLocaleString() || 0}
                  </p>
                </div>
                <div>
                  <p style={{ fontSize: "12px", color: "rgba(255,255,255,0.8)" }}>Mean Risk Score</p>
                  <p style={{ fontSize: "32px", fontWeight: "bold", color: "white" }}>
                    {dashboard.ml_metrics?.mean_predicted_risk ? (dashboard.ml_metrics.mean_predicted_risk * 100).toFixed(1) : "0"}%
                  </p>
                </div>
                <div>
                  <p style={{ fontSize: "12px", color: "rgba(255,255,255,0.8)" }}>High Risk Population</p>
                  <p style={{ fontSize: "32px", fontWeight: "bold", color: "white" }}>
                    {dashboard.ml_metrics?.high_risk_fraction ? (dashboard.ml_metrics.high_risk_fraction * 100).toFixed(1) : "0"}%
                  </p>
                </div>
              </div>
            </div>

            {/* Time Window Selector with 3D Effect */}
            <div style={{
              background: "rgba(255, 255, 255, 0.95)",
              backdropFilter: "blur(10px)",
              borderRadius: "20px",
              padding: "24px",
              marginBottom: "32px",
              boxShadow: "0 8px 32px rgba(0,0,0,0.1)"
            }}>
              <h3 style={{ marginBottom: "20px", fontSize: "18px", fontWeight: "600", color: "#1f2937" }}>
                ⏱️ Select Analysis Time Horizon
              </h3>
              <div style={{ display: "flex", gap: "16px", flexWrap: "wrap", marginBottom: "24px" }}>
                {windows.map(window => (
                  <button
                    key={window}
                    onClick={() => handleWindowChange(window)}
                    style={{
                      flex: 1,
                      padding: "16px 24px",
                      background: selectedWindow === window 
                        ? "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
                        : "linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%)",
                      color: selectedWindow === window ? "white" : "#374151",
                      border: "none",
                      borderRadius: "12px",
                      fontSize: "16px",
                      fontWeight: "600",
                      cursor: "pointer",
                      transition: "all 0.3s ease",
                      boxShadow: selectedWindow === window 
                        ? "0 4px 15px rgba(102, 126, 234, 0.4)" 
                        : "0 2px 4px rgba(0,0,0,0.05)",
                      transform: selectedWindow === window ? "scale(1.02)" : "scale(1)"
                    }}
                  >
                    {window} Days
                    <span style={{ 
                      display: "block", 
                      fontSize: "12px", 
                      marginTop: "4px",
                      opacity: 0.9
                    }}>
                      {window === 30 && "📊 Short-term"}
                      {window === 60 && "📈 Mid-term"}
                      {window === 90 && "🚀 Long-term"}
                    </span>
                  </button>
                ))}
              </div>
              
              {/* ROI Trend Chart */}
              {dashboard.roi_by_horizon && (
                <ROITrendChart roiData={dashboard.roi_by_horizon} selectedWindow={selectedWindow} />
              )}
            </div>

            {/* Current Window Badge */}
            <div style={{
              display: "inline-block",
              background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
              color: "white",
              padding: "8px 20px",
              borderRadius: "30px",
              marginBottom: "24px",
              fontSize: "14px",
              fontWeight: "600",
              boxShadow: "0 2px 8px rgba(102, 126, 234, 0.3)"
            }}>
              📊 {selectedWindow}-Day Window Analysis
            </div>

            {/* ROI Metrics Cards with Hover Effects */}
            <ROIMetrics interventionMetrics={windowData?.intervention_metrics} />

            {/* Two Column Layout for Charts */}
            <div style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(500px, 1fr))",
              gap: "24px",
              marginBottom: "24px"
            }}>
              {/* Risk Tier Distribution */}
              {windowData?.tier_distribution && (
                <RiskTierChart tierDistribution={windowData.tier_distribution} />
              )}
              
              {/* Model Health Metrics */}
              {dashboard.ml_metrics && (
                <ModelHealthMetrics mlMetrics={dashboard.ml_metrics} trainingAuc={dashboard.training_auc} />
              )}
            </div>

            {/* Catastrophe Metrics */}
            {windowData?.catastrophe_metrics && (
              <CatastropheMetrics catastropheMetrics={windowData.catastrophe_metrics} />
            )}

            {/* Recommendation Card */}
            {windowData?.recommended_decision && (
              <RecommendationCard recommendation={windowData.recommended_decision} />
            )}

            {/* Migration Summary with Animation */}
            {dashboard.migration_summary && Object.keys(dashboard.migration_summary).length > 0 && (
              <div style={{
                background: "rgba(255, 255, 255, 0.95)",
                backdropFilter: "blur(10px)",
                borderRadius: "20px",
                padding: "28px",
                marginBottom: "24px",
                boxShadow: "0 8px 32px rgba(0,0,0,0.1)"
              }}>
                <h3 style={{ marginBottom: "20px", fontSize: "20px", fontWeight: "600", color: "#1f2937" }}>
                  🔄 Risk Migration Trends
                </h3>
                <div style={{ display: "grid", gap: "16px" }}>
                  {Object.entries(dashboard.migration_summary).map(([period, summary], index) => (
                    <div 
                      key={period} 
                      style={{
                        padding: "20px",
                        background: "linear-gradient(135deg, #f9fafb 0%, #ffffff 100%)",
                        borderRadius: "16px",
                        border: "1px solid #e5e7eb",
                        transition: "all 0.3s ease",
                        animation: `slideIn 0.5s ease-out ${index * 0.1}s both`
                      }}
                      onMouseEnter={(e) => {
                        e.currentTarget.style.transform = "translateX(8px)";
                        e.currentTarget.style.boxShadow = "0 4px 12px rgba(0,0,0,0.1)";
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.transform = "translateX(0)";
                        e.currentTarget.style.boxShadow = "none";
                      }}
                    >
                      <p style={{ fontWeight: "700", marginBottom: "16px", fontSize: "16px", color: "#374151" }}>
                        {period.replace("_to_", " → ")} Days
                      </p>
                      <div style={{ display: "flex", gap: "32px", flexWrap: "wrap" }}>
                        <div>
                          <p style={{ fontSize: "12px", color: "#6b7280" }}>New High Risk</p>
                          <p style={{ fontSize: "24px", fontWeight: "bold", color: "#ef4444" }}>+{summary.net_new_high_risk_members}</p>
                        </div>
                        <div>
                          <p style={{ fontSize: "12px", color: "#6b7280" }}>Recovered</p>
                          <p style={{ fontSize: "24px", fontWeight: "bold", color: "#10b981" }}>↓{summary.net_recovered_members}</p>
                        </div>
                        <div>
                          <p style={{ fontSize: "12px", color: "#6b7280" }}>Upward Moves</p>
                          <p style={{ fontSize: "20px", fontWeight: "bold", color: "#f59e0b" }}>📈 {summary.total_upward_moves}</p>
                        </div>
                        <div>
                          <p style={{ fontSize: "12px", color: "#6b7280" }}>Downward Moves</p>
                          <p style={{ fontSize: "20px", fontWeight: "bold", color: "#10b981" }}>📉 {summary.total_downward_moves}</p>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
                <style>{`
                  @keyframes slideIn {
                    from {
                      opacity: 0;
                      transform: translateX(-20px);
                    }
                    to {
                      opacity: 1;
                      transform: translateX(0);
                    }
                  }
                `}</style>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
