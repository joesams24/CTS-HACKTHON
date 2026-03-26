// src/pages/AdminDashboard.jsx
import { useState, useMemo, useCallback, lazy, Suspense } from "react";
import FileUploader from "../components/FileUploader";


// Lazy load components for better performance
const RiskTierChart = lazy(() => import("../components/RiskTierChart"));
const ROIMetrics = lazy(() => import("../components/ROIMetrics"));
const CatastropheMetrics = lazy(() => import("../components/CatastropheMetrics"));
const RecommendationCard = lazy(() => import("../components/RecommendationCard"));
const ModelHealthMetrics = lazy(() => import("../components/ModelHealthMetrics"));
const ROITrendChart = lazy(() => import("../components/ROITrendChart"));

export default function AdminDashboard() {
  const [dashboard, setDashboard] = useState(null);
  const [selectedWindow, setSelectedWindow] = useState(30);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [uploadComplete, setUploadComplete] = useState(false);

  const windows = useMemo(() => [30, 60, 90], []);

  // Memoize window data
  const windowData = useMemo(() => {
    if (!dashboard || !dashboard.windows) return null;
    return dashboard.windows[selectedWindow];
  }, [dashboard, selectedWindow]);

  // Memoize ROI data for chart
  const roiData = useMemo(() => {
    if (!dashboard?.roi_by_horizon) return null;
    return dashboard.roi_by_horizon;
  }, [dashboard]);

  // Memoize ML metrics
  const mlMetrics = useMemo(() => {
    if (!dashboard?.ml_metrics) return null;
    return dashboard.ml_metrics;
  }, [dashboard]);

  const handleWindowChange = useCallback((window) => {
    setSelectedWindow(window);
  }, []);

  const handleDataReceived = useCallback((data) => {
    setDashboard(data);
    setError(null);
    setSelectedWindow(30);
    setUploadComplete(true); // Hide upload area after successful upload
  }, []);

  const handleReset = useCallback(() => {
    setDashboard(null);
    setUploadComplete(false);
    setSelectedWindow(30);
    setError(null);
  }, []);

  // Loading skeleton
  const LoadingSkeleton = () => (
    <div style={{
      background: "white",
      borderRadius: "20px",
      padding: "40px",
      textAlign: "center",
      animation: "pulse 1.5s ease-in-out infinite"
    }}>
      <div style={{ 
        width: "60px", 
        height: "60px", 
        margin: "0 auto 20px",
        border: "3px solid #e5e7eb",
        borderTopColor: "#667eea",
        borderRadius: "50%",
        animation: "spin 1s linear infinite"
      }} />
      <p style={{ color: "#6b7280" }}>Loading dashboard data...</p>
      <style>{`
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.7; }
        }
      `}</style>
    </div>
  );

  return (
    <div style={{ 
      minHeight: "100vh", 
      background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
      fontFamily: "'Inter', system-ui, -apple-system, sans-serif"
    }}>
      {/* Header */}
      <div style={{
        backgroundColor: "white",
        padding: "24px 32px",
        marginBottom: "32px",
        boxShadow: "0 2px 8px rgba(0,0,0,0.05)"
      }}>
        <div style={{ maxWidth: "1200px", margin: "0 auto" }}>
          <h1 style={{ 
            fontSize: "28px", 
            fontWeight: "700",
            background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
            marginBottom: "4px"
          }}>
            🏥 Member Risk Stratification Dashboard
          </h1>
          <p style={{ color: "#6b7280", fontSize: "14px" }}>
            AI-powered risk prediction and ROI simulation
          </p>
        </div>
      </div>

      <div style={{ maxWidth: "1200px", margin: "0 auto", padding: "0 32px 32px" }}>
        {/* Upload Section - Only show if upload not complete */}
        {!uploadComplete && !loading && (
          <div style={{
            background: "white",
            borderRadius: "20px",
            padding: "40px",
            marginBottom: "32px",
            boxShadow: "0 4px 20px rgba(0,0,0,0.08)",
            textAlign: "center",
            transition: "all 0.3s ease"
          }}>
            <div style={{ marginBottom: "24px" }}>
              <span style={{ fontSize: "48px" }}>📊</span>
            </div>
            <h2 style={{ fontSize: "24px", fontWeight: "600", marginBottom: "12px", color: "#1f2937" }}>
              Upload Member Data
            </h2>
            <p style={{ color: "#6b7280", marginBottom: "24px", fontSize: "14px" }}>
              Upload a CSV file to analyze member risk and generate ROI insights
            </p>
            <FileUploader 
              onResult={handleDataReceived}
              onLoading={setLoading}
              onError={setError}
            />
            {error && (
              <div style={{ 
                marginTop: "20px", 
                padding: "12px", 
                background: "#fee2e2", 
                borderRadius: "12px", 
                color: "#dc2626",
                fontSize: "14px"
              }}>
                ⚠️ {error}
              </div>
            )}
          </div>
        )}

        {/* Loading State */}
        {loading && (
          <div style={{
            background: "white",
            borderRadius: "20px",
            padding: "60px",
            textAlign: "center"
          }}>
            <div style={{ 
              width: "60px", 
              height: "60px", 
              margin: "0 auto 20px",
              border: "3px solid #e5e7eb",
              borderTopColor: "#667eea",
              borderRadius: "50%",
              animation: "spin 1s linear infinite"
            }} />
            <p style={{ fontSize: "18px", color: "#4b5563", marginBottom: "8px" }}>
              Processing your data...
            </p>
            <p style={{ fontSize: "14px", color: "#9ca3af" }}>
              Training risk model and simulating ROI
            </p>
            <style>{`
              @keyframes spin {
                to { transform: rotate(360deg); }
              }
            `}</style>
          </div>
        )}

        {/* Dashboard Content - Only show after upload complete */}
        {dashboard && !loading && uploadComplete && (
          <div>
            {/* Reset Button */}
            <div style={{ textAlign: "right", marginBottom: "20px" }}>
              <button
                onClick={handleReset}
                style={{
                  padding: "8px 20px",
                  background: "rgba(255,255,255,0.2)",
                  backdropFilter: "blur(10px)",
                  color: "white",
                  border: "1px solid rgba(255,255,255,0.3)",
                  borderRadius: "20px",
                  cursor: "pointer",
                  fontSize: "14px",
                  fontWeight: "500",
                  transition: "all 0.2s ease"
                }}
                onMouseEnter={(e) => e.currentTarget.style.background = "rgba(255,255,255,0.3)"}
                onMouseLeave={(e) => e.currentTarget.style.background = "rgba(255,255,255,0.2)"}
              >
                ↺ Upload New File
              </button>
            </div>

            {/* 1. Executive Summary - First */}
            <div style={{
              background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
              borderRadius: "20px",
              padding: "32px",
              marginBottom: "24px",
              boxShadow: "0 8px 32px rgba(102, 126, 234, 0.3)"
            }}>
              <div style={{ display: "flex", alignItems: "center", marginBottom: "16px" }}>
                <span style={{ fontSize: "28px", marginRight: "12px" }}>📋</span>
                <h2 style={{ fontSize: "20px", fontWeight: "600", color: "white", margin: 0 }}>
                  Executive Summary
                </h2>
              </div>
              <p style={{ fontSize: "15px", color: "rgba(255,255,255,0.95)", lineHeight: "1.6", marginBottom: "24px" }}>
                {dashboard.executive_summary}
              </p>
              <div style={{ 
                display: "grid", 
                gridTemplateColumns: "repeat(3, 1fr)", 
                gap: "24px",
                borderTop: "1px solid rgba(255,255,255,0.2)",
                paddingTop: "24px"
              }}>
                <div>
                  <p style={{ fontSize: "12px", color: "rgba(255,255,255,0.8)", marginBottom: "4px" }}>
                    Total Members
                  </p>
                  <p style={{ fontSize: "28px", fontWeight: "bold", color: "white" }}>
                    {dashboard.population_size?.toLocaleString() || 0}
                  </p>
                </div>
                <div>
                  <p style={{ fontSize: "12px", color: "rgba(255,255,255,0.8)", marginBottom: "4px" }}>
                    Mean Risk Score
                  </p>
                  <p style={{ fontSize: "28px", fontWeight: "bold", color: "white" }}>
                    {mlMetrics?.mean_predicted_risk ? (mlMetrics.mean_predicted_risk * 100).toFixed(1) : "0"}%
                  </p>
                </div>
                <div>
                  <p style={{ fontSize: "12px", color: "rgba(255,255,255,0.8)", marginBottom: "4px" }}>
                    High Risk Population
                  </p>
                  <p style={{ fontSize: "28px", fontWeight: "bold", color: "white" }}>
                    {mlMetrics?.high_risk_fraction ? (mlMetrics.high_risk_fraction * 100).toFixed(1) : "0"}%
                  </p>
                </div>
              </div>
            </div>

            {/* 2. Model Health Metrics - Second */}
            {mlMetrics && (
              <Suspense fallback={<LoadingSkeleton />}>
                <div style={{ marginBottom: "32px" }}>
                  <ModelHealthMetrics mlMetrics={mlMetrics} trainingAuc={dashboard.training_auc} />
                </div>
              </Suspense>
            )}

            {/* 3. Time Window Selector */}
            <div style={{
              background: "white",
              borderRadius: "20px",
              padding: "24px",
              marginBottom: "32px",
              boxShadow: "0 2px 8px rgba(0,0,0,0.05)"
            }}>
              <h3 style={{ marginBottom: "16px", fontSize: "18px", fontWeight: "600", color: "#1f2937" }}>
                ⏱️ Analysis Time Horizon
              </h3>
              <div style={{ display: "flex", gap: "12px", flexWrap: "wrap", marginBottom: "24px" }}>
                {windows.map(window => (
                  <button
                    key={window}
                    onClick={() => handleWindowChange(window)}
                    style={{
                      flex: 1,
                      padding: "12px 20px",
                      background: selectedWindow === window 
                        ? "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
                        : "#f3f4f6",
                      color: selectedWindow === window ? "white" : "#374151",
                      border: "none",
                      borderRadius: "12px",
                      fontSize: "14px",
                      fontWeight: "600",
                      cursor: "pointer",
                      transition: "all 0.2s ease"
                    }}
                  >
                    {window} Days
                  </button>
                ))}
              </div>
              
              {/* ROI Trend Chart */}
              {roiData && (
                <Suspense fallback={<div style={{ height: "200px" }} />}>
                  <ROITrendChart roiData={roiData} selectedWindow={selectedWindow} />
                </Suspense>
              )}
            </div>

            {/* Current Window Badge */}
            <div style={{
              display: "inline-block",
              background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
              color: "white",
              padding: "6px 20px",
              borderRadius: "30px",
              marginBottom: "20px",
              fontSize: "13px",
              fontWeight: "600"
            }}>
              📊 {selectedWindow}-Day Window Analysis
            </div>

            {/* 4. ROI Metrics */}
            {windowData && (
              <Suspense fallback={<LoadingSkeleton />}>
                <ROIMetrics interventionMetrics={windowData.intervention_metrics} />
              </Suspense>
            )}

            {/* 5. Risk Tier Distribution & Catastrophe Metrics - Two Column Layout */}
            {windowData && (
              <div style={{
                display: "grid",
                gridTemplateColumns: "repeat(2, 1fr)",
                gap: "24px",
                marginBottom: "24px"
              }}>
                <Suspense fallback={<LoadingSkeleton />}>
                  <RiskTierChart tierDistribution={windowData.tier_distribution} />
                </Suspense>
                
                <Suspense fallback={<LoadingSkeleton />}>
                  <CatastropheMetrics catastropheMetrics={windowData.catastrophe_metrics} />
                </Suspense>
              </div>
            )}

            {/* 6. Recommendation Card */}
            {windowData?.recommended_decision && (
              <Suspense fallback={<LoadingSkeleton />}>
                <RecommendationCard recommendation={windowData.recommended_decision} />
              </Suspense>
            )}

            {/* 7. Risk Migration Trends */}
            {dashboard.migration_summary && Object.keys(dashboard.migration_summary).length > 0 && (
              <div style={{
                background: "white",
                borderRadius: "20px",
                padding: "24px",
                marginTop: "24px",
                boxShadow: "0 2px 8px rgba(0,0,0,0.05)"
              }}>
                <h3 style={{ marginBottom: "20px", fontSize: "18px", fontWeight: "600", color: "#1f2937" }}>
                  🔄 Risk Migration Trends
                </h3>
                <div style={{ display: "grid", gap: "16px" }}>
                  {Object.entries(dashboard.migration_summary).map(([period, summary]) => (
                    <div key={period} style={{
                      padding: "16px",
                      background: "#f9fafb",
                      borderRadius: "12px",
                      border: "1px solid #e5e7eb"
                    }}>
                      <p style={{ fontWeight: "600", marginBottom: "12px", fontSize: "14px", color: "#374151" }}>
                        {period.replace("_to_", " → ")} Days
                      </p>
                      <div style={{ display: "flex", gap: "24px", flexWrap: "wrap", fontSize: "13px" }}>
                        <span>⬆️ New High Risk: <strong>{summary.net_new_high_risk_members}</strong></span>
                        <span>⬇️ Recovered: <strong>{summary.net_recovered_members}</strong></span>
                        <span>📈 Upward Moves: <strong>{summary.total_upward_moves}</strong></span>
                        <span>📉 Downward Moves: <strong>{summary.total_downward_moves}</strong></span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
