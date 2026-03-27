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
    setUploadComplete(true);
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
      background: "rgba(255,255,255,0.95)",
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
        borderTopColor: "#0891b2",
        borderRadius: "50%",
        animation: "spin 1s linear infinite"
      }} />
      <p style={{ color: "#4b5563" }}>Loading dashboard data...</p>
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
      background: "linear-gradient(135deg, #e0f2fe 0%, #f0f9ff 50%, #d9f0ec 100%)",
      fontFamily: "'Inter', system-ui, -apple-system, sans-serif",
      position: "relative",
      overflowX: "hidden"
    }}>
      {/* Medical Background Pattern */}
      <div style={{
        position: "fixed",
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        pointerEvents: "none",
        zIndex: 0,
        opacity: 0.08
      }}>
        <svg width="100%" height="100%" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <pattern id="medicalCross" patternUnits="userSpaceOnUse" width="50" height="50">
              <path d="M25 10 L25 40 M10 25 L40 25" stroke="#0f766e" strokeWidth="1.5" strokeLinecap="round"/>
              <circle cx="25" cy="25" r="6" stroke="#0f766e" strokeWidth="1" fill="none"/>
            </pattern>
            <pattern id="medicalHeart" patternUnits="userSpaceOnUse" width="70" height="70">
              <path d="M35 22 L38 19 L41 22 L44 19 L47 22 L44 30 L41 38 L38 30 L35 22 Z" fill="#0891b2" opacity="0.4"/>
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#medicalCross)" />
          <rect width="100%" height="100%" fill="url(#medicalHeart)" />
        </svg>
      </div>

      {/* Floating Medical Icons */}
      <div style={{
        position: "fixed",
        bottom: "30px",
        right: "30px",
        fontSize: "35px",
        opacity: 0.12,
        pointerEvents: "none",
        zIndex: 0
      }}>
         
      </div>
      <div style={{
        position: "fixed",
        top: "120px",
        left: "20px",
        fontSize: "55px",
        opacity: 0.08,
        pointerEvents: "none",
        zIndex: 0,
        transform: "rotate(-10deg)"
      }}>
        🫀
      </div>
      <div style={{
        position: "fixed",
        bottom: "150px",
        left: "40px",
        fontSize: "45px",
        opacity: 0.08,
        pointerEvents: "none",
        zIndex: 0
      }}>
        💉
      </div>
      <div style={{
        position: "fixed",
        top: "50%",
        right: "20px",
        fontSize: "50px",
        opacity: 0.06,
        pointerEvents: "none",
        zIndex: 0,
        transform: "translateY(-50%)"
      }}>
        📊
      </div>

      {/* Header with Medical Theme */}
      <div style={{
        background: "linear-gradient(135deg, #0f766e 0%, #0891b2 50%, #06b6d4 100%)",
        padding: "28px 32px",
        marginBottom: "32px",
        boxShadow: "0 4px 20px rgba(0,0,0,0.1)",
        position: "relative",
        zIndex: 1
      }}>
        <div style={{ maxWidth: "1200px", margin: "0 auto" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "16px", marginBottom: "8px" }}>
            <span style={{ fontSize: "40px", filter: "drop-shadow(0 2px 4px rgba(0,0,0,0.1))" }}>🏥</span>
            <div>
              <h1 style={{ 
                fontSize: "28px", 
                fontWeight: "700",
                color: "white",
                margin: 0,
                textShadow: "0 2px 4px rgba(0,0,0,0.1)"
              }}>
                Member Risk Stratification Dashboard
              </h1>
              <p style={{ color: "rgba(255,255,255,0.9)", fontSize: "14px", marginTop: "4px" }}>
                AI-powered risk prediction and ROI simulation for proactive care management
              </p>
            </div>
          </div>
          {/* Medical Decorative Line */}
          <div style={{
            marginTop: "16px",
            display: "flex",
            gap: "8px"
          }}>
            {[...Array(6)].map((_, i) => (
              <div key={i} style={{
                width: "50px",
                height: "3px",
                background: "rgba(255,255,255,0.3)",
                borderRadius: "2px"
              }} />
            ))}
          </div>
        </div>
      </div>

      <div style={{ maxWidth: "1200px", margin: "0 auto", padding: "0 32px 32px", position: "relative", zIndex: 1 }}>
        {/* Upload Section - Medical Card Style */}
        {!uploadComplete && !loading && (
          <div style={{
            background: "rgba(255, 255, 255, 0.95)",
            backdropFilter: "blur(10px)",
            borderRadius: "24px",
            padding: "40px",
            marginBottom: "32px",
            boxShadow: "0 8px 32px rgba(0,0,0,0.08)",
            textAlign: "center",
            transition: "all 0.3s ease",
            border: "1px solid rgba(8,145,178,0.2)"
          }}>
            <div style={{
              width: "80px",
              height: "80px",
              background: "linear-gradient(135deg, #0f766e 0%, #0891b2 100%)",
              borderRadius: "40px",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              margin: "0 auto 20px",
              boxShadow: "0 8px 20px rgba(8,145,178,0.3)"
            }}>
              <span style={{ fontSize: "40px" }}>📊</span>
            </div>
            <h2 style={{ fontSize: "24px", fontWeight: "600", marginBottom: "12px", color: "#0f766e" }}>
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
            background: "rgba(255,255,255,0.95)",
            borderRadius: "20px",
            padding: "60px",
            textAlign: "center"
          }}>
            <div style={{ 
              width: "60px", 
              height: "60px", 
              margin: "0 auto 20px",
              border: "3px solid #e5e7eb",
              borderTopColor: "#0891b2",
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

        {/* Dashboard Content */}
        {dashboard && !loading && uploadComplete && (
          <div>
            {/* Reset Button */}
            <div style={{ textAlign: "right", marginBottom: "20px" }}>
              <button
                onClick={handleReset}
                style={{
                  padding: "8px 20px",
                  background: "rgba(255,255,255,0.9)",
                  backdropFilter: "blur(10px)",
                  color: "#0f766e",
                  border: "1px solid rgba(8,145,178,0.3)",
                  borderRadius: "30px",
                  cursor: "pointer",
                  fontSize: "14px",
                  fontWeight: "500",
                  transition: "all 0.2s ease",
                  display: "inline-flex",
                  alignItems: "center",
                  gap: "6px"
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = "white";
                  e.currentTarget.style.transform = "translateY(-1px)";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = "rgba(255,255,255,0.9)";
                  e.currentTarget.style.transform = "translateY(0)";
                }}
              >
                <span>↺</span> Upload New File
              </button>
            </div>

            {/* 1. Executive Summary - Medical Theme */}
            <div style={{
              background: "linear-gradient(135deg, #0f766e 0%, #0891b2 100%)",
              borderRadius: "24px",
              padding: "32px",
              marginBottom: "24px",
              boxShadow: "0 8px 32px rgba(8,145,178,0.3)",
              position: "relative",
              overflow: "hidden"
            }}>
              <div style={{
                position: "absolute",
                top: -30,
                right: -30,
                width: "150px",
                height: "150px",
                background: "radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%)",
                borderRadius: "50%"
              }} />
              <div style={{ position: "relative", zIndex: 1 }}>
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
            </div>

            {/* 2. Model Health Metrics */}
            {mlMetrics && (
              <Suspense fallback={<LoadingSkeleton />}>
                <div style={{ marginBottom: "32px" }}>
                  <ModelHealthMetrics mlMetrics={mlMetrics} trainingAuc={dashboard.training_auc} />
                </div>
              </Suspense>
            )}

            {/* 3. Time Window Selector */}
            <div style={{
              background: "rgba(255,255,255,0.95)",
              backdropFilter: "blur(10px)",
              borderRadius: "20px",
              padding: "24px",
              marginBottom: "32px",
              boxShadow: "0 2px 8px rgba(0,0,0,0.05)",
              border: "1px solid rgba(8,145,178,0.2)"
            }}>
              <h3 style={{ marginBottom: "16px", fontSize: "18px", fontWeight: "600", color: "#0f766e", display: "flex", alignItems: "center", gap: "8px" }}>
                <span>⏱️</span> Analysis Time Horizon
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
                        ? "linear-gradient(135deg, #0f766e 0%, #0891b2 100%)"
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
              background: "linear-gradient(135deg, #0f766e 0%, #0891b2 100%)",
              color: "white",
              padding: "6px 20px",
              borderRadius: "30px",
              marginBottom: "20px",
              fontSize: "13px",
              fontWeight: "600",
              boxShadow: "0 2px 8px rgba(8,145,178,0.3)"
            }}>
              📊 {selectedWindow}-Day Window Analysis
            </div>

            {/* 4. ROI Metrics */}
            {windowData && (
              <Suspense fallback={<LoadingSkeleton />}>
                <ROIMetrics interventionMetrics={windowData.intervention_metrics} />
              </Suspense>
            )}

            {/* 5. Risk Tier Distribution & Catastrophe Metrics */}
            {windowData && (
              <div style={{
                display: "grid",
                gridTemplateColumns: "repeat(2, 1fr)",
                gap: "24px",
                marginBottom: "24px"
              }}>
                <Suspense fallback={<LoadingSkeleton />}>
                  <RiskTierChart 
                    tierDistribution={windowData.tier_distribution}
                    membersByTier={dashboard.members_by_tier || {}}
                  />
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
                background: "rgba(255,255,255,0.95)",
                backdropFilter: "blur(10px)",
                borderRadius: "20px",
                padding: "24px",
                marginTop: "24px",
                boxShadow: "0 2px 8px rgba(0,0,0,0.05)",
                border: "1px solid rgba(8,145,178,0.2)"
              }}>
                <h3 style={{ marginBottom: "20px", fontSize: "18px", fontWeight: "600", color: "#0f766e", display: "flex", alignItems: "center", gap: "8px" }}>
                  <span>🔄</span> Risk Migration Trends
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
