// src/components/ROITrendChart.jsx
import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, ComposedChart, ReferenceLine } from 'recharts';

const ROITrendChart = React.memo(({ roiData, selectedWindow, onWindowChange }) => {
  const [hoveredPoint, setHoveredPoint] = useState(null);
  const [animated, setAnimated] = useState(false);

  useEffect(() => {
    setAnimated(true);
    const timer = setTimeout(() => setAnimated(false), 1500);
    return () => clearTimeout(timer);
  }, [selectedWindow, roiData]);

  if (!roiData) return null;

  // Prepare chart data
  const chartData = Object.entries(roiData).map(([window, roi]) => ({
    days: parseInt(window),
    window: `${window} Days`,
    roi: roi,
    isSelected: parseInt(window) === selectedWindow,
    roiDisplay: `${roi}%`,
    color: parseInt(window) === 30 ? "#3b82f6" : parseInt(window) === 60 ? "#f59e0b" : "#10b981"
  })).sort((a, b) => a.days - b.days);

  // Calculate additional metrics
  const maxRoi = Math.max(...chartData.map(d => d.roi));
  const avgRoi = chartData.reduce((sum, d) => sum + d.roi, 0) / chartData.length;
  const projectedRoi = chartData[chartData.length - 1]?.roi ? (chartData[chartData.length - 1].roi * 1.2).toFixed(1) : 0;

  // Custom Tooltip with Animation
  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div style={{
          background: "linear-gradient(135deg, rgba(255,255,255,0.98) 0%, rgba(255,255,255,0.95) 100%)",
          padding: "14px 20px",
          borderRadius: "16px",
          boxShadow: "0 8px 32px rgba(0,0,0,0.15)",
          border: `2px solid ${data.color}`,
          backdropFilter: "blur(10px)",
          animation: "tooltipFadeIn 0.2s ease-out"
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "8px" }}>
            <span style={{ fontSize: "24px" }}>
              {data.days === 30 ? "📊" : data.days === 60 ? "📈" : "🚀"}
            </span>
            <div>
              <p style={{ fontWeight: "700", fontSize: "14px", color: "#1f2937", margin: 0 }}>
                {data.window}
              </p>
              <p style={{ fontSize: "10px", color: "#6b7280", margin: "2px 0 0 0" }}>
                Time Horizon
              </p>
            </div>
          </div>
          <div style={{ marginBottom: "8px" }}>
            <span style={{ fontSize: "36px", fontWeight: "bold", color: data.color }}>
              {data.roi}%
            </span>
            <span style={{ fontSize: "12px", color: "#6b7280" }}> ROI</span>
          </div>
          <div style={{
            width: "100%",
            height: "4px",
            background: "#e5e7eb",
            borderRadius: "2px",
            marginBottom: "8px",
            overflow: "hidden"
          }}>
            <div style={{
              width: `${(data.roi / maxRoi) * 100}%`,
              height: "100%",
              background: data.color,
              borderRadius: "2px",
              animation: "progressFill 1s ease-out"
            }} />
          </div>
          <p style={{ fontSize: "11px", color: "#6b7280", margin: "8px 0 0 0", lineHeight: "1.4" }}>
            {data.days === 30 && "Short-term ROI shows immediate impact from targeted interventions"}
            {data.days === 60 && "Mid-term ROI demonstrates growing benefits from expanded coverage"}
            {data.days === 90 && "Long-term ROI maximizes value through compounded savings"}
          </p>
        </div>
      );
    }
    return null;
  };

  // Custom Dot - Clean version without unwanted animations
  const CustomDot = (props) => {
    const { cx, cy, payload, index } = props;
    const isSelected = payload.isSelected;
    const size = isSelected ? 8 : 6;
    
    return (
      <g>
        {/* Static glow effect for selected point (no animation) */}
        {isSelected && (
          <circle
            cx={cx}
            cy={cy}
            r={size + 4}
            fill={payload.color}
            opacity={0.15}
          />
        )}
        {/* Main dot */}
        <circle
          cx={cx}
          cy={cy}
          r={size}
          fill={payload.color}
          stroke="white"
          strokeWidth={2}
          style={{
            cursor: "pointer",
            transition: "all 0.2s ease",
            transform: hoveredPoint === index ? "scale(1.15)" : "scale(1)"
          }}
          onMouseEnter={() => setHoveredPoint(index)}
          onMouseLeave={() => setHoveredPoint(null)}
          onClick={() => onWindowChange && onWindowChange(payload.days)}
        />
      </g>
    );
  };

  return (
    <div style={{
      background: "linear-gradient(135deg, #ffffff 0%, #faf5ff 100%)",
      borderRadius: "24px",
      padding: "24px",
      boxShadow: "0 8px 32px rgba(0,0,0,0.08)",
      marginTop: "20px",
      border: "1px solid rgba(102, 126, 234, 0.15)",
      position: "relative",
      overflow: "hidden"
    }}>
      <style>{`
        @keyframes shimmer {
          0% {
            transform: translateX(-100%);
          }
          100% {
            transform: translateX(100%);
          }
        }
        
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
        
        @keyframes tooltipFadeIn {
          from {
            opacity: 0;
            transform: scale(0.95);
          }
          to {
            opacity: 1;
            transform: scale(1);
          }
        }
        
        @keyframes progressFill {
          from {
            width: 0%;
          }
          to {
            width: ${({ width }) => width}%;
          }
        }
      `}</style>

      {/* Decorative Elements - Static */}
      <div style={{
        position: "absolute",
        top: -30,
        right: -30,
        width: "150px",
        height: "150px",
        background: "radial-gradient(circle, rgba(102,126,234,0.05) 0%, transparent 70%)",
        borderRadius: "50%",
        pointerEvents: "none"
      }} />
      <div style={{
        position: "absolute",
        bottom: -20,
        left: -20,
        width: "120px",
        height: "120px",
        background: "radial-gradient(circle, rgba(16,185,129,0.05) 0%, transparent 70%)",
        borderRadius: "50%",
        pointerEvents: "none"
      }} />

      {/* Header with Stats */}
      <div style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "flex-start",
        marginBottom: "24px",
        flexWrap: "wrap",
        gap: "16px"
      }}>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "8px" }}>
            <span style={{ fontSize: "28px" }}></span>
            <h3 style={{ fontSize: "18px", fontWeight: "700", color: "#1f2937", margin: 0 }}>
              ROI Trend Across Time Horizons
            </h3>
            <span style={{
              background: "linear-gradient(135deg, #10b981 0%, #059669 100%)",
              padding: "2px 10px",
              borderRadius: "20px",
              fontSize: "10px",
              fontWeight: "600",
              color: "white"
            }}>
              REAL-TIME
            </span>
          </div>
          <p style={{ fontSize: "12px", color: "#6b7280", margin: 0 }}>
            Interactive trend analysis with projected ROI values
          </p>
        </div>
        
        {/* Quick Stats */}
        <div style={{ display: "flex", gap: "12px" }}>
          <div style={{
            background: "linear-gradient(135deg, #f3f4f6 0%, #ffffff 100%)",
            padding: "8px 16px",
            borderRadius: "12px",
            textAlign: "center",
            border: "1px solid #e5e7eb"
          }}>
            <p style={{ fontSize: "10px", color: "#6b7280", margin: 0 }}>Average ROI</p>
            <p style={{ fontSize: "18px", fontWeight: "bold", color: "#667eea", margin: "2px 0 0 0" }}>
              {avgRoi.toFixed(1)}%
            </p>
          </div>
          <div style={{
            background: "linear-gradient(135deg, #f3f4f6 0%, #ffffff 100%)",
            padding: "8px 16px",
            borderRadius: "12px",
            textAlign: "center",
            border: "1px solid #e5e7eb"
          }}>
            <p style={{ fontSize: "10px", color: "#6b7280", margin: 0 }}>Max ROI</p>
            <p style={{ fontSize: "18px", fontWeight: "bold", color: "#10b981", margin: "2px 0 0 0" }}>
              {maxRoi}%
            </p>
          </div>
        </div>
      </div>

      {/* Main Chart */}
      <div style={{ height: "280px", width: "100%", marginBottom: "20px" }}>
        <ResponsiveContainer>
          <ComposedChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 10 }}>
            <defs>
              <linearGradient id="areaGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#667eea" stopOpacity={0.15}/>
                <stop offset="95%" stopColor="#667eea" stopOpacity={0}/>
              </linearGradient>
              <linearGradient id="lineGradient" x1="0" y1="0" x2="1" y2="0">
                <stop offset="0%" stopColor="#667eea"/>
                <stop offset="50%" stopColor="#8b5cf6"/>
                <stop offset="100%" stopColor="#a855f7"/>
              </linearGradient>
            </defs>
            
            <CartesianGrid 
              strokeDasharray="5 5" 
              stroke="#e5e7eb" 
              strokeWidth={1}
              horizontal={true}
              vertical={false}
            />
            
            <XAxis 
              dataKey="window" 
              stroke="#6b7280"
              tick={{ fontSize: 12, fill: "#6b7280", fontWeight: 500 }}
              axisLine={{ stroke: "#e5e7eb", strokeWidth: 1 }}
              tickLine={{ stroke: "#e5e7eb" }}
            />
            
            <YAxis 
              stroke="#6b7280"
              tick={{ fontSize: 12, fill: "#6b7280" }}
              axisLine={{ stroke: "#e5e7eb" }}
              tickLine={{ stroke: "#e5e7eb" }}
              unit="%"
              domain={[0, 'auto']}
            />
            
            <Tooltip content={<CustomTooltip />} cursor={{ stroke: "#667eea", strokeWidth: 1, strokeDasharray: "3 3" }} />
            
            <Area
              type="monotone"
              dataKey="roi"
              stroke="none"
              fill="url(#areaGradient)"
              isAnimationActive={animated}
              animationDuration={1000}
            />
            
            <Line
              type="monotone"
              dataKey="roi"
              stroke="url(#lineGradient)"
              strokeWidth={3}
              dot={<CustomDot />}
              activeDot={false}
              isAnimationActive={animated}
              animationDuration={1000}
              animationEasing="ease-out"
            />
            
            {/* Reference line for selected window - Static */}
            {chartData.find(d => d.isSelected) && (
              <ReferenceLine
                x={chartData.find(d => d.isSelected)?.window}
                stroke="#667eea"
                strokeWidth={1.5}
                strokeDasharray="5 5"
              />
            )}
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* Interactive ROI Cards */}
      <div style={{
        display: "flex",
        gap: "12px",
        marginBottom: "20px",
        flexWrap: "wrap"
      }}>
        {chartData.map((data, idx) => (
          <div
            key={idx}
            onClick={() => onWindowChange && onWindowChange(data.days)}
            style={{
              flex: 1,
              background: data.isSelected 
                ? `linear-gradient(135deg, ${data.color} 0%, ${data.color}dd 100%)`
                : "linear-gradient(135deg, #f9fafb 0%, #ffffff 100%)",
              padding: "12px",
              borderRadius: "12px",
              textAlign: "center",
              cursor: "pointer",
              transition: "all 0.2s ease",
              transform: hoveredPoint === idx ? "translateY(-2px)" : "translateY(0)",
              boxShadow: data.isSelected 
                ? `0 4px 12px ${data.color}40` 
                : "0 2px 4px rgba(0,0,0,0.05)",
              border: data.isSelected ? "none" : "1px solid #e5e7eb"
            }}
            onMouseEnter={() => setHoveredPoint(idx)}
            onMouseLeave={() => setHoveredPoint(null)}
          >
            <p style={{
              fontSize: "11px",
              fontWeight: "500",
              color: data.isSelected ? "rgba(255,255,255,0.9)" : "#6b7280",
              marginBottom: "4px"
            }}>
              {data.days} Days
            </p>
            <p style={{
              fontSize: "24px",
              fontWeight: "bold",
              color: data.isSelected ? "white" : data.color,
              margin: 0
            }}>
              {data.roi}%
            </p>
            <p style={{
              fontSize: "9px",
              color: data.isSelected ? "rgba(255,255,255,0.7)" : "#9ca3af",
              marginTop: "4px"
            }}>
              {data.days === 30 && "Quick Win"}
              {data.days === 60 && "Growth"}
              {data.days === 90 && "Maximum"}
            </p>
          </div>
        ))}
      </div>

      {/* Insight and Projection Section */}
      <div style={{
        background: "linear-gradient(135deg, #fef3c7 0%, #fde68a 100%)",
        borderRadius: "16px",
        padding: "14px 18px",
        position: "relative",
        overflow: "hidden"
      }}>
        <div style={{
          position: "absolute",
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: "linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent)",
          animation: "shimmer 2s infinite"
        }} />
        
        <div style={{ display: "flex", alignItems: "center", gap: "12px", flexWrap: "wrap" }}>
          <span style={{ fontSize: "24px" }}>💡</span>
          <div style={{ flex: 1 }}>
            <p style={{ fontSize: "12px", fontWeight: "600", color: "#92400e", margin: 0 }}>
              ROI Insight
            </p>
            <p style={{ fontSize: "11px", color: "#78350f", margin: "4px 0 0 0", lineHeight: "1.4" }}>
              {selectedWindow === 30 && `Short-term ROI of ${chartData.find(d => d.days === 30)?.roi}% shows quick returns from high-risk interventions.`}
              {selectedWindow === 60 && `Mid-term ROI of ${chartData.find(d => d.days === 60)?.roi}% demonstrates growing benefits from expanded coverage.`}
              {selectedWindow === 90 && `Long-term ROI of ${chartData.find(d => d.days === 90)?.roi}% maximizes value through compounded savings.`}
            </p>
          </div>
          <div style={{
            background: "rgba(255,255,255,0.5)",
            padding: "6px 12px",
            borderRadius: "10px",
            textAlign: "center"
          }}>
            <p style={{ fontSize: "10px", color: "#92400e", margin: 0 }}>Projected 120d</p>
            <p style={{ fontSize: "16px", fontWeight: "bold", color: "#92400e", margin: 0 }}>
              {projectedRoi}%
            </p>
          </div>
        </div>
      </div>

      {/* Footer Note */}
      <div style={{
        marginTop: "12px",
        display: "flex",
        justifyContent: "center",
        gap: "16px",
        fontSize: "10px",
        color: "#9ca3af"
      }}>
        <span>📊 Hover for details</span>
        <span>⚡ Click to select window</span>
        <span>📈 Data updates in real-time</span>
      </div>
    </div>
  );
});

export default ROITrendChart;
