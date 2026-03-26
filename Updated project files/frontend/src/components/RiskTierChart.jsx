// src/components/RiskTierChart.jsx
import React, { useState } from 'react';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from 'recharts';

const RiskTierChart = React.memo(({ tierDistribution, membersByTier = {} }) => {
  const [selectedTier, setSelectedTier] = useState(null);
  const [hoveredTier, setHoveredTier] = useState(null);
  const [showMemberDetails, setShowMemberDetails] = useState(false);

  if (!tierDistribution) return null;

  const tiers = [
    { name: "Very Low", color: "#10b981", gradient: "linear-gradient(135deg, #10b981 0%, #34d399 100%)", icon: "🟢", description: "Minimal risk, stable members", bgLight: "#d1fae5" },
    { name: "Low", color: "#3b82f6", gradient: "linear-gradient(135deg, #3b82f6 0%, #60a5fa 100%)", icon: "🔵", description: "Low risk, routine monitoring", bgLight: "#dbeafe" },
    { name: "Medium", color: "#f59e0b", gradient: "linear-gradient(135deg, #f59e0b 0%, #fbbf24 100%)", icon: "🟡", description: "Moderate risk, proactive care needed", bgLight: "#fed7aa" },
    { name: "High", color: "#ef4444", gradient: "linear-gradient(135deg, #ef4444 0%, #f87171 100%)", icon: "🟠", description: "High risk, intensive management", bgLight: "#fee2e2" },
    { name: "Very High", color: "#dc2626", gradient: "linear-gradient(135deg, #dc2626 0%, #ef4444 100%)", icon: "🔴", description: "Critical risk, immediate intervention", bgLight: "#fecaca" }
  ];

  const total = Object.values(tierDistribution).reduce((a, b) => a + b, 0);
  
  const pieData = tiers.map(tier => ({
    name: tier.name,
    value: tierDistribution[tier.name] || 0,
    percentage: total > 0 ? ((tierDistribution[tier.name] || 0) / total * 100).toFixed(1) : 0,
    color: tier.color,
    gradient: tier.gradient,
    icon: tier.icon,
    description: tier.description
  })).filter(d => d.value > 0);

  const getMemberDetailsForTier = (tierName) => {
    return membersByTier[tierName] || [];
  };

  const CustomTooltip = ({ active, payload }) => {
    const [tooltipPosition, setTooltipPosition] = useState({ x: 0, y: 0 });
    
    React.useEffect(() => {
      if (active && payload?.length) {
        const handleMouseMove = (e) => setTooltipPosition({ x: e.clientX, y: e.clientY });
        window.addEventListener('mousemove', handleMouseMove);
        return () => window.removeEventListener('mousemove', handleMouseMove);
      }
    }, [active]);

    if (active && payload?.length) {
      const data = payload[0].payload;
      return (
        <div style={{
          position: "fixed",
          left: tooltipPosition.x + 15,
          top: tooltipPosition.y - 80,
          background: "white",
          padding: "12px 16px",
          borderRadius: "12px",
          boxShadow: "0 4px 20px rgba(0,0,0,0.2)",
          border: `2px solid ${data.color}`,
          minWidth: "200px",
          zIndex: 1000,
          pointerEvents: "none"
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "8px" }}>
            <span style={{ fontSize: "20px" }}>{data.icon}</span>
            <span style={{ fontWeight: "700", fontSize: "14px", color: "#1f2937" }}>{data.name}</span>
          </div>
          <div style={{ marginBottom: "4px" }}>
            <span style={{ fontSize: "24px", fontWeight: "bold", color: data.color }}>
              {data.value.toLocaleString()}
            </span>
            <span style={{ fontSize: "14px", color: "#6b7280" }}> members</span>
          </div>
          <div style={{ marginBottom: "8px" }}>
            <span style={{ fontSize: "18px", fontWeight: "600", color: data.color }}>
              {data.percentage}%
            </span>
            <span style={{ fontSize: "12px", color: "#6b7280" }}> of population</span>
          </div>
          <div style={{ width: "100%", height: "4px", background: "#e5e7eb", borderRadius: "2px", marginBottom: "8px" }}>
            <div style={{ width: `${data.percentage}%`, height: "100%", background: data.gradient, borderRadius: "2px" }} />
          </div>
          <p style={{ fontSize: "11px", color: "#6b7280", margin: 0, lineHeight: "1.4" }}>{data.description}</p>
        </div>
      );
    }
    return null;
  };

  const CustomLegend = ({ payload }) => (
    <div style={{ display: "flex", flexWrap: "wrap", justifyContent: "center", gap: "12px", marginTop: "20px" }}>
      {payload.map((entry, index) => {
        const tier = tiers.find(t => t.name === entry.value);
        const count = tierDistribution[tier.name] || 0;
        const percentage = total > 0 ? ((count / total) * 100).toFixed(1) : 0;
        return (
          <div
            key={index}
            style={{
              display: "flex",
              alignItems: "center",
              gap: "8px",
              padding: "6px 12px",
              background: hoveredTier === tier.name ? tier.bgLight : "white",
              borderRadius: "30px",
              cursor: "pointer",
              transition: "all 0.2s ease",
              border: `1px solid ${hoveredTier === tier.name ? tier.color : "#e5e7eb"}`,
              transform: hoveredTier === tier.name ? "scale(1.05)" : "scale(1)"
            }}
            onMouseEnter={() => setHoveredTier(tier.name)}
            onMouseLeave={() => setHoveredTier(null)}
            onClick={() => {
              setSelectedTier(tier.name);
              setShowMemberDetails(true);
            }}
          >
            <div style={{ width: "12px", height: "12px", borderRadius: "50%", background: tier.gradient }} />
            <span style={{ fontSize: "13px", fontWeight: "500", color: "#374151" }}>{tier.icon} {tier.name}</span>
            <span style={{ fontSize: "12px", fontWeight: "700", color: tier.color, background: tier.bgLight, padding: "2px 8px", borderRadius: "20px" }}>{percentage}%</span>
          </div>
        );
      })}
    </div>
  );

  const MemberDetailsModal = ({ tier, onClose }) => {
    const [showCount, setShowCount] = useState(5);
    const members = getMemberDetailsForTier(tier);
    const tierInfo = tiers.find(t => t.name === tier);
    const totalMembers = tierDistribution[tier] || 0;
    const displayedMembers = members.slice(0, showCount);
    const hasMore = totalMembers > showCount;

    return (
      <div style={{
        position: "fixed",
        top: 0, left: 0, right: 0, bottom: 0,
        background: "rgba(0,0,0,0.5)",
        backdropFilter: "blur(4px)",
        zIndex: 2000,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: "20px"
      }} onClick={onClose}>
        <div style={{
          background: "white",
          borderRadius: "24px",
          maxWidth: "600px",
          width: "100%",
          maxHeight: "80vh",
          overflow: "auto",
          animation: "slideUp 0.3s ease-out"
        }} onClick={(e) => e.stopPropagation()}>
          <div style={{
            padding: "24px",
            background: tierInfo.gradient,
            borderRadius: "24px 24px 0 0",
            color: "white"
          }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <div>
                <span style={{ fontSize: "32px" }}>{tierInfo.icon}</span>
                <h3 style={{ fontSize: "24px", margin: "8px 0 0 0" }}>{tier} Risk Members</h3>
                <p style={{ fontSize: "14px", opacity: 0.9, margin: "4px 0 0 0" }}>Total: {totalMembers.toLocaleString()} members</p>
              </div>
              <button onClick={onClose} style={{
                background: "rgba(255,255,255,0.2)",
                border: "none",
                borderRadius: "50%",
                width: "36px",
                height: "36px",
                fontSize: "20px",
                cursor: "pointer",
                color: "white"
              }}>✕</button>
            </div>
          </div>

          <div style={{ padding: "24px" }}>
            <h4 style={{ fontSize: "16px", fontWeight: "600", marginBottom: "16px" }}>Member List</h4>
            <div style={{ display: "flex", flexDirection: "column", gap: "10px", marginBottom: "20px" }}>
              {displayedMembers.map((member, idx) => (
                <div key={idx} style={{
                  padding: "14px",
                  background: "#f9fafb",
                  borderRadius: "12px",
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  transition: "all 0.2s ease",
                  border: "1px solid #e5e7eb"
                }}
                onMouseEnter={(e) => e.currentTarget.style.background = tierInfo.bgLight}
                onMouseLeave={(e) => e.currentTarget.style.background = "#f9fafb"}>
                  <div>
                    <p style={{ fontWeight: "600", margin: 0, fontSize: "15px" }}>{member.name || member.id}</p>
                    <p style={{ fontSize: "12px", color: "#6b7280", margin: "4px 0 0 0" }}>
                      ID: {member.id} | Age: {member.age}
                    </p>
                  </div>
                  <div style={{ textAlign: "right" }}>
                    <p style={{ 
                      fontSize: "18px", 
                      fontWeight: "bold", 
                      color: tierInfo.color,
                      margin: 0
                    }}>
                      {(member.risk_score * 100).toFixed(1)}%
                    </p>
                    <p style={{ fontSize: "10px", color: "#6b7280", margin: "2px 0 0 0" }}>Risk Score</p>
                  </div>
                </div>
              ))}
            </div>

            {hasMore && (
              <button
                onClick={() => setShowCount(showCount + 5)}
                style={{
                  width: "100%",
                  padding: "12px",
                  background: tierInfo.gradient,
                  color: "white",
                  border: "none",
                  borderRadius: "12px",
                  fontSize: "14px",
                  fontWeight: "600",
                  cursor: "pointer",
                  transition: "all 0.2s ease"
                }}
                onMouseEnter={(e) => e.currentTarget.style.transform = "translateY(-2px)"}
                onMouseLeave={(e) => e.currentTarget.style.transform = "translateY(0)"}
              >
                 Members ({totalMembers} remaining)
              </button>
            )}

            {!hasMore && totalMembers > 0 && (
              <p style={{ textAlign: "center", fontSize: "12px", color: "#6b7280", marginTop: "16px" }}>
                ✓ Showing all {totalMembers} members
              </p>
            )}

            {totalMembers === 0 && (
              <p style={{ textAlign: "center", fontSize: "14px", color: "#6b7280", marginTop: "32px" }}>
                No members in this risk tier
              </p>
            )}
          </div>
        </div>
      </div>
    );
  };

  return (
    <div style={{
      background: "white",
      borderRadius: "24px",
      padding: "28px",
      boxShadow: "0 4px 20px rgba(0,0,0,0.08)",
      transition: "all 0.3s ease"
    }}>
      <style>{`
        @keyframes fadeInUp {
          from { opacity: 0; transform: translateY(20px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes rotateIn {
          from { opacity: 0; transform: rotate(-180deg) scale(0.5); }
          to { opacity: 1; transform: rotate(0) scale(1); }
        }
        @keyframes slideUp {
          from { opacity: 0; transform: translateY(50px); }
          to { opacity: 1; transform: translateY(0); }
        }
      `}</style>

      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "24px", flexWrap: "wrap", gap: "16px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
          <div style={{
            width: "48px",
            height: "48px",
            background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
            borderRadius: "16px",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: "24px",
            animation: "rotateIn 0.5s ease-out"
          }}>📊</div>
          <div>
            <h3 style={{ fontSize: "20px", fontWeight: "700", color: "#1f2937", margin: 0 }}>Risk Tier Distribution</h3>
            <p style={{ fontSize: "13px", color: "#6b7280", margin: "4px 0 0 0" }}>Click on any tier to view member details</p>
          </div>
        </div>
        <div style={{ background: "linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%)", padding: "8px 20px", borderRadius: "40px", textAlign: "center" }}>
          <p style={{ fontSize: "11px", color: "#6b7280", margin: 0 }}>Total Members</p>
          <p style={{ fontSize: "24px", fontWeight: "bold", color: "#1f2937", margin: 0, lineHeight: "1.2" }}>{total.toLocaleString()}</p>
        </div>
      </div>

      <div style={{ position: "relative", height: "350px", marginBottom: "20px" }}>
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={pieData}
              cx="50%"
              cy="50%"
              innerRadius={80}
              outerRadius={120}
              paddingAngle={3}
              dataKey="value"
              stroke="white"
              strokeWidth={3}
              animationBegin={0}
              animationDuration={1000}
              animationEasing="ease-out"
            >
              {pieData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.color} style={{
                  cursor: "pointer",
                  transition: "filter 0.3s ease",
                  filter: selectedTier === entry.name ? "brightness(1.1)" : "brightness(1)"
                }} />
              ))}
            </Pie>
            <Tooltip content={<CustomTooltip />} />
          </PieChart>
        </ResponsiveContainer>
        
        <div style={{
          position: "absolute",
          top: "50%",
          left: "50%",
          transform: "translate(-50%, -50%)",
          textAlign: "center",
          pointerEvents: "none"
        }}>
          <div style={{
            background: "rgba(255,255,255,0.95)",
            borderRadius: "50%",
            width: "100px",
            height: "100px",
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            boxShadow: "0 4px 12px rgba(0,0,0,0.1)"
          }}>
            <span style={{ fontSize: "28px" }}>🎯</span>
            <p style={{ fontSize: "12px", fontWeight: "600", color: "#1f2937", margin: "4px 0 0 0" }}>Risk Profile</p>
          </div>
        </div>
      </div>

      <CustomLegend payload={pieData.map(d => ({ value: d.name }))} />

      <div style={{
        marginTop: "24px",
        display: "grid",
        gridTemplateColumns: "repeat(3, 1fr)",
        gap: "12px",
        padding: "16px",
        background: "#f9fafb",
        borderRadius: "16px"
      }}>
        <div style={{ textAlign: "center" }}>
          <p style={{ fontSize: "11px", color: "#6b7280", marginBottom: "4px" }}>High Risk (High + Very High)</p>
          <p style={{ fontSize: "18px", fontWeight: "bold", color: "#ef4444" }}>{(tierDistribution["High"] || 0) + (tierDistribution["Very High"] || 0)}</p>
        </div>
        <div style={{ textAlign: "center" }}>
          <p style={{ fontSize: "11px", color: "#6b7280", marginBottom: "4px" }}>Medium Risk</p>
          <p style={{ fontSize: "18px", fontWeight: "bold", color: "#f59e0b" }}>{tierDistribution["Medium"] || 0}</p>
        </div>
        <div style={{ textAlign: "center" }}>
          <p style={{ fontSize: "11px", color: "#6b7280", marginBottom: "4px" }}>Low Risk (Very Low + Low)</p>
          <p style={{ fontSize: "18px", fontWeight: "bold", color: "#10b981" }}>{(tierDistribution["Very Low"] || 0) + (tierDistribution["Low"] || 0)}</p>
        </div>
      </div>

      {showMemberDetails && selectedTier && (
        <MemberDetailsModal tier={selectedTier} onClose={() => {
          setShowMemberDetails(false);
          setSelectedTier(null);
        }} />
      )}
    </div>
  );
});

export default RiskTierChart;
