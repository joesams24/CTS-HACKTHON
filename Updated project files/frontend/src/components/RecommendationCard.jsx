// src/components/RecommendationCard.jsx
export default function RecommendationCard({ recommendation }) {
  if (!recommendation) return null;

  const getIcon = () => {
    if (recommendation.recommendation.includes("Stabilize")) return "⚠️";
    if (recommendation.recommendation.includes("Aggressive")) return "🚀";
    return "💡";
  };

  return (
    <div style={{
      backgroundColor: "#fef3c7",
      borderRadius: "12px",
      padding: "24px",
      marginBottom: "24px",
      borderLeft: "4px solid #f59e0b"
    }}>
      <div style={{ display: "flex", alignItems: "center", marginBottom: "12px" }}>
        <span style={{ fontSize: "24px", marginRight: "12px" }}>{getIcon()}</span>
        <h3 style={{ fontSize: "18px", fontWeight: "600", color: "#92400e" }}>
          Recommended Strategy
        </h3>
      </div>
      
      <p style={{ fontSize: "16px", fontWeight: "500", color: "#78350f", marginBottom: "8px" }}>
        {recommendation.recommendation}
      </p>
      
      <p style={{ fontSize: "14px", color: "#92400e", marginBottom: "12px" }}>
        {recommendation.rationale}
      </p>
      
      <div style={{ display: "flex", gap: "8px", flexWrap: "wrap" }}>
        {recommendation.eligible_tiers?.map(tier => (
          <span key={tier} style={{
            backgroundColor: "#fed7aa",
            padding: "4px 12px",
            borderRadius: "20px",
            fontSize: "12px",
            fontWeight: "500",
            color: "#78350f"
          }}>
            {tier}
          </span>
        ))}
      </div>
    </div>
  );
}
