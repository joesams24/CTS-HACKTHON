// src/components/CatastropheMetrics.jsx
export default function CatastropheMetrics({ catastropheMetrics }) {
  if (!catastropheMetrics) return null;

  return (
    <div style={{
      backgroundColor: "white",
      borderRadius: "12px",
      padding: "24px",
      boxShadow: "0 1px 3px rgba(0,0,0,0.1)",
      marginBottom: "24px"
    }}>
      <h3 style={{ marginBottom: "20px", fontSize: "18px", fontWeight: "600" }}>
        🚨 Catastrophic Event Impact
      </h3>
      
      <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: "16px" }}>
        <div>
          <p style={{ fontSize: "14px", color: "#6b7280" }}>Baseline Events</p>
          <p style={{ fontSize: "20px", fontWeight: "bold", color: "#ef4444" }}>
            {catastropheMetrics.baseline_events}
          </p>
        </div>
        
        <div>
          <p style={{ fontSize: "14px", color: "#6b7280" }}>Treated Events</p>
          <p style={{ fontSize: "20px", fontWeight: "bold", color: "#f59e0b" }}>
            {catastropheMetrics.treated_events}
          </p>
        </div>
        
        <div>
          <p style={{ fontSize: "14px", color: "#6b7280" }}>✅ Avoided Events</p>
          <p style={{ fontSize: "24px", fontWeight: "bold", color: "#10b981" }}>
            {catastropheMetrics.avoided_events}
          </p>
        </div>
        
        <div>
          <p style={{ fontSize: "14px", color: "#6b7280" }}>💰 Acute Savings</p>
          <p style={{ fontSize: "20px", fontWeight: "bold", color: "#10b981" }}>
            ₹{catastropheMetrics.acute_savings?.toLocaleString("en-IN") || 0}
          </p>
        </div>
      </div>
    </div>
  );
}
