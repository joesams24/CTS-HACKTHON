export default function CareROI({ data }) {
  if (!data || data.length === 0) return null;

  return (
    <div>
      <h3>Care Intervention & ROI Simulation</h3>

      {data.map((row, idx) => (
        <div
          key={idx}
          style={{
            border: "1px solid #ccc",
            padding: "10px",
            marginBottom: "10px",
          }}
        >
          <p>
            <strong>Risk Tier:</strong> {row.risk_tier}
          </p>
          <p>
            <strong>Risk Probability:</strong> {row.risk_probability}
          </p>
          <p>
            <strong>Risk Score:</strong> {row.risk_score}
          </p>

          <p>
            <strong>Intervention:</strong> {row.intervention}
          </p>
          <p>
            <strong>Intervention Cost:</strong> ₹{row.intervention_cost}
          </p>

          <hr />

          <p>
            <strong>Expected Cost (Before):</strong> ₹
            {row.roi.expected_cost_before}
          </p>
          <p>
            <strong>Expected Cost (After):</strong> ₹
            {row.roi.expected_cost_after}
          </p>
          <p>
            <strong>Savings:</strong> ₹{row.roi.savings}
          </p>
          <p>
            <strong>ROI:</strong>{" "}
            <span style={{ color: row.roi.roi < 0 ? "red" : "green" }}>
              {row.roi.roi}
            </span>
          </p>
        </div>
      ))}
    </div>
  );
}
