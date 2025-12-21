// src/pages/AdminDashboard.jsx
import { useState } from "react";
import FileUploader from "../components/FileUploader";

export default function AdminDashboard() {
  const [dashboard, setDashboard] = useState(null);

  const formatCurrency = (value) =>
    value?.toLocaleString("en-IN", { maximumFractionDigits: 0 });

  return (
    <div style={{ padding: "24px" }}>
      <h1>Member Risk Stratification Dashboard</h1>

      <FileUploader onResult={setDashboard} />

      {!dashboard && (
        <p>Please upload a CSV file to generate policy-based insights.</p>
      )}

      {dashboard && (
        <>
          <p>
            Total population analyzed:{" "}
            <strong>{dashboard.population_size}</strong> members
          </p>

          {Object.entries(dashboard.windows).map(([window, data]) => (
            <div
              key={window}
              style={{
                border: "1px solid #ddd",
                padding: "16px",
                marginTop: "20px",
                borderRadius: "6px",
              }}
            >
              <h2>{window}-Day Policy Window</h2>

              <p>
                <strong>Policy strategy:</strong>{" "}
                {data.policy.policy_note ||
                  "Policy applied based on risk tiers"}
              </p>

              <h4>Intervention Summary</h4>

              <p>
                💰 Total intervention cost: ₹
                {formatCurrency(
                  data.intervention_metrics.total_intervention_cost
                )}
              </p>

              <p>
                💾 Total expected savings: ₹
                {formatCurrency(
                  data.intervention_metrics.total_expected_savings
                )}
              </p>

              <p>
                📈 Net benefit: ₹
                {formatCurrency(data.intervention_metrics.net_benefit)}
              </p>

              <p>
                📊 ROI:{" "}
                <strong>{data.intervention_metrics.roi_percent}%</strong>
              </p>

              <p>
                🧾 Remaining budget: ₹
                {formatCurrency(data.intervention_metrics.remaining_budget)}
              </p>
            </div>
          ))}
        </>
      )}
    </div>
  );
}
