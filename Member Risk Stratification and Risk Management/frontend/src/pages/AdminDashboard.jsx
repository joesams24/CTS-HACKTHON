import { useState } from "react";
import FileUploader from "../components/FileUploader";

export default function AdminDashboard() {
  const [dashboard, setDashboard] = useState(null);

  const formatCurrency = (value) =>
    value?.toLocaleString("en-IN", { maximumFractionDigits: 0 });

  return (
    <div style={{ padding: "24px", maxWidth: "1100px", margin: "0 auto" }}>
      <h1>Member Risk Stratification Dashboard (PoC)</h1>

      <FileUploader onResult={setDashboard} />

      {!dashboard && (
        <p>Please upload a CSV file to generate policy-based insights.</p>
      )}

      {dashboard && (
        <>
          {/* ================= BUSINESS METRICS ================= */}
          <section style={{ marginTop: "24px" }}>
            <h2>Business Overview</h2>
            <p>
              <strong>Total members analyzed:</strong>{" "}
              {dashboard.population_size}
            </p>
            <p>
              <strong>Executive Summary:</strong> {dashboard.executive_summary}
            </p>
          </section>

          {/* ================= ML METRICS ================= */}
          <section style={{ marginTop: "24px" }}>
            <h2>Model Health (ML Metrics)</h2>
            <ul>
              <li>
                Mean predicted risk:{" "}
                <strong>{dashboard.ml_metrics.mean_predicted_risk}</strong>
              </li>
              <li>
                Risk standard deviation:{" "}
                <strong>{dashboard.ml_metrics.risk_std_dev}</strong>
              </li>
              <li>
                High-risk population fraction:{" "}
                <strong>
                  {dashboard.ml_metrics.high_risk_fraction * 100}%
                </strong>
              </li>
              <li>
                Top decile average risk:{" "}
                <strong>{dashboard.ml_metrics.top_decile_avg_risk}</strong>
              </li>
            </ul>
          </section>

          {/* ================= ROI BY HORIZON ================= */}
          <section style={{ marginTop: "24px" }}>
            <h2>ROI by Time Horizon</h2>
            <ul>
              {Object.entries(dashboard.roi_by_horizon).map(([window, roi]) => (
                <li key={window}>
                  {window} days → <strong>{roi}%</strong>
                </li>
              ))}
            </ul>
          </section>

          {/* ================= WINDOW-BASED ANALYSIS ================= */}
          {Object.entries(dashboard.windows).map(([window, data]) => (
            <section
              key={window}
              style={{
                border: "1px solid #ddd",
                padding: "16px",
                marginTop: "32px",
                borderRadius: "6px",
              }}
            >
              <h2>{window}-Day Window Analysis</h2>

              {/* ---------- Intervention Metrics ---------- */}
              <h3>Intervention Metrics</h3>
              <p>
                💰 Intervention cost: ₹
                {formatCurrency(
                  data.intervention_metrics.total_intervention_cost
                )}
              </p>
              <p>
                💾 Expected savings: ₹
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

              {/* ---------- Tier Distribution ---------- */}
              <h3>Risk Tier Distribution</h3>
              <ul>
                {Object.entries(data.tier_distribution).map(([tier, count]) => (
                  <li key={tier}>
                    {tier}: {count}
                  </li>
                ))}
              </ul>

              {/* ---------- Catastrophe Metrics ---------- */}
              <h3>Catastrophic Event Impact</h3>
              <p>Baseline events: {data.catastrophe_metrics.baseline_events}</p>
              <p>Treated events: {data.catastrophe_metrics.treated_events}</p>
              <p>
                Avoided events:{" "}
                <strong>{data.catastrophe_metrics.avoided_events}</strong>
              </p>
              <p>
                Acute savings: ₹
                {formatCurrency(data.catastrophe_metrics.acute_savings)}
              </p>

              {/* ---------- Recommended Decision ---------- */}
              <h3>Recommended Intervention Strategy</h3>
              <p>
                <strong>{data.recommended_decision.recommendation}</strong>
              </p>
              <p>{data.recommended_decision.rationale}</p>
              <p>
                Eligible tiers:{" "}
                {data.recommended_decision.eligible_tiers.join(", ")}
              </p>
            </section>
          ))}

          {/* ================= MIGRATION METRICS ================= */}
          <section style={{ marginTop: "32px" }}>
            <h2>Risk Migration Metrics</h2>

            {Object.entries(dashboard.migration_summary).map(
              ([transition, summary]) => (
                <div
                  key={transition}
                  style={{
                    border: "1px dashed #ccc",
                    padding: "12px",
                    marginTop: "12px",
                  }}
                >
                  <h4>{transition.replace("_", " → ")}</h4>
                  <p>
                    Net new high-risk members:{" "}
                    <strong>{summary.net_new_high_risk_members}</strong>
                  </p>
                  <p>
                    Net recovered members:{" "}
                    <strong>{summary.net_recovered_members}</strong>
                  </p>
                  <p>Total upward moves: {summary.total_upward_moves}</p>
                  <p>Total downward moves: {summary.total_downward_moves}</p>
                </div>
              )
            )}
          </section>
        </>
      )}
    </div>
  );
}
