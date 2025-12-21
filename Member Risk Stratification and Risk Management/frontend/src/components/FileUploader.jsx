// src/components/FileUploader.jsx
import { useState } from "react";
import { runFullPipeline } from "../api/backend";

export default function FileUploader({ onResult }) {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState("");

  const handleUpload = async () => {
    if (!file) {
      alert("Please select a CSV file");
      return;
    }

    try {
      setLoading(true);
      setStatus("📤 Uploading CSV...");

      // Full backend pipeline:
      // Upload → Train → Policy-based ROI simulation
      setStatus("⚙️ Training risk model...");
      const result = await runFullPipeline(file);

      setStatus("✅ Policy simulation complete");
      onResult(result);
    } catch (err) {
      console.error(err);
      setStatus("❌ Pipeline failed. Check backend logs.");
      alert("Pipeline failed. Please check the backend logs.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ marginBottom: "24px" }}>
      <input
        type="file"
        accept=".csv"
        onChange={(e) => setFile(e.target.files[0])}
        disabled={loading}
      />

      <button
        onClick={handleUpload}
        disabled={loading}
        style={{ marginLeft: "12px" }}
      >
        {loading ? "Processing..." : "Upload & Run"}
      </button>

      {status && <p style={{ marginTop: "8px" }}>{status}</p>}
    </div>
  );
}
