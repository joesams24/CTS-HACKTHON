// src/components/FileUploader.jsx
import { useState } from "react";
import { runFullPipeline } from "../api/backend";

export default function FileUploader({ onResult, onLoading, onError }) {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState("");

  const handleUpload = async () => {
    if (!file) {
      alert("Please select a CSV file");
      return;
    }

    if (!file.name.endsWith('.csv')) {
      alert("Please upload a CSV file");
      return;
    }

    try {
      setLoading(true);
      onLoading && onLoading(true);
      setStatus("📤 Uploading CSV...");
      console.log("Starting upload...");

      setStatus("⚙️ Training risk model...");
      const result = await runFullPipeline(file);
      
      console.log("Pipeline result:", result);
      setStatus("✅ Analysis complete!");
      
      onResult(result);
    } catch (err) {
      console.error("Pipeline error:", err);
      setStatus("❌ Pipeline failed. Check backend logs.");
      onError && onError(err.message);
      alert(`Pipeline failed: ${err.message}`);
    } finally {
      setLoading(false);
      onLoading && onLoading(false);
    }
  };

  return (
    <div style={{ marginBottom: "24px" }}>
      <div style={{ display: "flex", alignItems: "center", gap: "12px", flexWrap: "wrap" }}>
        <input
          type="file"
          accept=".csv"
          onChange={(e) => {
            const selectedFile = e.target.files[0];
            console.log("File selected:", selectedFile?.name);
            setFile(selectedFile);
            setStatus("");
          }}
          disabled={loading}
          style={{ 
            flex: 1, 
            padding: "8px",
            border: "1px solid #d1d5db",
            borderRadius: "6px"
          }}
        />
        <button
          onClick={handleUpload}
          disabled={loading}
          style={{
            padding: "10px 24px",
            backgroundColor: loading ? "#9ca3af" : "#3b82f6",
            color: "white",
            border: "none",
            borderRadius: "6px",
            cursor: loading ? "not-allowed" : "pointer",
            fontWeight: "500"
          }}
        >
          {loading ? "Processing..." : "Upload & Analyze"}
        </button>
      </div>
      {status && (
        <p style={{ 
          marginTop: "12px", 
          fontSize: "14px",
          color: status.includes("✅") ? "#10b981" : status.includes("❌") ? "#ef4444" : "#6b7280"
        }}>
          {status}
        </p>
      )}
    </div>
  );
}
