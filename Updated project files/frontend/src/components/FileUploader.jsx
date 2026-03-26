// src/components/FileUploader.jsx
import { useState } from "react";
import { runFullPipeline } from "../api/backend";

export default function FileUploader({ onResult, onLoading, onError }) {
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState("");
  const [dragActive, setDragActive] = useState(false);
  const [isUploading, setIsUploading] = useState(false);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile && droppedFile.name.endsWith('.csv')) {
      setFile(droppedFile);
      setStatus("");
    } else {
      setStatus("❌ Please upload a valid CSV file");
      setTimeout(() => setStatus(""), 3000);
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setStatus("⚠️ Please select a CSV file");
      setTimeout(() => setStatus(""), 3000);
      return;
    }

    if (!file.name.endsWith('.csv')) {
      setStatus("❌ Please upload a CSV file");
      setTimeout(() => setStatus(""), 3000);
      return;
    }

    try {
      setIsUploading(true);
      onLoading(true);
      setStatus("📤 Uploading and processing...");
      
      const result = await runFullPipeline(file);
      
      setStatus("✅ Analysis complete!");
      onResult(result);
    } catch (err) {
      console.error("Pipeline error:", err);
      setStatus(`❌ ${err.message}`);
      onError(err.message);
      setTimeout(() => setStatus(""), 5000);
    } finally {
      setIsUploading(false);
      onLoading(false);
    }
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  };

  return (
    <div style={{
      position: "relative",
      width: "100%",
      maxWidth: "600px",
      margin: "0 auto"
    }}>
      <style>{`
        @keyframes slideUp {
          from {
            opacity: 0;
            transform: translateY(10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        
        .upload-zone {
          transition: all 0.2s ease;
        }
        
        .upload-zone:hover {
          transform: translateY(-2px);
        }
        
        .btn-hover {
          transition: all 0.2s ease;
        }
      `}</style>

      {/* Main Upload Zone */}
      <div
        className="upload-zone"
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        style={{
          background: dragActive 
            ? "linear-gradient(135deg, rgba(102,126,234,0.05) 0%, rgba(118,75,162,0.05) 100%)"
            : "linear-gradient(135deg, #ffffff 0%, #faf5ff 100%)",
          borderRadius: "24px",
          padding: "40px 32px",
          textAlign: "center",
          cursor: "pointer",
          border: dragActive 
            ? "2px solid #667eea"
            : "2px dashed #d1d5db",
          transition: "all 0.2s ease",
          boxShadow: dragActive 
            ? "0 4px 20px rgba(102,126,234,0.1)"
            : "0 2px 8px rgba(0,0,0,0.05)"
        }}
        onClick={() => !isUploading && document.getElementById('file-upload').click()}
      >
        {/* Icon */}
        <div style={{ marginBottom: "24px" }}>
          {isUploading ? (
            <div style={{
              width: "70px",
              height: "70px",
              margin: "0 auto",
              position: "relative"
            }}>
              <div style={{
                position: "absolute",
                top: 0,
                left: 0,
                right: 0,
                bottom: 0,
                border: "3px solid #e5e7eb",
                borderTopColor: "#667eea",
                borderRadius: "50%",
                animation: "spin 0.8s linear infinite"
              }} />
              <div style={{
                position: "absolute",
                top: "50%",
                left: "50%",
                transform: "translate(-50%, -50%)",
                fontSize: "28px"
              }}>
                ⚙️
              </div>
            </div>
          ) : (
            <div style={{
              width: "70px",
              height: "70px",
              margin: "0 auto",
              background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
              borderRadius: "50%",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              boxShadow: "0 4px 12px rgba(102,126,234,0.2)"
            }}>
              <span style={{ fontSize: "32px" }}>📊</span>
            </div>
          )}
        </div>
        
        {/* Title */}
        <h3 style={{
          fontSize: "22px",
          fontWeight: "700",
          background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
          WebkitBackgroundClip: "text",
          WebkitTextFillColor: "transparent",
          marginBottom: "12px"
        }}>
          {isUploading ? "Processing..." : "Upload Member Data"}
        </h3>
        
        <p style={{
          fontSize: "13px",
          color: "#6b7280",
          marginBottom: "24px"
        }}>
          {isUploading 
            ? "Please wait while we analyze your data..." 
            : "Drag & drop CSV or click to browse"}
        </p>
        
        {/* Hidden File Input */}
        <input
          id="file-upload"
          type="file"
          accept=".csv"
          onChange={(e) => {
            const selectedFile = e.target.files[0];
            if (selectedFile && selectedFile.name.endsWith('.csv')) {
              setFile(selectedFile);
              setStatus("");
            } else if (selectedFile) {
              setStatus("❌ Please select a CSV file");
              setTimeout(() => setStatus(""), 3000);
            }
          }}
          style={{ display: "none" }}
          disabled={isUploading}
        />
        
        {/* File Info */}
        {file && !isUploading && (
          <div style={{
            background: "#f9fafb",
            borderRadius: "16px",
            padding: "16px",
            marginBottom: "24px",
            border: "1px solid #e5e7eb"
          }}>
            <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
              <span style={{ fontSize: "24px" }}>📄</span>
              <div style={{ flex: 1, textAlign: "left" }}>
                <p style={{ fontWeight: "500", margin: 0, fontSize: "14px", color: "#1f2937" }}>
                  {file.name}
                </p>
                <p style={{ fontSize: "11px", color: "#6b7280", margin: "4px 0 0 0" }}>
                  {formatFileSize(file.size)}
                </p>
              </div>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  setFile(null);
                  setStatus("");
                }}
                style={{
                  background: "#fee2e2",
                  border: "none",
                  borderRadius: "50%",
                  width: "28px",
                  height: "28px",
                  cursor: "pointer",
                  fontSize: "14px"
                }}
              >
                ✕
              </button>
            </div>
          </div>
        )}
        
        {/* Upload Button */}
        {!isUploading && (
          <button
            onClick={(e) => {
              e.stopPropagation();
              handleUpload();
            }}
            disabled={!file}
            style={{
              width: "100%",
              padding: "12px",
              background: file 
                ? "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
                : "#e5e7eb",
              color: file ? "white" : "#9ca3af",
              border: "none",
              borderRadius: "16px",
              fontSize: "14px",
              fontWeight: "600",
              cursor: file ? "pointer" : "not-allowed",
              transition: "all 0.2s ease"
            }}
            onMouseEnter={(e) => {
              if (file) {
                e.currentTarget.style.transform = "translateY(-1px)";
                e.currentTarget.style.boxShadow = "0 4px 12px rgba(102,126,234,0.3)";
              }
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.transform = "translateY(0)";
              e.currentTarget.style.boxShadow = "none";
            }}
          >
            {file ? "🚀 Upload & Analyze" : "📁 Select CSV File"}
          </button>
        )}
        
        {/* Cancel Button */}
        {isUploading && (
          <button
            onClick={(e) => {
              e.stopPropagation();
              setIsUploading(false);
              onLoading(false);
              setStatus("Upload cancelled");
              setTimeout(() => setStatus(""), 2000);
            }}
            style={{
              width: "100%",
              padding: "12px",
              background: "#f3f4f6",
              color: "#ef4444",
              border: "1px solid #e5e7eb",
              borderRadius: "16px",
              fontSize: "14px",
              fontWeight: "600",
              cursor: "pointer"
            }}
          >
            Cancel Upload
          </button>
        )}
        
        {/* Status */}
        {status && (
          <div style={{
            marginTop: "20px",
            padding: "10px",
            background: status.includes("✅") 
              ? "#d1fae5"
              : status.includes("❌") || status.includes("⚠️")
              ? "#fee2e2"
              : "#dbeafe",
            borderRadius: "12px"
          }}>
            <p style={{
              fontSize: "12px",
              color: status.includes("✅") ? "#065f46" : status.includes("❌") || status.includes("⚠️") ? "#991b1b" : "#1e40af",
              margin: 0
            }}>
              {status}
            </p>
          </div>
        )}
      </div>
      
      {/* Info Badges */}
      {!isUploading && !file && (
        <div style={{
          marginTop: "20px",
          display: "flex",
          justifyContent: "center",
          gap: "16px",
          flexWrap: "wrap"
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
            <span style={{ fontSize: "16px" }}>📊</span>
            <span style={{ fontSize: "11px", color: "#6b7280" }}>CSV</span>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
            <span style={{ fontSize: "16px" }}>🔒</span>
            <span style={{ fontSize: "11px", color: "#6b7280" }}>Secure</span>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
            <span style={{ fontSize: "16px" }}>⚡</span>
            <span style={{ fontSize: "11px", color: "#6b7280" }}>Fast</span>
          </div>
        </div>
      )}
      
      {/* Requirements */}
      {!isUploading && !file && (
        <div style={{
          marginTop: "16px",
          padding: "12px",
          background: "#f9fafb",
          borderRadius: "12px"
        }}>
          <p style={{ fontSize: "10px", color: "#6b7280", margin: 0, textAlign: "center" }}>
            Required: age, time_in_hospital, number_inpatient, number_emergency, 
            num_medications, number_diagnoses, insulin, readmitted
          </p>
        </div>
      )}

      <style>{`
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}
