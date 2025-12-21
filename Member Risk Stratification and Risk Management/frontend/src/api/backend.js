const BASE_URL = "http://127.0.0.1:8000";

// -------------------- Upload CSV --------------------
export const uploadFile = async (file) => {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${BASE_URL}/upload`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    throw new Error("File upload failed");
  }

  return res.json();
};

// -------------------- Train Model --------------------
export const trainModel = async () => {
  const res = await fetch(`${BASE_URL}/train`, {
    method: "POST",
  });

  if (!res.ok) {
    throw new Error("Model training failed");
  }

  return res.json();
};

// -------------------- Window-aware Predictions --------------------
export const getPredictions = async (window = 30) => {
  const res = await fetch(`${BASE_URL}/predict-by-window?window=${window}`);

  if (!res.ok) {
    throw new Error("Prediction fetch failed");
  }

  return res.json();
};

// -------------------- Care + ROI Simulation --------------------
export const getCareSimulation = async (window = 30) => {
  const res = await fetch(`${BASE_URL}/care-simulation?window=${window}`);

  if (!res.ok) {
    throw new Error("Care simulation fetch failed");
  }

  return res.json();
};
