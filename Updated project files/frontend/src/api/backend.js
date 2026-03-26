// src/api/backend.js

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
    const error = await res.json();
    throw new Error(error.detail || "File upload failed");
  }

  return res.json();
};

// -------------------- Train Model --------------------
export const trainModel = async () => {
  const res = await fetch(`${BASE_URL}/train`, {
    method: "POST",
  });

  if (!res.ok) {
    const error = await res.json();
    throw new Error(error.detail || "Model training failed");
  }

  return res.json();
};

// -------------------- Admin Dashboard --------------------
export const getAdminDashboard = async () => {
  const res = await fetch(`${BASE_URL}/admin-dashboard`);

  if (!res.ok) {
    const error = await res.json();
    throw new Error(error.detail || "Admin dashboard fetch failed");
  }

  return res.json();
};

// -------------------- FULL PIPELINE --------------------
export const runFullPipeline = async (file) => {
  await uploadFile(file);
  const trainResult = await trainModel();
  const dashboardData = await getAdminDashboard();
  return {
    ...dashboardData,
    training_auc: trainResult.validation_auc
  };
};
