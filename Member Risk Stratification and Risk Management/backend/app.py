print("🔥🔥🔥 THIS app.py IS LOADED 🔥🔥🔥")

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import os

from utils.validators import validate_csv
from preprocessing.preprocess import preprocess_data
from modeling.train_model import train_xgboost
from modeling.risk_scoring import probability_to_score, assign_percentile_tiers
from modeling.deterioration import simulate_deterioration
from care.interventions import get_intervention
from care.roi import calculate_roi

app = FastAPI(title="Member Risk Stratification PoC")

# -------------------- CORS --------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

uploaded_data = None


# -------------------- Health Check --------------------
@app.get("/")
def health_check():
    return {"status": "Backend running"}


# -------------------- Step 2: Upload --------------------
@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    global uploaded_data

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="CSV only")

    df = validate_csv(file.file)
    uploaded_data = df

    return {
        "message": "File uploaded successfully",
        "rows": df.shape[0],
        "columns": df.shape[1]
    }


# -------------------- Step 3: Preprocess (Debug) --------------------
@app.get("/preprocess")
def preprocess_endpoint():
    if uploaded_data is None:
        raise HTTPException(status_code=400, detail="No data uploaded")

    X, y, _ = preprocess_data(uploaded_data)

    return {
        "features_shape": X.shape,
        "target_distribution": y.value_counts().to_dict(),
        "sample_features": X.head(3).to_dict(orient="records")
    }


# -------------------- Step 4: Train Model --------------------
@app.post("/train")
def train_model():
    if uploaded_data is None:
        raise HTTPException(status_code=400, detail="No data uploaded")

    auc = train_xgboost(uploaded_data)

    return {
        "message": "Model trained successfully",
        "validation_auc": round(auc, 3)
    }


# -------------------- Step 4b: Predict by Time Window --------------------
@app.get("/predict-by-window")
def predict_by_window(window: int = 30):
    if uploaded_data is None:
        raise HTTPException(status_code=400, detail="No data uploaded")

    if window not in [30, 60, 90]:
        raise HTTPException(status_code=400, detail="Window must be 30, 60, or 90")

    model = joblib.load("artifacts/xgb_model.pkl")

    X, _, _ = preprocess_data(uploaded_data)
    base_probs = model.predict_proba(X)[:, 1]

    adjusted_probs = [
        simulate_deterioration(p, window) for p in base_probs
    ]

    tiers = assign_percentile_tiers(adjusted_probs)

    results = []
    for p, tier in zip(adjusted_probs[:10], tiers[:10]):
        results.append({
            "risk_probability": round(float(p), 3),
            "risk_score": probability_to_score(p),
            "risk_tier": tier
        })

    return {
        "window_days": window,
        "sample_predictions": results
    }


# -------------------- Step 5: Care + ROI --------------------
@app.get("/care-simulation")
def care_simulation(window: int = 30):
    if uploaded_data is None:
        raise HTTPException(status_code=400, detail="No data uploaded")

    model = joblib.load("artifacts/xgb_model.pkl")

    X, _, _ = preprocess_data(uploaded_data)
    base_probs = model.predict_proba(X)[:, 1]

    adjusted_probs = [
        simulate_deterioration(p, window) for p in base_probs
    ]

    tiers = assign_percentile_tiers(adjusted_probs)

    BASE_COST = 100000

    results = []
    for p, tier in zip(adjusted_probs[:10], tiers[:10]):
        intervention = get_intervention(tier)

        roi = calculate_roi(
            base_cost=BASE_COST,
            intervention_cost=intervention["cost"],
            risk_probability=p,
            risk_reduction=intervention["risk_reduction"]
        )

        results.append({
            "risk_probability": round(float(p), 3),
            "risk_score": probability_to_score(p),
            "risk_tier": tier,
            "intervention": intervention["intervention"],
            "intervention_cost": intervention["cost"],
            "roi": roi
        })

    return {
        "window_days": window,
        "care_simulation_results": results
    }

@app.get("/__routes")
def list_routes():
    return [route.path for route in app.routes]
