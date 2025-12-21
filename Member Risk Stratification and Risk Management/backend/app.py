print("🔥🔥🔥 THIS app.py IS LOADED 🔥🔥🔥")

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
from collections import Counter

from analytics.budget_allocator import apply_budget_cap
from analytics.diminishing_returns import diminishing_effectiveness
from analytics.risk_realization import realized_savings_factor
from analytics.window_discount import window_discount_factor

from utils.validators import validate_csv
from preprocessing.preprocess import preprocess_data
from modeling.train_model import train_xgboost
from modeling.risk_scoring import assign_quantile_tiers
from modeling.deterioration import simulate_deterioration
from care.interventions import get_intervention

app = FastAPI(title="Member Risk Stratification PoC")

# -------------------- CONFIG --------------------
ANNUAL_BUDGET = 250_000_000
BASE_COST = 100_000

TIER_COST_MULTIPLIER = {
    "Very Low": 0.4,
    "Low": 0.6,
    "Medium": 1.0,
    "High": 1.5,
    "Very High": 2.2
}

# -------------------- POLICY TOGGLE --------------------
POLICY_BY_WINDOW = {
    30: {
        "eligible_tiers": ["Very High"],
        "high_coverage_multiplier": 0.0,
        "policy_note": "Short-term horizon → intervene only Very High risk members"
    },
    60: {
        "eligible_tiers": ["Very High", "High"],
        "high_coverage_multiplier": 0.5,
        "policy_note": "Mid-term horizon → Very High full, High partial coverage"
    },
    90: {
        "eligible_tiers": ["Very High", "High"],
        "high_coverage_multiplier": 1.0,
        "policy_note": "Long-term horizon → aggressive intervention strategy"
    }
}

# -------------------- CORS --------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

uploaded_data = None

# -------------------- Health --------------------
@app.get("/")
def health_check():
    return {"status": "Backend running"}

# -------------------- Upload --------------------
@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    global uploaded_data

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="CSV only")

    uploaded_data = validate_csv(file.file)

    return {
        "message": "File uploaded successfully",
        "rows": uploaded_data.shape[0],
        "columns": uploaded_data.shape[1]
    }

# -------------------- Train --------------------
@app.post("/train")
def train_model():
    if uploaded_data is None:
        raise HTTPException(status_code=400, detail="No data uploaded")

    auc = train_xgboost(uploaded_data)
    return {"validation_auc": round(auc, 3)}

# -------------------- ADMIN DASHBOARD --------------------
@app.get("/admin-dashboard")
def admin_dashboard():
    if uploaded_data is None:
        raise HTTPException(status_code=400, detail="No data uploaded")

    model = joblib.load("artifacts/xgb_model.pkl")
    X, _, _ = preprocess_data(uploaded_data)
    base_probs = model.predict_proba(X)[:, 1]

    dashboard = {
        "population_size": len(base_probs),
        "baseline_window": 30,
        "windows": {}
    }

    # -------- Baseline cutoffs (audit only) --------
    baseline_probs = [simulate_deterioration(p, 30) for p in base_probs]
    _, baseline_cutoffs = assign_quantile_tiers(baseline_probs)
    dashboard["baseline_cutoffs"] = baseline_cutoffs

    # ---------------- WINDOWS ----------------
    for window in [30, 60, 90]:
        adjusted_probs = [simulate_deterioration(p, window) for p in base_probs]
        tiers, cutoffs = assign_quantile_tiers(adjusted_probs)

        tier_counts = dict(Counter(tiers))

        policy = POLICY_BY_WINDOW[window]
        discount = window_discount_factor(window)

        remaining_budget = ANNUAL_BUDGET
        total_intervention_cost = 0
        total_expected_savings = 0

        # -------- Priority order --------
        for tier in ["Very High", "High"]:
            if tier not in policy["eligible_tiers"]:
                continue

            members = tier_counts.get(tier, 0)
            if members == 0:
                continue

            intervention = get_intervention(tier)

            coverage_multiplier = (
                policy["high_coverage_multiplier"]
                if tier == "High"
                else 1.0
            )

            target_members = int(members * coverage_multiplier)

            covered_members, spend = apply_budget_cap(
                tier=tier,
                members=target_members,
                cost_per_member=intervention["cost"],
                max_budget=remaining_budget,
                priority_weight=1.0
            )

            if covered_members == 0:
                continue

            remaining_budget -= spend
            coverage_ratio = covered_members / members

            effective_reduction = diminishing_effectiveness(
                base_effectiveness=intervention["risk_reduction"],
                coverage_ratio=coverage_ratio
            )

            tier_base_cost = BASE_COST * TIER_COST_MULTIPLIER[tier]
            avg_risk = sum(
                p for p, t in zip(adjusted_probs, tiers) if t == tier
            ) / members

            savings_pm = tier_base_cost * avg_risk * effective_reduction

            realized_savings = (
                savings_pm
                * realized_savings_factor(avg_risk)
                * discount
                * covered_members
            )

            total_intervention_cost += spend
            total_expected_savings += realized_savings

        dashboard["windows"][window] = {
            "policy": policy,
            "tier_distribution": tier_counts,
            "intervention_metrics": {
                "total_intervention_cost": round(total_intervention_cost, 2),
                "total_expected_savings": round(total_expected_savings, 2),
                "net_benefit": round(
                    total_expected_savings - total_intervention_cost, 2
                ),
                "roi_percent": round(
                    ((total_expected_savings - total_intervention_cost)
                     / total_intervention_cost) * 100
                    if total_intervention_cost > 0 else 0,
                    2
                ),
                "remaining_budget": round(remaining_budget, 2)
            },
            "tier_cutoffs": cutoffs
        }

    return dashboard
