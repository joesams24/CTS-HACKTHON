from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
from collections import Counter

from analytics.budget_allocator import apply_budget_cap
from analytics.diminishing_returns import diminishing_effectiveness
from analytics.risk_realization import realized_savings_factor
from analytics.window_discount import window_discount_factor
from analytics.risk_migration import build_migration_matrix
from analytics.migration_summary import summarize_migration
from analytics.decision_policy import (
    acute_realization_factor,
    recommended_intervention_policy
)

from utils.validators import validate_csv
from preprocessing.preprocess import preprocess_data
from modeling.train_model import train_xgboost
from modeling.deterioration import simulate_deterioration
from modeling.baseline_tiering import assign_tiers_from_cutoffs
from care.interventions import get_intervention


# -------------------- APP --------------------
app = FastAPI(title="Member Risk Stratification PoC")

# -------------------- CONFIG --------------------
ANNUAL_BUDGET = 250_000_000
BASE_COST = 100_000

ANALYSIS_WINDOWS = [30, 60, 90]

ACUTE_EVENT_COST = 250_000
BASE_CATASTROPHE_RATE = 0.025
TREATED_CATASTROPHE_RATE = 0.006

TIER_COST_MULTIPLIER = {
    "Very Low": 0.4,
    "Low": 0.6,
    "Medium": 1.0,
    "High": 1.5,
    "Very High": 2.2
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


# -------------------- HEALTH --------------------
@app.get("/")
def health_check():
    return {"status": "ok"}


# -------------------- UPLOAD --------------------
@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    global uploaded_data
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="CSV only")
    uploaded_data = validate_csv(file.file)
    return {
        "rows": uploaded_data.shape[0],
        "columns": uploaded_data.shape[1]
    }


# -------------------- TRAIN --------------------
@app.post("/train")
def train_model_endpoint():
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

    # ---------------- ML METRICS ----------------
    dashboard["ml_metrics"] = {
        "mean_predicted_risk": round(float(np.mean(base_probs)), 4),
        "risk_std_dev": round(float(np.std(base_probs)), 4),
        "high_risk_fraction": round(
            float(np.mean(base_probs >= np.quantile(base_probs, 0.6))), 4
        ),
        "top_decile_avg_risk": round(
            float(np.mean(base_probs[base_probs >= np.quantile(base_probs, 0.9)])),
            4
        )
    }

    # -------- BASELINE CUT-OFFS (computed once) --------
    baseline_probs = [simulate_deterioration(p, 30) for p in base_probs]
    q20, q40, q60, q80 = np.quantile(baseline_probs, [0.2, 0.4, 0.6, 0.8])

    baseline_cutoffs = {
        "Very Low": {"max": round(q20, 4)},
        "Low": {"min": round(q20, 4), "max": round(q40, 4)},
        "Medium": {"min": round(q40, 4), "max": round(q60, 4)},
        "High": {"min": round(q60, 4), "max": round(q80, 4)},
        "Very High": {"min": round(q80, 4)}
    }

    dashboard["baseline_cutoffs"] = baseline_cutoffs
    tiers_by_window = {}

    # ---------------- WINDOWS ----------------
    for window in ANALYSIS_WINDOWS:
        adjusted_probs = [simulate_deterioration(p, window) for p in base_probs]
        tiers = assign_tiers_from_cutoffs(adjusted_probs, baseline_cutoffs)

        tiers_by_window[window] = tiers
        tier_counts = dict(Counter(tiers))

        discount = window_discount_factor(window)
        remaining_budget = ANNUAL_BUDGET

        total_cost = 0.0
        total_savings = 0.0

        baseline_events = 0
        treated_events = 0
        avoided_events = 0
        avoided_catastrophe_savings = 0.0

        for tier in ["Very High", "High"]:
            members = tier_counts.get(tier, 0)
            if members == 0:
                continue

            intervention = get_intervention(tier)

            covered, spend = apply_budget_cap(
                tier=tier,
                members=members,
                cost_per_member=intervention["cost"],
                max_budget=remaining_budget,
                priority_weight=1.0
            )

            if covered == 0:
                continue

            remaining_budget -= spend
            total_cost += spend

            avg_risk = np.mean(
                [p for p, t in zip(adjusted_probs, tiers) if t == tier]
            )

            effectiveness = diminishing_effectiveness(
                intervention["risk_reduction"],
                covered / members
            )

            tier_cost = BASE_COST * TIER_COST_MULTIPLIER[tier]

            chronic_savings = (
                tier_cost
                * avg_risk
                * effectiveness
                * realized_savings_factor(avg_risk)
                * discount
                * covered
            )

            total_savings += chronic_savings

            # -------- Catastrophe avoidance (with realization timing) --------
            b_events = int(members * BASE_CATASTROPHE_RATE)
            t_events = int(covered * TREATED_CATASTROPHE_RATE)
            a_events = max(b_events - t_events, 0)

            baseline_events += b_events
            treated_events += t_events
            avoided_events += a_events

            avoided_catastrophe_savings += (
                a_events
                * ACUTE_EVENT_COST
                * acute_realization_factor(window)
            )

        total_savings += avoided_catastrophe_savings

        dashboard["windows"][window] = {
            "tier_distribution": tier_counts,
            "intervention_metrics": {
                "total_intervention_cost": round(total_cost, 2),
                "total_expected_savings": round(total_savings, 2),
                "net_benefit": round(total_savings - total_cost, 2),
                "roi_percent": round(
                    ((total_savings - total_cost) / total_cost) * 100, 2
                ),
            },
            "catastrophe_metrics": {
                "baseline_events": baseline_events,
                "treated_events": treated_events,
                "avoided_events": avoided_events,
                "acute_savings": round(avoided_catastrophe_savings, 2),
            },
            "recommended_decision": recommended_intervention_policy(window)
        }

    # ---------------- MIGRATION SUMMARY ----------------
    migration_summary = {}
    for w1, w2 in zip(ANALYSIS_WINDOWS[:-1], ANALYSIS_WINDOWS[1:]):
        matrix = build_migration_matrix(
            tiers_by_window[w1],
            tiers_by_window[w2]
        )
        migration_summary[f"{w1}_to_{w2}"] = summarize_migration(matrix)

    dashboard["migration_summary"] = migration_summary

    # ---------------- ROI BY HORIZON ----------------
    dashboard["roi_by_horizon"] = {
        w: dashboard["windows"][w]["intervention_metrics"]["roi_percent"]
        for w in ANALYSIS_WINDOWS
    }

    dashboard["executive_summary"] = (
        "Early horizons show lower or negative ROI due to front-loaded costs "
        "and delayed realization of avoided acute events. As the time horizon "
        "extends, prevention of catastrophic events and stabilization of "
        "high-risk members generate compounding economic value."
    )

    return dashboard
@app.get("/debug-data")
def debug_data():
    """Debug endpoint to check what data is available"""
    global uploaded_data
    if uploaded_data is None:
        return {"status": "no_data"}
    return {
        "status": "has_data",
        "rows": uploaded_data.shape[0],
        "columns": list(uploaded_data.columns),
        "sample": uploaded_data.head(2).to_dict()
    }
