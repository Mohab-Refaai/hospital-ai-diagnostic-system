"""Insurance cost prediction service (Tab 1).

Loads the artifact produced by `notebooks/Insurance_Cost_Full_Pipeline.ipynb`
(Part J) and reproduces the notebook's feature engineering + encoding + scaling
so raw form inputs can be fed straight to `predict()`.

Artifact keys: model, model_name, scaler, feature_columns, target_column.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd

ARTIFACT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "model_artifacts",
    "insurance_cost_model.joblib",
)

# Options exactly as they appear in the Kaggle dataset
SEX_OPTIONS = ["female", "male"]
SMOKER_OPTIONS = ["no", "yes"]
REGION_OPTIONS = ["northeast", "northwest", "southeast", "southwest"]

# Ranges of the training data (inputs are limited to these to avoid extrapolation)
AGE_RANGE = (18, 64)
BMI_RANGE = (15.0, 54.0)
CHILDREN_RANGE = (0, 5)

# Held-out test metrics reported by the notebook (Part H) for the saved XGBoost model
NOTEBOOK_METRICS = {"R2": 0.9070, "RMSE": 3662.12, "MAE": 2274.00}

def _group_of(col: str) -> str:
    """Map an engineered model column back to one of the 6 form inputs (SHAP values are additive,
    so summing them per input is exact; interaction terms count toward 'Smoker')."""
    if col.startswith("smoker"):
        return "Smoker"
    if col.startswith("bmi"):
        return "BMI"
    if col.startswith("region_"):
        return "Region"
    return {"age": "Age", "children": "Children", "sex_male": "Sex"}.get(col, col)


def bmi_category(b: float) -> str:
    """Same WHO cut-offs used in the notebook (Part C)."""
    if b < 18.5:
        return "underweight"
    if b < 25:
        return "normal"
    if b < 30:
        return "overweight"
    return "obese"


@dataclass
class Explanation:
    contributions: pd.DataFrame  # columns: feature, contribution (USD)
    base_value: float
    method: str


class InsuranceService:
    def __init__(self, artifact_path: str = ARTIFACT_PATH):
        self.artifact_path = artifact_path
        art = joblib.load(artifact_path)
        self.model = art["model"]
        self.model_name = art.get("model_name", type(self.model).__name__)
        self.scaler = art["scaler"]
        self.feature_columns = list(art["feature_columns"])
        self.target_column = art.get("target_column", "charges")
        self._explainer = None
        self._explainer_tried = False

    def warmup(self) -> None:
        """Run one prediction + explanation at startup so the first user request is fast (NFR: <1s)."""
        sample = dict(age=35, sex="female", bmi=27.4, children=2, smoker="no", region="southeast")
        self.predict(**sample)
        self.explain(**sample)

    # ------------------------------------------------------------------ features
    def build_features(self, age, sex, bmi, children, smoker, region) -> pd.DataFrame:
        flag = 1.0 if smoker == "yes" else 0.0
        row = {
            "age": float(age),
            "bmi": float(bmi),
            "children": float(children),
            "smoker_flag": flag,
            "smoker_x_age": flag * age,
            "smoker_x_bmi": flag * bmi,
            "smoker_x_children": flag * children,
            "sex_male": 1.0 if sex == "male" else 0.0,
            "smoker_yes": flag,
        }
        for r in REGION_OPTIONS:
            row[f"region_{r}"] = 1.0 if region == r else 0.0
        cat = bmi_category(bmi)
        for c in ("underweight", "normal", "overweight", "obese"):
            row[f"bmi_category_{c}"] = 1.0 if cat == c else 0.0
        # Keep ONLY the columns (and order) the model was trained on
        return pd.DataFrame([row]).reindex(columns=self.feature_columns, fill_value=0.0).astype(float)

    def _scaled(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(self.scaler.transform(X), columns=X.columns, index=X.index)

    # ---------------------------------------------------------------- prediction
    def predict(self, **inputs) -> float:
        X = self.build_features(**inputs)
        pred = float(self.model.predict(self._scaled(X))[0])
        return max(pred, 0.0)

    # --------------------------------------------------------------- explanation
    def _get_explainer(self):
        if self._explainer_tried:
            return self._explainer
        self._explainer_tried = True
        try:
            import shap

            self._explainer = shap.TreeExplainer(self.model)
        except Exception:
            self._explainer = None
        return self._explainer

    def explain(self, **inputs) -> Explanation:
        """Per-prediction feature contributions in USD (SHAP for tree models,
        coef × scaled value for linear models, global importance as last resort)."""
        X = self.build_features(**inputs)
        Xs = self._scaled(X)

        explainer = self._get_explainer()
        if explainer is not None:
            try:
                vals = np.asarray(explainer.shap_values(Xs))[0]
                base = float(np.ravel(explainer.expected_value)[0])
                return Explanation(self._group(vals, Xs.columns), base, "SHAP")
            except Exception:
                pass

        if hasattr(self.model, "coef_"):
            vals = np.ravel(self.model.coef_) * Xs.iloc[0].to_numpy()
            base = float(np.ravel(getattr(self.model, "intercept_", 0.0))[0])
            return Explanation(self._group(vals, Xs.columns), base, "Linear coefficients")

        if hasattr(self.model, "feature_importances_"):
            vals = np.asarray(self.model.feature_importances_)
            return Explanation(self._group(vals, Xs.columns), float("nan"), "Global feature importance")

        return Explanation(pd.DataFrame(columns=["feature", "contribution"]), float("nan"), "unavailable")

    @staticmethod
    def _group(values, columns) -> pd.DataFrame:
        df = pd.DataFrame({"feature": [_group_of(c) for c in columns], "contribution": values})
        df = df.groupby("feature", as_index=False)["contribution"].sum()
        return df.reindex(df["contribution"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def factor_summary(exp: "Explanation", inputs: dict, top: int = 3) -> str:
    """Plain-language 'top contributing factors' line for non-technical staff."""
    df = exp.contributions
    if df.empty:
        return ""
    total = float(df["contribution"].abs().sum()) or 1.0
    shown = {
        "Age": str(inputs["age"]), "BMI": f"{inputs['bmi']:.1f}", "Children": str(inputs["children"]),
        "Smoker": inputs["smoker"].capitalize(), "Sex": inputs["sex"].capitalize(),
        "Region": inputs["region"].capitalize(),
    }
    has_sign = exp.method in ("SHAP", "Linear coefficients")
    parts = []
    for _, r in df.head(top).iterrows():
        share = abs(r["contribution"]) / total
        level = "high impact" if share >= 0.35 else "moderate impact" if share >= 0.12 else "low impact"
        extra = (", raises cost" if r["contribution"] > 0 else ", lowers cost") if has_sign else ""
        parts.append(f"{r['feature']}={shown.get(r['feature'], '')} ({level}{extra})")
    return ", ".join(parts)
