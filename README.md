# Hospital Multi-Model AI Diagnostic & Cost Prediction System

Streamlit app built to the project BRD: **Login → dashboard with two independent tabs**.

| Tab | Model | Output |
|---|---|---|
| Insurance Cost Prediction | XGBoost (from `notebooks/Insurance_Cost_Full_Pipeline.ipynb`) | predicted charge + top factors + SHAP chart |
| Chest Disease Detection | YOLO / classifier (**added later**, see `model_artifacts/cv/README.md`) | image with bounding boxes, class, confidence, location; "No abnormal finding detected" when clean |

## Structure
```
app.py                               UI: login, top bar, two tabs, admin sidebar
requirements.txt
services/
  data_layer.py                      SQLite: hashed users + prediction log      (Data Layer)
  insurance_service.py               Model Service 1 (regression + SHAP)
  detection_service.py               Model Service 2 (YOLO ONNX / ultralytics / classifier)
  validation.py                      input validation for both tabs
model_artifacts/
  insurance_cost_model.joblib        <- produced by the notebook (Part J)   ** you must add **
  cv/                                <- put the detection model + config.json here later
data/                                SQLite DB is created here at runtime (git-ignored)
notebooks/                           the training notebook
.streamlit/config.toml, secrets.toml.example
```

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Users & login
* Passwords are stored **hashed** (PBKDF2-SHA256 + salt) in SQLite — never in plain text.
* Production: define users in Streamlit secrets (see `.streamlit/secrets.toml.example`; on Streamlit Cloud: *App settings → Secrets*).
* If no users are defined, **demo accounts** are created for evaluation: `admin / Admin@12345` (role admin) and `staff_ahmed / Staff@12345` (role staff).
  The admin sidebar warns while the default admin password is active. **Change these before a public deployment.**
* Admin users get a sidebar to add users and view usage / the prediction log (user, tab, result, latency — no patient data or images).
* Streamlit Cloud's disk is ephemeral: users added through the admin panel disappear on restart; users from secrets are re-created.

## Library versions (important)
`insurance_cost_model.joblib` is a pickle: scikit-learn / xgboost versions here must match the training environment.
In the notebook environment run `pip freeze | grep -i -E "scikit-learn|xgboost|joblib|numpy|pandas"` and pin them in `requirements.txt`.

## Deploy
Push to GitHub → share.streamlit.io → New app → pick the repo, branch, `app.py` → add Secrets.

> Decision-support tool for educational use — not a certified diagnostic device.
