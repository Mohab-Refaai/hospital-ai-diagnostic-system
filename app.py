"""Hospital Multi-Model AI Diagnostic & Cost Prediction System — Streamlit app.

Flow (per BRD): Login -> Dashboard with two independent tabs
  Tab 1  Insurance Cost Prediction   (regression, explained with SHAP)
  Tab 2  Chest Disease Detection     (class + bounding box + confidence)
"""
import time

import pandas as pd
import streamlit as st

from services import data_layer
from services.detection_service import DetectionService, describe_location, draw_detections
from services.insurance_service import (
    AGE_RANGE, BMI_RANGE, CHILDREN_RANGE, NOTEBOOK_METRICS, REGION_OPTIONS, SEX_OPTIONS, SMOKER_OPTIONS,
    InsuranceService, factor_summary,
)
from services.validation import ALLOWED_EXT, MAX_UPLOAD_MB, load_xray, validate_insurance_inputs

st.set_page_config(page_title="Hospital AI System", page_icon="🏥", layout="wide", initial_sidebar_state="collapsed")

BLUE = "#1565ff"
st.markdown(
    f"""
    <style>
    .block-container {{padding-top: 1.5rem; max-width: 1100px;}}
    /* login card */
    .login-icon {{width:56px;height:56px;border-radius:50%;background:{BLUE};color:#fff;font-size:30px;
                 display:flex;align-items:center;justify-content:center;margin:8px auto 10px;}}
    .login-title {{text-align:center;font-weight:700;font-size:1.35rem;}}
    .login-sub {{text-align:center;color:#8a8f98;font-size:.85rem;margin-bottom:.6rem;}}
    .login-foot {{text-align:center;color:#a0a4ab;font-size:.75rem;margin-top:.8rem;}}
    /* top bar */
    .st-key-topbar {{background:{BLUE};border-radius:8px;padding:.5rem 1rem;margin-bottom:1rem;}}
    .st-key-topbar p, .st-key-topbar .brand {{color:#fff;margin:0;}}
    .st-key-topbar .brand {{font-weight:700;font-size:1.15rem;line-height:2.2rem;}}
    .st-key-topbar button {{background:transparent;border:none;color:#fff;}}
    .st-key-topbar button p {{color:#fff;}}
    .avatar {{width:34px;height:34px;border-radius:50%;background:#fff;color:{BLUE};font-weight:700;
             display:flex;align-items:center;justify-content:center;margin-left:auto;}}
    /* result panels */
    .rbox {{border-radius:10px;padding:1rem 1.2rem;margin:.6rem 0;border:1px solid;}}
    .rbox.info {{background:#eaf1ff;border-color:#b9d0ff;}}   .rbox.info .rt {{color:{BLUE};}}
    .rbox.bad  {{background:#fff0ef;border-color:#f5c0bc;}}   .rbox.bad .rt {{color:#c0392b;}}
    .rbox.ok   {{background:#eefaf1;border-color:#b6e2c1;}}   .rbox.ok .rt {{color:#1e8e3e;}}
    .rbox .rt {{font-weight:700;margin-bottom:.3rem;}}
    .rbox .rv {{font-size:2.1rem;font-weight:700;margin:.1rem 0 .4rem;}}
    .rbox .rs {{font-size:.85rem;color:#555;}}
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------- cached services
@st.cache_resource
def bootstrap() -> bool:
    """Create DB/tables, seed users from st.secrets['users'] (if set) else demo accounts."""
    data_layer.init_db()
    try:
        secret_users = {k: dict(v) for k, v in st.secrets["users"].items()}
    except Exception:
        secret_users = {}
    if secret_users:
        data_layer.seed_users(secret_users)
    return data_layer.ensure_default_users()


@st.cache_resource(show_spinner="Loading insurance model...")
def get_insurance_service():
    try:
        svc = InsuranceService()
        svc.warmup()
        return svc, None
    except FileNotFoundError:
        return None, "`model_artifacts/insurance_cost_model.joblib` was not found. Run the notebook (Part J) and add the file."
    except Exception as e:
        return None, f"Could not load the insurance model: {e}"


@st.cache_resource(show_spinner="Loading detection model...")
def get_detection_service():
    return DetectionService()


bootstrap()


# ------------------------------------------------------------------- login page
def render_login():
    _, mid, _ = st.columns([1, 1.1, 1])
    with mid:
        st.write("")
        with st.container(border=True):
            st.markdown(
                '<div class="login-icon">✚</div><div class="login-title">Hospital AI System</div>'
                '<div class="login-sub">Sign in to continue</div>',
                unsafe_allow_html=True,
            )
            with st.form("login_form"):
                username = st.text_input("Username", placeholder="e.g. staff_ahmed")
                password = st.text_input("Password", type="password", placeholder="••••••••")
                go = st.form_submit_button("LOG IN", type="primary", width="stretch")
            st.markdown('<div class="login-foot">Authorized hospital staff only</div>', unsafe_allow_html=True)

        if go:
            wait = st.session_state.get("lock_until", 0) - time.time()
            if wait > 0:
                st.error(f"Too many failed attempts. Try again in {int(wait) + 1}s.")
                return
            role = data_layer.authenticate(username, password) if username and password else None
            if role:
                st.session_state.auth = {"username": username.strip(), "role": role}
                st.session_state.fails = 0
                st.rerun()
            st.session_state.fails = st.session_state.get("fails", 0) + 1
            if st.session_state.fails >= 5:
                st.session_state.lock_until, st.session_state.fails = time.time() + 30, 0
            st.error("Invalid username or password.")


# ------------------------------------------------------------------- top bar
def render_topbar(user: dict):
    with st.container(key="topbar"):
        c1, c2, c3 = st.columns([8, 1.3, 0.7], vertical_alignment="center")
        c1.markdown('<span class="brand">Hospital AI System</span>', unsafe_allow_html=True)
        if c2.button("Logout", key="logout_btn"):
            st.session_state.clear()
            st.rerun()
        c3.markdown(f'<div class="avatar" title="{user["username"]}">{user["username"][0].upper()}</div>', unsafe_allow_html=True)


# ------------------------------------------------------- Tab 1: insurance cost
def render_insurance_tab(user: dict):
    ins, err = get_insurance_service()
    st.markdown("#### Patient Details")
    if ins is None:
        st.error(err)
        return

    with st.form("ins_form"):
        left, right = st.columns(2)
        age = left.number_input("Age", AGE_RANGE[0], AGE_RANGE[1], 35, step=1, key="ins_age")
        children = right.number_input("Children", CHILDREN_RANGE[0], CHILDREN_RANGE[1], 2, step=1, key="ins_children")
        sex = left.selectbox("Sex", SEX_OPTIONS, format_func=str.capitalize, key="ins_sex")
        smoker = right.selectbox("Smoker", SMOKER_OPTIONS, format_func=str.capitalize, key="ins_smoker")
        bmi = left.number_input("BMI", BMI_RANGE[0], BMI_RANGE[1], 27.4, step=0.1, format="%.1f", key="ins_bmi")
        region = right.selectbox("Region", REGION_OPTIONS, index=2, format_func=str.capitalize, key="ins_region")
        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        inputs = dict(age=int(age), sex=sex, bmi=float(bmi), children=int(children), smoker=smoker, region=region)
        errors = validate_insurance_inputs(**inputs)
        if errors:
            st.session_state.pop("ins_result", None)
            for e in errors:
                st.error(e)
        else:
            t0 = time.perf_counter()
            pred = ins.predict(**inputs)
            exp = ins.explain(**inputs)
            latency = time.perf_counter() - t0
            st.session_state.ins_result = dict(inputs=inputs, pred=pred, exp=exp, latency=latency,
                                               summary=factor_summary(exp, inputs))
            data_layer.log_prediction(user["username"], "insurance", f"${pred:,.2f}", latency)

    res = st.session_state.get("ins_result")  # kept in session_state -> survives tab switches (FR-2.2)
    if res:
        i = res["inputs"]
        st.markdown(
            f'<div class="rbox info"><div class="rt">Predicted Insurance Charge</div>'
            f'<div class="rv">${res["pred"]:,.2f}</div>'
            f'<div class="rs">Top contributing factors: {res["summary"]}</div></div>',
            unsafe_allow_html=True,
        )
        st.caption(
            f"For: age {i['age']}, {i['sex']}, BMI {i['bmi']:.1f}, {i['children']} children, "
            f"smoker: {i['smoker']}, region: {i['region']} · computed in {res['latency']:.2f}s"
        )
        exp = res["exp"]
        if len(exp.contributions):
            st.markdown("**How each input moved the estimate**")
            top = exp.contributions.iloc[::-1].set_index("feature")
            st.bar_chart(
                pd.DataFrame({"Raises cost": top["contribution"].clip(lower=0),
                              "Lowers cost": top["contribution"].clip(upper=0)}),
                horizontal=True, color=["#d62728", "#2ca02c"],
            )
            note = (f"Effect in USD versus the average prediction (≈ ${exp.base_value:,.0f}); interaction effects "
                    "such as smoker × BMI are counted under Smoker." if exp.method == "SHAP"
                    else f"Explanation method: {exp.method}.")
            st.caption(note)
        st.caption(
            f"Model: {ins.model_name} · test-set R² {NOTEBOOK_METRICS['R2']:.2f}, MAE ≈ ${NOTEBOOK_METRICS['MAE']:,.0f}. "
            "Estimate only — not a quote."
        )


# ------------------------------------------------------ Tab 2: chest detection
def render_detection_tab(user: dict):
    det = get_detection_service()
    left, right = st.columns([1, 1.1], gap="large")

    with left:
        st.markdown("#### Upload Chest X-Ray")
        up = st.file_uploader("Drag & drop or Browse", type=list(ALLOWED_EXT), key="det_upload",
                              label_visibility="collapsed")
        st.caption(f"JPG / PNG / DICOM · max {MAX_UPLOAD_MB} MB")

        image, err = (None, None)
        if up is not None:
            image, err = load_xray(up)
            if err:
                st.error(err)

        # a new / removed file invalidates the previous result of THIS tab only
        file_key = f"{up.name}:{up.size}" if up is not None else None
        prev = st.session_state.get("det_result")
        if prev and prev["file_key"] != file_key:
            st.session_state.pop("det_result")

        if not det.ready:
            st.info(f"⏳ **Detection model not connected yet.** {det.message}")

        with st.expander("Advanced settings"):
            conf = st.slider("Detection sensitivity (confidence threshold)", 0.05, 0.95, det.default_conf, 0.05,
                             key="det_conf", help="Lower = more findings flagged (more false alarms).")

        clicked = st.button("Analyze", type="primary", key="det_analyze",
                            disabled=(image is None or not det.ready),
                            help=None if det.ready else "Available once the model is connected.")
        if clicked and image is not None:
            with st.spinner("Analyzing..."):
                try:
                    t0 = time.perf_counter()
                    result = det.predict(image, conf)
                    latency = time.perf_counter() - t0
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
                    result = None
            if result is not None:
                st.session_state.det_result = dict(
                    file_key=file_key, result=result, latency=latency, size=image.size,
                    annotated=draw_detections(image, result.findings) if result.has_boxes else image,
                )
                summary = ("; ".join(f"{f.label} {f.confidence:.2f}" for f in result.findings)
                           or "No abnormal finding")
                data_layer.log_prediction(user["username"], "chest_detection", summary, latency)

        res = st.session_state.get("det_result")
        if res:
            r, findings = res["result"], res["result"].findings
            if not findings:
                st.markdown(
                    '<div class="rbox ok"><div class="rt">Detection Result</div>'
                    '<div class="rv" style="font-size:1.3rem">No abnormal finding detected</div>'
                    '<div class="rs">Decision support only — a radiologist should still review the image.</div></div>',
                    unsafe_allow_html=True,
                )
            else:
                f = findings[0]
                loc = describe_location(f.box, res["size"]) if f.box else "Not available (classification model)"
                more = f" · {len(findings)} findings in total" if len(findings) > 1 else ""
                st.markdown(
                    f'<div class="rbox bad"><div class="rt">Detection Result{more}</div>'
                    f'<div class="rs" style="color:#222;font-size:.95rem">Class: <b>{f.label}</b><br>'
                    f'Confidence: <b>{f.confidence * 100:.0f}%</b><br>Location: {loc}</div>'
                    f'<div class="rs" style="margin-top:.5rem">Decision support only — refer to radiologist</div></div>',
                    unsafe_allow_html=True,
                )
                if len(findings) > 1:
                    st.dataframe(pd.DataFrame(
                        [{"Class": x.label, "Confidence": f"{x.confidence * 100:.0f}%",
                          "Location": describe_location(x.box, res["size"]) if x.box else "—"} for x in findings]),
                        hide_index=True, width="stretch")
            if r.scores:
                with st.expander("Class probabilities"):
                    st.dataframe(pd.DataFrame([(a, f"{b * 100:.1f}%") for a, b in r.scores], columns=["Class", "Probability"]),
                                 hide_index=True, width="stretch")
            st.caption(f"Analysis time: {res['latency']:.2f}s")

    with right:
        res = st.session_state.get("det_result")
        if res:
            st.image(res["annotated"], caption="Analyzed image", width="stretch")
        elif image is not None:
            st.image(image, caption="Uploaded image", width="stretch")


# ------------------------------------------------------------ admin sidebar
def render_admin_panel():
    with st.sidebar:
        st.subheader("Admin")
        if data_layer.uses_default_password("admin"):
            st.warning("The default admin password is still active. Define users in Streamlit secrets (see README).")
        with st.expander("Users", expanded=False):
            st.dataframe(pd.DataFrame(data_layer.list_users()), hide_index=True, width="stretch")
            with st.form("add_user", clear_on_submit=True):
                nu = st.text_input("New username")
                npw = st.text_input("Password (min 8 chars)", type="password")
                nrole = st.selectbox("Role", data_layer.ROLES)
                if st.form_submit_button("Add user"):
                    try:
                        data_layer.create_user(nu.strip(), npw, nrole)
                        st.success(f"User '{nu}' created.")
                    except ValueError as e:
                        st.error(str(e))
        with st.expander("Usage & prediction log", expanded=False):
            st.dataframe(pd.DataFrame(data_layer.usage_summary()), hide_index=True, width="stretch")
            st.dataframe(pd.DataFrame(data_layer.recent_logs(50)), hide_index=True, width="stretch")
        st.caption("Note: on Streamlit Community Cloud the SQLite file resets on restart; users from secrets are re-created.")


# ------------------------------------------------------------------------ main
user = st.session_state.get("auth")
if not user:
    render_login()
    st.stop()

render_topbar(user)
if user["role"] == "admin":
    render_admin_panel()

tab_ins, tab_det = st.tabs(["Insurance Cost Prediction", "Chest Disease Detection"])
with tab_ins:
    render_insurance_tab(user)
with tab_det:
    render_detection_tab(user)

st.caption("Decision-support tool for educational use — not a certified diagnostic device and not a substitute for a doctor's judgment.")
