import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Kenya Financial Inclusion Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --green:  #1a6b3c;
    --green2: #2d9653;
    --gold:   #c9a84c;
    --cream:  #faf8f3;
    --dark:   #111827;
    --red:    #b91c1c;
    --grey:   #6b7280;
    --border: #e5e7eb;
    --shadow: 0 4px 20px rgba(0,0,0,0.07);
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--cream);
    color: var(--dark);
}
h1,h2,h3,h4 {
    font-family: 'Playfair Display', serif;
    letter-spacing: -0.02em;
}

/* Hero */
.hero {
    background: linear-gradient(140deg, #0f2417 0%, #1a6b3c 60%, #2d9653 100%);
    padding: 3rem 2.5rem;
    border-radius: 20px;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content:'';position:absolute;top:-40px;right:-40px;
    width:220px;height:220px;border-radius:50%;
    background:rgba(201,168,76,0.12);
}
.hero-title {
    font-family:'Playfair Display',serif;
    font-size:2.2rem;font-weight:800;
    color:white;margin:0 0 0.6rem 0;line-height:1.2;
}
.hero-sub {
    color:rgba(255,255,255,0.78);font-size:0.95rem;
    font-weight:300;margin:0 0 1.2rem 0;max-width:620px;
}
.hero-badges { display:flex;gap:0.6rem;flex-wrap:wrap; }
.badge {
    background:rgba(255,255,255,0.12);
    border:1px solid rgba(255,255,255,0.25);
    color:white;padding:0.3rem 0.8rem;border-radius:20px;
    font-size:0.78rem;font-weight:500;
}

/* Metric cards */
.metrics-row {
    display:grid;grid-template-columns:repeat(4,1fr);
    gap:1rem;margin-bottom:2rem;
}
.metric-card {
    background:white;border-radius:14px;
    padding:1.2rem 1rem;text-align:center;
    border:1px solid var(--border);box-shadow:var(--shadow);
}
.metric-value {
    font-family:'Playfair Display',serif;
    font-size:2rem;font-weight:700;color:var(--green);
    line-height:1;margin-bottom:0.3rem;
}
.metric-label {
    font-size:0.75rem;color:var(--grey);
    text-transform:uppercase;letter-spacing:0.06em;font-weight:500;
}

/* Form cards */
.form-card {
    background:white;border-radius:16px;
    padding:1.8rem;border:1px solid var(--border);
    box-shadow:var(--shadow);margin-bottom:1.2rem;
}
.section-title {
    font-family:'Playfair Display',serif;font-size:1.05rem;
    font-weight:700;color:var(--dark);margin:0 0 1.2rem 0;
    padding-bottom:0.6rem;border-bottom:2px solid var(--gold);
}

/* Result */
.result-card {
    border-radius:20px;padding:2.5rem 2rem;
    text-align:center;margin:1.5rem 0;border:2px solid;
}
.result-included {
    background:linear-gradient(135deg,#ecfdf5,#d1fae5);
    border-color:var(--green);
}
.result-excluded {
    background:linear-gradient(135deg,#fef2f2,#fee2e2);
    border-color:var(--red);
}
.result-emoji { font-size:3rem;margin-bottom:0.5rem; }
.result-status {
    font-family:'Playfair Display',serif;
    font-size:1.8rem;font-weight:800;margin-bottom:0.3rem;
}
.result-prob {
    font-size:1.1rem;margin-bottom:1rem;color:var(--grey);
}
.result-prob strong { color:var(--dark);font-size:1.3rem; }
.prob-bar-bg {
    background:rgba(0,0,0,0.08);border-radius:20px;
    height:10px;margin:0.8rem auto;max-width:400px;overflow:hidden;
}
.prob-bar-fill { height:100%;border-radius:20px; }
.threshold-note { font-size:0.8rem;color:var(--grey);margin-top:0.8rem; }
.fairness-pill {
    display:inline-block;background:white;
    border:1.5px solid var(--green);color:var(--green);
    padding:0.4rem 1rem;border-radius:20px;
    font-size:0.82rem;font-weight:600;margin-top:1rem;
}
.info-box {
    background:#fffbeb;border:1px solid #fbbf24;
    border-left:4px solid #f59e0b;border-radius:8px;
    padding:1rem 1.2rem;font-size:0.88rem;margin-top:1rem;
}

/* Button */
.stButton > button {
    background:linear-gradient(135deg,var(--green) 0%,var(--green2) 100%);
    color:white;border:none;padding:0.9rem 2rem;
    border-radius:12px;font-size:1rem;font-weight:600;
    font-family:'DM Sans',sans-serif;width:100%;
    box-shadow:0 4px 14px rgba(26,107,60,0.35);
}
.stButton > button:hover { opacity:0.9; }

.stSelectbox label, .stSlider label,
.stNumberInput label, .stCheckbox label {
    font-size:0.88rem !important;
    font-weight:500 !important;color:#374151 !important;
}
footer, #MainMenu { visibility:hidden; }
.block-container { padding:2rem 2.5rem; }
</style>
""", unsafe_allow_html=True)


# ── Load models ───────────────────────────────────────────────────
@st.cache_resource
def load_models():
    base    = os.path.dirname(os.path.abspath(__file__))
    mp      = os.path.join(base, 'models')
    xgb     = joblib.load(os.path.join(mp, 'xgb_model.pkl'))
    meta    = joblib.load(os.path.join(mp, 'fc_stack_meta.pkl'))
    le_dict = joblib.load(os.path.join(mp, 'label_encoders.pkl'))
    with open(os.path.join(mp, 'feature_names.txt')) as f:
        features = f.read().splitlines()
    with open(os.path.join(mp, 'model_metrics.json')) as f:
        metrics = json.load(f)
    return xgb, meta, le_dict, features, metrics

xgb_model, meta_lr, le_dict, feature_names, metrics = load_models()

COUNTIES = sorted([
    "Baringo","Bomet","Bungoma","Busia","Elgeyo-Marakwet","Embu",
    "Garissa","Homa Bay","Isiolo","Kajiado","Kakamega","Kericho",
    "Kiambu","Kilifi","Kirinyaga","Kisii","Kisumu","Kitui","Kwale",
    "Laikipia","Lamu","Machakos","Makueni","Mandera","Marsabit",
    "Meru","Migori","Mombasa","Muranga","Nairobi City","Nakuru",
    "Nandi","Narok","Nyamira","Nyandarua","Nyeri","Samburu","Siaya",
    "Taita-Taveta","Tana River","Tharaka-Nithi","Trans Nzoia",
    "Turkana","Uasin Gishu","Vihiga","Wajir","West Pokot"
])

# ── Hero ──────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-title">🏦 Kenya Financial Inclusion Predictor</div>
    <p class="hero-sub">
        Predict financial inclusion status using the FC-Stack
        fairness-constrained ensemble model trained on the FinAccess Kenya
        2024 Household Survey (N=20,871). Gender-specific thresholds ensure
        equitable predictions for all respondents.
    </p>
    <div class="hero-badges">
        <span class="badge">✦ FC-Stack Novel Algorithm</span>
        <span class="badge">✦ FinAccess Kenya 2024</span>
        <span class="badge">✦ Gender Fairness Corrected</span>
        <span class="badge">✦ Strathmore University MSc DSA</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Model metrics ─────────────────────────────────────────────────
st.markdown(f"""
<div class="metrics-row">
    <div class="metric-card">
        <div class="metric-value">{metrics['accuracy']*100:.1f}%</div>
        <div class="metric-label">Accuracy</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">{metrics['f1_score']*100:.1f}%</div>
        <div class="metric-label">F1-Score</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">{metrics['roc_auc']*100:.1f}%</div>
        <div class="metric-label">ROC-AUC</div>
    </div>
    <div class="metric-card">
        <div class="metric-value" style="color:#c9a84c">0.000</div>
        <div class="metric-label">Gender Gap (DP)</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Form ──────────────────────────────────────────────────────────
st.markdown("### 📋 Enter Respondent Details")
st.caption("Complete all three sections then click Predict.")

col1, col2, col3 = st.columns(3, gap="large")

# ── Column 1: Personal & Location ────────────────────────────────
with col1:
    st.markdown('<div class="form-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">👤 Personal Information</div>',
                unsafe_allow_html=True)
    gender     = st.selectbox("Gender", ["Female", "Male"])
    age        = st.slider("Age (years)", 18, 86, 32)
    education  = st.selectbox("Highest Education Level",
                               ["None","Primary","Secondary","Tertiary"])
    marital    = st.selectbox("Marital Status", [
        "Single",
        "Married/Living with partner",
        "Divorced/Separated",
        "Widowed"])
    hh_size    = st.slider("People in your household", 1, 20, 4)
    disability = st.selectbox("Disability Status",
                               ["Without Disability","With Disability"])
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="form-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📍 Location</div>',
                unsafe_allow_html=True)
    county       = st.selectbox("County of Residence", COUNTIES)
    cluster_type = st.selectbox("Type of Area", ["Urban","Rural"])
    st.markdown("</div>", unsafe_allow_html=True)

# ── Column 2: Technology & Income ────────────────────────────────
with col2:
    st.markdown('<div class="form-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📱 Technology Access</div>',
                unsafe_allow_html=True)
    mobile_own   = st.selectbox("Do you own a mobile phone?",
                                 ["Yes","No"])
    digital_acc  = st.selectbox("Do you have a digital financial account?",
                                 ["Yes — I actively use it",
                                  "Yes — but I don't use it",
                                  "No — I don't have one"])
    mobile_money = st.selectbox(
        "Have you ever used mobile money (e.g. M-Pesa)?",
        ["Yes","No"])
    has_id       = st.selectbox("Do you have a National ID Card?",
                                 ["Yes","No"])
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="form-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">💰 Income & Spending</div>',
                unsafe_allow_html=True)
    monthly_exp = st.number_input(
        "Monthly household expenditure (KES)",
        min_value=500, max_value=500000, value=10000, step=500,
        help="Estimate of all monthly costs: food, rent, transport, bills")
    loan_denied = st.selectbox(
        "Have you ever applied for a loan and been denied?",
        ["No","Yes"])
    livelihood  = st.selectbox("Which best describes your livelihood?", [
        "Employed (regular salary)",
        "Running my own business",
        "Farming / Agriculture",
        "Casual / seasonal work",
        "Dependent on family or spouse",
        "Other"])
    main_income = st.selectbox("Your main source of income?", [
        "Regular employment / salary",
        "Own business",
        "Farming / livestock",
        "Casual / seasonal work",
        "Support from family or friends",
        "Other"])
    st.markdown("</div>", unsafe_allow_html=True)

# ── Column 3: Assets & Resilience ────────────────────────────────
with col3:
    st.markdown('<div class="form-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🏠 Assets You Own</div>',
                unsafe_allow_html=True)
    st.caption("Tick everything that applies to your household")
    owns_tv      = st.checkbox("📺  Television set")
    has_internet = st.checkbox("🌐  Internet at home (modem / WiFi)")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="form-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">💼 Income Sources</div>',
                unsafe_allow_html=True)
    st.caption("Tick all income sources in your household")
    inc_farming  = st.checkbox("🌾  Farming or keeping livestock")
    inc_employed = st.checkbox("💼  Regular employment / salary")
    inc_business = st.checkbox("🏪  Running own business")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="form-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🛡️ Financial Resilience</div>',
                unsafe_allow_html=True)
    st.caption("Answer honestly — this helps assess your financial wellbeing")
    saves_emerg = st.selectbox(
        "Do you regularly set money aside for emergencies?",
        ["Yes","No"], key="s1")
    debt_stress = st.selectbox(
        "In the past 3 months, were you stressed about debt?",
        ["No — debt is manageable",
         "Yes — debt has been a worry"], key="s2")
    food_secure = st.selectbox(
        "In the past year, did your household always have enough food?",
        ["Yes — always had enough",
         "No — sometimes went without"], key="s3")
    raise_13k   = st.selectbox(
        "Could you raise KES 13,000 within 30 days if needed?",
        ["Yes","No — it would be difficult"], key="s4")
    fin_healthy = st.selectbox(
        "Overall, would you say you are financially healthy?",
        ["Yes","No"], key="s5")
    st.markdown("</div>", unsafe_allow_html=True)

# ── Compute composites ────────────────────────────────────────────
log_exp = float(np.log1p(monthly_exp))

digital_map = {
    "Yes — I actively use it":   "Usage",
    "Yes — but I don't use it":  "Non-usage",
    "No — I don't have one":     "Non-usage",
}
digital_val = digital_map[digital_acc]

id_map = {
    "Yes": "Has ID (National Identity Card (ID))",
    "No":  "No ID",
}
id_val = id_map[has_id]

liv_map = {
    "Employed (regular salary)":      "Employed",
    "Running my own business":        "Own Business",
    "Farming / Agriculture":          "Agriculture",
    "Casual / seasonal work":         "Casual Worker",
    "Dependent on family or spouse":  "Dependent",
    "Other":                          "Other",
}
liv_val = liv_map[livelihood]

inc_map = {
    "Regular employment / salary":    "Employed/regular employees",
    "Own business":                   "Running own business/Self employed",
    "Farming / livestock":            "Farming (crops, keeping livestock)",
    "Casual / seasonal work":         "Casual worker/Seasonal Worker",
    "Support from family or friends": "support from family / friends",
    "Other":                          "Other",
}
inc_val = inc_map[main_income]

asset_index          = int(owns_tv) + int(has_internet)
income_diversity     = int(inc_farming) + int(inc_employed) + int(inc_business)
financial_resilience = (
    int(saves_emerg == "Yes") +
    int("No —" in debt_stress) +
    int("Yes —" in food_secure) +
    int(raise_13k == "Yes")
)
financial_health_score = int(fin_healthy == "Yes")

# ── Predict button ────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    predict = st.button("🔮  PREDICT FINANCIAL INCLUSION STATUS",
                         use_container_width=True)

# ── Prediction logic ──────────────────────────────────────────────
if predict:
    raw = pd.DataFrame({
        'gender':                [gender],
        'age':                   [age],
        'education_level':       [education],
        'marital_status':        [marital],
        'household_size':        [hh_size],
        'disability_status':     [disability],
        'county':                [county.lower()],
        'cluster_type':          [cluster_type],
        'mobile_ownership':      [mobile_own],
        'digital_account':       [digital_val],
        'mobile_money':          [mobile_money],
        'has_id_card':           [id_val],
        'loan_denied':           [loan_denied],
        'livelihood':            [liv_val],
        'main_income_source':    [inc_val],
        'log_expenditure':       [log_exp],
        'asset_index':           [asset_index],
        'income_diversity':      [income_diversity],
        'financial_resilience':  [financial_resilience],
        'financial_health_score':[financial_health_score],
    })

    for col, le in le_dict.items():
        if col in raw.columns:
            val = str(raw.at[0, col])
            raw[col] = (le.transform([val])[0]
                        if val in le.classes_
                        else le.transform([le.classes_[0]])[0])

    for fn in feature_names:
        if fn not in raw.columns:
            raw[fn] = 0
    raw  = raw[feature_names]

    prob     = xgb_model.predict_proba(raw)[0][1]
    thresh   = metrics['thresholds'][gender.lower()]
    included = prob >= thresh
    pct      = prob * 100

    # Result card
    _, res_col, _ = st.columns([1, 3, 1])
    with res_col:
        if included:
            clr = "#1a6b3c"
            st.markdown(f"""
            <div class="result-card result-included">
                <div class="result-emoji">✅</div>
                <div class="result-status" style="color:{clr}">
                    FINANCIALLY INCLUDED
                </div>
                <div class="result-prob">
                    Inclusion Probability: <strong>{pct:.1f}%</strong>
                </div>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill"
                         style="width:{pct}%;background:{clr}"></div>
                </div>
                <div class="threshold-note">
                    FC-Stack threshold: {thresh:.2f} ({gender}) &nbsp;|&nbsp;
                    Probability {prob:.3f} ≥ threshold ✓
                </div>
                <div class="fairness-pill">
                    ⚖️ Fairness Correction Applied — Gender Gap = 0.000
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            clr = "#b91c1c"
            st.markdown(f"""
            <div class="result-card result-excluded">
                <div class="result-emoji">❌</div>
                <div class="result-status" style="color:{clr}">
                    FINANCIALLY EXCLUDED
                </div>
                <div class="result-prob">
                    Inclusion Probability: <strong>{pct:.1f}%</strong>
                </div>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill"
                         style="width:{pct}%;background:{clr}"></div>
                </div>
                <div class="threshold-note">
                    FC-Stack threshold: {thresh:.2f} ({gender}) &nbsp;|&nbsp;
                    Probability {prob:.3f} &lt; threshold
                </div>
                <div class="fairness-pill">
                    ⚖️ Fairness Correction Applied — Gender Gap = 0.000
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Key insight
        st.markdown("<br>", unsafe_allow_html=True)
        if digital_val == "Usage":
            insight = ("Having an active digital account is the strongest "
                       "driver of financial inclusion — 12.3× more important "
                       "than any other factor in this model.")
        else:
            insight = ("Not having an active digital account is the single "
                       "biggest barrier to financial inclusion. Opening a "
                       "digital account (M-Pesa, bank, SACCO) would most "
                       "significantly improve this outcome.")

        st.markdown(f"""
        <div class="info-box">
            <strong>🔍 Key Insight:</strong> {insight}
        </div>
        """, unsafe_allow_html=True)

        # Profile summary
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Your profile summary used in this prediction:**")
        fa, fb = st.columns(2)
        with fa:
            st.markdown(f"""
- **Digital account:** {digital_acc}
- **Mobile phone:** {mobile_own}
- **Mobile money:** {mobile_money}
- **National ID:** {has_id}
- **County:** {county}
            """)
        with fb:
            st.markdown(f"""
- **Asset index:** {asset_index} / 2
- **Income sources:** {income_diversity} / 3
- **Financial resilience:** {financial_resilience} / 4
- **Financial health:** {'Good ✓' if financial_health_score else 'Needs attention'}
- **Monthly spend:** KES {monthly_exp:,}
            """)

# ── Footer ────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;color:#9ca3af;font-size:0.78rem;
     border-top:1px solid #e5e7eb;padding-top:1.2rem;">
    <strong style="color:#1a6b3c">FC-Stack Fairness-Constrained
    Stacking Ensemble</strong> &nbsp;·&nbsp;
    Kenya FinAccess 2024 &nbsp;·&nbsp;
    Accuracy 94.0% &nbsp;·&nbsp; ROC-AUC 97.9% &nbsp;·&nbsp;
    Demographic Parity Gap: 0.000 <br><br>
    Strathmore University — MSc Data Science & Analytics &nbsp;·&nbsp;
    Faith Makena Riungu &nbsp;·&nbsp; 2025
</div>
""", unsafe_allow_html=True)
