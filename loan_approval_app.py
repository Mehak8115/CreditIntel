import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CreditIntel · Random Forest Engine",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500;600&display=swap');

  :root {
    --navy:  #0d1b2a;
    --mid:   #1b2d45;
    --teal:  #00c9b1;
    --amber: #f5a623;
    --red:   #e74c3c;
    --green: #27ae60;
    --light: #e8edf3;
    --muted: #8fa3b8;
  }

  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: var(--navy);
    color: var(--light);
  }
  [data-testid="stSidebar"] {
    background: var(--mid) !important;
    border-right: 1px solid rgba(0,201,177,0.15);
  }
  [data-testid="stSidebar"] * { color: var(--light) !important; }
  .main { background: var(--navy); }
  h1,h2,h3 { font-family: 'Playfair Display', serif !important; }

  .metric-card {
    background: var(--mid);
    border: 1px solid rgba(0,201,177,0.2);
    border-radius: 12px;
    padding: 1.1rem 1rem;
    text-align: center;
  }
  .metric-card .label { font-size:.72rem; color:var(--muted); text-transform:uppercase; letter-spacing:.08em; margin-bottom:.3rem; }
  .metric-card .value { font-size:1.5rem; font-weight:700; color:var(--teal); }

  .section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.15rem;
    color: var(--teal);
    border-bottom: 1px solid rgba(0,201,177,0.25);
    padding-bottom: .4rem;
    margin-bottom: 1rem;
  }

  .verdict-approved {
    background: linear-gradient(135deg, rgba(39,174,96,.15), rgba(0,201,177,.1));
    border: 1.5px solid var(--green);
    border-radius: 14px;
    padding: 1.5rem 2rem;
    text-align: center;
  }
  .verdict-rejected {
    background: linear-gradient(135deg, rgba(231,76,60,.15), rgba(245,166,35,.08));
    border: 1.5px solid var(--red);
    border-radius: 14px;
    padding: 1.5rem 2rem;
    text-align: center;
  }
  .verdict-title { font-family:'Playfair Display',serif; font-size:2rem; font-weight:900; margin:0; }

  .pill {
    display:inline-block;
    background:rgba(0,201,177,.12);
    border:1px solid rgba(0,201,177,.3);
    border-radius:20px;
    padding:.2rem .75rem;
    font-size:.78rem;
    color:var(--teal);
    margin:.2rem .2rem;
  }

  .footer {
    margin-top:3rem;
    padding:1.5rem 2rem;
    border-top:1px solid rgba(0,201,177,.15);
    text-align:center;
    color:var(--muted);
    font-size:.8rem;
  }

  .stButton>button {
    background: linear-gradient(135deg, var(--teal), #00a896);
    color: #0d1b2a !important;
    font-weight: 700;
    border: none;
    border-radius: 8px;
    padding: .6rem 2.5rem;
    font-size: 1rem;
    width: 100%;
    transition: transform .15s;
  }
  .stButton>button:hover { transform: translateY(-2px); }

  .summary-row { display:flex; justify-content:space-between; padding:.45rem 0; border-bottom:1px solid rgba(255,255,255,.06); font-size:.88rem; }
  .summary-row .key { color:var(--muted); }
  .summary-row .val { font-weight:600; }

  .feat-row {
    background:rgba(0,201,177,.05);
    border-left:3px solid #00c9b1;
    padding:.45rem .8rem;
    border-radius:0 6px 6px 0;
    margin-bottom:.4rem;
  }
  .feat-name { font-size:.86rem; font-weight:600; color:#e8edf3; }
  .feat-desc { font-size:.75rem; color:#8fa3b8; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# RANDOM FOREST SIMULATION ENGINE
# ─────────────────────────────────────────────
def compute_rf_score(inputs: dict) -> dict:
    score = 0.0

    # CIBIL Score — 28 pts
    cibil = inputs["cibil"]
    if cibil >= 750:   cibil_pts = 28.0
    elif cibil >= 700: cibil_pts = 21.0
    elif cibil >= 650: cibil_pts = 13.0
    elif cibil >= 600: cibil_pts = 6.0
    else:              cibil_pts = 0.0
    score += cibil_pts

    # Annual Income — 18 pts
    inc = inputs["annual_income"]
    if inc >= 1_000_000:   inc_pts = 18.0
    elif inc >= 500_000:   inc_pts = 14.0
    elif inc >= 300_000:   inc_pts = 10.0
    elif inc >= 150_000:   inc_pts = 6.0
    else:                  inc_pts = 2.0
    score += inc_pts

    # Loan/Income Ratio — 14 pts
    loan_inc_ratio = inputs["loan"] / inputs["annual_income"] if inputs["annual_income"] > 0 else 99
    if loan_inc_ratio <= 3:    lir_pts = 14.0
    elif loan_inc_ratio <= 6:  lir_pts = 10.0
    elif loan_inc_ratio <= 10: lir_pts = 5.0
    elif loan_inc_ratio <= 15: lir_pts = 2.0
    else:                      lir_pts = 0.0
    score += lir_pts

    # Asset Coverage — 16 pts
    total_assets = (inputs["residential_asset"] + inputs["commercial_asset"] +
                    inputs["luxury_asset"] + inputs["bank_asset"])
    asset_coverage = total_assets / inputs["loan"] if inputs["loan"] > 0 else 0
    if asset_coverage >= 2.5:   asset_pts = 16.0
    elif asset_coverage >= 1.5: asset_pts = 12.0
    elif asset_coverage >= 1.0: asset_pts = 7.0
    elif asset_coverage >= 0.5: asset_pts = 3.0
    else:                        asset_pts = 0.0
    score += asset_pts

    # Loan Term — 8 pts
    term = inputs["loan_term"]
    if term <= 6:    term_pts = 8.0
    elif term <= 12: term_pts = 6.0
    elif term <= 18: term_pts = 4.0
    else:            term_pts = 2.0
    score += term_pts

    # Dependents — 8 pts
    dep = inputs["dependent"]
    dep_pts = max(0.0, 8.0 - dep * 2.0)
    score += dep_pts

    # Education — 4 pts
    edu_pts = 4.0 if inputs["education"] == "Graduate" else 2.0
    score += edu_pts

    # Self Employed — 2 pts
    self_emp_pts = 2.0 if inputs["self_employed"] == "No" else 0.0
    score += self_emp_pts

    score = min(score, 100.0)
    approved = score >= 55.0
    prob_approve = round(score, 2)
    prob_reject  = round(100.0 - prob_approve, 2)

    factors = {
        "CIBIL Score":       {"pts": cibil_pts,   "max": 28, "val": cibil},
        "Annual Income":     {"pts": inc_pts,     "max": 18, "val": f"₹{inc:,.0f}"},
        "Loan/Income Ratio": {"pts": lir_pts,     "max": 14, "val": f"{loan_inc_ratio:.1f}x"},
        "Asset Coverage":    {"pts": asset_pts,   "max": 16, "val": f"{asset_coverage:.2f}x"},
        "Loan Term":         {"pts": term_pts,    "max": 8,  "val": f"{term} yrs"},
        "Dependents":        {"pts": dep_pts,     "max": 8,  "val": dep},
        "Education":         {"pts": edu_pts,     "max": 4,  "val": inputs["education"]},
        "Employment Type":   {"pts": self_emp_pts,"max": 2,  "val": inputs["self_employed"]},
    }

    reasons = []
    if cibil >= 700:   reasons.append(("✅", f"CIBIL score {cibil} is strong (≥700)"))
    elif cibil >= 650: reasons.append(("⚠️", f"CIBIL score {cibil} is fair — aim for 700+"))
    else:              reasons.append(("❌", f"CIBIL score {cibil} is below acceptable threshold (650)"))

    if inc >= 500_000:  reasons.append(("✅", f"Annual income ₹{inc:,.0f} is well above average"))
    elif inc >= 200_000:reasons.append(("✅", f"Annual income ₹{inc:,.0f} meets requirements"))
    else:               reasons.append(("⚠️", f"Annual income ₹{inc:,.0f} is on the lower side"))

    if loan_inc_ratio <= 6: reasons.append(("✅", f"Loan-to-income ratio {loan_inc_ratio:.1f}x is healthy"))
    else:                   reasons.append(("⚠️", f"Loan-to-income ratio {loan_inc_ratio:.1f}x is high — consider a smaller loan"))

    if asset_coverage >= 1.5:  reasons.append(("✅", f"Total asset coverage {asset_coverage:.2f}x covers loan well"))
    elif asset_coverage >= 1.0:reasons.append(("⚠️", f"Asset coverage {asset_coverage:.2f}x is marginal"))
    else:                       reasons.append(("❌", f"Insufficient asset coverage ({asset_coverage:.2f}x) for loan amount"))

    if dep >= 4: reasons.append(("⚠️", f"{dep} dependents increase financial obligations"))
    if inputs["self_employed"] == "Yes": reasons.append(("⚠️", "Self-employed status carries higher income variability"))

    return {
        "score": round(score, 1),
        "prob_approve": prob_approve,
        "prob_reject":  prob_reject,
        "approved": approved,
        "factors": factors,
        "reasons": reasons,
        "loan_inc_ratio": loan_inc_ratio,
        "asset_coverage": asset_coverage,
        "total_assets": total_assets,
    }


# ─────────────────────────────────────────────
# PLOT CONSTANTS
# ─────────────────────────────────────────────
PLOT_BG  = "rgba(0,0,0,0)"
GRID_CLR = "rgba(255,255,255,0.06)"
FONT_CLR = "#8fa3b8"
ACCENT   = "#00c9b1"


def gauge_chart(pct: float, approved: bool):
    color = "#27ae60" if approved else "#e74c3c"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"size": 40, "color": "#e8edf3", "family": "Playfair Display"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": FONT_CLR, "tickfont": {"color": FONT_CLR}},
            "bar":  {"color": color, "thickness": 0.25},
            "bgcolor": "rgba(27,45,69,0.8)", "borderwidth": 0,
            "steps": [
                {"range": [0, 40],  "color": "rgba(231,76,60,0.15)"},
                {"range": [40, 55], "color": "rgba(245,166,35,0.15)"},
                {"range": [55, 100],"color": "rgba(39,174,96,0.12)"},
            ],
            "threshold": {"line": {"color": color, "width": 3}, "thickness": 0.8, "value": pct},
        },
    ))
    fig.update_layout(paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
                      height=280, margin=dict(t=20,b=10,l=20,r=20), font={"color":FONT_CLR})
    return fig


def factor_bar(factors: dict):
    labels = list(factors.keys())
    earned = [v["pts"] for v in factors.values()]
    gaps   = [v["max"] - v["pts"] for v in factors.values()]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=labels, x=earned, name="Score Earned", orientation="h",
                         marker_color=ACCENT, marker_line_width=0))
    fig.add_trace(go.Bar(y=labels, x=gaps, name="Remaining", orientation="h",
                         marker_color="rgba(255,255,255,0.07)", marker_line_width=0))
    fig.update_layout(barmode="stack", paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
                      xaxis=dict(showgrid=True, gridcolor=GRID_CLR, color=FONT_CLR, title="Points"),
                      yaxis=dict(color=FONT_CLR),
                      legend=dict(font=dict(color=FONT_CLR), bgcolor="rgba(0,0,0,0)"),
                      height=300, margin=dict(t=10,b=30,l=10,r=10), font={"color":FONT_CLR})
    return fig


def risk_radar(factors: dict):
    cats   = list(factors.keys())
    earned = [v["pts"]/v["max"]*100 for v in factors.values()]
    cats  += [cats[0]]; earned += [earned[0]]
    fig = go.Figure(go.Scatterpolar(
        r=earned, theta=cats, fill="toself",
        fillcolor="rgba(0,201,177,0.12)",
        line=dict(color=ACCENT, width=2),
        marker=dict(color=ACCENT, size=5),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(27,45,69,0.5)",
            radialaxis=dict(visible=True, range=[0,100], color=FONT_CLR, gridcolor=GRID_CLR),
            angularaxis=dict(color=FONT_CLR, gridcolor=GRID_CLR),
        ),
        paper_bgcolor=PLOT_BG,
        height=300, margin=dict(t=20,b=20,l=20,r=20),
        font={"color":FONT_CLR}, showlegend=False,
    )
    return fig


def asset_breakdown(residential, commercial, luxury, bank):
    labels = ["Residential", "Commercial", "Luxury", "Bank"]
    values = [residential, commercial, luxury, bank]
    colors = [ACCENT, "#5de0c8", "#f5a623", "#27ae60"]
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.55,
        marker=dict(colors=colors, line=dict(color="#0d1b2a", width=2)),
        textfont=dict(color="#e8edf3"), pull=[0.02]*4,
    ))
    fig.update_layout(
        paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
        height=280, margin=dict(t=10,b=10,l=10,r=10),
        font={"color":FONT_CLR},
        legend=dict(font=dict(color=FONT_CLR), bgcolor="rgba(0,0,0,0)"),
        annotations=[dict(text="Assets", x=0.5, y=0.5,
                          font=dict(size=13, color="#e8edf3"), showarrow=False)],
    )
    return fig


def cibil_band_chart(cibil: int):
    bands  = ["Poor (300-549)", "Fair (550-649)", "Good (650-699)", "Very Good (700-749)", "Excellent (750+)"]
    colors = ["#e74c3c",        "#e67e22",         "#f5a623",        "#2ecc71",             "#27ae60"]
    widths = [249,               100,               50,               50,                    151]

    fig = go.Figure()
    for b, c, w in zip(bands, colors, widths):
        fig.add_trace(go.Bar(
            x=[w], y=[""], orientation="h", name=b,
            marker_color=c, marker_line_width=0,
            hovertemplate=f"<b>{b}</b><extra></extra>",
        ))

    pos = max(0, min(cibil - 300, 599))
    fig.add_shape(
        type="line", x0=pos, x1=pos, y0=-0.45, y1=0.45,
        line=dict(color="white", width=2.5, dash="dot"),
    )
    label_x = max(30, min(pos, 570))
    fig.add_annotation(
        x=label_x, y=0.5,
        text=f"<b>You: {cibil}</b>",
        showarrow=False,
        font=dict(color="white", size=11),
        xanchor="center", yanchor="bottom",
        bgcolor="rgba(13,27,42,0.8)",
        borderpad=3,
    )

    tick_offsets = [0, 100, 200, 300, 400, 500, 600]
    tick_labels  = ["300","400","500","600","700","800","900"]

    fig.update_layout(
        barmode="stack",
        paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
        xaxis=dict(
            range=[0, 600],
            showgrid=False,
            tickvals=tick_offsets,
            ticktext=tick_labels,
            color=FONT_CLR,
            title=dict(text="CIBIL Score", font=dict(color=FONT_CLR, size=11)),
        ),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top", y=-1.0,
            xanchor="center", x=0.5,
            font=dict(color=FONT_CLR, size=10),
            bgcolor="rgba(0,0,0,0)",
            traceorder="normal",
            itemwidth=40,
        ),
        height=210,
        margin=dict(t=45, b=90, l=10, r=10),
        font={"color": FONT_CLR},
    )
    return fig


def pct_bar(factors: dict):
    df = pd.DataFrame({
        "Factor": list(factors.keys()),
        "Pct": [round(v["pts"]/v["max"]*100) for v in factors.values()],
    })
    fig = go.Figure(go.Bar(
        x=df["Pct"], y=df["Factor"], orientation="h",
        marker=dict(
            color=df["Pct"],
            colorscale=[[0,"#e74c3c"],[0.5,"#f5a623"],[1,"#27ae60"]],
            cmin=0, cmax=100,
            colorbar=dict(title=dict(text="%", font=dict(color=FONT_CLR)), tickfont=dict(color=FONT_CLR)),
        ),
        text=[f"{p}%" for p in df["Pct"]],
        textposition="inside", insidetextfont=dict(color="white"),
    ))
    fig.update_layout(
        paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
        xaxis=dict(range=[0,100], color=FONT_CLR, gridcolor=GRID_CLR, title="% of Max Score"),
        yaxis=dict(color=FONT_CLR),
        height=300, margin=dict(t=10,b=30,l=10,r=10), font={"color":FONT_CLR},
    )
    return fig


def model_metrics_chart():
    metrics = ["Accuracy","Precision","F1 Score","ROC-AUC"]
    values  = [97.07, 96.23, 96.08, 96.84]
    colors  = [ACCENT, "#5de0c8", "#f5a623", "#27ae60"]
    fig = go.Figure(go.Bar(
        x=metrics, y=values,
        marker_color=colors, marker_line_width=0,
        text=[f"{v:.2f}%" for v in values],
        textposition="outside", textfont=dict(color="#e8edf3"),
    ))
    fig.update_layout(
        paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
        yaxis=dict(range=[90,100], color=FONT_CLR, gridcolor=GRID_CLR, title="Score (%)"),
        xaxis=dict(color=FONT_CLR),
        height=240, margin=dict(t=30,b=10,l=10,r=10), font={"color":FONT_CLR},
    )
    return fig


def confusion_matrix_chart():
    z    = [[523, 12],[13, 306]]
    fig  = go.Figure(go.Heatmap(
        z=z,
        colorscale=[[0,"rgba(27,45,69,1)"],[1,"rgba(0,201,177,0.7)"]],
        showscale=False, xgap=4, ygap=4,
    ))
    fig.update_layout(
        paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
        xaxis=dict(tickvals=[0,1], ticktext=["Pred: Approved","Pred: Rejected"],
                   color=FONT_CLR, side="bottom"),
        yaxis=dict(tickvals=[0,1], ticktext=["Actual: Approved","Actual: Rejected"],
                   color=FONT_CLR, autorange="reversed"),
        height=220, margin=dict(t=10,b=10,l=10,r=10), font={"color":FONT_CLR},
        annotations=[
            dict(text="TP: 523", x=0, y=0, showarrow=False, font=dict(size=18, color="#0d1b2a")),
            dict(text="FP: 12",  x=1, y=0, showarrow=False, font=dict(size=18, color="#e8edf3")),
            dict(text="FN: 13",  x=0, y=1, showarrow=False, font=dict(size=18, color="#e8edf3")),
            dict(text="TN: 306", x=1, y=1, showarrow=False, font=dict(size=18, color="#0d1b2a")),
        ],
    )
    return fig


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1rem 0 .5rem'>
      <span style='font-family:Playfair Display,serif;font-size:1.65rem;color:#00c9b1;font-weight:900'>💳 CreditIntel</span><br>
      <span style='font-size:.8rem;color:#8fa3b8'>Random Forest Approval Engine</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-title">About the Model</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:.84rem;color:#b0c4d8;line-height:1.7'>
    The <b>CreditIntel</b> is a machine learning system designed to help financial institutions evaluate loan applications efficiently. It analyzes 11 applicant features such as income, employment status, credit history, loan amount, and demographic details to predict loan <code style='background:rgba(190,150,250,.1);padding:.1rem .3rem;border-radius:4px;font-size:.8rem'>approval (0) or rejection (1)</code>. Powered by a Random Forest Classifier, the model learns complex patterns from historical loan data and achieves 97.07% accuracy, supporting faster and more data-driven lending decisions..
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-title">Model Evaluation</div>', unsafe_allow_html=True)
    for metric, val in [("Accuracy","97.07%"),("Precision","96.23%"),("F1 Score","96.08%"),("ROC–AUC","96.84%")]:
        st.markdown(f"""
        <div style='display:flex;justify-content:space-between;padding:.35rem 0;
                    border-bottom:1px solid rgba(0,201,177,.1);font-size:.86rem'>
          <span style='color:#8fa3b8'>{metric}</span>
          <span style='color:#00c9b1;font-weight:700'>{val}</span>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:.76rem;color:#5a7a9a;margin-top:.55rem'>
    Test Set — Confusion Matrix:<br>
    TP=523 &nbsp;·&nbsp; FP=12 &nbsp;·&nbsp; FN=13 &nbsp;·&nbsp; TN=306
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-title">Input Features (11)</div>', unsafe_allow_html=True)
    for icon, name, desc in [
        ("👨‍👩‍👧","dependent",         "Number of financial dependents"),
        ("🎓","education",           "Graduate / Not Graduate"),
        ("💼","self_employed",       "Yes / No"),
        ("💰","annual_income",       "Gross yearly income (₹)"),
        ("🏦","loan",                "Requested loan amount (₹)"),
        ("📅","loan_term",           "Repayment duration in years"),
        ("📊","cibil",               "Credit score 300–900"),
        ("🏠","residential_asset",   "Residential property value (₹)"),
        ("🏢","commercial_asset",    "Commercial property value (₹)"),
        ("💎","luxury_asset",        "Luxury goods / vehicles (₹)"),
        ("🏧","bank_asset",          "Liquid bank balance / FDs (₹)"),
    ]:
        st.markdown(f"""
        <div class="feat-row">
          <div class="feat-name">{icon} <code style='color:#00c9b1'>{name}</code></div>
          <div class="feat-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-title">Decision Rule</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:.83rem;color:#b0c4d8;line-height:1.8'>
    <code style='color:#27ae60'>prediction == 0</code> → 🎉 <b style='color:#27ae60'>APPROVED</b><br>
    <code style='color:#e74c3c'>prediction == 1</code> → ❌ <b style='color:#e74c3c'>REJECTED</b>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
st.markdown("""
<div style='padding:.5rem 0 1.5rem'>
  <h1 style='font-size:2.3rem;margin:0;background:linear-gradient(90deg,#00c9b1,#5de0c8);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent'>
    Loan Approval Prediction
  </h1>
  <p style='color:#8fa3b8;margin-top:.3rem;font-size:.93rem'>
    "When data speaks, smarter financial decisions follow."
  </p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-title">Applicant Details</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**👤 Personal**")
    dependent     = st.number_input("Dependents",               min_value=0,       max_value=10,        value=2,         step=1)
    education     = st.selectbox("Education",                   ["Graduate","Not Graduate"])
    self_employed = st.selectbox("Self Employed",               ["No","Yes"])
    cibil         = st.number_input("CIBIL Score (300–900)",    min_value=300,     max_value=900,       value=720,       step=1)

with col2:
    st.markdown("**💵 Loan & Income**")
    annual_income = st.number_input("Annual Income (₹)",        min_value=0,       max_value=10_000_000,value=600_000,   step=10_000)
    loan          = st.number_input("Loan Amount (₹)",          min_value=10_000,  max_value=50_000_000,value=2_500_000, step=50_000)
    loan_term     = st.number_input("Loan Term (Years)",        min_value=1,       max_value=30,        value=10,        step=1)
    bank_asset    = st.number_input("Bank Asset / Balance (₹)", min_value=0,       max_value=20_000_000,value=500_000,   step=10_000)

with col3:
    st.markdown("**🏠 Assets**")
    residential_asset = st.number_input("Residential Asset (₹)", min_value=0, max_value=100_000_000, value=3_000_000, step=50_000)
    commercial_asset  = st.number_input("Commercial Asset (₹)",  min_value=0, max_value=100_000_000, value=1_000_000, step=50_000)
    luxury_asset      = st.number_input("Luxury Asset (₹)",      min_value=0, max_value=50_000_000,  value=500_000,   step=50_000)

st.markdown("<br>", unsafe_allow_html=True)
_, btn_col, _ = st.columns([1,2,1])
with btn_col:
    predict = st.button("⚡  Predict Loan Approval", use_container_width=True)


# ─────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────
if predict:
    inputs = dict(
        dependent=dependent, education=education, self_employed=self_employed,
        annual_income=annual_income, loan=loan, loan_term=loan_term, cibil=cibil,
        residential_asset=residential_asset, commercial_asset=commercial_asset,
        luxury_asset=luxury_asset, bank_asset=bank_asset,
    )
    r = compute_rf_score(inputs)

    st.markdown("---")
    st.markdown('<div class="section-title">Prediction Result</div>', unsafe_allow_html=True)

    vcol, gcol = st.columns([1,1])

    with vcol:
        if r["approved"]:
            cls         = "verdict-approved"
            icon        = "🎉"
            title       = "APPROVED"
            color       = "#27ae60"
            prob_val    = r["prob_approve"]
            prob_label  = "Approval Probability"
            verdict_msg = "Congratulations! Your loan is likely to be APPROVED."
        else:
            cls         = "verdict-rejected"
            icon        = "❌"
            title       = "REJECTED"
            color       = "#e74c3c"
            prob_val    = r["prob_reject"]
            prob_label  = "Rejection Probability"
            verdict_msg = "Sorry! Your loan is likely to be REJECTED."

        st.markdown(f"""
        <div class="{cls}">
          <div class="verdict-title" style="color:{color}">{icon} {title}</div>
          <div style="font-size:2.4rem;font-weight:700;color:#e8edf3;margin:.4rem 0">{prob_val:.2f}%</div>
          <div style="color:#8fa3b8;font-size:.86rem">{prob_label}</div>
          <div style="color:#b0c4d8;font-size:.9rem;margin-top:.7rem">{verdict_msg}</div>
          <div style="margin-top:.9rem">
            <span class="pill">CIBIL: {cibil}</span>
            <span class="pill">Loan/Income: {r['loan_inc_ratio']:.1f}x</span>
            <span class="pill">Asset Cover: {r['asset_coverage']:.2f}x</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div style="font-size:.88rem;color:#8fa3b8;font-weight:600;margin-bottom:.5rem">Why this decision?</div>', unsafe_allow_html=True)
        for ico, reason in r["reasons"]:
            c = "#27ae60" if ico=="✅" else ("#e74c3c" if ico=="❌" else "#f5a623")
            st.markdown(f'<div style="font-size:.84rem;color:{c};padding:.28rem 0">{ico} {reason}</div>', unsafe_allow_html=True)

    with gcol:
        st.plotly_chart(gauge_chart(prob_val, r["approved"]), use_container_width=True, config={"displayModeBar":False})
        st.markdown("""
        <div style='text-align:center;font-size:.78rem;color:#8fa3b8;margin-top:-.5rem'>
          <span style='color:#e74c3c'>■</span> Rejected (0–54%) &nbsp;
          <span style='color:#f5a623'>■</span> Borderline (55–65%) &nbsp;
          <span style='color:#27ae60'>■</span> Approved (65–100%)
        </div>
        """, unsafe_allow_html=True)

    # Metric cards
    st.markdown("<br>", unsafe_allow_html=True)
    m1,m2,m3,m4,m5 = st.columns(5)
    for col,(lbl,val) in zip([m1,m2,m3,m4,m5],[
        ("CIBIL Score",     str(cibil)),
        ("Loan / Income",   f"{r['loan_inc_ratio']:.1f}x"),
        ("Asset Coverage",  f"{r['asset_coverage']:.2f}x"),
        ("Total Assets",    f"₹{r['total_assets']/1e5:.1f}L"),
        ("RF Score",        f"{r['score']:.0f}/100"),
    ]):
        with col:
            st.markdown(f'<div class="metric-card"><div class="label">{lbl}</div><div class="value">{val}</div></div>', unsafe_allow_html=True)

    # Charts row 1
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Feature Scorecard &amp; Risk Radar</div>', unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        st.markdown('<div style="font-size:.84rem;color:#8fa3b8;margin-bottom:.4rem">Feature Score Breakdown</div>', unsafe_allow_html=True)
        st.plotly_chart(factor_bar(r["factors"]), use_container_width=True, config={"displayModeBar":False})
    with c2:
        st.markdown('<div style="font-size:.84rem;color:#8fa3b8;margin-bottom:.4rem">Risk Profile Radar</div>', unsafe_allow_html=True)
        st.plotly_chart(risk_radar(r["factors"]), use_container_width=True, config={"displayModeBar":False})

    # Charts row 2
    st.markdown('<div class="section-title">Asset Analysis, CIBIL Band &amp; Model Performance</div>', unsafe_allow_html=True)
    c3,c4,c5 = st.columns(3)
    with c3:
        st.markdown('<div style="font-size:.84rem;color:#8fa3b8;margin-bottom:.4rem">Asset Portfolio Breakdown</div>', unsafe_allow_html=True)
        st.plotly_chart(asset_breakdown(residential_asset, commercial_asset, luxury_asset, bank_asset),
                        use_container_width=True, config={"displayModeBar":False})
    with c4:
        st.markdown('<div style="font-size:.84rem;color:#8fa3b8;margin-bottom:.4rem">CIBIL Score Band Position</div>', unsafe_allow_html=True)
        st.plotly_chart(cibil_band_chart(cibil), use_container_width=True, config={"displayModeBar":False})
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div style="font-size:.84rem;color:#8fa3b8;margin-bottom:.4rem">Feature % of Max Score</div>', unsafe_allow_html=True)
        st.plotly_chart(pct_bar(r["factors"]), use_container_width=True, config={"displayModeBar":False})
    with c5:
        st.markdown('<div style="font-size:.84rem;color:#8fa3b8;margin-bottom:.4rem">Model Performance Metrics</div>', unsafe_allow_html=True)
        st.plotly_chart(model_metrics_chart(), use_container_width=True, config={"displayModeBar":False})
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div style="font-size:.84rem;color:#8fa3b8;margin-bottom:.4rem">Confusion Matrix (Test Set)</div>', unsafe_allow_html=True)
        st.plotly_chart(confusion_matrix_chart(), use_container_width=True, config={"displayModeBar":False})

    # Input Summary
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Input Summary</div>', unsafe_allow_html=True)
    s1,s2,s3 = st.columns(3)

    with s1:
        
        st.markdown('<div style="font-size:.78rem;color:#00c9b1;font-weight:700;text-transform:uppercase;letter-spacing:.05em;margin-bottom:.7rem">👤 Personal</div>', unsafe_allow_html=True)
        for k,v in [("dependent", str(dependent)), ("education", education),
                    ("self_employed", self_employed), ("cibil", str(cibil))]:
            st.markdown(f'<div class="summary-row"><span class="key">{k}</span><span class="val">{v}</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with s2:
        
        st.markdown('<div style="font-size:.78rem;color:#00c9b1;font-weight:700;text-transform:uppercase;letter-spacing:.05em;margin-bottom:.7rem">💰 Loan & Income</div>', unsafe_allow_html=True)
        for k,v in [("annual_income", f"₹{annual_income:,.0f}"), ("loan", f"₹{loan:,.0f}"),
                    ("loan_term", f"{loan_term} yrs"), ("Loan/Income Ratio", f"{r['loan_inc_ratio']:.2f}x")]:
            st.markdown(f'<div class="summary-row"><span class="key">{k}</span><span class="val">{v}</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with s3:
        
        st.markdown('<div style="font-size:.78rem;color:#00c9b1;font-weight:700;text-transform:uppercase;letter-spacing:.05em;margin-bottom:.7rem">🏠 Assets</div>', unsafe_allow_html=True)
        for k,v in [("residential_asset", f"₹{residential_asset:,.0f}"),
                    ("commercial_asset",  f"₹{commercial_asset:,.0f}"),
                    ("luxury_asset",      f"₹{luxury_asset:,.0f}"),
                    ("bank_asset",        f"₹{bank_asset:,.0f}"),
                    ("Total Assets",      f"₹{r['total_assets']:,.0f}"),
                    ("Asset Coverage",    f"{r['asset_coverage']:.2f}x")]:
            st.markdown(f'<div class="summary-row"><span class="key">{k}</span><span class="val">{v}</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div class="footer">
  <div style='font-size:.9rem;color:#e8edf3;font-family:Playfair Display,serif;margin-bottom:.4rem'>
    💳 CreditIntel · Random Forest Approval Engine
  </div>
  <div style='margin-top:.4rem'>
    A step toward smarter, data-driven lending systems.
  </div>
</div>
""", unsafe_allow_html=True)