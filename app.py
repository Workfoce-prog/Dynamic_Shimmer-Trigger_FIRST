# --- FIRST Dynamic System: single-file Streamlit app (safe, with summary & actions) ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ---------------- Model ----------------
def _sigmoid(x):
    return 1/(1+np.exp(-x))

class FirstDynamicModel:
    def __init__(self, alpha=0.75, lambda_ewma=0.35,
                 thresholds=None, betas=None, gammas=None):
        self.alpha = float(alpha)
        self.lambda_ewma = float(lambda_ewma)
        self.thresholds = thresholds or {
            "AG_down": 0.30, "GA_up": 0.38, "AR_up": 0.62, "RA_down": 0.54
        }
        self.betas = betas or {
            "Unemp_Norm": 1.0, "Evict_Norm": 0.8, "Food_Norm": 0.7,
            "Shutoff_Norm": 0.5, "Attendance_Norm": 0.4, "FRL_Norm": 0.6
        }
        self.gammas = gammas or {
            "BenefitUptake": 0.9, "OutreachIntensity": 0.6, "CommunityPartnerCoverage": 0.5
        }

    def composite_index(self, row: pd.Series) -> float:
        z = 0.0
        for k, b in self.betas.items():  z += b * float(row.get(k, 0.0))
        for k, g in self.gammas.items(): z -= g * float(row.get(k, 0.0))
        return float(_sigmoid(z))

    def next_F(self, f_prev: float, row: pd.Series) -> float:
        comp   = self.composite_index(row)
        f_tilde = (1 - self.lambda_ewma) * f_prev + self.lambda_ewma * comp
        f_next  = self.alpha * f_prev + (1 - self.alpha) * f_tilde
        return float(np.clip(f_next, 0.0, 1.0))

    def rag_transition(self, f: float, last: str) -> str:
        th = self.thresholds
        if last == "Red":   return "Amber" if f < th["RA_down"] else "Red"
        if last == "Green": return "Amber" if f > th["GA_up"] else "Green"
        if f > th["AR_up"]: return "Red"
        if f < th["AG_down"]: return "Green"
        return "Amber"

    def simulate(self, df: pd.DataFrame, f0: float|None=None, rag0: str="Amber") -> pd.DataFrame:
        out = df.copy().reset_index(drop=True)
        if out.empty:
            return out.assign(F_t=[], RAG=[])
        first_row = out.iloc[0]
        f_prev = self.composite_index(first_row) if f0 is None else float(f0)
        rag_prev = rag0
        F, RAG = [], []
        for _, row in out.iterrows():
            f_next = self.next_F(f_prev, row)
            rag = self.rag_transition(f_next, rag_prev)
            F.append(f_next); RAG.append(rag)
            f_prev, rag_prev = f_next, rag
        out["F_t"] = F; out["RAG"] = RAG
        return out

# ---------------- App ----------------
st.set_page_config(page_title="FIRST Dynamic System", layout="wide")
st.title("FIRST Dynamic System — Risk & RAG Simulator")

# Sidebar controls
st.sidebar.header("Model Settings")
alpha = st.sidebar.slider("Alpha (persistence)", 0.0, 1.0, 0.75, 0.01)
lam   = st.sidebar.slider("Lambda (EWMA toward composite)", 0.0, 1.0, 0.35, 0.01)
AGd   = st.sidebar.slider("Green if below (AG_down)", 0.0, 1.0, 0.30, 0.01)
GAu   = st.sidebar.slider("Leave Green if above (GA_up)", 0.0, 1.0, 0.38, 0.01)
ARu   = st.sidebar.slider("Enter Red if above (AR_up)", 0.0, 1.0, 0.62, 0.01)
RAd   = st.sidebar.slider("Leave Red if below (RA_down)", 0.0, 1.0, 0.54, 0.01)

# Data load: CSV upload or bundled fallback
uploaded = st.file_uploader("Upload FIRST panel CSV", type=["csv"])
df = None
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read uploaded CSV: {e}")

if df is None:
    try:
        df = pd.read_csv("sample_first_panel.csv")
        st.info("Using bundled sample_first_panel.csv")
    except Exception:
        st.warning("No CSV found; showing a tiny synthetic sample.")
        dates = pd.date_range("2024-01-01", periods=8, freq="MS")
        rows = []
        for geo in ["Hennepin","Ramsey"]:
            bias = 0.02 if geo=="Hennepin" else -0.01
            for d in dates:
                base = float(0.5 + bias + 0.1*np.sin(2*np.pi*(d.month/12.0)))
                rows.append({
                    "geo": geo, "date": d.date().isoformat(),
                    "Unemp_Norm": float(np.clip(base + np.random.normal(0,0.07),0,1)),
                    "Evict_Norm": float(np.clip(0.45 + np.random.normal(0,0.08),0,1)),
                    "Food_Norm": float(np.clip(0.40 + np.random.normal(0,0.08),0,1)),
                    "Shutoff_Norm": float(np.clip(0.30 + np.random.normal(0,0.06),0,1)),
                    "Attendance_Norm": float(np.clip(0.35 + np.random.normal(0,0.07),0,1)),
                    "FRL_Norm": float(np.clip(0.55 + np.random.normal(0,0.06),0,1)),
                    "BenefitUptake": float(np.clip(0.40 + np.random.normal(0,0.03),0, 1)),
                    "OutreachIntensity": float(np.clip(0.50 + np.random.normal(0,0.04),0, 1)),
                    "CommunityPartnerCoverage": float(np.clip(0.60 + np.random.normal(0,0.03),0,1)),
                })
        df = pd.DataFrame(rows)

# Basic checks
required_min = {"geo","date"}
if not required_min.issubset(df.columns):
    st.error("CSV must include at least: geo, date, plus *_Norm and intervention columns.")
    st.stop()

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])
geos = sorted(df["geo"].astype(str).unique().tolist())
if not geos:
    st.error("No geographies found in 'geo' column.")
    st.stop()

geo = st.selectbox("Geography", geos)

# Run model
model = FirstDynamicModel(
    alpha=alpha, lambda_ewma=lam,
    thresholds={"AG_down":AGd,"GA_up":GAu,"AR_up":ARu,"RA_down":RAd}
)
df_geo = df[df["geo"] == geo].sort_values("date").reset_index(drop=True)
if df_geo.empty:
    st.warning("No rows for the selected geography.")
    st.stop()

sim = model.simulate(df_geo)
if sim.empty:
    st.warning("Simulation returned no rows.")
    st.stop()

# ---------- Executive summary + margin-to-threshold gauge ----------
def summarize(ft, rag, series, AGd, GAu, ARu, RAd):
    # Trend over last ~3 months
    if len(series) >= 4:
        delta = float(series.iloc[-1] - series.iloc[-4]); per_month = delta / 3.0
    elif len(series) >= 2:
        delta = float(series.iloc[-1] - series.iloc[-2]); per_month = delta
    else:
        per_month = 0.0

    if per_month < -0.01: trend_word, trend_icon = "improving", "📉"
    elif per_month > 0.01: trend_word, trend_icon = "worsening", "📈"
    else: trend_word, trend_icon = "stable", "➖"

    if rag == "Green":
        next_line = "Leave Green (GA↑)"; margin = float(GAu - ft); risk_level = "low"; chip = "🟢 Green"
    elif rag == "Amber":
        to_red   = float(ARu - ft); to_green = float(ft - AGd)
        if abs(to_red) < abs(to_green):
            next_line, margin = "Enter Red (AR↑)", to_red
        else:
            next_line, margin = "Back to Green (AG↓)", to_green
        risk_level = "elevated"; chip = "🟡 Amber"
    else:
        next_line = "Leave Red (RA↓)"; margin = float(ft - RAd); risk_level = "high"; chip = "🔴 Red"

    # Progress gauge toward next decision line
    if rag == "Green":
        denom = max(GAu - 0.0, 1e-6); pct = max(0.0, min(1.0, (GAu - ft) / denom))
    elif rag == "Amber" and next_line.startswith("Enter Red"):
        denom = max(ARu - AGd, 1e-6); pct = max(0.0, min(1.0, (ARu - ft) / denom))
    elif rag == "Amber":
        denom = max(ARu - AGd, 1e-6); pct = max(0.0, min(1.0, (ft - AGd) / denom))
    else:
        denom = max(1.0 - RAd, 1e-6); pct = max(0.0, min(1.0, (ft - RAd) / denom))

    return {
        "chip": chip, "trend_icon": trend_icon, "trend_word": trend_word, "per_month": per_month,
        "next_line": next_line, "margin": margin, "risk_level": risk_level, "pct_to_line": pct
    }

last = sim.iloc[-1]
ft  = float(last["F_t"])
rag = str(last["RAG"])
summary = summarize(ft, rag, sim["F_t"], AGd, GAu, ARu, RAd)

# One-line executive summary
if summary["risk_level"] == "low":
    st.success(f"{summary['chip']}: {summary['trend_icon']} {summary['trend_word']}. "
               f"Margin to {summary['next_line']}: {summary['margin']:+.03f}")
elif summary["risk_level"] == "elevated":
    st.warning(f"{summary['chip']}: {summary['trend_icon']} {summary['trend_word']}. "
               f"Closest decision: {summary['next_line']} (margin {summary['margin']:+.03f})")
else:
    st.error(f"{summary['chip']}: {summary['trend_icon']} {summary['trend_word']}. "
             f"Next objective: {summary['next_line']} (need {abs(summary['margin']):.03f})")

# Quick metrics
c1, c2, c3 = st.columns(3)
c1.metric("Current risk (Fₜ)", f"{ft:.02f}", f"{summary['per_month']:+.03f}/mo")
c2.metric("Next decision line", summary["next_line"])
c3.metric("Margin to line", f"{summary['margin']:+.03f}")

# Simple gauge
st.progress(int(round(summary["pct_to_line"] * 100)))

# ---------- Recommended actions (leadership-ready) ----------
NEAR = 0.03  # tune as desired

def make_actions(ft, rag, per_month, AGd, GAu, ARu, RAd):
    actions = []
    worsening = per_month >  0.01
    improving = per_month < -0.01
    near_leave_green = (rag == "Green") and ((GAu - ft) <= NEAR)
    near_enter_red   = (ARu - ft) <= NEAR  # works for Green/Amber
    near_back_green  = (rag == "Amber") and ((ft - AGd) <= NEAR)
    near_leave_red   = (rag == "Red")   and ((ft - RAd) <= NEAR)

    if rag == "Green":
        actions += [
            "Maintain baseline monitoring (monthly) and partner check-ins.",
            "Continue standard outreach cadence; keep navigator capacity warm.",
        ]
        if worsening or near_leave_green:
            actions += [
                "Pre-caution: +10% outreach (×1.10) for 2 months in top-need tracts.",
                "Light-touch eligibility texting to lift SNAP/WIC uptake +5% (×1.05).",
                "Set weekly review until risk is ≥0.05 below the Leave-Green line (GA↑).",
            ]

    elif rag == "Amber":
        actions += [
            "Activate heightened monitoring (weekly) and hotspot map for targeting.",
            "Increase outreach intensity +20% (×1.20) in top zip codes; shift events to schools/clinics.",
            "Boost benefits navigation +10% (×1.10) completions using auto-booking/text nudges.",
        ]
        if worsening or near_enter_red:
            actions.append("Pre-stage surge: extend pantry hours, line up mobile distributions, pre-notify partners.")
        if improving or near_back_green:
            actions.append("Hold for 2 cycles; plan step-down if Fₜ < AG↓ for 2 consecutive months.")

    else:  # Red
        actions += [
            "Launch surge within 72 hours: extended pantry hours, mobile pop-ups, temp staffing.",
            "Targeted outreach +40% (×1.40) in top tracts; door-to-door or school-based events.",
            "Fast-track SNAP/WIC: goal +15% application completions in 30 days.",
            "Coordinate utilities/housing for shutoff & eviction prevention clinics.",
            "Daily monitoring; stand-down only after Fₜ < RA↓ for 2 consecutive months.",
        ]
        if improving and near_leave_red:
            actions.append("Plan Red→Amber transition: taper surge 25% while retaining navigator coverage.")

    return actions

actions = make_actions(ft=ft, rag=rag, per_month=summary["per_month"],
                       AGd=AGd, GAu=GAu, ARu=ARu, RAd=RAd)

st.subheader("Recommended actions")
for a in actions:
    st.markdown(f"- {a}")

# Small action plan table
plan_rows = []
prio = 1
for a in actions:
    if "outreach" in a.lower():
        lever, target = "OutreachIntensity", "×1.10–×1.40 (by need)"
    elif "snap" in a.lower() or "wic" in a.lower() or "benefit" in a.lower():
        lever, target = "BenefitUptake", "+5–15% completions"
    elif "partner" in a.lower() or "mobile" in a.lower() or "pantry" in a.lower():
        lever, target = "CommunityPartnerCoverage", "+0.05–0.10 coverage"
    else:
        lever, target = "Governance/Monitoring", "Weekly / Daily"
    plan_rows.append({"Priority": prio, "Action": a, "Lever": lever, "Target": target})
    prio += 1

st.caption("Action plan (prioritized)")
st.dataframe(pd.DataFrame(plan_rows), use_container_width=True)

# ---------- Chart + Table ----------
st.subheader(f"Latent Risk over Time — {geo}")
fig, ax = plt.subplots()
ax.plot(sim["date"], sim["F_t"], label="F_t")
ax.axhspan(0, AGd, alpha=0.1, label="Green band")
ax.axhspan(AGd, ARu, alpha=0.1, label="Amber band")
ax.axhspan(ARu, 1.0, alpha=0.1, label="Red band")
ax.set_ylim(0, 1)
ax.legend()
st.pyplot(fig)

st.subheader("Simulated (last 12 rows)")
st.dataframe(sim.tail(12), use_container_width=True)

# Optional: export
st.download_button(
    "Download simulated CSV",
    sim.to_csv(index=False),
    file_name=f"FIRST_simulated_{geo}.csv",
    mime="text/csv"
)
