
import os, io, ssl, smtplib, zipfile
from email.message import EmailMessage
from pathlib import Path
from datetime import datetime, timedelta
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import boto3

st.set_page_config(page_title="FIRST — Full Streamlit App", layout="wide")
st.title("FIRST — Food Insecurity Score Tracker (Full App)")
st.caption("Upload Excel, analyze RAG, simulate dynamics, generate reports, email/S3 exports.")

BASE_DIR = Path(__file__).parent
EXPORT_DIR = BASE_DIR / "assets" / "exports"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------- Data loading --------------------
@st.cache_data
def load_excel(path_or_buf):
    try:
        df = pd.read_excel(path_or_buf, sheet_name="Food Insecurity Inputs")
    except Exception:
        xl = pd.ExcelFile(path_or_buf)
        df = xl.parse(xl.sheet_names[0])
    # normalize key columns
    rename_map = {
        "Communitypartner coverage": "CommunityPartnerCoverage",
        "CommunityPartner coverage": "CommunityPartnerCoverage",
        "communitypartner coverage": "CommunityPartnerCoverage",
        "Date": "date", "DATE":"date",
        "Geo":"geo","County":"geo","county":"geo"
    }
    for k,v in list(rename_map.items()):
        if k in df.columns:
            df.rename(columns={k:v}, inplace=True)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    elif "Date" in df.columns:
        df["date"] = pd.to_datetime(df["Date"], errors="coerce")
    else:
        df["date"] = pd.date_range("2024-01-01", periods=len(df), freq="D")
    if "geo" not in df.columns:
        df["geo"] = "Unknown"
    df = df.dropna(subset=["date"]).copy()
    return df

def resolve_sample_path():
    for cand in [
        BASE_DIR/"data"/"Food_Insecurity_Input_With_Actions.xlsx",
        BASE_DIR/"Food_Insecurity_Input_With_Actions.xlsx",
    ]:
        if cand.exists(): return cand
    return None

st.sidebar.header("Data Source")
uploaded = st.sidebar.file_uploader("Upload Excel (sheet: 'Food Insecurity Inputs')", type=["xlsx"])
use_sample = st.sidebar.checkbox("Use bundled sample dataset", value=True)

sample_path = resolve_sample_path()
if uploaded is not None:
    df = load_excel(uploaded)
elif use_sample and sample_path:
    df = load_excel(sample_path)
else:
    st.warning("Upload a dataset or enable 'Use bundled sample dataset'.")
    st.stop()

# -------------------- Dynamic model & triggers --------------------
def _sigmoid(x): return 1/(1+np.exp(-x))

ALPHA = st.sidebar.slider("α (persistence)", 0.0, 1.0, 0.75, 0.01)
LAMBDA = st.sidebar.slider("λ (EWMA toward composite)", 0.0, 1.0, 0.35, 0.01)
THRESHOLDS = {
    "AG_down": st.sidebar.slider("Green if below (AG↓)", 0.0, 1.0, 0.30, 0.01),
    "GA_up":   st.sidebar.slider("Leave Green if above (GA↑)", 0.0, 1.0, 0.38, 0.01),
    "AR_up":   st.sidebar.slider("Enter Red if above (AR↑)", 0.0, 1.0, 0.62, 0.01),
    "RA_down": st.sidebar.slider("Leave Red if below (RA↓)", 0.0, 1.0, 0.54, 0.01),
}

BETAS = {"Unemp_Norm":1.0,"Evict_Norm":0.8,"Food_Norm":0.7,"Shutoff_Norm":0.5,"Attendance_Norm":0.4,"FRL_Norm":0.6}
GAMMAS = {"BenefitUptake":0.9,"OutreachIntensity":0.6,"CommunityPartnerCoverage":0.5}

# ensure columns exist
for col in list(BETAS.keys()) + list(GAMMAS.keys()):
    if col not in df.columns: df[col] = 0.5

def composite_index(row):
    z = 0.0
    for k,b in BETAS.items(): z += b * float(row.get(k,0.0))
    for k,g in GAMMAS.items(): z -= g * float(row.get(k,0.0))
    return float(_sigmoid(z))

def next_F(f_prev, row):
    comp = composite_index(row)
    f_tilde = (1 - LAMBDA) * f_prev + LAMBDA * comp
    f_next  = ALPHA * f_prev + (1 - ALPHA) * f_tilde
    return float(np.clip(f_next, 0.0, 1.0))

def rag_transition(f, last):
    th = THRESHOLDS
    if last == "Red":   return "Amber" if f < th["RA_down"] else "Red"
    if last == "Green": return "Amber" if f > th["GA_up"] else "Green"
    if f > th["AR_up"]: return "Red"
    if f < th["AG_down"]: return "Green"
    return "Amber"

@st.cache_data
def apply_dynamic_triggers(df_in: pd.DataFrame, alpha, lam, thresholds):
    # use provided sliders (captured by closure, but also pass for cache key)
    df_sorted = df_in.sort_values(["geo","date"]).copy()
    F_list, RAG_list, Trig_list, Margin_list, NextLine_list = [], [], [], [], []
    AGd, GAu, ARu, RAd = thresholds["AG_down"], thresholds["GA_up"], thresholds["AR_up"], thresholds["RA_down"]

    def summarize_point(ft, rag, prev_f):
        if rag == "Green":
            return "Leave Green (GA↑)", float(GAu - ft)
        elif rag == "Amber":
            to_red, to_green = float(ARu - ft), float(ft - AGd)
            if abs(to_red) < abs(to_green): return "Enter Red (AR↑)", to_red
            else: return "Back to Green (AG↓)", to_green
        else:
            return "Leave Red (RA↓)", float(ft - RAd)

    for geo, sub in df_sorted.groupby("geo", sort=False):
        prev_f = None; prev_rag = None
        for _, row in sub.iterrows():
            if prev_f is None:
                prev_f = composite_index(row); prev_rag = "Amber"
            f_next = next_F(prev_f, row)
            rag = rag_transition(f_next, prev_rag)
            trig = ""
            if prev_rag is not None and rag != prev_rag:
                if rag == "Red": trig = "Enter Red (AR↑)"
                elif prev_rag == "Red": trig = "Leave Red (RA↓)"
                elif rag == "Green": trig = "Back to Green (AG↓)"
                elif prev_rag == "Green": trig = "Leave Green (GA↑)"
                else: trig = f"Amber shift ({prev_rag}→{rag})"
            nxt, margin = summarize_point(f_next, rag, prev_f)
            F_list.append(f_next); RAG_list.append(rag); Trig_list.append(trig); Margin_list.append(round(margin,3)); NextLine_list.append(nxt)
            prev_f, prev_rag = f_next, rag
    out = df_sorted.copy()
    out["F_t"] = F_list; out["RAG"] = RAG_list
    out["Latent_Trigger"] = Trig_list
    out["Margin_to_Next_Line"] = Margin_list
    out["Next_Decision_Line"] = NextLine_list
    return out

res = apply_dynamic_triggers(df, ALPHA, LAMBDA, THRESHOLDS)

# -------------------- Filters --------------------
st.sidebar.header("Filters")
geos = sorted(res["geo"].dropna().astype(str).unique().tolist())
geo_sel = st.sidebar.multiselect("County/Geo", geos, default=geos)
min_d, max_d = pd.to_datetime(res["date"]).min(), pd.to_datetime(res["date"]).max()
date_range = st.sidebar.date_input("Date range", [min_d, max_d])

mask = res["geo"].isin(geo_sel)
if isinstance(date_range, (list, tuple)) and len(date_range)==2:
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    mask &= (res["date"]>=start) & (res["date"]<=end)

res_f = res.loc[mask].copy()

# -------------------- KPIs --------------------
c1,c2,c3,c4 = st.columns(4)
c1.metric("Rows", len(res_f))
c2.metric("Mean F_t", f"{res_f['F_t'].mean():.3f}")
c3.metric("Severe/High (Red/Amber)", int((res_f['RAG'].isin(['Red','Amber'])).sum()))
c4.metric("Green", int((res_f['RAG']=='Green').sum()))

# -------------------- Visuals --------------------
st.subheader("Food Insecurity Risk Levels (RAG Summary)")
rag_counts = res_f["RAG"].value_counts().reindex(["Red","Amber","Green"]).fillna(0)
fig1, ax1 = plt.subplots(figsize=(5,4))
rag_counts.plot(kind="bar", ax=ax1); ax1.set_xlabel("RAG"); ax1.set_ylabel("Records")
st.pyplot(fig1)

st.subheader("Average Risk Score by County")
county_risk = res_f.groupby("geo")["F_t"].mean().sort_values(ascending=False)
fig2, ax2 = plt.subplots(figsize=(7,4))
county_risk.plot(kind="barh", ax=ax2); ax2.invert_yaxis(); ax2.set_xlabel("Average F_t"); ax2.set_ylabel("County")
st.pyplot(fig2)

st.subheader("Trend of Average Risk Score Over Time")
date_risk = res_f.groupby("date")["F_t"].mean().reset_index()
fig3, ax3 = plt.subplots(figsize=(8,4))
ax3.plot(date_risk["date"], date_risk["F_t"], marker="o"); ax3.set_xlabel("Date"); ax3.set_ylabel("Average F_t")
st.pyplot(fig3)

# -------------------- Recommended actions table --------------------
st.subheader("Recommended Actions (by county)")
def action_text(rag):
    if rag=="Red": return "Surge operations (pantry hours, mobile), +40% outreach, fast-track SNAP/WIC, daily monitoring"
    if rag=="Amber": return "Heightened monitoring, +20% outreach in hotspots, +10% benefit navigation, prep surge"
    return "Maintain baseline monitoring, standard outreach; precaution if approaching GA↑"
summary = (
    res_f.groupby("geo").agg(
        Avg_F_t=("F_t","mean"),
        RAG_Mode=("RAG", lambda x: x.mode().iat[0] if not x.mode().empty else ""),
        Next_Line=("Next_Decision_Line", lambda x: x.mode().iat[0] if not x.mode().empty else ""),
        Mean_Margin=("Margin_to_Next_Line","mean"),
    ).reset_index()
)
summary["Recommended_Actions"] = summary["RAG_Mode"].map(action_text)
st.dataframe(summary, use_container_width=True)

# -------------------- Exports (PNGs/PDF/CSV/ZIP) --------------------
st.subheader("Reports & Exports")

def export_static_pngs(df_source, out_dir=EXPORT_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    # trend
    tr = df_source.groupby("date")["F_t"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10,5)); ax.plot(tr["date"], tr["F_t"], marker="o")
    ax.set_title("Trend of Average Risk Score Over Time"); ax.set_xlabel("Date"); ax.set_ylabel("Average F_t")
    fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(out_dir/"trend_average_risk_over_time.png", dpi=160); plt.close(fig)
    # rag
    rag = df_source["RAG"].value_counts().reindex(["Red","Amber","Green"]).fillna(0)
    fig, ax = plt.subplots(figsize=(7,5)); rag.plot(kind="bar", ax=ax)
    ax.set_title("Food Insecurity Risk Levels (RAG Summary)"); ax.set_xlabel("RAG"); ax.set_ylabel("Number of Records")
    fig.tight_layout(); fig.savefig(out_dir/"rag_summary.png", dpi=160); plt.close(fig)
    # county
    cr = df_source.groupby("geo")["F_t"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8,5)); cr.plot(kind="barh", ax=ax)
    ax.set_title("Average Risk Score by County"); ax.set_xlabel("Average F_t"); ax.set_ylabel("County")
    fig.tight_layout(); fig.savefig(out_dir/"avg_risk_by_county.png", dpi=160); plt.close(fig)

def build_pdf_with_county_profiles(df_source, out_path=EXPORT_DIR/"FIRST_report_with_profiles.pdf"):
    with PdfPages(out_path) as pdf:
        # cover
        fig, ax = plt.subplots(figsize=(8.5, 11)); ax.axis("off")
        ax.text(0.1, 0.9, "FIRST — Food Insecurity Score Tracker", fontsize=22, weight="bold")
        ax.text(0.1, 0.86, f"Rows: {len(df_source)}", fontsize=11)
        ax.text(0.1, 0.84, f"Date Range: {pd.to_datetime(df_source['date']).min().date()} → {pd.to_datetime(df_source['date']).max().date()}", fontsize=11)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        for county in sorted(df_source["geo"].dropna().unique().tolist()):
            sub = df_source[df_source["geo"]==county]
            rows = len(sub); avg_risk = sub["F_t"].mean()
            rag_mode = sub["RAG"].mode().iat[0] if not sub["RAG"].mode().empty else ""
            next_line = sub["Next_Decision_Line"].mode().iat[0] if not sub["Next_Decision_Line"].mode().empty else ""

            fig, ax = plt.subplots(figsize=(8.5, 11)); ax.axis("off")
            ax.text(0.1, 0.93, f"County Profile — {county}", fontsize=18, weight="bold")
            ax.text(0.1, 0.90, f"Rows: {rows}    Avg F_t: {avg_risk:.3f}    RAG: {rag_mode}", fontsize=11)
            ax.text(0.1, 0.87, f"Next decision line: {next_line}", fontsize=10)

            # Trend
            trend = sub.groupby("date")["F_t"].mean()
            ax2 = fig.add_axes([0.1, 0.55, 0.8, 0.25])
            ax2.plot(trend.index, trend.values, marker="o"); ax2.set_title("Risk Trend"); ax2.set_xlabel(""); ax2.set_ylabel("Avg F_t")

            # RAG
            rag = sub["RAG"].value_counts().reindex(["Red","Amber","Green"]).fillna(0)
            ax3 = fig.add_axes([0.1, 0.35, 0.8, 0.15])
            rag.plot(kind="bar", ax=ax3); ax3.set_title("RAG Distribution"); ax3.set_xlabel(""); ax3.set_ylabel("Count")

            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
    return out_path

def _county_csv_bytes(df_src, county_name): return df_src[df_src["geo"]==county_name].to_csv(index=False).encode("utf-8")

colA, colB, colC = st.columns(3)
with colA:
    if st.button("Export Static PNGs (current filters)"):
        export_static_pngs(res_f)
        st.success(f"PNG charts saved to {EXPORT_DIR}")
with colB:
    if st.button("Build County Profiles PDF (current filters)"):
        pdf_path = build_pdf_with_county_profiles(res_f)
        with open(pdf_path, "rb") as f:
            st.download_button("Download County Profiles PDF", f, file_name=Path(pdf_path).name, mime="application/pdf")
with colC:
    export_counties = st.multiselect("Per‑county CSV export", geos, default=geo_sel)
    if st.button("Export Selected Counties as ZIP"):
        if not export_counties: st.warning("Select counties first.")
        else:
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                for c in export_counties:
                    z.writestr(f"{c.replace(' ','_')}.csv", res_f[res_f["geo"]==c].to_csv(index=False))
            st.download_button("Download counties_export.zip", data=buf.getvalue(), file_name="counties_export.zip", mime="application/zip")

# -------------------- Email & S3 --------------------
st.subheader("Email & S3 (optional)")

def send_email_with_attachments(subject, body, to_addr, attachments):
    # Read SMTP config from st.secrets if present
    host = st.secrets.get("SMTP_SERVER", "")
    port = int(st.secrets.get("SMTP_PORT", 587))
    user = st.secrets.get("SMTP_USER", "")
    pwd  = st.secrets.get("SMTP_PASSWORD", "")
    if not all([host, port, user, pwd, to_addr]):
        st.error("Missing SMTP settings or recipient. Set SMTP_* in Streamlit secrets.")
        return False
    msg = EmailMessage()
    msg["From"] = user; msg["To"] = to_addr; msg["Subject"] = subject
    msg.set_content(body)
    for path in attachments:
        p = Path(path)
        if not p.exists(): continue
        data = p.read_bytes()
        maintype, subtype = ("application","octet-stream")
        msg.add_attachment(data, maintype=maintype, subtype=subtype, filename=p.name)
    context = ssl.create_default_context()
    with smtplib.SMTP(host, port) as server:
        server.starttls(context=context)
        server.login(user, pwd)
        server.send_message(msg)
    return True

def s3_upload(path, bucket, key, region=None):
    region = region or st.secrets.get("AWS_REGION", "us-east-1")
    ak = st.secrets.get("AWS_ACCESS_KEY_ID", None)
    sk = st.secrets.get("AWS_SECRET_ACCESS_KEY", None)
    if not (ak and sk and bucket):
        st.error("Missing AWS secrets or bucket name. Set AWS_* in Streamlit secrets.")
        return None
    s3 = boto3.client("s3", region_name=region, aws_access_key_id=ak, aws_secret_access_key=sk)
    s3.upload_file(str(path), bucket, key)
    return f"s3://{bucket}/{key}"

col1, col2 = st.columns(2)
with col1:
    st.caption("Attach latest charts & PDF, and send via SMTP")
    to_addr = st.text_input("To (email)")
    subj = st.text_input("Subject", "FIRST weekly summary")
    if st.button("Send Email with Attachments"):
        # make sure artifacts exist
        export_static_pngs(res_f)
        pdf_path = build_pdf_with_county_profiles(res_f)
        files = [EXPORT_DIR/"trend_average_risk_over_time.png",
                 EXPORT_DIR/"rag_summary.png",
                 EXPORT_DIR/"avg_risk_by_county.png",
                 pdf_path]
        ok = send_email_with_attachments(subj, "Automated summary attached.", to_addr, files)
        if ok: st.success("Email sent (if SMTP settings were valid).")

with col2:
    st.caption("Upload latest PDF to S3")
    bucket = st.text_input("S3 Bucket")
    prefix = st.text_input("S3 Prefix", "first-exports/")
    if st.button("Upload County Profiles PDF to S3"):
        pdf_path = build_pdf_with_county_profiles(res_f)
        key = f"{prefix.rstrip('/')}/{Path(pdf_path).name}"
        uri = s3_upload(pdf_path, bucket, key)
        if uri: st.success(f"Uploaded: {uri}")

# -------------------- Data download --------------------
st.subheader("Download Enriched Dataset")
csv_bytes = res.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV (all rows)", data=csv_bytes, file_name="FIRST_enriched_with_triggers.csv", mime="text/csv")

# save an Excel also to exports
xlsx_path = EXPORT_DIR / "FIRST_enriched_with_triggers.xlsx"
with pd.ExcelWriter(xlsx_path, engine="openpyxl") as w:
    res.to_excel(w, index=False, sheet_name="With Triggers")
st.caption(f"Saved Excel to {xlsx_path}")
