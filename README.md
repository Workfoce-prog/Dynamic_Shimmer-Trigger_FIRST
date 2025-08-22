
# FIRST — Full Streamlit App

Run an end-to-end Food Insecurity Score Tracker: upload Excel, compute dynamic latent risk (**F_t**), RAG with hysteresis,
visualize, export PNG/PDF/CSV, and optionally email or upload to S3.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Data format
- Sheet: **Food Insecurity Inputs**
- Columns (min): `geo`, `date` (or `Date`), normalized drivers: `Unemp_Norm`, `Evict_Norm`, `Food_Norm`, `Shutoff_Norm`, `Attendance_Norm`, `FRL_Norm`,
  and levers: `BenefitUptake`, `OutreachIntensity`, `CommunityPartnerCoverage` (case/spacing variants auto-normalized).

If you do not upload a file, the app loads `data/Food_Insecurity_Input_With_Actions.xlsx` (sample).

## Email
Set these in **Streamlit secrets**:
```
SMTP_SERVER=...
SMTP_PORT=587
SMTP_USER=...
SMTP_PASSWORD=...
```
Then use **Send Email with Attachments** in the app.

## S3 Upload
Add to **Streamlit secrets**:
```
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
```
Then use **Upload County Profiles PDF to S3** in the app.

## Exports
- PNGs: `assets/exports/trend_average_risk_over_time.png`, `rag_summary.png`, `avg_risk_by_county.png`
- County Profiles PDF: `assets/exports/FIRST_report_with_profiles.pdf`
- Enriched dataset (triggers): CSV download + Excel saved in `assets/exports/`

