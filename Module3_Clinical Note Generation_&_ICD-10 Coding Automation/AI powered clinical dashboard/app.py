import streamlit as st
import pandas as pd
from PIL import Image
import plotly.express as px
import os

# ==============================
# 🎨 Streamlit Configuration
# ==============================
st.set_page_config(page_title="AI-Powered Clinical Dashboard", page_icon="🩺", layout="wide")

# ==============================
# 💅 Custom CSS Styling
# ==============================
st.markdown("""
<style>
body {
    background-color: #fffaf5; /* soft cream tone */
    color: #1c1e21;
    font-family: 'Segoe UI', sans-serif;
}
h1 {
    color: #004d80 !important;
    text-align: center;
    font-weight: 700;
}
h2, h3 {
    color: #004d80 !important;
}
.subtext {
    text-align: center;
    font-style: italic;
    color: #6c757d;
    margin-top: -10px;
}
.block-container {
    padding-top: 1rem;
}
.footer {
    text-align: center;
    color: #5f6368;
    background-color: #f1f3f4;
    padding: 15px;
    border-radius: 8px;
    margin-top: 30px;
}
.condition-keywords {
    background-color: #fff1b8;  /* warm yellow highlight */
    padding: 10px 14px;
    border-left: 6px solid #f4b400; /* golden edge */
    border-radius: 8px;
    color: #102a43;
    font-size: 1.2em;
    font-weight: 600;
    display: inline-block;
    margin-top: 5px;
}
.note-box {
    background-color: #eaf4ff; /* subtle blue box */
    padding: 12px;
    border-radius: 10px;
    color: #1c1e21;
    font-size: 0.95em;
    line-height: 1.5em;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# 📥 Load Unified Dataset
# ==============================
@st.cache_data
def load_data():
    return pd.read_csv("/content/UnifiedDataset_with_images.csv")

df = load_data()

# ==============================
# 🏥 HEADER
# ==============================
st.title("AI-Powered Clinical Dashboard")
st.markdown("<p class='subtext'>Integrated Clinical Insights & EHR Visualization System </p>", unsafe_allow_html=True)
st.divider()

# ==============================
# 🧭 MODE TOGGLE
# ==============================
view_mode = st.radio("Choose view mode:", ["📋 Overview Table", "🔍 Detailed View"], horizontal=True)

# ==============================
# 📋 OVERVIEW MODE
# ==============================
if view_mode == "📋 Overview Table":
    st.subheader("🧾 Patient Records Overview")

    # Handle merged-column variations automatically
    condition_col = "Medical Condition"
    if "Medical Condition_x" in df.columns:
        condition_col = "Medical Condition_x"
    elif "Medical Condition_y" in df.columns:
        condition_col = "Medical Condition_y"

    display_cols = ["patient_id", "Name", "Age", "Gender", condition_col, "Predicted_ICD"]

    st.dataframe(df[display_cols].rename(columns={condition_col: "Medical Condition"}), use_container_width=True)
    st.markdown("Switch to *'Detailed View'* for in-depth visualization of each patient record.")

# ==============================
# 🔍 DETAILED VIEW MODE
# ==============================
else:
    st.sidebar.header("🔍 Search / Filter Patient")
    query = st.sidebar.text_input("Enter patient name or ID")

    filtered_df = df.copy()
    if query:
        filtered_df = df[df["Name"].str.contains(query, case=False, na=False) |
                         df["patient_id"].astype(str).str.contains(query, case=False, na=False)]

    patient = st.sidebar.selectbox("Select Patient", filtered_df["Name"].unique())
    record = filtered_df[filtered_df["Name"] == patient].iloc[0]

    # ✅ Handle column variation
    condition_col = "Medical Condition"
    if "Medical Condition_x" in df.columns:
        condition_col = "Medical Condition_x"
    elif "Medical Condition_y" in df.columns:
        condition_col = "Medical Condition_y"

    col1, col2 = st.columns([1, 1])

    # --- LEFT COLUMN (Patient Info & X-ray) ---
    with col1:
        st.subheader("🧍 Patient & Admission Details")
        st.write(f"**Patient ID:** {record['patient_id']}")
        st.write(f"**Name:** {record['Name']}")
        st.write(f"**Age / Gender:** {record['Age']} / {record['Gender']}")
        st.write(f"**Blood Type:** {record['Blood Type']}")
        st.write(f"**Condition:** {record[condition_col]}")
        st.write(f"**Admission Type:** {record['Admission Type']}")
        st.write(f"**Doctor:** {record['Doctor']}")
        st.write(f"**Hospital:** {record['Hospital']}")
        st.write(f"**Medication:** {record['Medication']}")
        st.write(f"**Date:** {record['Date of Admission']} → {record['Discharge Date']}")
        st.write(f"**Test Results:** {record['Test Results']}")

               

    # --- RIGHT COLUMN (Notes & ICD Info) ---
    with col2:
        st.subheader("📝 AI-Generated Clinical Note")
        st.markdown(f"<div class='note-box'>{record['clinical_note']}</div>", unsafe_allow_html=True)

        st.subheader("💊 Predicted ICD-10 Code")
        st.success(record["Predicted_ICD"])

        if "condition_keywords" in record and pd.notna(record["condition_keywords"]):
            st.subheader("🔤 Condition Keywords")
            st.markdown(f"<div class='condition-keywords'>{record['condition_keywords'].capitalize()}</div>", unsafe_allow_html=True)

    # --- CHARTS AREA ---
    st.divider()
    st.subheader("📊 Data Insights Overview")

    colA, colB, colC = st.columns(3)

    with colA:
        condition_counts = df[condition_col].value_counts().reset_index()
        condition_counts.columns = ["Condition", "Count"]
        fig1 = px.pie(condition_counts, names="Condition", values="Count",
                      title="Medical Condition Distribution",
                      color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig1, use_container_width=True)

    with colB:
        icd_counts = df["Predicted_ICD"].value_counts().reset_index()
        icd_counts.columns = ["ICD Code", "Count"]
        fig2 = px.bar(icd_counts.head(10), x="ICD Code", y="Count",
                      title="Top 10 ICD-10 Codes",
                      color="Count", color_continuous_scale="Blues")
        st.plotly_chart(fig2, use_container_width=True)

    with colC:
        if "Date of Admission" in df.columns:
            timeline = df["Date of Admission"].value_counts().sort_index().reset_index()
            timeline.columns = ["Date", "Admissions"]
            fig3 = px.line(timeline, x="Date", y="Admissions", title="Patient Admissions Over Time",
                           markers=True, line_shape="spline", color_discrete_sequence=["#72aee6"])
            st.plotly_chart(fig3, use_container_width=True)

# ==============================
# 👣 FOOTER
# ==============================
st.divider()
st.markdown("""
<div class="footer">
    <h4>👩‍💻 Developed by <b>Lalithanjali Kodavali</b></h4>
    <a href="https://github.com/lalithanjali04" target="_blank">🌐 GitHub</a> |
    <a href="https://www.linkedin.com/in/lalithanjali-kodavali/" target="_blank">💼 LinkedIn</a>
</div>
""", unsafe_allow_html=True)
