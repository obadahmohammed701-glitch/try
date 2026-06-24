# app.py — Streamlit dashboard for Zanjabeel Forecast System
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime, timedelta
import streamlit_authenticator as stauth

# =====================================================================
# PAGE SETUP
# =====================================================================
st.set_page_config(
    page_title="نظام توقع المبيعات — زنجبيل",
    page_icon="🍖",
    layout="wide"
)

# =====================================================================
# LOGIN
# =====================================================================
credentials = {
    "usernames": {
        "admin": {
            "name": "المدير",
            "password": "$2b$12$.bgkkBEGcggBbdfHaHjaxeI4mCjVcJknqA6267qe929/bWdZt0r0."
        },
        "zanjabeel": {
            "name": "مطعم زنجبيل",
            "password": "$2b$12$JADj/4A/WZX0iUA0Gubqzu6fA2cfk7ptZlcqaArFZoSO.diy0HuRO"
        }
    }
}

authenticator = stauth.Authenticate(
    credentials, "zanjabeel_cookie", "zanjabeel_key_2025", cookie_expiry_days=7
)

authenticator.login(location="main", fields={
    "Form name": "🍖 نظام توقع المبيعات — زنجبيل",
    "Username": "اسم المستخدم",
    "Password": "كلمة المرور",
    "Login": "دخول"
})

name        = st.session_state.get("name")
auth_status = st.session_state.get("authentication_status")
username    = st.session_state.get("username")

if auth_status is False:
    st.error("❌ اسم المستخدم أو كلمة المرور غير صحيحة")
    st.stop()
elif auth_status is None:
    st.info("أدخل بياناتك للدخول")
    st.stop()

authenticator.logout("🚪 خروج", location="sidebar")
st.sidebar.success(f"👋 أهلاً، {name}")

# =====================================================================
# MAIN DASHBOARD
# =====================================================================
st.markdown("""
<div style="background:linear-gradient(90deg,#1e293b,#0f172a);color:#f8fafc;
padding:20px 24px;border-radius:16px;margin-bottom:16px;">
  <h2 style="margin:0;font-size:22px;">🍖 نظام توقع المبيعات — زنجبيل</h2>
  <div style="opacity:0.7;margin-top:4px;font-size:13px;">
    توقعات الإنتاج اليومية بالذكاء الاصطناعي
  </div>
</div>
""", unsafe_allow_html=True)

# =====================================================================
# LOAD FORECAST RESULTS
# =====================================================================
FORECAST_PATH = "forecast_results.csv"

if not os.path.exists(FORECAST_PATH):
    st.warning("⚠️ لا توجد توقعات بعد. شغّل pipeline أولاً.")
    st.info("افتح التيرمنال واكتب: python run_all.py")
    st.stop()

df = pd.read_csv(FORECAST_PATH, encoding="utf-8-sig")
df["Sales_Date"] = pd.to_datetime(df["Sales_Date"])

# =====================================================================
# DATE SELECTOR — the feature you asked for
# =====================================================================
st.sidebar.markdown("---")
st.sidebar.subheader("📅 اختيار يوم التوقع")

available_dates = sorted(df["Sales_Date"].unique())
available_dates_str = [pd.Timestamp(d).strftime("%Y-%m-%d (%A)") for d in available_dates]

selected_idx = st.sidebar.selectbox(
    "اختر اليوم",
    range(len(available_dates_str)),
    format_func=lambda x: available_dates_str[x]
)

selected_date = available_dates[selected_idx]
day_df = df[df["Sales_Date"] == selected_date].copy()

# =====================================================================
# SUMMARY METRICS
# =====================================================================
col1, col2, col3 = st.columns(3)
col1.metric("📅 اليوم المختار", pd.Timestamp(selected_date).strftime("%Y-%m-%d"))
col2.metric("📦 عدد الأصناف", f"{len(day_df):,}")
col3.metric("📊 إجمالي الوحدات المتوقعة", f"{day_df['Model_Prediction'].sum():,.0f}")

st.markdown("---")

# =====================================================================
# FORECAST TABLE
# =====================================================================
st.subheader("📋 جدول توقعات الإنتاج")

# Add recommended stock column
day_df["الكمية_الموصى_بها"] = (day_df["Model_Prediction"] * 1.2).apply(np.ceil).astype(int)
day_df["التوقع"] = day_df["Model_Prediction"].round(1)

# Filter controls
col_a, col_b = st.columns(2)
with col_a:
    min_qty = st.slider("الحد الأدنى للكمية المتوقعة", 0, 50, 5)
with col_b:
    search = st.text_input("🔍 ابحث عن صنف", "")

filtered = day_df[day_df["Model_Prediction"] >= min_qty]
if search:
    filtered = filtered[filtered["Item_Name"].str.contains(search, na=False)]

filtered = filtered.sort_values("Model_Prediction", ascending=False).reset_index(drop=True)

st.dataframe(
    filtered[["Item_Name", "التوقع", "الكمية_الموصى_بها"]].rename(columns={
        "Item_Name": "اسم الصنف",
        "التوقع": "الكمية المتوقعة",
        "الكمية_الموصى_بها": "الكمية الموصى بتحضيرها"
    }),
    use_container_width=True,
    height=500
)

# =====================================================================
# DOWNLOAD BUTTON
# =====================================================================
csv_export = filtered[["Item_Name", "التوقع", "الكمية_الموصى_بها"]].to_csv(
    index=False, encoding="utf-8-sig"
)
st.download_button(
    label="📥 تحميل التوقعات كملف Excel",
    data=csv_export,
    file_name=f"forecast_{pd.Timestamp(selected_date).strftime('%Y-%m-%d')}.csv",
    mime="text/csv"
)