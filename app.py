# app.py — Complete cloud-ready Zanjabeel Forecast System
# The shop owner uploads files here — model trains and predicts automatically

import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import glob
import joblib
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import streamlit_authenticator as stauth

# ============================================================
# PAGE SETUP
# ============================================================
st.set_page_config(
    page_title="نظام توقع المبيعات — زنجبيل",
    page_icon="🍖",
    layout="wide"
)

# ============================================================
# LOGIN
# ============================================================
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
    credentials, "zanjabeel_cookie", "zanjabeel_key_2025",
    cookie_expiry_days=7
)

authenticator.login(location="main", fields={
    "Form name": "🍖 نظام توقع المبيعات — زنجبيل",
    "Username": "اسم المستخدم",
    "Password": "كلمة المرور",
    "Login": "دخول"
})

auth_status = st.session_state.get("authentication_status")
name = st.session_state.get("name")

if auth_status is False:
    st.error("❌ اسم المستخدم أو كلمة المرور غير صحيحة")
    st.stop()
elif auth_status is None:
    st.markdown("""
    <div style="text-align:center;padding:60px 20px;">
        <div style="font-size:56px">🍖</div>
        <div style="font-size:22px;font-weight:900;margin-top:10px">زنجبيل — نظام التنبؤ الذكي</div>
        <div style="color:#64748b;margin-top:6px;font-size:13px">أدخل بياناتك للوصول إلى النظام</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

authenticator.logout("🚪 خروج", location="sidebar")
st.sidebar.success(f"👋 أهلاً، {name}")

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div style="background:linear-gradient(90deg,#1e293b,#0f172a);color:#f8fafc;
padding:20px 24px;border-radius:16px;margin-bottom:16px;">
  <h2 style="margin:0;font-size:22px;">🍖 نظام توقع المبيعات — زنجبيل</h2>
  <div style="opacity:0.7;margin-top:4px;font-size:13px;">
      ارفع بيانات المبيعات وسيقوم النظام بتوليد توقعات الإنتاج تلقائياً
  </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# ALL PIPELINE FUNCTIONS (embedded — no external file calls)
# ============================================================

MODIFIER_PREFIXES = [
    'اضافة', 'إضافة', 'اضافه', 'إضافه',
    'بدون', 'ملاحظة', 'ملاحظه', 'note', 'خاص'
]

def is_noise_item(name):
    name = str(name).strip()
    return any(name.startswith(p) for p in MODIFIER_PREFIXES)

@st.cache_data
def run_full_pipeline(uploaded_files_bytes, filenames):
    """
    Takes uploaded file bytes, runs the complete pipeline,
    returns forecast dataframe.
    Cached — only reruns if files change.
    """
    # Step 1: Load and combine all uploaded files
    dfs = []
    for file_bytes, fname in zip(uploaded_files_bytes, filenames):
        try:
            df = pd.read_excel(io.BytesIO(file_bytes))
            dfs.append(df)
        except Exception as e:
            st.warning(f"⚠️ خطأ في قراءة {fname}: {e}")

    if not dfs:
        return None, "لا توجد ملفات صالحة"

    combined = pd.concat(dfs, ignore_index=True)

    # Step 2: Parse datetime
    date_col = "تاريخ الفاتورة"
    if date_col not in combined.columns:
        return None, f"العمود '{date_col}' غير موجود في البيانات"

    combined[date_col] = (
        combined[date_col].astype(str)
        .str.replace("ص", "AM", regex=False)
        .str.replace("م", "PM", regex=False)
    )
    combined[date_col] = pd.to_datetime(combined[date_col], dayfirst=True, errors="coerce")
    combined = combined.dropna(subset=[date_col])
    combined["Sales_Date"] = pd.to_datetime(combined[date_col].dt.date)

    # Step 3: Daily aggregation
    item_col = "اسم الصنف"
    qty_col  = "الكمية"

    if item_col not in combined.columns or qty_col not in combined.columns:
        return None, "أعمدة اسم الصنف أو الكمية غير موجودة"

    combined[qty_col] = pd.to_numeric(combined[qty_col], errors="coerce").fillna(0)

    daily = (
        combined.groupby(["Sales_Date", item_col])[qty_col]
        .sum()
        .reset_index()
        .rename(columns={item_col: "Item_Name", qty_col: "Daily_Qty"})
    )

    # Clean item names
    daily["Item_Name"] = daily["Item_Name"].astype(str).str.lstrip("=+\t ")

    # Remove modifiers
    before = len(daily)
    daily = daily[~daily["Item_Name"].apply(is_noise_item)].reset_index(drop=True)

    # Remove zero/negative qty
    daily = daily[daily["Daily_Qty"] > 0].reset_index(drop=True)

    # Step 4: Zero-padding
    full_dates  = pd.date_range(daily["Sales_Date"].min(), daily["Sales_Date"].max(), freq="D")
    unique_items = daily["Item_Name"].unique()
    grid = pd.MultiIndex.from_product([full_dates, unique_items], names=["Sales_Date","Item_Name"])
    daily = (
        daily.set_index(["Sales_Date","Item_Name"])
        .reindex(grid).reset_index()
    )
    daily["Daily_Qty"] = daily["Daily_Qty"].fillna(0)

    # Step 5: Feature engineering
    daily = daily.sort_values(["Item_Name","Sales_Date"]).reset_index(drop=True)
    daily["Day_Of_Week"]    = daily["Sales_Date"].dt.dayofweek
    daily["Month"]          = daily["Sales_Date"].dt.month
    daily["Day_Of_Month"]   = daily["Sales_Date"].dt.day
    daily["Is_Weekend"]     = daily["Day_Of_Week"].isin([4,5]).astype(int)
    daily["Is_Salary_Period"] = (
        (daily["Day_Of_Month"] >= 22) | (daily["Day_Of_Month"] <= 2)
    ).astype(int)
    daily["Lag_1"]          = daily.groupby("Item_Name")["Daily_Qty"].shift(1)
    daily["Lag_7"]          = daily.groupby("Item_Name")["Daily_Qty"].shift(7)
    daily["Rolling_Mean_7"] = (
        daily.groupby("Item_Name")["Daily_Qty"]
        .transform(lambda x: x.shift(1).rolling(7).mean())
    )
    daily = daily.dropna().reset_index(drop=True)

    # Step 6: Train model
    VOLUME_THRESHOLD = 15.0
    item_volumes     = daily.groupby("Item_Name")["Daily_Qty"].mean()
    high_vol_items   = item_volumes[item_volumes >= VOLUME_THRESHOLD].index.tolist()

    cutoff = daily["Sales_Date"].max() - pd.Timedelta(days=14)
    train  = daily[daily["Sales_Date"] <= cutoff]
    test   = daily[daily["Sales_Date"] >  cutoff]

    features = [
        "Day_Of_Week","Month","Is_Weekend","Day_Of_Month",
        "Is_Salary_Period","Lag_1","Lag_7","Rolling_Mean_7"
    ]

    all_preds = []

    for category, item_list in [
        ("High-Volume", high_vol_items),
        ("Low-Volume", [i for i in item_volumes.index if i not in high_vol_items])
    ]:
        tr = train[train["Item_Name"].isin(item_list)]
        te = test[test["Item_Name"].isin(item_list)]
        if len(tr) == 0 or len(te) == 0:
            continue

        X_tr = tr[features + ["Item_Name"]].copy()
        X_te = te[features + ["Item_Name"]].copy()
        y_tr = tr["Daily_Qty"]

        X_tr["Item_Name"] = X_tr["Item_Name"].astype("category")
        X_te["Item_Name"] = X_te["Item_Name"].astype("category")

        model = lgb.LGBMRegressor(
            n_estimators=100, learning_rate=0.05,
            random_state=42, verbose=-1
        )
        model.fit(X_tr, y_tr)

        preds = np.clip(model.predict(X_te), 0, None)
        buffer = 0.15 if category == "High-Volume" else 0.30

        meta = te[["Sales_Date","Item_Name","Daily_Qty"]].copy()
        meta["Model_Prediction"]   = np.round(preds, 1)
        meta["Recommended_Stock"]  = np.ceil(preds * (1 + buffer)).astype(int)
        meta["Category"]           = category
        all_preds.append(meta)

    if not all_preds:
        return None, "لم يتمكن النموذج من التدريب"

    results = pd.concat(all_preds).sort_values(["Sales_Date","Item_Name"])
    return results, None


# ============================================================
# FILE UPLOAD UI
# ============================================================
st.subheader("📂 رفع ملفات البيانات")

uploaded_files = st.file_uploader(
    "ارفع ملفات Excel للمبيعات (يمكن رفع أكثر من ملف)",
    type=["xlsx", "xls"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("👆 ارفع ملفات بيانات المبيعات لبدء التنبؤ")
    st.stop()

# ============================================================
# RUN PIPELINE
# ============================================================
with st.spinner("⏳ جاري تحليل البيانات وتدريب النموذج... (30-60 ثانية)"):
    files_bytes = [f.read() for f in uploaded_files]
    filenames   = [f.name for f in uploaded_files]

    results_df, error = run_full_pipeline(
        tuple(files_bytes),
        tuple(filenames)
    )

if error:
    st.error(f"❌ {error}")
    st.stop()

st.success(f"✅ اكتمل التحليل — {results_df['Item_Name'].nunique():,} صنف، {results_df['Sales_Date'].nunique()} يوم")

# ============================================================
# DATE SELECTOR
# ============================================================
st.sidebar.markdown("---")
st.sidebar.subheader("📅 اختيار يوم التوقع")

available_dates = sorted(results_df["Sales_Date"].unique())
date_labels     = [pd.Timestamp(d).strftime("%Y-%m-%d (%A)") for d in available_dates]

selected_idx  = st.sidebar.selectbox("اختر اليوم", range(len(date_labels)),
                                      format_func=lambda x: date_labels[x])
selected_date = available_dates[selected_idx]
day_df        = results_df[results_df["Sales_Date"] == selected_date].copy()

# ============================================================
# METRICS
# ============================================================
col1, col2, col3, col4 = st.columns(4)
col1.metric("📅 اليوم", pd.Timestamp(selected_date).strftime("%Y-%m-%d"))
col2.metric("📦 عدد الأصناف", f"{len(day_df):,}")
col3.metric("📊 إجمالي الوحدات المتوقعة", f"{day_df['Model_Prediction'].sum():,.0f}")
col4.metric("📦 الكمية الموصى بتحضيرها", f"{day_df['Recommended_Stock'].sum():,.0f}")

st.markdown("---")

# ============================================================
# FORECAST TABLE WITH FILTERS
# ============================================================
st.subheader("📋 جدول توقعات الإنتاج")

cola, colb = st.columns(2)
with cola:
    min_qty = st.slider("الحد الأدنى للكمية", 0, 50, 5)
with colb:
    search = st.text_input("🔍 ابحث عن صنف", "")

filtered = day_df[day_df["Model_Prediction"] >= min_qty].copy()
if search:
    filtered = filtered[filtered["Item_Name"].str.contains(search, na=False)]

filtered = filtered.sort_values("Model_Prediction", ascending=False).reset_index(drop=True)

st.dataframe(
    filtered[["Item_Name","Model_Prediction","Recommended_Stock","Category"]].rename(columns={
        "Item_Name":          "اسم الصنف",
        "Model_Prediction":   "الكمية المتوقعة",
        "Recommended_Stock":  "الكمية الموصى بتحضيرها",
        "Category":           "الفئة"
    }),
    use_container_width=True,
    height=500
)

# ============================================================
# DOWNLOAD
# ============================================================
csv_out = filtered[["Item_Name","Model_Prediction","Recommended_Stock"]].to_csv(
    index=False, encoding="utf-8-sig"
)
st.download_button(
    "📥 تحميل التوقعات",
    data=csv_out,
    file_name=f"forecast_{pd.Timestamp(selected_date).strftime('%Y-%m-%d')}.csv",
    mime="text/csv"
)