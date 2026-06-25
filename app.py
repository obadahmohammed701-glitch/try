# app.py — Zanjabeel Forecast System (Cloud Ready)
# Full pipeline embedded: load → clean → aggregate → pad → feature engineer → train → predict
# No external script calls. Owner uploads Excel files → system trains and forecasts automatically.

import io
import hashlib
import numpy as np
import pandas as pd
import lightgbm as lgb
import streamlit as st
import streamlit_authenticator as stauth
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
# PIPELINE CONSTANTS
# ============================================================
MODIFIER_PREFIXES = [
    'اضافة', 'إضافة', 'اضافه', 'إضافه',
    'بدون', 'بدون ',
    'ملاحظة', 'ملاحظه', 'note',
    'خاص', 'تعليق',
]
VOLUME_THRESHOLD = 15.0
FEATURES = [
    "Day_Of_Week", "Month", "Is_Weekend", "Day_Of_Month",
    "Is_Salary_Period", "Lag_1", "Lag_7", "Rolling_Mean_7"
]

# ============================================================
# PIPELINE FUNCTIONS
# ============================================================

def is_noise_item(name: str) -> bool:
    name = str(name).strip()
    return any(name.startswith(p) for p in MODIFIER_PREFIXES)


def step1_load(uploaded_files_bytes, filenames):
    """Read uploaded Excel bytes and combine into one DataFrame."""
    dfs = []
    for file_bytes, fname in zip(uploaded_files_bytes, filenames):
        try:
            df = pd.read_excel(io.BytesIO(file_bytes))
            dfs.append(df)
        except Exception as e:
            st.warning(f"⚠️ خطأ في قراءة {fname}: {e}")
    if not dfs:
        raise ValueError("لا توجد ملفات صالحة للقراءة")
    combined = pd.concat(dfs, ignore_index=True)
    for col in combined.select_dtypes(include=["object"]).columns:
        combined[col] = combined[col].astype(str)
    return combined


def step2_parse_datetime(df):
    """Parse Arabic datetime column (ص/م → AM/PM)."""
    date_col = "تاريخ الفاتورة"
    if date_col not in df.columns:
        raise KeyError(f"العمود '{date_col}' غير موجود. الأعمدة المتاحة: {list(df.columns)}")
    df = df.copy()
    date_series = (
        df[date_col].astype(str)
        .str.replace("ص", "AM", regex=False)
        .str.replace("م", "PM", regex=False)
        .str.strip()
    )
    df[date_col] = pd.to_datetime(date_series, dayfirst=True, errors="coerce")
    missing = df[date_col].isna().sum()
    if missing > 0:
        st.warning(f"⚠️ {missing} صف لم يتم تحليل تاريخه وسيتم تجاهله.")
    df = df.dropna(subset=[date_col])
    df["Sales_Date"] = pd.to_datetime(df[date_col].dt.date)
    return df


def step3_aggregate(df):
    """Daily aggregation → remove modifiers → remove zero/negative qty."""
    item_col = "اسم الصنف"
    qty_col  = "الكمية"
    for col in [item_col, qty_col]:
        if col not in df.columns:
            raise KeyError(f"العمود '{col}' غير موجود في البيانات")

    df = df.copy()
    df[qty_col] = pd.to_numeric(df[qty_col], errors="coerce").fillna(0)

    agg = (
        df.groupby(["Sales_Date", item_col])[qty_col]
        .sum()
        .reset_index()
        .rename(columns={item_col: "Item_Name", qty_col: "Daily_Qty"})
    )
    # Clean leading special chars
    agg["Item_Name"] = agg["Item_Name"].astype(str).str.lstrip("=+\t ")
    # Remove modifier/noise items
    agg = agg[~agg["Item_Name"].apply(is_noise_item)].reset_index(drop=True)
    # Remove zero/negative quantity rows
    agg = agg[agg["Daily_Qty"] > 0].reset_index(drop=True)
    return agg


def step35_fuzzy_standardize(df, threshold=0.85):
    """TF-IDF fuzzy name deduplication for Arabic item names."""
    unique_names = df["Item_Name"].unique().tolist()
    if len(unique_names) < 2:
        return df

    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 3))
    tfidf_matrix = vectorizer.fit_transform(unique_names)
    sim_matrix = cosine_similarity(tfidf_matrix)

    name_mapping = {}
    for i, name_i in enumerate(unique_names):
        if name_i in name_mapping:
            continue
        similar_idx = np.where(sim_matrix[i] >= threshold)[0]
        for j in similar_idx:
            if j != i:
                variant = unique_names[j]
                if variant not in name_mapping:
                    name_mapping[variant] = name_i

    df["Item_Name"] = df["Item_Name"].replace(name_mapping)
    df = (
        df.groupby(["Sales_Date", "Item_Name"])["Daily_Qty"]
        .sum()
        .reset_index()
    )
    return df


def step4_zero_pad(df):
    """Fill missing date-item combinations with 0."""
    full_dates   = pd.date_range(df["Sales_Date"].min(), df["Sales_Date"].max(), freq="D")
    unique_items = df["Item_Name"].unique()
    grid = pd.MultiIndex.from_product(
        [full_dates, unique_items], names=["Sales_Date", "Item_Name"]
    )
    padded = (
        df.set_index(["Sales_Date", "Item_Name"])
        .reindex(grid)
        .reset_index()
    )
    padded["Daily_Qty"] = padded["Daily_Qty"].fillna(0)
    padded = padded.sort_values(["Sales_Date", "Item_Name"]).reset_index(drop=True)
    return padded


def step5_feature_engineer(df):
    """Add calendar, lag, and rolling features."""
    df = df.sort_values(["Item_Name", "Sales_Date"]).reset_index(drop=True)
    df["Day_Of_Week"]      = df["Sales_Date"].dt.dayofweek
    df["Month"]            = df["Sales_Date"].dt.month
    df["Day_Of_Month"]     = df["Sales_Date"].dt.day
    df["Is_Weekend"]       = df["Day_Of_Week"].isin([4, 5]).astype(int)
    df["Is_Salary_Period"] = (
        (df["Day_Of_Month"] >= 22) | (df["Day_Of_Month"] <= 2)
    ).astype(int)
    df["Lag_1"]            = df.groupby("Item_Name")["Daily_Qty"].shift(1)
    df["Lag_7"]            = df.groupby("Item_Name")["Daily_Qty"].shift(7)
    df["Rolling_Mean_7"]   = (
        df.groupby("Item_Name")["Daily_Qty"]
        .transform(lambda x: x.shift(1).rolling(7).mean())
    )
    df = df.dropna().reset_index(drop=True)
    return df


def step6_train_predict(df):
    """Dual LightGBM training (High-Volume / Low-Volume) with buffer stock."""
    item_volumes     = df.groupby("Item_Name")["Daily_Qty"].mean()
    high_vol_items   = item_volumes[item_volumes >= VOLUME_THRESHOLD].index.tolist()
    low_vol_items    = [i for i in item_volumes.index if i not in high_vol_items]

    cutoff = df["Sales_Date"].max() - pd.Timedelta(days=14)
    train  = df[df["Sales_Date"] <= cutoff]
    test   = df[df["Sales_Date"] >  cutoff]

    if len(test) == 0:
        raise ValueError(
            "البيانات غير كافية للتنبؤ — يجب أن تحتوي على أكثر من 14 يوماً من المبيعات"
        )

    all_preds = []

    for category, item_list in [("High-Volume", high_vol_items), ("Low-Volume", low_vol_items)]:
        tr = train[train["Item_Name"].isin(item_list)]
        te = test[test["Item_Name"].isin(item_list)]
        if len(tr) == 0 or len(te) == 0:
            continue

        X_tr = tr[FEATURES + ["Item_Name"]].copy()
        X_te = te[FEATURES + ["Item_Name"]].copy()
        y_tr = tr["Daily_Qty"]

        X_tr["Item_Name"] = X_tr["Item_Name"].astype("category")
        X_te["Item_Name"] = X_te["Item_Name"].astype("category")

        model = lgb.LGBMRegressor(
            n_estimators=100, learning_rate=0.05,
            random_state=42, verbose=-1
        )
        model.fit(X_tr, y_tr)

        preds  = np.clip(model.predict(X_te), 0, None)
        buffer = 0.15 if category == "High-Volume" else 0.30

        meta = te[["Sales_Date", "Item_Name", "Daily_Qty"]].copy()
        meta["Model_Prediction"]  = np.round(preds, 1)
        meta["Recommended_Stock"] = np.ceil(preds * (1 + buffer)).astype(int)
        meta["Category"]          = category
        all_preds.append(meta)

    if not all_preds:
        raise ValueError("لم يتمكن النموذج من التدريب — تحقق من البيانات")

    results = pd.concat(all_preds).sort_values(["Sales_Date", "Item_Name"])
    return results


# ============================================================
# MAIN CACHED PIPELINE ENTRY POINT
# ============================================================

@st.cache_data(show_spinner=False)
def run_full_pipeline(cache_key: str, files_bytes: list, filenames: list):
    """
    cache_key: MD5 hash of all file contents — ensures rerun only when files change.
    Returns (results_df, error_message).
    """
    try:
        raw      = step1_load(files_bytes, filenames)
        parsed   = step2_parse_datetime(raw)
        daily    = step3_aggregate(parsed)
        daily    = step35_fuzzy_standardize(daily, threshold=0.85)
        padded   = step4_zero_pad(daily)
        featured = step5_feature_engineer(padded)
        results  = step6_train_predict(featured)
        return results, None
    except Exception as e:
        return None, str(e)


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
# READ FILES — seek(0) prevents empty-bytes bug on Streamlit Cloud
# ============================================================
files_bytes = []
for f in uploaded_files:
    f.seek(0)
    files_bytes.append(f.read())

filenames  = [f.name for f in uploaded_files]
cache_key  = hashlib.md5(b"".join(files_bytes)).hexdigest()

# ============================================================
# RUN PIPELINE
# ============================================================
with st.spinner("⏳ جاري تحليل البيانات وتدريب النموذج... (30-60 ثانية)"):
    results_df, error = run_full_pipeline(cache_key, files_bytes, filenames)

if error:
    st.error(f"❌ {error}")
    st.stop()

st.success(
    f"✅ اكتمل التحليل — "
    f"{results_df['Item_Name'].nunique():,} صنف، "
    f"{results_df['Sales_Date'].nunique()} يوم في فترة الاختبار"
)

# ============================================================
# DATE SELECTOR
# ============================================================
st.sidebar.markdown("---")
st.sidebar.subheader("📅 اختيار يوم التوقع")

available_dates = sorted(results_df["Sales_Date"].unique())
date_labels     = [pd.Timestamp(d).strftime("%Y-%m-%d (%A)") for d in available_dates]

selected_idx  = st.sidebar.selectbox(
    "اختر اليوم", range(len(date_labels)),
    format_func=lambda x: date_labels[x]
)
selected_date = available_dates[selected_idx]
day_df        = results_df[results_df["Sales_Date"] == selected_date].copy()

# ============================================================
# METRICS ROW
# ============================================================
col1, col2, col3, col4 = st.columns(4)
col1.metric("📅 اليوم",                    pd.Timestamp(selected_date).strftime("%Y-%m-%d"))
col2.metric("📦 عدد الأصناف",             f"{len(day_df):,}")
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
    filtered = filtered[
        filtered["Item_Name"].str.contains(search, na=False, case=False)
    ]

filtered = filtered.sort_values("Model_Prediction", ascending=False).reset_index(drop=True)

st.dataframe(
    filtered[["Item_Name", "Model_Prediction", "Recommended_Stock", "Category"]].rename(columns={
        "Item_Name":         "اسم الصنف",
        "Model_Prediction":  "الكمية المتوقعة",
        "Recommended_Stock": "الكمية الموصى بتحضيرها",
        "Category":          "الفئة"
    }),
    use_container_width=True,
    height=500
)

# ============================================================
# DOWNLOAD
# ============================================================
# AFTER (fixed code)
csv_df = filtered[["Item_Name", "Model_Prediction", "Recommended_Stock"]].rename(columns={
    "Item_Name":         "اسم الصنف",
    "Model_Prediction":  "الكمية المتوقعة",
    "Recommended_Stock": "الكمية الموصى بتحضيرها",
})
csv_out = csv_df.to_csv(index=False, encoding="utf-8-sig")
st.download_button(
    "📥 تحميل التوقعات CSV",
    data=csv_out,
    file_name=f"forecast_{pd.Timestamp(selected_date).strftime('%Y-%m-%d')}.csv",
    mime="text/csv"
)