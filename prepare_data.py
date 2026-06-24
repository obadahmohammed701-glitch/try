import glob
import os
import pandas as pd


# =====================================================================
# STEP 1: LOAD AND COMBINE FILES (WITH BULLETPROOF CACHING)
# =====================================================================
def load_and_combine_files(folder_path):
    """Finds all Excel files in the specified folder, reads them,

    and concatenates them. Caches to CSV/Parquet for speed.
    """
    # We will use a flat CSV cache as a safe baseline to prevent type errors
    cache_file = os.path.join(folder_path, "combined_raw_data_cache.csv")

    # 1. Check if the cache file already exists from a previous run
    if os.path.exists(cache_file):
        print(
            f"⚡ Cache found! Loading optimized data instantly from: {os.path.basename(cache_file)}"
        )
        # low_memory=False ensures pandas reads large files without guessing types incorrectly
        return pd.read_csv(cache_file, low_memory=False)

    # 2. If no cache exists, scan and read the Excel files
    search_pattern = os.path.join(folder_path, "*.xlsx")
    all_files = glob.glob(search_pattern)

    if not all_files:
        raise FileNotFoundError(
            f"❌ File Error: No Excel files (.xlsx) found in: {folder_path}"
        )

    print(
        f"No cache found. Processing {len(all_files)} Excel files. Please wait, this takes a moment..."
    )
    df_list = []

    for file in all_files:
        print(f" Reading -> {os.path.basename(file)}")
        try:
            individual_df = pd.read_excel(file)
            df_list.append(individual_df)
        except Exception as file_err:
            print(
                f"❌ Error reading specific file {os.path.basename(file)}: {file_err}"
            )
            raise file_err

    # Stitch them all vertically
    combined_df = pd.concat(df_list, axis=0, ignore_index=True)

    # 3. Clean up object types before writing to cache
    # This turns any mixed columns safely into text so the exporter doesn't crash
    for col in combined_df.select_dtypes(include=["object"]).columns:
        combined_df[col] = combined_df[col].astype(str)

    print("Saving combined data to local cache to bypass Excel processing next time...")
    combined_df.to_csv(cache_file, index=False)

    print(
        f"✅ Step 1 Complete! Total rows loaded into memory: {len(combined_df)}"
    )
    return combined_df


# =====================================================================
# STEP 2: CLEAN AND PARSE DATETIME (WITH ARABIC FIX)
# =====================================================================
def clean_and_parse_datetime(df, date_column_name):
    """Pre-processes Arabic AM/PM indicators ('ص'/'م') and extracts a date-only timeline."""
    if date_column_name not in df.columns:
        raise KeyError(
            f"❌ Target column '{date_column_name}' not found. Available columns are: {list(df.columns)}"
        )

    df = df.copy()
    print(f"\nParsing dates from column: '{date_column_name}'...")

    # Force conversion to string to manipulate the raw text characters safely
    date_series = df[date_column_name].astype(str)

    # Replace Arabic AM/PM markers with standard ones
    date_series = date_series.str.replace("ص", "AM", regex=False)
    date_series = date_series.str.replace("م", "PM", regex=False)
    date_series = date_series.str.strip()

    # Parse strings into actual computer-readable dates
    df[date_column_name] = pd.to_datetime(
        date_series, dayfirst=True, errors="coerce"
    )

    # Drop invalid rows if any are found
    missing_dates = df[date_column_name].isna().sum()
    if missing_dates > 0:
        print(
            f"⚠️ Note: {missing_dates} rows could not be parsed as dates and will be skipped."
        )
        df = df.dropna(subset=[date_column_name])

    # Extract date-only format (YYYY-MM-DD)
    df["Sales_Date"] = pd.to_datetime(df[date_column_name].dt.date)

    min_date = df["Sales_Date"].min().strftime("%Y-%m-%d")
    max_date = df["Sales_Date"].max().strftime("%Y-%m-%d")

    print("✅ Step 2 Complete! Datetime successfully parsed.")
    print(f"Dataset Row Count: {len(df)}")
    print(f"Timeline Range: {min_date} to {max_date}\n")

    return df


# =====================================================================
# PIPELINE EXECUTION
# =====================================================================
if __name__ == "__main__":
    target_folder = r"C:\Users\obada\Desktop\Zanjabeel"

    try:
        # Run Step 1
        raw_data = load_and_combine_files(target_folder)

        # Run Step 2
        date_column_header = "تاريخ الفاتورة"
        cleaned_data = clean_and_parse_datetime(raw_data, date_column_header)

        print("--- Pipeline Verification Check ---")
        print(cleaned_data[[date_column_header, "Sales_Date"]].head())

    except Exception as e:
        print(f"\n❌ Pipeline stopped due to error:\n{e}")
    


# =====================================================================
# STEP 3: DAILY AGGREGATION
# =====================================================================
def perform_daily_aggregation(df, date_col, item_col, qty_col):
    """Step 3: Groups transactional data by Date and Item Name,

    summing up the total daily quantities sold.
    """
    df = df.copy()

    # Safety check: Catch spelling/column name errors instantly before grouping
    for col in [date_col, item_col, qty_col]:
        if col not in df.columns:
            raise KeyError(
                f"❌ Bug Alert: Column '{col}' missing! Current columns: {list(df.columns)}"
            )

    print("Collapsing transactions into daily item totals...")

    # 1. Group by Date and Item, then sum the Quantities
    aggregated_df = (
        df.groupby([date_col, item_col])[qty_col].sum().reset_index()
    )

    # 2. Rename columns to clean, standard English headers for modeling
    rename_mapping = {
        date_col: "Sales_Date",
        item_col: "Item_Name",
        qty_col: "Daily_Qty",
    }
    aggregated_df = aggregated_df.rename(columns=rename_mapping)

    # =====================================================================
    # 🌟 ADD THE FIX RIGHT HERE! 🌟
    # =====================================================================
    # Convert item name to string and strip out any leading '=', '+', or spaces
    # that cause Excel formula bugs and fracture your ML categories.
    aggregated_df["Item_Name"] = aggregated_df["Item_Name"].astype(str)
    aggregated_df["Item_Name"] = aggregated_df["Item_Name"].str.lstrip(
        "=+\t "
    )
    # =====================================================================
# =====================================================================
    # FILTER MODIFIERS AND NOISE ITEMS
    # Remove items that are toppings, removals, or notes — not real menu items
    # These corrupt the model because they're not stockable inventory
    # =====================================================================
    modifier_prefixes = [
        'اضافة', 'إضافة', 'اضافه', 'إضافه',  # additions
        'بدون', 'بدون ',                          # removals ("without")
        'ملاحظة', 'ملاحظه', 'note',              # notes
        'خاص', 'تعليق',                           # special instructions
    ]

    def is_noise_item(name):
        name = str(name).strip()
        return any(name.startswith(prefix) for prefix in modifier_prefixes)

    before_filter = len(aggregated_df)
    aggregated_df = aggregated_df[
        ~aggregated_df["Item_Name"].apply(is_noise_item)
    ].reset_index(drop=True)
    after_filter = len(aggregated_df)

    print(f"✅ Modifier filter: removed {before_filter - after_filter} noise rows")
    print(f"   Real menu items remaining: {aggregated_df['Item_Name'].nunique()}")
    # 3. Sort chronologically by date and item so it reads like a timeline
    aggregated_df = aggregated_df.sort_values(
        by=["Sales_Date", "Item_Name"]
    ).reset_index(drop=True)

    print("✅ Step 3 Complete! Data collapsed into a clean time-series.")
    print(f"Row count condensed from {len(df)} down to {len(aggregated_df)}.")

    return aggregated_df
# =====================================================================
# STEP 3.5: FUZZY NAME STANDARDIZATION (TF-IDF SIMILARITY)
# Merges items that are the same product but spelled differently
# =====================================================================
def standardize_item_names(df, similarity_threshold=0.85):
    """
    Uses TF-IDF cosine similarity to find item names that are
    very similar to each other and merges them into one standard name.

    Example:
        "اضافة جبنه" and "اضافه جبنه" → both become "اضافة جبنه"
        "شاورما صغير" and "ساندويش شاورما صغير" → merged if similarity > 0.85

    threshold: 0.85 means 85% similar. Higher = stricter matching.
    Lower values merge more aggressively. 0.85 is safe for Arabic names.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    unique_names = df["Item_Name"].unique().tolist()
    print(f"\nRunning fuzzy name standardization on {len(unique_names)} unique items...")

    # Build TF-IDF matrix — treats each character as a token (char-level)
    # This works better for Arabic than word-level because Arabic words vary
    # in spelling due to ة vs ه, ا vs أ vs إ, etc.
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 3))
    tfidf_matrix = vectorizer.fit_transform(unique_names)

    # Calculate similarity between all pairs
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # Build a mapping: variant name → canonical (standard) name
    # We always keep the name that appears first (alphabetically or by frequency)
    name_mapping = {}
    merged_count = 0

    for i, name_i in enumerate(unique_names):
        if name_i in name_mapping:
            continue  # already mapped to something else

        # Find all names similar to this one
        similar_indices = np.where(similarity_matrix[i] >= similarity_threshold)[0]

        for j in similar_indices:
            if j != i:
                variant_name = unique_names[j]
                if variant_name not in name_mapping:
                    # Map the variant to the canonical name
                    name_mapping[variant_name] = name_i
                    merged_count += 1

    # Apply the mapping
    df["Item_Name"] = df["Item_Name"].replace(name_mapping)

    # Re-aggregate because merging names creates duplicate rows for same date+item
    df = df.groupby(
        ["Sales_Date", "Item_Name"]
    )["Daily_Qty"].sum().reset_index()

    print(f"✅ Fuzzy matching complete!")
    print(f"   Items merged: {merged_count}")
    print(f"   Items before: {len(unique_names)}")
    print(f"   Items after:  {df['Item_Name'].nunique()}")

    # Show what was merged (for your review)
    if name_mapping:
        print("\n   Merged pairs (variant → canonical):")
        for variant, canonical in list(name_mapping.items())[:10]:
            print(f"   '{variant}' → '{canonical}'")
        if len(name_mapping) > 10:
            print(f"   ... and {len(name_mapping) - 10} more")

    return df
#-------
# =====================================================================
# STEP 4: ZERO-PADDING (TIMELINE SEQUENCE GRID GENERATOR)
# =====================================================================
def handle_zero_padding(df):
    """Enforces absolute chronological continuity for every individual item.

    Constructs a complete Cartesian grid combination of every calendar date
    and every unique menu item, substituting true zeros for dates without sales.
    """
    df = df.copy()
    print("\nGenerating complete multi-product calendar grid for zero-padding...")

    # 1. Extract all discrete items and establish an unbroken, linear day-by-day calendar
    unique_items = df["Item_Name"].unique()
    full_date_range = pd.date_range(
        start=df["Sales_Date"].min(), end=df["Sales_Date"].max(), freq="D"
    )

    # 2. Build a structural master framework index cross-multiplying items and calendar days
    grid = pd.MultiIndex.from_product(
        [full_date_range, unique_items], names=["Sales_Date", "Item_Name"]
    )

    # 3. Project current sales counts into the comprehensive grid slots
    padded_df = (
        df.set_index(["Sales_Date", "Item_Name"])
        .reindex(grid)
        .reset_index()
    )

    # 4. Reallocate clean numerical zeros to historical voids (NaN entries)
    padded_df["Daily_Qty"] = padded_df["Daily_Qty"].fillna(0)

    # 5. Order layout linearly by calendar timeline sequence
    padded_df = padded_df.sort_values(by=["Sales_Date", "Item_Name"]).reset_index(
        drop=True
    )

    print("✅ Step 4 Complete! Missing timeline gaps filled with 0.")
    print(f"Row count expanded from {len(df)} up to {len(padded_df)} entries.")
    return padded_df


# =====================================================================
# RUNTIME CONTROL PIPELINE
# =====================================================================
if __name__ == "__main__":
    # Define primary workspace storage folder location path
    target_folder = r"C:\Users\obada\Desktop\Zanjabeel"

    try:
        # Step 1 & 2: Execution and clean extraction phase
        raw_data = load_and_combine_files(target_folder)
        cleaned_data = clean_and_parse_datetime(raw_data, "تاريخ الفاتورة")

        # Step 3: Compress transactional sales history items into daily balances
        print("\n--- Running Step 3: Daily Aggregation ---")
        daily_data = perform_daily_aggregation(
            cleaned_data, "Sales_Date", "اسم الصنف", "الكمية"
        )

        
       # Step 3.5: Standardize similar item names before zero-padding
        print("\n--- Running Step 3.5: Fuzzy Name Standardization ---")
        daily_data = standardize_item_names(daily_data, similarity_threshold=0.85)

        # Step 4: Zero-padding (now on cleaner, merged item names)
        print("\n--- Running Step 4: Zero-Padding ---")
        final_padded_data = handle_zero_padding(daily_data)

        # Output file routing path address setup
        output_file_path = os.path.join(target_folder, "cleaned_daily_sales.csv")

        # Save data table explicitly applying UTF-8-SIG encoding signature for clean Arabic text support
        final_padded_data.to_csv(output_file_path, index=False, encoding="utf-8-sig")

        print(f"\n💾 Data cached and updated on your desktop!")
        print(f"--> Target Output Destination: 'cleaned_daily_sales.csv'")

        print("\n--- Step 4 Zero-Padding Preview Check ---")
        print(final_padded_data.head(10))

    except Exception as e:
        print(f"\n❌ Pipeline stopped due to execution error:\n{e}")