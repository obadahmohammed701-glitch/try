import os
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =====================================================================
# CONFIGURATION & ROUTING
# =====================================================================
DATA_PATH = r"C:\Users\obada\Desktop\Zanjabeel\cleaned_daily_sales.csv"
VOLUME_THRESHOLD = 15.0  # Threshold to separate Main Meals from Add-ons/Notes


def load_and_prepare_features(file_path):
    """Loads the zero-padded dataset and engineers all predictive features

    (Calendar markers, Jordan Salary Windows, Lags, and Rolling Averages).
    """
    print("⏳ Loading cleaned daily time-series grid...")
    df = pd.read_csv(file_path)
    df["Sales_Date"] = pd.to_datetime(df["Sales_Date"])

    # PHASE A: CALENDAR & REGIONAL ECONOMIC FEATURES
    print("Engineering calendar and regional economic features...")
    df["Day_Of_Week"] = df["Sales_Date"].dt.dayofweek
    df["Month"] = df["Sales_Date"].dt.month
    df["Day_Of_Month"] = df["Sales_Date"].dt.day
    df["Is_Weekend"] = df["Day_Of_Week"].isin([4, 5]).astype(int)

    # THE JORDAN SALARY IMPACT FEATURE (Tuned to start on the 22nd)
    df["Is_Salary_Period"] = (
        (df["Day_Of_Month"] >= 22) | (df["Day_Of_Month"] <= 2)
    ).astype(int)

    # PHASE B: LAG & ROLLING WINDOW FEATURES
    print("Calculating historical lag and rolling features...")
    df = df.sort_values(by=["Item_Name", "Sales_Date"]).reset_index(drop=True)
    df["Lag_1"] = df.groupby("Item_Name")["Daily_Qty"].shift(1)
    df["Lag_7"] = df.groupby("Item_Name")["Daily_Qty"].shift(7)
    df["Rolling_Mean_7"] = (
        df.groupby("Item_Name")["Daily_Qty"]
        .transform(lambda x: x.shift(1).rolling(window=7).mean())
    )

    df = df.dropna().reset_index(drop=True)
    return df


# =====================================================================
# MULTI-ENGINE TRAINING PIPELINE
# =====================================================================
if __name__ == "__main__":
    try:
        data = load_and_prepare_features(DATA_PATH)

        # AUTOMATED VOLUME ROUTING
        item_volumes = data.groupby("Item_Name", observed=False)["Daily_Qty"].mean()
        high_volume_items = item_volumes[
            item_volumes >= VOLUME_THRESHOLD
        ].index.tolist()

        print(f"\n🤖 Automated Inventory Category Routing:")
        print(f"   --> High-Volume Items detected: {len(high_volume_items)}")
        print(
            f"   --> Low-Volume Items detected:  {len(item_volumes) - len(high_volume_items)}"
        )

        # Chronological Split (14 Days Holdout)
        cutoff_date = data["Sales_Date"].max() - pd.Timedelta(days=14)
        train_set = data[data["Sales_Date"] <= cutoff_date]
        test_set = data[data["Sales_Date"] > cutoff_date]

        features = [
            "Day_Of_Week",
            "Month",
            "Is_Weekend",
            "Day_Of_Month",
            "Is_Salary_Period",
            "Lag_1",
            "Lag_7",
            "Rolling_Mean_7",
        ]
        target = "Daily_Qty"
        all_predictions = []

        # ROUTING LOOP AND CUSTOM MODEL TRAINING
        for category, item_list in [
            ("High-Volume", high_volume_items),
            (
                "Low-Volume",
                [i for i in item_volumes.index if i not in high_volume_items],
            ),
        ]:
            train_sub = train_set[train_set["Item_Name"].isin(item_list)]
            test_sub = test_set[test_set["Item_Name"].isin(item_list)]

            if len(train_sub) == 0 or len(test_sub) == 0:
                continue

            X_tr, y_tr = (
                train_sub[features + ["Item_Name"]].copy(),
                train_sub[target],
            )
            X_te, y_te = (
                test_sub[features + ["Item_Name"]].copy(),
                test_sub[target],
            )

            X_tr["Item_Name"] = X_tr["Item_Name"].astype("category")
            X_te["Item_Name"] = X_te["Item_Name"].astype("category")

            print(f"\n🚀 Training Dedicated Engine for {category} Stream...")
            model = lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.05,
                random_state=42,
                verbose=-1,
            )
            model.fit(X_tr, y_tr)

            sub_preds = model.predict(X_te)
            sub_preds = np.clip(sub_preds, a_min=0, a_max=None)

            sub_mae = mean_absolute_error(y_te, sub_preds)
            sub_rmse = np.sqrt(mean_squared_error(y_te, sub_preds))
            print(
                f"   📊 {category} Engine Performance -> MAE: {sub_mae:.2f} | RMSE: {sub_rmse:.2f}"
            )

            test_meta = test_sub[
                ["Sales_Date", "Item_Name", "Daily_Qty"]
            ].copy()
            test_meta["Model_Prediction"] = np.round(sub_preds, 1)
            test_meta["Forecast_Error"] = np.round(
                test_meta["Model_Prediction"] - test_meta["Daily_Qty"], 1
            )
            all_predictions.append(test_meta)

            # Save each model separately with its category name
            import joblib
            os.makedirs("models", exist_ok=True)
            model_filename = f"models/{category.replace('-','_').lower()}_model.pkl"
            joblib.dump(model, model_filename)
            print(f"   💾 Saved: {model_filename}")

        # =====================================================================
        # THIS BLOCK IS OUTSIDE THE LOOP — notice 8 spaces indent not 12
        # =====================================================================

        # Combine all predictions from both engines
        evaluation_df = pd.concat(all_predictions).sort_values(
            by=["Sales_Date", "Item_Name"]
        )

        overall_mae  = mean_absolute_error(evaluation_df["Daily_Qty"], evaluation_df["Model_Prediction"])
        overall_rmse = np.sqrt(mean_squared_error(evaluation_df["Daily_Qty"], evaluation_df["Model_Prediction"]))
        overall_r2   = r2_score(evaluation_df["Daily_Qty"], evaluation_df["Model_Prediction"])

        print(f"\n🎯 COMBINED MULTI-ENGINE SYSTEM METRICS:")
        print(f"👉 Overall MAE:  {overall_mae:.2f} units")
        print(f"👉 Overall RMSE: {overall_rmse:.2f} units")
        print(f"👉 Overall R²:   {overall_r2:.4f}")

        # Save full evaluation results
        output_path = r"C:\Users\obada\Desktop\Zanjabeel\forecast_results.csv"
        evaluation_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"\n💾 Forecast results saved to: {output_path}")

        # Show tomorrow's forecast
        tomorrow_date = evaluation_df["Sales_Date"].min()
        tomorrow_df = evaluation_df[
            evaluation_df["Sales_Date"] == tomorrow_date
        ].sort_values("Item_Name").reset_index(drop=True)

        print("\n" + "=" * 65)
        print(f"📋 FORECAST FOR: {tomorrow_date.strftime('%Y-%m-%d')}")
        print("=" * 65)
        for _, row in tomorrow_df.iterrows():
            recommended = int(np.ceil(row["Model_Prediction"] * 1.2))
            print(f"  {row['Item_Name']:<35} | {row['Model_Prediction']:.1f} → stock: {recommended}")
        print("=" * 65)

    except Exception as e:
        print(f"\n❌ Error:\n{e}")
        import traceback
        traceback.print_exc()