# run_all.py
# Entry point for the entire pipeline
# Run: python run_all.py

import auth
import sys
import subprocess

def main():
    print("=== Zanjabeel Forecast System ===")
    user = input("Username: ").strip()
    pw   = input("Password: ").strip()

    if not auth.verify_user(user, pw):
        print("❌ Access Denied.")
        sys.exit(1)

    print("✅ Access Granted. Starting pipeline...\n")

    # Step 1 — Prepare data
    print("─── Step 1: Preparing Data ───")
    result = subprocess.run(["python", "prepare_data.py"], check=False)
    if result.returncode != 0:
        print("❌ prepare_data.py failed. Check errors above.")
        sys.exit(1)

    # Step 2 — Train and predict
    print("\n─── Step 2: Training Model & Predicting ───")
    result = subprocess.run(["python", "train_model.py"], check=False)
    if result.returncode != 0:
        print("❌ train_model.py failed. Check errors above.")
        sys.exit(1)

    print("\n✅ Pipeline complete! Predictions saved to forecast_results.csv")

if __name__ == "__main__":
    main()