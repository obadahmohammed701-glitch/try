# make_password.py
# Run: python make_password.py

import streamlit_authenticator as stauth

# Put your passwords here
passwords = ["admin1234", "cafe2025", "shop2025"]

print("\n=== كلمات المرور المشفرة ===\n")

for pw in passwords:
    hashed = stauth.Hasher.hash(pw)
    print(f"الأصلية : {pw}")
    print(f"المشفرة : {hashed}")
    print()