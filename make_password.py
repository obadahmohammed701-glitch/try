# make_password.py — Run this to generate a hashed password for a new customer
# Usage: python make_password.py

import bcrypt

print("=" * 50)
print("  مولّد كلمة المرور — زنجبيل")
print("=" * 50)

username  = input("\nاسم المستخدم (بالإنجليزي، بدون مسافات): ").strip()
shop_name = input("اسم المطعم (بالعربي): ").strip()
password  = input("كلمة المرور: ").strip()

hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

print("\n" + "=" * 50)
print("✅ انسخ هذا الكود وأضفه في app.py داخل credentials:")
print("=" * 50)
print(f'''
        "{username}": {{
            "name": "{shop_name}",
            "password": "{hashed}"
        }},
''')
print("=" * 50)