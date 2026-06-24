# auth.py
import bcrypt

# This is your "Phonebook" of allowed users and their hashed passwords
# Note: Use the 'b' prefix for bcrypt compatibility
USERS = {
    "admin": b"$2b$12$.bgkkBEGcggBbdfHaHjaxeI4mCjVcJknqA6267qe929/bWdZt0r0.",
    "cafe":  b"$2b$12$JADj/4A/WZX0iUA0Gubqzu6fA2cfk7ptZlcqaArFZoSO.diy0HuRO",
    "shop":  b"$2b$12$agOmCU3slyVIsJzd54UjiemwrlFfXggmQVTZ50yqskFHpJjZy/2K6"
}

def verify_user(username, password):
    if username in USERS:
        # bcrypt needs the password and the hash to both be bytes
        stored_hash = USERS[username]
        password_bytes = password.encode('utf-8')
        return bcrypt.checkpw(password_bytes, stored_hash)
    return False