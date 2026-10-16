from werkzeug.security import generate_password_hash

passwords = {
    "Citizen": "Citizen@123",
    "Administrator": "Admin@123",
    "Engineer": "Engineer@123",
    "Maintenance": "Maintenance@123"
}

for role, password in passwords.items():
    hashed = generate_password_hash(password)

    print(role)
    print("Password:", password)
    print("Hash:", hashed)
    print("-" * 60)