from werkzeug.security import generate_password_hash, check_password_hash

password = "Test@123"

hashed_password = generate_password_hash(password)

print("Original password:", password)
print("Hashed password:", hashed_password)

print("Correct password:",
      check_password_hash(hashed_password, "Test@123"))

print("Wrong password:",
      check_password_hash(hashed_password, "WrongPassword"))