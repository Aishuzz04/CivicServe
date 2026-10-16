from werkzeug.security import generate_password_hash
from database import get_connection


passwords = {
    1: "Citizen@123",
    2: "Admin@123",
    3: "Engineer@123",
    4: "Maintenance@123"
}


connection = get_connection()
cursor = connection.cursor()

for user_id, password in passwords.items():

    hashed_password = generate_password_hash(password)

    cursor.execute(
        """
        UPDATE users
        SET password_hash = %s
        WHERE user_id = %s;
        """,
        (hashed_password, user_id)
    )

connection.commit()

cursor.close()
connection.close()

print("Passwords updated successfully!")