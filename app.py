from flask import Flask, session, request, render_template, redirect, url_for, send_from_directory, flash, jsonify
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename
from database import get_connection
from config import Config
import os
import datetime
import uuid
import psycopg2
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import math
import re
from image_validator import validate_complaint_image, validate_image_security, calculate_image_similarity

EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)*\.[a-zA-Z]{2,}$")

def is_valid_email(email: str) -> bool:
    if not email:
        return False
    return bool(EMAIL_REGEX.match(email.strip()))

def is_valid_phone(phone: str) -> bool:
    if not phone:
        return False
    cleaned = re.sub(r'[\s\-()]', '', phone)
    return bool(re.match(r'^\+?[0-9]{10,15}$', cleaned))

def calculate_text_similarity(text1, text2):
    """
    Calculate text similarity using TF-IDF and Cosine Similarity.

    Returns a similarity score between 0 and 100.
    """

    text1 = (text1 or "").strip().lower()
    text2 = (text2 or "").strip().lower()

    if not text1 or not text2:
        return 0

    # Create TF-IDF vectors for both texts
    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    try:
        tfidf_matrix = vectorizer.fit_transform([
            text1,
            text2
        ])

        # Calculate cosine similarity
        similarity = cosine_similarity(
            tfidf_matrix[0:1],
            tfidf_matrix[1:2]
        )[0][0]

        return round(similarity * 100, 2)

    except ValueError:
        # Happens when there are no meaningful words
        # after preprocessing
        return 0

def calculate_tfidf_similarity(text1, text2):
    """
    Calculate text similarity using TF-IDF and Cosine Similarity.

    Returns a value between 0 and 100.
    """

    text1 = (text1 or "").strip().lower()
    text2 = (text2 or "").strip().lower()

    if not text1 or not text2:
        return 0

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english"
        )

        tfidf_matrix = vectorizer.fit_transform([
            text1,
            text2
        ])

        similarity = cosine_similarity(
            tfidf_matrix[0:1],
            tfidf_matrix[1:2]
        )[0][0]

        return round(similarity * 100, 2)

    except ValueError:
        return 0

def calculate_distance_km(lat1, lon1, lat2, lon2):
    """
    Calculate approximate distance between two coordinates
    using the Haversine formula.
    """

    if None in (lat1, lon1, lat2, lon2):
        return None

    try:
        lat1 = float(lat1)
        lon1 = float(lon1)
        lat2 = float(lat2)
        lon2 = float(lon2)
    except (TypeError, ValueError):
        return None

    earth_radius = 6371.0

    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        +
        math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    c = 2 * math.atan2(
        math.sqrt(a),
        math.sqrt(1 - a)
    )

    return earth_radius * c

def calculate_location_similarity(distance_km):
    """
    Calculate location similarity based on geographical distance.
    Returns a value between 0 and 100.
    """

    if distance_km is None:
        return 0

    if distance_km <= 0.1:
        return 100

    elif distance_km <= 0.5:
        return 80

    elif distance_km <= 1.0:
        return 60

    return 0

def get_root_parent_complaint_id(cursor, complaint_id):
    """
    Given a complaint_id, traverse up the parent chain to find the canonical root parent complaint.
    The root parent is the original (first) complaint in the cluster.
    """
    if not complaint_id:
        return None
    try:
        current_id = int(complaint_id)
    except (ValueError, TypeError):
        return complaint_id

    visited = set()
    while current_id and current_id not in visited:
        visited.add(current_id)
        # 1. Check parent_complaint_id in complaints
        cursor.execute(
            "SELECT parent_complaint_id FROM complaints WHERE complaint_id = %s",
            (current_id,)
        )
        row = cursor.fetchone()
        if row and row[0] and row[0] != current_id:
            current_id = row[0]
            continue

        # 2. If parent_complaint_id is NULL, check if there is an accepted duplicate or similarity link
        # to an older (earlier created) complaint
        cursor.execute(
            """
            SELECT cs.similar_complaint_id 
            FROM complaint_similarity cs
            JOIN complaints c_sim ON cs.similar_complaint_id = c_sim.complaint_id
            JOIN complaints c_curr ON cs.complaint_id = c_curr.complaint_id
            WHERE cs.complaint_id = %s
              AND cs.decision != 'Separate'
              AND cs.decision != 'Rejected'
              AND (c_sim.created_at < c_curr.created_at OR c_sim.complaint_id < c_curr.complaint_id)
            ORDER BY 
              CASE WHEN cs.decision = 'Duplicate' THEN 1 ELSE 2 END,
              cs.combined_score DESC,
              c_sim.created_at ASC
            LIMIT 1
            """,
            (current_id,)
        )
        sim_row = cursor.fetchone()
        if sim_row and sim_row[0] and sim_row[0] != current_id:
            current_id = sim_row[0]
            continue

        break

    return current_id

def find_similar_complaints(
    cursor,
    complaint_id,
    category,
    title,
    description,
    location,
    latitude,
    longitude,
    image_path=None
):
    """
    Find previous complaints that may be duplicates using:
    1. Category match
    2. NLP TF-IDF text similarity
    3. Haversine location & coordinate distance
    4. Computer Vision CLIP photo similarity

    Resolves all candidate matches to the canonical root parent complaint.
    Detects photo reuse at different locations (photo/location mismatch).
    If all components are similar, flags as confirmed duplicate.
    Returns a list of top similar complaints.
    """

    # -----------------------------------
    # Get previous complaints (including image_path)
    # -----------------------------------
    cursor.execute(
        """
        SELECT
            complaint_id,
            title,
            description,
            category,
            location,
            latitude,
            longitude,
            status,
            created_at,
            image_path,
            parent_complaint_id
        FROM complaints
        WHERE complaint_id != %s
          AND verification_status != %s
        ORDER BY created_at DESC
        LIMIT 100
        """,
        (
            complaint_id,
            "Rejected"
        )
    )

    rows = cursor.fetchall()
    candidates_by_canonical = {}

    # -----------------------------------
    # Compare complaints
    # -----------------------------------
    for row in rows:
        previous_id = row[0]
        previous_title = row[1]
        previous_description = row[2]
        previous_category = row[3]
        previous_location = row[4]
        previous_latitude = row[5]
        previous_longitude = row[6]
        previous_status = row[7]
        previous_created_at = row[8]
        previous_image_path = row[9]
        previous_parent_id = row[10]

        # -----------------------------------
        # Category similarity
        # -----------------------------------
        category_score = 0
        if category and previous_category:
            if category.strip().lower() == previous_category.strip().lower():
                category_score = 100

        # -----------------------------------
        # NLP TEXT SIMILARITY (TF-IDF + Cosine)
        # -----------------------------------
        current_complaint_text = f"{title} {description}".strip()
        previous_complaint_text = f"{previous_title} {previous_description}".strip()
        text_score = calculate_tfidf_similarity(
            current_complaint_text,
            previous_complaint_text
        )

        # -----------------------------------
        # Location similarity (Text + Coordinates)
        # -----------------------------------
        location_score = calculate_text_similarity(
            location,
            previous_location
        )

        distance_km = calculate_distance_km(
            latitude,
            longitude,
            previous_latitude,
            previous_longitude
        )

        coordinate_score = calculate_location_similarity(
            distance_km
        )

        final_location_score = max(
            location_score,
            coordinate_score
        )

        # -----------------------------------
        # PHOTO SIMILARITY (CLIP Vision Embeddings)
        # -----------------------------------
        image_score = None
        if image_path and previous_image_path:
            try:
                image_score = calculate_image_similarity(image_path, previous_image_path)
            except Exception as e:
                print(f"Error comparing photo #{complaint_id} with #{previous_id}: {e}")
                image_score = None

        # -----------------------------------
        # Overall similarity score
        # -----------------------------------
        if image_score is not None:
            # 4-factor scoring: Category (15%), Text (35%), Location (25%), Photo (25%)
            overall_score = (
                category_score * 0.15
                + text_score * 0.35
                + final_location_score * 0.25
                + image_score * 0.25
            )
        else:
            # 3-factor fallback when photos are absent
            overall_score = (
                category_score * 0.20
                + text_score * 0.60
                + final_location_score * 0.20
            )

        overall_score = round(overall_score, 2)

        # -----------------------------------
        # REUSED PHOTO AT DIFFERENT LOCATION DETECTION
        # -----------------------------------
        is_same_photo = (image_score is not None and image_score >= 70)
        is_different_location = (
            (distance_km is not None and distance_km > 0.5) or
            (distance_km is None and final_location_score <= 30)
        )
        is_photo_location_mismatch = is_same_photo and is_different_location

        # -----------------------------------
        # "if everything is similar that complaint will be confirm duplicate"
        # -----------------------------------
        is_confirmed_duplicate = False
        if not is_photo_location_mismatch:
            # Condition 1: Same category, nearby/same location, similar text, and highly similar photo
            if (category_score >= 100 and final_location_score >= 60 and text_score >= 55
                    and image_score is not None and image_score >= 70 and overall_score >= 70):
                is_confirmed_duplicate = True
            # Condition 2: Same category, nearby location, and exact/identical photo match (>= 90%)
            elif (category_score >= 100 and final_location_score >= 50
                  and image_score is not None and image_score >= 90):
                is_confirmed_duplicate = True
            # Condition 3: When no photo on previous, but identical location, high text, and same category
            elif (category_score >= 100 and final_location_score >= 85 and text_score >= 80 and overall_score >= 80):
                is_confirmed_duplicate = True

        print(
            f"Comparing Complaint #{complaint_id} with #{previous_id}: "
            f"Category={category_score}%, Text={text_score}%, Location={final_location_score}%, "
            f"Photo={image_score if image_score is not None else 'N/A'}%, Overall={overall_score}%, "
            f"Confirmed Duplicate={is_confirmed_duplicate}, Photo Mismatch={is_photo_location_mismatch}"
        )

        # Keep candidates with overall score >= 50%, or if confirmed duplicate, or photo location mismatch
        if overall_score >= 50 or is_confirmed_duplicate or is_photo_location_mismatch:
            # Resolve to canonical root parent complaint (e.g. #62 rather than child #63)
            canonical_id = get_root_parent_complaint_id(cursor, previous_id) or previous_id

            # If canonical_id differs from previous_id, fetch root parent's metadata
            if canonical_id != previous_id:
                cursor.execute(
                    """
                    SELECT complaint_id, title, description, category, location, status, created_at, image_path
                    FROM complaints
                    WHERE complaint_id = %s
                    """,
                    (canonical_id,)
                )
                root_meta = cursor.fetchone()
                if root_meta:
                    cand_id = root_meta[0]
                    cand_title = root_meta[1]
                    cand_desc = root_meta[2]
                    cand_cat = root_meta[3]
                    cand_loc = root_meta[4]
                    cand_status = root_meta[5]
                    cand_created_at = root_meta[6]
                else:
                    cand_id = previous_id
                    cand_title = previous_title
                    cand_desc = previous_description
                    cand_cat = previous_category
                    cand_loc = previous_location
                    cand_status = previous_status
                    cand_created_at = previous_created_at
            else:
                cand_id = previous_id
                cand_title = previous_title
                cand_desc = previous_description
                cand_cat = previous_category
                cand_loc = previous_location
                cand_status = previous_status
                cand_created_at = previous_created_at

            # Merge or create entry for this canonical root parent
            if cand_id not in candidates_by_canonical:
                candidates_by_canonical[cand_id] = {
                    "complaint_id": cand_id,
                    "title": cand_title,
                    "description": cand_desc,
                    "category": cand_cat,
                    "location": cand_loc,
                    "status": cand_status,
                    "created_at": cand_created_at,
                    "category_similarity": category_score,
                    "text_similarity": text_score,
                    "location_similarity": final_location_score,
                    "image_similarity": image_score,
                    "similarity": overall_score,
                    "is_confirmed_duplicate": is_confirmed_duplicate,
                    "is_photo_location_mismatch": is_photo_location_mismatch,
                    "distance_km": round(distance_km, 2) if distance_km is not None else None
                }
            else:
                existing = candidates_by_canonical[cand_id]
                if overall_score > existing["similarity"]:
                    existing["similarity"] = overall_score
                    existing["category_similarity"] = category_score
                    existing["text_similarity"] = text_score
                    existing["location_similarity"] = final_location_score
                    existing["image_similarity"] = image_score
                    existing["distance_km"] = round(distance_km, 2) if distance_km is not None else None
                if is_confirmed_duplicate:
                    existing["is_confirmed_duplicate"] = True
                if is_photo_location_mismatch:
                    existing["is_photo_location_mismatch"] = True

    # -----------------------------------
    # Highest similarity / confirmed duplicates / mismatch alerts first
    # -----------------------------------
    similar_complaints = list(candidates_by_canonical.values())
    similar_complaints.sort(
        key=lambda x: (
            x.get("is_confirmed_duplicate", False) or x.get("is_photo_location_mismatch", False),
            x["similarity"]
        ),
        reverse=True
    )

    # Show top 3
    return similar_complaints[:3]

app = Flask(__name__)
app.secret_key = Config.SECRET_KEY

# Image upload settings
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)),"uploads")
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE

class NormalizeSlashMiddleware:
    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        if 'PATH_INFO' in environ:
            p = environ['PATH_INFO']
            p = p.replace('%5C', '/').replace('%5c', '/').replace('\\', '/')
            if '/uploads/' in p and not p.startswith('/uploads/'):
                p = '/uploads/' + p.split('/uploads/', 1)[1]
            environ['PATH_INFO'] = p
        return self.wsgi_app(environ, start_response)

app.wsgi_app = NormalizeSlashMiddleware(app.wsgi_app)

# -----------------------------------
# Check Allowed Image
# -----------------------------------

def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/test-db")
def test_database():

    try:
        connection = get_connection()
        cursor = connection.cursor()

        cursor.execute("SELECT 1;")
        result = cursor.fetchone()

        cursor.close()
        connection.close()

        if result[0] == 1:
            return "PostgreSQL connected successfully!"

    except Exception as e:
        return f"Database connection failed: {e}"


@app.route("/users")
def get_users():

    try:
        connection = get_connection()
        cursor = connection.cursor()

        cursor.execute("""
            SELECT
                user_id,
                first_name,
                last_name,
                email,
                role,
                is_active
            FROM users
            ORDER BY user_id;
        """)

        users = cursor.fetchall()

        cursor.close()
        connection.close()

        return str(users)

    except Exception as e:
        return f"Database error: {e}"

@app.route("/register", methods=["GET", "POST"])
def register():

    # -----------------------------------
    # Display Registration Page
    # -----------------------------------
    if request.method == "GET":
        return render_template("citizen_registration.html")


    # -----------------------------------
    # Get form data
    # -----------------------------------
    first_name = request.form.get("first_name", "").strip()
    last_name = request.form.get("last_name", "").strip()
    email = request.form.get("email", "").strip()
    phone = request.form.get("phone", "").strip()
    password = request.form.get("password", "")
    confirm_password = request.form.get("confirm_password", "")
    address = request.form.get("address", "").strip()


    # -----------------------------------
    # Basic validation
    # -----------------------------------
    if not first_name or not last_name or not email or not phone or not password or not address:

        return render_template(
            "citizen_registration.html",
            error="Please fill in all required fields.",
            first_name=first_name,
            last_name=last_name,
            email=email,
            phone=phone,
            address=address
        )

    # -----------------------------------
    # Email and Phone format validation
    # -----------------------------------
    if not is_valid_email(email):
        return render_template(
            "citizen_registration.html",
            error="Please enter a valid email address (e.g. name@example.com).",
            first_name=first_name,
            last_name=last_name,
            email=email,
            phone=phone,
            address=address
        )

    if not is_valid_phone(phone):
        return render_template(
            "citizen_registration.html",
            error="Please enter a valid 10-digit mobile number.",
            first_name=first_name,
            last_name=last_name,
            email=email,
            phone=phone,
            address=address
        )


    # -----------------------------------
    # Password match
    # -----------------------------------
    if password != confirm_password:

        return render_template(
            "citizen_registration.html",
            error="Passwords do not match.",
            first_name=first_name,
            last_name=last_name,
            email=email,
            phone=phone,
            address=address
        )


    # -----------------------------------
    # Password length
    # -----------------------------------
    if len(password) < 8:

        return render_template(
            "citizen_registration.html",
            error="Password must contain at least 8 characters.",
            first_name=first_name,
            last_name=last_name,
            email=email,
            phone=phone,
            address=address
        )


    try:

        # -----------------------------------
        # Database connection
        # -----------------------------------
        connection = get_connection()
        cursor = connection.cursor()


        # -----------------------------------
        # Check existing email
        # -----------------------------------
        cursor.execute(
            """
            SELECT user_id
            FROM users
            WHERE email = %s;
            """,
            (email,)
        )

        existing_user = cursor.fetchone()


        if existing_user:

            cursor.close()
            connection.close()

            return render_template(
                "citizen_registration.html",
                error="An account with this email already exists.",
                first_name=first_name,
                last_name=last_name,
                email=email,
                phone=phone,
                address=address
            )


        # -----------------------------------
        # Hash password
        # -----------------------------------
        password_hash = generate_password_hash(password)


        # -----------------------------------
        # Insert Citizen
        # -----------------------------------
        cursor.execute(
            """
            INSERT INTO users
            (
                first_name,
                last_name,
                email,
                phone,
                password_hash,
                role,
                department_id,
                address,
                created_at,
                is_active
            )
            VALUES
            (
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                CURRENT_TIMESTAMP,
                TRUE
            );
            """,
            (
                first_name,
                last_name,
                email,
                phone,
                password_hash,
                "Citizen",
                None,
                address
            )
        )


        # -----------------------------------
        # Save changes
        # -----------------------------------
        connection.commit()

        cursor.close()
        connection.close()


        # -----------------------------------
        # Registration successful
        # -----------------------------------
        return render_template(
            "citizen_registration.html",
            success="Registration successful! You can now login to your account."
        )


    except Exception as e:

        print("Registration Error:", e)

        return render_template(
            "citizen_registration.html",
            error="Something went wrong while creating your account. Please try again."
        )

@app.route("/login", methods=["GET", "POST"])
def login():

    # Display login page
    if request.method == "GET":
        return render_template("citizen_login.html")

    # Get form data
    email = request.form.get("email", "").strip()
    password = request.form.get("password", "")

    # Basic validation
    if not email or not password:
        return render_template(
            "citizen_login.html",
            error="Please enter your email and password.",
            email=email
        )

    try:
        connection = get_connection()
        cursor = connection.cursor()

        # Find user
        cursor.execute(
            """
            SELECT
                user_id,
                first_name,
                last_name,
                email,
                password_hash,
                role,
                department_id,
                is_active,
                must_change_password
            FROM users
            WHERE email = %s;
            """,
            (email,)
        )

        user = cursor.fetchone()

        cursor.close()
        connection.close()

        # User not found
        if user is None:
            return render_template(
                "citizen_login.html",
                error="Invalid email or password.",
                email=email
            )

        (
            user_id,
            first_name,
            last_name,
            email,
            password_hash,
            role,
            department_id,
            is_active,
            must_change_password
        ) = user

        # Account inactive
        if not is_active:
            return render_template(
                "citizen_login.html",
                error="Your account is inactive. Please contact the administrator.",
                email=email
            )

        # Password incorrect
        if not check_password_hash(password_hash, password):
            return render_template(
                "citizen_login.html",
                error="Invalid email or password.",
                email=email
            )

        # Create session
        session["user_id"] = user_id
        session["first_name"] = first_name
        session["last_name"] = last_name
        session["role"] = role
        session["department_id"] = department_id
        session["must_change_password"] = bool(must_change_password)

        # First-login password change
        if must_change_password:
            return redirect(url_for("change_password"))

        # Role validation
        valid_roles = ["Citizen", "Administrator", "Department Head", "Engineer", "Maintenance"]
        if role not in valid_roles:
            return render_template(
                "citizen_login.html",
                error="Invalid user role.",
                email=email
            )

        dashboard_url = get_dashboard_url_for_role(role)
        return redirect(dashboard_url)

    except Exception as e:

        print("Login Error:", e)

        return render_template(
            "citizen_login.html",
            error="Something went wrong. Please try again.",
            email=email
        )

# Helper for role-based dashboard URL
def get_dashboard_url_for_role(role):
    if role == "Citizen":
        return url_for("citizen_dashboard")
    elif role == "Administrator":
        return url_for("admin_dashboard")
    elif role == "Department Head":
        return url_for("department_head_dashboard")
    elif role == "Engineer":
        return url_for("engineer_dashboard")
    elif role == "Maintenance":
        return url_for("maintenance_dashboard")
    return url_for("login")

# ==========================================
# FIRST-LOGIN PASSWORD CHANGE ENFORCEMENT
# ==========================================
@app.before_request
def enforce_first_login_password_change():
    if "user_id" in session and session.get("must_change_password"):
        allowed_endpoints = {"change_password", "logout", "static"}
        if request.endpoint and request.endpoint not in allowed_endpoints:
            return redirect(url_for("change_password"))


@app.route("/change-password", methods=["GET", "POST"])
def change_password():
    if "user_id" not in session:
        return redirect(url_for("login"))

    user_id = session.get("user_id")

    # If the user does not need to change password, send to their dashboard
    if not session.get("must_change_password"):
        role = session.get("role")
        return redirect(get_dashboard_url_for_role(role))

    if request.method == "GET":
        return render_template("change_password.html")

    current_password = request.form.get("current_password", "").strip()
    new_password = request.form.get("new_password", "").strip()
    confirm_password = request.form.get("confirm_password", "").strip()

    if not current_password or not new_password or not confirm_password:
        return render_template(
            "change_password.html",
            error="Please fill in all password fields."
        )

    if len(new_password) < 8:
        return render_template(
            "change_password.html",
            error="New password must be at least 8 characters long."
        )

    if new_password != confirm_password:
        return render_template(
            "change_password.html",
            error="New password and confirm password do not match."
        )

    if new_password == current_password:
        return render_template(
            "change_password.html",
            error="Your new password must be different from your temporary password."
        )

    connection = None
    cursor = None
    try:
        connection = get_connection()
        cursor = connection.cursor()

        cursor.execute("SELECT password_hash FROM users WHERE user_id = %s;", (user_id,))
        row = cursor.fetchone()

        if not row or not check_password_hash(row[0], current_password):
            return render_template(
                "change_password.html",
                error="The current temporary password you entered is incorrect."
            )

        new_hash = generate_password_hash(new_password)
        cursor.execute("""
            UPDATE users
            SET password_hash = %s,
                must_change_password = FALSE
            WHERE user_id = %s;
        """, (new_hash, user_id))
        connection.commit()

        # Update session flags
        session["must_change_password"] = False

        flash("Your password has been changed successfully! Welcome to your dashboard.", "success")
        dashboard_url = get_dashboard_url_for_role(role)
        return redirect(dashboard_url)

    except Exception as e:
        print("Change Password Error:", e)
        if connection:
            connection.rollback()
        return render_template(
            "change_password.html",
            error="An error occurred while updating your password. Please try again."
        )
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

# ==========================================
# DYNAMIC WELCOME SPLASH SCREEN
# ==========================================
@app.route("/welcome")
def welcome():
    if "user_id" not in session:
        return redirect(url_for("login"))

    role = session.get("role")
    dashboard_url = get_dashboard_url_for_role(role)
    return redirect(dashboard_url)

    first_name = session.get("first_name", "User")

    ROLE_METADATA = {
        "Citizen": {
            "badge_text": "Citizen Civic Portal",
            "portal_title": "Public Services & Civic Engagement",
            "subtitle": "Track neighborhood issues, register civic complaints, and monitor municipal resolutions in real-time.",
            "icon": "fa-users",
            "badge_classes": "bg-blue-500/15 text-blue-300 border border-blue-500/30",
            "accent_gradient": "from-blue-400 via-sky-300 to-cyan-300",
            "glow_color": "rgba(59, 130, 246, 0.35)",
            "progress_bar": "bg-gradient-to-r from-blue-500 via-sky-400 to-cyan-400",
            "highlights": [
                {"icon": "fa-bullhorn", "title": "Smart Complaint Filing", "desc": "Custom department forms & photo upload"},
                {"icon": "fa-location-crosshairs", "title": "GPS Geo-Tagging", "desc": "Precise one-click location mapping"},
                {"icon": "fa-clock-rotate-left", "title": "Live Status Tracking", "desc": "Real-time updates on repair progress"}
            ]
        },
        "Administrator": {
            "badge_text": "Central Administration Center",
            "portal_title": "Municipal Governance & Oversight",
            "subtitle": "Managing departmental officers, automated duplicate detection, complaint verification, and municipal operations.",
            "icon": "fa-shield-halved",
            "badge_classes": "bg-purple-500/15 text-purple-300 border border-purple-500/30",
            "accent_gradient": "from-purple-400 via-violet-300 to-indigo-300",
            "glow_color": "rgba(168, 85, 247, 0.35)",
            "progress_bar": "bg-gradient-to-r from-purple-500 via-violet-400 to-indigo-400",
            "highlights": [
                {"icon": "fa-copy", "title": "Smart Duplicate Review", "desc": "TF-IDF & Haversine similarity matching"},
                {"icon": "fa-building-shield", "title": "Department Verification", "desc": "Dispatch verified issues to departments"},
                {"icon": "fa-user-shield", "title": "Account Governance", "desc": "3-strike spam enforcement & metrics"}
            ]
        },
        "Department Head": {
            "badge_text": "Department Operations Center",
            "portal_title": "Department Operations & Workflow",
            "subtitle": "Overseeing municipal infrastructure, dispatching engineers for inspection, and approving maintenance resolutions.",
            "icon": "fa-building-columns",
            "badge_classes": "bg-teal-500/15 text-teal-300 border border-teal-500/30",
            "accent_gradient": "from-teal-400 via-emerald-300 to-cyan-300",
            "glow_color": "rgba(20, 184, 166, 0.35)",
            "progress_bar": "bg-gradient-to-r from-teal-500 via-emerald-400 to-cyan-400",
            "highlights": [
                {"icon": "fa-user-gear", "title": "Work Order Allocation", "desc": "Assign field engineers & repair teams"},
                {"icon": "fa-clipboard-check", "title": "Inspection Verification", "desc": "Review diagnostic findings & cost estimates"},
                {"icon": "fa-check-double", "title": "Completion Verification", "desc": "Verify completed maintenance and close tickets"}
            ]
        },
        "Engineer": {
            "badge_text": "Field Engineering Console",
            "portal_title": "Technical Inspection & Assessment",
            "subtitle": "Conducting on-site diagnostics, evaluating structural infrastructure defects, and filing technical reports.",
            "icon": "fa-helmet-safety",
            "badge_classes": "bg-emerald-500/15 text-emerald-300 border border-emerald-500/30",
            "accent_gradient": "from-emerald-400 via-teal-300 to-green-300",
            "glow_color": "rgba(16, 185, 129, 0.35)",
            "progress_bar": "bg-gradient-to-r from-emerald-500 via-teal-400 to-green-400",
            "highlights": [
                {"icon": "fa-magnifying-glass-location", "title": "Site Inspection", "desc": "Evaluate damage severity on assigned sites"},
                {"icon": "fa-file-invoice-dollar", "title": "Cost & Materials", "desc": "Itemize required repair parts & budget"},
                {"icon": "fa-camera", "title": "Diagnostic Evidence", "desc": "Upload on-site inspection photos"}
            ]
        },
        "Maintenance": {
            "badge_text": "Field Operations Console",
            "portal_title": "Repair & Physical Work Execution",
            "subtitle": "Executing infrastructure repairs, updating active task milestones, and submitting photo proof of completion.",
            "icon": "fa-screwdriver-wrench",
            "badge_classes": "bg-amber-500/15 text-amber-300 border border-amber-500/30",
            "accent_gradient": "from-amber-400 via-orange-300 to-yellow-300",
            "glow_color": "rgba(245, 158, 11, 0.35)",
            "progress_bar": "bg-gradient-to-r from-amber-500 via-orange-400 to-yellow-400",
            "highlights": [
                {"icon": "fa-tools", "title": "Work Order Execution", "desc": "Active maintenance & physical repair"},
                {"icon": "fa-camera-retro", "title": "Work Proof Upload", "desc": "Upload after-repair evidence photos"},
                {"icon": "fa-circle-check", "title": "Status Reporting", "desc": "Instant progress updates to department"}
            ]
        }
    }

    meta = ROLE_METADATA.get(role, {
        "badge_text": f"{role} Workspace",
        "portal_title": "CivicServe Workspace",
        "subtitle": "Connecting to municipal services network...",
        "icon": "fa-city",
        "badge_classes": "bg-blue-500/15 text-blue-300 border border-blue-500/30",
        "accent_gradient": "from-blue-400 to-indigo-300",
        "glow_color": "rgba(59, 130, 246, 0.35)",
        "progress_bar": "bg-gradient-to-r from-blue-500 to-indigo-500",
        "highlights": [
            {"icon": "fa-shield-halved", "title": "Secure Session", "desc": "Encrypted role-authenticated access"},
            {"icon": "fa-database", "title": "Infrastructure Core", "desc": "Unified public service data"},
            {"icon": "fa-bolt", "title": "Real-time Sync", "desc": "Live event and status synchronization"}
        ]
    })

    return render_template(
        "welcome.html",
        first_name=first_name,
        role=role,
        dashboard_url=dashboard_url,
        meta=meta
    )

#Citizen
@app.route("/citizen-dashboard")
def citizen_dashboard():
    if "user_id" not in session:
        return redirect(url_for("login"))

    if session.get("role") != "Citizen":
        return "Access denied."

    connection = None
    cursor = None

    try:
        connection = get_connection()
        cursor = connection.cursor()

        # =========================
        # Recent Complaints
        # =========================
        cursor.execute(
            """
            SELECT
                c.complaint_id,
                c.category,
                c.location,
                c.status,
                c.created_at
            FROM complaints c
            WHERE c.user_id = %s
            ORDER BY c.created_at DESC
            LIMIT 5
            """,
            (session["user_id"],)
        )

        complaint_rows = cursor.fetchall()

        recent_complaints = []

        for row in complaint_rows:
            recent_complaints.append({
                "complaint_id": row[0],
                "category": row[1],
                "location": row[2],
                "status": row[3],
                "created_at": row[4]
            })

        # =========================
        # Calendar Complaints
        # =========================
        cursor.execute(
            """
            SELECT
                c.complaint_id,
                c.category,
                c.location,
                c.status,
                c.created_at
            FROM complaints c
            WHERE c.user_id = %s
            ORDER BY c.created_at DESC
            """,
            (session["user_id"],)
        )

        calendar_rows = cursor.fetchall()

        calendar_complaints = []

        for row in calendar_rows:
            calendar_complaints.append({
                "complaint_id": row[0],
                "category": row[1],
                "location": row[2],
                "status": row[3],
                "created_at": row[4].strftime("%Y-%m-%d") if row[4] else None
            })

        # =========================
        # Complaint Summary
        # =========================
        cursor.execute(
            """
            SELECT
                COUNT(*) AS total_complaints,

                COUNT(*) FILTER (
                    WHERE LOWER(status) = 'pending'
                ) AS pending,

                COUNT(*) FILTER (
                    WHERE LOWER(status) = 'under verification'
                ) AS under_verification,

                COUNT(*) FILTER (
                    WHERE LOWER(status) IN
                    ('verified', 'assigned', 'inspection', 'maintenance')
                ) AS in_progress,

                COUNT(*) FILTER (
                    WHERE LOWER(status) = 'completed'
                ) AS completed,

                COUNT(*) FILTER (
                    WHERE LOWER(status) = 'closed'
                ) AS closed,

                COUNT(*) FILTER (
                    WHERE LOWER(status) = 'rejected'
                ) AS rejected

            FROM complaints
            WHERE user_id = %s
            """,
            (session["user_id"],)
        )

        summary_row = cursor.fetchone()

        complaint_summary = {
            "total": summary_row[0],
            "pending": summary_row[1],
            "under_verification": summary_row[2],
            "in_progress": summary_row[3],
            "completed": summary_row[4],
            "closed": summary_row[5],
            "rejected": summary_row[6]
        }

        cursor.close()
        connection.close()

        return render_template(
            "citizen_dashboard.html",
            first_name=session.get("first_name"),
            recent_complaints=recent_complaints,
            calendar_complaints=calendar_complaints,
            complaint_summary=complaint_summary
        )

    except Exception as e:
        print("Citizen Dashboard Error:", repr(e))

        if cursor:
            cursor.close()

        if connection:
            connection.close()

        return render_template(
            "citizen_dashboard.html",
            first_name=session.get("first_name"),
            recent_complaints=[],
            calendar_complaints=[],
            complaint_summary={
                "total": 0,
                "pending": 0,
                "under_verification": 0,
                "in_progress": 0,
                "completed": 0,
                "closed": 0,
                "rejected": 0
            }
        )

#Administrator
@app.route("/admin-dashboard")
def admin_dashboard():

    # ==========================================
    # CHECK LOGIN
    # ==========================================

    if "user_id" not in session:
        return redirect(url_for("login"))


    # ==========================================
    # ONLY ADMINISTRATOR
    # ==========================================

    if session.get("role") != "Administrator":
        return "Access denied."


    connection = None
    cursor = None

    try:

        connection = get_connection()
        cursor = connection.cursor()


        # ==========================================
        # 1. TOTAL COMPLAINTS
        # ==========================================

        cursor.execute("""
            SELECT COUNT(*)
            FROM complaints;
        """)

        total_complaints = cursor.fetchone()[0]


        # ==========================================
        # 2. PENDING COMPLAINTS
        # ==========================================

        cursor.execute("""
            SELECT COUNT(*)
            FROM complaints
            WHERE status IN (
                'Pending',
                'Under Verification'
            );
        """)

        pending_complaints = cursor.fetchone()[0]


        # ==========================================
        # 3. ASSIGNED COMPLAINTS
        # ==========================================

        cursor.execute("""
            SELECT COUNT(*)
            FROM complaints
            WHERE status IN (
                'Assigned',
                'Inspection',
                'Maintenance'
            );
        """)

        assigned_complaints = cursor.fetchone()[0]


        # ==========================================
        # 4. RESOLVED COMPLAINTS
        # ==========================================

        cursor.execute("""
            SELECT COUNT(*)
            FROM complaints
            WHERE status IN (
                'Completed',
                'Closed'
            );
        """)

        resolved_complaints = cursor.fetchone()[0]

        # Complaints by Status
        cursor.execute("""
            SELECT status, COUNT(*)
            FROM complaints
            GROUP BY status
            ORDER BY COUNT(*) DESC;
        """)

        status_rows = cursor.fetchall()

        status_labels = [row[0] if row[0] else "Unknown" for row in status_rows]
        status_data = [row[1] for row in status_rows]

        # Complaints by Category (Normalized)
        cursor.execute("""
            SELECT 
                CASE 
                    WHEN LOWER(category) LIKE '%road%' THEN 'Roads'
                    WHEN LOWER(category) LIKE '%light%' THEN 'Street Lights'
                    WHEN LOWER(category) LIKE '%water%' THEN 'Water Supply'
                    WHEN LOWER(category) LIKE '%drain%' THEN 'Drainage'
                    WHEN LOWER(category) LIKE '%waste%' OR LOWER(category) LIKE '%garb%' THEN 'Waste Management'
                    WHEN LOWER(category) LIKE '%park%' THEN 'Parks'
                    ELSE INITCAP(COALESCE(category, 'General Infrastructure'))
                END AS norm_cat,
                COUNT(*)
            FROM complaints
            GROUP BY norm_cat
            ORDER BY COUNT(*) DESC;
        """)

        category_rows = cursor.fetchall()

        category_labels = [row[0] if row[0] else "Unknown" for row in category_rows]
        category_data = [row[1] for row in category_rows]

        # Additional Executive System Metrics
        cursor.execute("SELECT COUNT(*) FROM users;")
        total_users = cursor.fetchone()[0] or 0

        cursor.execute("SELECT COUNT(*) FROM departments WHERE is_active = TRUE;")
        total_departments = cursor.fetchone()[0] or 0

        cursor.execute("""
            SELECT COUNT(*) 
            FROM complaints 
            WHERE status = 'Under Verification' OR is_duplicate = TRUE;
        """)
        pending_verifications = cursor.fetchone()[0] or 0

        # System resolution rate
        resolution_rate = round((resolved_complaints / total_complaints * 100), 1) if total_complaints > 0 else 0

        # Department Operations Overview
        cursor.execute("""
            SELECT 
                d.department_id,
                d.department_name,
                d.description,
                COUNT(c.complaint_id) AS total_dept_complaints,
                COUNT(c.complaint_id) FILTER (WHERE c.status NOT IN ('Completed', 'Closed', 'Rejected')) AS active_complaints,
                COUNT(c.complaint_id) FILTER (WHERE c.status IN ('Completed', 'Closed')) AS resolved_complaints,
                u.first_name,
                u.last_name
            FROM departments d
            LEFT JOIN complaints c ON d.department_id = c.department_id
            LEFT JOIN users u ON d.department_id = u.department_id AND u.role = 'Department Head'
            GROUP BY d.department_id, d.department_name, d.description, u.first_name, u.last_name
            ORDER BY total_dept_complaints DESC, d.department_name ASC;
        """)
        dept_rows = cursor.fetchall()
        department_overview = []
        for d in dept_rows:
            head_name = f"{d[6]} {d[7]}" if d[6] else None
            department_overview.append({
                'id': d[0],
                'name': d[1],
                'description': d[2],
                'total': d[3],
                'active': d[4],
                'resolved': d[5],
                'head_name': head_name
            })

        # ==========================================
        # 7. RECENT COMPLAINTS
        # ==========================================

        cursor.execute("""
            SELECT
                c.complaint_id,
                c.title,
                c.category,
                c.location,
                c.status,
                c.priority,
                c.created_at,
                u.first_name,
                u.last_name,
                d.department_name
            FROM complaints c

            JOIN users u
                ON c.user_id = u.user_id

            LEFT JOIN departments d
                ON c.department_id = d.department_id

            ORDER BY c.created_at DESC

            LIMIT 6;
        """)

        recent_complaints = cursor.fetchall()


        # ==========================================
        # CLOSE DATABASE
        # ==========================================

        cursor.close()
        connection.close()


        # ==========================================
        # SEND DATA TO HTML
        # ==========================================

        return render_template(
            "admin_dashboard.html",
            total_complaints=total_complaints,
            pending_complaints=pending_complaints,
            assigned_complaints=assigned_complaints,
            resolved_complaints=resolved_complaints,
            status_labels=status_labels,
            status_data=status_data,
            category_labels=category_labels,
            category_data=category_data,
            recent_complaints=recent_complaints,
            total_users=total_users,
            total_departments=total_departments,
            pending_verifications=pending_verifications,
            resolution_rate=resolution_rate,
            department_overview=department_overview
        )


    except Exception as e:

        print("Admin Dashboard Error:", e)

        if cursor:
            cursor.close()

        if connection:
            connection.close()

        return "Unable to load administrator dashboard."

# Department Head
@app.route("/department-head-dashboard")
@app.route("/department-head/dashboard")
def department_head_dashboard():

    # ==========================================
    # CHECK LOGIN
    # ==========================================

    if "user_id" not in session:
        return redirect(url_for("login"))

    # ==========================================
    # ONLY DEPARTMENT HEAD
    # ==========================================

    if session.get("role") != "Department Head":
        return "Access denied."

    department_id = session.get("department_id")

    if not department_id:
        return "No department assigned to this account."

    connection = None
    cursor = None

    try:

        connection = get_connection()
        cursor = connection.cursor()

        # ==========================================
        # DEPARTMENT NAME
        # ==========================================

        cursor.execute(
            """
            SELECT department_name
            FROM departments
            WHERE department_id = %s
            """,
            (department_id,)
        )

        department_row = cursor.fetchone()

        if department_row is None:
            return "Department not found."

        department_name = department_row[0]

        # ==========================================
        # TOTAL VERIFIED COMPLAINTS
        # ==========================================

        cursor.execute(
            """
            SELECT COUNT(*)
            FROM complaints
            WHERE department_id = %s
              AND verification_status = 'Verified'
              AND (is_duplicate IS FALSE OR is_duplicate IS NULL)
              AND status != 'Duplicate'
            """,
            (department_id,)
        )

        total = cursor.fetchone()[0]

        # ==========================================
        # PENDING VERIFIED COMPLAINTS
        # ==========================================

        cursor.execute(
            """
            SELECT COUNT(*)
            FROM complaints
            WHERE department_id = %s
              AND verification_status = 'Verified'
              AND status = 'Pending'
              AND (is_duplicate IS FALSE OR is_duplicate IS NULL)
              AND status != 'Duplicate'
            """,
            (department_id,)
        )

        pending = cursor.fetchone()[0]

        # ==========================================
        # ASSIGNED COMPLAINTS
        # ==========================================

        cursor.execute(
            """
            SELECT COUNT(*)
            FROM complaints
            WHERE department_id = %s
              AND verification_status = 'Verified'
              AND status = 'Assigned'
              AND (is_duplicate IS FALSE OR is_duplicate IS NULL)
              AND status != 'Duplicate'
            """,
            (department_id,)
        )

        assigned = cursor.fetchone()[0]

        # ==========================================
        # INSPECTION COMPLAINTS
        # ==========================================

        cursor.execute(
            """
            SELECT COUNT(*)
            FROM complaints
            WHERE department_id = %s
              AND verification_status = 'Verified'
              AND status = 'Inspection'
              AND (is_duplicate IS FALSE OR is_duplicate IS NULL)
              AND status != 'Duplicate'
            """,
            (department_id,)
        )

        inspection = cursor.fetchone()[0]

        # ==========================================
        # MAINTENANCE COMPLAINTS
        # ==========================================

        cursor.execute(
            """
            SELECT COUNT(*)
            FROM complaints
            WHERE department_id = %s
              AND verification_status = 'Verified'
              AND status = 'Maintenance'
              AND (is_duplicate IS FALSE OR is_duplicate IS NULL)
              AND status != 'Duplicate'
            """,
            (department_id,)
        )

        maintenance = cursor.fetchone()[0]

        # ==========================================
        # IN-PROGRESS COMPLAINTS
        #
        # Includes all complaints currently being
        # worked on.
        # ==========================================

        cursor.execute(
            """
            SELECT COUNT(*)
            FROM complaints
            WHERE department_id = %s
              AND verification_status = 'Verified'
              AND status IN (
                  'Assigned',
                  'Inspection',
                  'Maintenance',
                  'In Progress'
              )
              AND (is_duplicate IS FALSE OR is_duplicate IS NULL)
              AND status != 'Duplicate'
            """,
            (department_id,)
        )

        in_progress = cursor.fetchone()[0]

        # ==========================================
        # RESOLVED COMPLAINTS
        # ==========================================

        cursor.execute(
            """
            SELECT COUNT(*)
            FROM complaints
            WHERE department_id = %s
              AND verification_status = 'Verified'
              AND status IN (
                  'Completed',
                  'Closed'
              )
              AND (is_duplicate IS FALSE OR is_duplicate IS NULL)
              AND status != 'Duplicate'
            """,
            (department_id,)
        )

        resolved = cursor.fetchone()[0]

        # ==========================================
        # DUPLICATE / LINKED REPORTS
        # ==========================================

        cursor.execute(
            """
            SELECT COUNT(*)
            FROM complaints
            WHERE department_id = %s
              AND (is_duplicate IS TRUE OR status = 'Duplicate')
            """,
            (department_id,)
        )

        duplicate_count = cursor.fetchone()[0]

        # ==========================================
        # RECENT VERIFIED COMPLAINTS
        #
        # IMPORTANT:
        # Only Admin Verified complaints appear here.
        #
        # Column order matches the HTML:
        #
        # c[0] = complaint_id
        # c[1] = title
        # c[2] = category
        # c[3] = location
        # c[4] = priority
        # c[5] = status
        # c[6] = created_by
        # c[7] = created_at
        # ==========================================

        cursor.execute(
            """
            SELECT
                c.complaint_id,
                c.title,
                c.category,
                c.location,
                c.priority,
                c.status,
                u.first_name || ' ' || u.last_name AS created_by,
                c.created_at,
                eng.first_name || ' ' || COALESCE(eng.last_name, '') AS engineer_name,
                maint.first_name || ' ' || COALESCE(maint.last_name, '') AS maintenance_name,
                c.assigned_engineer_id,
                c.assigned_maintenance_id,
                c.is_duplicate,
                c.parent_complaint_id
            FROM complaints c
            JOIN users u ON c.user_id = u.user_id
            LEFT JOIN users eng ON c.assigned_engineer_id = eng.user_id
            LEFT JOIN users maint ON c.assigned_maintenance_id = maint.user_id
            WHERE c.department_id = %s
              AND c.verification_status = 'Verified'
            ORDER BY c.created_at DESC
            LIMIT 10
            """,
            (department_id,)
        )

        recent_complaints_rows = cursor.fetchall()
        recent_complaints = []
        for rc in recent_complaints_rows:
            recent_complaints.append({
                "complaint_id": rc[0],
                "title": rc[1],
                "category": rc[2],
                "location": rc[3],
                "priority": rc[4],
                "status": rc[5],
                "created_by": rc[6],
                "created_at": rc[7],
                "engineer_name": rc[8].strip() if rc[8] else None,
                "maintenance_name": rc[9].strip() if rc[9] else None,
                "assigned_engineer_id": rc[10],
                "assigned_maintenance_id": rc[11],
                "is_duplicate": rc[12] or False,
                "parent_complaint_id": rc[13]
            })

        # ==========================================
        # PENDING INSPECTION REPORTS (FOR VERIFICATION)
        # ==========================================
        cursor.execute(
            """
            SELECT
                r.report_id,
                r.complaint_id,
                c.title,
                c.category,
                c.location,
                c.priority,
                r.inspection_date,
                r.severity,
                r.findings,
                r.recommendation,
                r.required_materials,
                r.estimated_cost,
                r.inspection_image,
                r.status,
                r.created_at,
                eng.first_name || ' ' || COALESCE(eng.last_name, '') AS engineer_name
            FROM inspection_reports r
            JOIN complaints c ON r.complaint_id = c.complaint_id
            JOIN users eng ON r.engineer_id = eng.user_id
            WHERE c.department_id = %s
              AND r.status = 'Pending Verification'
            ORDER BY r.created_at DESC
            """,
            (department_id,)
        )
        pending_insp_rows = cursor.fetchall()
        pending_inspections = []
        for pi in pending_insp_rows:
            pending_inspections.append({
                "report_id": pi[0],
                "complaint_id": pi[1],
                "title": pi[2],
                "category": pi[3],
                "location": pi[4],
                "priority": pi[5],
                "inspection_date": pi[6],
                "severity": pi[7],
                "findings": pi[8],
                "recommendation": pi[9],
                "required_materials": pi[10],
                "estimated_cost": pi[11],
                "inspection_image": pi[12],
                "status": pi[13],
                "created_at": pi[14],
                "engineer_name": pi[15].strip()
            })
        pending_inspections_count = len(pending_inspections)

        # ==========================================
        # PENDING MAINTENANCE WORK VERIFICATIONS
        # ==========================================
        cursor.execute(
            """
            SELECT
                mw.maintenance_id,
                mw.complaint_id,
                c.title,
                c.category,
                c.location,
                c.priority,
                c.status,
                mw.work_status,
                mw.work_description,
                mw.remarks,
                mw.completion_report,
                mw.before_image,
                mw.after_image,
                mw.completion_date,
                mw.created_at,
                maint.first_name || ' ' || COALESCE(maint.last_name, '') AS maintenance_name,
                c.image_path AS complaint_image
            FROM maintenance_work mw
            JOIN complaints c ON mw.complaint_id = c.complaint_id
            JOIN users maint ON mw.maintenance_staff_id = maint.user_id
            WHERE c.department_id = %s
              AND mw.work_status = 'Completed'
              AND (mw.verification_status = 'Pending' OR mw.verification_status IS NULL)
            ORDER BY mw.created_at DESC
            """,
            (department_id,)
        )
        pending_maint_rows = cursor.fetchall()
        pending_maintenance = []
        for pm in pending_maint_rows:
            pending_maintenance.append({
                "maintenance_id": pm[0],
                "complaint_id": pm[1],
                "title": pm[2],
                "category": pm[3],
                "location": pm[4],
                "priority": pm[5],
                "complaint_status": pm[6],
                "work_status": pm[7],
                "work_description": pm[8],
                "remarks": pm[9],
                "completion_report": pm[10],
                "before_image": pm[11],
                "after_image": pm[12],
                "completion_date": pm[13],
                "created_at": pm[14],
                "maintenance_name": pm[15].strip(),
                "complaint_image": pm[16]
            })
        pending_maintenance_count = len(pending_maintenance)

        # ==========================================
        # CLOSE DATABASE
        # ==========================================

        cursor.close()
        connection.close()

        # ==========================================
        # SEND DATA TO TEMPLATE
        # ==========================================

        return render_template(
            "department_head_dashboard.html",

            first_name=session.get("first_name"),
            last_name=session.get("last_name"),

            department_name=department_name,

            total=total,
            pending=pending,
            assigned=assigned,
            inspection=inspection,
            maintenance=maintenance,
            in_progress=in_progress,
            resolved=resolved,
            duplicate_count=duplicate_count,

            recent_complaints=recent_complaints,
            pending_inspections=pending_inspections,
            pending_inspections_count=pending_inspections_count,
            pending_maintenance=pending_maintenance,
            pending_maintenance_count=pending_maintenance_count
        )

    except Exception as e:

        print("Department Head Dashboard Error:", repr(e))

        if cursor:
            cursor.close()

        if connection:
            connection.close()

        return "Unable to load department head dashboard."

@app.route("/department-head-complaints")
def department_head_complaints():

    if "user_id" not in session:
        return redirect(url_for("login"))

    if session.get("role") != "Department Head":
        return "Access denied."

    department_id = session.get("department_id")

    if not department_id:
        return "No department assigned to this account."

    connection = None
    cursor = None

    try:
        connection = get_connection()
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                c.complaint_id,
                c.title,
                c.location,
                c.priority,
                c.status,
                c.verification_status,
                u.first_name,
                u.last_name,
                c.created_at,
                c.assigned_engineer_id,
                c.assigned_maintenance_id,

                mw.maintenance_id,
                mw.work_status,
                mw.verification_status,
                mw.completion_date,
                c.is_duplicate,
                c.parent_complaint_id

            FROM complaints c

            JOIN users u
                ON c.user_id = u.user_id

            LEFT JOIN maintenance_work mw
                ON mw.maintenance_id = (
                    SELECT mw2.maintenance_id
                    FROM maintenance_work mw2
                    WHERE mw2.complaint_id = c.complaint_id
                    ORDER BY mw2.maintenance_id DESC
                    LIMIT 1
                )

            WHERE c.department_id = %s
              AND c.verification_status = 'Verified'

            ORDER BY c.created_at DESC
            """,
            (department_id,)
        )

        complaints = cursor.fetchall()

        cursor.close()
        connection.close()

        return render_template(
            "department_head_complaints.html",
            first_name=session.get("first_name"),
            last_name=session.get("last_name"),
            complaints=complaints
        )

    except Exception as e:

        print("Department Head Complaints Error:", e)

        if cursor:
            cursor.close()

        if connection:
            connection.close()

        return "Unable to load department complaints."

@app.route("/department-head/work-details/<int:complaint_id>")
def department_head_work_details(complaint_id):

    if "user_id" not in session:
        return redirect(url_for("login"))

    if session.get("role") != "Department Head":
        return "Access denied."

    department_id = session.get("department_id")

    if not department_id:
        return "No department assigned to this account."

    connection = None
    cursor = None

    try:
        connection = get_connection()
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                c.complaint_id,
                c.title,
                c.description,
                c.category,
                c.location,
                c.priority,
                c.status,
                c.created_at,

                c.assigned_engineer_id,
                c.assigned_maintenance_id,

                engineer.first_name,
                engineer.last_name,

                worker.first_name,
                worker.last_name,

                mw.maintenance_id,
                mw.assigned_date,
                mw.start_date,
                mw.completion_date,
                mw.work_status,
                mw.work_description,
                mw.remarks,
                mw.completion_report,
                mw.verification_status,
                c.image_path,
                mw.after_image

            FROM complaints c

            LEFT JOIN users engineer
                ON c.assigned_engineer_id = engineer.user_id
                AND engineer.role = 'Engineer'

            LEFT JOIN users worker
                ON c.assigned_maintenance_id = worker.user_id
                AND worker.role = 'Maintenance'

            LEFT JOIN maintenance_work mw
                ON mw.maintenance_id = (
                    SELECT mw2.maintenance_id
                    FROM maintenance_work mw2
                    WHERE mw2.complaint_id = c.complaint_id
                    ORDER BY mw2.maintenance_id DESC
                    LIMIT 1
                )

            WHERE c.complaint_id = %s
              AND c.department_id = %s

            """,
            (
                complaint_id,
                department_id
            )
        )

        work = cursor.fetchone()

        if not work:
            cursor.close()
            connection.close()
            return "Complaint not found."

        # Fetch inspection report if available
        cursor.execute(
            """
            SELECT
                r.report_id,
                r.inspection_date,
                r.findings,
                r.severity,
                r.estimated_cost,
                r.recommendation,
                r.inspection_image,
                r.created_at,
                eng.first_name,
                eng.last_name,
                r.required_materials,
                r.status,
                r.supervisor_remarks,
                r.verified_at
            FROM inspection_reports r
            LEFT JOIN users eng ON r.engineer_id = eng.user_id
            WHERE r.complaint_id = %s
            ORDER BY r.report_id DESC
            LIMIT 1
            """,
            (complaint_id,)
        )
        insp_row = cursor.fetchone()
        inspection_report = None
        if insp_row:
            inspection_report = {
                "report_id": insp_row[0],
                "inspection_date": insp_row[1],
                "findings": insp_row[2],
                "severity": insp_row[3],
                "estimated_cost": insp_row[4],
                "recommendation": insp_row[5],
                "inspection_image": insp_row[6],
                "created_at": insp_row[7],
                "engineer_name": f"{insp_row[8] or ''} {insp_row[9] or ''}".strip(),
                "required_materials": insp_row[10],
                "status": insp_row[11] or "Pending Verification",
                "supervisor_remarks": insp_row[12],
                "verified_at": insp_row[13]
            }

        cursor.close()
        connection.close()

        return render_template(
            "department_head_work_details.html",
            first_name=session.get("first_name"),
            last_name=session.get("last_name"),
            work=work,
            inspection_report=inspection_report
        )

    except Exception as e:

        print("Department Head Work Details Error:", e)

        if cursor:
            cursor.close()

        if connection:
            connection.close()

        return "Unable to load work details."


@app.route("/department-head/inspection-report/approve/<int:report_id>", methods=["POST"])
def department_head_approve_inspection(report_id):
    if "user_id" not in session:
        return redirect(url_for("login"))
    if session.get("role") not in ("Department Head", "department_head"):
        return "Access denied."

    supervisor_id = session.get("user_id")
    department_id = session.get("department_id")

    connection = None
    cursor = None
    try:
        connection = get_connection()
        cursor = connection.cursor()

        # Verify report belongs to supervisor's department
        cursor.execute(
            """
            SELECT r.report_id, r.complaint_id, c.title
            FROM inspection_reports r
            JOIN complaints c ON r.complaint_id = c.complaint_id
            WHERE r.report_id = %s AND c.department_id = %s
            """,
            (report_id, department_id)
        )
        row = cursor.fetchone()
        if not row:
            flash("Inspection report not found or does not belong to your department.", "error")
            cursor.close()
            connection.close()
            return redirect(url_for("department_head_dashboard"))

        complaint_id = row[1]
        complaint_title = row[2]

        # Update inspection report status to Approved
        cursor.execute(
            """
            UPDATE inspection_reports
            SET status = 'Approved',
                verified_by = %s,
                verified_at = CURRENT_TIMESTAMP
            WHERE report_id = %s
            """,
            (supervisor_id, report_id)
        )

        # Update complaint status to Maintenance (Authorize recommended maintenance work)
        cursor.execute(
            """
            UPDATE complaints
            SET status = 'Maintenance',
                updated_at = CURRENT_TIMESTAMP
            WHERE complaint_id = %s
            """,
            (complaint_id,)
        )

        connection.commit()
        cursor.close()
        connection.close()

        flash(f"Inspection Report #REP-{report_id} for '{complaint_title}' approved! Maintenance work has been authorized.", "success")
        referrer = request.referrer
        if referrer:
            return redirect(referrer)
        return redirect(url_for("department_head_dashboard"))

    except Exception as e:
        print("Approve Inspection Error:", repr(e))
        if connection:
            connection.rollback()
        if cursor:
            cursor.close()
        if connection:
            connection.close()
        flash("Failed to approve inspection report.", "error")
        return redirect(url_for("department_head_dashboard"))


@app.route("/department-head/inspection-report/request-revision/<int:report_id>", methods=["POST"])
def department_head_request_revision_inspection(report_id):
    if "user_id" not in session:
        return redirect(url_for("login"))
    if session.get("role") not in ("Department Head", "department_head"):
        return "Access denied."

    supervisor_id = session.get("user_id")
    department_id = session.get("department_id")
    remarks = request.form.get("remarks", "").strip()

    if not remarks:
        flash("Please provide revision remarks for the engineer.", "error")
        referrer = request.referrer
        if referrer:
            return redirect(referrer)
        return redirect(url_for("department_head_dashboard"))

    connection = None
    cursor = None
    try:
        connection = get_connection()
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT r.report_id, r.complaint_id, c.title
            FROM inspection_reports r
            JOIN complaints c ON r.complaint_id = c.complaint_id
            WHERE r.report_id = %s AND c.department_id = %s
            """,
            (report_id, department_id)
        )
        row = cursor.fetchone()
        if not row:
            flash("Inspection report not found or does not belong to your department.", "error")
            cursor.close()
            connection.close()
            return redirect(url_for("department_head_dashboard"))

        complaint_id = row[1]
        complaint_title = row[2]

        # Update inspection report status to Revision Requested and store supervisor remarks
        cursor.execute(
            """
            UPDATE inspection_reports
            SET status = 'Revision Requested',
                supervisor_remarks = %s,
                verified_by = %s,
                verified_at = CURRENT_TIMESTAMP
            WHERE report_id = %s
            """,
            (remarks, supervisor_id, report_id)
        )

        # Complaint remains in Inspection status
        cursor.execute(
            """
            UPDATE complaints
            SET status = 'Inspection',
                updated_at = CURRENT_TIMESTAMP
            WHERE complaint_id = %s
            """,
            (complaint_id,)
        )

        connection.commit()
        cursor.close()
        connection.close()

        flash(f"Inspection Report #REP-{report_id} for '{complaint_title}' returned to engineer for revision.", "info")
        referrer = request.referrer
        if referrer:
            return redirect(referrer)
        return redirect(url_for("department_head_dashboard"))

    except Exception as e:
        print("Request Revision Error:", repr(e))
        if connection:
            connection.rollback()
        if cursor:
            cursor.close()
        if connection:
            connection.close()
        flash("Failed to request revision for inspection report.", "error")
        return redirect(url_for("department_head_dashboard"))


@app.route("/department-head/verify-inspections")
def department_head_verify_inspections():
    if "user_id" not in session:
        return redirect(url_for("login"))
    if session.get("role") not in ("Department Head", "department_head"):
        return "Access denied."

    department_id = session.get("department_id")
    if not department_id:
        return "No department assigned to this account."

    connection = None
    cursor = None
    try:
        connection = get_connection()
        cursor = connection.cursor()

        cursor.execute("SELECT department_name FROM departments WHERE department_id = %s;", (department_id,))
        dept_row = cursor.fetchone()
        department_name = dept_row[0] if dept_row else "Department"

        cursor.execute(
            """
            SELECT
                r.report_id,
                r.complaint_id,
                c.title,
                c.category,
                c.location,
                c.priority,
                r.inspection_date,
                r.severity,
                r.findings,
                r.recommendation,
                r.required_materials,
                r.estimated_cost,
                r.inspection_image,
                r.status,
                r.created_at,
                eng.first_name || ' ' || COALESCE(eng.last_name, '') AS engineer_name,
                eng.email AS engineer_email,
                r.supervisor_remarks,
                r.verified_at,
                sup.first_name || ' ' || COALESCE(sup.last_name, '') AS supervisor_name
            FROM inspection_reports r
            JOIN complaints c ON r.complaint_id = c.complaint_id
            JOIN users eng ON r.engineer_id = eng.user_id
            LEFT JOIN users sup ON r.verified_by = sup.user_id
            WHERE c.department_id = %s
            ORDER BY
                CASE 
                    WHEN r.status = 'Pending Verification' THEN 1
                    WHEN r.status = 'Revision Requested' THEN 2
                    ELSE 3
                END,
                r.created_at DESC
            """,
            (department_id,)
        )
        report_rows = cursor.fetchall()
        reports = []
        for r in report_rows:
            reports.append({
                "report_id": r[0],
                "complaint_id": r[1],
                "title": r[2],
                "category": r[3],
                "location": r[4],
                "priority": r[5],
                "inspection_date": r[6],
                "severity": r[7],
                "findings": r[8],
                "recommendation": r[9],
                "required_materials": r[10],
                "estimated_cost": r[11],
                "inspection_image": r[12],
                "status": r[13],
                "created_at": r[14],
                "engineer_name": r[15].strip(),
                "engineer_email": r[16],
                "supervisor_remarks": r[17],
                "verified_at": r[18],
                "supervisor_name": r[19].strip() if r[19] else ""
            })

        pending_count = sum(1 for rep in reports if rep["status"] == "Pending Verification")
        approved_count = sum(1 for rep in reports if rep["status"] == "Approved")
        revision_count = sum(1 for rep in reports if rep["status"] == "Revision Requested")
        total_count = len(reports)

        return render_template(
            "department_head_verify_inspections.html",
            first_name=session.get("first_name"),
            last_name=session.get("last_name"),
            department_name=department_name,
            reports=reports,
            pending_count=pending_count,
            approved_count=approved_count,
            revision_count=revision_count,
            total_count=total_count
        )

    except Exception as e:
        print("Verify Inspections Page Error:", repr(e))
        return f"Unable to load inspection verification page: {e}"
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


@app.route("/department-head/assign-complaints", methods=["GET", "POST"])
@app.route("/department-head/assignments", methods=["GET", "POST"])
def department_head_assign_complaints():
    if "user_id" not in session:
        return redirect(url_for("login"))
    if session.get("role") not in ("Department Head", "department_head"):
        return "Access denied."

    department_id = session.get("department_id")
    assigned_by = session.get("user_id")
    if not department_id:
        return "No department assigned to this account."

    connection = None
    cursor = None
    try:
        connection = get_connection()
        cursor = connection.cursor()

        # Handle POST assignment
        if request.method == "POST":
            action = request.form.get("action")
            complaint_id_raw = request.form.get("complaint_id")
            selected_user_id_raw = request.form.get("user_id")

            if not complaint_id_raw or not selected_user_id_raw or not action:
                flash("Please select both a complaint and a staff member to assign.", "error")
                return redirect(url_for("department_head_assign_complaints"))

            try:
                complaint_id = int(complaint_id_raw)
                selected_user_id = int(selected_user_id_raw)
            except ValueError:
                flash("Invalid complaint or staff ID.", "error")
                return redirect(url_for("department_head_assign_complaints"))

            # Verify complaint belongs to supervisor's department and is not a duplicate
            cursor.execute(
                """
                SELECT complaint_id, title, status, is_duplicate, parent_complaint_id FROM complaints
                WHERE complaint_id = %s AND department_id = %s AND verification_status = 'Verified'
                """,
                (complaint_id, department_id)
            )
            c_row = cursor.fetchone()
            if not c_row:
                flash("Complaint not found or does not belong to your department.", "error")
                return redirect(url_for("department_head_assign_complaints"))

            if c_row[3] or c_row[2] == 'Duplicate':
                flash(f"Complaint #{complaint_id} is a confirmed duplicate linked to Main Ticket #{c_row[4]} and cannot be assigned independently.", "error")
                return redirect(url_for("department_head_assign_complaints"))

            c_title = c_row[1]

            if action == "assign_engineer":
                # Verify user is active Engineer in department
                cursor.execute(
                    "SELECT first_name, last_name FROM users WHERE user_id = %s AND role = 'Engineer' AND department_id = %s AND is_active = TRUE",
                    (selected_user_id, department_id)
                )
                eng_row = cursor.fetchone()
                if not eng_row:
                    flash("Selected engineer is not active or not in your department.", "error")
                    return redirect(url_for("department_head_assign_complaints"))

                eng_name = f"{eng_row[0]} {eng_row[1] or ''}".strip()

                cursor.execute(
                    """
                    UPDATE complaints
                    SET assigned_engineer_id = %s,
                        status = 'Assigned',
                        updated_at = CURRENT_TIMESTAMP
                    WHERE complaint_id = %s AND department_id = %s
                    """,
                    (selected_user_id, complaint_id, department_id)
                )
                connection.commit()
                flash(f"Complaint #{complaint_id} ('{c_title}') assigned to Engineer {eng_name} for inspection.", "success")
                return redirect(url_for("department_head_assign_complaints"))

            elif action == "assign_maintenance":
                # Verify user is active Maintenance in department
                cursor.execute(
                    "SELECT first_name, last_name FROM users WHERE user_id = %s AND role = 'Maintenance' AND department_id = %s AND is_active = TRUE",
                    (selected_user_id, department_id)
                )
                maint_row = cursor.fetchone()
                if not maint_row:
                    flash("Selected maintenance staff is not active or not in your department.", "error")
                    return redirect(url_for("department_head_assign_complaints"))

                maint_name = f"{maint_row[0]} {maint_row[1] or ''}".strip()

                cursor.execute(
                    """
                    UPDATE complaints
                    SET assigned_maintenance_id = %s,
                        status = 'Maintenance',
                        updated_at = CURRENT_TIMESTAMP
                    WHERE complaint_id = %s AND department_id = %s
                    """,
                    (selected_user_id, complaint_id, department_id)
                )

                # Ensure maintenance_work record exists or is updated
                cursor.execute(
                    "SELECT maintenance_id FROM maintenance_work WHERE complaint_id = %s",
                    (complaint_id,)
                )
                mw_row = cursor.fetchone()
                if mw_row:
                    cursor.execute(
                        """
                        UPDATE maintenance_work
                        SET maintenance_staff_id = %s,
                            assigned_by = %s,
                            work_status = 'Assigned',
                            assigned_date = CURRENT_DATE
                        WHERE maintenance_id = %s
                        """,
                        (selected_user_id, assigned_by, mw_row[0])
                    )
                else:
                    cursor.execute(
                        """
                        INSERT INTO maintenance_work (
                            complaint_id, assigned_by, maintenance_staff_id,
                            assigned_date, work_status, verification_status
                        ) VALUES (%s, %s, %s, CURRENT_DATE, 'Assigned', 'Pending')
                        """,
                        (complaint_id, assigned_by, selected_user_id)
                    )

                connection.commit()
                flash(f"Complaint #{complaint_id} ('{c_title}') assigned to Field Worker {maint_name} for maintenance repair.", "success")
                return redirect(url_for("department_head_assign_complaints"))

        # GET request: fetch data
        cursor.execute("SELECT department_name FROM departments WHERE department_id = %s;", (department_id,))
        dept_row = cursor.fetchone()
        department_name = dept_row[0] if dept_row else "Department"

        # Fetch available Engineers in department with active workload
        cursor.execute(
            """
            SELECT 
                u.user_id,
                u.first_name,
                u.last_name,
                u.email,
                u.phone,
                (SELECT COUNT(*) FROM complaints c WHERE c.assigned_engineer_id = u.user_id AND c.status IN ('Assigned', 'Inspection')) AS active_jobs
            FROM users u
            WHERE u.role = 'Engineer' AND u.department_id = %s AND u.is_active = TRUE
            ORDER BY u.first_name, u.last_name
            """,
            (department_id,)
        )
        engineer_rows = cursor.fetchall()
        engineers = []
        for e in engineer_rows:
            engineers.append({
                "user_id": e[0],
                "first_name": e[1],
                "last_name": e[2] or "",
                "name": f"{e[1]} {e[2] or ''}".strip(),
                "email": e[3],
                "phone": e[4] or "-",
                "active_jobs": e[5] or 0
            })

        # Fetch available Maintenance staff in department with active workload
        cursor.execute(
            """
            SELECT 
                u.user_id,
                u.first_name,
                u.last_name,
                u.email,
                u.phone,
                (SELECT COUNT(*) FROM maintenance_work m WHERE m.maintenance_staff_id = u.user_id AND m.work_status IN ('Assigned', 'In Progress', 'Pending Verification')) AS active_tasks
            FROM users u
            WHERE u.role = 'Maintenance' AND u.department_id = %s AND u.is_active = TRUE
            ORDER BY u.first_name, u.last_name
            """,
            (department_id,)
        )
        worker_rows = cursor.fetchall()
        field_workers = []
        for w in worker_rows:
            field_workers.append({
                "user_id": w[0],
                "first_name": w[1],
                "last_name": w[2] or "",
                "name": f"{w[1]} {w[2] or ''}".strip(),
                "email": w[3],
                "phone": w[4] or "-",
                "active_tasks": w[5] or 0
            })

        # Fetch complaints for this department
        cursor.execute(
            """
            SELECT
                c.complaint_id,
                c.title,
                c.description,
                c.category,
                c.location,
                c.priority,
                c.status,
                c.created_at,
                c.assigned_engineer_id,
                c.assigned_maintenance_id,
                eng.first_name || ' ' || COALESCE(eng.last_name, '') AS engineer_name,
                maint.first_name || ' ' || COALESCE(maint.last_name, '') AS maintenance_name,
                ir.report_id,
                ir.status AS inspection_report_status,
                ir.recommendation,
                ir.required_materials
            FROM complaints c
            LEFT JOIN users eng ON c.assigned_engineer_id = eng.user_id
            LEFT JOIN users maint ON c.assigned_maintenance_id = maint.user_id
            LEFT JOIN (
                SELECT DISTINCT ON (complaint_id) report_id, complaint_id, status, recommendation, required_materials
                FROM inspection_reports
                ORDER BY complaint_id, created_at DESC
            ) ir ON c.complaint_id = ir.complaint_id
            WHERE c.department_id = %s
              AND c.verification_status = 'Verified'
              AND c.status NOT IN ('Rejected', 'Duplicate')
              AND (c.is_duplicate IS FALSE OR c.is_duplicate IS NULL)
            ORDER BY
                CASE
                    WHEN c.status IN ('Pending', 'Verified') AND c.assigned_engineer_id IS NULL AND c.assigned_maintenance_id IS NULL THEN 1
                    WHEN c.status = 'Maintenance' AND c.assigned_maintenance_id IS NULL THEN 2
                    WHEN c.status IN ('Assigned', 'Inspection') THEN 3
                    ELSE 4
                END,
                c.created_at DESC
            """,
            (department_id,)
        )
        complaint_rows = cursor.fetchall()
        complaints = []
        for cr in complaint_rows:
            stage = "other"
            if cr[6] in ('Pending', 'Verified') and not cr[8] and not cr[9]:
                stage = "need_engineer"
            elif cr[6] == 'Maintenance' and not cr[9]:
                stage = "need_maintenance"
            elif cr[8] or cr[9]:
                stage = "assigned"

            complaints.append({
                "complaint_id": cr[0],
                "title": cr[1],
                "description": cr[2],
                "category": cr[3],
                "location": cr[4],
                "priority": cr[5],
                "status": cr[6],
                "created_at": cr[7],
                "assigned_engineer_id": cr[8],
                "assigned_maintenance_id": cr[9],
                "engineer_name": cr[10].strip() if cr[10] else None,
                "maintenance_name": cr[11].strip() if cr[11] else None,
                "inspection_report_id": cr[12],
                "inspection_report_status": cr[13],
                "recommendation": cr[14],
                "required_materials": cr[15],
                "stage": stage
            })

        need_engineer_count = sum(1 for c in complaints if c["stage"] == "need_engineer")
        need_maintenance_count = sum(1 for c in complaints if c["stage"] == "need_maintenance")
        assigned_count = sum(1 for c in complaints if c["stage"] == "assigned")
        total_complaints_count = len(complaints)

        return render_template(
            "department_head_assign_complaints.html",
            first_name=session.get("first_name"),
            last_name=session.get("last_name"),
            department_name=department_name,
            complaints=complaints,
            engineers=engineers,
            field_workers=field_workers,
            need_engineer_count=need_engineer_count,
            need_maintenance_count=need_maintenance_count,
            assigned_count=assigned_count,
            total_complaints_count=total_complaints_count,
            total_engineers_count=len(engineers),
            total_workers_count=len(field_workers)
        )

    except Exception as e:
        print("Assign Complaints Error:", repr(e))
        return f"Unable to load assignment page: {e}"
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


@app.route("/department-head/verify-work/<int:complaint_id>",methods=["GET", "POST"])
def department_head_verify_work(complaint_id):

    if "user_id" not in session:
        return redirect(url_for("login"))

    if session.get("role") != "Department Head":
        return "Access denied."

    department_id = session.get("department_id")
    department_head_id = session.get("user_id")

    if not department_id:
        return "No department assigned to this account."

    connection = None
    cursor = None

    try:

        connection = get_connection()
        cursor = connection.cursor()

        # =========================================================
        # GET WORK ORDER
        # =========================================================

        cursor.execute(
            """
            SELECT
                mw.maintenance_id,
                mw.complaint_id,
                mw.assigned_by,
                mw.maintenance_staff_id,
                mw.assigned_date,
                mw.start_date,
                mw.completion_date,
                mw.work_status,
                mw.work_description,
                mw.remarks,
                mw.before_image,
                mw.after_image,
                mw.completion_report,
                mw.verified_by,
                mw.verification_status,
                mw.verified_at,

                c.title,
                c.description,
                c.category,
                c.location,
                c.priority,
                c.status,

                u.first_name,
                u.last_name

            FROM maintenance_work mw

            JOIN complaints c
                ON mw.complaint_id = c.complaint_id

            JOIN users u
                ON mw.maintenance_staff_id = u.user_id

            WHERE mw.complaint_id = %s
              AND c.department_id = %s

            ORDER BY mw.maintenance_id DESC

            LIMIT 1
            """,
            (
                complaint_id,
                department_id
            )
        )

        work = cursor.fetchone()

        if not work:
            cursor.close()
            connection.close()

            return "Work order not found."

        # =========================================================
        # POST - VERIFY / REJECT
        # =========================================================

        if request.method == "POST":

            action = request.form.get("action")

            # -----------------------------------------------------
            # VERIFY
            # -----------------------------------------------------

            if action == "verify":

                if work[7] != "Completed":
                    flash("Only completed work can be verified.", "error")
                    return redirect(url_for("department_head_dashboard"))

                if work[14] != "Pending" and work[14] is not None:
                    flash("This work order has already been reviewed.", "info")
                    return redirect(url_for("department_head_dashboard"))

                cursor.execute(
                    """
                    UPDATE maintenance_work
                    SET
                        verification_status = 'Verified',
                        verified_by = %s,
                        verified_at = CURRENT_TIMESTAMP
                    WHERE maintenance_id = %s
                      AND work_status = 'Completed'
                    """,
                    (
                        department_head_id,
                        work[0]
                    )
                )

                # Only after Department Head verifies does complaint status become Completed
                cursor.execute(
                    """
                    UPDATE complaints
                    SET
                        status = 'Completed',
                        updated_at = CURRENT_TIMESTAMP
                    WHERE complaint_id = %s
                      AND department_id = %s
                    """,
                    (
                        complaint_id,
                        department_id
                    )
                )

                # Synchronize any linked child duplicate complaints
                cursor.execute(
                    """
                    UPDATE complaints
                    SET
                        status = 'Completed',
                        updated_at = CURRENT_TIMESTAMP
                    WHERE parent_complaint_id = %s
                      AND (is_duplicate IS TRUE OR status = 'Duplicate')
                    """,
                    (complaint_id,)
                )

                # Send notifications to main complainant
                cursor.execute(
                    """
                    INSERT INTO notifications (user_id, complaint_id, message, is_read, created_at)
                    SELECT 
                        user_id, 
                        complaint_id, 
                        'Your complaint #' || %s || ' maintenance repair work has been verified and marked as Completed by the Department Head.',
                        FALSE, 
                        CURRENT_TIMESTAMP
                    FROM complaints 
                    WHERE complaint_id = %s;
                    """,
                    (complaint_id, complaint_id)
                )

                # Send notifications to linked duplicate complainants
                cursor.execute(
                    """
                    INSERT INTO notifications (user_id, complaint_id, message, is_read, created_at)
                    SELECT 
                        user_id, 
                        complaint_id, 
                        'Main Complaint #' || %s || ' (which your complaint #' || complaint_id || ' is linked to) has been verified and marked as Completed by the department.',
                        FALSE, 
                        CURRENT_TIMESTAMP
                    FROM complaints 
                    WHERE parent_complaint_id = %s AND (is_duplicate IS TRUE OR status = 'Duplicate');
                    """,
                    (complaint_id, complaint_id)
                )

                connection.commit()

                cursor.close()
                connection.close()

                flash(f"Maintenance work for Complaint #{complaint_id} verified successfully! Status is now Completed.", "success")
                return redirect(url_for("department_head_dashboard"))

            # -----------------------------------------------------
            # REJECT
            # -----------------------------------------------------

            elif action == "reject":

                if work[7] != "Completed":
                    flash("Only completed work can be rejected.", "error")
                    return redirect(url_for("department_head_dashboard"))

                cursor.execute(
                    """
                    UPDATE maintenance_work
                    SET
                        verification_status = 'Revision Requested',
                        verified_by = %s,
                        verified_at = CURRENT_TIMESTAMP,
                        work_status = 'In Progress',
                        completion_date = NULL
                    WHERE maintenance_id = %s
                      AND work_status = 'Completed'
                    """,
                    (
                        department_head_id,
                        work[0]
                    )
                )

                cursor.execute(
                    """
                    UPDATE complaints
                    SET
                        status = 'In Progress',
                        updated_at = CURRENT_TIMESTAMP
                    WHERE complaint_id = %s
                      AND department_id = %s
                    """,
                    (
                        complaint_id,
                        department_id
                    )
                )

                connection.commit()

                cursor.close()
                connection.close()

                flash(f"Work order #{complaint_id} returned to maintenance worker for revisions.", "info")
                return redirect(url_for("department_head_dashboard"))

            else:
                return "Invalid verification action."

        # =========================================================
        # GET - SHOW VERIFICATION PAGE
        # =========================================================

        cursor.close()
        connection.close()

        return render_template(
            "department_head_verify_work.html",
            first_name=session.get("first_name"),
            last_name=session.get("last_name"),
            work=work
        )

    except Exception as e:

        print("Department Head Verify Work Error:", e)

        if cursor:
            cursor.close()

        if connection:
            connection.close()

        return "Unable to load work verification."

@app.route("/department-head/assign-work/<int:complaint_id>", methods=["GET", "POST"])
def department_head_assign_work(complaint_id):

    if "user_id" not in session:
        return redirect(url_for("login"))

    if session.get("role") != "Department Head":
        return "Access denied."

    department_id = session.get("department_id")
    assigned_by = session.get("user_id")

    if not department_id:
        return "No department assigned to this account."

    connection = None
    cursor = None

    try:
        connection = get_connection()
        cursor = connection.cursor()

        # ---------------------------------------------------------
        # Get the complaint
        # ---------------------------------------------------------
        cursor.execute(
            """
            SELECT
                c.complaint_id,
                c.title,
                c.description,
                c.location,
                c.priority,
                c.status,
                c.verification_status
            FROM complaints c
            WHERE c.complaint_id = %s
              AND c.department_id = %s
              AND c.verification_status = 'Verified'
              AND (c.is_duplicate IS FALSE OR c.is_duplicate IS NULL)
              AND c.status != 'Duplicate'
            """,
            (complaint_id, department_id)
        )

        complaint = cursor.fetchone()

        if not complaint:
            cursor.close()
            connection.close()
            flash("Complaint not found or not available for your department.", "error")
            return redirect(url_for("department_head_complaints"))

        # ---------------------------------------------------------
        # Do not allow reassignment if already assigned
        # ---------------------------------------------------------
        cursor.execute(
            """
            SELECT
                assigned_engineer_id,
                assigned_maintenance_id
            FROM complaints
            WHERE complaint_id = %s
            """,
            (complaint_id,)
        )

        assignment = cursor.fetchone()

        if request.method == "GET" and assignment and (
            assignment[0] is not None or
            assignment[1] is not None
        ):
            cursor.close()
            connection.close()
            flash(f"Complaint #{complaint_id} has already been assigned.", "info")
            return redirect(url_for("department_head_complaints"))

        # ---------------------------------------------------------
        # Handle POST
        # ---------------------------------------------------------
        if request.method == "POST":

            assignment_type = request.form.get("assignment_type")
            assigned_candidates = [uid.strip() for uid in request.form.getlist("assigned_user_id") if uid and str(uid).strip()]
            assigned_user_id = assigned_candidates[0] if assigned_candidates else None

            if not assignment_type or not assigned_user_id:
                flash("Please select a work type and a person to assign.", "error")
                return redirect(url_for("department_head_assign_work", complaint_id=complaint_id))

            try:
                assigned_user_id = int(assigned_user_id)
            except ValueError:
                flash("Invalid user selected.", "error")
                return redirect(url_for("department_head_assign_work", complaint_id=complaint_id))

            # =====================================================
            # DIRECT FIELD WORK / MAINTENANCE WORKER
            # =====================================================
            if assignment_type == "field_worker":

                cursor.execute(
                    """
                    SELECT user_id, first_name, last_name
                    FROM users
                    WHERE user_id = %s
                      AND role = 'Maintenance'
                      AND department_id = %s
                      AND is_active = TRUE
                    """,
                    (assigned_user_id, department_id)
                )

                worker = cursor.fetchone()

                if not worker:
                    flash("Selected field worker is not available for this department.", "error")
                    return redirect(url_for("department_head_assign_work", complaint_id=complaint_id))

                worker_name = f"{worker[1]} {worker[2] or ''}".strip()

                # -------------------------------------------------
                # Update the complaint
                # -------------------------------------------------
                cursor.execute(
                    """
                    UPDATE complaints
                    SET
                        assigned_maintenance_id = %s,
                        assigned_engineer_id = NULL,
                        status = 'Maintenance',
                        updated_at = CURRENT_TIMESTAMP
                    WHERE complaint_id = %s
                      AND department_id = %s
                    """,
                    (
                        assigned_user_id,
                        complaint_id,
                        department_id
                    )
                )

                if cursor.rowcount == 0:
                    connection.rollback()
                    flash("The complaint could not be updated. Please try again.", "error")
                    return redirect(url_for("department_head_complaints"))

                # -------------------------------------------------
                # Create maintenance work record
                # -------------------------------------------------
                cursor.execute(
                    """
                    INSERT INTO maintenance_work (
                        complaint_id,
                        assigned_by,
                        maintenance_staff_id,
                        assigned_date,
                        work_status,
                        verification_status
                    )
                    VALUES (
                        %s,
                        %s,
                        %s,
                        CURRENT_DATE,
                        'Assigned',
                        'Pending'
                    )
                    """,
                    (
                        complaint_id,
                        assigned_by,
                        assigned_user_id
                    )
                )

                connection.commit()
                cursor.close()
                connection.close()

                flash(f"Complaint #{complaint_id} successfully assigned to Field Worker {worker_name}.", "success")
                return redirect(
                    url_for("department_head_complaints")
                )

            # =====================================================
            # ENGINEER INSPECTION
            # =====================================================
            elif assignment_type == "engineer":

                cursor.execute(
                    """
                    SELECT user_id, first_name, last_name
                    FROM users
                    WHERE user_id = %s
                      AND role = 'Engineer'
                      AND department_id = %s
                      AND is_active = TRUE
                    """,
                    (assigned_user_id, department_id)
                )

                engineer = cursor.fetchone()

                if not engineer:
                    flash("Selected engineer is not available for this department.", "error")
                    return redirect(url_for("department_head_assign_work", complaint_id=complaint_id))

                eng_name = f"{engineer[1]} {engineer[2] or ''}".strip()

                cursor.execute(
                    """
                    UPDATE complaints
                    SET
                        assigned_engineer_id = %s,
                        assigned_maintenance_id = NULL,
                        status = 'Assigned',
                        updated_at = CURRENT_TIMESTAMP
                    WHERE complaint_id = %s
                      AND department_id = %s
                    """,
                    (
                        assigned_user_id,
                        complaint_id,
                        department_id
                    )
                )

                if cursor.rowcount == 0:
                    connection.rollback()
                    flash("The complaint could not be updated. Please try again.", "error")
                    return redirect(url_for("department_head_complaints"))

                connection.commit()
                cursor.close()
                connection.close()

                flash(f"Complaint #{complaint_id} successfully assigned to Engineer {eng_name} for inspection.", "success")
                return redirect(
                    url_for("department_head_complaints")
                )

            else:
                flash("Invalid assignment type.", "error")
                return redirect(url_for("department_head_assign_work", complaint_id=complaint_id))

        # =========================================================
        # GET: Load available Engineers
        # and Maintenance Workers
        # =========================================================

        cursor.execute(
            """
            SELECT
                user_id,
                first_name,
                last_name
            FROM users
            WHERE role = 'Engineer'
              AND department_id = %s
              AND is_active = TRUE
            ORDER BY first_name, last_name
            """,
            (department_id,)
        )

        engineers = cursor.fetchall()

        cursor.execute(
            """
            SELECT
                user_id,
                first_name,
                last_name
            FROM users
            WHERE role = 'Maintenance'
              AND department_id = %s
              AND is_active = TRUE
            ORDER BY first_name, last_name
            """,
            (department_id,)
        )

        field_workers = cursor.fetchall()

        cursor.close()
        connection.close()

        return render_template(
            "department_head_assign_work.html",
            first_name=session.get("first_name"),
            last_name=session.get("last_name"),
            complaint=complaint,
            engineers=engineers,
            field_workers=field_workers
        )

    except Exception as e:

        print("Department Head Assign Work Error:", e)

        # Roll back any partial database operation
        if connection:
            try:
                connection.rollback()
            except Exception:
                pass

        if cursor:
            try:
                cursor.close()
            except Exception:
                pass

        if connection:
            try:
                connection.close()
            except Exception:
                pass

        return "Unable to load work assignment page."

@app.route("/department-head/assign-maintenance", methods=["GET", "POST"])
def department_head_assign_maintenance():

    if "user_id" not in session:
        return redirect(url_for("login"))

    if session.get("role") != "Department Head":
        return "Access denied."

    department_id = session.get("department_id")

    if not department_id:
        return "No department assigned to this account."

    connection = None
    cursor = None

    try:
        connection = get_connection()
        cursor = connection.cursor()

        # ---------------------------------------
        # Get active Maintenance staff
        # from the same department
        # ---------------------------------------
        cursor.execute(
            """
            SELECT
                user_id,
                first_name,
                last_name,
                email
            FROM users
            WHERE role = 'Maintenance'
              AND department_id = %s
              AND is_active = TRUE
            ORDER BY first_name, last_name
            """,
            (department_id,)
        )

        maintenance_rows = cursor.fetchall()

        maintenance_staff = []

        for row in maintenance_rows:
            maintenance_staff.append({
                "maintenance_id": row[0],
                "name": f"{row[1]} {row[2]}",
                "email": row[3]
            })

        # ---------------------------------------
        # Complaints sent to Maintenance
        # but not yet assigned to a person
        # ---------------------------------------
        cursor.execute(
            """
            SELECT
                complaint_id,
                title,
                location,
                priority,
                status
            FROM complaints
            WHERE department_id = %s
              AND verification_status = 'Verified'
              AND status = 'Maintenance'
              AND assigned_maintenance_id IS NULL
            ORDER BY created_at DESC
            """,
            (department_id,)
        )

        complaint_rows = cursor.fetchall()

        maintenance_complaints = []

        for row in complaint_rows:
            maintenance_complaints.append({
                "complaint_id": row[0],
                "title": row[1],
                "location": row[2],
                "priority": row[3],
                "status": row[4]
            })

        # ---------------------------------------
        # POST - assign maintenance staff
        # ---------------------------------------
        if request.method == "POST":

            complaint_id = request.form.get("complaint_id")
            maintenance_id = request.form.get("maintenance_id")

            if not complaint_id or not maintenance_id:
                return render_template(
                    "assign_maintenance.html",
                    first_name=session.get("first_name"),
                    last_name=session.get("last_name"),
                    maintenance_staff=maintenance_staff,
                    maintenance_complaints=maintenance_complaints,
                    error="Please select both a complaint and maintenance staff."
                )

            try:
                complaint_id = int(complaint_id)
                maintenance_id = int(maintenance_id)
            except ValueError:
                return "Invalid complaint or maintenance staff."

            # ---------------------------------------
            # Verify complaint belongs to this
            # department and is waiting for assignment
            # ---------------------------------------
            cursor.execute(
                """
                SELECT complaint_id
                FROM complaints
                WHERE complaint_id = %s
                  AND department_id = %s
                  AND verification_status = 'Verified'
                  AND status = 'Maintenance'
                  AND assigned_maintenance_id IS NULL
                """,
                (complaint_id, department_id)
            )

            complaint = cursor.fetchone()

            if complaint is None:
                connection.rollback()

                return render_template(
                    "assign_maintenance.html",
                    first_name=session.get("first_name"),
                    last_name=session.get("last_name"),
                    maintenance_staff=maintenance_staff,
                    maintenance_complaints=maintenance_complaints,
                    error="Complaint not found or it has already been assigned."
                )

            # ---------------------------------------
            # Verify maintenance staff belongs to
            # this department
            # ---------------------------------------
            cursor.execute(
                """
                SELECT user_id
                FROM users
                WHERE user_id = %s
                  AND role = 'Maintenance'
                  AND department_id = %s
                  AND is_active = TRUE
                """,
                (maintenance_id, department_id)
            )

            maintenance_user = cursor.fetchone()

            if maintenance_user is None:
                connection.rollback()

                return render_template(
                    "assign_maintenance.html",
                    first_name=session.get("first_name"),
                    last_name=session.get("last_name"),
                    maintenance_staff=maintenance_staff,
                    maintenance_complaints=maintenance_complaints,
                    error="Invalid maintenance staff selected."
                )

            # ---------------------------------------
            # Assign maintenance staff
            # ---------------------------------------
            cursor.execute(
                """
                UPDATE complaints
                SET
                    assigned_maintenance_id = %s,
                    updated_at = CURRENT_TIMESTAMP
                WHERE complaint_id = %s
                  AND department_id = %s
                """,
                (
                    maintenance_id,
                    complaint_id,
                    department_id
                )
            )

            # Ensure maintenance_work record exists
            cursor.execute(
                """
                SELECT maintenance_id FROM maintenance_work
                WHERE complaint_id = %s
                """,
                (complaint_id,)
            )
            existing_mw = cursor.fetchone()
            if not existing_mw:
                cursor.execute(
                    """
                    INSERT INTO maintenance_work (
                        complaint_id,
                        assigned_by,
                        maintenance_staff_id,
                        assigned_date,
                        work_status,
                        verification_status
                    )
                    VALUES (
                        %s,
                        %s,
                        %s,
                        CURRENT_DATE,
                        'Assigned',
                        'Pending'
                    )
                    """,
                    (
                        complaint_id,
                        session.get("user_id"),
                        maintenance_id
                    )
                )
            else:
                cursor.execute(
                    """
                    UPDATE maintenance_work
                    SET
                        maintenance_staff_id = %s,
                        work_status = 'Assigned',
                        assigned_date = CURRENT_DATE
                    WHERE maintenance_id = %s
                    """,
                    (
                        maintenance_id,
                        existing_mw[0]
                    )
                )

            connection.commit()

            return redirect(
                url_for("department_head_complaints")
            )

        cursor.close()
        connection.close()

        return render_template(
            "assign_maintenance.html",
            first_name=session.get("first_name"),
            last_name=session.get("last_name"),
            maintenance_staff=maintenance_staff,
            maintenance_complaints=maintenance_complaints
        )

    except Exception as e:

        print(
            "Department Head Assign Maintenance Error:",
            repr(e)
        )

        if connection:
            connection.rollback()

        if cursor:
            cursor.close()

        if connection:
            connection.close()

        return "Unable to assign maintenance staff."


#Engineer
@app.route("/engineer-dashboard")
def engineer_dashboard():

    if "user_id" not in session:
        return redirect(url_for("login"))

    if session.get("role") != "Engineer":
        return "Access denied."

    engineer_id = session.get("user_id")

    connection = None
    cursor = None

    try:
        connection = get_connection()
        cursor = connection.cursor()

        # -----------------------------------------
        # Assigned complaints
        # -----------------------------------------
        cursor.execute(
            """
            SELECT
                c.complaint_id,
                c.title,
                c.category,
                c.location,
                c.priority,
                c.status,
                c.created_at,
                c.updated_at,
                c.latitude,
                c.longitude,
                c.image_path
            FROM complaints c
            WHERE c.assigned_engineer_id = %s
            ORDER BY c.updated_at DESC, c.created_at DESC
            LIMIT 25
            """,
            (engineer_id,)
        )

        rows = cursor.fetchall()

        assignments = []

        for row in rows:
            assignments.append({
                "complaint_id": row[0],
                "title": row[1],
                "category": row[2],
                "location": row[3],
                "priority": row[4],
                "status": row[5],
                "assigned_date": row[6],
                "latitude": float(row[8]) if row[8] is not None else None,
                "longitude": float(row[9]) if row[9] is not None else None,
                "image_path": row[10]
            })

        # -----------------------------------------
        # Dashboard counts
        # -----------------------------------------
        cursor.execute(
            """
            SELECT COUNT(*)
            FROM complaints
            WHERE assigned_engineer_id = %s
              AND status = 'Assigned'
            """,
            (engineer_id,)
        )
        assigned_count = cursor.fetchone()[0]

        cursor.execute(
            """
            SELECT COUNT(*)
            FROM complaints
            WHERE assigned_engineer_id = %s
              AND status = 'Inspection'
            """,
            (engineer_id,)
        )
        inspection_count = cursor.fetchone()[0]

        cursor.execute(
            """
            SELECT COUNT(*)
            FROM complaints
            WHERE assigned_engineer_id = %s
              AND status = 'Completed'
            """,
            (engineer_id,)
        )
        completed_count = cursor.fetchone()[0]

        cursor.execute(
            """
            SELECT COUNT(*)
            FROM complaints
            WHERE assigned_engineer_id = %s
              AND priority = 'High'
              AND status NOT IN ('Completed', 'Closed')
            """,
            (engineer_id,)
        )
        high_priority = cursor.fetchone()[0]

        cursor.execute(
            """
            SELECT COUNT(*)
            FROM complaints
            WHERE assigned_engineer_id = %s
              AND status = 'Maintenance'
            """,
            (engineer_id,)
        )
        maintenance_count = cursor.fetchone()[0]

        # Total inspection reports filed by this engineer
        cursor.execute(
            """
            SELECT COUNT(*)
            FROM inspection_reports
            WHERE engineer_id = %s
            """,
            (engineer_id,)
        )
        reports_count = cursor.fetchone()[0]

        # Recent inspection reports
        cursor.execute(
            """
            SELECT
                r.report_id,
                r.complaint_id,
                c.title,
                c.category,
                r.inspection_date,
                r.severity,
                r.recommendation,
                r.estimated_cost
            FROM inspection_reports r
            JOIN complaints c ON r.complaint_id = c.complaint_id
            WHERE r.engineer_id = %s
            ORDER BY r.created_at DESC
            LIMIT 5
            """,
            (engineer_id,)
        )
        recent_report_rows = cursor.fetchall()
        recent_reports = []
        for rr in recent_report_rows:
            recent_reports.append({
                "report_id": rr[0],
                "complaint_id": rr[1],
                "title": rr[2],
                "category": rr[3],
                "inspection_date": rr[4],
                "severity": rr[5],
                "recommendation": rr[6],
                "estimated_cost": rr[7]
            })

        cursor.close()
        connection.close()

        return render_template(
            "engineer_dashboard.html",
            first_name=session.get("first_name"),
            last_name=session.get("last_name"),
            assignments=assignments,
            assigned_count=assigned_count,
            inspection_count=inspection_count,
            completed_count=completed_count,
            high_priority=high_priority,
            maintenance_count=maintenance_count,
            reports_count=reports_count,
            recent_reports=recent_reports
        )

    except Exception as e:
        print("Engineer Dashboard Error:", repr(e))

        if cursor:
            try:
                cursor.close()
            except Exception:
                pass

        if connection:
            try:
                connection.close()
            except Exception:
                pass

        return "Unable to load engineer dashboard."

@app.route("/engineer-assigned")
def engineer_assigned():

    if "user_id" not in session:
        return redirect(url_for("login"))

    if session.get("role") != "Engineer":
        return "Access denied."

    engineer_id = session.get("user_id")

    filter_status = request.args.get("status", "").strip()
    filter_priority = request.args.get("priority", "").strip()
    search_query = request.args.get("search", "").strip()

    connection = None
    cursor = None

    try:
        connection = get_connection()
        cursor = connection.cursor()

        query = """
            SELECT
                c.complaint_id,
                c.title,
                c.category,
                c.location,
                c.priority,
                c.status,
                c.created_at,
                c.updated_at
            FROM complaints c
            WHERE c.assigned_engineer_id = %s
        """
        params = [engineer_id]

        if filter_status:
            query += " AND c.status = %s"
            params.append(filter_status)

        if filter_priority:
            query += " AND c.priority = %s"
            params.append(filter_priority)

        if search_query:
            query += " AND (c.title ILIKE %s OR c.location ILIKE %s OR c.category ILIKE %s OR CAST(c.complaint_id AS TEXT) ILIKE %s)"
            like_val = f"%{search_query}%"
            params.extend([like_val, like_val, like_val, like_val])

        query += " ORDER BY c.updated_at DESC, c.created_at DESC"

        cursor.execute(query, tuple(params))
        rows = cursor.fetchall()

        assignments = []

        for row in rows:
            assignments.append({
                "complaint_id": row[0],
                "title": row[1],
                "category": row[2],
                "location": row[3],
                "priority": row[4],
                "status": row[5],
                "assigned_date": row[6],
                "updated_at": row[7],
                "deadline": None
            })

        cursor.close()
        connection.close()

        return render_template(
            "engineer_assigned.html",
            first_name=session.get("first_name"),
            last_name=session.get("last_name"),
            assignments=assignments,
            current_status=filter_status,
            current_priority=filter_priority,
            search_query=search_query
        )

    except Exception as e:

        print("Engineer Assignments Error:", repr(e))

        if cursor:
            cursor.close()

        if connection:
            connection.close()

        return "Unable to load your assignments."

@app.route("/engineer/complaint/<int:complaint_id>")
def engineer_view_complaint(complaint_id):

    if "user_id" not in session:
        return redirect(url_for("login"))

    if session.get("role") != "Engineer":
        return "Access denied."

    engineer_id = session.get("user_id")

    connection = None
    cursor = None

    try:
        connection = get_connection()
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                c.complaint_id,
                c.title,
                c.description,
                c.category,
                c.location,
                c.road_name,
                c.landmark,
                c.location_details,
                c.latitude,
                c.longitude,
                c.image_path,
                c.priority,
                c.status,
                c.verification_status,
                c.created_at,
                c.updated_at,
                u.first_name,
                u.last_name
            FROM complaints c
            JOIN users u
                ON c.user_id = u.user_id
            WHERE c.complaint_id = %s
              AND c.assigned_engineer_id = %s
            """,
            (complaint_id, engineer_id)
        )

        row = cursor.fetchone()

        if row is None:
            cursor.close()
            connection.close()
            return "Complaint not found or this complaint is not assigned to you.", 404

        complaint = {
            "complaint_id": row[0],
            "title": row[1],
            "description": row[2],
            "category": row[3],
            "location": row[4],
            "road_name": row[5],
            "landmark": row[6],
            "location_details": row[7],
            "latitude": row[8],
            "longitude": row[9],
            "image_path": row[10],
            "priority": row[11],
            "status": row[12],
            "verification_status": row[13],
            "created_at": row[14],
            "updated_at": row[15],
            "citizen_first_name": row[16],
            "citizen_last_name": row[17]
        }

        # Query existing inspection report
        cursor.execute(
            """
            SELECT
                r.report_id,
                r.complaint_id,
                r.engineer_id,
                r.inspection_date,
                r.findings,
                r.severity,
                r.estimated_cost,
                r.recommendation,
                r.inspection_image,
                r.created_at,
                r.required_materials,
                r.status,
                r.supervisor_remarks,
                r.verified_by,
                r.verified_at
            FROM inspection_reports r
            WHERE r.complaint_id = %s
            ORDER BY r.report_id DESC
            LIMIT 1
            """,
            (complaint_id,)
        )
        insp_row = cursor.fetchone()
        inspection_report = None
        if insp_row:
            inspection_report = {
                "report_id": insp_row[0],
                "complaint_id": insp_row[1],
                "engineer_id": insp_row[2],
                "inspection_date": insp_row[3],
                "findings": insp_row[4],
                "severity": insp_row[5],
                "estimated_cost": insp_row[6],
                "recommendation": insp_row[7],
                "inspection_image": insp_row[8],
                "created_at": insp_row[9],
                "required_materials": insp_row[10],
                "status": insp_row[11] or "Pending Verification",
                "supervisor_remarks": insp_row[12],
                "verified_by": insp_row[13],
                "verified_at": insp_row[14]
            }

        cursor.close()
        connection.close()

        return render_template(
            "engineer_view_complaint.html",
            first_name=session.get("first_name"),
            last_name=session.get("last_name"),
            complaint=complaint,
            inspection_report=inspection_report
        )

    except Exception as e:

        print("Engineer View Complaint Error:", repr(e))

        if cursor:
            cursor.close()

        if connection:
            connection.close()

        return "Unable to load complaint details."

@app.route("/engineer/submit-inspection/<int:complaint_id>", methods=["POST"])
def engineer_submit_inspection(complaint_id):

    if "user_id" not in session:
        return redirect(url_for("login"))

    if session.get("role") != "Engineer":
        return "Access denied."

    engineer_id = session.get("user_id")

    findings = request.form.get("findings", "").strip()
    inspection_date = request.form.get("inspection_date", "").strip()
    severity = request.form.get("severity", "Medium").strip()
    estimated_cost = request.form.get("estimated_cost", "").strip()
    recommendation = request.form.get("recommendation", "").strip()
    required_materials = request.form.get("required_materials", "").strip()

    if not findings:
        flash("Inspection findings are required.", "error")
        return redirect(url_for("engineer_view_complaint", complaint_id=complaint_id))

    cost_val = None
    if estimated_cost:
        try:
            cost_val = float(estimated_cost.replace(",", "").replace("₹", ""))
        except ValueError:
            cost_val = None

    date_val = None
    if inspection_date:
        try:
            date_val = datetime.datetime.strptime(inspection_date, "%Y-%m-%d").date()
        except Exception:
            date_val = datetime.date.today()
    else:
        date_val = datetime.date.today()

    # Handle image upload
    inspection_image_path = None
    if "inspection_image" in request.files:
        file = request.files["inspection_image"]
        if file and file.filename != "":
            ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
            if ext in ALLOWED_EXTENSIONS:
                unique_fn = f"insp_{complaint_id}_{int(datetime.datetime.now().timestamp())}.{ext}"
                folder = os.path.join(app.config["UPLOAD_FOLDER"], "inspections")
                os.makedirs(folder, exist_ok=True)
                full_path = os.path.join(folder, unique_fn)
                file.save(full_path)
                inspection_image_path = f"uploads/inspections/{unique_fn}"

    connection = None
    cursor = None

    try:
        connection = get_connection()
        cursor = connection.cursor()

        # Check complaint assignment
        cursor.execute(
            "SELECT complaint_id, status FROM complaints WHERE complaint_id = %s AND assigned_engineer_id = %s",
            (complaint_id, engineer_id)
        )
        row = cursor.fetchone()
        if not row:
            cursor.close()
            connection.close()
            flash("Complaint not found or not assigned to you.", "error")
            return redirect(url_for("engineer_assigned"))

        # Check if inspection report already exists
        cursor.execute(
            "SELECT report_id, inspection_image FROM inspection_reports WHERE complaint_id = %s ORDER BY report_id DESC LIMIT 1",
            (complaint_id,)
        )
        existing_report = cursor.fetchone()

        if existing_report:
            report_id = existing_report[0]
            old_img = existing_report[1]
            img_to_save = inspection_image_path if inspection_image_path else old_img
            cursor.execute(
                """
                UPDATE inspection_reports
                SET findings = %s,
                    inspection_date = %s,
                    severity = %s,
                    estimated_cost = %s,
                    recommendation = %s,
                    required_materials = %s,
                    inspection_image = %s,
                    engineer_id = %s,
                    status = 'Pending Verification',
                    supervisor_remarks = NULL,
                    verified_by = NULL,
                    verified_at = NULL
                WHERE report_id = %s
                """,
                (findings, date_val, severity, cost_val, recommendation, required_materials, img_to_save, engineer_id, report_id)
            )
        else:
            cursor.execute(
                """
                INSERT INTO inspection_reports (
                    complaint_id, engineer_id, inspection_date, findings,
                    severity, estimated_cost, recommendation, required_materials,
                    inspection_image, status
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, 'Pending Verification'
                )
                """,
                (complaint_id, engineer_id, date_val, findings, severity, cost_val, recommendation, required_materials, inspection_image_path)
            )

        # Update complaint status to Inspection (under verification; engineer cannot approve directly)
        cursor.execute(
            """
            UPDATE complaints
            SET status = 'Inspection',
                updated_at = CURRENT_TIMESTAMP
            WHERE complaint_id = %s
            """,
            (complaint_id,)
        )

        connection.commit()
        cursor.close()
        connection.close()

        flash(f"Inspection report for Complaint #{complaint_id} submitted for Department Head verification. Complaint status is Under Inspection.", "success")
        return redirect(url_for("engineer_view_complaint", complaint_id=complaint_id))

    except Exception as e:
        print("Submit Inspection Error:", repr(e))
        if connection:
            connection.rollback()
        if cursor:
            cursor.close()
        if connection:
            connection.close()
        flash("Failed to submit inspection report. Please try again.", "error")
        return redirect(url_for("engineer_view_complaint", complaint_id=complaint_id))

@app.route("/engineer/inspection-reports")
def engineer_inspection_reports():

    if "user_id" not in session:
        return redirect(url_for("login"))

    if session.get("role") != "Engineer":
        return "Access denied."

    engineer_id = session.get("user_id")

    filter_severity = request.args.get("severity", "").strip()
    search_query = request.args.get("search", "").strip()

    connection = None
    cursor = None

    try:
        connection = get_connection()
        cursor = connection.cursor()

        query = """
            SELECT
                r.report_id,
                r.complaint_id,
                c.title,
                c.category,
                c.location,
                c.status,
                r.inspection_date,
                r.findings,
                r.severity,
                r.estimated_cost,
                r.recommendation,
                r.inspection_image,
                r.created_at
            FROM inspection_reports r
            JOIN complaints c ON r.complaint_id = c.complaint_id
            WHERE r.engineer_id = %s
        """
        params = [engineer_id]

        if filter_severity:
            query += " AND r.severity = %s"
            params.append(filter_severity)

        if search_query:
            query += " AND (c.title ILIKE %s OR c.location ILIKE %s OR r.findings ILIKE %s OR r.recommendation ILIKE %s OR CAST(r.complaint_id AS TEXT) ILIKE %s)"
            lq = f"%{search_query}%"
            params.extend([lq, lq, lq, lq, lq])

        query += " ORDER BY r.created_at DESC"

        cursor.execute(query, tuple(params))
        rows = cursor.fetchall()

        reports = []
        total_cost = 0.0
        high_count = 0

        for r in rows:
            cost = float(r[9]) if r[9] is not None else 0.0
            total_cost += cost
            if r[8] in ["High", "Critical"]:
                high_count += 1

            reports.append({
                "report_id": r[0],
                "complaint_id": r[1],
                "title": r[2],
                "category": r[3],
                "location": r[4],
                "status": r[5],
                "inspection_date": r[6],
                "findings": r[7],
                "severity": r[8],
                "estimated_cost": r[9],
                "recommendation": r[10],
                "inspection_image": r[11],
                "created_at": r[12]
            })

        cursor.close()
        connection.close()

        return render_template(
            "engineer_inspection_reports.html",
            first_name=session.get("first_name"),
            last_name=session.get("last_name"),
            reports=reports,
            total_reports=len(reports),
            high_count=high_count,
            total_cost=total_cost,
            current_severity=filter_severity,
            search_query=search_query
        )

    except Exception as e:
        print("Engineer Inspection Reports Error:", repr(e))
        if cursor:
            cursor.close()
        if connection:
            connection.close()
        return "Unable to load inspection reports."

@app.route("/engineer/inspections")
def engineer_inspection():

    if "user_id" not in session:
        return redirect(url_for("login"))

    if session.get("role") != "Engineer":
        return "Access denied."

    engineer_id = session.get("user_id")

    connection = None
    cursor = None

    try:
        connection = get_connection()
        cursor = connection.cursor()

        # Complaints assigned to this engineer
        cursor.execute(
            """
            SELECT
                c.complaint_id,
                c.title,
                c.description,
                c.category,
                c.location,
                c.priority,
                c.status,
                c.created_at,
                c.updated_at
            FROM complaints c
            WHERE c.assigned_engineer_id = %s
              AND c.status IN ('Assigned', 'Inspection')
            ORDER BY c.created_at DESC
            """,
            (engineer_id,)
        )

        inspection_rows = cursor.fetchall()

        inspections = []

        for row in inspection_rows:
            inspections.append({
                "complaint_id": row[0],
                "title": row[1],
                "description": row[2],
                "category": row[3],
                "location": row[4],
                "priority": row[5],
                "status": row[6],
                "created_at": row[7],
                "updated_at": row[8]
            })

        # Statistics
        under_inspection = sum(
            1 for item in inspections
            if item["status"] == "Inspection"
        )

        ready_count = sum(
            1 for item in inspections
            if item["status"] == "Assigned"
        )

        cursor.execute(
            """
            SELECT COUNT(*)
            FROM complaints
            WHERE assigned_engineer_id = %s
              AND status = 'Completed'
            """,
            (engineer_id,)
        )

        inspection_done = cursor.fetchone()[0]

        cursor.close()
        connection.close()

        return render_template(
            "engineer_inspection.html",
            first_name=session.get("first_name"),
            last_name=session.get("last_name"),
            inspections=inspections,
            under_inspection=under_inspection,
            ready_count=ready_count,
            inspection_done=inspection_done
        )

    except Exception as e:
        print("Engineer Inspection Error:", repr(e))

        if cursor:
            cursor.close()

        if connection:
            connection.close()

        return "Unable to load inspections."

@app.route("/engineer/start-inspection/<int:complaint_id>")
def engineer_start_inspection(complaint_id):

    if "user_id" not in session:
        return redirect(url_for("login"))

    if session.get("role") != "Engineer":
        return "Access denied."

    engineer_id = session.get("user_id")

    connection = None
    cursor = None

    try:
        connection = get_connection()
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT complaint_id, status 
            FROM complaints 
            WHERE complaint_id = %s AND assigned_engineer_id = %s
            """,
            (complaint_id, engineer_id)
        )
        row = cursor.fetchone()
        if not row:
            cursor.close()
            connection.close()
            flash("Complaint not found or it is not assigned to you.", "error")
            return redirect(url_for("engineer_assigned"))

        # If complaint status is Assigned or Pending, advance to Inspection
        if row[1] in ("Assigned", "Pending"):
            cursor.execute(
                """
                UPDATE complaints
                SET
                    status = 'Inspection',
                    updated_at = CURRENT_TIMESTAMP
                WHERE complaint_id = %s
                  AND assigned_engineer_id = %s
                """,
                (complaint_id, engineer_id)
            )
            connection.commit()
            flash(f"Inspection started for Complaint #{complaint_id}. Complaint is now Under Inspection.", "success")

        cursor.close()
        connection.close()

        return redirect(url_for("engineer_view_complaint", complaint_id=complaint_id))

    except Exception as e:
        print("Start Inspection Error:", repr(e))

        if connection:
            connection.rollback()

        if cursor:
            cursor.close()

        if connection:
            connection.close()

        return "Unable to start inspection."

@app.route("/engineer/update-inspection", methods=["POST"])
def engineer_update_inspection():
    if "user_id" not in session:
        return redirect(url_for("login"))

    flash("Engineers are not permitted to change complaint status. Only Department Heads can update or verify complaint status.", "error")
    return redirect(url_for("engineer_inspection"))

# =========================================================
# MAINTENANCE PORTAL
# =========================================================

@app.route("/maintenance-dashboard")
@app.route("/maintenance/dashboard")
def maintenance_dashboard():
    if "user_id" not in session:
        return redirect(url_for("login"))

    if session.get("role") != "Maintenance":
        return "Access denied."

    maintenance_id = session.get("user_id")

    connection = None
    cursor = None

    try:
        connection = get_connection()
        cursor = connection.cursor()

        # =========================================================
        # COUNTS
        # =========================================================
        cursor.execute(
            """
            SELECT
                COUNT(*),
                COUNT(*) FILTER (WHERE mw.work_status = 'Assigned' OR (mw.work_status IS NULL AND c.status = 'Maintenance')),
                COUNT(*) FILTER (WHERE mw.work_status = 'In Progress' OR (mw.work_status IS NULL AND c.status = 'In Progress')),
                COUNT(*) FILTER (WHERE mw.work_status = 'Completed' OR (mw.work_status IS NULL AND c.status = 'Completed')),
                COUNT(*) FILTER (WHERE c.priority = 'High' AND (COALESCE(mw.work_status, c.status) NOT IN ('Completed', 'Cancelled')))
            FROM complaints c
            LEFT JOIN maintenance_work mw
                ON mw.complaint_id = c.complaint_id AND mw.maintenance_staff_id = %s
            WHERE c.assigned_maintenance_id = %s OR mw.maintenance_staff_id = %s
            """,
            (maintenance_id, maintenance_id, maintenance_id)
        )
        counts = cursor.fetchone()
        total_count = counts[0] or 0
        assigned_count = counts[1] or 0
        in_progress = counts[2] or 0
        completed_count = counts[3] or 0
        high_priority = counts[4] or 0

        # =========================================================
        # RECENT WORK ORDERS
        # =========================================================
        cursor.execute(
            """
            SELECT
                c.complaint_id,
                c.title,
                c.location,
                c.category,
                c.priority,
                COALESCE(mw.work_status, c.status) AS work_status,
                COALESCE(mw.assigned_date, c.created_at::date) AS assigned_date,
                COALESCE(mw.created_at, c.created_at) AS created_at,
                mw.start_date,
                c.description,
                COALESCE(c.latitude, p.latitude) AS latitude,
                COALESCE(c.longitude, p.longitude) AS longitude,
                c.image_path
            FROM complaints c
            LEFT JOIN maintenance_work mw
                ON mw.complaint_id = c.complaint_id AND mw.maintenance_staff_id = %s
            LEFT JOIN complaints p
                ON c.parent_complaint_id = p.complaint_id
            WHERE c.assigned_maintenance_id = %s OR mw.maintenance_staff_id = %s
            ORDER BY
                CASE
                    WHEN mw.work_status = 'In Progress' OR c.status = 'In Progress' THEN 1
                    WHEN mw.work_status = 'Assigned' OR c.status = 'Maintenance' THEN 2
                    ELSE 3
                END,
                COALESCE(mw.created_at, c.created_at) DESC
            LIMIT 25
            """,
            (maintenance_id, maintenance_id, maintenance_id)
        )

        rows = cursor.fetchall()

        work_orders = []
        for row in rows:
            work_orders.append({
                "complaint_id": row[0],
                "title": row[1],
                "location": row[2],
                "work_type": row[3],
                "priority": row[4],
                "status": row[5],
                "assigned_date": row[6],
                "created_at": row[7],
                "start_date": row[8],
                "description": row[9],
                "latitude": float(row[10]) if row[10] is not None else None,
                "longitude": float(row[11]) if row[11] is not None else None,
                "image_path": row[12]
            })

        cursor.close()
        connection.close()

        return render_template(
            "maintenance_dashboard.html",
            first_name=session.get("first_name"),
            last_name=session.get("last_name"),
            assigned_count=assigned_count,
            in_progress=in_progress,
            completed_count=completed_count,
            high_priority=high_priority,
            total_count=total_count,
            work_orders=work_orders
        )

    except Exception as e:
        print("Maintenance Dashboard Error:", repr(e))
        if cursor:
            cursor.close()
        if connection:
            connection.close()
        return "Unable to load maintenance dashboard."


@app.route("/maintenance/work-orders")
def maintenance_work_orders():
    if "user_id" not in session:
        return redirect(url_for("login"))

    if session.get("role") != "Maintenance":
        return "Access denied."

    maintenance_id = session.get("user_id")
    filter_status = request.args.get("status", "all").strip()
    filter_priority = request.args.get("priority", "all").strip()
    search_query = request.args.get("q", "").strip()

    connection = None
    cursor = None

    try:
        connection = get_connection()
        cursor = connection.cursor()

        # Count summary for tabs
        cursor.execute(
            """
            SELECT
                COUNT(*),
                COUNT(*) FILTER (WHERE mw.work_status = 'Assigned'),
                COUNT(*) FILTER (WHERE mw.work_status = 'In Progress'),
                COUNT(*) FILTER (WHERE mw.work_status = 'Completed'),
                COUNT(*) FILTER (WHERE c.priority = 'High' AND mw.work_status NOT IN ('Completed', 'Cancelled'))
            FROM maintenance_work mw
            JOIN complaints c
                ON mw.complaint_id = c.complaint_id
            WHERE mw.maintenance_staff_id = %s
            """,
            (maintenance_id,)
        )
        counts = cursor.fetchone()
        total_count = counts[0] or 0
        assigned_count = counts[1] or 0
        in_progress_count = counts[2] or 0
        completed_count = counts[3] or 0
        high_priority_count = counts[4] or 0

        # Query work orders with dynamic filtering
        query = """
            SELECT
                c.complaint_id,
                c.title,
                c.description,
                c.location,
                c.category,
                c.priority,
                mw.work_status,
                mw.assigned_date,
                mw.start_date,
                mw.completion_date,
                mw.work_description,
                mw.remarks,
                mw.after_image,
                c.image_path,
                mw.maintenance_id
            FROM maintenance_work mw
            JOIN complaints c
                ON mw.complaint_id = c.complaint_id
            WHERE mw.maintenance_staff_id = %s
        """
        params = [maintenance_id]

        if filter_status and filter_status != "all":
            query += " AND mw.work_status = %s"
            params.append(filter_status)

        if filter_priority and filter_priority != "all":
            query += " AND c.priority = %s"
            params.append(filter_priority)

        if search_query:
            query += """ AND (
                c.title ILIKE %s OR 
                c.description ILIKE %s OR 
                c.location ILIKE %s OR 
                c.category ILIKE %s OR 
                CAST(c.complaint_id AS TEXT) ILIKE %s
            )"""
            wildcard = f"%{search_query}%"
            params.extend([wildcard, wildcard, wildcard, wildcard, wildcard])

        query += """
            ORDER BY 
                CASE 
                    WHEN mw.work_status = 'In Progress' THEN 1
                    WHEN mw.work_status = 'Assigned' THEN 2
                    WHEN mw.work_status = 'Completed' THEN 3
                    ELSE 4
                END,
                CASE WHEN c.priority = 'High' THEN 1 WHEN c.priority = 'Medium' THEN 2 ELSE 3 END,
                mw.assigned_date DESC NULLS LAST,
                c.complaint_id DESC
        """

        cursor.execute(query, tuple(params))
        rows = cursor.fetchall()

        work_orders = []
        for r in rows:
            work_orders.append({
                "complaint_id": r[0],
                "title": r[1],
                "description": r[2],
                "location": r[3],
                "category": r[4],
                "priority": r[5],
                "status": r[6],
                "assigned_date": r[7],
                "start_date": r[8],
                "completion_date": r[9],
                "work_description": r[10],
                "remarks": r[11],
                "after_image": r[12],
                "image_path": r[13],
                "maintenance_id": r[14]
            })

        cursor.close()
        connection.close()

        return render_template(
            "maintenance_work_orders.html",
            first_name=session.get("first_name"),
            last_name=session.get("last_name"),
            work_orders=work_orders,
            total_count=total_count,
            assigned_count=assigned_count,
            in_progress_count=in_progress_count,
            completed_count=completed_count,
            high_priority_count=high_priority_count,
            current_status=filter_status,
            current_priority=filter_priority,
            search_query=search_query
        )

    except Exception as e:
        print("Maintenance Work Orders Error:", repr(e))
        if cursor:
            cursor.close()
        if connection:
            connection.close()
        return "Unable to load maintenance work orders."


@app.route("/maintenance/start-work/<int:complaint_id>", methods=["GET", "POST"])
def maintenance_start_work(complaint_id):
    if "user_id" not in session:
        return redirect(url_for("login"))

    if session.get("role") != "Maintenance":
        return "Access denied."

    maintenance_staff_id = session.get("user_id")

    connection = None
    cursor = None

    try:
        connection = get_connection()
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT maintenance_id, work_status
            FROM maintenance_work
            WHERE complaint_id = %s
              AND maintenance_staff_id = %s
            ORDER BY maintenance_id DESC
            LIMIT 1
            """,
            (complaint_id, maintenance_staff_id)
        )
        row = cursor.fetchone()

        if not row:
            flash("Work order not found or not assigned to you.", "error")
            cursor.close()
            connection.close()
            return redirect(url_for("maintenance_work_orders"))

        maintenance_id, work_status = row

        if work_status == "Assigned":
            cursor.execute(
                """
                UPDATE maintenance_work
                SET work_status = 'In Progress',
                    start_date = CURRENT_DATE
                WHERE maintenance_id = %s
                """,
                (maintenance_id,)
            )

            cursor.execute(
                """
                UPDATE complaints
                SET status = 'In Progress',
                    updated_at = CURRENT_TIMESTAMP
                WHERE complaint_id = %s
                """,
                (complaint_id,)
            )

            connection.commit()
            flash(f"Work order #{complaint_id} has been started! Status is now In Progress.", "success")
        elif work_status == "In Progress":
            flash(f"Work order #{complaint_id} is already in progress.", "info")
        else:
            flash(f"Work order #{complaint_id} cannot be started from status '{work_status}'.", "warning")

        cursor.close()
        connection.close()

        referrer = request.referrer
        if referrer and ("maintenance" in referrer):
            return redirect(referrer)
        return redirect(url_for("maintenance_work_orders"))

    except Exception as e:
        print("Maintenance Start Work Error:", repr(e))
        if cursor:
            cursor.close()
        if connection:
            connection.close()
        flash("Failed to start work order.", "error")
        return redirect(url_for("maintenance_work_orders"))


@app.route("/maintenance/view/<int:complaint_id>", methods=["GET", "POST"])
def maintenance_view(complaint_id):
    if "user_id" not in session:
        return redirect(url_for("login"))

    if session.get("role") != "Maintenance":
        return "Access denied."

    maintenance_staff_id = session.get("user_id")

    connection = None
    cursor = None

    try:
        connection = get_connection()
        cursor = connection.cursor()

        # =========================================================
        # POST - Update Maintenance Work
        # =========================================================
        if request.method == "POST":
            new_status = request.form.get("status")

            allowed_statuses = ["In Progress", "Completed"]
            if new_status not in allowed_statuses:
                flash("Invalid work status requested.", "error")
                return redirect(url_for("maintenance_view", complaint_id=complaint_id))

            cursor.execute(
                """
                SELECT
                    maintenance_id,
                    work_status
                FROM maintenance_work
                WHERE complaint_id = %s
                  AND maintenance_staff_id = %s
                ORDER BY maintenance_id DESC
                LIMIT 1
                """,
                (complaint_id, maintenance_staff_id)
            )

            maintenance_work = cursor.fetchone()

            if not maintenance_work:
                flash("Maintenance work order not found.", "error")
                cursor.close()
                connection.close()
                return redirect(url_for("maintenance_dashboard"))

            maintenance_id = maintenance_work[0]
            current_work_status = maintenance_work[1]

            # -----------------------------------------------------
            # START WORK
            # -----------------------------------------------------
            if new_status == "In Progress":
                if current_work_status != "Assigned":
                    flash("This work order cannot be started because it is not currently Assigned.", "error")
                    cursor.close()
                    connection.close()
                    return redirect(url_for("maintenance_view", complaint_id=complaint_id))

                cursor.execute(
                    """
                    UPDATE maintenance_work
                    SET
                        work_status = 'In Progress',
                        start_date = CURRENT_DATE
                    WHERE maintenance_id = %s
                      AND maintenance_staff_id = %s
                      AND work_status = 'Assigned'
                    """,
                    (maintenance_id, maintenance_staff_id)
                )

                cursor.execute(
                    """
                    UPDATE complaints
                    SET
                        status = 'In Progress',
                        updated_at = CURRENT_TIMESTAMP
                    WHERE complaint_id = %s
                    """,
                    (complaint_id,)
                )

                connection.commit()
                flash(f"Work order #{complaint_id} started! Status is now In Progress.", "success")

            # -----------------------------------------------------
            # COMPLETE WORK
            # -----------------------------------------------------
            elif new_status == "Completed":
                if current_work_status != "In Progress":
                    flash("Work must be marked In Progress before it can be completed.", "error")
                    cursor.close()
                    connection.close()
                    return redirect(url_for("maintenance_view", complaint_id=complaint_id))

                work_description = request.form.get("work_description", "").strip()
                remarks = request.form.get("remarks", "").strip()
                completion_report = request.form.get("completion_report", "").strip()

                if not work_description:
                    flash("Please enter the work description of repairs completed.", "error")
                    cursor.close()
                    connection.close()
                    return redirect(url_for("maintenance_view", complaint_id=complaint_id))

                if not completion_report:
                    flash("Please enter the completion report summary.", "error")
                    cursor.close()
                    connection.close()
                    return redirect(url_for("maintenance_view", complaint_id=complaint_id))

                # Handle optional proof of completion image upload
                after_image_path = None
                if "after_image" in request.files:
                    image_file = request.files["after_image"]
                    if image_file and image_file.filename != "":
                        ext = image_file.filename.rsplit(".", 1)[-1].lower() if "." in image_file.filename else ""
                        if ext in ALLOWED_EXTENSIONS:
                            unique_fn = f"completion_{complaint_id}_{uuid.uuid4().hex[:8]}.{ext}"
                            maint_folder = os.path.join(app.config["UPLOAD_FOLDER"], "maintenance")
                            os.makedirs(maint_folder, exist_ok=True)
                            image_path_full = os.path.join(maint_folder, unique_fn)
                            image_file.save(image_path_full)
                            after_image_path = f"uploads/maintenance/{unique_fn}"

                if after_image_path:
                    cursor.execute(
                        """
                        UPDATE maintenance_work
                        SET
                            work_status = 'Completed',
                            completion_date = CURRENT_DATE,
                            work_description = %s,
                            remarks = %s,
                            completion_report = %s,
                            after_image = %s,
                            verification_status = 'Pending'
                        WHERE maintenance_id = %s
                          AND maintenance_staff_id = %s
                        """,
                        (
                            work_description,
                            remarks,
                            completion_report,
                            after_image_path,
                            maintenance_id,
                            maintenance_staff_id
                        )
                    )
                else:
                    cursor.execute(
                        """
                        UPDATE maintenance_work
                        SET
                            work_status = 'Completed',
                            completion_date = CURRENT_DATE,
                            work_description = %s,
                            remarks = %s,
                            completion_report = %s,
                            verification_status = 'Pending'
                        WHERE maintenance_id = %s
                          AND maintenance_staff_id = %s
                        """,
                        (
                            work_description,
                            remarks,
                            completion_report,
                            maintenance_id,
                            maintenance_staff_id
                        )
                    )

                # Keep complaint status as Maintenance awaiting Department Head verification
                cursor.execute(
                    """
                    UPDATE complaints
                    SET
                        updated_at = CURRENT_TIMESTAMP
                    WHERE complaint_id = %s
                    """,
                    (complaint_id,)
                )

                connection.commit()
                flash(f"Work order #{complaint_id} completion report submitted successfully! It is now awaiting Department Head verification.", "success")

            cursor.close()
            connection.close()

            return redirect(url_for("maintenance_view", complaint_id=complaint_id))

        # =========================================================
        # GET - Work Order Details
        # =========================================================
        cursor.execute(
            """
            SELECT
                c.complaint_id,
                c.title,
                c.description,
                c.category,
                c.location,
                c.priority,
                c.status,
                c.verification_status,
                c.created_at,
                c.updated_at,
                u.first_name,
                u.last_name,
                c.image_path
            FROM complaints c
            JOIN users u
                ON c.user_id = u.user_id
            WHERE c.complaint_id = %s
              AND (c.assigned_maintenance_id = %s OR EXISTS (
                  SELECT 1 FROM maintenance_work mw2
                  WHERE mw2.complaint_id = c.complaint_id
                    AND mw2.maintenance_staff_id = %s
              ))
            """,
            (complaint_id, maintenance_staff_id, maintenance_staff_id)
        )

        complaint = cursor.fetchone()

        if not complaint:
            cursor.close()
            connection.close()
            flash("Work order not found or not assigned to your account.", "error")
            return redirect(url_for("maintenance_dashboard"))

        cursor.execute(
            """
            SELECT
                maintenance_id,
                complaint_id,
                assigned_by,
                maintenance_staff_id,
                assigned_date,
                start_date,
                completion_date,
                work_status,
                work_description,
                remarks,
                before_image,
                after_image,
                completion_report,
                verified_by,
                verification_status,
                verified_at,
                created_at
            FROM maintenance_work
            WHERE complaint_id = %s
              AND maintenance_staff_id = %s
            ORDER BY maintenance_id DESC
            LIMIT 1
            """,
            (complaint_id, maintenance_staff_id)
        )

        maintenance_work = cursor.fetchone()

        if not maintenance_work:
            cursor.close()
            connection.close()
            flash("Maintenance work record not found.", "error")
            return redirect(url_for("maintenance_dashboard"))

        # Fetch engineer inspection report if available
        cursor.execute(
            """
            SELECT
                r.report_id,
                r.inspection_date,
                r.findings,
                r.severity,
                r.estimated_cost,
                r.recommendation,
                r.inspection_image,
                r.created_at,
                eng.first_name,
                eng.last_name,
                r.required_materials,
                r.status
            FROM inspection_reports r
            LEFT JOIN users eng ON r.engineer_id = eng.user_id
            WHERE r.complaint_id = %s
            ORDER BY r.report_id DESC
            LIMIT 1
            """,
            (complaint_id,)
        )
        insp_row = cursor.fetchone()
        inspection_report = None
        if insp_row:
            inspection_report = {
                "report_id": insp_row[0],
                "inspection_date": insp_row[1],
                "findings": insp_row[2],
                "severity": insp_row[3],
                "estimated_cost": insp_row[4],
                "recommendation": insp_row[5],
                "inspection_image": insp_row[6],
                "created_at": insp_row[7],
                "engineer_name": f"{insp_row[8] or ''} {insp_row[9] or ''}".strip(),
                "required_materials": insp_row[10],
                "status": insp_row[11] or "Pending Verification"
            }

        cursor.close()
        connection.close()

        return render_template(
            "maintenance_view.html",
            first_name=session.get("first_name"),
            last_name=session.get("last_name"),
            complaint=complaint,
            maintenance_work=maintenance_work,
            inspection_report=inspection_report
        )

    except Exception as e:
        print("Maintenance View Error:", repr(e))
        if cursor:
            cursor.close()
        if connection:
            connection.close()
        return "Unable to load work order."


@app.route("/maintenance/profile", methods=["GET", "POST"])
def maintenance_profile():
    if "user_id" not in session:
        return redirect(url_for("login"))

    if session.get("role") != "Maintenance":
        return "Access denied."

    user_id = session.get("user_id")

    connection = None
    cursor = None

    try:
        connection = get_connection()
        cursor = connection.cursor()

        if request.method == "POST":
            action = request.form.get("action")

            if action == "update_details":
                phone = request.form.get("phone", "").strip()
                address = request.form.get("address", "").strip()
                cursor.execute(
                    """
                    UPDATE users
                    SET phone = %s, address = %s
                    WHERE user_id = %s
                    """,
                    (phone, address, user_id)
                )
                connection.commit()
                flash("Contact details updated successfully.", "success")

            elif action == "change_password":
                current_password = request.form.get("current_password", "")
                new_password = request.form.get("new_password", "")
                confirm_password = request.form.get("confirm_password", "")

                cursor.execute("SELECT password_hash FROM users WHERE user_id = %s", (user_id,))
                user_pw = cursor.fetchone()

                if not user_pw or not check_password_hash(user_pw[0], current_password):
                    flash("Current password is incorrect.", "error")
                elif len(new_password) < 6:
                    flash("New password must be at least 6 characters long.", "error")
                elif new_password != confirm_password:
                    flash("New passwords do not match.", "error")
                else:
                    new_hash = generate_password_hash(new_password)
                    cursor.execute(
                        """
                        UPDATE users
                        SET password_hash = %s,
                            must_change_password = FALSE
                        WHERE user_id = %s
                        """,
                        (new_hash, user_id)
                    )
                    connection.commit()
                    flash("Password has been changed successfully.", "success")

            cursor.close()
            connection.close()
            return redirect(url_for("maintenance_profile"))

        # GET: retrieve profile and statistics
        cursor.execute(
            """
            SELECT
                u.user_id,
                u.first_name,
                u.last_name,
                u.email,
                u.phone,
                u.address,
                u.role,
                u.created_at,
                d.department_name
            FROM users u
            LEFT JOIN departments d
                ON u.department_id = d.department_id
            WHERE u.user_id = %s
            """,
            (user_id,)
        )
        user_data = cursor.fetchone()

        if not user_data:
            cursor.close()
            connection.close()
            return "User not found."

        # Performance / Task Metrics
        cursor.execute(
            """
            SELECT
                COUNT(*),
                COUNT(*) FILTER (WHERE work_status = 'Assigned'),
                COUNT(*) FILTER (WHERE work_status = 'In Progress'),
                COUNT(*) FILTER (WHERE work_status = 'Completed')
            FROM maintenance_work
            WHERE maintenance_staff_id = %s
            """,
            (user_id,)
        )
        stat_row = cursor.fetchone()
        total_tasks = stat_row[0] or 0
        assigned_tasks = stat_row[1] or 0
        in_progress_tasks = stat_row[2] or 0
        completed_tasks = stat_row[3] or 0
        resolution_rate = round((completed_tasks / total_tasks * 100), 1) if total_tasks > 0 else 100.0

        # Recent completed jobs history
        cursor.execute(
            """
            SELECT
                c.complaint_id,
                c.title,
                c.location,
                c.category,
                mw.completion_date,
                mw.work_description
            FROM maintenance_work mw
            JOIN complaints c
                ON mw.complaint_id = c.complaint_id
            WHERE mw.maintenance_staff_id = %s
              AND mw.work_status = 'Completed'
            ORDER BY mw.completion_date DESC NULLS LAST, mw.maintenance_id DESC
            LIMIT 5
            """,
            (user_id,)
        )
        recent_completed_rows = cursor.fetchall()
        recent_completed = []
        for rc in recent_completed_rows:
            recent_completed.append({
                "complaint_id": rc[0],
                "title": rc[1],
                "location": rc[2],
                "category": rc[3],
                "completion_date": rc[4],
                "work_description": rc[5]
            })

        cursor.close()
        connection.close()

        profile = {
            "user_id": user_data[0],
            "first_name": user_data[1],
            "last_name": user_data[2],
            "email": user_data[3],
            "phone": user_data[4] or "",
            "address": user_data[5] or "",
            "role": user_data[6],
            "created_at": user_data[7],
            "department_name": user_data[8] or "General Public Works"
        }

        return render_template(
            "maintenance_profile.html",
            first_name=session.get("first_name"),
            last_name=session.get("last_name"),
            profile=profile,
            total_tasks=total_tasks,
            assigned_tasks=assigned_tasks,
            in_progress_tasks=in_progress_tasks,
            completed_tasks=completed_tasks,
            resolution_rate=resolution_rate,
            recent_completed=recent_completed
        )

    except Exception as e:
        print("Maintenance Profile Error:", repr(e))
        if cursor:
            cursor.close()
        if connection:
            connection.close()
        return "Unable to load maintenance profile."

@app.route("/logout")
def logout():

    session.clear()

    return redirect(url_for("login"))

@app.route("/api/validate-complaint-image", methods=["POST"])
def api_validate_complaint_image():
    if "user_id" not in session:
        return jsonify({"success": False, "can_submit": False, "error": "Authentication required."}), 401

    image_file = request.files.get("image")
    if not image_file or not image_file.filename:
        return jsonify({"success": False, "can_submit": False, "error": "No image file provided."}), 400

    if not allowed_file(image_file.filename):
        return jsonify({"success": False, "can_submit": False, "error": "Invalid file format. Please upload JPG, JPEG, PNG, or WEBP."}), 400

    category = request.form.get("category", "").strip() or "Roads"
    title = request.form.get("title", "").strip()
    description = request.form.get("description", "").strip()

    try:
        val_result = validate_complaint_image(
            file_or_path=image_file,
            category=category,
            title=title,
            description=description,
            allow_uncertain=True
        )
        return jsonify(val_result)
    except Exception as e:
        print("API Image Validation Error:", repr(e))
        return jsonify({
            "success": False,
            "can_submit": False,
            "decision": "error",
            "error": "Mandatory image verification encountered an error. Validation cannot be bypassed."
        }), 500

@app.route("/register-complaint", methods=["GET", "POST"])
def register_complaint():
    if "user_id" not in session:
        return redirect(url_for("login"))

    if session.get("role") != "Citizen":
        return "Access denied."

    connection = None
    cursor = None

    try:
        connection = get_connection()
        cursor = connection.cursor()

        cursor.execute("""
            SELECT department_id, department_name
            FROM departments
            WHERE is_active = TRUE
            ORDER BY department_name
        """)
        department_rows = cursor.fetchall()
        departments = [{"department_id": r[0], "department_name": r[1]} for r in department_rows]

        if request.method == "GET":
            selected_dept = request.args.get("dept") or request.args.get("category") or ""
            initial_lat = request.args.get("lat") or ""
            initial_lng = request.args.get("lng") or ""
            auto_gps = request.args.get("auto_gps") or ""
            return_url = request.referrer or url_for("citizen_dashboard")
            cursor.close()
            connection.close()
            return render_template(
                "register_complaint.html",
                first_name=session.get("first_name"),
                departments=departments,
                selected_dept=selected_dept,
                initial_lat=initial_lat,
                initial_lng=initial_lng,
                auto_gps=auto_gps,
                return_url=return_url
            )

        # Get form data
        category = request.form.get("category", "").strip()
        title = request.form.get("title", "").strip()
        description = request.form.get("description", "").strip()
        location = request.form.get("location", "").strip()
        road_name = request.form.get("road_name", "").strip()
        landmark = request.form.get("landmark", "").strip()
        location_details = request.form.get("location_details", "").strip()

        latitude = request.form.get("latitude")
        longitude = request.form.get("longitude")
        return_url = request.form.get("return_url") or url_for("citizen_dashboard")

        if not category or not title or not description or not location:
            cursor.close()
            connection.close()
            return render_template(
                "register_complaint.html",
                first_name=session.get("first_name"),
                departments=departments,
                selected_dept=category,
                return_url=return_url,
                error="Please fill in all required fields."
            )

        latitude_value = float(latitude) if latitude else None
        longitude_value = float(longitude) if longitude else None

        # Image Handling - Strictly REQUIRED
        image = request.files.get("image")
        if not image or not image.filename or not allowed_file(image.filename):
            cursor.close()
            connection.close()
            return render_template(
                "register_complaint.html",
                first_name=session.get("first_name"),
                departments=departments,
                selected_dept=category,
                return_url=return_url,
                error="Please upload a valid photo (JPG, JPEG, or PNG) of the issue. A photo is required."
            )

        # Collect department-specific extra details
        dept_extra = []
        if category == "Roads":
            if request.form.get("dept_roads_issue_type"):
                dept_extra.append(f"• Specific Issue: {request.form.get('dept_roads_issue_type').strip()}")
            if request.form.get("dept_roads_size"):
                dept_extra.append(f"• Defect Size: {request.form.get('dept_roads_size').strip()}")
            if request.form.get("dept_roads_traffic_impact"):
                dept_extra.append(f"• Traffic Disruption: {request.form.get('dept_roads_traffic_impact').strip()}")
            if request.form.get("dept_roads_affected_length"):
                dept_extra.append(f"• Affected Length: {request.form.get('dept_roads_affected_length').strip()}")

        elif category == "Street Lights":
            if request.form.get("dept_lights_issue_type"):
                dept_extra.append(f"• Specific Issue: {request.form.get('dept_lights_issue_type').strip()}")
            if request.form.get("dept_lights_pole_number"):
                dept_extra.append(f"• Pole ID / Number: {request.form.get('dept_lights_pole_number').strip()}")
            if request.form.get("dept_lights_count"):
                dept_extra.append(f"• Affected Lights: {request.form.get('dept_lights_count').strip()}")
            if request.form.get("dept_lights_hazard"):
                dept_extra.append(f"• Electrical Hazard: {request.form.get('dept_lights_hazard').strip()}")

        elif category == "Water Supply":
            if request.form.get("dept_water_issue_type"):
                dept_extra.append(f"• Specific Issue: {request.form.get('dept_water_issue_type').strip()}")
            if request.form.get("dept_water_severity"):
                dept_extra.append(f"• Severity / Outage: {request.form.get('dept_water_severity').strip()}")
            if request.form.get("dept_water_duration"):
                dept_extra.append(f"• Disruption Duration: {request.form.get('dept_water_duration').strip()}")
            if request.form.get("dept_water_affected_homes"):
                dept_extra.append(f"• Households Affected: {request.form.get('dept_water_affected_homes').strip()}")

        elif category == "Drainage":
            if request.form.get("dept_drainage_issue_type"):
                dept_extra.append(f"• Specific Issue: {request.form.get('dept_drainage_issue_type').strip()}")
            if request.form.get("dept_drainage_flood_level"):
                dept_extra.append(f"• Waterlogging Level: {request.form.get('dept_drainage_flood_level').strip()}")
            if request.form.get("dept_drainage_hazard"):
                dept_extra.append(f"• Safety Hazard Alert: {request.form.get('dept_drainage_hazard').strip()}")

        elif category == "Waste Management":
            if request.form.get("dept_waste_issue_type"):
                dept_extra.append(f"• Specific Issue: {request.form.get('dept_waste_issue_type').strip()}")
            if request.form.get("dept_waste_type"):
                dept_extra.append(f"• Waste Category: {request.form.get('dept_waste_type').strip()}")
            if request.form.get("dept_waste_volume"):
                dept_extra.append(f"• Estimated Volume: {request.form.get('dept_waste_volume').strip()}")
            if request.form.get("dept_waste_blocking"):
                dept_extra.append(f"• Road / Drain Obstruction: {request.form.get('dept_waste_blocking').strip()}")

        elif category == "Parks":
            if request.form.get("dept_parks_issue_type"):
                dept_extra.append(f"• Specific Issue: {request.form.get('dept_parks_issue_type').strip()}")
            if request.form.get("dept_parks_area"):
                dept_extra.append(f"• Park Area / Section: {request.form.get('dept_parks_area').strip()}")
            if request.form.get("dept_parks_hazard"):
                dept_extra.append(f"• Visitor Safety Risk: {request.form.get('dept_parks_hazard').strip()}")

        full_description = description
        if dept_extra:
            full_description = f"{description}\n\n--- Department Specific Details ({category}) ---\n" + "\n".join(dept_extra)

        # -------------------------------------------------------------
        # MANDATORY SERVER-SIDE COMPUTER VISION VALIDATION
        # (CLIP Image Relevance + Pretrained AI-Image Detector)
        # -------------------------------------------------------------
        val_result = validate_complaint_image(
            file_or_path=image,
            category=category,
            title=title,
            description=full_description,
            allow_uncertain=True
        )

        if not val_result.get("can_submit", False):
            cursor.close()
            connection.close()
            error_reason = val_result.get("error")
            if not error_reason:
                if val_result.get("decision") == "block_unrelated":
                    rel_msg = val_result.get("relevance", {}).get("message", "The uploaded photo does not appear relevant to the selected category.")
                    error_reason = f"Image Verification Failed: {rel_msg} Please upload a clear photo showing the actual {category} issue."
                elif val_result.get("decision") == "block_ai":
                    ai_score = val_result.get("ai_detection", {}).get("ai_score", "High")
                    error_reason = f"Submission Blocked: The uploaded photo was detected as suspected AI-generated ({ai_score}% probability). Municipal policy requires authentic camera-captured photographs."
                else:
                    error_reason = "Image validation failed. Please upload a clear, authentic photo of the issue."

            return render_template(
                "register_complaint.html",
                first_name=session.get("first_name"),
                departments=departments,
                selected_dept=category,
                return_url=return_url,
                error=error_reason
            )

        original_filename = secure_filename(image.filename)
        unique_filename = f"{uuid.uuid4().hex}_{original_filename}"
        os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
        image.save(file_path)
        image_path = "uploads/" + unique_filename

        # Extract validation metrics
        is_ai_suspected = val_result.get("ai_detection", {}).get("is_ai_suspected", False) if val_result.get("ai_detection") else False
        ai_score = val_result.get("ai_detection", {}).get("ai_score", 0.0) if val_result.get("ai_detection") else 0.0
        relevance_score = val_result.get("relevance", {}).get("score", 0.0) if val_result.get("relevance") else 0.0
        relevance_status = val_result.get("relevance", {}).get("status", "Relevant") if val_result.get("relevance") else "Relevant"
        validation_json = json.dumps(val_result)

        # Match department_id
        dept_id = None
        for d in departments:
            if d["department_name"].strip().lower() == category.lower():
                dept_id = d["department_id"]
                break

        # Insert Complaint with Image Verification metrics
        cursor.execute(
            """
            INSERT INTO complaints(
                user_id, department_id, title, description, category,
                location, road_name, landmark, location_details, latitude, longitude,
                image_path, priority, status, verification_status,
                is_ai_suspected, ai_detection_score, image_relevance_score, image_relevance_status, image_validation_details,
                created_at, updated_at
            )
            VALUES
            (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            RETURNING complaint_id;
            """,
            (
                session["user_id"],
                dept_id,
                title,
                full_description,
                category,
                location,
                road_name or None,
                landmark or None,
                location_details or None,
                latitude_value,
                longitude_value,
                image_path,
                "Medium",
                "Pending",
                "Pending",
                is_ai_suspected,
                ai_score,
                relevance_score,
                relevance_status,
                validation_json
            )
        )
        complaint_id = cursor.fetchone()[0]

        # -------------------------------------------------------------
        # DUPLICATE DETECTION & LOGGING TO `complaint_similarity`
        # -------------------------------------------------------------
        similar_matches = find_similar_complaints(
            cursor, complaint_id, category, title, full_description, location, latitude_value, longitude_value,
            image_path=file_path
        )

        parent_complaint_id = None
        is_confirmed = False
        if similar_matches:
            top_match = similar_matches[0]
            canonical_parent_id = get_root_parent_complaint_id(cursor, top_match["complaint_id"]) or top_match["complaint_id"]
            similarity_score = top_match["similarity"]
            is_confirmed = top_match.get("is_confirmed_duplicate", False)
            is_photo_mismatch = top_match.get("is_photo_location_mismatch", False)

            if is_confirmed and not is_photo_mismatch:
                # User requirement: "if everything is similar that complaint will be confirm duplicate"
                cursor.execute(
                    """
                    UPDATE complaints
                    SET is_duplicate = TRUE,
                        parent_complaint_id = %s,
                        status = 'Duplicate'
                    WHERE complaint_id = %s;
                    """,
                    (canonical_parent_id, complaint_id)
                )
                decision_val = "Duplicate"
            else:
                # Link to canonical root parent as candidate for administrator review
                cursor.execute(
                    """
                    UPDATE complaints
                    SET parent_complaint_id = %s,
                        is_duplicate = FALSE
                    WHERE complaint_id = %s;
                    """,
                    (canonical_parent_id, complaint_id)
                )
                decision_val = "Pending"

            # Save similarity details for audit and admin verification
            cursor.execute(
                """
                INSERT INTO complaint_similarity
                (
                    complaint_id,
                    similar_complaint_id,
                    text_similarity,
                    location_distance_m,
                    image_similarity,
                    combined_score,
                    decision,
                    created_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP);
                """,
                (
                    complaint_id,
                    canonical_parent_id,
                    # Database stores TF-IDF text similarity between 0 and 1
                    top_match["text_similarity"] / 100,
                    (top_match["distance_km"] * 1000)
                    if top_match["distance_km"] is not None
                    else None,
                    # Photo similarity between 0 and 1
                    (top_match["image_similarity"] / 100)
                    if top_match["image_similarity"] is not None
                    else None,
                    # Combined score between 0 and 1
                    similarity_score / 100,
                    decision_val
                )
            )

        connection.commit()
        cursor.close()
        connection.close()

        if is_confirmed:
            success_msg = (
                f"Complaint submitted successfully! Your report (ID #{complaint_id}) "
                f"has been matched with existing Complaint #{top_match['complaint_id']} "
                f"and automatically confirmed as a linked duplicate."
            )
        else:
            success_msg = f"Complaint submitted successfully! Your Complaint ID is #{complaint_id}."

        return render_template(
            "register_complaint.html",
            first_name=session.get("first_name"),
            departments=departments,
            complaint_id=complaint_id,
            success=success_msg,
            similar_complaint_id=top_match["complaint_id"] if similar_matches else None,
            validation_result=val_result,
            return_url=return_url
        )

    except Exception as e:
        print("Complaint Registration Error:", repr(e))
        if connection:
            connection.rollback()
            connection.close()
        return render_template("register_complaint.html", first_name=session.get("first_name"), departments=[], error="Something went wrong.")

@app.route('/track-complaints')
def track_complaints():

    if 'user_id' not in session or session.get('role') != 'Citizen':
        return redirect(url_for('login'))

    connection = None
    cursor = None

    try:
        connection = get_connection()
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                c.complaint_id,
                c.title,
                c.description,
                c.category,
                c.location,
                c.priority,
                c.status,
                c.verification_status,
                c.created_at,
                c.updated_at,
                d.department_name,
                c.is_duplicate,
                c.parent_complaint_id,
                p.title AS parent_title,
                p.status AS parent_status
            FROM complaints c
            LEFT JOIN departments d
                ON c.department_id = d.department_id
            LEFT JOIN complaints p
                ON c.parent_complaint_id = p.complaint_id
            WHERE c.user_id = %s
            ORDER BY c.created_at DESC
            """,
            (session['user_id'],)
        )

        complaint_rows = cursor.fetchall()

        complaints = []

        for row in complaint_rows:
            complaints.append({
                "complaint_id": row[0],
                "title": row[1],
                "description": row[2],
                "category": row[3],
                "location": row[4],
                "priority": row[5],
                "status": row[6],
                "verification_status": row[7],
                "created_at": row[8],
                "updated_at": row[9],
                "department_name": row[10],
                "is_duplicate": row[11] or False,
                "parent_complaint_id": row[12],
                "parent_title": row[13],
                "parent_status": row[14]
            })

        cursor.close()
        connection.close()

        return render_template(
            'track_complaints.html',
            first_name=session.get('first_name'),
            complaints=complaints
        )

    except Exception as e:

        print("Track Complaints Error:", repr(e))

        if cursor:
            cursor.close()

        if connection:
            connection.close()

        return render_template(
            'track_complaints.html',
            first_name=session.get('first_name'),
            complaints=[]
        )

@app.route('/notifications')
def notifications():
    if 'user_id' not in session or session.get('role') != 'Citizen':
        return redirect(url_for('login'))

    connection = None
    cursor = None

    try:
        connection = get_connection()
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                c.complaint_id,
                c.category,
                c.status,
                c.created_at,
                c.updated_at
            FROM complaints c
            WHERE c.user_id = %s
            ORDER BY c.updated_at DESC NULLS LAST, c.created_at DESC
            """,
            (session['user_id'],)
        )

        complaint_rows = cursor.fetchall()

        notifications_list = []

        for row in complaint_rows:
            complaint_id = row[0]
            category = row[1]
            status = row[2]
            created_at = row[3]
            updated_at = row[4]

            if status == "Assigned":
                message = f"Your complaint #{complaint_id} has been assigned to the concerned department."

            elif status == "Inspection":
                message = f"Your complaint #{complaint_id} is currently under inspection."

            elif status == "Maintenance":
                message = f"Work has started on your complaint #{complaint_id} ({category})."

            elif status == "Completed":
                message = f"Your complaint #{complaint_id} has been resolved successfully."

            elif status == "Rejected":
                message = f"Your complaint #{complaint_id} has been rejected after verification."

            elif status == "Verified":
                message = f"Your complaint #{complaint_id} has been verified and is being processed."

            elif status == "Under Verification":
                message = f"Your complaint #{complaint_id} is currently under verification."

            else:
                message = f"Your complaint #{complaint_id} is currently {status}."

            notifications_list.append({
                "complaint_id": complaint_id,
                "message": message,
                "status": status,
                "created_at": created_at,
                "updated_at": updated_at
            })

        cursor.close()
        connection.close()

        return render_template(
            'notifications.html',
            first_name=session.get('first_name'),
            notifications=notifications_list
        )

    except Exception as e:
        print("Notifications Error:", repr(e))

        if cursor:
            cursor.close()

        if connection:
            connection.close()

        return render_template(
            'notifications.html',
            first_name=session.get('first_name'),
            notifications=[]
        )


@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'user_id' not in session or session.get('role') != 'Citizen':
        return redirect(url_for('login'))

    connection = None
    cursor = None

    try:
        connection = get_connection()
        cursor = connection.cursor()

        user_id = session['user_id']

        # Update profile
        if request.method == 'POST':
            first_name = request.form.get('first_name', '').strip()
            last_name = request.form.get('last_name', '').strip()
            phone = request.form.get('phone', '').strip()
            address = request.form.get('address', '').strip()

            if not first_name:
                return render_template(
                    'profile.html',
                    first_name=first_name,
                    last_name=last_name,
                    email=session.get('email', ''),
                    phone=phone,
                    address=address,
                    error="First name is required."
                )

            if phone and not is_valid_phone(phone):
                return render_template(
                    'profile.html',
                    first_name=first_name,
                    last_name=last_name,
                    email=session.get('email', ''),
                    phone=phone,
                    address=address,
                    error="Please enter a valid phone number (10 to 15 digits)."
                )

            cursor.execute(
                """
                UPDATE users
                SET first_name = %s,
                    last_name = %s,
                    phone = %s,
                    address = %s
                WHERE user_id = %s
                """,
                (
                    first_name,
                    last_name,
                    phone,
                    address,
                    user_id
                )
            )

            connection.commit()

            # Keep session information updated
            session['first_name'] = first_name
            session['last_name'] = last_name

            # Fetch the updated profile from the database
            cursor.execute(
                """
                SELECT
                    first_name,
                    last_name,
                    email,
                    phone,
                    address
                FROM users
                WHERE user_id = %s
                """,
                (user_id,)
            )

            updated_row = cursor.fetchone()

            return render_template(
                'profile.html',
                first_name=updated_row[0],
                last_name=updated_row[1],
                email=updated_row[2],
                phone=updated_row[3],
                address=updated_row[4],
                success="Profile updated successfully."
            )

        # Load current profile information
        cursor.execute(
            """
            SELECT
                first_name,
                last_name,
                email,
                phone,
                address
            FROM users
            WHERE user_id = %s
            """,
            (user_id,)
        )

        row = cursor.fetchone()

        if not row:
            return "User not found."

        first_name = row[0]
        last_name = row[1]
        email = row[2]
        phone = row[3]
        address = row[4]

        cursor.close()
        connection.close()

        return render_template(
            'profile.html',
            first_name=first_name,
            last_name=last_name,
            email=email,
            phone=phone,
            address=address
        )

    except Exception as e:
        print("Profile Error:", repr(e))

        if connection:
            connection.rollback()

        if cursor:
            cursor.close()

        if connection:
            connection.close()

        return render_template(
            'profile.html',
            first_name=session.get('first_name', ''),
            last_name=session.get('last_name', ''),
            email=session.get('email', ''),
            phone='',
            address='',
            error="Unable to update profile."
        )

@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    if 'user_id' not in session or session.get('role') != 'Citizen':
        return redirect(url_for('login'))

    connection = None
    cursor = None

    try:
        connection = get_connection()
        cursor = connection.cursor()

        user_id = session['user_id']

        # Handle feedback submission
        if request.method == 'POST':

            complaint_id = request.form.get('complaint_id')
            rating = request.form.get('rating')
            comments = request.form.get('feedback', '').strip()

            # Convert empty complaint ID to NULL
            if not complaint_id:
                complaint_id = None
            else:
                try:
                    complaint_id = int(complaint_id)
                except ValueError:
                    complaint_id = None

            # Validate rating
            try:
                rating = int(rating)
            except (TypeError, ValueError):
                rating = 0

            if rating < 1 or rating > 5:
                return render_template(
                    'feedback.html',
                    first_name=session.get('first_name'),
                    complaints=[],
                    error="Please select a rating from 1 to 5 stars."
                )

            if not comments:
                return render_template(
                    'feedback.html',
                    first_name=session.get('first_name'),
                    complaints=[],
                    error="Please enter your feedback."
                )

            # If a complaint is selected, verify that it belongs to this citizen
            # and has been completed or closed.
            if complaint_id is not None:

                cursor.execute(
                    """
                    SELECT complaint_id
                    FROM complaints
                    WHERE complaint_id = %s
                      AND user_id = %s
                      AND status IN ('Completed', 'Closed')
                    """,
                    (complaint_id, user_id)
                )

                complaint = cursor.fetchone()

                if not complaint:
                    return render_template(
                        'feedback.html',
                        first_name=session.get('first_name'),
                        complaints=[],
                        error="You can only give feedback for your completed complaints."
                    )

                # Check whether feedback already exists
                cursor.execute(
                    """
                    SELECT feedback_id
                    FROM feedback
                    WHERE complaint_id = %s
                      AND user_id = %s
                    LIMIT 1
                    """,
                    (complaint_id, user_id)
                )

                existing_feedback = cursor.fetchone()

                if existing_feedback:
                    return render_template(
                        'feedback.html',
                        first_name=session.get('first_name'),
                        complaints=[],
                        error="You have already submitted feedback for this complaint."
                    )

            # Insert feedback
            cursor.execute(
                """
                INSERT INTO feedback
                    (complaint_id, user_id, rating, comments)
                VALUES
                    (%s, %s, %s, %s)
                """,
                (
                    complaint_id,
                    user_id,
                    rating,
                    comments
                )
            )

            connection.commit()

            # Reload completed complaints for the page
            cursor.execute(
                """
                SELECT
                    c.complaint_id,
                    c.category,
                    c.location,
                    c.status
                FROM complaints c
                WHERE c.user_id = %s
                  AND c.status IN ('Completed', 'Closed')
                ORDER BY c.created_at DESC
                """,
                (user_id,)
            )

            complaint_rows = cursor.fetchall()

            complaints = []

            for row in complaint_rows:
                complaints.append({
                    "complaint_id": row[0],
                    "category": row[1],
                    "location": row[2],
                    "status": row[3]
                })

            cursor.close()
            connection.close()

            return render_template(
                'feedback.html',
                first_name=session.get('first_name'),
                complaints=complaints,
                success="Your feedback has been submitted successfully."
            )

        # Load completed/closed complaints for the dropdown
        cursor.execute(
            """
            SELECT
                c.complaint_id,
                c.category,
                c.location,
                c.status
            FROM complaints c
            WHERE c.user_id = %s
              AND c.status IN ('Completed', 'Closed')
            ORDER BY c.created_at DESC
            """,
            (user_id,)
        )

        complaint_rows = cursor.fetchall()

        complaints = []

        for row in complaint_rows:
            complaints.append({
                "complaint_id": row[0],
                "category": row[1],
                "location": row[2],
                "status": row[3]
            })

        cursor.close()
        connection.close()

        return render_template(
            'feedback.html',
            first_name=session.get('first_name'),
            complaints=complaints
        )

    except Exception as e:
        print("Feedback Error:", repr(e))

        if connection:
            connection.rollback()

        if cursor:
            cursor.close()

        if connection:
            connection.close()

        return render_template(
            'feedback.html',
            first_name=session.get('first_name'),
            complaints=[],
            error="Unable to submit feedback."
        )

# ==========================================
# ADMIN - ALL COMPLAINTS
# ==========================================

@app.route("/admin/complaints", methods=["GET"])
def admin_complaints():

    # -----------------------------------
    # Check login
    # -----------------------------------

    if "user_id" not in session:
        return redirect(url_for("login"))

    # -----------------------------------
    # Only Administrator
    # -----------------------------------

    if session.get("role") != "Administrator":
        return "Access denied."

    connection = None
    cursor = None

    try:
        # -----------------------------------
        # Get filters
        # -----------------------------------

        search_query = request.args.get("search", "").strip()
        status_filter = request.args.get("status", "").strip()
        category_filter = request.args.get("category", "").strip()
        complaint_id = request.args.get("complaint_id")

        # -----------------------------------
        # Database connection
        # -----------------------------------

        connection = get_connection()
        cursor = connection.cursor()

        # If complaint_id is provided, show complaint details
        if complaint_id:
            try:
                complaint_id = int(complaint_id)
            except ValueError:
                return "Invalid complaint ID."

            query = """
                        SELECT
                            c.complaint_id,
                            c.title,
                            c.description,
                            c.category,
                            c.location,
                            c.latitude,
                            c.longitude,
                            c.image_path,
                            c.priority,
                            c.status,
                            c.verification_status,
                            c.department_id,
                            c.created_at,
                            c.updated_at,
                            u.first_name || ' ' || u.last_name AS citizen_name,
                            u.email AS citizen_email,
                            u.phone AS citizen_phone,
                            d.department_name,
                            c.is_duplicate,
                            c.parent_complaint_id,
                            c.is_ai_suspected,
                            c.ai_detection_score,
                            c.image_relevance_score,
                            c.image_relevance_status,
                            c.image_validation_details
                        FROM complaints c
                        JOIN users u ON c.user_id = u.user_id
                        LEFT JOIN departments d
                            ON c.department_id = d.department_id
                        WHERE c.complaint_id = %s
                    """

            cursor.execute(query, (complaint_id,))
            row = cursor.fetchone()

            if not row:
                cursor.close()
                connection.close()
                return "Complaint not found."

            complaint = {
                "complaint_id": row[0],
                "title": row[1],
                "description": row[2],
                "category": row[3],
                "location": row[4],
                "latitude": row[5],
                "longitude": row[6],
                "image_path": row[7],
                "priority": row[8],
                "status": row[9],
                "verification_status": row[10],
                "department_id": row[11],
                "created_at": row[12],
                "updated_at": row[13],
                "citizen_name": row[14],
                "citizen_email": row[15],
                "citizen_phone": row[16],
                "department_name": row[17],
                "is_duplicate": row[18] or False,
                "parent_complaint_id": row[19],
                "is_ai_suspected": bool(row[20]) if row[20] is not None else False,
                "ai_detection_score": round(float(row[21]), 1) if row[21] is not None else None,
                "image_relevance_score": round(float(row[22]), 1) if row[22] is not None else None,
                "image_relevance_status": row[23] or "Pending",
                "image_validation_details": row[24]
            }

            # Fetch any child duplicate reports linked to this complaint
            cursor.execute(
                """
                SELECT 
                    c.complaint_id,
                    c.title,
                    c.category,
                    c.location,
                    c.status,
                    c.created_at,
                    u.first_name || ' ' || u.last_name AS citizen_name,
                    ROUND(COALESCE(cs.combined_score, 0) * 100, 1) AS similarity_score
                FROM complaints c
                JOIN users u ON c.user_id = u.user_id
                LEFT JOIN complaint_similarity cs 
                    ON cs.complaint_id = c.complaint_id AND cs.similar_complaint_id = %s
                WHERE c.parent_complaint_id = %s AND (c.is_duplicate IS TRUE OR c.status = 'Duplicate')
                ORDER BY c.created_at DESC
                """,
                (complaint_id, complaint_id)
            )
            linked_rows = cursor.fetchall()
            linked_duplicates = []
            for lr in linked_rows:
                linked_duplicates.append({
                    "complaint_id": lr[0],
                    "title": lr[1],
                    "category": lr[2],
                    "location": lr[3],
                    "status": lr[4],
                    "created_at": lr[5],
                    "citizen_name": lr[6],
                    "similarity_score": lr[7]
                })

            parent_complaint = None
            if row[18] and row[19]:
                cursor.execute(
                    """
                    SELECT c.complaint_id, c.title, c.category, c.status, c.created_at,
                           u.first_name || ' ' || u.last_name AS citizen_name
                    FROM complaints c
                    JOIN users u ON c.user_id = u.user_id
                    WHERE c.complaint_id = %s
                    """,
                    (row[19],)
                )
                p_row = cursor.fetchone()
                if p_row:
                    parent_complaint = {
                        "complaint_id": p_row[0],
                        "title": p_row[1],
                        "category": p_row[2],
                        "status": p_row[3],
                        "created_at": p_row[4],
                        "citizen_name": p_row[5]
                    }

            cursor.close()
            connection.close()

            return render_template(
                "admin_complaint_details.html",
                complaint=complaint,
                linked_duplicates=linked_duplicates,
                parent_complaint=parent_complaint
            )

        # -----------------------------------
        # Base query
        # -----------------------------------

        query = """
            SELECT
                c.complaint_id,
                c.title,
                c.description,
                c.category,
                c.location,
                c.latitude,
                c.longitude,
                c.image_path,
                c.priority,
                c.status,
                c.verification_status,
                c.department_id,
                c.created_at,
                c.updated_at,
                u.first_name || ' ' || u.last_name AS citizen_name,
                c.is_duplicate,
                c.parent_complaint_id

            FROM complaints c

            JOIN users u
                ON c.user_id = u.user_id

            WHERE 1 = 1
        """

        params = []

        # -----------------------------------
        # Search
        # -----------------------------------

        if search_query:

            query += """
                AND (
                    CAST(c.complaint_id AS TEXT) ILIKE %s
                    OR c.title ILIKE %s
                    OR c.location ILIKE %s
                    OR c.category ILIKE %s
                )
            """

            search_term = f"%{search_query}%"

            params.extend([
                search_term,
                search_term,
                search_term,
                search_term
            ])

        # -----------------------------------
        # Status filter
        # -----------------------------------

        if status_filter:

            query += """
                AND LOWER(c.status) = LOWER(%s)
            """

            params.append(status_filter)

        # -----------------------------------
        # Category filter
        # -----------------------------------

        if category_filter:

            query += """
                AND LOWER(c.category) = LOWER(%s)
            """

            params.append(category_filter)

        # -----------------------------------
        # Newest first
        # -----------------------------------

        query += """
            ORDER BY c.created_at DESC
        """

        # -----------------------------------
        # Execute
        # -----------------------------------

        cursor.execute(query, params)

        rows = cursor.fetchall()

        # -----------------------------------
        # Convert tuples to dictionaries
        # -----------------------------------

        complaints = []

        for row in rows:

            complaints.append({
                "complaint_id": row[0],
                "title": row[1],
                "description": row[2],
                "category": row[3],
                "location": row[4],
                "latitude": row[5],
                "longitude": row[6],
                "image_path": row[7],
                "priority": row[8],
                "status": row[9],
                "verification_status": row[10],
                "department_id": row[11],
                "created_at": row[12],
                "updated_at": row[13],
                "citizen_name": row[14],
                "is_duplicate": row[15] or False,
                "parent_complaint_id": row[16]
            })

        # -----------------------------------
        # Close database
        # -----------------------------------

        cursor.close()
        connection.close()

        # -----------------------------------
        # Send to HTML
        # -----------------------------------

        return render_template(
            "admin_complaints.html",
            complaints=complaints,
            search=search_query,
            selected_status=status_filter,
            selected_category=category_filter
        )

    except Exception as e:

        print("Admin Complaints Error:", e)

        if cursor:
            cursor.close()

        if connection:
            connection.close()

        return f"Admin Complaints Error: {e}"

@app.route('/admin/verify-complaints', methods=["GET", "POST"])
def verify_complaints():

    # Check administrator login
    if "user_id" not in session:
        return redirect(url_for("login"))

    if session.get("role") != "Administrator":
        return redirect(url_for("login"))

    connection = None
    cursor = None

    try:
        connection = get_connection()
        cursor = connection.cursor()

        # -----------------------------------
        # VERIFY / REJECT COMPLAINT
        # -----------------------------------
        if request.method == "POST":

            complaint_id = request.form.get("complaint_id")
            action = request.form.get("action")

            if not complaint_id:
                return "Complaint ID is required."

            try:
                complaint_id = int(complaint_id)
            except ValueError:
                return "Invalid complaint ID."

            # -----------------------------------
            # VERIFY COMPLAINT (INDEPENDENT)
            # -----------------------------------
            if action == "verify":

                # Get complaint category
                cursor.execute(
                    """
                    SELECT category
                    FROM complaints
                    WHERE complaint_id = %s
                    """,
                    (complaint_id,)
                )

                complaint_row = cursor.fetchone()

                if not complaint_row:
                    return "Complaint not found."

                category = complaint_row[0]

                # Match category directly to department
                department_id = None

                if category:
                    cursor.execute(
                        """
                        SELECT department_id
                        FROM departments
                        WHERE LOWER(TRIM(department_name)) = LOWER(TRIM(%s))
                        LIMIT 1
                        """,
                        (category,)
                    )

                    department_row = cursor.fetchone()

                    if department_row:
                        department_id = department_row[0]

                # Department must exist
                if department_id is None:
                    connection.rollback()
                    return render_template(
                        "department_assignment_error.html",
                        message=(
                            f'No department is available for the category "{category}". '
                            'Please add the appropriate department before verifying this complaint.'
                        )
                    )

                # Verify complaint as an independent complaint + assign department
                cursor.execute(
                    """
                    UPDATE complaints
                    SET
                        department_id = %s,
                        verification_status = %s,
                        status = %s,
                        is_duplicate = FALSE,
                        parent_complaint_id = NULL,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE complaint_id = %s
                    """,
                    (
                        department_id,
                        "Verified",
                        "Pending",
                        complaint_id
                    )
                )

                # If there were any pending similarity candidate reviews, mark them as Separate
                cursor.execute(
                    """
                    UPDATE complaint_similarity
                    SET 
                        decision = 'Separate',
                        reviewed_by = %s,
                        reviewed_at = CURRENT_TIMESTAMP
                    WHERE complaint_id = %s AND decision = 'Pending';
                    """,
                    (session.get("user_id"), complaint_id)
                )

                # Send verification notification to citizen
                cursor.execute(
                    """
                    INSERT INTO notifications (user_id, complaint_id, message, is_read, created_at)
                    SELECT 
                        user_id, 
                        complaint_id, 
                        'Your complaint #' || %s || ' has been verified and assigned to the ' || COALESCE((SELECT department_name FROM departments WHERE department_id = %s), 'department') || ' for action.',
                        FALSE, 
                        CURRENT_TIMESTAMP
                    FROM complaints 
                    WHERE complaint_id = %s;
                    """,
                    (complaint_id, department_id, complaint_id)
                )
                flash(f"Complaint #{complaint_id} verified and assigned to department successfully.", "success")

            # -----------------------------------
            # REJECT COMPLAINT & APPLY 3-STRIKE RULE
            # -----------------------------------
            elif action == "reject":

                # Find submitting user ID
                cursor.execute(
                    "SELECT user_id FROM complaints WHERE complaint_id = %s;",
                    (complaint_id,)
                )
                user_row = cursor.fetchone()

                if user_row:
                    user_id = user_row[0]

                    # 1. Update complaint status
                    cursor.execute(
                        """
                        UPDATE complaints
                        SET
                            verification_status = %s,
                            status = %s,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE complaint_id = %s
                        """,
                        ("Rejected", "Rejected", complaint_id)
                    )

                    # 2. Update complaint_similarity if pending
                    cursor.execute(
                        """
                        UPDATE complaint_similarity
                        SET 
                            decision = 'Rejected',
                            reviewed_by = %s,
                            reviewed_at = CURRENT_TIMESTAMP
                        WHERE complaint_id = %s AND decision = 'Pending';
                        """,
                        (session.get("user_id"), complaint_id)
                    )

                    # 3. Increment user strike count
                    cursor.execute(
                        """
                        UPDATE users
                        SET strike_count = COALESCE(strike_count, 0) + 1
                        WHERE user_id = %s
                        RETURNING strike_count;
                        """,
                        (user_id,)
                    )
                    strike_count = cursor.fetchone()[0]

                    # 4. Ban user on 3rd strike
                    if strike_count >= 3:
                        cursor.execute(
                            """
                            UPDATE users
                            SET is_active = FALSE
                            WHERE user_id = %s;
                            """,
                            (user_id,)
                        )
                        flash(f"Complaint #{complaint_id} rejected. Citizen account suspended due to 3 strikes.", "error")
                    else:
                        flash(f"Complaint #{complaint_id} rejected. Strike issued to citizen ({strike_count}/3).", "warning")

            # -----------------------------------
            # CONFIRM DUPLICATE (PRESERVE ORIGINAL, LINK DUPLICATE, HALT DOWNSTREAM)
            # -----------------------------------
            elif action in ("confirm_duplicate", "link"):
                parent_id = request.form.get("parent_id") or request.form.get("parent_complaint_id")
                if parent_id:
                    try:
                        parent_id = int(parent_id)
                    except ValueError:
                        return "Invalid parent complaint ID."

                    # User requirement: guarantee duplicate merges directly to canonical root parent (e.g. #62 not #63)
                    canonical_parent_id = get_root_parent_complaint_id(cursor, parent_id)
                    if canonical_parent_id:
                        parent_id = canonical_parent_id

                    # 1. Update duplicate record:
                    # - Link to main complaint (parent_complaint_id = parent_id)
                    # - Set is_duplicate = TRUE
                    # - Set status = 'Duplicate'
                    # - Set verification_status = 'Verified'
                    # - Inherit department_id from parent complaint so reporting is consistent
                    cursor.execute(
                        """
                        UPDATE complaints
                        SET 
                            parent_complaint_id = %s,
                            is_duplicate = TRUE,
                            status = 'Duplicate',
                            verification_status = 'Verified',
                            department_id = COALESCE(department_id, (SELECT department_id FROM complaints WHERE complaint_id = %s)),
                            updated_at = CURRENT_TIMESTAMP
                        WHERE complaint_id = %s;
                        """,
                        (parent_id, parent_id, complaint_id)
                    )

                    # 2. Log administrator decision in complaint_similarity (pointing to root parent)
                    cursor.execute(
                        """
                        UPDATE complaint_similarity
                        SET 
                            similar_complaint_id = %s,
                            decision = 'Duplicate',
                            reviewed_by = %s,
                            reviewed_at = CURRENT_TIMESTAMP
                        WHERE complaint_id = %s;
                        """,
                        (parent_id, session.get("user_id"), complaint_id)
                    )

                    # 3. Notify citizen who filed duplicate report
                    cursor.execute(
                        """
                        INSERT INTO notifications (user_id, complaint_id, message, is_read, created_at)
                        SELECT 
                            user_id, 
                            complaint_id, 
                            'Your complaint #' || %s || ' was reviewed and confirmed as a duplicate of Main Complaint #' || %s || '. The municipal team is actively resolving the main complaint, and your report is linked to its progress.',
                            FALSE, 
                            CURRENT_TIMESTAMP
                        FROM complaints 
                        WHERE complaint_id = %s;
                        """,
                        (complaint_id, parent_id, complaint_id)
                    )

                    flash(f"Complaint #{complaint_id} confirmed as duplicate of Main Complaint #{parent_id}. Original record #{parent_id} preserved as main issue; duplicate will NOT proceed independently to department assignment.", "success")

            # -----------------------------------
            # KEEP SEPARATE (MARK AS DISTINCT INDEPENDENT COMPLAINT)
            # -----------------------------------
            elif action in ("keep_separate", "unlink"):
                cursor.execute(
                    """
                    UPDATE complaints
                    SET 
                        parent_complaint_id = NULL,
                        is_duplicate = FALSE,
                        status = 'Pending',
                        updated_at = CURRENT_TIMESTAMP
                    WHERE complaint_id = %s;
                    """,
                    (complaint_id,)
                )

                cursor.execute(
                    """
                    UPDATE complaint_similarity
                    SET 
                        decision = 'Separate',
                        reviewed_by = %s,
                        reviewed_at = CURRENT_TIMESTAMP
                    WHERE complaint_id = %s;
                    """,
                    (session.get("user_id"), complaint_id)
                )

                flash(f"Complaint #{complaint_id} marked as an independent complaint. You may now verify it for department dispatch.", "info")

            else:
                return "Invalid action."

            connection.commit()

            # Redirect back to refresh the list after POST
            return redirect(url_for("verify_complaints"))

        # -----------------------------------
        # GET COMPLAINTS WAITING FOR VERIFICATION / DUPLICATE REVIEW
        # -----------------------------------
        tab_filter = request.args.get('tab', 'all').strip()

        # Query stats overview
        cursor.execute("""
            SELECT 
                COUNT(*) FILTER (WHERE c.verification_status = 'Pending' AND (c.is_duplicate IS FALSE OR c.is_duplicate IS NULL)),
                COUNT(*) FILTER (WHERE c.verification_status = 'Pending' AND (c.parent_complaint_id IS NOT NULL OR cs.combined_score >= 0.5)),
                COUNT(*) FILTER (WHERE c.is_duplicate = TRUE)
            FROM complaints c
            LEFT JOIN LATERAL (
                SELECT combined_score
                FROM complaint_similarity
                WHERE complaint_id = c.complaint_id
                ORDER BY combined_score DESC
                LIMIT 1
            ) cs ON true;
        """)
        stats_counts = cursor.fetchone()
        pending_count = stats_counts[0] or 0
        potential_duplicate_count = stats_counts[1] or 0
        confirmed_duplicate_count = stats_counts[2] or 0

        where_clause = "WHERE c.verification_status = 'Pending' OR (c.is_duplicate = TRUE AND c.status = 'Duplicate')"
        if tab_filter == 'pending':
            where_clause = "WHERE c.verification_status = 'Pending' AND (c.is_duplicate IS FALSE OR c.is_duplicate IS NULL)"
        elif tab_filter == 'candidates':
            where_clause = "WHERE c.verification_status = 'Pending' AND (c.parent_complaint_id IS NOT NULL OR cs.combined_score >= 0.5)"
        elif tab_filter == 'confirmed':
            where_clause = "WHERE c.is_duplicate = TRUE"

        cursor.execute(
            f"""
            SELECT
                c.complaint_id,
                c.title,
                c.description,
                c.category,
                c.location,
                c.latitude,
                c.longitude,
                c.image_path,
                c.priority,
                c.status,
                c.verification_status,
                c.created_at,
                c.parent_complaint_id,
                u.first_name || ' ' || COALESCE(u.last_name, '') AS citizen_name,

                COALESCE(p.complaint_id, sim_p.complaint_id) AS parent_id,
                COALESCE(p.title, sim_p.title) AS parent_title,
                COALESCE(p.description, sim_p.description) AS parent_description,
                COALESCE(p.location, sim_p.location) AS parent_location,
                COALESCE(p.status, sim_p.status) AS parent_status,

                cs.similarity_id,
                cs.combined_score,
                cs.text_similarity,
                cs.location_distance_m,
                cs.image_similarity,
                cs.decision AS similarity_decision,
                c.is_duplicate,
                COALESCE(p.image_path, sim_p.image_path) AS parent_image_path,
                COALESCE(p.created_at, sim_p.created_at) AS parent_created_at,
                c.is_ai_suspected,
                c.ai_detection_score,
                c.image_relevance_score,
                c.image_relevance_status,
                c.image_validation_details

            FROM complaints c

            JOIN users u
                ON c.user_id = u.user_id

            LEFT JOIN complaints p
                ON c.parent_complaint_id = p.complaint_id

            LEFT JOIN LATERAL (
                SELECT similarity_id, similar_complaint_id, combined_score, text_similarity, location_distance_m, image_similarity, decision
                FROM complaint_similarity
                WHERE complaint_id = c.complaint_id
                ORDER BY combined_score DESC
                LIMIT 1
            ) cs ON true

            LEFT JOIN complaints sim_p
                ON cs.similar_complaint_id = sim_p.complaint_id

            {where_clause}

            ORDER BY 
                CASE 
                    WHEN c.verification_status = 'Pending' AND (cs.combined_score >= 0.5 OR c.parent_complaint_id IS NOT NULL) THEN 1
                    WHEN c.verification_status = 'Pending' THEN 2
                    ELSE 3
                END,
                c.created_at DESC
            """
        )

        rows = cursor.fetchall()
        complaints = []

        for row in rows:
            is_dup = bool(row[25])
            raw_parent_id = row[14]

            # Resolve to canonical root parent complaint (e.g. #62 not #63)
            parent_id_val = get_root_parent_complaint_id(cursor, raw_parent_id) if raw_parent_id else None
            parent_title = row[15]
            parent_desc = row[16]
            parent_loc = row[17]
            parent_status = row[18]
            parent_img = row[26]
            parent_created = row[27]

            # If raw_parent_id was a child and resolved to root parent, fetch root parent's actual info
            if parent_id_val and parent_id_val != raw_parent_id:
                cursor.execute(
                    """
                    SELECT title, description, location, status, image_path, created_at
                    FROM complaints WHERE complaint_id = %s
                    """,
                    (parent_id_val,)
                )
                p_meta = cursor.fetchone()
                if p_meta:
                    parent_title, parent_desc, parent_loc, parent_status, parent_img, parent_created = p_meta

            image_sim = round(float(row[23]) * 100, 2) if row[23] is not None else None
            dist_m = round(float(row[22]), 1) if row[22] is not None else None

            # Detect suspicious photo reuse / location mismatch (same photo, different location)
            is_photo_mismatch = False
            if image_sim is not None and image_sim >= 70:
                if dist_m is not None and dist_m > 500:
                    is_photo_mismatch = True
                elif dist_m is None and row[4] and parent_loc:
                    loc_text_sim = calculate_text_similarity(row[4], parent_loc)
                    if loc_text_sim <= 30:
                        is_photo_mismatch = True

            complaints.append({
                "complaint_id": row[0],
                "title": row[1],
                "description": row[2],
                "category": row[3],
                "location": row[4],
                "latitude": row[5],
                "longitude": row[6],
                "image_path": row[7],
                "priority": row[8],
                "status": row[9],
                "verification_status": row[10],
                "created_at": row[11],
                "parent_complaint_id": row[12],
                "citizen_name": row[13],
                "parent_id": parent_id_val,
                "parent_title": parent_title,
                "parent_description": parent_desc,
                "parent_location": parent_loc,
                "parent_status": parent_status,
                "similarity_id": row[19],
                "similarity_score": round(float(row[20]) * 100, 2) if row[20] is not None else None,
                "text_similarity": round(float(row[21]) * 100, 2) if row[21] is not None else None,
                "location_distance_m": dist_m,
                "image_similarity": image_sim,
                "similarity_decision": row[24],
                "is_duplicate": is_dup,
                "is_linked": bool(row[12]) or is_dup,
                "parent_image_path": parent_img,
                "parent_created_at": parent_created,
                "is_photo_location_mismatch": is_photo_mismatch,
                "is_ai_suspected": bool(row[28]) if row[28] is not None else False,
                "ai_detection_score": round(float(row[29]), 1) if row[29] is not None else None,
                "image_relevance_score": round(float(row[30]), 1) if row[30] is not None else None,
                "image_relevance_status": row[31] or "Pending",
                "image_validation_details": row[32]
            })

        return render_template(
            "verify_complaints.html",
            complaints=complaints,
            tab_filter=tab_filter,
            pending_count=pending_count,
            potential_duplicate_count=potential_duplicate_count,
            confirmed_duplicate_count=confirmed_duplicate_count
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        print("Verify Complaints Error:", e)

        if connection and not connection.closed:
            try:
                connection.rollback()
            except Exception:
                pass

        return f"Verify Complaints Error: {e}", 500

    finally:
        if cursor and not cursor.closed:
            cursor.close()
        if connection and not connection.closed:
            connection.close()

@app.route("/admin/departments")
def departments():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    if session.get('role') != 'Administrator':
        return "Access denied."

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            d.department_id,
            d.department_name,
            d.description,
            d.created_at,
            d.is_active,
            COUNT(c.complaint_id) AS complaint_count
        FROM departments d
        LEFT JOIN complaints c
            ON d.department_id = c.department_id
        GROUP BY
            d.department_id,
            d.department_name,
            d.description,
            d.created_at,
            d.is_active
        ORDER BY d.department_id;
    """)

    rows = cursor.fetchall()

    departments = []

    for row in rows:
        departments.append({
            'department_id': row[0],
            'department_name': row[1],
            'description': row[2],
            'created_at': row[3],
            'is_active': row[4],
            'complaint_count': row[5]
        })

    cursor.close()
    connection.close()

    return render_template(
        'departments.html',
        departments=departments
    )

@app.route('/admin/departments/toggle/<int:department_id>', methods=['POST'])
def toggle_department(department_id):

    if 'user_id' not in session:
        return redirect(url_for('login'))

    if session.get('role') != 'Administrator':
        return "Access denied."

    connection = None
    cursor = None

    try:
        connection = get_connection()
        cursor = connection.cursor()

        # Get current department status
        cursor.execute("""
            SELECT department_name, is_active
            FROM departments
            WHERE department_id = %s
        """, (department_id,))

        department = cursor.fetchone()

        if not department:
            return "Department not found."

        department_name = department[0]
        is_active = department[1]

        # -----------------------------------
        # If department is ACTIVE,
        # check whether it has processing complaints
        # before disabling
        # -----------------------------------

        if is_active:

            cursor.execute("""
                SELECT COUNT(*)
                FROM complaints
                WHERE department_id = %s
                AND status NOT IN ('Completed', 'Closed', 'Rejected')
            """, (department_id,))

            processing_count = cursor.fetchone()[0]

            if processing_count > 0:

                connection.rollback()

                return render_template(
                    "department_status_error.html",
                    message=(
                        f'Cannot disable "{department_name}" because '
                        f'it has {processing_count} complaint(s) currently being processed. '
                        'Please complete or close those complaints before disabling the department.'
                    )
                )

        # -----------------------------------
        # Toggle department status
        # -----------------------------------

        new_status = not is_active

        cursor.execute("""
            UPDATE departments
            SET is_active = %s
            WHERE department_id = %s
        """, (new_status, department_id))

        connection.commit()

        if new_status:
            flash(
                f'Department "{department_name}" has been enabled.',
                'success'
            )
        else:
            flash(
                f'Department "{department_name}" has been disabled.',
                'success'
            )

        return redirect(url_for('departments'))

    except Exception as e:

        print("Toggle Department Error:", e)

        if connection:
            connection.rollback()

        return f"Toggle Department Error: {e}"

    finally:

        if cursor:
            cursor.close()

        if connection:
            connection.close()

@app.route('/admin/departments/add', methods=['GET', 'POST'])
def add_department():

    if "user_id" not in session:
        return redirect(url_for("login"))

    if session.get("role") != "Administrator":
        return "Access denied."

    if request.method == "GET":

        return render_template(
            "department_form.html",
            title="Add Department",
            button_text="Add Department",
            department=None
        )

    department_name = request.form.get(
        "department_name", ""
    ).strip()

    description = request.form.get(
        "description", ""
    ).strip()

    if not department_name:

        return render_template(
            "department_form.html",
            title="Add Department",
            button_text="Add Department",
            department=None,
            error="Department name is required."
        )

    connection = None
    cursor = None

    try:

        connection = get_connection()
        cursor = connection.cursor()

        # Check duplicate department
        cursor.execute(
            """
            SELECT department_id
            FROM departments
            WHERE LOWER(department_name) = LOWER(%s)
            """,
            (department_name,)
        )

        existing = cursor.fetchone()

        if existing:

            cursor.close()
            connection.close()

            return render_template(
                "department_form.html",
                title="Add Department",
                button_text="Add Department",
                department=None,
                error="A department with this name already exists."
            )

        cursor.execute(
            """
            INSERT INTO departments
            (
                department_name,
                description,
                created_at
            )
            VALUES
            (
                %s,
                %s,
                CURRENT_TIMESTAMP
            )
            """,
            (
                department_name,
                description
            )
        )

        connection.commit()

        cursor.close()
        connection.close()

        return redirect(url_for("departments"))

    except Exception as e:

        print("Add Department Error:", e)

        if cursor:
            cursor.close()

        if connection:
            connection.close()

        return f"Add Department Error: {e}"
@app.route('/admin/departments/edit/<int:department_id>', methods=['GET', 'POST'])
def edit_department(department_id):

    if "user_id" not in session:
        return redirect(url_for("login"))

    if session.get("role") != "Administrator":
        return "Access denied."

    connection = None
    cursor = None

    try:

        connection = get_connection()
        cursor = connection.cursor()

        # Get department
        cursor.execute(
            """
            SELECT
                department_id,
                department_name,
                description
            FROM departments
            WHERE department_id = %s
            """,
            (department_id,)
        )

        row = cursor.fetchone()

        if not row:

            cursor.close()
            connection.close()

            return "Department not found."

        department = {
            "department_id": row[0],
            "department_name": row[1],
            "description": row[2]
        }


        # GET
        if request.method == "GET":

            cursor.close()
            connection.close()

            return render_template(
                "department_form.html",
                title="Edit Department",
                button_text="Update Department",
                department=department
            )


        # POST
        department_name = request.form.get(
            "department_name", ""
        ).strip()

        description = request.form.get(
            "description", ""
        ).strip()


        if not department_name:

            cursor.close()
            connection.close()

            return render_template(
                "department_form.html",
                title="Edit Department",
                button_text="Update Department",
                department=department,
                error="Department name is required."
            )


        # Check duplicate name
        cursor.execute(
            """
            SELECT department_id
            FROM departments
            WHERE LOWER(department_name) = LOWER(%s)
              AND department_id != %s
            """,
            (
                department_name,
                department_id
            )
        )

        existing = cursor.fetchone()

        if existing:

            cursor.close()
            connection.close()

            department["department_name"] = department_name
            department["description"] = description

            return render_template(
                "department_form.html",
                title="Edit Department",
                button_text="Update Department",
                department=department,
                error="Another department already has this name."
            )


        # Update
        cursor.execute(
            """
            UPDATE departments
            SET
                department_name = %s,
                description = %s
            WHERE department_id = %s
            """,
            (
                department_name,
                description,
                department_id
            )
        )

        connection.commit()

        cursor.close()
        connection.close()

        return redirect(url_for("departments"))


    except Exception as e:

        print("Edit Department Error:", e)

        if cursor:
            cursor.close()

        if connection:
            connection.close()

        return f"Edit Department Error: {e}"

@app.route('/admin/departments/delete/<int:department_id>', methods=['POST'])
def delete_department(department_id):

    if 'user_id' not in session:
        return redirect(url_for('login'))

    if session.get('role') != 'Administrator':
        return "Access denied."

    connection = None
    cursor = None

    try:
        connection = get_connection()
        cursor = connection.cursor()

        # -----------------------------------
        # Get department
        # -----------------------------------

        cursor.execute("""
            SELECT department_name
            FROM departments
            WHERE department_id = %s
        """, (department_id,))

        department = cursor.fetchone()

        if not department:
            flash("Department not found.", "error")
            return redirect(url_for('departments'))

        department_name = department[0]

        # -----------------------------------
        # Check complaint history
        # -----------------------------------

        cursor.execute("""
            SELECT COUNT(*)
            FROM complaints
            WHERE department_id = %s
        """, (department_id,))

        complaint_count = cursor.fetchone()[0]

        # -----------------------------------
        # Department has complaint history
        # -----------------------------------

        if complaint_count > 0:

            connection.rollback()

            return render_template(
                "department_delete_error.html",
                message=(
                    f'Cannot delete "{department_name}" because it has '
                    f'{complaint_count} complaint(s) associated with it. '
                    'Departments with complaint history cannot be deleted. '
                    'You can disable the department instead.'
                )
            )

        # -----------------------------------
        # No complaint history → allow delete
        # -----------------------------------

        cursor.execute("""
            DELETE FROM departments
            WHERE department_id = %s
        """, (department_id,))

        connection.commit()

        flash(
            f'Department "{department_name}" has been deleted successfully.',
            'success'
        )

        return redirect(url_for('departments'))

    except Exception as e:

        print("Delete Department Error:", e)

        if connection:
            connection.rollback()

        return f"Delete Department Error: {e}"

    finally:

        if cursor:
            cursor.close()

        if connection:
            connection.close()

@app.route('/admin/reports')
def reports():
    if session.get('role') != 'Administrator':
        return redirect(url_for('login'))

    dept_filter = request.args.get('department_id', type=int)
    time_filter = request.args.get('time_range', default='all')

    connection = None
    cursor = None
    try:
        connection = get_connection()
        cursor = connection.cursor()

        # Build time filter SQL clause
        time_clause = ""
        if time_filter == '7days':
            time_clause = " AND c.created_at >= NOW() - INTERVAL '7 days' "
        elif time_filter == '30days':
            time_clause = " AND c.created_at >= NOW() - INTERVAL '30 days' "
        elif time_filter == 'this_month':
            time_clause = " AND c.created_at >= date_trunc('month', CURRENT_DATE) "
        elif time_filter == 'this_year':
            time_clause = " AND c.created_at >= date_trunc('year', CURRENT_DATE) "

        # 1. Departments List (for dropdown filter)
        cursor.execute("SELECT department_id, department_name FROM departments ORDER BY department_name;")
        all_departments = cursor.fetchall()

        # 2. Department-wise complaint statistics
        dept_stats_query = f"""
            SELECT 
                d.department_id,
                d.department_name,
                COUNT(c.complaint_id) AS total_complaints,
                COUNT(CASE WHEN c.status = 'Pending' THEN 1 END) AS pending_count,
                COUNT(CASE WHEN c.status IN ('Verified', 'Inspection', 'Maintenance', 'Assigned') THEN 1 END) AS in_progress_count,
                COUNT(CASE WHEN c.status = 'Completed' THEN 1 END) AS completed_count,
                COUNT(CASE WHEN c.status = 'Rejected' THEN 1 END) AS rejected_count,
                COUNT(CASE WHEN c.status NOT IN ('Completed', 'Rejected') AND c.created_at < NOW() - INTERVAL '7 days' THEN 1 END) AS overdue_count,
                AVG(CASE WHEN c.status = 'Completed' AND c.updated_at >= c.created_at THEN EXTRACT(EPOCH FROM (c.updated_at - c.created_at)) / 86400 END) AS avg_resolution_days
            FROM departments d
            LEFT JOIN complaints c ON d.department_id = c.department_id {time_clause}
            GROUP BY d.department_id, d.department_name
            ORDER BY total_complaints DESC;
        """
        cursor.execute(dept_stats_query)
        dept_stats_rows = cursor.fetchall()
        department_stats = []
        chart_dept_labels = []
        chart_dept_totals = []
        chart_dept_completed = []
        chart_dept_pending = []
        chart_dept_overdue = []

        for r in dept_stats_rows:
            d_id, d_name, tot, pend, in_prog, comp, rej, od, avg_d = r
            res_rate = round((comp / tot * 100), 1) if tot > 0 else 0
            avg_days_val = round(float(avg_d), 1) if avg_d is not None else None
            department_stats.append({
                'department_id': d_id,
                'department_name': d_name,
                'total': tot,
                'pending': pend,
                'in_progress': in_prog,
                'completed': comp,
                'rejected': rej,
                'overdue': od,
                'resolution_rate': res_rate,
                'avg_resolution_days': avg_days_val
            })
            chart_dept_labels.append(d_name)
            chart_dept_totals.append(tot)
            chart_dept_completed.append(comp)
            chart_dept_pending.append(pend + in_prog)
            chart_dept_overdue.append(od)

        # 3. Overall KPI Summary Metrics
        dept_where = ""
        query_params = []
        if dept_filter:
            dept_where = " AND c.department_id = %s "
            query_params.append(dept_filter)

        summary_query = f"""
            SELECT 
                COUNT(*) AS total,
                COUNT(CASE WHEN status IN ('Pending', 'Verified', 'Inspection', 'Maintenance', 'Assigned') THEN 1 END) AS pending_total,
                COUNT(CASE WHEN status = 'Completed' THEN 1 END) AS completed_total,
                COUNT(CASE WHEN status = 'Rejected' THEN 1 END) AS rejected_total,
                COUNT(CASE WHEN status NOT IN ('Completed', 'Rejected') AND created_at < NOW() - INTERVAL '7 days' THEN 1 END) AS overdue_total
            FROM complaints c
            WHERE 1=1 {dept_where} {time_clause};
        """
        cursor.execute(summary_query, query_params)
        sum_row = cursor.fetchone()
        total_complaints = sum_row[0] or 0
        pending_complaints = sum_row[1] or 0
        completed_complaints = sum_row[2] or 0
        rejected_complaints = sum_row[3] or 0
        overdue_complaints = sum_row[4] or 0
        resolution_rate = round((completed_complaints / total_complaints * 100), 1) if total_complaints > 0 else 0

        # 4. Average Resolution Time (overall)
        avg_res_query = f"""
            SELECT 
                AVG(EXTRACT(EPOCH FROM (updated_at - created_at)) / 86400) AS avg_days,
                AVG(EXTRACT(EPOCH FROM (updated_at - created_at)) / 3600) AS avg_hours
            FROM complaints c
            WHERE status = 'Completed' AND updated_at IS NOT NULL AND updated_at >= created_at
            {dept_where} {time_clause};
        """
        cursor.execute(avg_res_query, query_params)
        avg_row = cursor.fetchone()
        avg_resolution_days = round(float(avg_row[0]), 1) if avg_row and avg_row[0] is not None else 0
        avg_resolution_hours = round(float(avg_row[1]), 1) if avg_row and avg_row[1] is not None else 0

        # 5. Frequently Reported Problem Locations (Hotspots)
        locations_query = f"""
            SELECT 
                COALESCE(NULLIF(TRIM(location), ''), 'Unknown Location') AS loc,
                COUNT(*) AS total_reports,
                COUNT(CASE WHEN status = 'Completed' THEN 1 END) AS resolved_count,
                COUNT(CASE WHEN status NOT IN ('Completed', 'Rejected') AND created_at < NOW() - INTERVAL '7 days' THEN 1 END) AS overdue_count,
                MAX(category) AS sample_category,
                MAX(created_at) AS latest_report
            FROM complaints c
            WHERE 1=1 {dept_where} {time_clause}
            GROUP BY loc
            ORDER BY total_reports DESC, overdue_count DESC
            LIMIT 8;
        """
        cursor.execute(locations_query, query_params)
        hotspot_rows = cursor.fetchall()
        hotspot_locations = []
        for loc, tot, res, od, cat, latest in hotspot_rows:
            loc_rate = round((res / tot * 100), 1) if tot > 0 else 0
            hotspot_locations.append({
                'location': loc,
                'total_reports': tot,
                'resolved_count': res,
                'overdue_count': od,
                'category': cat or 'General',
                'resolution_rate': loc_rate,
                'latest_report': latest.strftime('%b %d, %Y') if latest else 'N/A'
            })

        # 6. Critical Overdue Complaints
        overdue_list_query = f"""
            SELECT 
                c.complaint_id,
                c.title,
                d.department_name,
                c.location,
                c.priority,
                c.status,
                c.created_at,
                ROUND(EXTRACT(EPOCH FROM (NOW() - c.created_at)) / 86400) AS days_open
            FROM complaints c
            LEFT JOIN departments d ON c.department_id = d.department_id
            WHERE c.status NOT IN ('Completed', 'Rejected') 
              AND c.created_at < NOW() - INTERVAL '7 days'
              {dept_where} {time_clause}
            ORDER BY c.created_at ASC
            LIMIT 10;
        """
        cursor.execute(overdue_list_query, query_params)
        overdue_complaints_list = cursor.fetchall()

        # 7. Category Distribution (for chart)
        category_query = f"""
            SELECT category, COUNT(*) 
            FROM complaints c
            WHERE 1=1 {dept_where} {time_clause}
            GROUP BY category
            ORDER BY COUNT(*) DESC;
        """
        cursor.execute(category_query, query_params)
        category_distribution = cursor.fetchall()

        # 8. Monthly Trends (last 6 months)
        trend_query = f"""
            SELECT 
                TO_CHAR(created_at, 'Mon YYYY') AS month_label,
                COUNT(*) AS total_received,
                COUNT(CASE WHEN status = 'Completed' THEN 1 END) AS total_resolved
            FROM complaints c
            WHERE created_at >= NOW() - INTERVAL '6 months'
            {dept_where}
            GROUP BY TO_CHAR(created_at, 'Mon YYYY'), date_trunc('month', created_at)
            ORDER BY date_trunc('month', created_at) ASC;
        """
        cursor.execute(trend_query, query_params)
        monthly_trends = cursor.fetchall()

        cursor.close()
        connection.close()

        # JSON serialized for client-side Chart.js
        chart_data = {
            'dept_labels': chart_dept_labels,
            'dept_totals': chart_dept_totals,
            'dept_completed': chart_dept_completed,
            'dept_pending': chart_dept_pending,
            'dept_overdue': chart_dept_overdue,
            'status_counts': [completed_complaints, pending_complaints, overdue_complaints, rejected_complaints],
            'status_labels': ['Completed', 'In Progress / Pending', 'Overdue (>7 Days)', 'Rejected'],
            'categories': [row[0] for row in category_distribution],
            'category_counts': [row[1] for row in category_distribution],
            'trend_months': [row[0] for row in monthly_trends],
            'trend_received': [row[1] for row in monthly_trends],
            'trend_resolved': [row[2] for row in monthly_trends]
        }

        return render_template(
            'reports.html',
            all_departments=all_departments,
            selected_dept=dept_filter,
            selected_time=time_filter,
            department_stats=department_stats,
            total_complaints=total_complaints,
            pending_complaints=pending_complaints,
            completed_complaints=completed_complaints,
            rejected_complaints=rejected_complaints,
            overdue_complaints=overdue_complaints,
            resolution_rate=resolution_rate,
            avg_resolution_days=avg_resolution_days,
            avg_resolution_hours=avg_resolution_hours,
            hotspot_locations=hotspot_locations,
            overdue_complaints_list=overdue_complaints_list,
            category_distribution=category_distribution,
            monthly_trends=monthly_trends,
            chart_data_json=json.dumps(chart_data)
        )
    except Exception as e:
        print("Error in reports route:", e)
        if cursor:
            cursor.close()
        if connection:
            connection.close()
        return render_template(
            'reports.html',
            error=str(e),
            all_departments=[],
            department_stats=[],
            hotspot_locations=[],
            overdue_complaints_list=[],
            total_complaints=0,
            pending_complaints=0,
            completed_complaints=0,
            rejected_complaints=0,
            overdue_complaints=0,
            resolution_rate=0,
            avg_resolution_days=0,
            avg_resolution_hours=0,
            chart_data_json='{}'
        )

# ==========================================
# ADMIN: DEPARTMENT HEADS MANAGEMENT
# ==========================================
@app.route('/admin/department-heads')
def admin_department_heads():
    if session.get('role') != 'Administrator':
        return redirect(url_for('login'))

    connection = None
    cursor = None
    try:
        connection = get_connection()
        cursor = connection.cursor()

        # Get all department heads with department name
        cursor.execute("""
            SELECT 
                u.user_id,
                u.first_name,
                u.last_name,
                u.email,
                u.phone,
                u.department_id,
                d.department_name,
                u.is_active,
                u.must_change_password,
                u.created_at
            FROM users u
            LEFT JOIN departments d ON u.department_id = d.department_id
            WHERE u.role = 'Department Head'
            ORDER BY u.user_id ASC;
        """)
        rows = cursor.fetchall()
        department_heads = []
        for r in rows:
            department_heads.append({
                'user_id': r[0],
                'first_name': r[1],
                'last_name': r[2],
                'email': r[3],
                'phone': r[4],
                'department_id': r[5],
                'department_name': r[6],
                'is_active': r[7],
                'must_change_password': r[8],
                'created_at': r[9]
            })

        # Get active departments for add/edit modal dropdown
        cursor.execute("""
            SELECT department_id, department_name
            FROM departments
            WHERE is_active = TRUE
            ORDER BY department_name ASC;
        """)
        dept_rows = cursor.fetchall()
        active_departments_list = [{'department_id': d[0], 'department_name': d[1]} for d in dept_rows]

        # Statistics
        total_heads = len(department_heads)
        active_heads = sum(1 for h in department_heads if h['is_active'])
        pending_password_changes = sum(1 for h in department_heads if h['must_change_password'])

        cursor.execute("SELECT COUNT(*) FROM departments;")
        total_departments = cursor.fetchone()[0]

        assigned_dept_ids = {h['department_id'] for h in department_heads if h['department_id']}
        assigned_departments = len(assigned_dept_ids)

        return render_template(
            'admin_department_heads.html',
            department_heads=department_heads,
            active_departments_list=active_departments_list,
            total_heads=total_heads,
            active_heads=active_heads,
            pending_password_changes=pending_password_changes,
            total_departments=total_departments,
            assigned_departments=assigned_departments
        )
    except Exception as e:
        print("Admin Department Heads Error:", e)
        return f"Error loading department heads: {e}"
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


@app.route('/admin/department-heads/add', methods=['POST'])
def add_department_head():
    if session.get('role') != 'Administrator':
        return redirect(url_for('login'))

    first_name = request.form.get('first_name', '').strip()
    last_name = request.form.get('last_name', '').strip()
    email = request.form.get('email', '').strip().lower()
    phone = request.form.get('phone', '').strip()
    department_id = request.form.get('department_id')
    temporary_password = request.form.get('temporary_password', '').strip()

    if not first_name or not email or not department_id or not temporary_password:
        flash("Please fill in all required fields (First Name, Email, Department, and Fixed Password).", "error")
        return redirect(url_for('admin_department_heads'))

    if not is_valid_email(email):
        flash("Please enter a valid email address (e.g. name@example.com).", "error")
        return redirect(url_for('admin_department_heads'))

    if phone and not is_valid_phone(phone):
        flash("Please enter a valid phone number (10 to 15 digits).", "error")
        return redirect(url_for('admin_department_heads'))

    connection = None
    cursor = None
    try:
        connection = get_connection()
        cursor = connection.cursor()

        # Check for existing email
        cursor.execute("SELECT user_id FROM users WHERE LOWER(email) = LOWER(%s);", (email,))
        if cursor.fetchone():
            flash(f"A user account with email '{email}' already exists.", "error")
            return redirect(url_for('admin_department_heads'))

        # Hash the fixed password provided by the admin
        password_hash = generate_password_hash(temporary_password)

        cursor.execute("""
            INSERT INTO users (
                first_name, last_name, email, phone, password_hash,
                role, department_id, is_active, must_change_password, created_at
            ) VALUES (
                %s, %s, %s, %s, %s,
                'Department Head', %s, TRUE, TRUE, CURRENT_TIMESTAMP
            );
        """, (first_name, last_name, email, phone, password_hash, int(department_id)))
        connection.commit()

        flash(
            f"<strong>Department Head created successfully!</strong><br>"
            f"Email: <code class='bg-white px-1.5 py-0.5 rounded text-blue-700 font-mono'>{email}</code><br>"
            f"Initial Fixed Password: <code class='bg-white px-1.5 py-0.5 rounded text-amber-700 font-mono'>{temporary_password}</code><br>"
            f"<em>(The Department Head must change this password on their first login)</em>",
            "success"
        )
        return redirect(url_for('admin_department_heads'))

    except Exception as e:
        print("Add Department Head Error:", e)
        if connection:
            connection.rollback()
        flash(f"Error adding department head: {e}", "error")
        return redirect(url_for('admin_department_heads'))
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


@app.route('/admin/department-heads/edit/<int:user_id>', methods=['POST'])
def edit_department_head(user_id):
    if session.get('role') != 'Administrator':
        return redirect(url_for('login'))

    first_name = request.form.get('first_name', '').strip()
    last_name = request.form.get('last_name', '').strip()
    email = request.form.get('email', '').strip().lower()
    phone = request.form.get('phone', '').strip()
    department_id = request.form.get('department_id')

    if not first_name or not email or not department_id:
        flash("Please fill in all required fields (First Name, Email, Department).", "error")
        return redirect(url_for('admin_department_heads'))

    if not is_valid_email(email):
        flash("Please enter a valid email address (e.g. name@example.com).", "error")
        return redirect(url_for('admin_department_heads'))

    if phone and not is_valid_phone(phone):
        flash("Please enter a valid phone number (10 to 15 digits).", "error")
        return redirect(url_for('admin_department_heads'))

    dept_val = int(department_id) if department_id else None

    connection = None
    cursor = None
    try:
        connection = get_connection()
        cursor = connection.cursor()

        cursor.execute("SELECT user_id FROM users WHERE LOWER(email) = LOWER(%s) AND user_id != %s;", (email, user_id))
        if cursor.fetchone():
            flash(f"Email '{email}' is already in use by another user.", "error")
            return redirect(url_for('admin_department_heads'))

        cursor.execute("""
            UPDATE users
            SET first_name = %s,
                last_name = %s,
                email = %s,
                phone = %s,
                department_id = %s
            WHERE user_id = %s AND role = 'Department Head';
        """, (first_name, last_name, email, phone, dept_val, user_id))
        connection.commit()

        flash("Department Head profile updated successfully.", "success")
        return redirect(url_for('admin_department_heads'))

    except Exception as e:
        print("Edit Department Head Error:", e)
        if connection:
            connection.rollback()
        flash(f"Error updating department head: {e}", "error")
        return redirect(url_for('admin_department_heads'))
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


@app.route('/admin/department-heads/reset-password/<int:user_id>', methods=['POST'])
def reset_department_head_password(user_id):
    if session.get('role') != 'Administrator':
        return redirect(url_for('login'))

    temporary_password = request.form.get('temporary_password', '').strip()
    if not temporary_password:
        flash("Please enter a new temporary password.", "error")
        return redirect(url_for('admin_department_heads'))

    connection = None
    cursor = None
    try:
        connection = get_connection()
        cursor = connection.cursor()

        password_hash = generate_password_hash(temporary_password)

        cursor.execute("""
            UPDATE users
            SET password_hash = %s,
                must_change_password = TRUE
            WHERE user_id = %s AND role = 'Department Head';
        """, (password_hash, user_id))
        connection.commit()

        flash(
            f"Password reset successfully! New temporary password: "
            f"<code class='bg-white px-1.5 py-0.5 rounded text-amber-700 font-mono'>{temporary_password}</code>. "
            f"The department head will be forced to change this on their next login.",
            "success"
        )
        return redirect(url_for('admin_department_heads'))

    except Exception as e:
        print("Reset Password Error:", e)
        if connection:
            connection.rollback()
        flash(f"Error resetting password: {e}", "error")
        return redirect(url_for('admin_department_heads'))
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


@app.route('/admin/department-heads/toggle/<int:user_id>', methods=['POST'])
def toggle_department_head(user_id):
    if session.get('role') != 'Administrator':
        return redirect(url_for('login'))

    connection = None
    cursor = None
    try:
        connection = get_connection()
        cursor = connection.cursor()

        cursor.execute("""
            UPDATE users
            SET is_active = NOT is_active
            WHERE user_id = %s AND role = 'Department Head';
        """, (user_id,))
        connection.commit()

        flash("Department head status updated successfully.", "success")
        return redirect(url_for('admin_department_heads'))
    except Exception as e:
        print("Toggle Dept Head Status Error:", e)
        if connection:
            connection.rollback()
        flash(f"Error toggling status: {e}", "error")
        return redirect(url_for('admin_department_heads'))
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


# ==========================================
# ADMIN: USERS DIRECTORY & DETAILS
# ==========================================
@app.route('/admin/users')
def users():
    if session.get('role') != 'Administrator':
        return redirect(url_for('login'))

    search = request.args.get('search', '').strip()
    selected_role = request.args.get('role', '').strip()
    selected_status = request.args.get('status', '').strip()
    selected_sort = request.args.get('sort', 'role').strip()

    connection = None
    cursor = None
    try:
        connection = get_connection()
        cursor = connection.cursor()

        query = """
            SELECT 
                u.user_id,
                u.first_name,
                u.last_name,
                u.email,
                u.phone,
                u.address,
                u.role,
                u.department_id,
                d.department_name,
                u.is_active,
                u.must_change_password,
                u.created_at,
                COALESCE(u.strike_count, 0) AS strike_count,
                (SELECT COUNT(*) FROM complaints c WHERE c.user_id = u.user_id) AS citizen_complaint_count,
                (SELECT COUNT(*) FROM complaints c WHERE c.assigned_engineer_id = u.user_id) AS engineer_assigned_count,
                (SELECT COUNT(*) FROM maintenance_work m WHERE m.maintenance_staff_id = u.user_id) AS maintenance_work_count
            FROM users u
            LEFT JOIN departments d ON u.department_id = d.department_id
            WHERE 1=1
        """
        params = []

        if search:
            query += """
                AND (
                    LOWER(u.first_name) LIKE LOWER(%s)
                    OR LOWER(COALESCE(u.last_name, '')) LIKE LOWER(%s)
                    OR LOWER(u.email) LIKE LOWER(%s)
                    OR LOWER(COALESCE(u.phone, '')) LIKE LOWER(%s)
                )
            """
            s_param = f"%{search}%"
            params.extend([s_param, s_param, s_param, s_param])

        if selected_role:
            query += " AND u.role = %s"
            params.append(selected_role)

        if selected_status == 'active':
            query += " AND u.is_active = TRUE"
        elif selected_status == 'inactive':
            query += " AND u.is_active = FALSE"

        # Logical arrangement: Default to Role Hierarchy, with department and name grouping
        if selected_sort == 'name':
            query += " ORDER BY u.first_name ASC, u.last_name ASC;"
        elif selected_sort == 'dept':
            query += " ORDER BY COALESCE(d.department_name, 'zzz') ASC, u.first_name ASC;"
        elif selected_sort == 'newest':
            query += " ORDER BY u.created_at DESC;"
        elif selected_sort == 'id':
            query += " ORDER BY u.user_id ASC;"
        else:
            # Default: Role Hierarchy
            selected_sort = 'role'
            query += """
                ORDER BY 
                    CASE u.role 
                        WHEN 'Administrator' THEN 1
                        WHEN 'Department Head' THEN 2
                        WHEN 'Engineer' THEN 3
                        WHEN 'Maintenance' THEN 4
                        WHEN 'Citizen' THEN 5
                        ELSE 6 
                    END ASC,
                    COALESCE(d.department_name, '') ASC,
                    u.first_name ASC,
                    u.last_name ASC;
            """

        cursor.execute(query, tuple(params))
        rows = cursor.fetchall()

        users_list = []
        for r in rows:
            users_list.append({
                'user_id': r[0],
                'first_name': r[1],
                'last_name': r[2],
                'email': r[3],
                'phone': r[4],
                'address': r[5],
                'role': r[6],
                'department_id': r[7],
                'department_name': r[8],
                'is_active': r[9],
                'must_change_password': r[10],
                'created_at': r[11],
                'strike_count': r[12],
                'citizen_complaint_count': r[13],
                'engineer_assigned_count': r[14],
                'maintenance_work_count': r[15],
                'created_at_formatted': r[11].strftime('%b %d, %Y, %I:%M %p') if r[11] else '—'
            })

        # Statistics overview
        cursor.execute("""
            SELECT 
                COUNT(*),
                COUNT(*) FILTER (WHERE role = 'Citizen'),
                COUNT(*) FILTER (WHERE role = 'Department Head'),
                COUNT(*) FILTER (WHERE role = 'Engineer'),
                COUNT(*) FILTER (WHERE role = 'Maintenance'),
                COUNT(*) FILTER (WHERE role = 'Administrator'),
                COUNT(*) FILTER (WHERE is_active = TRUE)
            FROM users;
        """)
        stats_row = cursor.fetchone()
        total_users = stats_row[0] or 0
        citizen_count = stats_row[1] or 0
        dept_head_count = stats_row[2] or 0
        engineer_count = stats_row[3] or 0
        maintenance_count = stats_row[4] or 0
        admin_count = stats_row[5] or 0
        active_users_count = stats_row[6] or 0
        staff_count = engineer_count + maintenance_count

        users_json = json.dumps(users_list, default=str)

        # Pagination to eliminate page scroll and provide sleek UI
        page = request.args.get('page', 1, type=int)
        per_page = 8  # 8 users per page fits cleanly on single desktop screen without scroll
        total_filtered_users = len(users_list)
        total_pages = max(1, math.ceil(total_filtered_users / per_page))
        if page < 1:
            page = 1
        elif page > total_pages:
            page = total_pages
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        paginated_users = users_list[start_idx:end_idx]

        return render_template(
            'users.html',
            users_list=paginated_users,
            users_json=users_json,
            total_users=total_users,
            citizen_count=citizen_count,
            dept_head_count=dept_head_count,
            engineer_count=engineer_count,
            maintenance_count=maintenance_count,
            admin_count=admin_count,
            staff_count=staff_count,
            active_users_count=active_users_count,
            search=search,
            selected_role=selected_role,
            selected_status=selected_status,
            selected_sort=selected_sort,
            page=page,
            total_pages=total_pages,
            per_page=per_page,
            total_filtered_users=total_filtered_users,
            showing_start=start_idx + 1 if total_filtered_users > 0 else 0,
            showing_end=min(end_idx, total_filtered_users)
        )
    except Exception as e:
        print("Admin Users Error:", e)
        return f"Error loading users: {e}"
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


@app.route('/admin/users/toggle/<int:user_id>', methods=['POST'])
def toggle_user_status(user_id):
    if session.get('role') != 'Administrator':
        return redirect(url_for('login'))

    if user_id == session.get('user_id'):
        flash("You cannot deactivate your own administrator account.", "error")
        return redirect(url_for('users'))

    connection = None
    cursor = None
    try:
        connection = get_connection()
        cursor = connection.cursor()

        cursor.execute("UPDATE users SET is_active = NOT is_active WHERE user_id = %s;", (user_id,))
        connection.commit()

        flash("User account status has been updated.", "success")
        return redirect(url_for('users'))
    except Exception as e:
        print("Toggle User Status Error:", e)
        if connection:
            connection.rollback()
        flash(f"Error toggling user status: {e}", "error")
        return redirect(url_for('users'))
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


@app.route('/admin/settings')
def settings():
    if session.get('role') != 'Administrator':
        return redirect(url_for('login'))
    return redirect(url_for('admin_dashboard'))


@app.route('/uploads/<path:filename>')
@app.route('/static/uploads/<path:filename>')
def uploaded_file(filename):
    clean = filename.replace('\\', '/').strip('/')
    while clean.startswith('uploads/'):
        clean = clean[len('uploads/'):]

    upload_folder = app.config['UPLOAD_FOLDER']
    target_path = os.path.join(upload_folder, clean)

    if os.path.isfile(target_path):
        directory = os.path.dirname(target_path)
        base = os.path.basename(target_path)
        return send_from_directory(directory, base)

    base_name = os.path.basename(clean)
    for root_dir, _, files in os.walk(upload_folder):
        if base_name in files:
            return send_from_directory(root_dir, base_name)

    return send_from_directory(upload_folder, clean)


# ==========================================
# DEPARTMENT HEAD SUB-ROUTES
# ==========================================
# ==========================================
# DEPARTMENT HEAD: ENGINEERS MANAGEMENT
# ==========================================
@app.route('/department-head/engineers')
def department_head_engineers():
    if session.get('role') not in ('Department Head', 'department_head'):
        return redirect(url_for('login'))

    dept_id = session.get('department_id')
    if not dept_id:
        return "No department assigned to this account."

    connection = None
    cursor = None
    try:
        connection = get_connection()
        cursor = connection.cursor()

        cursor.execute("SELECT department_name FROM departments WHERE department_id = %s;", (dept_id,))
        dept_row = cursor.fetchone()
        department_name = dept_row[0] if dept_row else "Department"

        cursor.execute("""
            SELECT 
                u.user_id,
                u.first_name,
                u.last_name,
                u.email,
                u.phone,
                u.is_active,
                u.must_change_password,
                u.created_at,
                (SELECT COUNT(*) FROM complaints c WHERE c.assigned_engineer_id = u.user_id AND c.status IN ('Assigned', 'Inspection')) AS active_jobs,
                (SELECT COUNT(*) FROM inspection_reports r WHERE r.engineer_id = u.user_id) AS total_inspections
            FROM users u
            WHERE u.role = 'Engineer' AND u.department_id = %s
            ORDER BY u.user_id ASC;
        """, (dept_id,))
        rows = cursor.fetchall()

        engineers = []
        for r in rows:
            engineers.append({
                'user_id': r[0],
                'first_name': r[1],
                'last_name': r[2],
                'email': r[3],
                'phone': r[4],
                'is_active': r[5],
                'must_change_password': r[6],
                'created_at': r[7],
                'active_jobs': r[8] or 0,
                'total_inspections': r[9] or 0
            })

        total_engineers = len(engineers)
        active_engineers = sum(1 for e in engineers if e['is_active'])
        pending_password_changes = sum(1 for e in engineers if e['must_change_password'])
        total_active_jobs = sum(e['active_jobs'] for e in engineers)

        return render_template(
            'department_head_engineers.html',
            engineers=engineers,
            department_name=department_name,
            total_engineers=total_engineers,
            active_engineers=active_engineers,
            pending_password_changes=pending_password_changes,
            total_active_jobs=total_active_jobs
        )
    except Exception as e:
        print("Dept Head Engineers Error:", e)
        return f"Error loading engineers: {e}"
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


@app.route('/department-head/engineers/add', methods=['POST'])
def add_engineer():
    if session.get('role') not in ('Department Head', 'department_head'):
        return redirect(url_for('login'))

    dept_id = session.get('department_id')
    if not dept_id:
        return "No department assigned."

    first_name = request.form.get('first_name', '').strip()
    last_name = request.form.get('last_name', '').strip()
    email = request.form.get('email', '').strip().lower()
    phone = request.form.get('phone', '').strip()
    temporary_password = request.form.get('temporary_password', '').strip()

    target_url = request.referrer or url_for('department_head_engineers')

    if not first_name or not email or not temporary_password:
        flash("Please fill in First Name, Email, and Temporary Password.", "error")
        return redirect(target_url)

    if not is_valid_email(email):
        flash("Please enter a valid email address (e.g. name@example.com).", "error")
        return redirect(target_url)

    if phone and not is_valid_phone(phone):
        flash("Please enter a valid phone number (10 to 15 digits).", "error")
        return redirect(target_url)

    connection = None
    cursor = None
    try:
        connection = get_connection()
        cursor = connection.cursor()

        cursor.execute("SELECT user_id FROM users WHERE LOWER(email) = LOWER(%s);", (email,))
        if cursor.fetchone():
            flash(f"A user with email '{email}' already exists.", "error")
            return redirect(target_url)

        password_hash = generate_password_hash(temporary_password)

        cursor.execute("""
            INSERT INTO users (
                first_name, last_name, email, phone, password_hash,
                role, department_id, is_active, must_change_password, created_at
            ) VALUES (
                %s, %s, %s, %s, %s,
                'Engineer', %s, TRUE, TRUE, CURRENT_TIMESTAMP
            );
        """, (first_name, last_name, email, phone, password_hash, dept_id))
        connection.commit()

        flash(
            f"<strong>Engineer added successfully!</strong><br>"
            f"Email: <code class='bg-white px-1.5 py-0.5 rounded text-blue-700 font-mono'>{email}</code><br>"
            f"Initial Temporary Password: <code class='bg-white px-1.5 py-0.5 rounded text-amber-700 font-mono'>{temporary_password}</code><br>"
            f"<em>(The Engineer must change this password upon their first login)</em>",
            "success"
        )
        return redirect(target_url)
    except Exception as e:
        print("Add Engineer Error:", e)
        if connection:
            connection.rollback()
        flash(f"Error adding engineer: {e}", "error")
        return redirect(target_url)
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


@app.route('/department-head/engineers/edit/<int:user_id>', methods=['POST'])
def edit_engineer(user_id):
    if session.get('role') not in ('Department Head', 'department_head'):
        return redirect(url_for('login'))

    dept_id = session.get('department_id')
    first_name = request.form.get('first_name', '').strip()
    last_name = request.form.get('last_name', '').strip()
    email = request.form.get('email', '').strip().lower()
    phone = request.form.get('phone', '').strip()

    if not first_name or not email:
        flash("Please fill in First Name and Email.", "error")
        return redirect(url_for('department_head_engineers'))

    if not is_valid_email(email):
        flash("Please enter a valid email address (e.g. name@example.com).", "error")
        return redirect(url_for('department_head_engineers'))

    if phone and not is_valid_phone(phone):
        flash("Please enter a valid phone number (10 to 15 digits).", "error")
        return redirect(url_for('department_head_engineers'))

    connection = None
    cursor = None
    try:
        connection = get_connection()
        cursor = connection.cursor()

        cursor.execute("SELECT user_id FROM users WHERE LOWER(email) = LOWER(%s) AND user_id != %s;", (email, user_id))
        if cursor.fetchone():
            flash(f"Email '{email}' is already in use.", "error")
            return redirect(url_for('department_head_engineers'))

        cursor.execute("""
            UPDATE users
            SET first_name = %s, last_name = %s, email = %s, phone = %s
            WHERE user_id = %s AND role = 'Engineer' AND department_id = %s;
        """, (first_name, last_name, email, phone, user_id, dept_id))
        connection.commit()

        flash("Engineer details updated successfully.", "success")
        return redirect(url_for('department_head_engineers'))
    except Exception as e:
        print("Edit Engineer Error:", e)
        if connection:
            connection.rollback()
        flash(f"Error updating engineer: {e}", "error")
        return redirect(url_for('department_head_engineers'))
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


@app.route('/department-head/engineers/reset-password/<int:user_id>', methods=['POST'])
def reset_engineer_password(user_id):
    if session.get('role') not in ('Department Head', 'department_head'):
        return redirect(url_for('login'))

    dept_id = session.get('department_id')
    temporary_password = request.form.get('temporary_password', '').strip()
    if not temporary_password:
        flash("Please enter a new temporary password.", "error")
        return redirect(url_for('department_head_engineers'))

    connection = None
    cursor = None
    try:
        connection = get_connection()
        cursor = connection.cursor()

        password_hash = generate_password_hash(temporary_password)

        cursor.execute("""
            UPDATE users
            SET password_hash = %s, must_change_password = TRUE
            WHERE user_id = %s AND role = 'Engineer' AND department_id = %s;
        """, (password_hash, user_id, dept_id))
        connection.commit()

        flash(
            f"Password reset successfully! Temporary password: "
            f"<code class='bg-white px-1.5 py-0.5 rounded text-amber-700 font-mono'>{temporary_password}</code>. "
            f"The engineer will be required to change it on their next login.",
            "success"
        )
        return redirect(url_for('department_head_engineers'))
    except Exception as e:
        print("Reset Engineer Password Error:", e)
        if connection:
            connection.rollback()
        flash(f"Error resetting password: {e}", "error")
        return redirect(url_for('department_head_engineers'))
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


@app.route('/department-head/engineers/toggle/<int:user_id>', methods=['POST'])
def toggle_engineer_status(user_id):
    if session.get('role') not in ('Department Head', 'department_head'):
        return redirect(url_for('login'))

    dept_id = session.get('department_id')
    connection = None
    cursor = None
    try:
        connection = get_connection()
        cursor = connection.cursor()

        cursor.execute("""
            UPDATE users
            SET is_active = NOT is_active
            WHERE user_id = %s AND role = 'Engineer' AND department_id = %s;
        """, (user_id, dept_id))
        connection.commit()

        flash("Engineer account status updated.", "success")
        return redirect(url_for('department_head_engineers'))
    except Exception as e:
        print("Toggle Engineer Status Error:", e)
        if connection:
            connection.rollback()
        flash(f"Error toggling status: {e}", "error")
        return redirect(url_for('department_head_engineers'))
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


# ==========================================
# DEPARTMENT HEAD: FIELD WORKERS MANAGEMENT
# ==========================================
@app.route('/department-head/field-workers')
def department_head_field_workers():
    if session.get('role') not in ('Department Head', 'department_head'):
        return redirect(url_for('login'))

    dept_id = session.get('department_id')
    if not dept_id:
        return "No department assigned to this account."

    connection = None
    cursor = None
    try:
        connection = get_connection()
        cursor = connection.cursor()

        cursor.execute("SELECT department_name FROM departments WHERE department_id = %s;", (dept_id,))
        dept_row = cursor.fetchone()
        department_name = dept_row[0] if dept_row else "Department"

        cursor.execute("""
            SELECT 
                u.user_id,
                u.first_name,
                u.last_name,
                u.email,
                u.phone,
                u.is_active,
                u.must_change_password,
                u.created_at,
                (SELECT COUNT(*) FROM maintenance_work m WHERE m.maintenance_staff_id = u.user_id AND m.work_status IN ('Assigned', 'In Progress', 'Pending Verification')) AS active_tasks,
                (SELECT COUNT(*) FROM maintenance_work m WHERE m.maintenance_staff_id = u.user_id AND m.work_status = 'Completed') AS completed_tasks
            FROM users u
            WHERE u.role = 'Maintenance' AND u.department_id = %s
            ORDER BY u.user_id ASC;
        """, (dept_id,))
        rows = cursor.fetchall()

        field_workers = []
        for r in rows:
            field_workers.append({
                'user_id': r[0],
                'first_name': r[1],
                'last_name': r[2],
                'email': r[3],
                'phone': r[4],
                'is_active': r[5],
                'must_change_password': r[6],
                'created_at': r[7],
                'active_tasks': r[8] or 0,
                'completed_tasks': r[9] or 0
            })

        total_workers = len(field_workers)
        active_workers = sum(1 for w in field_workers if w['is_active'])
        pending_password_changes = sum(1 for w in field_workers if w['must_change_password'])
        total_active_tasks = sum(w['active_tasks'] for w in field_workers)

        return render_template(
            'department_head_field_workers.html',
            field_workers=field_workers,
            department_name=department_name,
            total_workers=total_workers,
            active_workers=active_workers,
            pending_password_changes=pending_password_changes,
            total_active_tasks=total_active_tasks
        )
    except Exception as e:
        print("Dept Head Field Workers Error:", e)
        return f"Error loading field workers: {e}"
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


@app.route('/department-head/field-workers/add', methods=['POST'])
def add_field_worker():
    if session.get('role') not in ('Department Head', 'department_head'):
        return redirect(url_for('login'))

    dept_id = session.get('department_id')
    if not dept_id:
        return "No department assigned."

    first_name = request.form.get('first_name', '').strip()
    last_name = request.form.get('last_name', '').strip()
    email = request.form.get('email', '').strip().lower()
    phone = request.form.get('phone', '').strip()
    temporary_password = request.form.get('temporary_password', '').strip()

    target_url = request.referrer or url_for('department_head_field_workers')

    if not first_name or not email or not temporary_password:
        flash("Please fill in First Name, Email, and Temporary Password.", "error")
        return redirect(target_url)

    if not is_valid_email(email):
        flash("Please enter a valid email address (e.g. name@example.com).", "error")
        return redirect(target_url)

    if phone and not is_valid_phone(phone):
        flash("Please enter a valid phone number (10 to 15 digits).", "error")
        return redirect(target_url)

    connection = None
    cursor = None
    try:
        connection = get_connection()
        cursor = connection.cursor()

        cursor.execute("SELECT user_id FROM users WHERE LOWER(email) = LOWER(%s);", (email,))
        if cursor.fetchone():
            flash(f"A user with email '{email}' already exists.", "error")
            return redirect(target_url)

        password_hash = generate_password_hash(temporary_password)

        cursor.execute("""
            INSERT INTO users (
                first_name, last_name, email, phone, password_hash,
                role, department_id, is_active, must_change_password, created_at
            ) VALUES (
                %s, %s, %s, %s, %s,
                'Maintenance', %s, TRUE, TRUE, CURRENT_TIMESTAMP
            );
        """, (first_name, last_name, email, phone, password_hash, dept_id))
        connection.commit()

        flash(
            f"<strong>Field Worker added successfully!</strong><br>"
            f"Email: <code class='bg-white px-1.5 py-0.5 rounded text-blue-700 font-mono'>{email}</code><br>"
            f"Initial Temporary Password: <code class='bg-white px-1.5 py-0.5 rounded text-amber-700 font-mono'>{temporary_password}</code><br>"
            f"<em>(The worker must change this password upon their first login)</em>",
            "success"
        )
        return redirect(target_url)
    except Exception as e:
        print("Add Field Worker Error:", e)
        if connection:
            connection.rollback()
        flash(f"Error adding field worker: {e}", "error")
        return redirect(target_url)
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


@app.route('/department-head/field-workers/edit/<int:user_id>', methods=['POST'])
def edit_field_worker(user_id):
    if session.get('role') not in ('Department Head', 'department_head'):
        return redirect(url_for('login'))

    dept_id = session.get('department_id')
    first_name = request.form.get('first_name', '').strip()
    last_name = request.form.get('last_name', '').strip()
    email = request.form.get('email', '').strip().lower()
    phone = request.form.get('phone', '').strip()

    if not first_name or not email:
        flash("Please fill in First Name and Email.", "error")
        return redirect(url_for('department_head_field_workers'))

    if not is_valid_email(email):
        flash("Please enter a valid email address (e.g. name@example.com).", "error")
        return redirect(url_for('department_head_field_workers'))

    if phone and not is_valid_phone(phone):
        flash("Please enter a valid phone number (10 to 15 digits).", "error")
        return redirect(url_for('department_head_field_workers'))

    connection = None
    cursor = None
    try:
        connection = get_connection()
        cursor = connection.cursor()

        cursor.execute("SELECT user_id FROM users WHERE LOWER(email) = LOWER(%s) AND user_id != %s;", (email, user_id))
        if cursor.fetchone():
            flash(f"Email '{email}' is already in use.", "error")
            return redirect(url_for('department_head_field_workers'))

        cursor.execute("""
            UPDATE users
            SET first_name = %s, last_name = %s, email = %s, phone = %s
            WHERE user_id = %s AND role = 'Maintenance' AND department_id = %s;
        """, (first_name, last_name, email, phone, user_id, dept_id))
        connection.commit()

        flash("Field worker details updated successfully.", "success")
        return redirect(url_for('department_head_field_workers'))
    except Exception as e:
        print("Edit Field Worker Error:", e)
        if connection:
            connection.rollback()
        flash(f"Error updating field worker: {e}", "error")
        return redirect(url_for('department_head_field_workers'))
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


@app.route('/department-head/field-workers/reset-password/<int:user_id>', methods=['POST'])
def reset_field_worker_password(user_id):
    if session.get('role') not in ('Department Head', 'department_head'):
        return redirect(url_for('login'))

    dept_id = session.get('department_id')
    temporary_password = request.form.get('temporary_password', '').strip()
    if not temporary_password:
        flash("Please enter a new temporary password.", "error")
        return redirect(url_for('department_head_field_workers'))

    connection = None
    cursor = None
    try:
        connection = get_connection()
        cursor = connection.cursor()

        password_hash = generate_password_hash(temporary_password)

        cursor.execute("""
            UPDATE users
            SET password_hash = %s, must_change_password = TRUE
            WHERE user_id = %s AND role = 'Maintenance' AND department_id = %s;
        """, (password_hash, user_id, dept_id))
        connection.commit()

        flash(
            f"Password reset successfully! Temporary password: "
            f"<code class='bg-white px-1.5 py-0.5 rounded text-amber-700 font-mono'>{temporary_password}</code>. "
            f"The worker will be required to change it on their next login.",
            "success"
        )
        return redirect(url_for('department_head_field_workers'))
    except Exception as e:
        print("Reset Field Worker Password Error:", e)
        if connection:
            connection.rollback()
        flash(f"Error resetting password: {e}", "error")
        return redirect(url_for('department_head_field_workers'))
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


@app.route('/department-head/field-workers/toggle/<int:user_id>', methods=['POST'])
def toggle_field_worker_status(user_id):
    if session.get('role') not in ('Department Head', 'department_head'):
        return redirect(url_for('login'))

    dept_id = session.get('department_id')
    connection = None
    cursor = None
    try:
        connection = get_connection()
        cursor = connection.cursor()

        cursor.execute("""
            UPDATE users
            SET is_active = NOT is_active
            WHERE user_id = %s AND role = 'Maintenance' AND department_id = %s;
        """, (user_id, dept_id))
        connection.commit()

        flash("Field worker account status updated.", "success")
        return redirect(url_for('department_head_field_workers'))
    except Exception as e:
        print("Toggle Field Worker Status Error:", e)
        if connection:
            connection.rollback()
        flash(f"Error toggling status: {e}", "error")
        return redirect(url_for('department_head_field_workers'))
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


@app.route('/department-head/reports')
def department_head_reports():
    if session.get('role') not in ('Department Head', 'department_head'):
        return redirect(url_for('login'))

    return render_template(
        'department_head_reports.html',
        first_name=session.get('first_name'),
        last_name=session.get('last_name')
    )


@app.route('/department-head/profile', methods=['GET', 'POST'])
def department_head_profile():
    if session.get('role') not in ('Department Head', 'department_head'):
        return redirect(url_for('login'))

    user_id = session.get('user_id')
    if not user_id:
        return redirect(url_for('login'))

    connection = None
    cursor = None
    try:
        connection = get_connection()
        cursor = connection.cursor()

        if request.method == 'POST':
            first_name = request.form.get('first_name', '').strip()
            last_name = request.form.get('last_name', '').strip()
            phone = request.form.get('phone', '').strip()

            if not first_name:
                flash('First name is required.', 'error')
            else:
                cursor.execute("""
                    UPDATE users
                    SET first_name = %s,
                        last_name = %s,
                        phone = %s
                    WHERE user_id = %s
                """, (first_name, last_name, phone, user_id))
                connection.commit()

                session['first_name'] = first_name
                session['last_name'] = last_name
                flash('Profile updated successfully!', 'success')

        cursor.execute("""
            SELECT 
                u.first_name,
                u.last_name,
                u.email,
                u.phone,
                d.department_name,
                u.created_at,
                u.role
            FROM users u
            LEFT JOIN departments d ON u.department_id = d.department_id
            WHERE u.user_id = %s
        """, (user_id,))
        user_row = cursor.fetchone()

        if not user_row:
            flash('User record not found.', 'error')
            return redirect(url_for('department_head_dashboard'))

        user_data = {
            'first_name': user_row[0] or '',
            'last_name': user_row[1] or '',
            'email': user_row[2] or '',
            'phone': user_row[3] or '',
            'department_name': user_row[4] or 'Department',
            'created_at': user_row[5],
            'role': user_row[6]
        }

        return render_template(
            'department_head_profile.html',
            first_name=user_data['first_name'],
            last_name=user_data['last_name'],
            email=user_data['email'],
            phone=user_data['phone'],
            department_name=user_data['department_name'],
            created_at=user_data['created_at'],
            role=user_data['role']
        )
    except Exception as e:
        print("Department Head Profile Error:", repr(e))
        if connection:
            connection.rollback()
        flash('An error occurred while loading or updating your profile.', 'error')
        return redirect(url_for('department_head_dashboard'))
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


if __name__ == "__main__":
    app.run(debug=True)