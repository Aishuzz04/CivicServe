import os
import sys
import io
import json

sys.path.insert(0, ".")
from app import app
from database import get_connection
from config import Config

def run_e2e_tests():
    print("==================================================================")
    print("   CIVICSERVE COMPREHENSIVE E2E IMAGE VALIDATION TEST SUITE       ")
    print("==================================================================")

    # Set up Flask test client
    app.config["TESTING"] = True
    client = app.test_client()

    # Find a Citizen user_id from the database
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT user_id, first_name, email FROM users WHERE role = 'Citizen' LIMIT 1;")
    user_row = cur.fetchone()
    if not user_row:
        # Create a citizen user for test
        cur.execute("INSERT INTO users(first_name, last_name, email, password_hash, role) VALUES ('Test', 'Citizen', 'test_citizen_cv@example.com', 'dummy_hash', 'Citizen') RETURNING user_id;")
        user_id = cur.fetchone()[0]
        first_name = "Test"
        conn.commit()
    else:
        user_id, first_name, _ = user_row
    
    cur.close()
    conn.close()

    print(f"Using Citizen user_id: {user_id} ({first_name})")

    # Helper to authenticate session in test client
    def set_session(c, uid, name):
        with c.session_transaction() as sess:
            sess["user_id"] = uid
            sess["role"] = "Citizen"
            sess["first_name"] = name

    # -------------------------------------------------------------
    # TEST 1: API Endpoint with Relevant Road Pothole Photo
    # -------------------------------------------------------------
    print("\n[TEST 1] POST /api/validate-complaint-image (Relevant Road Pothole)")
    set_session(client, user_id, first_name)
    with open("test_fixtures/real_road_pothole.jpg", "rb") as f:
        img_bytes = f.read()

    res = client.post(
        "/api/validate-complaint-image",
        data={
            "image": (io.BytesIO(img_bytes), "pothole.jpg", "image/jpeg"),
            "category": "Roads",
            "title": "Severe pothole on highway",
            "description": "Hazardous road surface damage in Peruva"
        },
        content_type="multipart/form-data"
    )
    print("  Status code:", res.status_code)
    data = res.get_json()
    print("  Decision:", data.get("decision"), "| Can Submit:", data.get("can_submit"))
    print("  Relevance:", data.get("relevance", {}).get("status"), f"({data.get('relevance', {}).get('score')}%)")
    print("  AI Detection:", data.get("ai_detection", {}).get("status"), f"(AI Score: {data.get('ai_detection', {}).get('ai_score')}%)")
    assert res.status_code == 200
    assert data["can_submit"] is True
    assert data["decision"] == "allow"
    assert data["relevance"]["status"] == "Relevant"
    print("  -> PASSED: API verified relevant road photo.")

    # -------------------------------------------------------------
    # TEST 2: API Endpoint with Unrelated Pet Photo
    # -------------------------------------------------------------
    print("\n[TEST 2] POST /api/validate-complaint-image (Unrelated Dog Pet Image vs Roads)")
    with open("test_fixtures/unrelated_dog_pet.jpg", "rb") as f:
        dog_bytes = f.read()

    res = client.post(
        "/api/validate-complaint-image",
        data={
            "image": (io.BytesIO(dog_bytes), "dog.jpg", "image/jpeg"),
            "category": "Roads",
            "title": "Pothole problem",
            "description": "Road asphalt broken"
        },
        content_type="multipart/form-data"
    )
    data = res.get_json()
    print("  Status code:", res.status_code)
    print("  Decision:", data.get("decision"), "| Can Submit:", data.get("can_submit"))
    print("  Message:", data.get("relevance", {}).get("message"))
    assert res.status_code == 200
    assert data["can_submit"] is False
    assert data["decision"] == "block_unrelated"
    print("  -> PASSED: API blocked unrelated pet photo.")

    # -------------------------------------------------------------
    # TEST 3: Form Submission with Relevant Road Photo (Mandatory Server-Side Check)
    # -------------------------------------------------------------
    print("\n[TEST 3] POST /register-complaint (Relevant Road Pothole)")
    res = client.post(
        "/register-complaint",
        data={
            "category": "Roads",
            "title": "Road pothole near bus stand",
            "description": "Deep dangerous pothole damaging vehicles daily.",
            "location": "Peruva Junction",
            "road_name": "Main Highway",
            "landmark": "Near bus stand",
            "location_details": "Right in front of bus stand gate",
            "latitude": "9.5916",
            "longitude": "76.5222",
            "image": (io.BytesIO(img_bytes), "pothole.jpg", "image/jpeg")
        },
        content_type="multipart/form-data"
    )
    html = res.data.decode("utf-8")
    print("  Status code:", res.status_code)
    assert res.status_code == 200
    assert ("Complaint submitted successfully" in html or "Similar Complaint Found" in html or "successfully registered" in html)
    # Verify in Database that complaint was inserted with image verification metrics
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT complaint_id, is_ai_suspected, ai_detection_score, image_relevance_score, image_relevance_status
        FROM complaints 
        WHERE user_id = %s 
        ORDER BY complaint_id DESC LIMIT 1;
    """, (user_id,))
    c_row = cur.fetchone()
    print(f"  Inserted DB Record: Complaint #{c_row[0]}, AI Suspected: {c_row[1]}, AI Score: {c_row[2]}%, Rel Score: {c_row[3]}%, Rel Status: {c_row[4]}")
    assert c_row is not None
    assert c_row[1] is False
    assert c_row[4] == "Relevant"
    cur.close()
    conn.close()
    print("  -> PASSED: Successfully submitted complaint and saved image verification metrics to DB.")

    # -------------------------------------------------------------
    # TEST 4: Form Submission with Clearly Unrelated Image (Mandatory Server-Side Block)
    # -------------------------------------------------------------
    print("\n[TEST 4] POST /register-complaint with Unrelated Dog Image (MUST BE BLOCKED)")
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM complaints;")
    count_before = cur.fetchone()[0]
    cur.close()
    conn.close()

    res = client.post(
        "/register-complaint",
        data={
            "category": "Roads",
            "title": "Road broken here",
            "description": "Pothole on the road",
            "location": "Piravom",
            "road_name": "Town Road",
            "landmark": "Opposite shop",
            "location_details": "Near cross junction",
            "image": (io.BytesIO(dog_bytes), "dog.jpg", "image/jpeg")
        },
        content_type="multipart/form-data"
    )
    html = res.data.decode("utf-8")
    print("  Status code:", res.status_code)
    assert res.status_code == 200
    assert "Image Verification Failed" in html or "depict a pet animal" in html
    assert "Complaint submitted successfully" not in html

    # Verify Database count did NOT increase
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM complaints;")
    count_after = cur.fetchone()[0]
    cur.close()
    conn.close()
    assert count_before == count_after
    print("  -> PASSED: Mandatory server-side validation strictly blocked unrelated photo from DB!")

    # -------------------------------------------------------------
    # TEST 5: Form Submission with Corrupted File Bytes (Security Guard)
    # -------------------------------------------------------------
    print("\n[TEST 5] POST /register-complaint with Corrupted Image Bytes (Security Guard)")
    corrupted_data = b"NOT_A_REAL_IMAGE_DATA_CORRUPTED_BYTES"
    res = client.post(
        "/register-complaint",
        data={
            "category": "Roads",
            "title": "Road issue",
            "description": "Road problem",
            "location": "Town",
            "road_name": "Main Street",
            "landmark": "Corner",
            "location_details": "Exact corner",
            "image": (io.BytesIO(corrupted_data), "corrupted.jpg", "image/jpeg")
        },
        content_type="multipart/form-data"
    )
    html = res.data.decode("utf-8")
    assert "Corrupted or invalid image file" in html or "validation failed" in html
    assert "Complaint submitted successfully" not in html
    print("  -> PASSED: Corrupted file was safely intercepted and blocked.")

    # -------------------------------------------------------------
    # TEST 6: Category Mismatch (Pothole submitted for Street Lights)
    # -------------------------------------------------------------
    print("\n[TEST 6] POST /register-complaint with Category Mismatch (Pothole submitted for Street Lights)")
    res = client.post(
        "/register-complaint",
        data={
            "category": "Street Lights",
            "title": "Streetlight dark",
            "description": "Light pole completely dead at night",
            "location": "Market Road",
            "road_name": "Market Road",
            "landmark": "Near bank",
            "location_details": "Pole SL-10",
            "image": (io.BytesIO(img_bytes), "pothole.jpg", "image/jpeg")
        },
        content_type="multipart/form-data"
    )
    html = res.data.decode("utf-8")
    assert "show Roads instead of the selected category" in html or "Verification Failed" in html
    assert "Complaint submitted successfully" not in html
    print("  -> PASSED: Category mismatch blocked.")

    # -------------------------------------------------------------
    # TEST 7: AI-Generated Image under 'flag' policy vs 'reject' policy
    # -------------------------------------------------------------
    print("\n[TEST 7] AI-Generated Image Detection Policy")
    with open("test_fixtures/ai_waifu.png", "rb") as f:
        ai_bytes = f.read()

    # Under 'flag' policy:
    Config.AI_IMAGE_POLICY = "flag"
    res_flag = client.post(
        "/api/validate-complaint-image",
        data={
            "image": (io.BytesIO(ai_bytes), "ai.png", "image/png"),
            "category": "Roads",
            "title": "AI Image test",
            "description": "Test"
        },
        content_type="multipart/form-data"
    )
    data_flag = res_flag.get_json()
    print("  Policy 'flag' -> AI Score:", data_flag.get("ai_detection", {}).get("ai_score"), "% | Decision:", data_flag.get("decision"))
    assert data_flag["ai_detection"]["is_ai_suspected"] is True

    # Under 'reject' policy:
    Config.AI_IMAGE_POLICY = "reject"
    # Test server-side check with policy = reject
    res_reject = client.post(
        "/register-complaint",
        data={
            "category": "Roads",
            "title": "AI Image complaint",
            "description": "AI test",
            "location": "Town",
            "road_name": "Road",
            "landmark": "Mark",
            "location_details": "Spot",
            "image": (io.BytesIO(ai_bytes), "ai.png", "image/png")
        },
        content_type="multipart/form-data"
    )
    html_rej = res_reject.data.decode("utf-8")
    assert "Complaint submitted successfully" not in html_rej
    assert "Verification Failed" in html_rej or "Submission Blocked" in html_rej
    Config.AI_IMAGE_POLICY = "flag"  # Reset
    print("  -> PASSED: AI-generated detection and policy enforcement confirmed.")

    # -------------------------------------------------------------
    # TEST 8: Admin Verification Page renders CV badges
    # -------------------------------------------------------------
    print("\n[TEST 8] GET /admin/verify-complaints (Admin CV Verification Badges)")
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT user_id, first_name FROM users WHERE role = 'Administrator' LIMIT 1;")
    admin_row = cur.fetchone()
    cur.close()
    conn.close()
    admin_id = admin_row[0] if admin_row else 1
    admin_name = admin_row[1] if admin_row else "Admin"

    with client.session_transaction() as sess:
        sess["user_id"] = admin_id
        sess["role"] = "Administrator"
        sess["first_name"] = admin_name

    res_admin = client.get("/admin/verify-complaints", follow_redirects=True)
    html_admin = res_admin.data.decode("utf-8")
    assert res_admin.status_code == 200
    assert "Match:" in html_admin or "Relevance" in html_admin or "Authentic" in html_admin or "AI" in html_admin
    print("  -> PASSED: Admin Verification portal renders CV verification badges.")

    print("\n==================================================================")
    print("   ALL 8 END-TO-END IMAGE VALIDATION TESTS PASSED SUCCESSFULLY!   ")
    print("==================================================================")

if __name__ == "__main__":
    run_e2e_tests()
