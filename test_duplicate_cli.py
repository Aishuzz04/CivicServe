import sys
import os

# Ensure the project directory is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database import get_connection
from app import (
    find_similar_complaints,
    calculate_tfidf_similarity,
    calculate_text_similarity,
    calculate_distance_km,
    calculate_location_similarity,
)

def run_duplicate_test():
    print("=" * 70)
    print(" CIVICSERVE - SMART DUPLICATE DETECTION VERIFICATION TEST")
    print("=" * 70)

    conn = get_connection()
    cursor = conn.cursor()

    test_complaint_ids = []

    try:
        # Find a valid citizen user id
        cursor.execute("SELECT user_id FROM users WHERE role = 'Citizen' LIMIT 1")
        user_row = cursor.fetchone()
        user_id = user_row[0] if user_row else 1

        # Find or use department for Water Supply
        cursor.execute("SELECT department_id FROM departments WHERE department_name ILIKE '%water%' LIMIT 1")
        dept_row = cursor.fetchone()
        dept_id = dept_row[0] if dept_row else None

        # -------------------------------------------------------------
        # 1. Define Test Complaints
        # -------------------------------------------------------------
        complaint_1 = {
            "title": "[TEST] Broken water pipe leaking on Main Road",
            "description": "Severe drinking water pipeline leak causing road flooding and low water pressure near City Hospital.",
            "category": "Water Supply",
            "location": "Main Road, Near City Hospital",
            "road_name": "Main Road",
            "landmark": "Near City Hospital",
            "location_details": "Opposite gate 2",
            "latitude": 9.931200,
            "longitude": 76.267300,
        }

        complaint_2 = {
            "title": "[TEST] Water leakage from pipe on Main Road",
            "description": "Huge water leakage from broken pipe flooding the street in front of City Hospital.",
            "category": "Water Supply",
            "location": "Main Road, Opp City Hospital",
            "road_name": "Main Road",
            "landmark": "Opp City Hospital",
            "location_details": "Near casualty entrance",
            "latitude": 9.931450,
            "longitude": 76.267520,
        }

        # -------------------------------------------------------------
        # 2. Insert Complaint 1 (Existing Baseline Complaint)
        # -------------------------------------------------------------
        cursor.execute(
            """
            INSERT INTO complaints (
                user_id, department_id, title, description, category,
                location, road_name, landmark, location_details,
                latitude, longitude, priority, status, verification_status,
                created_at, updated_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'Medium', 'Pending', 'Pending', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            RETURNING complaint_id;
            """,
            (
                user_id, dept_id,
                complaint_1["title"], complaint_1["description"], complaint_1["category"],
                complaint_1["location"], complaint_1["road_name"], complaint_1["landmark"], complaint_1["location_details"],
                complaint_1["latitude"], complaint_1["longitude"]
            )
        )
        c1_id = cursor.fetchone()[0]
        test_complaint_ids.append(c1_id)
        print(f"\n[+] Inserted Baseline Complaint #{c1_id}:")
        print(f"    Title:       {complaint_1['title']}")
        print(f"    Category:    {complaint_1['category']}")
        print(f"    Location:    {complaint_1['location']} ({complaint_1['latitude']}, {complaint_1['longitude']})")

        # -------------------------------------------------------------
        # 3. Insert Complaint 2 (New Candidate Duplicate Complaint)
        # -------------------------------------------------------------
        cursor.execute(
            """
            INSERT INTO complaints (
                user_id, department_id, title, description, category,
                location, road_name, landmark, location_details,
                latitude, longitude, priority, status, verification_status,
                created_at, updated_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'Medium', 'Pending', 'Pending', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            RETURNING complaint_id;
            """,
            (
                user_id, dept_id,
                complaint_2["title"], complaint_2["description"], complaint_2["category"],
                complaint_2["location"], complaint_2["road_name"], complaint_2["landmark"], complaint_2["location_details"],
                complaint_2["latitude"], complaint_2["longitude"]
            )
        )
        c2_id = cursor.fetchone()[0]
        test_complaint_ids.append(c2_id)
        print(f"\n[+] Inserted Candidate Complaint #{c2_id}:")
        print(f"    Title:       {complaint_2['title']}")
        print(f"    Category:    {complaint_2['category']}")
        print(f"    Location:    {complaint_2['location']} ({complaint_2['latitude']}, {complaint_2['longitude']})")

        conn.commit()

        # -------------------------------------------------------------
        # 4. Run Duplicate Detection Pipeline via find_similar_complaints
        # -------------------------------------------------------------
        print("\n" + "-" * 70)
        print(" EXECUTING find_similar_complaints() FOR COMPLAINT #" + str(c2_id))
        print("-" * 70)

        similar_matches = find_similar_complaints(
            cursor=cursor,
            complaint_id=c2_id,
            category=complaint_2["category"],
            title=complaint_2["title"],
            description=complaint_2["description"],
            location=complaint_2["location"],
            latitude=complaint_2["latitude"],
            longitude=complaint_2["longitude"]
        )

        print(f"\nTotal duplicate candidates flagged (score >= 60): {len(similar_matches)}")

        # Find match for our baseline complaint
        matched_c1 = next((m for m in similar_matches if m["complaint_id"] == c1_id), None)

        if not matched_c1:
            print(f"[!] Warning: Baseline Complaint #{c1_id} was not returned in candidates.")
            return

        # -------------------------------------------------------------
        # 5. Detailed Breakdown of the Calculation
        # -------------------------------------------------------------
        text1 = f"{complaint_2['title']} {complaint_2['description']}".strip()
        text2 = f"{complaint_1['title']} {complaint_1['description']}".strip()

        raw_text_score = calculate_tfidf_similarity(text1, text2)
        raw_cat_score = 100 if complaint_2["category"].strip().lower() == complaint_1["category"].strip().lower() else 0
        raw_dist_km = calculate_distance_km(
            complaint_2["latitude"], complaint_2["longitude"],
            complaint_1["latitude"], complaint_1["longitude"]
        )
        raw_coord_score = calculate_location_similarity(raw_dist_km)
        raw_loc_text_score = calculate_text_similarity(complaint_2["location"], complaint_1["location"])
        final_loc_score = max(raw_loc_text_score, raw_coord_score)

        weighted_text = round(raw_text_score * 0.60, 2)
        weighted_cat = round(raw_cat_score * 0.20, 2)
        weighted_loc = round(final_loc_score * 0.20, 2)
        total_score = round(weighted_text + weighted_cat + weighted_loc, 2)

        print("\n" + "=" * 70)
        print(f" SIMILARITY SCORE BREAKDOWN (Complaint #{c2_id} vs #{c1_id})")
        print("=" * 70)
        print(f"1. NLP Text Similarity (TF-IDF & Cosine Similarity) [60% Weight]:")
        print(f"   - Text A: \"{text2}\"")
        print(f"   - Text B: \"{text1}\"")
        print(f"   - Raw Score:            {raw_text_score:.2f}%")
        print(f"   - Weighted Contribution: {weighted_text:.2f} / 60.00")
        print()
        print(f"2. Category Similarity [20% Weight]:")
        print(f"   - Category A: \"{complaint_1['category']}\"")
        print(f"   - Category B: \"{complaint_2['category']}\"")
        print(f"   - Match:                 {'EXACT MATCH (100%)' if raw_cat_score == 100 else 'NO MATCH (0%)'}")
        print(f"   - Weighted Contribution: {weighted_cat:.2f} / 20.00")
        print()
        print(f"3. Location Similarity (Haversine Formula & Text) [20% Weight]:")
        print(f"   - Coordinates A:        ({complaint_1['latitude']}, {complaint_1['longitude']})")
        print(f"   - Coordinates B:        ({complaint_2['latitude']}, {complaint_2['longitude']})")
        print(f"   - Haversine Distance:   {raw_dist_km * 1000:.2f} meters ({raw_dist_km:.4f} km)")
        print(f"   - Coordinate Score:     {raw_coord_score:.2f}% (<= 100m band)")
        print(f"   - Text Location Score:  {raw_loc_text_score:.2f}%")
        print(f"   - Final Location Score: {final_loc_score:.2f}% (max of coord & text)")
        print(f"   - Weighted Contribution: {weighted_loc:.2f} / 20.00")
        print("-" * 70)
        print(f" TOTAL OVERALL SIMILARITY SCORE: {total_score:.2f}%")
        print(f" Algorithm Returned Score:     {matched_c1['similarity']:.2f}%")
        print(f" Duplicate Threshold Check:    {'PASS (>= 60% Duplicate Candidate)' if total_score >= 60 else 'FAIL'}")
        print(f" Auto-Link Threshold Check:    {'ELIGIBLE FOR AUTO-LINK (>= 75%)' if total_score >= 75 else 'NEEDS MANUAL ADMIN REVIEW'}")
        print("=" * 70)

        # -------------------------------------------------------------
        # 6. Verify Self-Exclusion & Rejected Exclusion
        # -------------------------------------------------------------
        print("\n" + "-" * 70)
        print(" VERIFYING FILTER CONSTRAINTS")
        print("-" * 70)

        # A) Self-exclusion test
        is_self_included = any(m["complaint_id"] == c2_id for m in similar_matches)
        print(f"[*] Self-Exclusion Test (c2_id != c2_id): {'PASSED (Not included)' if not is_self_included else 'FAILED'}")

        # B) Rejected status exclusion test
        cursor.execute("UPDATE complaints SET verification_status = 'Rejected' WHERE complaint_id = %s", (c1_id,))
        conn.commit()

        matches_after_reject = find_similar_complaints(
            cursor=cursor,
            complaint_id=c2_id,
            category=complaint_2["category"],
            title=complaint_2["title"],
            description=complaint_2["description"],
            location=complaint_2["location"],
            latitude=complaint_2["latitude"],
            longitude=complaint_2["longitude"]
        )
        is_rejected_included = any(m["complaint_id"] == c1_id for m in matches_after_reject)
        print(f"[*] Rejected Exclusion Test (status != 'Rejected'): {'PASSED (Excluded)' if not is_rejected_included else 'FAILED'}")

    finally:
        # -------------------------------------------------------------
        # 7. Clean up Test Data
        # -------------------------------------------------------------
        print("\n" + "-" * 70)
        print(" CLEANING UP TEST DATA")
        print("-" * 70)
        if test_complaint_ids:
            cursor.execute(
                "DELETE FROM complaint_similarity WHERE complaint_id = ANY(%s) OR similar_complaint_id = ANY(%s)",
                (test_complaint_ids, test_complaint_ids)
            )
            cursor.execute(
                "DELETE FROM complaints WHERE complaint_id = ANY(%s)",
                (test_complaint_ids,)
            )
            conn.commit()
            print(f"[+] Successfully purged test complaint IDs: {test_complaint_ids}")
        cursor.close()
        conn.close()

if __name__ == "__main__":
    run_duplicate_test()
