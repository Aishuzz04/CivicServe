import time
import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def capture_duplicate_review_screen():
    print("Setting up Headless Chrome...")
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--window-size=1440,1200")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(options=options)

    try:
        base_url = "http://127.0.0.1:5000"
        
        # 1. Open Login Page
        print(f"Navigating to {base_url}/login ...")
        driver.get(f"{base_url}/login")
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.NAME, "email")))

        # 2. Enter Admin Credentials
        print("Logging in as Administrator (admin@test.com)...")
        driver.find_element(By.NAME, "email").send_keys("admin@test.com")
        driver.find_element(By.NAME, "password").send_keys("Admin@123")
        driver.find_element(By.CSS_SELECTOR, "button[type='submit']").click()

        # 3. Wait for redirect
        time.sleep(2)
        print("Current URL after login:", driver.current_url)

        # 4. Navigate to Duplicate / Verification Review Page
        print(f"Navigating to {base_url}/admin/verify-complaints ...")
        driver.get(f"{base_url}/admin/verify-complaints")
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "main")))
        time.sleep(2)

        # 5. Define Screenshot Output Paths
        artifact_dir = r"C:\Users\aiswa\.gemini\antigravity\brain\67df713b-867d-4302-8712-ad264af8ca37"
        artifact_path = os.path.join(artifact_dir, "fig_8_6_duplicate_review_screen.png")
        static_img_dir = r"C:\Users\aiswa\PycharmProjects\smart_public_infra\static\images"
        os.makedirs(static_img_dir, exist_ok=True)
        static_img_path = os.path.join(static_img_dir, "fig_8_6_duplicate_review_screen.png")

        # 6. Save Screenshots
        driver.save_screenshot(artifact_path)
        driver.save_screenshot(static_img_path)
        print(f"[+] Screenshot successfully saved to:\n  - {artifact_path}\n  - {static_img_path}")

        # Check page title and verify content elements
        page_source = driver.page_source
        has_duplicate_badge = "Overall Similarity" in page_source or "Possible Similar Complaint" in page_source or "Linked Duplicate" in page_source
        print("[+] Duplicate detection elements present in page:", has_duplicate_badge)

    finally:
        driver.quit()

if __name__ == "__main__":
    capture_duplicate_review_screen()
