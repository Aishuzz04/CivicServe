import time
import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

ROLES_CONFIG = [
    {
        "role": "Citizen",
        "email": "anu@test.com",
        "password": "Citizen@123",
        "expected_badge": "Citizen Portal",
        "expected_dashboard": "/citizen-dashboard",
        "screenshot_name": "welcome_splash_citizen.png"
    },
    {
        "role": "Administrator",
        "email": "admin@test.com",
        "password": "Admin@123",
        "expected_badge": "Administrator Control Center",
        "expected_dashboard": "/admin-dashboard",
        "screenshot_name": "welcome_splash_admin.png"
    },
    {
        "role": "Department Head",
        "email": "streetlight.head@municipal.com",
        "password": "Head@123",
        "expected_badge": "Department Head Workspace",
        "expected_dashboard": "/department-head-dashboard",
        "screenshot_name": "welcome_splash_dept_head.png"
    },
    {
        "role": "Engineer",
        "email": "arun@test.com",
        "password": "Engineer@123",
        "expected_badge": "Field Engineering Console",
        "expected_dashboard": "/engineer-dashboard",
        "screenshot_name": "welcome_splash_engineer.png"
    },
    {
        "role": "Maintenance",
        "email": "rahul@test.com",
        "password": "Maintenance@123",
        "expected_badge": "Field Maintenance Operations",
        "expected_dashboard": "/maintenance-dashboard",
        "screenshot_name": "welcome_splash_maintenance.png"
    }
]

def run_tests():
    print("=" * 70)
    print(" CIVICSERVE - WELCOME SPLASH SCREEN ROLE-BASED VERIFICATION")
    print("=" * 70)

    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--window-size=1440,1050")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")

    driver = webdriver.Chrome(options=options)
    base_url = "http://127.0.0.1:5000"
    artifact_dir = r"C:\Users\aiswa\.gemini\antigravity\brain\67df713b-867d-4302-8712-ad264af8ca37"
    static_img_dir = r"C:\Users\aiswa\PycharmProjects\smart_public_infra\static\images"
    os.makedirs(static_img_dir, exist_ok=True)

    results = []

    try:
        for idx, conf in enumerate(ROLES_CONFIG, 1):
            print(f"\n[{idx}/5] Testing Role: {conf['role']} ({conf['email']})")
            
            # Step 1: Open Login Page with clean session
            driver.delete_all_cookies()
            driver.get(f"{base_url}/login")
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.NAME, "email")))
            
            # Step 2: Submit Credentials
            driver.find_element(By.NAME, "email").clear()
            driver.find_element(By.NAME, "email").send_keys(conf["email"])
            driver.find_element(By.NAME, "password").clear()
            driver.find_element(By.NAME, "password").send_keys(conf["password"])
            driver.find_element(By.CSS_SELECTOR, "button[type='submit']").click()

            # Step 3 & 4: Verify arrival at /welcome and validate Welcome Card elements
            WebDriverWait(driver, 10).until(lambda d: "/welcome" in d.current_url)
            print("  -> Landed on:", driver.current_url)

            # Wait specifically for the splash card with expected badge to be rendered
            WebDriverWait(driver, 10).until(
                lambda d: conf["expected_badge"] in d.find_element(By.CLASS_NAME, "splash-card").text
            )
            card = driver.find_element(By.CLASS_NAME, "splash-card")
            card_text = card.text
            print(f"  -> Verified Badge: \"{conf['expected_badge']}\"")
            greeting_lines = [line for line in card_text.splitlines() if 'Welcome' in line]
            if greeting_lines:
                safe_text = greeting_lines[0].encode('ascii', errors='ignore').decode('ascii')
                print(f"  -> Greeting text found: {safe_text}")

            # Capture Screenshot of Welcome Splash Screen
            artifact_img = os.path.join(artifact_dir, conf["screenshot_name"])
            static_img = os.path.join(static_img_dir, conf["screenshot_name"])
            driver.save_screenshot(artifact_img)
            driver.save_screenshot(static_img)
            print(f"  -> Screenshot saved: {conf['screenshot_name']}")

            # Step 5: Wait for automatic 3.5s redirection to dashboard
            print("  -> Waiting for 3.5s auto-redirect...")
            WebDriverWait(driver, 10).until(lambda d: conf["expected_dashboard"] in d.current_url)
            print(f"  -> Successfully redirected to dashboard: {driver.current_url}")

            # Step 6: Test Security / One-Time Guard
            # Try to manually revisit /welcome directly
            print("  -> Testing direct manual access to /welcome after flag consumed...")
            driver.get(f"{base_url}/welcome")
            time.sleep(1)
            # Should immediately bounce back to dashboard
            assert conf["expected_dashboard"] in driver.current_url, f"Guard failed! Manual access stayed at {driver.current_url}"
            print(f"  -> Guard PASSED: Immediately bounced back to {driver.current_url}")

            results.append({
                "role": conf["role"],
                "badge": conf["expected_badge"],
                "redirect_url": driver.current_url,
                "screenshot": conf["screenshot_name"],
                "status": "PASSED"
            })

            # Logout before next test
            driver.get(f"{base_url}/logout")
            time.sleep(0.5)

    finally:
        driver.quit()

    print("\n" + "=" * 70)
    print(" VERIFICATION SUMMARY")
    print("=" * 70)
    for r in results:
        print(f"[*] {r['role']:<16} | Badge: {r['badge']:<30} | Status: {r['status']}")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
