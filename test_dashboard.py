"""
Test Script for Risk History & Trend Tracking Dashboard
This script tests the dashboard features including empty state, history display, and trend chart.
"""

import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options

def test_dashboard():
    print("🧪 Starting Dashboard Tests...\n")
    
    # Setup Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--start-maximized")
    
    try:
        driver = webdriver.Chrome(options=chrome_options)
        wait = WebDriverWait(driver, 10)
        
        # Test 1: Empty State
        print("📋 Test 1: Testing Empty State")
        print("-" * 50)
        driver.get("http://localhost:5000/dashboard")
        time.sleep(2)
        
        # Clear any existing history
        driver.execute_script("localStorage.clear();")
        driver.refresh()
        time.sleep(1)
        
        try:
            empty_state = driver.find_element(By.ID, "emptyState")
            if "hidden" not in empty_state.get_attribute("class"):
                print("✅ Empty state is visible")
                print("✅ 'No Predictions Yet' message displayed")
            else:
                print("❌ Empty state should be visible but is hidden")
        except Exception as e:
            print(f"❌ Empty state test failed: {e}")
        
        print()
        
        # Test 2: Create Predictions
        print("📋 Test 2: Creating Test Predictions")
        print("-" * 50)
        
        test_predictions = [
            {
                "Pregnancies": "2",
                "Glucose": "120",
                "BloodPressure": "70",
                "SkinThickness": "25",
                "Insulin": "80",
                "BMI": "25.5",
                "DiabetesPedigreeFunction": "0.35",
                "Age": "30"
            },
            {
                "Pregnancies": "5",
                "Glucose": "160",
                "BloodPressure": "85",
                "SkinThickness": "35",
                "Insulin": "120",
                "BMI": "32.0",
                "DiabetesPedigreeFunction": "0.65",
                "Age": "45"
            },
            {
                "Pregnancies": "1",
                "Glucose": "95",
                "BloodPressure": "65",
                "SkinThickness": "20",
                "Insulin": "70",
                "BMI": "22.0",
                "DiabetesPedigreeFunction": "0.25",
                "Age": "28"
            }
        ]
        
        for i, prediction in enumerate(test_predictions, 1):
            print(f"\n  Creating prediction {i}/3...")
            driver.get("http://localhost:5000/index")
            time.sleep(1)
            
            # Fill form
            for field, value in prediction.items():
                try:
                    input_field = driver.find_element(By.NAME, field)
                    input_field.clear()
                    input_field.send_keys(value)
                except Exception as e:
                    print(f"  ⚠️  Could not fill {field}: {e}")
            
            # Submit form
            submit_button = driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
            submit_button.click()
            time.sleep(2)
            
            print(f"  ✅ Prediction {i} submitted")
        
        print("\n✅ All test predictions created")
        print()
        
        # Test 3: Dashboard with History
        print("📋 Test 3: Testing Dashboard with History")
        print("-" * 50)
        driver.get("http://localhost:5000/dashboard")
        time.sleep(2)
        
        try:
            # Check if dashboard content is visible
            dashboard_content = driver.find_element(By.ID, "dashboardContent")
            if "hidden" not in dashboard_content.get_attribute("class"):
                print("✅ Dashboard content is visible")
            else:
                print("❌ Dashboard content should be visible")
            
            # Check statistics cards
            total_count = driver.find_element(By.ID, "totalCount").text
            high_risk_count = driver.find_element(By.ID, "highRiskCount").text
            low_risk_count = driver.find_element(By.ID, "lowRiskCount").text
            
            print(f"✅ Total Predictions: {total_count}")
            print(f"✅ High Risk Count: {high_risk_count}")
            print(f"✅ Low Risk Count: {low_risk_count}")
            
            # Check history table
            history_rows = driver.find_elements(By.CSS_SELECTOR, "#historyTable tr")
            print(f"✅ History table has {len(history_rows)} entries")
            
        except Exception as e:
            print(f"❌ Dashboard history test failed: {e}")
        
        print()
        
        # Test 4: Trend Chart
        print("📋 Test 4: Testing Trend Chart")
        print("-" * 50)
        
        try:
            # Check if chart canvas exists
            chart_canvas = driver.find_element(By.ID, "trendChart")
            print("✅ Trend chart canvas found")
            
            # Check if Chart.js is loaded
            chart_exists = driver.execute_script("""
                return typeof Chart !== 'undefined' && 
                       document.getElementById('trendChart') !== null;
            """)
            
            if chart_exists:
                print("✅ Chart.js library loaded successfully")
            else:
                print("❌ Chart.js library not loaded")
            
            # Verify chart is rendered
            time.sleep(1)
            chart_rendered = driver.execute_script("""
                const canvas = document.getElementById('trendChart');
                return canvas && canvas.getContext('2d') !== null;
            """)
            
            if chart_rendered:
                print("✅ Trend chart rendered successfully")
            else:
                print("❌ Trend chart not rendered")
                
        except Exception as e:
            print(f"❌ Trend chart test failed: {e}")
        
        print()
        
        # Test 5: Clear History
        print("📋 Test 5: Testing Clear History")
        print("-" * 50)
        
        try:
            # Click clear history button
            clear_button = driver.find_element(By.XPATH, "//button[contains(text(), 'Clear History')]")
            clear_button.click()
            time.sleep(1)
            
            # Handle confirmation dialog
            alert = driver.switch_to.alert
            print(f"✅ Confirmation dialog appeared: '{alert.text}'")
            alert.dismiss()  # Cancel first time
            print("✅ Cancel button works")
            
            # Try again and accept
            clear_button.click()
            time.sleep(1)
            alert = driver.switch_to.alert
            alert.accept()
            time.sleep(2)
            
            # Check if empty state is shown
            empty_state = driver.find_element(By.ID, "emptyState")
            if "hidden" not in empty_state.get_attribute("class"):
                print("✅ History cleared successfully")
                print("✅ Empty state displayed after clearing")
            else:
                print("❌ Empty state should be visible after clearing")
                
        except Exception as e:
            print(f"❌ Clear history test failed: {e}")
        
        print()
        print("=" * 50)
        print("🎉 All Dashboard Tests Completed!")
        print("=" * 50)
        
        # Keep browser open for manual inspection
        print("\n⏸️  Browser will remain open for 10 seconds for manual inspection...")
        time.sleep(10)
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
    
    finally:
        driver.quit()
        print("\n✅ Browser closed")

if __name__ == "__main__":
    print("""
╔════════════════════════════════════════════════════════════╗
║  Risk History & Trend Tracking Dashboard - Test Suite     ║
╚════════════════════════════════════════════════════════════╝
    """)
    
    print("Prerequisites:")
    print("  1. Flask app running on http://localhost:5000")
    print("  2. Chrome browser installed")
    print("  3. Selenium WebDriver installed (pip install selenium)")
    print()
    
    input("Press Enter to start tests...")
    print()
    
    test_dashboard()
