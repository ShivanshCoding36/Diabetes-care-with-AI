# 🧪 Manual Testing Guide - Risk History & Trend Tracking Dashboard

## Prerequisites
- Flask app running on `http://localhost:5000`
- Modern web browser (Chrome, Firefox, Edge, or Safari)

---

## Test 1: Empty State ✅

### Steps:
1. Open browser console (F12)
2. Run: `localStorage.clear()`
3. Navigate to: `http://localhost:5000/dashboard`

### Expected Results:
- ✅ Empty state container is visible
- ✅ Icon showing empty clipboard/document
- ✅ Message: "No Predictions Yet"
- ✅ Description: "Start by making your first diabetes risk prediction"
- ✅ Blue "Make Prediction" button visible
- ✅ Dashboard content (stats, chart, table) is hidden

### Screenshot Checklist:
- [ ] Empty state displays correctly
- [ ] Button is clickable and navigates to `/index`

---

## Test 2: Creating Predictions 📝

### Steps:
1. Navigate to: `http://localhost:5000/index`
2. Fill in the form with these test values:

**Prediction 1 (Low Risk):**
- Pregnancies: `1`
- Glucose: `95`
- Blood Pressure: `65`
- Skin Thickness: `20`
- Insulin: `70`
- BMI: `22.0`
- Diabetes Pedigree: `0.25`
- Age: `28`

3. Click "Get Prediction"
4. Wait for result page
5. Note the prediction result

**Prediction 2 (High Risk):**
- Pregnancies: `5`
- Glucose: `160`
- Blood Pressure: `85`
- Skin Thickness: `35`
- Insulin: `120`
- BMI: `32.0`
- Diabetes Pedigree: `0.65`
- Age: `45`

6. Repeat steps 1-5

**Prediction 3 (Low Risk):**
- Pregnancies: `2`
- Glucose: `110`
- Blood Pressure: `70`
- Skin Thickness: `25`
- Insulin: `80`
- BMI: `24.5`
- Diabetes Pedigree: `0.35`
- Age: `32`

7. Repeat steps 1-5

### Expected Results:
- ✅ Form accepts all inputs
- ✅ Prediction result displays
- ✅ "View History" button appears (if implemented)
- ✅ Data is saved to localStorage

### Verify in Console:
```javascript
// Check localStorage
JSON.parse(localStorage.getItem('predictionHistory'))
// Should show array with 3 predictions
```

---

## Test 3: Dashboard with History 📊

### Steps:
1. Navigate to: `http://localhost:5000/dashboard`
2. Wait for page to load (2-3 seconds)

### Expected Results:

#### Statistics Cards:
- ✅ **Total Predictions**: Shows `3`
- ✅ **High Risk**: Shows count of "Diabetic" predictions
- ✅ **Low Risk**: Shows count of "Not Diabetic" predictions
- ✅ Cards have appropriate colors (blue, red, green)
- ✅ Icons display correctly

#### History Table:
- ✅ Table header shows: Date, Result, Glucose, BMI, Age
- ✅ 3 rows of data displayed
- ✅ Most recent prediction at the top
- ✅ Date/time formatted correctly
- ✅ Result shows colored badge (red for Diabetic, green for Not Diabetic)
- ✅ Glucose, BMI, Age values match input data
- ✅ Table is scrollable if needed

#### UI Elements:
- ✅ "Clear History" button visible in top-right
- ✅ Empty state is hidden
- ✅ Dashboard content is visible

### Screenshot Checklist:
- [ ] Statistics cards display correctly
- [ ] History table shows all predictions
- [ ] Layout is responsive

---

## Test 4: Trend Chart 📈

### Steps:
1. On dashboard page, scroll to "Risk Trend Over Time" section
2. Observe the chart

### Expected Results:
- ✅ Chart canvas is visible
- ✅ Line chart displays with blue line
- ✅ X-axis shows dates of predictions
- ✅ Y-axis shows "High Risk" (1) and "Low Risk" (0)
- ✅ Data points connect with smooth line
- ✅ Chart is responsive to window resize
- ✅ Hover over points shows tooltip (if Chart.js loaded)

### Verify in Console:
```javascript
// Check if Chart.js is loaded
typeof Chart !== 'undefined'
// Should return: true

// Check if chart instance exists
document.getElementById('trendChart')
// Should return: canvas element
```

### Screenshot Checklist:
- [ ] Chart renders correctly
- [ ] Line connects all data points
- [ ] Axes are labeled properly

---

## Test 5: Clear History 🗑️

### Steps:
1. On dashboard, click "Clear History" button
2. Observe confirmation dialog

### Expected Results:
- ✅ Browser confirmation dialog appears
- ✅ Message: "Are you sure you want to clear all prediction history?"
- ✅ Two options: OK and Cancel

#### Test Cancel:
3. Click "Cancel"
- ✅ Dialog closes
- ✅ History remains intact
- ✅ Dashboard still shows data

#### Test Confirm:
4. Click "Clear History" again
5. Click "OK"
- ✅ Page reloads
- ✅ Empty state is now visible
- ✅ Dashboard content is hidden
- ✅ Statistics show 0
- ✅ Chart is hidden
- ✅ Table is empty

### Verify in Console:
```javascript
// After clearing
localStorage.getItem('predictionHistory')
// Should return: null
```

---

## Test 6: Dark Mode Compatibility 🌙

### Steps:
1. Navigate to dashboard with history
2. Toggle dark mode (if available in navbar)

### Expected Results:
- ✅ Background changes to dark
- ✅ Text remains readable (light colors)
- ✅ Cards have dark background
- ✅ Chart adapts to dark theme
- ✅ Table rows have dark background
- ✅ Buttons maintain visibility
- ✅ No contrast issues

---

## Test 7: Responsive Design 📱

### Steps:
1. Open dashboard with history
2. Resize browser window to mobile size (375px width)
3. Or use browser DevTools device emulation

### Expected Results:
- ✅ Statistics cards stack vertically
- ✅ Table is horizontally scrollable
- ✅ Chart resizes appropriately
- ✅ Navigation menu adapts
- ✅ Text remains readable
- ✅ Buttons are touch-friendly
- ✅ No horizontal overflow

---

## Test 8: Navigation Links 🔗

### Steps:
1. From dashboard, click each navigation link

### Expected Results:
- ✅ "Home" → navigates to `/`
- ✅ "Prediction" → navigates to `/index`
- ✅ "Dashboard" → stays on `/dashboard` (highlighted)
- ✅ "Lifestyle" → navigates to `/life`
- ✅ "Chatbot" → navigates to `/chatbot`
- ✅ "Explore" → navigates to `/explore`
- ✅ "Forum" → navigates to `/forum`

---

## Test 9: Data Persistence 💾

### Steps:
1. Create 2-3 predictions
2. Navigate to dashboard
3. Close browser completely
4. Reopen browser
5. Navigate to dashboard

### Expected Results:
- ✅ History is still present
- ✅ Statistics show correct counts
- ✅ Chart displays previous data
- ✅ Table shows all previous predictions
- ✅ Data persists across sessions

---

## Test 10: Edge Cases 🔍

### Test Multiple Predictions Same Day:
1. Create 5+ predictions in quick succession
2. Check dashboard

**Expected:**
- ✅ All predictions saved
- ✅ Chart shows all points
- ✅ Table displays all entries
- ✅ No duplicate entries

### Test Very Old Predictions:
1. In console, manually add old prediction:
```javascript
let history = JSON.parse(localStorage.getItem('predictionHistory') || '[]');
history.push({
  date: '2023-01-01T00:00:00.000Z',
  result: 'Not Diabetic',
  inputs: {
    Pregnancies: '1', Glucose: '100', BloodPressure: '70',
    SkinThickness: '20', Insulin: '80', BMI: '23',
    DiabetesPedigreeFunction: '0.3', Age: '30'
  }
});
localStorage.setItem('predictionHistory', JSON.stringify(history));
location.reload();
```

**Expected:**
- ✅ Old prediction appears in table
- ✅ Chart includes old data point
- ✅ Date formatted correctly

---

## ✅ Test Summary Checklist

- [ ] Empty state displays correctly
- [ ] Predictions save to localStorage
- [ ] Dashboard shows statistics correctly
- [ ] History table displays all predictions
- [ ] Trend chart renders properly
- [ ] Clear history works with confirmation
- [ ] Dark mode compatible
- [ ] Responsive on mobile
- [ ] Navigation links work
- [ ] Data persists across sessions
- [ ] Edge cases handled properly

---

## 🐛 Bug Reporting Template

If you find any issues, report them with:

**Bug Title:** [Brief description]

**Steps to Reproduce:**
1. 
2. 
3. 

**Expected Behavior:**

**Actual Behavior:**

**Screenshots:**

**Browser:** [Chrome/Firefox/Safari/Edge]
**Version:** 
**OS:** [Windows/Mac/Linux]

---

## 📝 Notes

- All data is stored in browser's localStorage
- Clearing browser data will delete history
- No backend database required for this feature
- Chart.js loaded from CDN (requires internet)
