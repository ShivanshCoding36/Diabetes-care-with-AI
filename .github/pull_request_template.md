## 📌 Description

This PR implements a **Personalized Risk History & Trend Tracking Dashboard** that enables users to track their diabetes risk predictions over time and visualize trends for better long-term health monitoring.

### Key Changes:
- Added new `/dashboard` route to display prediction history
- Implemented localStorage-based history tracking for predictions
- Created interactive trend chart using Chart.js to visualize risk over time
- Added statistics cards showing total predictions, high risk count, and low risk count
- Built history table displaying date, result, glucose, BMI, and age for each prediction
- Updated navigation menus across all pages to include Dashboard link
- Refactored prediction form to automatically save results to history
- Implemented clear history functionality with confirmation dialog
- Added empty state for users with no prediction history

Fixes: # (issue number, if applicable)

---

## 🔧 Type of Change

- [ ] 🐛 Bug fix
- [x] ✨ New feature
- [ ] 📝 Documentation update
- [x] ♻️ Refactor / Code cleanup
- [x] 🎨 UI / Styling change
- [ ] 🚀 Other (please describe):

---

## 🧪 How Has This Been Tested?

- [x] Manual testing
  - Tested prediction form submission and history saving
  - Verified dashboard displays correct statistics and charts
  - Tested clear history functionality
  - Verified empty state displays correctly
  - Tested navigation links across all pages
  - Verified dark mode compatibility
  - Tested responsive design on mobile devices
- [ ] Automated tests
- [ ] Not tested (please explain why)

**Test Steps:**
1. Navigate to `/index` and submit multiple predictions
2. Visit `/dashboard` to view history
3. Verify trend chart updates correctly
4. Test clear history button
5. Confirm data persists after browser refresh

---

## 📸 Screenshots (if applicable)

### Dashboard with History
<!-- Add screenshot of dashboard showing statistics, chart, and history table -->

### Empty State
<!-- Add screenshot of empty state when no predictions exist -->

### Trend Chart
<!-- Add screenshot of the trend visualization -->

---

## ✅ Checklist

- [x] My code follows the project's coding style
- [x] I have tested my changes
- [x] I have updated documentation where necessary
- [x] This PR does not introduce breaking changes

---

## 📝 Additional Notes

### Technical Implementation:
- Uses **localStorage** for client-side data persistence (no backend changes required)
- **Chart.js** library loaded via CDN for trend visualization
- Data structure stores date, result, and all input parameters
- Fully responsive design with Tailwind CSS
- Compatible with existing dark mode implementation

### Benefits:
- Enables long-term health monitoring and trend analysis
- Encourages proactive health management
- Improves user engagement with the platform
- Provides real-world healthcare value beyond single-session predictions

### Future Enhancements (Optional):
- Backend database integration for multi-device sync
- User authentication for personalized history
- Export history as PDF/CSV
- Advanced analytics and insights

### Files Changed:
- `app.py` - Added dashboard route
- `templates/index.html` - Refactored with history tracking
- `templates/home.html` - Added dashboard navigation link
- `templates/dashboard.html` - New dashboard page (created)
