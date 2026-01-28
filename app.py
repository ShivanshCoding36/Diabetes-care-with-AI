import os
import base64
import io
import logging
import pickle
import re
import sys
import threading
import joblib
import json
import warnings
from datetime import datetime, timezone
from config import Config

# Suppress scikit-learn version warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
# --- External Libraries ---

import os
from dotenv import load_dotenv
load_dotenv()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from flask import Flask, jsonify, render_template, request, make_response, session, redirect, url_for, flash
from flask_cors import CORS
from flask_mail import Mail, Message
from flask_babel import Babel, gettext as _
from dotenv import load_dotenv

# --- New Auth Imports ---
from flask_login import LoginManager, login_user, current_user, logout_user, login_required
from flask_bcrypt import Bcrypt

# --- Database & New AI SDK Imports ---
from models import db, Post, User, PredictionHistory  # Feature #113: Database Model
# from google import genai     # Fix #112: New Google GenAI SDK
import google.generativeai as genai

# from google import genai     # Fix #112: New Google GenAI SDK
import google.generativeai as genai


# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config.from_object(Config)
# --- Database Configuration (Feature #113) ---
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///forum.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize Database
db.init_app(app)
with app.app_context():
    db.create_all()

# --- Auth Configuration (New) ---
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- App Configuration ---
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.jinja_env.auto_reload = True
CORS(app)

# --- i18n Configuration ---
SUPPORTED_LANGUAGES = {
    'en': 'English',
    'es': 'Español',
    'hi': 'हिन्दी',
    'fr': 'Français',
    'zh': '中文'
}
DEFAULT_LANGUAGE = 'en'

app.config['BABEL_DEFAULT_LOCALE'] = DEFAULT_LANGUAGE
app.config['BABEL_TRANSLATION_DIRECTORIES'] = 'translations'

babel = Babel(app)

def get_locale():
    # Check for user's language preference in cookie
    user_language = request.cookies.get('language')
    if user_language and user_language in SUPPORTED_LANGUAGES:
        return user_language
    
    # Fall back to browser's preferred language
    best_match = request.accept_languages.best_match(SUPPORTED_LANGUAGES.keys())
    if best_match:
        return best_match
    
    # Default to English
    return DEFAULT_LANGUAGE

babel.init_app(app, locale_selector=get_locale)

@app.context_processor
def inject_i18n_context():
    """Inject language context into all templates."""
    return {
        'languages': SUPPORTED_LANGUAGES,
        'current_language': get_locale(),
        'current_year': datetime.now().year,
        'current_user': current_user # Make user available in all templates
    }

@app.route('/api/set-language', methods=['POST'])
def set_language():
    """Set user's preferred language via cookie."""
    data = request.get_json()
    
    if not data or 'language' not in data:
        return jsonify({'error': 'Language code is required'}), 400
    
    language = data['language']
    
    if language not in SUPPORTED_LANGUAGES:
        return jsonify({
            'error': 'Unsupported language',
            'supported': list(SUPPORTED_LANGUAGES.keys())
        }), 400
    
    response = make_response(jsonify({
        'success': True,
        'language': language,
        'language_name': SUPPORTED_LANGUAGES[language]
    }))
    
    # Set cookie for 1 year
    response.set_cookie('language', language, max_age=365*24*60*60, samesite='Lax')
    
    return response

# --- Flask-Mail Configuration ---
app.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER', 'smtp.gmail.com')
app.config['MAIL_PORT'] = int(os.getenv('MAIL_PORT', 587))
app.config['MAIL_USE_TLS'] = os.getenv('MAIL_USE_TLS', 'true').lower() == 'true'
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.getenv('MAIL_DEFAULT_SENDER', 'noreply@diabetescare.com')

mail = Mail(app)

# --- Global Variables ---
FORUM_TOPICS = ['General', 'Diet', 'Exercise', 'Medication', 'Lifestyle', 'Support']
users = {}  # {user_id: {email, username, preferences, subscribed_posts}}
notifications = []  # [{id, user_id, type, message, post_id, read, timestamp}]
auth_users = {
    # email: {name, email, password_hash}
}


# Setup logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s: %(message)s')

# --- Load ML Model and Scaler ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "diabetes_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
try:
    model = joblib.load(MODEL_PATH)
    print(MODEL_PATH)
    print(os.path.exists(MODEL_PATH))
    logging.info("Diabetes model loaded successfully.")
except Exception as e:
    logging.error(f"Model load error: {e}")
    model = None

try:
    scaler = joblib.load(SCALER_PATH)
    logging.info("Scaler loaded successfully.")
except Exception as e:
    logging.error(f"Scaler load error: {e}")
    scaler = None

try:
    df = pd.read_csv('diabetes.csv')
    logging.info("Dataset loaded.")
except Exception as e:
    logging.error(f"CSV load error: {e}")
    df = None

# --- Gemini AI Client Initialization (Fix #112) ---
try:
    if app.config.get("GEMINI_API_KEY") and app.config["GEMINI_API_KEY"] != "your_gemini_api_key_here":
        client = genai.Client(api_key = app.config["GEMINI_API_KEY"])
        logging.info("Gemini Client initialized successfully.")
    else:
        client = None
        logging.warning("Gemini API Key not configured. Chatbot features will be limited.")
except Exception as e:
    logging.warning(f"Gemini Client initialization skipped: {e}")
    client = None

def get_gemini_response(user_message):
    """Get response from Gemini using the new SDK."""
    try:
        if not client:
            return "Error: Gemini API Key not found or Client not initialized."

        response = client.models.generate_content(
            model='gemini-2.0-flash', 
            contents=user_message
        )
        return response.text
    except Exception as e:
        logging.error(f"Gemini error: {e}")
        return "Sorry, Gemini service is unavailable right now."

# --- Routes ---

@app.route('/')
def root():
    return render_template('home.html')

@app.route('/index')
def home():
    return render_template('index.html')

# --- Authentication Routes ---

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.get_json() if request.is_json else request.form
        username = data.get('username')
        password = data.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            session['user_id'] = user.id
            session['username'] = user.username
            if request.is_json:
                return jsonify({"success": True, "message": "Login successful", "username": user.username})
            flash(_('Login successful!'), 'success')
            return redirect(url_for('dashboard'))
        else:
            if request.is_json:
                return jsonify({"success": False, "message": "Invalid username or password"}), 401
            flash(_('Invalid username or password'), 'error')
    
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        data = request.get_json() if request.is_json else request.form
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')
        
        # Validate input
        if not username or not email or not password:
            if request.is_json:
                return jsonify({"success": False, "message": "All fields are required"}), 400
            flash(_('All fields are required'), 'error')
            return render_template('signup.html')
        
        # Check if user already exists
        if User.query.filter_by(username=username).first():
            if request.is_json:
                return jsonify({"success": False, "message": "Username already exists"}), 400
            flash(_('Username already exists'), 'error')
            return render_template('signup.html')
        
        if User.query.filter_by(email=email).first():
            if request.is_json:
                return jsonify({"success": False, "message": "Email already registered"}), 400
            flash(_('Email already registered'), 'error')
            return render_template('signup.html')
        
        # Create new user
        new_user = User(username=username, email=email)
        new_user.set_password(password)
        
        try:
            db.session.add(new_user)
            db.session.commit()
            
            # Log in the user
            session['user_id'] = new_user.id
            session['username'] = new_user.username
            
            if request.is_json:
                return jsonify({"success": True, "message": "Account created successfully", "username": new_user.username})
            flash(_('Account created successfully!'), 'success')
            return redirect(url_for('dashboard'))
        except Exception as e:
            db.session.rollback()
            logging.error(f"Signup error: {e}")
            if request.is_json:
                return jsonify({"success": False, "message": "An error occurred"}), 500
            flash(_('An error occurred. Please try again.'), 'error')
    
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.clear()
    flash(_('Logged out successfully'), 'success')
    return redirect(url_for('root'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        flash(_('Please login to access dashboard'), 'error')
        return redirect(url_for('login'))
    
    user = User.query.get(session['user_id'])
    if not user:
        session.clear()
        return redirect(url_for('login'))
    
    # Get user's prediction history
    predictions = PredictionHistory.query.filter_by(user_id=user.id).order_by(PredictionHistory.timestamp.desc()).all()
    
    return render_template('dashboard.html', user=user, predictions=predictions)

@app.route('/api/dashboard/trend-data')
def dashboard_trend_data():
    """API endpoint to get trend data for charts"""
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    
    user_id = session['user_id']
    predictions = PredictionHistory.query.filter_by(user_id=user_id).order_by(PredictionHistory.timestamp.asc()).all()
    
    # Prepare data for charts
    trend_data = {
        "dates": [],
        "glucose": [],
        "bmi": [],
        "blood_pressure": [],
        "predictions": [],
        "risk_scores": []
    }
    
    for pred in predictions:
        trend_data["dates"].append(pred.timestamp.strftime('%Y-%m-%d %H:%M'))
        trend_data["glucose"].append(pred.glucose)
        trend_data["bmi"].append(pred.bmi)
        trend_data["blood_pressure"].append(pred.blood_pressure)
        trend_data["predictions"].append(pred.prediction)
        trend_data["risk_scores"].append(pred.risk_score if pred.risk_score else 0)
    
    # Calculate statistics
    stats = {
        "total_predictions": len(predictions),
        "diabetic_count": sum(1 for p in predictions if p.prediction == 1),
        "average_glucose": sum(p.glucose for p in predictions) / len(predictions) if predictions else 0,
        "average_bmi": sum(p.bmi for p in predictions) / len(predictions) if predictions else 0,
    }
    
    return jsonify({
        "trend_data": trend_data,
        "stats": stats
    })

def generate_prediction_explanation(result, confidence_score, feature_importance, features):
    """Generate a human-readable explanation for the prediction."""
    if confidence_score is None:
        return _("Prediction made, but confidence score unavailable.")
    
    # Risk level based on confidence
    if confidence_score >= 70:
        risk_level = _("High Risk")
    elif confidence_score >= 40:
        risk_level = _("Moderate Risk") 
    else:
        risk_level = _("Low Risk")
    
    explanation = _("Based on your health metrics, the AI predicts you are {result} with {confidence:.1f}% confidence ({risk_level}).\n\n").format(
        result=result.lower(), confidence=confidence_score, risk_level=risk_level)
    
    explanation += _("Key factors influencing this prediction:\n")
    
    for i, (feature_name, contribution, original_value) in enumerate(feature_importance[:3]):
        factor_type = _("increases") if contribution > 0 else _("decreases")
        explanation += _("• {feature}: {value} ({factor_type} diabetes risk)\n").format(
            feature=feature_name, value=original_value, factor_type=factor_type)
    
    # Add general advice
    if result == _("Diabetic"):
        explanation += _("\n⚠️ This suggests potential diabetes risk. Please consult a healthcare professional for proper diagnosis and guidance.")
    else:
        explanation += _("\n✅ This suggests low diabetes risk, but maintain healthy lifestyle habits.")
    
    return explanation

@app.route('/predict', methods=['POST'])
def predict():
    try:
        expected_features = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]
        features = [float(request.form.get(f, 0)) for f in expected_features]

        # --- Fix #115: Negative Validation ---
        if any(f < 0 for f in features):
             return render_template('index.html', prediction_text=_("Error: Input values cannot be negative."))

        if scaler is None or model is None:
            print(scaler)
            print(model)
            return render_template('index.html', prediction_text=_("Model not available."))

        final_input = scaler.transform([features])
        prediction = model.predict(final_input)
        
        # Get probability score if available
        risk_score = None
        if hasattr(model, 'predict_proba'):
            try:
                proba = model.predict_proba(final_input)
                risk_score = float(proba[0][1])  # Probability of being diabetic
            except:
                pass
        
        result = _("Diabetic") if prediction[0] == 1 else _("Not Diabetic")
        
        # --- EXPLAINABILITY AND CONFIDENCE (Issue #53) ---
        confidence_score = risk_score * 100 if risk_score is not None else None
        
        # Feature importance from model coefficients
        feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        coefficients = model.coef_[0]  # LogisticRegression coefficients
        
        # Calculate feature contributions (coefficient * scaled_feature_value)
        scaled_features = final_input[0]  # Already scaled
        feature_contributions = coefficients * scaled_features
        
        # Get top contributing features
        feature_importance = list(zip(feature_names, feature_contributions, features))
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Generate explanation
        explanation = generate_prediction_explanation(result, confidence_score, feature_importance, features)
        
        # Save prediction history for logged-in users
        if 'user_id' in session:
            try:
                prediction_record = PredictionHistory(
                    user_id=session['user_id'],
                    pregnancies=int(features[0]),
                    glucose=features[1],
                    blood_pressure=features[2],
                    skin_thickness=features[3],
                    insulin=features[4],
                    bmi=features[5],
                    dpf=features[6],
                    age=int(features[7]),
                    prediction=int(prediction[0]),
                    risk_score=risk_score
                )
                db.session.add(prediction_record)
                db.session.commit()
            except Exception as e:
                logging.error(f"Error saving prediction history: {e}")
                db.session.rollback()

        # --- SAVE HISTORY IF LOGGED IN (New) ---
        if current_user.is_authenticated:
            new_pred = Prediction(
                glucose=int(features[1]),
                bmi=float(features[5]),
                age=int(features[7]),
                result=result,
                author=current_user
            )
            db.session.add(new_pred)
            db.session.commit()

        return render_template('index.html', 
                             prediction_text=result,
                             confidence_score=confidence_score,
                             explanation=explanation,
                             top_features=feature_importance[:3])
    except Exception as e:
        logging.error(f"Predict error: {e}")
        return render_template('index.html', prediction_text=_("Error during prediction."))

# Rate limiting for email - simple in-memory store
email_rate_limit = {}  # {ip: [timestamps]}
EMAIL_RATE_LIMIT = 5  # max emails per hour
EMAIL_RATE_WINDOW = 3600  # 1 hour in seconds


@app.route('/api/send-prediction-email', methods=['POST'])
def send_prediction_email():
    """Send prediction results via email."""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        email = data.get('email', '').strip()
        prediction_result = data.get('prediction', '')
        input_values = data.get('inputValues', {})
        
        # Validate email
        if not email or '@' not in email or '.' not in email:
            return jsonify({'error': 'Please provide a valid email address'}), 400
        
        if not prediction_result:
            return jsonify({'error': 'No prediction result to send'}), 400
        
        # Rate limiting
        client_ip = request.remote_addr
        current_time = datetime.now().timestamp()
        
        if client_ip in email_rate_limit:
            # Clean old timestamps
            email_rate_limit[client_ip] = [
                ts for ts in email_rate_limit[client_ip] 
                if current_time - ts < EMAIL_RATE_WINDOW
            ]
            if len(email_rate_limit[client_ip]) >= EMAIL_RATE_LIMIT:
                return jsonify({'error': 'Rate limit exceeded. Please try again later.'}), 429
        else:
            email_rate_limit[client_ip] = []
        
        # Build email content
        report_date = datetime.now().strftime('%B %d, %Y at %I:%M %p')
        
        # HTML email template
        html_body = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; background-color: #f0f4f8; margin: 0; padding: 20px; }}
        .container {{ max-width: 600px; margin: 0 auto; background: white; border-radius: 16px; overflow: hidden; box-shadow: 0 4px 20px rgba(0,0,0,0.1); }}
        .header {{ background: linear-gradient(135deg, #2563eb, #4f46e5); color: white; padding: 30px; text-align: center; }}
        .header h1 {{ margin: 0; font-size: 24px; }}
        .header p {{ margin: 10px 0 0; opacity: 0.9; }}
        .content {{ padding: 30px; }}
        .result-box {{ background: linear-gradient(135deg, #06b6d4, #3b82f6); color: white; padding: 20px; border-radius: 12px; text-align: center; margin-bottom: 25px; }}
        .result-box h2 {{ margin: 0 0 10px; font-size: 18px; opacity: 0.9; }}
        .result-box p {{ margin: 0; font-size: 28px; font-weight: bold; }}
        .values-section {{ background: #f8fafc; border-radius: 12px; padding: 20px; margin-bottom: 25px; }}
        .values-section h3 {{ margin: 0 0 15px; color: #1e40af; font-size: 16px; }}
        .value-row {{ display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #e2e8f0; }}
        .value-row:last-child {{ border-bottom: none; }}
        .value-label {{ color: #64748b; }}
        .value-data {{ color: #1e293b; font-weight: 600; }}
        .disclaimer {{ background: #fef3c7; border: 1px solid #f59e0b; border-radius: 8px; padding: 15px; margin-bottom: 20px; }}
        .disclaimer p {{ margin: 0; color: #92400e; font-size: 13px; }}
        .footer {{ text-align: center; padding: 20px; color: #64748b; font-size: 12px; border-top: 1px solid #e2e8f0; }}
        .footer a {{ color: #2563eb; text-decoration: none; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>⚕️ Diabetes Prediction Report</h1>
            <p>{report_date}</p>
        </div>
        <div class="content">
            <div class="result-box">
                <h2>PREDICTION RESULT</h2>
                <p>{prediction_result}</p>
            </div>
            <div class="values-section">
                <h3>📊 Input Values</h3>
                <div class="value-row"><span class="value-label">Pregnancies</span><span class="value-data">{input_values.get('Pregnancies', 'N/A')}</span></div>
                <div class="value-row"><span class="value-label">Glucose Level</span><span class="value-data">{input_values.get('Glucose', 'N/A')} mg/dL</span></div>
                <div class="value-row"><span class="value-label">Blood Pressure</span><span class="value-data">{input_values.get('BloodPressure', 'N/A')} mm Hg</span></div>
                <div class="value-row"><span class="value-label">Skin Thickness</span><span class="value-data">{input_values.get('SkinThickness', 'N/A')} mm</span></div>
                <div class="value-row"><span class="value-label">Insulin Level</span><span class="value-data">{input_values.get('Insulin', 'N/A')} μU/mL</span></div>
                <div class="value-row"><span class="value-label">BMI</span><span class="value-data">{input_values.get('BMI', 'N/A')} kg/m²</span></div>
                <div class="value-row"><span class="value-label">Diabetes Pedigree</span><span class="value-data">{input_values.get('DiabetesPedigreeFunction', 'N/A')}</span></div>
                <div class="value-row"><span class="value-label">Age</span><span class="value-data">{input_values.get('Age', 'N/A')} years</span></div>
            </div>
            <div class="disclaimer">
                <p><strong>⚠️ Disclaimer:</strong> This prediction is for informational purposes only and should not be considered medical advice. Please consult a healthcare professional for proper diagnosis and treatment.</p>
            </div>
        </div>
        <div class="footer">
            <p>Sent from <a href="#">Diabetes Care with AI</a></p>
            <p>© {datetime.now().year} Diabetes Care Platform</p>
        </div>
    </div>
</body>
</html>
"""
        
        # Plain text fallback
        text_body = f"""
DIABETES PREDICTION REPORT
==========================
Date: {report_date}

RESULT: {prediction_result}

INPUT VALUES:
- Pregnancies: {input_values.get('Pregnancies', 'N/A')}
- Glucose Level: {input_values.get('Glucose', 'N/A')} mg/dL
- Blood Pressure: {input_values.get('BloodPressure', 'N/A')} mm Hg
- Skin Thickness: {input_values.get('SkinThickness', 'N/A')} mm
- Insulin Level: {input_values.get('Insulin', 'N/A')} μU/mL
- BMI: {input_values.get('BMI', 'N/A')} kg/m²
- Diabetes Pedigree Function: {input_values.get('DiabetesPedigreeFunction', 'N/A')}
- Age: {input_values.get('Age', 'N/A')} years

DISCLAIMER: This prediction is for informational purposes only
and should not be considered medical advice. Please consult a
healthcare professional for proper diagnosis.

---
Sent from Diabetes Care Platform
"""
        
        # Check if mail is configured
        if not app.config.get('MAIL_USERNAME'):
            return jsonify({'error': 'Email service is not configured'}), 503
        
        # Send email
        msg = Message(
            subject=f"Your Diabetes Prediction Report - {datetime.now().strftime('%B %d, %Y')}",
            recipients=[email],
            body=text_body,
            html=html_body
        )
        mail.send(msg)
        
        # Record for rate limiting
        email_rate_limit[client_ip].append(current_time)
        
        logging.info(f"Prediction report sent to {email}")
        return jsonify({'success': True, 'message': 'Report sent successfully!'})
        
    except Exception as e:
        logging.error(f"Email send error: {e}")
        return jsonify({'error': 'Failed to send email. Please try again later.'}), 500


@app.route('/explore')
def explore():
    if df is None:
        return "Dataset not loaded", 500
    try:
        corr = df.corr(numeric_only=True)
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title('Correlation Matrix')
        plt.tight_layout()
        img = io.BytesIO()
        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)
        heatmap_url = base64.b64encode(img.getvalue()).decode()

        fig_glucose = px.histogram(df, x="Glucose", nbins=20, title="Glucose Distribution")
        fig_bmi = px.histogram(df, x="BMI", nbins=20, title="BMI Distribution")

        return render_template('explore.html',
                               heatmap_url=heatmap_url,
                               hist_glucose_html=fig_glucose.to_html(full_html=False, include_plotlyjs='cdn'),
                               hist_bmi_html=fig_bmi.to_html(full_html=False, include_plotlyjs='cdn'))
    except Exception as e:
        logging.error(f"Explore error: {e}")
        return "Error generating plots", 500

@app.route('/chatbot')
def chatbot_page():
    return render_template('chatbot.html')

@app.route('/life')
def life():
    return render_template('life.html')

def generate_fallback_plan(age, bmi, bmi_category, activity, age_group):
    """Generate a rule-based lifestyle plan as fallback."""
    plan = {'diet': [], 'exercise': [], 'stress': [], 'sleep': [], 'monitoring': []}
    
    # Diet recommendations based on BMI
    if bmi_category == 'underweight':
        plan['diet'] = [
            'Increase caloric intake with nutrient-dense foods like nuts, avocados, and whole grains.',
            'Eat 5-6 smaller meals throughout the day to boost calorie consumption.',
            'Include protein-rich foods at every meal to support healthy weight gain.',
            'Consider healthy smoothies with fruits, yogurt, and nut butters.'
        ]
    elif bmi_category == 'normal weight':
        plan['diet'] = [
            'Maintain a balanced diet with plenty of vegetables, lean proteins, and whole grains.',
            'Practice portion control and mindful eating habits.',
            'Limit processed foods and added sugars to prevent blood sugar spikes.',
            'Stay hydrated with water and limit sugary beverages.'
        ]
    elif bmi_category == 'overweight':
        plan['diet'] = [
            'Focus on a moderate calorie deficit with nutrient-dense, low-glycemic foods.',
            'Fill half your plate with non-starchy vegetables at each meal.',
            'Choose lean proteins and limit saturated fats.',
            'Reduce refined carbohydrates and opt for whole grain alternatives.'
        ]
    else:  # obese
        plan['diet'] = [
            'Work with a dietitian to create a sustainable calorie-controlled meal plan.',
            'Prioritize high-fiber foods to improve satiety and blood sugar control.',
            'Eliminate sugary drinks and limit fruit juices.',
            'Practice meal prepping to avoid unhealthy food choices.'
        ]
    
    # Exercise recommendations based on activity level
    if activity == 'low':
        plan['exercise'] = [
            'Start with 15-20 minute walks daily and gradually increase duration.',
            'Try chair exercises or gentle stretching if mobility is limited.',
            'Set reminders to stand and move every hour if you have a sedentary job.',
            'Consider swimming or water aerobics for low-impact cardio.'
        ]
    elif activity == 'moderate':
        plan['exercise'] = [
            'Aim for 150 minutes of moderate aerobic activity per week.',
            'Add 2-3 strength training sessions to build muscle and improve insulin sensitivity.',
            'Include flexibility exercises like yoga or stretching.',
            'Try interval training to boost cardiovascular health.'
        ]
    else:  # high
        plan['exercise'] = [
            'Maintain your excellent activity level with varied workouts.',
            'Ensure adequate rest days to prevent overtraining.',
            'Focus on proper nutrition timing around workouts.',
            'Consider working with a trainer to optimize your routine.'
        ]
    
    # Stress management based on age
    if age >= 50:
        plan['stress'] = [
            'Practice daily meditation or deep breathing exercises for 10-15 minutes.',
            'Join social groups or community activities to stay connected.',
            'Consider gentle yoga or tai chi for combined physical and mental benefits.'
        ]
    else:
        plan['stress'] = [
            'Identify stress triggers and develop healthy coping mechanisms.',
            'Schedule regular breaks and leisure activities.',
            'Practice mindfulness or use stress-management apps.'
        ]
    
    # Sleep recommendations
    if age >= 65:
        plan['sleep'] = [
            'Aim for 7-8 hours of sleep with a consistent bedtime routine.',
            'Limit daytime naps to 20-30 minutes if needed.',
            'Avoid screens and caffeine at least 2 hours before bed.'
        ]
    else:
        plan['sleep'] = [
            'Prioritize 7-9 hours of quality sleep each night.',
            'Create a dark, cool, and quiet sleep environment.',
            'Maintain consistent sleep and wake times, even on weekends.'
        ]
    
    # Health monitoring
    plan['monitoring'] = [
        'Check blood glucose levels as recommended by your healthcare provider.',
        'Monitor blood pressure regularly, especially if overweight.',
        'Schedule regular check-ups and diabetes screenings.'
    ]
    
    return plan

@app.route('/api/lifestyle-plan', methods=['POST'])
def generate_lifestyle_plan():
    """Generate a personalized lifestyle plan based on user inputs."""
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    age = data.get('age')
    bmi = data.get('bmi')
    activity = data.get('activity')
    
    # Validate inputs
    if not all([age, bmi, activity]):
        return jsonify({'error': 'Age, BMI, and activity level are required'}), 400
    
    try:
        age = int(age)
        bmi = float(bmi)
    except (ValueError, TypeError):
        return jsonify({'error': 'Invalid age or BMI value'}), 400
    
    if age < 1 or age > 120:
        return jsonify({'error': 'Age must be between 1 and 120'}), 400
    if bmi < 10 or bmi > 50:
        return jsonify({'error': 'BMI must be between 10 and 50'}), 400
    if activity not in ['low', 'moderate', 'high']:
        return jsonify({'error': 'Activity must be low, moderate, or high'}), 400
    
    # Determine BMI category
    if bmi < 18.5:
        bmi_category = 'underweight'
    elif bmi < 25:
        bmi_category = 'normal weight'
    elif bmi < 30:
        bmi_category = 'overweight'
    else:
        bmi_category = 'obese'
    
    # Determine age group
    if age < 30:
        age_group = 'young adult'
    elif age < 50:
        age_group = 'middle-aged adult'
    elif age < 65:
        age_group = 'older adult'
    else:
        age_group = 'senior'
    
    # Try to get AI-generated plan (Updated for Fix #112)
    try:
        if client:
            prompt = f"""Generate a personalized lifestyle plan for diabetes prevention/management with these details:
- Age: {age} years ({age_group})
- BMI: {bmi} ({bmi_category})
- Physical Activity Level: {activity}

Provide specific, actionable recommendations in these categories:
1. Diet (3-4 specific tips)
2. Exercise (3-4 specific recommendations based on their activity level)
3. Stress Management (2-3 techniques)
4. Sleep & Recovery (2-3 tips)
5. Health Monitoring (2-3 suggestions)

Keep each tip concise (1-2 sentences). Focus on diabetes-relevant advice. Format as JSON with keys: diet, exercise, stress, sleep, monitoring (each containing an array of strings)."""

            response = client.models.generate_content(
                model='gemini-2.0-flash',
                contents=prompt
            )
            response_text = response.text
            
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                plan = json.loads(json_match.group())
                return jsonify({
                    'success': True,
                    'plan': plan,
                    'profile': {
                        'age': age,
                        'age_group': age_group,
                        'bmi': bmi,
                        'bmi_category': bmi_category,
                        'activity': activity
                    },
                    'source': 'ai'
                })
    except Exception as e:
        logging.warning(f"AI plan generation failed, using fallback: {e}")
    
    # Fallback: Rule-based plan generation
    plan = generate_fallback_plan(age, bmi, bmi_category, activity, age_group)
    
    return jsonify({
        'success': True,
        'plan': plan,
        'profile': {
            'age': age,
            'age_group': age_group,
            'bmi': bmi,
            'bmi_category': bmi_category,
            'activity': activity
        },
        'source': 'rules'
    })

@app.route('/generate', methods=['POST'])
def generate():
    try:
        user_input = request.json.get('message')
        if not user_input:
            return jsonify({'reply': "Please say something!"})

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return jsonify({'reply': "Error: Gemini API Key not configured."})

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=user_input
        )
        reply_text = response.text
        
        return jsonify({'reply': reply_text})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'reply': "Sorry, I'm having trouble connecting to the AI right now."})

# --- Forum Backend (Updated for Feature #113: SQLite) ---

def parse_post_timestamp(timestamp_str):
    """
    Parse ISO timestamp string to datetime object.
    Returns UTC datetime objects for consistent timezone handling.
    """
    if not timestamp_str:
        return datetime.min.replace(tzinfo=timezone.utc)

    # Remove 'Z' suffix if present and parse
    ts = timestamp_str.rstrip('Z')
    try:
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)

def filter_posts(posts_list, search=None, start_date=None, end_date=None, topic=None):
    """Filter posts by search term, date range, and topic."""
    filtered = posts_list.copy()

    if search and search.strip():
        search_lower = search.lower()
        filtered = [p for p in filtered if search_lower in p.get('content', '').lower()]

    if topic and topic.strip():
        filtered = [p for p in filtered if p.get('topic') == topic]

    if start_date:
        filtered = [p for p in filtered if parse_post_timestamp(p.get('timestamp')) >= start_date]

    if end_date:
        filtered = [p for p in filtered if parse_post_timestamp(p.get('timestamp')) <= end_date]

    return filtered

def paginate_posts(posts_list, page=1, per_page=10):
    import math
    total = len(posts_list)
    total_pages = math.ceil(total / per_page) if total > 0 else 1
    page = max(1, min(page, total_pages))
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    
    return {
        'posts': posts_list[start_idx:end_idx],
        'total': total,
        'page': page,
        'per_page': per_page,
        'total_pages': total_pages
    }

@app.route('/forum')
def forum():
    return render_template('forum.html')

@app.route('/api/posts', methods=['GET', 'POST'])
def posts_api():
    if request.method == 'GET':
        search = request.args.get('search', '').strip() or None
        start_date_str = request.args.get('start_date', '').strip()
        end_date_str = request.args.get('end_date', '').strip()
        topic = request.args.get('topic', '').strip() or None

        try:
            page = int(request.args.get('page', 1))
            if page < 1: page = 1
        except ValueError:
            return jsonify({"error": "Page must be a positive integer"}), 400

        try:
            per_page = int(request.args.get('per_page', 10))
            if per_page < 1 or per_page > 50: per_page = 10
        except ValueError:
            return jsonify({"error": "per_page must be between 1 and 50"}), 400

        # Parse dates
        start_date = None
        end_date = None
        if start_date_str:
            try:
                start_date = datetime.fromisoformat(start_date_str)
                if start_date.tzinfo is None:
                    start_date = start_date.replace(tzinfo=timezone.utc)
            except ValueError:
                return jsonify({"error": "Invalid date format. Use YYYY-MM-DD"}), 400

        if end_date_str:
            try:
                end_date = datetime.fromisoformat(end_date_str)
                end_date = end_date.replace(hour=23, minute=59, second=59)
                if end_date.tzinfo is None:
                    end_date = end_date.replace(tzinfo=timezone.utc)
            except ValueError:
                return jsonify({"error": "Invalid date format. Use YYYY-MM-DD"}), 400

        if start_date and end_date and start_date > end_date:
            return jsonify({"error": "start_date must be before or equal to end_date"}), 400

        # --- DATABASE CHANGE: Fetch all posts from DB ---
        all_db_posts = Post.query.all()
        # Convert DB models to list of dicts for filtering logic
        posts_list = [p.to_dict() for p in all_db_posts]

        # Sort posts by timestamp (newest first)
        sorted_posts = sorted(posts_list, key=lambda x: x['timestamp'], reverse=True)

        # Apply filters
        filtered_posts = filter_posts(sorted_posts, search=search, start_date=start_date, end_date=end_date, topic=topic)

        # Paginate results
        result = paginate_posts(filtered_posts, page=page, per_page=per_page)

        return jsonify({
            'posts': result['posts'],
            'pagination': {
                'total': result['total'],
                'page': result['page'],
                'per_page': result['per_page'],
                'total_pages': result['total_pages']
            }
        })
    
    elif request.method == 'POST':
        data = request.json
        content = data.get('content', '').strip()
        author_id = data.get('author_id', 'anonymous')
        parent_id = data.get('parent_id')  # For replies
        topic = data.get('topic', '').strip() or None

        if not content:
            return jsonify({"error": "Content is required"}), 400

        if topic and topic not in FORUM_TOPICS:
            return jsonify({"error": f"Invalid topic. Available topics: {FORUM_TOPICS}"}), 400

        # --- DATABASE CHANGE: Create Post in DB ---
        new_post = Post(
            content=content,
            author_id=author_id,
            parent_id=parent_id,
            topic=topic,
            timestamp=datetime.now(timezone.utc)
        )
        db.session.add(new_post)
        db.session.commit()

        # Convert to dict for response and notifications
        post_dict = new_post.to_dict()

        # Process notifications asynchronously
        # We pass the dict because Threads don't share Flask-SQLAlchemy session context easily
        # threading.Thread(target=process_post_notifications, args=(post_dict,)).start()  # <--- COMMENTED OUT TO FIX LAG

        return jsonify(post_dict), 201

# --- Notification System (Updated) ---

def extract_mentions(content):
    """Extract @username mentions from post content."""
    return re.findall(r'@(\w+)', content)

def send_email_notification(to_email, subject, body):
    """Send email notification asynchronously."""
    try:
        if not app.config['MAIL_USERNAME']:
            logging.warning("Email not configured, skipping notification")
            return False
        
        with app.app_context():
            msg = Message(subject=subject, recipients=[to_email], body=body)
            mail.send(msg)
            logging.info(f"Email sent to {to_email}")
            return True
    except Exception as e:
        logging.error(f"Failed to send email: {e}")
        return False

def create_notification(user_id, notif_type, message, post_id=None):
    """Create an in-app notification."""
    notification = {
        'id': len(notifications) + 1,
        'user_id': user_id,
        'type': notif_type,
        'message': message,
        'post_id': post_id,
        'read': False,
        'timestamp': datetime.utcnow().isoformat() + 'Z'
    }
    notifications.append(notification)
    return notification

def process_post_notifications(post):
    """Process notifications for a new post (replies, mentions)."""
    # Need app context to query DB for parent post if needed
    with app.app_context():
        content = post.get('content', '')
        author_id = post.get('author_id', 'anonymous')
        post_id = post.get('id')
        parent_id = post.get('parent_id')
        
        # Handle reply notifications
        if parent_id:
            # --- DATABASE QUERY FOR PARENT POST ---
            parent_post_obj = Post.query.get(parent_id)
            
            if parent_post_obj:
                parent_post = parent_post_obj.to_dict()
                
                if parent_post.get('author_id') != author_id:
                    original_author = parent_post.get('author_id')
                    if original_author in users:
                        user = users[original_author]
                        # Create in-app notification
                        create_notification(
                            original_author, 
                            'reply', 
                            f"Someone replied to your post",
                            post_id
                        )
                        # Send email if enabled
                        if user.get('preferences', {}).get('email_replies', True):
                            send_email_notification(
                                user['email'],
                                "New reply to your post - Diabetes Care Forum",
                                f"Someone replied to your post:\n\n{content[:200]}...\n\nVisit the forum to see the full reply."
                            )
        
        # Handle mention notifications
        mentions = extract_mentions(content)
        logging.info(f"Processing mentions: {mentions}, registered users: {list(users.keys())}")
        
        # Get author's username for the notification message
        author_username = users.get(author_id, {}).get('username', 'Someone')
        
        for username in mentions:
            # Find user by username (case-insensitive)
            mentioned_user = next(
                (uid for uid, u in users.items() if u.get('username', '').lower() == username.lower()),
                None
            )
            logging.info(f"Looking for @{username}, found user_id: {mentioned_user}")
            if mentioned_user and mentioned_user != author_id:
                user = users[mentioned_user]
                create_notification(
                    mentioned_user,
                    'mention',
                    f"@{author_username} mentioned you in a post",
                    post_id
                )
                logging.info(f"Created mention notification for {mentioned_user}")
                if user.get('preferences', {}).get('email_mentions', True):
                    send_email_notification(
                        user['email'],
                        "You were mentioned - Diabetes Care Forum",
                        f"You were mentioned in a post:\n\n{content[:200]}...\n\nVisit the forum to see the full post."
                    )
        
        # Notify subscribers of the thread
        if parent_id:
            for uid, user in users.items():
                if parent_id in user.get('subscribed_posts', []) and uid != author_id:
                    create_notification(
                        uid,
                        'subscription',
                        f"New activity in a thread you're following",
                        post_id
                    )
                    if user.get('preferences', {}).get('email_subscriptions', True):
                        send_email_notification(
                            user['email'],
                            "New activity in subscribed thread - Diabetes Care Forum",
                            f"There's new activity in a thread you're following:\n\n{content[:200]}..."
                        )

# --- User & Notification API Endpoints ---

@app.route('/api/users', methods=['GET', 'POST'])
def users_api():
    """Register or update a user for notifications, or list all users."""
    if request.method == 'GET':
        # Debug endpoint to see registered users
        return jsonify({"users": {uid: {"username": u.get("username"), "email": u.get("email")} for uid, u in users.items()}})
    
    data = request.json
    user_id = data.get('user_id')
    email = data.get('email', '').strip()
    username = data.get('username', '').strip()
    
    if not user_id or not email:
        return jsonify({"error": "user_id and email are required"}), 400
    
    if user_id in users:
        # Update existing user
        users[user_id]['email'] = email
        if username:
            users[user_id]['username'] = username
    else:
        # Create new user
        users[user_id] = {
            'email': email,
            'username': username or user_id,
            'preferences': {
                'email_replies': True,
                'email_mentions': True,
                'email_subscriptions': True
            },
            'subscribed_posts': []
        }
    
    logging.info(f"User registered/updated: {user_id} -> {users[user_id]}")
    return jsonify({"message": "User registered", "user": users[user_id]}), 200

@app.route('/api/users/<user_id>/preferences', methods=['GET', 'PUT'])
def user_preferences(user_id):
    """Get or update user notification preferences."""
    if user_id not in users:
        return jsonify({"error": "User not found"}), 404
    
    if request.method == 'GET':
        return jsonify(users[user_id].get('preferences', {}))
    
    elif request.method == 'PUT':
        data = request.json
        prefs = users[user_id].get('preferences', {})
        
        if 'email_replies' in data:
            prefs['email_replies'] = bool(data['email_replies'])
        if 'email_mentions' in data:
            prefs['email_mentions'] = bool(data['email_mentions'])
        if 'email_subscriptions' in data:
            prefs['email_subscriptions'] = bool(data['email_subscriptions'])
        
        users[user_id]['preferences'] = prefs
        return jsonify({"message": "Preferences updated", "preferences": prefs})

@app.route('/api/users/<user_id>/subscribe/<int:post_id>', methods=['POST', 'DELETE'])
def subscribe_post(user_id, post_id):
    """Subscribe or unsubscribe from a post/thread."""
    if user_id not in users:
        return jsonify({"error": "User not found"}), 404
    
    subscribed = users[user_id].setdefault('subscribed_posts', [])
    
    if request.method == 'POST':
        if post_id not in subscribed:
            subscribed.append(post_id)
        return jsonify({"message": "Subscribed", "subscribed_posts": subscribed})
    
    elif request.method == 'DELETE':
        if post_id in subscribed:
            subscribed.remove(post_id)
        return jsonify({"message": "Unsubscribed", "subscribed_posts": subscribed})

@app.route('/api/notifications/<user_id>', methods=['GET'])
def get_notifications(user_id):
    """Get notifications for a user."""
    # Allow fetching even if user not fully registered yet
    user_notifications = [n for n in notifications if n['user_id'] == user_id]
    
    # Sort by timestamp descending
    user_notifications.sort(key=lambda x: x['timestamp'], reverse=True)
    
    unread_count = sum(1 for n in user_notifications if not n['read'])
    
    return jsonify({
        "notifications": user_notifications[:50],
        "unread_count": unread_count
    })

@app.route('/api/notifications/<user_id>/read', methods=['POST'])
def mark_notifications_read(user_id):
    """Mark notifications as read and optionally delete them."""
    data = request.json or {}
    notification_ids = data.get('notification_ids', [])
    mark_all = data.get('mark_all', False)
    delete_read = data.get('delete', True)  # Delete by default
    
    global notifications
    count = 0
    
    if delete_read:
        # Remove notifications instead of just marking as read
        if mark_all:
            before_count = len([n for n in notifications if n['user_id'] == user_id])
            notifications = [n for n in notifications if n['user_id'] != user_id]
            count = before_count
        else:
            before_count = len(notifications)
            notifications = [n for n in notifications if not (n['user_id'] == user_id and n['id'] in notification_ids)]
            count = before_count - len(notifications)
    else:
        # Just mark as read
        for n in notifications:
            if n['user_id'] == user_id:
                if mark_all or n['id'] in notification_ids:
                    if not n['read']:
                        n['read'] = True
                        count += 1
    
    return jsonify({"message": f"Processed {count} notifications"})


@app.route('/privacy')
def Privacy():
    return render_template('privacy.html')

@app.route('/terms')
def Privacy():
    return render_template('terms.html')
# --- NEW AUTH ROUTES (Register, Login, Logout, Dashboard) ---


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
