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

from dotenv import load_dotenv
load_dotenv()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns

from flask import Flask, jsonify, render_template, request, make_response, redirect, url_for, flash
from flask_cors import CORS
from flask_mail import Mail, Message
from flask_babel import Babel, gettext as _
from flask_login import LoginManager, login_user, current_user, logout_user, login_required
from flask_bcrypt import Bcrypt

from models import db, Post, User, PredictionHistory

import google.generativeai as genai

# ------------------- APP SETUP -------------------

app = Flask(__name__)
app.config.from_object(Config)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///forum.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)
with app.app_context():
    db.create_all()

bcrypt = Bcrypt(app)

login_manager = LoginManager(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

CORS(app)

# ------------------- BABEL -------------------

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
    user_language = request.cookies.get('language')
    if user_language in SUPPORTED_LANGUAGES:
        return user_language
    return DEFAULT_LANGUAGE

babel.init_app(app, locale_selector=get_locale)

# ------------------- MAIL -------------------

app.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER', 'smtp.gmail.com')
app.config['MAIL_PORT'] = int(os.getenv('MAIL_PORT', 587))
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.getenv('MAIL_DEFAULT_SENDER', 'noreply@diabetescare.com')

mail = Mail(app)

# ------------------- LOGGING -------------------

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s: %(message)s')

# ------------------- ML MODEL -------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "diabetes_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except:
    model, scaler = None, None

try:
    df = pd.read_csv('diabetes.csv')
except:
    df = None

# ------------------- GEMINI FIX -------------------

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

def get_gemini_response(user_message):
    try:
        response = gemini_model.generate_content(user_message)
        return response.text
    except Exception as e:
        logging.error(f"Gemini error: {e}")
        return "Sorry, Gemini service is unavailable right now."

# ------------------- ROUTES -------------------

@app.route('/')
def root():
    return render_template('home.html')

@app.route('/index')
def home():
    return render_template('index.html')

# ------------------- AUTH -------------------

@app.route("/register", methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))

    if request.method == 'POST':
        hashed = bcrypt.generate_password_hash(request.form['password']).decode('utf-8')
        user = User(
            username=request.form['username'],
            email=request.form['email'],
            password=hashed
        )
        db.session.add(user)
        db.session.commit()
        flash('Account created successfully!', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route("/login", methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))

    if request.method == 'POST':
        user = User.query.filter_by(email=request.form['email']).first()
        if user and bcrypt.check_password_hash(user.password, request.form['password']):
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash('Login Unsuccessful. Check email and password', 'danger')

    return render_template('login.html')

@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route("/dashboard")
@login_required
def dashboard():
    predictions = PredictionHistory.query.filter_by(user_id=current_user.id).all()
    return render_template('dashboard.html', predictions=predictions)

# ------------------- PREDICTION -------------------

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [
            float(request.form.get(f, 0)) for f in
            ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
        ]

        if any(f < 0 for f in features):
            return render_template('index.html', prediction_text=_("Negative values not allowed"))

        if scaler is None or model is None:
            return render_template('index.html', prediction_text=_("Model not available"))

        final_input = scaler.transform([features])
        prediction = model.predict(final_input)[0]

        result = _("Diabetic") if prediction == 1 else _("Not Diabetic")

        if current_user.is_authenticated:
            record = PredictionHistory(
                user_id=current_user.id,
                glucose=features[1],
                bmi=features[5],
                age=int(features[7]),
                prediction=int(prediction)
            )
            db.session.add(record)
            db.session.commit()

        return render_template('index.html', prediction_text=_("Prediction: %(result)s", result=result))

    except Exception as e:
        logging.error(f"Predict error: {e}")
        return render_template('index.html', prediction_text=_("Error during prediction"))

# ------------------- CHATBOT -------------------

@app.route('/chatbot')
def chatbot_page():
    return render_template('chatbot.html')

@app.route('/generate', methods=['POST'])
def generate():
    try:
        user_input = request.json.get('message')
        if not user_input:
            return jsonify({'reply': "Please say something!"})

        response = gemini_model.generate_content(user_input)
        return jsonify({'reply': response.text})

    except Exception as e:
        logging.error(e)
        return jsonify({'reply': "AI service unavailable"})

# ------------------- EXPLORE -------------------

@app.route('/explore')
def explore():
    if df is None:
        return "Dataset not loaded", 500

    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    heatmap_url = base64.b64encode(img.getvalue()).decode()

    fig = px.histogram(df, x="Glucose")

    return render_template(
        'explore.html',
        heatmap_url=heatmap_url,
        hist_glucose_html=fig.to_html(full_html=False, include_plotlyjs='cdn')
    )

# ------------------- RUN -------------------

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
