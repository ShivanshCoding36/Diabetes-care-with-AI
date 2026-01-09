
# Diabetes Care Home

A comprehensive web-based platform for diabetes risk assessment, health data visualization, and AI-powered health assistance.

[![Live Demo](https://img.shields.io/badge/demo-live-brightgreen)](https://diabetes-care-with-ai-5-nd7x.onrender.com)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/flask-2.x-lightgrey.svg)](https://flask.palletsprojects.com/)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)](CONTRIBUTING.md)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [API Endpoints](#api-endpoints)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Diabetes Care Home is an open-source application that combines machine learning with interactive health tools to support diabetes awareness and management. The platform enables users to:

- Assess their diabetes risk using a trained machine learning model
- Explore and visualize health data patterns through interactive charts
- Get instant answers to health queries via an AI-powered chatbot
- Access evidence-based lifestyle recommendations
- Connect with a community through a discussion forum

### Why This Project?

Diabetes affects millions worldwide, and early detection is crucial for effective management. This platform aims to make diabetes risk assessment accessible to everyone while providing educational resources and community support.

> **Medical Disclaimer:** This application is intended for educational and informational purposes only. It does not provide medical advice, diagnosis, or treatment recommendations. Always consult a qualified healthcare professional for medical concerns.

---

## Features

### 1. Diabetes Risk Prediction

The core feature uses a machine learning model trained on the Pima Indians Diabetes Dataset. Users input eight health parameters:

| Parameter | Description | Unit |
|-----------|-------------|------|
| Pregnancies | Number of pregnancies | Count |
| Glucose | Plasma glucose concentration (2 hours after glucose tolerance test) | mg/dL |
| Blood Pressure | Diastolic blood pressure | mm Hg |
| Skin Thickness | Triceps skin fold thickness | mm |
| Insulin | 2-Hour serum insulin | mu U/ml |
| BMI | Body Mass Index | kg/m² |
| Diabetes Pedigree Function | Genetic predisposition score | Score |
| Age | Age of the individual | Years |

The model processes these inputs through a StandardScaler for normalization and returns a binary prediction (Diabetic / Not Diabetic).

### Note on Prediction Inputs

The diabetes prediction model is trained on the Pima Indians Diabetes Dataset.
Due to limitations of the dataset, gender and individual height/weight values are not included as input features.
BMI is used as a combined indicator of height and weight.

### 2. Data Exploration Dashboard

Interactive visualization tools for understanding diabetes-related health patterns:

- **Correlation Heatmap** — Displays relationships between all health variables using Seaborn
- **Glucose Distribution** — Interactive histogram showing glucose level patterns across the dataset
- **BMI Distribution** — Visual analysis of Body Mass Index distribution

All charts are rendered using Plotly for interactivity and Matplotlib/Seaborn for statistical visualizations.

### 3. AI-Powered Chatbot

An intelligent assistant powered by Google Gemini API that can:

- Answer diabetes-related health questions
- Provide general information about symptoms, causes, and management
- Offer guidance on lifestyle modifications
- Respond in a conversational, user-friendly manner

### 4. Lifestyle Recommendations

Curated, evidence-based guidance covering:

- Dietary recommendations for blood sugar management
- Exercise and physical activity guidelines
- Stress management techniques
- Sleep hygiene practices
- Regular monitoring schedules

### 5. Community Forum

A simple discussion platform where users can:

- Share their experiences and stories
- Ask questions to the community
- Provide peer support
- View posts in reverse chronological order

---

## Technology Stack

### Backend

| Technology | Purpose | Version |
|------------|---------|---------|
| Python | Core programming language | 3.8+ |
| Flask | Web application framework | 2.x |
| Flask-CORS | Cross-Origin Resource Sharing | Latest |
| Scikit-learn | Machine learning model | Latest |
| Pandas | Data manipulation and analysis | Latest |
| NumPy | Numerical computing | Latest |
| Gunicorn | Production WSGI server | Latest |

### Frontend

| Technology | Purpose |
|------------|---------|
| HTML5 | Page structure and semantics |
| CSS3 | Styling and animations |
| Tailwind CSS | Utility-first CSS framework |
| JavaScript | Client-side interactivity |

### Data Visualization

| Technology | Purpose |
|------------|---------|
| Plotly | Interactive web-based charts |
| Matplotlib | Static chart generation |
| Seaborn | Statistical data visualization |

### AI Integration

| Technology | Purpose |
|------------|---------|
| Google Gemini API | Conversational AI chatbot |
| google-generativeai | Python SDK for Gemini |

### Deployment

| Technology | Purpose |
|------------|---------|
| Render | Cloud hosting platform |
| Gunicorn | Production-grade WSGI server |
| python-dotenv | Environment variable management |
| Docker | Containerization |

---

## Getting Started

### Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.8 or higher**
  ```bash
  # Check Python version
  python --version

```

* **pip package manager**
```bash
# Check pip version
pip --version

```


* **Git**
```bash
# Check Git version
git --version

```


* **Docker Desktop** (Optional, for containerized setup)
* Required if you plan to run the application using Docker.


* **Google Gemini API Key** — Required for the chatbot feature
* Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
* Sign in with your Google account
* Click "Create API Key"
* Copy and save the key securely



### Installation (Manual Method)

#### Step 1: Clone the Repository

```bash
# Clone via HTTPS
git clone [https://github.com/your-username/diabetes-care-home.git](https://github.com/your-username/diabetes-care-home.git)

# Or clone via SSH
git clone git@github.com:your-username/diabetes-care-home.git

# Navigate to project directory
cd diabetes-care-home

```

#### Step 2: Create Virtual Environment

Creating a virtual environment isolates project dependencies from your system Python.

**Windows (Command Prompt):**

```cmd
python -m venv venv
venv\Scripts\activate

```

**Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1

```

**macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate

```

You should see `(venv)` prefix in your terminal indicating the virtual environment is active.

#### Step 3: Install Dependencies

```bash
# Upgrade pip to latest version
pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt

```

This installs the following packages:

* flask, flask-cors
* numpy, pandas
* matplotlib, seaborn, plotly
* scikit-learn
* python-dotenv
* google-generativeai
* gunicorn

#### Step 4: Configure Environment Variables

Create a `.env` file in the project root directory:

**Windows (Command Prompt):**

```cmd
echo GEMINI_API_KEY=your_api_key_here > .env

```

**Windows (PowerShell):**

```powershell
"GEMINI_API_KEY=your_api_key_here" | Out-File -FilePath .env -Encoding utf8

```

**macOS / Linux:**

```bash
echo "GEMINI_API_KEY=your_api_key_here" > .env

```

Or manually create the file with the following content:

```env
GEMINI_API_KEY=your_actual_gemini_api_key

```

> **Security Note:** Never commit your `.env` file to version control. It should already be listed in `.gitignore`.

#### Step 5: Verify Model Files

Ensure the following pre-trained model files exist in the project root:

* `diabetes_model.pkl` — Trained classification model
* `scaler.pkl` — Feature scaler for input normalization
* `diabetes.csv` — Dataset for visualization features

If missing, you can retrain the model using the provided Jupyter notebook:

```bash
# Install Jupyter if not available
pip install jupyter

# Launch Jupyter and run train.ipynb
jupyter notebook train.ipynb

```

#### Step 6: Run the Application

**Development Mode:**

```bash
python app.py

```

**Production Mode (using Gunicorn):**

```bash
gunicorn app:app --bind 0.0.0.0:5000

```

#### Step 7: Access the Application

Open your web browser and navigate to:

```
http://localhost:5000

```

You should see the Diabetes Care Home landing page.

### 🐳 Installation (Docker Method)

If you prefer to run the application in a consistent containerized environment, follow these steps instead of the manual installation.

1. **Prerequisite:** Ensure **Docker Desktop** is installed and running on your machine.
2. **Configure Env:** Create the `.env` file as described in **Step 4** above.
3. **Run:** Open your terminal in the project root and run:
```bash
docker-compose up --build

```


4. **Access:** Open your browser to `http://localhost:5000`.

To stop the container, press `Ctrl+C` in the terminal.

---

## Project Structure

```
diabetes-care-home/
│
├── app.py                      # Main Flask application
│   ├── Route definitions       # All URL endpoints
│   ├── ML model loading        # Pickle file loading
│   ├── Gemini integration      # AI chatbot logic
│   └── Forum API               # Post management
│
├── diabetes_model.pkl          # Trained scikit-learn model
├── scaler.pkl                  # StandardScaler for feature normalization
├── diabetes.csv                # Pima Indians Diabetes Dataset
│
├── train.ipynb                 # Jupyter notebook for model training
│
├── requirements.txt            # Python package dependencies
├── Dockerfile                  # Docker build instructions
├── docker-compose.yml          # Container orchestration
├── .env                        # Environment variables (not in repo)
├── .gitignore                  # Git ignore rules
│
├── static/                     # Static assets
│   └── doctor.png              # Image assets
│
└── templates/                  # Jinja2 HTML templates
    ├── home.html               # Landing page (route: /)
    ├── index.html              # Prediction form (route: /index)
    ├── explore.html            # Data visualization (route: /explore)
    ├── chatbot.html            # AI chatbot interface (route: /chatbot)
    ├── life.html               # Lifestyle tips (route: /life)
    └── forum.html              # Community forum (route: /forum)

```

---

## API Endpoints

### Web Routes

| Method | Endpoint | Description |
| --- | --- | --- |
| GET | `/` | Landing page |
| GET | `/index` | Diabetes prediction form |
| POST | `/predict` | Process prediction form submission |
| GET | `/explore` | Data visualization dashboard |
| GET | `/chatbot` | AI chatbot interface |
| GET | `/life` | Lifestyle recommendations page |
| GET | `/forum` | Community forum page |

### API Routes

| Method | Endpoint | Description | Request Body | Response |
| --- | --- | --- | --- | --- |
| POST | `/generate` | Send message to AI chatbot | `{"message": "your question"}` | `{"reply": "AI response"}` |
| GET | `/api/posts` | Retrieve all forum posts | — | Array of post objects |
| POST | `/api/posts` | Create new forum post | `{"content": "post text"}` | Created post object |

### Example API Usage

**Chat with AI Assistant:**

```bash
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"message": "What are the symptoms of diabetes?"}'

```

**Create Forum Post:**

```bash
curl -X POST http://localhost:5000/api/posts \
  -H "Content-Type: application/json" \
  -d '{"content": "Hello, this is my first post!"}'

```

**Get All Forum Posts:**

```bash
curl http://localhost:5000/api/posts

```

---

## Contributing

We welcome contributions from developers of all skill levels.

Follow the Code Of Conduct: [CODE_OF_CONDUCT.md](https://www.google.com/search?q=CODE_OF_CONDUCT.md)

---

## License

This project is open source and available under the [MIT License](https://www.google.com/search?q=LICENSE).

```
MIT License

Copyright (c) 2024 Anshika Singh

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

```

---

## Author

**Anshika Singh**

---

## Acknowledgments

* [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) — UCI Machine Learning Repository
* [Google Gemini](https://deepmind.google/technologies/gemini/) — AI chatbot capabilities
* [Flask](https://flask.palletsprojects.com/) — Web framework
* [Tailwind CSS](https://tailwindcss.com/) — UI styling

---

<p align="center">
<strong>Making diabetes care smarter and more accessible.</strong>
</p>
