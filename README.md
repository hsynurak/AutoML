# AutoML Dashboard: End-to-End Machine Learning Platform

**AutoML Dashboard** is a full-stack web application designed to democratize machine learning. It allows users to upload raw data, perform automated exploratory data analysis (EDA), and train robust ML models without writing a single line of code.

The project is built with a **Micro-service oriented architecture**, separating the calculation engine (FastAPI) from the user interface (Streamlit), communicating via RESTful APIs.

---

## Key Features

### 1. Automated Data Analysis
* Upload CSV files and get instant insights.
* Generates a detailed **HTML Report** (Distribution, Correlations, Missing Values).
* Provides a statistical summary JSON for the frontend.

### 2. Smart Feature Engineering (The Engine)
The system uses a custom `AutoPreprocessor` pipeline that automatically handles complex data scenarios:
* ** Date Engineering:** Automatically detects datetime columns and extracts features like *Month, Year, Season, Is_Weekend*.
* ** Robust Scaling:** Uses `RobustScaler` instead of Standard Scaler to handle **outliers** effectively.
* ** Rare Label Encoding:** Automatically groups infrequent categories (high cardinality) into an "Other" category to prevent memory explosion.

### 3. Model Training & Persistence
* Supports both **Classification** and **Regression** tasks.
* **Random Forest** implementation with parallel processing (`n_jobs=-1`).
* **Pipeline Serialization:** Saves not just the model, but the entire preprocessing pipeline (`.pkl`), ensuring 100% reproducibility during prediction.

---

## Tech Stack & Architecture

* **Backend:** FastAPI, Uvicorn, Pydantic.
* **Frontend:** Streamlit.
* **ML Core:** Scikit-learn, Pandas, Numpy, Joblib.
* **Design Patterns:** OOP (Abstract Base Classes), Pipeline Pattern.

### Project Structure
```text
automl-platform/
├── app/
│   ├── main.py              # FastAPI Entry Point & Routes
│   ├── ml_engine/           # Core ML Logic
│   │   ├── base.py          # Abstract Base Model (Interface)
│   │   ├── analyzer.py      # Automated EDA Logic
│   │   ├── preprocessor.py  # Custom Feature Engineering Pipeline
│   │   └── models/          # Model Wrappers (RandomForest etc.)
├── frontend/
│   └── main.py              # Streamlit UI Logic
├── data/                    # Data Storage (Git-Ignored)
│   ├── raw/                 # Uploaded CSVs
│   ├── models/              # Trained Pipelines (.pkl)
│   └── reports/             # Generated HTML Reports
└── requirements.txt         # Project Dependencies
```
---

## Installation & Usage

### 1. Clone the Repository
  ```
    git clone [https://github.com/hsynurak/AutoML.git]
    cd automl-platform
  ```
### 2. Setup Environment
  ```
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
  ```
### 3. Run the Application
You need to run the Backend and Frontend in separate terminals.
  #### 3-1. Terminal 1: Backend (API)
    ```
      uvicorn app.main:app --reload
      # Server will start at [http://127.0.0.1:8000](http://127.0.0.1:8000)
    ```
  #### 3-2. Terminal 2: Frontend (UI)
    ```
      streamlit run frontend/main.py
      # UI will open at http://localhost:8501
    ```

---

## Roadmap

- [ ] **NLP Support**
- [ ] **Model Comparison**
- [ ] **Dockerization**
- [ ] **Async Training**
