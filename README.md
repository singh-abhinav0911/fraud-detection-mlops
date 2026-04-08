# 🔍 Fraud Detection MLOps System

An end-to-end machine learning system for real-time credit card fraud detection — built with production-grade tools including FastAPI, MLflow, XGBoost, and Docker.

> **AUC Score: 0.98** | Trained on 284,807 real transactions | REST API with live predictions

---

## 📌 Project Overview

This project simulates how fraud detection systems work at real banks and fintech companies. It goes beyond a simple notebook — it is a fully deployable ML pipeline with experiment tracking, a REST API, containerization, and CI/CD.

---

## 🏗️ Architecture

```
Raw Data (Kaggle)
      ↓
Data Preprocessing + SMOTE (imbalance fix)
      ↓
Model Training (XGBoost) + Experiment Tracking (MLflow)
      ↓
REST API (FastAPI) → Docker Container
      ↓
CI/CD (GitHub Actions)
```

---

## 🛠️ Tech Stack

| Layer              | Tool                          |
|--------------------|-------------------------------|
| Language           | Python 3.11                   |
| ML Model           | XGBoost                       |
| Imbalance Fix      | SMOTE (imbalanced-learn)      |
| Experiment Tracking| MLflow                        |
| API Framework      | FastAPI + Uvicorn             |
| Containerization   | Docker                        |
| CI/CD              | GitHub Actions                |
| Data Processing    | Pandas, NumPy, Scikit-learn   |

---

## 📁 Project Structure

```
fraud-detection-mlops/
├── data/                   # Dataset (not tracked in Git)
├── notebooks/
│   └── 01_eda.ipynb        # EDA, model training, MLflow tracking
├── src/
│   ├── app.py              # FastAPI application
│   ├── model.pkl           # Trained XGBoost model
│   └── scaler.pkl          # StandardScaler
├── .github/
│   └── workflows/          # CI/CD pipeline (coming soon)
├── .gitignore
├── Dockerfile              # Docker configuration
├── requirements.txt        # Python dependencies
└── README.md
```

---

## 📊 Model Performance

| Metric        | Score  |
|---------------|--------|
| AUC Score     | 0.98   |
| Fraud Recall  | 85%    |
| Precision     | 73%    |
| F1 Score      | 0.78   |
| Accuracy      | ~100%  |

The dataset is heavily imbalanced — only 0.17% of transactions are fraudulent (492 out of 284,807). SMOTE was used to oversample the minority class, improving fraud recall from 80% to 85% and AUC from 0.94 to 0.98.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- Git
- Docker (optional)

### 1. Clone the repository
```bash
git clone https://github.com/singh-abhinav0911/fraud-detection-mlops.git
cd fraud-detection-mlops
```

### 2. Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the dataset
Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the `data/` folder.

### 5. Run the notebook
Open `notebooks/01_eda.ipynb` in VS Code or Jupyter and run all cells. This trains the model and saves `model.pkl` and `scaler.pkl` to the `src/` folder.

### 6. Start the API
```bash
cd src
uvicorn app:app --reload
```

API will be live at `http://127.0.0.1:8000`

---

## 🔌 API Usage

### Base URL
```
http://127.0.0.1:8000
```

### Endpoints

| Method | Endpoint   | Description              |
|--------|------------|--------------------------|
| GET    | `/`        | Health check             |
| POST   | `/predict` | Predict fraud            |
| GET    | `/docs`    | Interactive API docs (Swagger UI) |

### Example Request
```bash
POST /predict
Content-Type: application/json

{
  "features": [0.0, -1.35, -0.07, 2.53, 1.37, -0.33, 0.46, 0.23,
               0.09, 0.36, 0.09, -0.55, -0.61, -0.99, -0.31, 1.46,
               -0.47, 0.20, 0.02, 0.40, 0.25, -0.01, 0.27, -0.11,
               0.06, 0.12, -0.18, 0.13, -0.02, 149.62]
}
```

### Example Response
```json
{
  "fraud": false,
  "probability": 0.0012,
  "risk": "LOW"
}
```

The input is a list of 30 features: `Time`, `V1`–`V28` (PCA-anonymized bank features), and `Amount`.

---

## 📈 MLflow Experiment Tracking

This project uses MLflow to log every experiment — parameters, metrics, and saved models.

### Start MLflow UI
```bash
mlflow ui --backend-store-uri file:///path/to/fraud-detection-mlops/mlruns
```

Open `http://127.0.0.1:5000` to view the dashboard.

Tracked per run:
- Parameters: model type, SMOTE usage, test size
- Metrics: AUC, F1, precision, recall
- Artifacts: saved model

---

## 🐳 Docker (Coming Soon)

```bash
docker build -t fraud-detection-api .
docker run -p 8000:8000 fraud-detection-api
```

---

## 📖 Dataset

- **Source**: [Credit Card Fraud Detection — Kaggle (MLG-ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size**: 284,807 transactions
- **Fraud cases**: 492 (0.17%)
- **Features**: 30 (Time, V1–V28 via PCA, Amount)
- **Target**: `Class` (0 = Normal, 1 = Fraud)

The features V1–V28 are the result of PCA transformation applied by the bank to protect sensitive customer information while preserving the statistical patterns needed for fraud detection.

---

## 💡 Key Learnings

- Handling severely imbalanced datasets using SMOTE
- Building production-style ML pipelines beyond Jupyter notebooks
- Tracking ML experiments with MLflow
- Serving ML models via REST APIs using FastAPI
- Containerizing applications with Docker
- Automating workflows with CI/CD (GitHub Actions)

---

## 🗺️ Roadmap

- [x] Data preprocessing pipeline
- [x] XGBoost model with SMOTE
- [x] MLflow experiment tracking
- [x] FastAPI REST API
- [ ] Docker containerization
- [ ] GitHub Actions CI/CD
- [ ] Streamlit monitoring dashboard
- [ ] Data drift detection with Evidently AI

---

## 👤 Author

**Abhinav Singh**
- GitHub: [@singh-abhinav0911](https://github.com/singh-abhinav0911)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).