# Customer Churn Prediction MLOps Pipeline

This project demonstrates an end-to-end MLOps pipeline for predicting customer churn. It focuses on **production-readiness**, featuring a containerized REST API, automated testing, strict code quality checks, and a fully automated CI/CD pipeline that publishes versioned Docker images to the GitHub Container Registry (GHCR).

## Model Performance

**Current Model Accuracy:** <!-- ACCURACY_START -->0.0000<!-- ACCURACY_END -->



## Architecture & Flow
1. **Training Pipeline (`src/`):** Downloads the IBM Telco Customer Churn dataset, cleans the data, and trains a Scikit-Learn `RandomForestClassifier`. Crucially, it saves a Scikit-Learn Pipeline object (preprocessing + model) as `model.joblib` to prevent train-serving skew.
2. **API Serving (`app/`):** A FastAPI application loads the `model.joblib` pipeline and exposes a `/predict` endpoint. It uses Pydantic for strict input data validation.
3. **Containerization:** The application and its dependencies are packaged into a Docker image using uv for ultra-fast dependency resolution.
4. **CI/CD (GitHub Actions):** On every push to main, the pipeline automatically:
  - Syncs dependencies using uv.
  - Trains the model from scratch.
  - Runs unit tests (Pytest) to validate both API behavior and model accuracy thresholds.
  - Lints the codebase (Ruff).
  - Builds the Docker image and pushes it to GHCR with a version tag based on the Git commit SHA.

## Tech Stack
Category	Technology
Language	Python 3.12
Machine Learning	Scikit-Learn, Pandas, Joblib
API Framework	FastAPI, Uvicorn, Pydantic
Dependency Mgmt	uv, pyproject.toml
Containerization	Docker, Docker Compose
CI/CD & Registry	GitHub Actions, GitHub Container Registry (GHCR)
Code Quality	Ruff (Linting & Formatting), Pre-commit hooks


## Project Structure
```
churn-prediction/
├── .github/workflows/
│   └── ci.yml               # CI/CD pipeline definition
├── app/
│   └── main.py              # FastAPI application and endpoints
├── src/│   ├── preprocess.py        # Data loading, cleaning, and SK-Learn Pipeline
│   └── train.py             # Model training script
├── tests/
│   ├── test_app.py          # API unit tests (Pytest)
│   └── test_model.py        # Data science tests (model accuracy validation)
├── .dockerignore
├── .env                     # Local environment variables (ports)
├── .pre-commit-config.yaml  # Local git hook configuration
├── docker-compose.yml       # Local container orchestration
├── Dockerfile               # Production container definition
├── model.joblib             # Serialized ML pipeline (generated on train)
└── pyproject.toml           # Project dependencies and Ruff config
```

## Getting Started (Local Development)

### Prerequisites
- Python 3.12+
- uv installed globally
- Docker & Docker Compose

#### 1. Install Dependencies

This project uses `uv` for dependency management.

```bash
uv sync
```

#### 2. Train the Model

Run the training script to download the data and generate `model.joblib`.

```bash
uv run python -m src.train
```

#### 3. Run the API Locally (Without Docker)

```bash
uv run uvicorn app.main:app --reload
```
Visit `http://127.0.0.1:8000/docs` to view the interactive Swagger UI.

#### 4. Run the API Locally (With Docker Compose)
Using Docker Compose reads the port configuration from the `.env` file.

```bash
docker compose up --build -d
```

## Testing & Code Quality
To ensure production readiness, this project enforces strict testing and linting.

- Run Tests: `uv run pytest`
- Lint Code: `uv run ruff check .`

Note: This project uses `pre-commit` hooks. If installed, Ruff will automatically check and format code before allowing a `git commit`.

## Continuous Deployment (CD)
Every time code is merged into the main branch, GitHub Actions builds a Docker image and pushes it to the GitHub Container Registry (GHCR).

Images are versioned using the Git commit SHA (e.g., `sha-a1b2c3d`) alongside a `latest` tag for convenience.

### Run the Deployed Image
You can pull and run the latest deployed model directly from GHCR without needing to clone the repository:

```bash
# Replace YOUR_GITHUB_USERNAME with your actual username
docker run -d -p 8000:8000 ghcr.io/YOUR_GITHUB_USERNAME/churn-prediction:latest
```

## Future Improvements

- **Model Registry:** Integrate MLflow to track model experiments, parameters, and artifacts.
- **Data Drift Monitoring:** Add Evidently AI to detect when incoming API data distribution shifts from the training data.
- **Cloud Deployment:** Add a Terraform script and a CD step to deploy the GHCR image to AWS ECS or GCP Cloud Run.




## How to use the deployed image

Open your terminal and run this command (replace YOUR_GITHUB_USERNAME with your actual username, and make sure it's lowercase).
```bash
docker run -d -p 8000:8000 ghcr.io/YOUR_GITHUB_USERNAME/churn-prediction:latest
```
or specific registry:
```bash
# Run the specific commit version
docker run -d -p 8000:8000 ghcr.io/YOUR_GITHUB_USERNAME/churn-prediction:sha-a1b2c3d
```
Docker will download your image from GitHub and run it. Go to http://127.0.0.1:8000/docs and test your API.

```bash
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 1,
  "PhoneService": "No",
  "MultipleLines": "No phone service",
  "InternetService": "DSL",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 29.85,
  "TotalCharges": 29.85
}
```