import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# The URL for the raw Telco Churn dataset (hosted on GitHub for easy access)
DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"


def load_data():
    """Downloads data from the web and returns a pandas DataFrame."""
    df = pd.read_csv(DATA_URL)
    return df


def clean_data(df):
    """Cleans the dataframe and returns X (features) and y (target)."""
    # 1. Drop customerID (useless for modeling)
    df = df.drop("customerID", axis=1)

    # 2. Convert TotalCharges to numeric (it has empty strings that become NaN)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    # Fill NaNs with 0 (representing new customers)
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # 3. Map target 'Churn' to 1 (Yes) and 0 (No)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # 4. Split into X and y
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    return X, y


def build_pipeline(X):
    """Creates a Scikit-Learn Pipeline that handles preprocessing and modeling."""

    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # Create preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    # Create the full pipeline with a RandomForest model
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    return pipeline
