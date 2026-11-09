import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_FILE = "Housing.csv"
TARGET = "price"
MODEL_FILE = "house_price_model.pkl"

def main():
    df = pd.read_csv(DATA_FILE)
    if TARGET not in df.columns:
        raise ValueError(f"Dataset must contain a '{TARGET}' column.")

    df = df.dropna(subset=[TARGET]).copy()
    X = df.drop(columns=[TARGET])
    y = pd.to_numeric(df[TARGET], errors="coerce")
    valid = y.notna()
    X, y = X.loc[valid], y.loc[valid]

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    transformers = []
    if numeric_features:
        transformers.append(("num", numeric_pipeline, numeric_features))
    if categorical_features:
        transformers.append(("cat", categorical_pipeline, categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers)
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(mean_squared_error(y_test, predictions) ** 0.5),
        "r2": float(r2_score(y_test, predictions))
    }

    bundle = {
        "model": model,
        "features": X.columns.tolist(),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "target": TARGET,
        "metrics": metrics,
        "actual_test": y_test.tolist(),
        "predicted_test": predictions.tolist()
    }
    joblib.dump(bundle, MODEL_FILE)

    print("Model trained and saved successfully.")
    print(f"MAE:  {metrics['mae']:,.2f}")
    print(f"RMSE: {metrics['rmse']:,.2f}")
    print(f"R²:   {metrics['r2']:.4f}")

if __name__ == "__main__":
    main()
