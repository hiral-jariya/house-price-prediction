import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

st.set_page_config(
    page_title="House Price Analytics",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_FILE = Path("Housing.csv")
MODEL_FILE = Path("house_price_model.pkl")

st.markdown("""
<style>
    .main {background-color: #f7f8fc;}
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    div[data-testid="stMetric"] {
        background: white; padding: 16px; border-radius: 12px;
        border: 1px solid #e9eaf0;
    }
    .hero {
        padding: 25px; border-radius: 16px; color: white;
        background: linear-gradient(110deg, #173b57, #287a78);
        margin-bottom: 18px;
    }
    .hero h1 {color: white; margin: 0;}
    .hero p {color: #e5f2f2; margin-bottom: 0;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)

@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_FILE)

def money(value):
    # Dataset currency/unit is preserved; no currency symbol is assumed.
    return f"{value:,.2f}"

def show_header(title, subtitle):
    st.markdown(
        f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True
    )

if not DATA_FILE.exists():
    st.error("Housing.csv was not found. Put it in the same folder as app.py.")
    st.stop()
if not MODEL_FILE.exists():
    st.error("house_price_model.pkl was not found. Run `python train_model.py` first.")
    st.stop()

try:
    df = load_data()
    bundle = load_bundle()
    model = bundle["model"]
    features = bundle.get("features", [c for c in df.columns if c != "price"])
    numeric_features = bundle.get("numeric_features", df[features].select_dtypes(include="number").columns.tolist())
    categorical_features = bundle.get("categorical_features", df[features].select_dtypes(exclude="number").columns.tolist())
except Exception as exc:
    st.error(f"Could not load the dataset/model: {exc}")
    st.stop()

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

st.sidebar.title("🏡 House Price AI")
st.sidebar.caption("Prediction & Real Estate Analytics")
page = st.sidebar.radio(
    "Navigate",
    ["Dashboard", "Explore Dataset", "Predict Price", "Model Performance", "Prediction History"]
)
st.sidebar.divider()
st.sidebar.caption("Machine Learning model: Linear Regression")

# ---------------- Dashboard ----------------
if page == "Dashboard":
    show_header("House Price Dashboard", "Explore housing data and estimate property prices.")
    if "price" not in df.columns:
        st.error("The dataset must contain a 'price' target column.")
        st.stop()

    prices = pd.to_numeric(df["price"], errors="coerce").dropna()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Properties", f"{len(df):,}")
    c2.metric("Average Price", money(prices.mean()))
    c3.metric("Minimum Price", money(prices.min()))
    c4.metric("Maximum Price", money(prices.max()))

    left, right = st.columns(2)
    with left:
        st.subheader("House Price Distribution")
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(prices, kde=True, ax=ax)
        ax.set_xlabel("Price (dataset units)")
        ax.set_ylabel("Number of houses")
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.subheader("Average Price by Bedrooms")
        if "bedrooms" in df.columns:
            chart_df = df.copy()
            chart_df["price"] = pd.to_numeric(chart_df["price"], errors="coerce")
            grouped = chart_df.groupby("bedrooms", as_index=False)["price"].mean()
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.barplot(data=grouped, x="bedrooms", y="price", ax=ax)
            ax.set_xlabel("Bedrooms")
            ax.set_ylabel("Average price")
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("The dataset does not contain a bedrooms column.")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(8), use_container_width=True)

# ---------------- Explore Dataset ----------------
elif page == "Explore Dataset":
    show_header("Explore the Dataset", "Understand the features used by the prediction model.")
    a, b, c = st.columns(3)
    a.metric("Rows", f"{df.shape[0]:,}")
    b.metric("Columns", f"{df.shape[1]:,}")
    c.metric("Missing Values", f"{int(df.isna().sum().sum()):,}")

    st.subheader("Dataset")
    st.dataframe(df, use_container_width=True, height=350)

    st.subheader("Column Information")
    info = pd.DataFrame({
        "Column": df.columns,
        "Data Type": [str(df[col].dtype) for col in df.columns],
        "Missing Values": [int(df[col].isna().sum()) for col in df.columns],
        "Unique Values": [int(df[col].nunique()) for col in df.columns]
    })
    st.dataframe(info, use_container_width=True)

    st.subheader("Correlation Heatmap")
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Not enough numeric columns for a correlation heatmap.")

    st.download_button(
        "Download Dataset CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="housing_dataset.csv",
        mime="text/csv"
    )

# ---------------- Predict Price ----------------
elif page == "Predict Price":
    show_header("Predict House Price", "Enter property details to get a model-based estimate.")
    st.info("The prediction uses the dataset's original price units. It is an estimate, not a verified market valuation.")

    with st.form("prediction_form"):
        st.subheader("Property Details")
        cols = st.columns(2)
        user_values = {}

        for index, feature in enumerate(features):
            container = cols[index % 2]
            with container:
                if feature in numeric_features:
                    series = pd.to_numeric(df[feature], errors="coerce").dropna()
                    default = float(series.median()) if not series.empty else 0.0
                    minimum = float(series.min()) if not series.empty else 0.0
                    maximum = float(series.max()) if not series.empty else max(default * 2, 1.0)
                    if minimum == maximum:
                        maximum = minimum + 1.0
                    user_values[feature] = st.number_input(
                        feature.replace("_", " ").title(),
                        min_value=minimum,
                        max_value=maximum,
                        value=min(max(default, minimum), maximum),
                        step=1.0
                    )
                else:
                    choices = df[feature].dropna().astype(str).unique().tolist()
                    if not choices:
                        choices = ["Unknown"]
                    user_values[feature] = st.selectbox(
                        feature.replace("_", " ").title(), choices
                    )

        submitted = st.form_submit_button("Predict House Price", type="primary", use_container_width=True)

    if submitted:
        input_df = pd.DataFrame([user_values], columns=features)
        try:
            predicted_price = float(model.predict(input_df)[0])
            st.success("Prediction generated successfully!")
            st.metric("Estimated House Price", money(predicted_price))
            st.caption("This estimate is generated by the trained Linear Regression model.")

            record = user_values.copy()
            record["predicted_price"] = predicted_price
            record["timestamp"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.prediction_history.append(record)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

# ---------------- Model Performance ----------------
elif page == "Model Performance":
    show_header("Model Performance", "Evaluate how the trained regression model performs on test data.")
    metrics = bundle.get("metrics", {})
    if metrics:
        m1, m2, m3 = st.columns(3)
        m1.metric("R² Score", f"{metrics.get('r2', 0):.4f}")
        m2.metric("MAE", money(metrics.get("mae", 0)))
        m3.metric("RMSE", money(metrics.get("rmse", 0)))
    else:
        st.warning("Your model file does not contain evaluation metrics. Run train_model.py again.")

    actual = bundle.get("actual_test", [])
    predicted = bundle.get("predicted_test", [])
    if actual and predicted:
        result_df = pd.DataFrame({"Actual Price": actual, "Predicted Price": predicted})
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.scatterplot(data=result_df, x="Actual Price", y="Predicted Price", ax=ax)
        low = min(result_df.min())
        high = max(result_df.max())
        ax.plot([low, high], [low, high], linestyle="--")
        ax.set_title("Actual vs Predicted Prices")
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Test Predictions")
        st.dataframe(result_df.head(20), use_container_width=True)
        st.download_button(
            "Download Test Predictions",
            data=result_df.to_csv(index=False).encode("utf-8"),
            file_name="model_test_predictions.csv",
            mime="text/csv"
        )
    else:
        st.info("Test prediction details are not available. Retrain using the supplied train_model.py.")

# ---------------- Prediction History ----------------
elif page == "Prediction History":
    show_header("Prediction History", "Review estimates generated during this browser session.")
    history = st.session_state.prediction_history
    if history:
        history_df = pd.DataFrame(history)
        st.dataframe(history_df, use_container_width=True)
        st.download_button(
            "Download Prediction History",
            data=history_df.to_csv(index=False).encode("utf-8"),
            file_name="prediction_history.csv",
            mime="text/csv"
        )
        if st.button("Clear History"):
            st.session_state.prediction_history = []
            st.rerun()
    else:
        st.info("No predictions yet. Go to Predict Price and generate your first estimate.")
