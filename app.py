
import streamlit as st
import pandas as pd
import joblib

# Load Trained Model
model_data = joblib.load("house_price_model.pkl")

model = model_data["model"]
features = model_data["features"]
numeric_features = model_data["numeric_features"]
categorical_features = model_data["categorical_features"]

# Load Dataset for Input Options
data = pd.read_csv("Housing.csv")

# Page Configuration
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.stApp {
    background-color: #f5f7fb;
}
.main-title {
    text-align: center;
    color: #173b65;
    font-size: 38px;
    font-weight: bold;
}
.subtitle {
    text-align: center;
    color: #555;
    font-size: 17px;
}
</style>
""", unsafe_allow_html=True)

# Application Heading
st.markdown(
    '<p class="main-title">🏠 House Price Predictor</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Predict your dream house price using Machine Learning</p>',
    unsafe_allow_html=True
)

st.divider()

st.subheader("Enter Property Details")

user_input = {}

# Input Form
with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    for i, feature in enumerate(features):

        current_col = col1 if i % 2 == 0 else col2

        with current_col:

            if feature in numeric_features:

                values = pd.to_numeric(
                    data[feature], errors="coerce"
                ).dropna()

                minimum = float(values.min())
                maximum = float(values.max())
                default = float(values.median())

                if minimum == maximum:
                    user_input[feature] = st.number_input(
                        feature.replace("_", " ").title(),
                        value=default
                    )
                else:
                    user_input[feature] = st.number_input(
                        feature.replace("_", " ").title(),
                        min_value=minimum,
                        max_value=maximum,
                        value=default
                    )

            else:

                options = data[feature].dropna().unique().tolist()

                user_input[feature] = st.selectbox(
                    feature.replace("_", " ").title(),
                    options
                )

    submit = st.form_submit_button(
        "Predict House Price",
        use_container_width=True
    )

# Prediction
if submit:

    input_data = pd.DataFrame(
        [user_input],
        columns=features
    )

    prediction = model.predict(input_data)[0]

    st.divider()

    st.success("House Price Prediction Completed!")

    st.subheader("Estimated House Price")

    st.markdown(
        f"""
        <div style="
            background-color:#173b65;
            padding:25px;
            border-radius:15px;
            text-align:center;
        ">
            <h2 style="color:white;">
                ₹ {prediction:,.2f}
            </h2>
            <p style="color:white;">
                Predicted Property Value
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.info(
        "This price is an estimated value generated "
        "by a Machine Learning model. Actual market "
        "prices may vary."
    )
