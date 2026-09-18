"""
Sydney Housing Price Prediction — Streamlit App
Part 6: Final Deployment

Run with:
    pip install streamlit joblib scikit-learn pandas
    streamlit run app.py

Loads the model trained and saved in the project notebook
(housing_price_model.joblib, feature_cols.joblib) and lets a user
enter property details to get a predicted sale price.
"""

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Sydney Housing Price Predictor", page_icon="🏠")

st.title("🏠 Sydney Housing Price Predictor")
st.write(
    "Estimate a sale price for a property in **Blacktown**, **Parramatta**, "
    "or **Randwick**, based on a model trained on 100 recent sold listings."
)


@st.cache_resource
def load_model():
    model = joblib.load("housing_price_model.joblib")
    feature_cols = joblib.load("feature_cols.joblib")
    return model, feature_cols


try:
    model, feature_cols = load_model()
except FileNotFoundError:
    st.error(
        "Model files not found. Run the project notebook first so it creates "
        "`housing_price_model.joblib` and `feature_cols.joblib` in this folder."
    )
    st.stop()

st.header("Enter property details")

col1, col2 = st.columns(2)

with col1:
    suburb = st.selectbox("Suburb", ["Blacktown", "Parramatta", "Randwick"])
    property_type = st.selectbox(
        "Property type", ["house", "apartment", "townhouse", "unit", "studio"]
    )
    bedrooms = st.number_input("Bedrooms", min_value=0, max_value=10, value=3)
    bathrooms = st.number_input("Bathrooms", min_value=0, max_value=10, value=2)

with col2:
    parking = st.number_input("Parking spaces", min_value=0, max_value=10, value=1)
    land_size = st.number_input(
        "Land size (m²) — enter 0 for apartments/units with no private land",
        min_value=0,
        max_value=5000,
        value=0,
    )
    days_since_first_sale = st.number_input(
        "Days since first sale in training data (leave default unless you know this)",
        min_value=0,
        max_value=1000,
        value=90,
    )

if st.button("Predict sale price", type="primary"):
    total_rooms = bedrooms + bathrooms
    bed_bath_ratio = bedrooms / bathrooms if bathrooms > 0 else bedrooms
    has_land = int(land_size > 0)

    input_row = pd.DataFrame([{
        "Suburb": suburb,
        "Property_Type": property_type,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Parking": parking,
        "Land_Size_m2": land_size,
        "Total_Rooms": total_rooms,
        "Bed_Bath_Ratio": bed_bath_ratio,
        "Has_Land": has_land,
        "Days_Since_First_Sale": days_since_first_sale,
    }])[feature_cols]

    prediction = model.predict(input_row)[0]

    st.success(f"### Estimated sale price: ${prediction:,.0f}")
    st.caption(
        "This is a prototype decision-support estimate from a model trained on only "
        "100 properties across three suburbs — treat it as a starting point for "
        "discussion, not a formal valuation. See the project report for known "
        "limitations and failure cases."
    )

st.divider()
st.caption(
    "Built for the SIT720 Sydney Housing Price Prediction mini project. "
    "Model: see notebook Section 3.6 for which algorithm was selected and why."
)
