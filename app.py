import streamlit as st
import pickle
import pandas as pd

st.set_page_config(page_title="House Price Predictor", layout="wide")

# Load model
with open("house_price_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

st.title("🏠 House Price Predictor")
st.caption("End-to-End XGBoost ML Project")

st.divider()

# ---- Row 1 ----
col1, col2, col3, col4 = st.columns(4)

with col1:
    bedrooms = st.number_input("Bedrooms", 1, 10, 3)
with col2:
    bathrooms = st.number_input("Bathrooms", 1.0, 5.0, 2.0, step=0.5)
with col3:
    floors = st.number_input("Floors", 1, 3, 1)
with col4:
    condition = st.slider("Condition", 1, 5, 3)

# ---- Row 2 ----
col1, col2, col3, col4 = st.columns(4)

with col1:
    sqft_living = st.number_input("Living Area", 500, 5000, 1500)
with col2:
    sqft_lot = st.number_input("Lot Area", 500, 20000, 5000)
with col3:
    sqft_basement = st.number_input("Basement", 0, 2000, 0)
with col4:
    yr_built = st.number_input("Year Built", 1900, 2024, 2000)

# ---- Row 3 ----
col1, col2, col3, col4 = st.columns(4)

with col1:
    waterfront = st.selectbox("Waterfront", [0, 1])
with col2:
    view = st.slider("View", 0, 4, 0)
with col3:
    sale_year = st.number_input("Sale Year", 2014, 2025, 2014)
with col4:
    sale_month = st.slider("Sale Month", 1, 12, 6)

# ---- Row 4 (Location) ----
col1, col2 = st.columns(2)

with col1:
    city = st.text_input("City", "Seattle")
with col2:
    statezip = st.text_input("StateZip", "WA 98178")

st.divider()

# Predict button centered
center = st.columns([1,2,1])[1]
predict = center.button("Predict Price 💰", use_container_width=True)

if predict:
    input_dict = {
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'sqft_living': sqft_living,
        'sqft_lot': sqft_lot,
        'floors': floors,
        'waterfront': waterfront,
        'view': view,
        'condition': condition,
        'sqft_basement': sqft_basement,
        'yr_built': yr_built,
        'city': city,
        'statezip': statezip,
        'sale_year': sale_year,
        'sale_month': sale_month
    }

    input_df = pd.DataFrame([input_dict])
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    prediction = model.predict(input_df)[0]

    st.divider()

    center = st.columns([1,2,1])[1]
    with center:
        st.markdown("### 💰 Estimated House Price")

        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #1f2937, #111827);
            padding: 30px;
            border-radius: 16px;
            text-align: center;
            margin-top: 10px;
            box-shadow: 0 8px 20px rgba(0,0,0,0.3);
        ">
            <h1 style="color:#22c55e; font-size:52px; margin:0;">
                ${prediction:,.0f}
            </h1>
            <p style="color:#9ca3af; margin-top:8px;">
                Predicted using XGBoost Model
            </p>
        </div>
        """, unsafe_allow_html=True)
