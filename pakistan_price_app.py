import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("pakistan_property_data.csv")

# Define features and target
features = ['City', 'Area', 'Size (Marla)', 'House Style', 'Bedrooms', 'Bathrooms', 'Garage', 'Garden', 'Year Built']
target = 'Price (PKR)'

X = data[features]
y = data[target]

# Define preprocessing
categorical_features = ['City', 'Area', 'House Style']
numeric_features = ['Size (Marla)', 'Bedrooms', 'Bathrooms', 'Garage', 'Garden', 'Year Built']

preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
], remainder='passthrough')

# Define model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# Streamlit UI
st.title("🏘️ Pakistan Property Price Predictor")

city = st.selectbox("City", sorted(data['City'].unique()))
area = st.selectbox("Area", sorted(data[data['City'] == city]['Area'].unique()))
size = st.slider("Plot Size (in Marla)", 3.0, 30.0, 10.0, step=0.5)
house_style = st.selectbox("House Style", sorted(data['House Style'].unique()))
bedrooms = st.slider("Number of Bedrooms", 2, 7, 3)
bathrooms = st.slider("Number of Bathrooms", 1, 5, 2)
garage = st.selectbox("Garage", [0, 1])
garden = st.selectbox("Garden", [0, 1])
year_built = st.slider("Year Built", 1995, 2023, 2010)

if st.button("Predict Price"):
    input_df = pd.DataFrame([{
        'City': city,
        'Area': area,
        'Size (Marla)': size,
        'House Style': house_style,
        'Bedrooms': bedrooms,
        'Bathrooms': bathrooms,
        'Garage': garage,
        'Garden': garden,
        'Year Built': year_built
    }])

    predicted_price = model.predict(input_df)[0]
    st.success(f"Estimated Price: PKR {predicted_price:,.0f}")

# Optional model evaluation
if st.checkbox("Show Model Evaluation"):
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    st.write(f"Mean Absolute Error on test set: PKR {mae:,.0f}")
    fig, ax = plt.subplots()
    ax.scatter(y_test, preds, alpha=0.5)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    ax.set_xlabel("Actual Price")
    ax.set_ylabel("Predicted Price")
    ax.set_title("Actual vs Predicted Prices")
    st.pyplot(fig)
