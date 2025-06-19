import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go 
import tensorflow as tf
import os
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

# ---------- Page Configuration ----------
st.set_page_config(page_title="🌱 Crop Yield Insights", layout="wide")

# ---------- Load Data ----------
@st.cache_data
def load_data():
    df = pd.read_csv("final_combined_dataset.csv")
    weather_df = pd.read_csv("forecast_weather_data_with_ndvi.csv")
    model = tf.keras.models.load_model("crop_yield_model.h5", compile=False)
    return df, weather_df, model

df, weather_df, model = load_data()
# --- Derive NDVI from Weather ---
avg_temp = weather_df['Temperature_C'].mean()
avg_rain = weather_df['Rainfall_mm'].mean()
avg_humidity = weather_df['Humidity_%'].mean()

# Synthetic NDVI proxy influenced by weather parameters
weather_based_ndvi = 0.27 + ((avg_temp - 25) * 0.001) + ((avg_rain - 5) * 0.0005) + ((avg_humidity - 60) * 0.0003)
weather_based_ndvi = np.clip(weather_based_ndvi, 0.2, 0.9)  # Keep it in realistic NDVI range

# Display it with style
st.markdown(f"""
<div style='font-size:16px; padding:10px; background-color:#1f1f1f; border-radius:10px;'>
<b>🌿 Estimated NDVI from Weather Forecast:</b> <span style='color:#ffa726;'>{weather_based_ndvi:.3f}</span><br>
<small>NDVI is estimated using average temperature, rainfall, and humidity from the forecast above.</small>
</div>
""", unsafe_allow_html=True)

df['NDVI_mean'] = df['NDVI_mean'].fillna(df['NDVI_mean'].mean())
df['Year'] = df['Year'].astype(int)
le = LabelEncoder()
df['Crop_enc'] = le.fit_transform(df['Crop'])

# ---------- Sidebar UI ----------
with st.sidebar:
    st.header("🔍 Select Crop(s)")
    crops = df["Crop"].unique()
    selected_crops = []

    for crop in crops:
        col1, col2 = st.columns([1, 3])
        with col1:
            img_path = f"crop_images/{crop.lower().replace(',', '').replace('(', '').replace(')', '').replace(' ', '_')}.jpg"
            if os.path.exists(img_path):
                st.image(img_path, width=100)
            else:
                st.write("🚫")
        
        with col2:
            checkbox_key = crop.lower().replace(",", "").replace("(", "").replace(")", "").replace(".", "").replace(" ", "_")
            if st.checkbox(f"{crop}", key=f"chk_{checkbox_key}", value=True):
                selected_crops.append(crop)
            
            with st.expander("ℹ️ View Conditions"):
                # Manually coded info for each crop
                if crop == "Apples":
                    st.markdown("🌡️ Temp: 15–25°C\n\n🌧️ Rainfall: 100–125 cm\n\n🌱 Soil: Well-drained loamy")
                elif crop == "Areca nuts":
                    st.markdown("🌡️ Temp: 20–35°C\n\n🌧️ Rainfall: 150–200 cm\n\n🌱 Soil: Laterite or red loam")
                elif crop == "Cauliflowers and broccoli":
                    st.markdown("🌡️ Temp: 15–20°C\n\n🌧️ Rainfall: 50–80 cm\n\n🌱 Soil: Sandy loam")
                elif crop == "Chillies and pepper, green (Capsicum spp. Pimenta spp.)":
                    st.markdown("🌡️ Temp: 20–30°C\n\n🌧️ Rainfall: 60–120 cm\n\n🌱 Soil: Loamy, rich in organic matter")
                elif crop == "Coconuts, in shell":
                    st.markdown("🌡️ Temp: 25–35°C\n\n🌧️ Rainfall: 150–250 cm\n\n🌱 Soil: Sandy loam")
                elif crop == "Coffee, green":
                    st.markdown("🌡️ Temp: 18–30°C\n\n🌧️ Rainfall: 120–200 cm\n\n🌱 Soil: Well-drained, slightly acidic")
                elif crop == "Maize (corn)":
                    st.markdown("🌡️ Temp: 21–30°C\n\n🌧️ Rainfall: 50–100 cm\n\n🌱 Soil: Loamy, fertile")
                elif crop == "Peas, green":
                    st.markdown("🌡️ Temp: 13–18°C\n\n🌧️ Rainfall: 60–80 cm\n\n🌱 Soil: Loamy, well-drained")
                elif crop == "Potatoes":
                    st.markdown("🌡️ Temp: 15–20°C\n\n🌧️ Rainfall: 100–150 cm\n\n🌱 Soil: Sandy loam")
                elif crop == "Rice":
                    st.markdown("🌡️ Temp: 20–30°C\n\n🌧️ Rainfall: 100–200 cm\n\n🌱 Soil: Clayey, water-retentive")
                elif crop == "Sugar cane":
                    st.markdown("🌡️ Temp: 20–35°C\n\n🌧️ Rainfall: 75–150 cm\n\n🌱 Soil: Deep rich loam")
                elif crop == "Tea leaves":
                    st.markdown("🌡️ Temp: 16–30°C\n\n🌧️ Rainfall: 150–300 cm\n\n🌱 Soil: Slightly acidic")
                elif crop == "Wheat":
                    st.markdown("🌡️ Temp: 10–25°C\n\n🌧️ Rainfall: 50–100 cm\n\n🌱 Soil: Clay loam or loamy")

    st.markdown("---")
    year_range = st.slider("📆 Select Year Range", min_value=int(df["Year"].min()), max_value=2023, value=(2013, 2023))
    show_weather = st.checkbox("☁️ Show Weather Impact Analysis", value=True)

# ---------- Filtered Data ----------
df_filtered = df[(df["Crop"].isin(selected_crops)) & (df["Year"].between(year_range[0], year_range[1]))]

# ---------- Dataset Metrics ----------
st.title("🌾 Crop Yield Prediction & Dashboard")
st.markdown("##### 🚀 Powered by Satellite NDVI + Weather + ML Model")

col1, col2, col3, col4 = st.columns(4)
col1.metric("📊 Total Records", len(df_filtered))
col2.metric("📅 Year Range", f"{year_range[0]} - {year_range[1]}")
col3.metric("🌿 Crops Selected", len(selected_crops))
col4.metric("🛰 Avg. NDVI", round(df_filtered['NDVI_mean'].mean(), 3))

# ------------------ Yield Trend Line ------------------
st.subheader("📈 Crop Yield Trends (kg/ha)")
fig_line = px.line(df_filtered, x='Year', y='Yield_kg_per_ha', color='Crop', markers=True)
fig_line.update_layout(template="plotly_dark", title_x=0.5)
st.plotly_chart(fig_line, use_container_width=True)

# ------------------ NDVI vs Yield ------------------
st.subheader("🛰️ NDVI vs Crop Yield")
fig_ndvi = px.scatter(df_filtered, x='NDVI_mean', y='Yield_kg_per_ha', color='Crop', trendline="ols")
fig_ndvi.update_layout(template="plotly_dark", title_x=0.5)
st.plotly_chart(fig_ndvi, use_container_width=True)

# ------------------ Predict Future Yields ------------------
st.subheader("🔮 Predict Future Crop Yields (2024 - 2030)")
future_years = list(range(2024, 2031))
forecast_data = []

for crop in selected_crops:
    crop_encoded = le.transform([crop])[0]
    for year in future_years:
        input_array = np.array([[crop_encoded, year, weather_based_ndvi]])
        predicted_yield = model.predict(input_array, verbose=0)[0][0]
        forecast_data.append([crop, year, weather_based_ndvi, predicted_yield])

forecast_df = pd.DataFrame(forecast_data, columns=["Crop", "Year", "NDVI_mean", "Predicted_Yield_kg_per_ha"])
fig_forecast = px.line(forecast_df, x="Year", y="Predicted_Yield_kg_per_ha", color="Crop", markers=True,
                       title="📉 Predicted Yield Trends (Based on Weather-Inferred NDVI)")
fig_forecast.update_layout(template="plotly_dark", title_x=0.5)
st.plotly_chart(fig_forecast, use_container_width=True)

# ------------------ Box Plot Yield by Crop ------------------
st.subheader("📦 Yield Distribution by Crop")
fig_box = px.box(df_filtered, x="Crop", y="Yield_kg_per_ha", color="Crop", title="Yield Spread")
fig_box.update_layout(template="plotly_dark", title_x=0.5)
st.plotly_chart(fig_box, use_container_width=True)

# ------------------ NDVI Heatmap ------------------
st.subheader("🌿 NDVI Heatmap (Crop vs Year)")
ndvi_pivot = df.pivot_table(index="Crop", columns="Year", values="NDVI_mean")
fig_heatmap = px.imshow(
    ndvi_pivot,
    aspect="auto",
    color_continuous_scale="Viridis",
    labels=dict(x="Year", y="Crop", color="NDVI Mean"),
    title="Average NDVI Variation per Crop Over the Years"
)
fig_heatmap.update_layout(template="plotly_dark", title_x=0.5)
st.plotly_chart(fig_heatmap, use_container_width=True)

# ------------------ Feature Correlation Matrix ------------------
st.subheader("🧠 Feature Correlation Matrix")
corr_matrix = df[["NDVI_mean", "Yield_kg_per_ha"]].corr()
fig_corr = px.imshow(
    corr_matrix,
    text_auto=True,
    color_continuous_scale="RdBu_r",
    title="Correlation Between NDVI and Crop Yield"
)
fig_corr.update_layout(template="plotly_dark", title_x=0.5)
st.plotly_chart(fig_corr, use_container_width=True)

# ------------------ Bar: Top 10 Crops ------------------
st.subheader("🏆 Top Performing Crops (Avg Yield)")
avg_yield = df.groupby("Crop")["Yield_kg_per_ha"].mean().sort_values(ascending=False).head(10)
fig_bar = px.bar(avg_yield, x=avg_yield.index, y=avg_yield.values,
                 title="Top 10 Crops by Avg Yield", labels={"y": "Yield (kg/ha)", "x": "Crop"})
fig_bar.update_layout(template="plotly_dark", title_x=0.5)
st.plotly_chart(fig_bar, use_container_width=True)

# ---------- Weather Forecast Analysis ----------
if show_weather:
    st.subheader("☁️ Weather Forecast Analysis")
    weather_df['DateTime'] = pd.to_datetime(weather_df['DateTime'])
    weather_df['Temp_C'] = weather_df['Temperature_C'] - 273.15

    col5, col6, col7 = st.columns(3)
    with col5:
        fig_temp = px.line(weather_df, x='DateTime', y='Temp_C', title="🌡️ Temperature (°C)")
        st.plotly_chart(fig_temp, use_container_width=True)
    with col6:
        fig_rain = px.bar(weather_df, x='DateTime', y='Rainfall_mm', title="🌧️ Rainfall (mm)")
        st.plotly_chart(fig_rain, use_container_width=True)
    with col7:
        fig_humid = px.line(weather_df, x='DateTime', y='Humidity_%', title="💧 Humidity (%)")
        st.plotly_chart(fig_humid, use_container_width=True)

    st.markdown("#### 🧪 Weather Impact Interpretation")
    st.markdown("""
    - 🌡️ High temperature may reduce yield potential.
    - 🌧️ Rainfall improves soil moisture but excessive may harm.
    - 💧 Balanced humidity supports crop health.
    """)
    fig_ndvi_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=weather_based_ndvi,
    title={'text': "NDVI Derived from Weather"},
    gauge={
        'axis': {'range': [0.2, 0.9]},
        'bar': {'color': "green"},
        'steps': [
            {'range': [0.2, 0.4], 'color': "#ef5350"},
            {'range': [0.4, 0.6], 'color': "#ffca28"},
            {'range': [0.6, 0.9], 'color': "#66bb6a"}
        ]
    }
))
st.plotly_chart(fig_ndvi_gauge, use_container_width=True)


# ---------- Footer ----------
st.markdown("---")
st.success("🎯 Dashboard built with real FAO + NDVI + Weather data | Model MAE: ~16340.32 kg/ha")
st.caption("🔗 Project by Person C | Hackathon-ready visual analytics")


