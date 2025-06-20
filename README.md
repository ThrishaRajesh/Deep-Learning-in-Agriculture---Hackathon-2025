###🌾​̲ ​̲𝑺​̲𝒎​̲𝒂​̲𝒓​̲𝒕​̲ ​̲𝑨​̲𝒈​̲𝒓​̲𝒊​̲𝒄​̲𝒖​̲𝒍​̲𝒕​̲𝒖​̲𝒓​̲𝒆​̲ ​̲𝑰​̲𝒏​̲𝒕​̲𝒆​̲𝒍​̲𝒍​̲𝒊​̲𝒈​̲𝒆​̲𝒏​̲𝒄​̲𝒆​̲ ​̲𝑺​̲𝒚​̲𝒔​̲𝒕​̲𝒆​̲𝒎​̲ ​̲|​̲ ​̲𝑫​̲𝒆​̲𝒆​̲𝒑​̲ ​̲𝑳​̲𝒆​̲𝒂​̲𝒓​̲𝒏​̲𝒊​̲𝒏​̲𝒈​̲ ​̲+​̲ ​̲𝑺​̲𝒕​̲𝒓​̲𝒆​̲𝒂​̲𝒎​̲𝒍​̲𝒊​̲𝒕​̲ ​̲+​̲ ​̲𝑺​̲𝒂​̲𝒕​̲𝒆​̲𝒍​̲𝒍​̲𝒊​̲𝒕​̲𝒆​̲ ​̲𝑫​̲𝒂​̲𝒕​̲𝒂

An end-to-end smart agriculture solution powered by **Deep Learning**, **Computer Vision**, **Weather Forecasting**, and **Satellite NDVI Analysis**, built for **real-time decision support in farming**.

🚀 Built during **Internship - 2 - Hackathon-2025** to address productivity, disease management, and crop planning with ease.

---

## ✅ Core Modules

1. **🍃 Leaf Disease Detection via Image Upload**
2. **📷 Real-Time Leaf Disease Detection via Webcam**
3. **🌱 Crop Yield Prediction using Satellite NDVI + Weather**
4. **🔥 Grad-CAM Visual Explanations**
5. **📊 Analytics Dashboard with Visual Insights**
6. **🖼️ Unified Streamlit Web Interface**

---

## 🧠 Key Features

- ✅ Deep Learning–powered image classification (MobileNetV2)
- ✅ Grad-CAM explainability for model transparency
- ✅ Real-time detection from camera (OpenCV)
- ✅ Crop yield prediction using tabular + NDVI + weather fusion
- ✅ Weather-NDVI correlation analysis
- ✅ Interactive dashboard built with Plotly + Streamlit

---

## 🧰 Tech Stack

| Layer              | Tools/Technologies                             |
|-------------------|-------------------------------------------------|
| Frontend UI        | Streamlit                                      |
| Deep Learning      | TensorFlow / Keras, MobileNetV2, Grad-CAM      |
| Weather Forecast   | OpenWeatherMap API                             |
| Satellite Data     | NDVI data from Sentinel-2 / Google Earth Engine|
| Visualization      | Plotly, Matplotlib, Seaborn                    |
| Tabular Modelling  | Scikit-learn, Pandas, NumPy                    |

---

## 👥 Team Roles & Responsibilities

### 👨‍💻 Person A – **Real-Time Detection via Webcam**

- 🎥 Captures live frames via **OpenCV**
- 🧠 Model trained using MobileNetV2
- 🔍 Runs predictions on each frame in real-time

> 📁 `leaf_disease_camera.py`

---

### 👩‍💻 Person B – **Leaf Disease Detection + Grad-CAM**

- 📦 Uses **PlantVillage dataset**
- 📤 Upload image → Predict → Display heatmap
- 🧠 Loads same DL model as Person A
- 🖼️ Uses Grad-CAM to visualize the model’s attention

> 📁 `leaf_disease_upload.py`

---

### 👩‍💻 Person C – **Crop Yield Prediction + Dashboard**

- 🔬 Fuses NDVI + Year + Crop into a prediction model
- 📊 Dashboard visualizes:
  - Yield trends
  - NDVI correlations
  - Future predictions (2024–2030)
- 🔮 Uses synthetic NDVI from weather for yield forecasting

> 📁 `Visualization.py`

---
