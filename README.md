# 🌾 Smart Agriculture Intelligence System | Deep Learning

An end-to-end smart agriculture solution powered by **Deep Learning**, **Computer Vision**, **Weather Forecasting**, and **Satellite NDVI Analysis**, built for **real-time decision support in farming**.

🚀 Built during **UG - BE - Internship - 2 - Hackathon-2025** to address productivity, disease management, and crop planning with ease.

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

### 👨‍💻 Yashaswini K M – **Real-Time Detection via Webcam**

- 🎥 Captures live frames via **OpenCV**
- 🧠 Uses a pre-trained **MobileNetV2 model**
- 🔍 Runs predictions on each frame in real-time

> 📁 `leaf_disease_camera.py`

---

### 👩‍💻 Vismaya M – **Leaf Disease Detection + Grad-CAM**

- 📊 Trained a custom CNN model on the PlantVillage dataset
- 📤 Allows users to upload leaf images for disease classification
- 🖼️ Implements Grad-CAM to show what parts of the leaf the model focused on
- 📚 Shows detailed disease information and treatment suggestions for user awareness

> 📁 `leaf_disease_upload.py`

---

### 👩‍💻 Thrisha R – **Crop Yield Prediction + Dashboard**

- 🔬 Fuses NDVI + Year + Crop into a prediction model
- 📊 Dashboard visualizes:
  - Yield trends
  - NDVI correlations
  - Future predictions (2024–2030)
- 🔮 Uses synthetic NDVI from weather for yield forecasting

> 📁 `Visualization.py`

---
