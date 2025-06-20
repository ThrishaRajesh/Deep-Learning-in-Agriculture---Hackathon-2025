import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st
from utils.gradcam import make_gradcam_heatmap, overlay_gradcam
import cv2




# 🌿 Set page config
st.set_page_config(page_title="🌿 Leaf Disease Detector", layout="centered")

# 🌿 Title with gradient
st.markdown("""
<h1 class="main-title">🌿 Leaf Disease Detector 🌿</h1>
<style>
.main-title {
    text-align: center;
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(to right, #27ae60, #2ecc71);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-top: 1rem;
    margin-bottom: 2rem;
}
</style>
""", unsafe_allow_html=True)

# 🍃 Animated emoji leaves
st.markdown("""
<div class="emoji-leaf-container">
  <span class="leaf-emoji" style="left: 5%; animation-delay: 0s;">🍃</span>
  <span class="leaf-emoji" style="left: 15%; animation-delay: 2s;">🍂</span>
  <span class="leaf-emoji" style="left: 30%; animation-delay: 1s;">🌿</span>
  <span class="leaf-emoji" style="left: 45%; animation-delay: 3s;">🍃</span>
  <span class="leaf-emoji" style="left: 60%; animation-delay: 2.5s;">🍂</span>
  <span class="leaf-emoji" style="left: 75%; animation-delay: 4s;">🌿</span>
  <span class="leaf-emoji" style="left: 90%; animation-delay: 3s;">🍃</span>
</div>

<style>
.emoji-leaf-container {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    z-index: 1;
    overflow: hidden;
}

.leaf-emoji {
    position: absolute;
    font-size: 2rem;
    animation: emojiFall 12s linear infinite;
    opacity: 0.8;
}

@keyframes emojiFall {
    0% { transform: translateY(-100px) rotate(0deg); opacity: 0.7; }
    100% { transform: translateY(100vh) rotate(360deg); opacity: 0; }
}
</style>
""", unsafe_allow_html=True)

# Click-to-play hidden audio workaround
if st.button("🎵 Click here once to start soothing nature sounds"):
    st.markdown("""
    <audio autoplay loop style="display:none;">
      <source src="https://www.bensound.com/bensound-music/bensound-sunny.mp3" type="audio/mp3">
    </audio>
    """, unsafe_allow_html=True)








# Define paths
MODEL_PATH = "model/mobilenet_model.h5"
CLASS_INDEX_PATH = "training/class_indices.json"

@st.cache_resource
def load_model_and_classes():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_INDEX_PATH, "r") as f:
        class_indices = json.load(f)
    return model, class_indices

model, class_indices = load_model_and_classes()


# Reverse class_indices to map label -> class name
index_to_class = {v: k for k, v in class_indices.items()}

# Image Preprocessing
def preprocess_image(image, target_size=(224, 224)):
    img = image.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Prediction
def predict(img):
    img = img.convert("RGB").resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    preds = model.predict(img_array)
    pred_class_index = np.argmax(preds)
    pred_class = index_to_class[pred_class_index]
    confidence = float(np.max(preds))

    # Grad-CAM
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name="Conv_1")
    img_np = np.array(img.resize((224, 224)))
    cam = overlay_gradcam(img_np, heatmap)

    return pred_class, confidence, cam


# UI

st.write("Upload a plant leaf image and I’ll predict the disease using MobileNetV2.")

uploaded_file = st.file_uploader("📁 Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", width=300)

    if st.button('Predict Disease'):
        with st.container():
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            pred_class, confidence, cam = predict(img)
            st.success(f'🌿 Predicted: {pred_class} ({confidence:.2%} confidence)')


            col1, col2 = st.columns(2)

            with col1:
                st.image(img, caption="Original Leaf", use_container_width=True)

            with col2:
                heatmap_rgb = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
                st.image(heatmap_rgb, caption="Grad-CAM: Where the model focused ", use_container_width=True)

            if pred_class == "Pepper__bell___Bacterial_spot":
                explanation = """
                The model focused on dark, uneven spots and irregular edges of the leaf — classic signs of bacterial spot caused by *Xanthomonas*.
                """
            elif pred_class == "Pepper__bell___healthy":
                explanation = """
                The model detected a smooth, consistent green coloration without any blemishes — indicating a healthy pepper plant.
                """
            elif pred_class == "Potato___Early_blight":
                explanation = """
                The model identified brown lesions with concentric rings on older leaves — a common early blight pattern.
                """
            elif pred_class == "Potato___Late_blight":
                explanation = """
                It spotted large, dark blotches on leaves with water-soaked appearance — key symptoms of aggressive late blight infection.
                """
            elif pred_class == "Potato___healthy":
                explanation = """
                The model saw fresh, vibrant green leaves with no discoloration — a good indicator of plant health.
                """
            elif pred_class == "Tomato_Bacterial_spot":
                explanation = """
                The model found multiple small, greasy lesions on the leaf — typical of bacterial spot in tomatoes.
                """
            elif pred_class == "Tomato_Early_blight":
                explanation = """
                It focused on older leaves showing target-like spots with a yellow halo — symptoms of early blight.
                """
            elif pred_class == "Tomato_Late_blight":
                explanation = """
                The model detected rapidly spreading dark lesions, often accompanied by moldy growth — hallmarks of late blight.
                """
            elif pred_class == "Tomato_Leaf_Mold":
                explanation = """
                It picked up on yellowing on top of the leaves and olive-colored mold underneath — a clear sign of leaf mold.
                """
            elif pred_class == "Tomato_Septoria_leaf_spot":
                explanation = """
                The model noticed many tiny brown circular spots with distinct dark borders — a classic case of Septoria leaf spot.
                """
            elif pred_class == "Tomato_Spider_mites_Two_spotted_spider_mite":
                explanation = """
                It identified yellow speckling and delicate webbing — commonly left behind by two-spotted spider mites.
                """
            elif pred_class == "Tomato__Target_Spot":
                explanation = """
                The model found circular lesions with concentric rings — resembling a target pattern — a signature of Target Spot.
                """
            elif pred_class == "Tomato__Tomato_YellowLeaf__Curl_Virus":
                explanation = """
                It detected curled, yellowing leaves — typically caused by TYLCV, a viral disease spread by whiteflies.
                """
            elif pred_class == "Tomato__Tomato_mosaic_virus":
                explanation = """
                The model spotted mosaic-like mottling and leaf distortion — signs of a viral infection by Tomato Mosaic Virus.
                """
            elif pred_class == "Tomato_healthy":
                explanation = """
                The model recognized evenly shaped, dark green leaves without any abnormal textures — indicating a healthy tomato plant.
                """
            else:
                explanation = "The model made its prediction based on learned visual features specific to this class."

            # Show explanation inside an expandable box
            with st.expander("🔍 Why This Prediction?"):
                st.markdown(explanation)

            disease_info = {
                "Pepper__bell___Bacterial_spot": {
                    "description": "A bacterial disease causing small water-soaked lesions that turn dark and enlarge.",
                    "cause": "Xanthomonas campestris bacteria.",
                    "treatment": "Remove infected plants, apply copper-based bactericides."
                    },
                "Pepper__bell___healthy": {
                    "description": "No disease symptoms detected. Healthy plant with vibrant leaves.",
                    "cause": "N/A",
                    "treatment": "Continue regular care and monitoring."
                    },
                "Potato___Early_blight": {
                    "description": "Brown leaf spots with concentric rings, often starting on older leaves.",
                    "cause": "Fungal infection by *Alternaria solani*.",
                    "treatment": "Use fungicides like chlorothalonil, rotate crops."
                    },
                "Potato___Late_blight": {
                    "description": "Dark, rapidly expanding lesions on leaves and stems.",
                    "cause": "*Phytophthora infestans* fungus-like organism.",
                    "treatment": "Apply systemic fungicides, remove infected plants."
                    },
                "Potato___healthy": {
                    "description": "Potato plant appears healthy with no visible disease.",
                    "cause": "N/A",
                    "treatment": "Maintain good irrigation and monitor regularly."
                    },
                "Tomato_Bacterial_spot": {
                    "description": "Tiny dark lesions on leaves and fruits; may appear greasy.",
                    "cause": "*Xanthomonas* bacteria.",
                    "treatment": "Avoid overhead watering, apply copper-based sprays."
                    },
                "Tomato_Early_blight": {
                    "description": "Dark spots with concentric rings and yellowing leaves.",
                    "cause": "*Alternaria solani* fungus.",
                    "treatment": "Apply fungicides; remove affected leaves."
                    },
                "Tomato_Late_blight": {
                    "description": "Irregularly shaped water-soaked lesions with white fungal growth.",
                    "cause": "*Phytophthora infestans*.",
                    "treatment": "Destroy infected plants; use fungicides preventively."
                    },
                "Tomato_Leaf_Mold": {
                    "description": "Yellow spots on upper leaf surface, olive mold underneath.",
                    "cause": "*Fulvia fulva* fungus.",
                    "treatment": "Ensure ventilation, use sulfur-based fungicides."
                    },
                "Tomato_Septoria_leaf_spot": {
                    "description": "Numerous small, round leaf spots with gray centers.",
                    "cause": "*Septoria lycopersici* fungus.",
                    "treatment": "Use protective fungicides, remove debris."
                    },
                "Tomato_Spider_mites_Two_spotted_spider_mite": {
                    "description": "Yellow speckles and webbing on the underside of leaves.",
                    "cause": "Infestation by *Tetranychus urticae* (spider mites).",
                    "treatment": "Spray neem oil or insecticidal soap."
                    },
                "Tomato__Target_Spot": {
                    "description": "Dark circular lesions with yellow halos, resembling target rings.",
                    "cause": "*Corynespora cassiicola* fungus.",
                    "treatment": "Use fungicides, improve air circulation."
                    },
                "Tomato__Tomato_YellowLeaf__Curl_Virus": {
                    "description": "Curling, yellowing leaves with stunted growth.",
                    "cause": "TYLCV transmitted by whiteflies.",
                    "treatment": "Remove infected plants; control whitefly vectors."
                    },
                "Tomato__Tomato_mosaic_virus": {
                    "description": "Mosaic-like mottling on leaves, distorted fruits.",
                    "cause": "Tomato Mosaic Virus (ToMV).",
                    "treatment": "Destroy infected plants, disinfect tools."
                    },
                "Tomato_healthy": {
                    "description": "Tomato plant shows no symptoms. Lush green leaves.",
                    "cause": "N/A",
                    "treatment": "Keep up healthy care and prevent stress."
                    }
                }

            # Display detailed info
            if pred_class in disease_info:
                with st.expander("📘 Disease Information"):
                    st.markdown(f"**📝 Description:** {disease_info[pred_class]['description']}")
                    st.markdown(f"**🦠 Cause:** {disease_info[pred_class]['cause']}")
                    st.markdown(f"**💊 Treatment:** {disease_info[pred_class]['treatment']}")
            # Expander: Could the model be wrong?
            if confidence < 0.50:
                with st.expander("⚠️ Why This Might Be Wrong"):
                    st.warning("The model's confidence is below 50%.")
                    st.markdown("""
                    - The image might be **blurry or poorly lit**.
                    - Some diseases share **similar patterns** (e.g., Early vs Late Blight).
                    - The leaf might be **partially occluded** or damaged in a way not seen during training.
                    - The model was trained for limited epochs .
                    - The model might need **fine-tuning or more data** for this particular disease.
                    """)
            else:
                with st.expander("📌 How Confident is This Prediction?"):
                    st.success("Confidence is reasonably high. But if you're unsure, consider verifying with an expert.")



