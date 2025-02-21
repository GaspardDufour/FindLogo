import streamlit as st
import cv2
import numpy as np
import torch
import os
import requests
from PIL import Image
from ultralytics import YOLO
import tempfile
from network_swinir import SwinIR

# URL du modèle SwinIR
MODEL_URL = "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth"
MODEL_PATH = "SwinIR_model.pth"

# Charger le modèle YOLO (remplace par ton modèle)
YOLO_MODEL = YOLO("last.pt")

# Dictionnaire des classes détectées par YOLO
class_dict = {
    0: "Microsoft",
    1: "Google",
    2: "Intel",
    3: "Nvidia",
    4: "Apple",
    # Ajoute d'autres classes ici si nécessaire
}

# --- Télécharger SwinIR si nécessaire ---
def download_model():
    if not os.path.exists(MODEL_PATH):
        st.warning("📥 Téléchargement du modèle SwinIR en cours...")
        response = requests.get(MODEL_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        st.success("✅ Modèle téléchargé avec succès !")

# --- Charger SwinIR ---
@st.cache_resource
def load_swinir_model():
    download_model()
    model = SwinIR(
        upscale=4, img_size=64, window_size=8, img_range=1.,
        depths=[6, 6, 6, 6, 6, 6, 6, 6, 6], embed_dim=240,
        num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8], mlp_ratio=2,
        upsampler='nearest+conv', resi_connection='3conv'
    )
    
    pretrained_model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    param_key_g = 'params_ema' if 'params_ema' in pretrained_model else 'params'
    model.load_state_dict(pretrained_model[param_key_g], strict=False)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return model.to(device), device

# --- Prétraitement SwinIR ---
def preprocess_image(img):
    img = np.array(img).astype(np.float32) / 255.
    img = img[:, :, [2, 1, 0]]  # BGR → RGB
    img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float().unsqueeze(0)
    return img

# --- Post-traitement SwinIR ---
def postprocess_image(output):
    output = output.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output, (1, 2, 0))
    output = output[:, :, [2, 1, 0]]  # RGB → BGR
    output = (output * 255.0).round().astype(np.uint8)
    
    # Amélioration de la netteté
    kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
    return cv2.filter2D(output, -1, kernel)

# --- Fonction d'amélioration SwinIR ---
def enhance_image(image, model, device):
    img_tensor = preprocess_image(image).to(device)
    with torch.no_grad():
        output = model(img_tensor)
    return postprocess_image(output)

# --- Interface Streamlit - Amélioration d'image ---
def enhance_image_ui():
    st.title("Amélioration d'Image avec SwinIR")
    model, device = load_swinir_model()
    uploaded_file = st.file_uploader("Choisissez une image à améliorer", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Image originale", use_container_width=True)
        with st.spinner("Amélioration en cours..."):
            enhanced_img = enhance_image(image, model, device)
        # Convertir BGR → RGB avant affichage et téléchargement
        enhanced_img_rgb = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)
         # Affichage correct
        st.image(enhanced_img_rgb, caption="Image Améliorée", use_container_width=True)

        # Téléchargement avec les vraies couleurs
        st.download_button("Télécharger l'image améliorée",
                           data=cv2.imencode(".jpg", enhanced_img_rgb)[1].tobytes(),
                           file_name="image_amelioree.jpg",
                           mime="image/jpeg")

# --- Interface Streamlit - Détection de logos ---
def detect_logo_ui():
    st.title("Détection de Logo avec YOLO")
    uploaded_file = st.file_uploader("Choisissez une image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        # Sauvegarde temporaire
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            image.save(temp_file.name, "JPEG")
            image_path = temp_file.name

        # Exécuter YOLO
        results = YOLO_MODEL(image_path)
        detected_classes = []
        
        for result in results:
            for box in result.boxes.data:
                x1, y1, x2, y2, conf, cls = box.tolist()
                cls = int(cls)  # Convertir en entier
                
                # Dessiner la boîte
                cv2.rectangle(image_np, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
                
                # Ajouter la classe détectée
                detected_classes.append(cls)
                
                # Ajouter le texte de la classe détectée
                class_name = class_dict.get(cls, f"Classe {cls}")
                cv2.putText(image_np, class_name, (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Afficher l'image avec les détections
        st.image(image_np, caption="Résultat de la détection", use_container_width=True)

        # Afficher les classes détectées
        if detected_classes:
            st.write("### Logos détectés :")
            for cls in detected_classes:
                st.write(f"➡️ Logo détecté : **{class_dict.get(cls, f'Classe {cls}')}**")
        else:
            st.write("Aucun logo détecté.")

# --- Page d'accueil ---
def accueil():
    st.title("📌 Choisissez un service")
    option = st.radio("🔍 Sélectionnez l'option :", ("Détection de Logos", "Améliorateur d'Images"))

    if option == "Améliorateur d'Images":
        enhance_image_ui()
    else:
        detect_logo_ui()

# --- Exécution ---
if __name__ == "__main__":
    accueil()
