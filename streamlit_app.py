import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile

# Charger le modèle YOLO
model = YOLO("yolo11n.pt")  # Remplace par ton modèle

st.title("Détection de Logo avec YOLO")

# Upload d'image
uploaded_file = st.file_uploader("Choisissez une image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Lire l'image
    image = Image.open(uploaded_file)

    # Convertir en RGB si nécessaire
    if image.mode == "P":
        image = image.convert("RGBA")
    else:
        image = image.convert("RGB")

    image_np = np.array(image)  # Convertir en numpy array pour OpenCV

    # Sauvegarde temporaire
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        image.save(temp_file.name, "JPEG")
        image_path = temp_file.name

    # Exécuter YOLO
    results = model(image_path)

    detected_classes = []  # Liste des classes détectées

    # Dessiner les bounding boxes
    for result in results:
        for box in result.boxes.data:
            x1, y1, x2, y2, _, cls = box.tolist()
            cls = int(cls)
            detected_classes.append(cls)

            # Dessiner la bounding box
            cv2.rectangle(image_np, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)

    # Afficher l'image avec les bounding boxes
    st.image(image_np, caption="Résultat de la détection", use_container_width=True)

    # Afficher les classes détectées
    if detected_classes:
        st.write("### Classes détectées :")
        for cls in detected_classes:
            st.write(f"➡️ Classe détectée : **{cls}**")
    else:
        st.write("Aucune classe détectée.")
