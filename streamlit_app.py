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
    # Lire l'image avec PIL pour l'affichage, mais la convertir en format OpenCV (BGR)
    image = Image.open(uploaded_file)
    image_np = np.array(image)  # Convertir en numpy array pour OpenCV

    # Convertir en BGR (format OpenCV attendu)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Exécuter YOLO
    results = model(image_cv)

    detected_classes = []  # Liste des classes détectées

    # Dessiner les bounding boxes
    for result in results:
        for box in result.boxes.data:
            x1, y1, x2, y2, _, cls = box.tolist()
            cls = str(cls)
            detected_classes.append(cls)

            # Dessiner la bounding box
            cv2.rectangle(image_cv, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)

    # Afficher l'image avec les bounding boxes
    st.image(image_cv, caption="Résultat de la détection", use_container_width=True)

    # Afficher les classes détectées
    if detected_classes:
        st.write("### Classes détectées :")
        for cls in detected_classes:
            st.write(f"➡️ Classe détectée : **{cls}**")
    else:
        st.write("Aucune classe détectée.")
