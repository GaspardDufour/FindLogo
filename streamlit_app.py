import streamlit as st
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile

# Charger le modèle YOLO
model = YOLO("yolo11n.pt")  # Remplace par le chemin vers ton modèle entraîné

st.title("Détection de Logo avec YOLO")

# Upload d'image
uploaded_file = st.file_uploader("Choisissez une image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Lire l'image
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    
    # Sauvegarde temporaire de l'image pour la passer au modèle
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        image.save(temp_file.name)
        image_path = temp_file.name
    
    # Exécuter la détection YOLO
    results = model(image_path)
    
    # Dessiner les boîtes de détection
    for result in results:
        for box in result.boxes.data:
            x1, y1, x2, y2, conf, cls = box.tolist()
            cls = int(cls)
            cv2.rectangle(image_np, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
            cv2.putText(image_np, f"Classe {cls}", (int(x1), int(y1) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Afficher l'image avec les détections
    st.image(image_np, caption="Résultat de la détection", use_container_width=True)
