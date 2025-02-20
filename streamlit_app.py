import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile

# Charger le modèle YOLO
model = YOLO("last.pt")  # Remplace par le chemin vers ton modèle

# Dictionnaire des classes et leurs noms
class_dict = {
    0: "Microsoft",  # Exemple de classe 0
    1: "Google",  # Exemple de classe 1
    2: "Intel",  # Exemple de classe 2
    3: "Nvidia",  # Exemple de classe 3
    4: "Apple",  # Exemple de classe 4
    # Ajoute d'autres classes ici si nécessaire
}

def detect_logo():
    st.title("Détection de Logo avec YOLO")

    # Upload d'image
    uploaded_file = st.file_uploader("Choisissez une image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Lire l'image
        image = Image.open(uploaded_file)

        # Convertir en RGB si nécessaire
        if image.mode != "RGB":
            image = image.convert("RGB")

        image_np = np.array(image)

        # Sauvegarde temporaire
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            image.save(temp_file.name, "JPEG")
            image_path = temp_file.name

        # Exécuter YOLO
        results = model(image_path)
        detected_classes = []
        # Dessiner les boîtes et écrire le nom de la classe
        for result in results:
            for box in result.boxes.data:
                x1, y1, x2, y2, conf, cls = box.tolist()
                cls = int(cls)  # Convertir en entier

                # Dessiner la boîte
                cv2.rectangle(image_np, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)

                # Ajouter la classe détectée à la liste
                detected_classes.append(cls)

                # Ajouter le texte de la classe détectée à côté de la boîte
                class_name = class_dict.get(cls, f"Classe {cls}")  # Utiliser le dictionnaire pour récupérer le nom de la classe
                cv2.putText(image_np, class_name, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Afficher l'image avec les détections
        st.image(image_np, caption="Résultat de la détection", use_container_width=True)

        # Afficher les classes détectées
        if detected_classes:
            st.write("### Logos détectés :")
            for cls in detected_classes:
                class_name = class_dict.get(cls, f"Classe {cls}")
                st.write(f"➡️ Logo détecté : **{class_name}**")
        else:
            st.write("Aucun logo détecté.")

def enhance_image():
    st.title("Amélioration d'Image")

    # Upload d'image
    uploaded_file = st.file_uploader("Choisissez une image à améliorer", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Lire l'image
        image = Image.open(uploaded_file)

        # Convertir en RGB si nécessaire
        if image.mode != "RGB":
            image = image.convert("RGB")

        image_np = np.array(image)

        # Appliquer un traitement pour améliorer l'image (exemple avec OpenCV)
        enhanced_image_np = cv2.convertScaleAbs(image_np, alpha=1.2, beta=30)  # Amélioration de la luminosité et du contraste

        # Convertir l'image améliorée pour l'affichage
        enhanced_image = Image.fromarray(enhanced_image_np)

        # Afficher l'image améliorée
        st.image(enhanced_image, caption="Image Améliorée", use_container_width=True)

# Page d'accueil pour choisir un service
def accueil():
    st.title("Choisissez un service")
    st.write("Sélectionnez l'option qui vous intéresse :")
    
    option = st.radio("Services disponibles", ("Détecteur de Logos", "Améliorateur d'Images"))
    
    if option == "Détecteur de Logos":
        detect_logo()
        
    elif option == "Améliorateur d'Images":
        enhance_image()

# Fonction principale
def main():
    accueil()

if __name__ == "__main__":
    main()
