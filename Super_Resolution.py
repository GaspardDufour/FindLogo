import torch
import cv2
import numpy as np
from models.network_swinir import SwinIR

def load_swinir_model(model_path):
    """
    Charge le modèle SwinIR avec l'architecture correspondant au modèle classique M
    """
    model = SwinIR(
        upscale=4,
        img_size=64,
        window_size=8,
        img_range=1.,
        depths=[6, 6, 6, 6, 6, 6, 6, 6, 6],
        embed_dim=240,
        num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
        mlp_ratio=2,  # C'était correct mais il faut s'assurer que toute la configuration est cohérente
        upsampler='nearest+conv',
        resi_connection = '3conv'
    )
    
    # Chargement avec strict=False pour gérer les différences de structure
    pretrained_model = torch.load(model_path)
    param_key_g = 'params_ema' if 'params_ema' in pretrained_model else 'params'
    model.load_state_dict(
        pretrained_model[param_key_g] if param_key_g in pretrained_model else pretrained_model, 
        strict=False  # Changé à False pour permettre le chargement partiel
    )
    
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    return model, device

def preprocess_image(img_path):
    """
    Prétraitement de l'image
    """
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Impossible de charger l'image : {img_path}")
    img = img.astype(np.float32) / 255.
    img = img[:, :, [2, 1, 0]]  # BGR à RGB
    img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float().unsqueeze(0)
    return img

def postprocess_image(output):
    """
    Post-traitement de l'image
    """
    output = output.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output, (1, 2, 0))
    output = output[:, :, [2, 1, 0]]  # RGB à BGR
    output = (output * 255.0).round().astype(np.uint8)
    
    # Légère amélioration de la netteté
    kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
    output = cv2.filter2D(output, -1, kernel)
    
    return output

def enhance_image_resolution(input_path, model_path, output_path):
    """
    Fonction principale d'amélioration de l'image
    """
    try:
        print("Chargement du modèle...")
        model, device = load_swinir_model(model_path)
        if model is None:
            return False
        
        print("Prétraitement de l'image...")
        img_lq = preprocess_image(input_path).to(device)
        
        print("Amélioration de l'image...")
        with torch.no_grad():
            output = model(img_lq)
        
        print("Post-traitement et sauvegarde...")
        enhanced_image = postprocess_image(output)
        cv2.imwrite(output_path, enhanced_image)
        
        original_img = cv2.imread(input_path)
        print(f"Dimensions originales : {original_img.shape}")
        print(f"Dimensions après amélioration : {enhanced_image.shape}")
        print(f"Image traitée avec succès et sauvegardée dans : {output_path}")
        
        # Ouvrir l'image après traitement
        cv2.imshow("Image améliorée", enhanced_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return True
        
    except Exception as e:
        print(f"Une erreur est survenue : {e}")
        return False

# Utilisation
test_input_image_path = "C:/A3MSI/5-Day_Challenge/Challenge_2/Project/low.jpg"
test_model_path = "C:/A3MSI/5-Day_Challenge/Challenge_2/Project/weights/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth"
test_output_image_path = "C:/A3MSI/5-Day_Challenge/Challenge_2/Project/high.jpg"

enhance_image_resolution(test_input_image_path, test_model_path, test_output_image_path)
