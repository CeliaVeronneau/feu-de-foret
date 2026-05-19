
# TODO : recopier le code precedent et completer la logique de prediction
# Pensez a charger le modele avec @st.cache_resource pour ne le charger qu'une fois


import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageClassification

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ── Configuration de la page ──
st.set_page_config(
    page_title="Detecteur de feux de foret",
    page_icon="🔥",
    layout="wide"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Charger le modele une seule fois ──
@st.cache_resource
def load_model():
    model = AutoModelForImageClassification.from_pretrained("fire_model")
    model.to(device)
    model.eval()
    return model

model = load_model()

# Classes du modele : 0 = fire, 1 = no_fire
class_names = ["fire", "no_fire"]

# ── Transform d'evaluation, sans augmentation ──
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ── Wrapper HuggingFace pour GradCAM ──
class HFWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(pixel_values=x).logits

wrapped_model = HFWrapper(model).to(device)
wrapped_model.eval()

# GradCAM a besoin des gradients sur le backbone
for param in model.efficientnet.parameters():
    param.requires_grad = True

target_layers = [model.efficientnet.encoder.blocks[-1].projection.project_conv]

# ── Sidebar ──
st.sidebar.title("A propos")
st.sidebar.info(
    "Systeme d'aide a la detection — toute alerte doit etre confirmee par un operateur humain avant mobilisation."
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Modele :** EfficientNet-B0")
st.sidebar.markdown("**Classes :** fire, no_fire")

# ── Page principale ──
st.title("Detecteur de feux de foret")
st.markdown(
    "Uploadez une image pour obtenir une prediction "
    "avec score de confiance et visualisation GradCAM."
)

# ── Upload d'image ──
uploaded_file = st.file_uploader(
    "Choisir une image",
    type=["jpg", "jpeg", "png"],
    help="Formats acceptes : JPG, JPEG, PNG"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Image originale")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Analyse")

        with st.spinner("Analyse en cours..."):
            # 1. Appliquer val_transform a l'image
            input_tensor = val_transform(image).unsqueeze(0).to(device)

            # 2. Predire avec le modele
            with torch.no_grad():
                logits = wrapped_model(input_tensor)

            # 3. Extraire les probabilites avec softmax
            probs = torch.softmax(logits, dim=1)[0]
            pred_idx = torch.argmax(probs).item()
            pred_class = class_names[pred_idx]
            confidence = probs[pred_idx].item() * 100

            # 4. Afficher la prediction avec st.metric
            st.metric(
                label="Prediction",
                value=pred_class,
                delta=f"{confidence:.1f}%"
            )
            st.progress(int(confidence))

            st.markdown("**Probabilites :**")
            st.write(f"- fire : {probs[0].item() * 100:.1f}%")
            st.write(f"- no_fire : {probs[1].item() * 100:.1f}%")

            # 5. Generer et afficher le GradCAM
            resized_img = image.resize((224, 224))
            rgb_img = np.array(resized_img).astype(np.float32) / 255.0

            cam = GradCAM(
                model=wrapped_model,
                target_layers=target_layers
            )

            targets = [ClassifierOutputTarget(pred_idx)]
            grayscale_cam = cam(
                input_tensor=input_tensor,
                targets=targets
            )[0]

            gradcam_overlay = show_cam_on_image(
                rgb_img,
                grayscale_cam,
                use_rgb=True
            )

            st.image(
                gradcam_overlay,
                caption="Heatmap GradCAM",
                use_container_width=True
            )

    # ── Disclaimer en bas ──
    st.markdown("---")
    st.error(
        "ALERTE : ce systeme est une aide a la detection. Toute prediction doit etre confirmee par un operateur humain avant mobilisation."
    )

else:
    st.info("Uploadez une image pour commencer l'analyse.")

with st.expander("Comment ca marche ?"):
    st.markdown(
        """
        Cette application utilise un modele EfficientNet-B0 entraine pour distinguer deux classes :
        **fire** et **no_fire**.

        1. L'image uploadée est redimensionnée en 224x224 pixels.
        2. Le modele calcule une probabilité pour chaque classe.
        3. La classe avec la probabilité la plus élevée est affichée comme prédiction.
        4. GradCAM met en évidence les zones de l'image qui ont le plus influencé la décision.

        Important : ce système peut se tromper, notamment avec des couchers de soleil,
        de la brume, de la fumée ambiguë ou des lumières chaudes.
        """
    )
