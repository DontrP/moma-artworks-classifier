#########################
# ARTWORKS CLASSIFIER
#########################

import streamlit as st
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from PIL import Image
import os
import io

#########################
# SETTINGS
#########################
model_file = "models/momaclassifier_resnet50.pt"
image_folder = "demo_images"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_index = {"Drawing": 0, "Photograph": 1, "Print": 2}


#########################
# MODEL LOADING
#########################
@st.cache_resource
def load_model():
    num_class = len(class_index)
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_class)
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.to(device)
    model.eval()
    return model


model = load_model()

#########################
# TRANSFORM PIPELINE
#########################
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


#########################
# PREDICTION FUNCTION
#########################
def predict_image_class(image_path, model, transform, class_index, device):
    try:
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image)
        input_batch = input_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_batch)

        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        _, predicted_class_index = torch.max(output, 1)

        index_to_class = {v: k for k, v in class_index.items()}
        predicted_class_name = index_to_class[predicted_class_index.item()]
        class_probabilities = {
            index_to_class[i]: prob.item() for i, prob in enumerate(probabilities)
        }

        return image, predicted_class_name, class_probabilities
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None, None


#########################
# UI
#########################
st.title("MoMA ARTWORKS CLASSIFIER")
st.write("Classify artworks into Drawing, Photograph, and Print")

# side panel
st.sidebar.header("Options")

# select from collections

# Choose from collection
available_images = [
    f for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

selected_image = None
if available_images:
    selected_image = st.sidebar.selectbox(
        "Choose from Collection", ["None"] + available_images
    )
else:
    st.sidebar.warning("No images found in the demo_images folder.")

# Upload image
uploaded_file = st.sidebar.file_uploader(
    "Or Upload Your Own Image", type=["jpg", "jpeg", "png"]
)

# prediction
if uploaded_file is not None:
    st.info("Using uploaded image for classification.")
    image, predicted_class, probabilities = predict_image_class(
        uploaded_file, model, transform, class_index, device
    )

elif selected_image and selected_image != "None":
    st.info(f"Using selected image: {selected_image}")
    image_path = os.path.join(image_folder, selected_image)
    image, predicted_class, probabilities = predict_image_class(
        image_path, model, transform, class_index, device
    )

else:
    image, predicted_class, probabilities = None, None, None

# show results
if image:
    st.image(image, caption="Input Image", width=300)
    st.subheader("Prediction Results (Probabilities)")
    for cls, prob in probabilities.items():
        st.write(f"{cls}: {prob:.3f}")

    st.success(f"Final Predicted Class: **{predicted_class}**")
else:
    st.warning("Please choose an image from the sidebar or upload your own.")
