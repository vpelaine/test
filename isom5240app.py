import streamlit as st
from transformers import pipeline
from PIL import Image

# Title
st.title("Age Classification using ViT")

# Load the age classification pipeline
age_classifier = pipeline("image-classification",
                          model="nateraw/vit-age-classifier")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Classify age
    age_predictions = age_classifier(image)
    age_predictions = sorted(age_predictions, key=lambda x: x['score'], reverse=True)

    # Display results
    st.subheader("Predicted Age Range")
    st.write(f"Age range: {age_predictions[0]['label']}")

    # Show all predictions with scores
    st.write("All predictions:")
    for pred in age_predictions:
        st.write(f"{pred['label']}: {pred['score']:.4f}")

