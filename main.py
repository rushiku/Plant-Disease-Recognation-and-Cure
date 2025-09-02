import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import requests

st.set_page_config(
    page_title="Plant Disease Recognition",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar
with st.sidebar:
    st.title("🌱 Plant Disease Dashboard")
    st.markdown("---")
    app_mode = st.selectbox("📄 Choose the page", ["Home", "About", "Disease Recognition"], index=0)
    st.markdown("---")
    st.info("Developed by Rushikesh Kulkarni 🌿")

# Load model once and cache it
@st.cache_resource
def load_model():
    # Initialize ResNet50 architecture
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 97)  # 97 classes

    # Dropbox direct download link
    model_url = "https://www.dropbox.com/scl/fi/dshwigc3i4e0ssrs56lsy/resnet50_plant_disease.pth?rlkey=1xqtgtllyzvjr460gr4y6fwsi&st=f8j52l2r&dl=1"
    model_path = "resnet50_plant_disease.pth"

    # Download if model doesn't exist
    if not os.path.exists(model_path):
        st.info("Downloading the model file. This may take a few minutes...")
        r = requests.get(model_url, stream=True)
        with open(model_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        st.success("Model downloaded successfully!")

    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# Image preprocessing for ResNet50
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image).convert('RGB')
    return preprocess(image).unsqueeze(0)  # Add batch dimension

# PyTorch model prediction
def model_prediction(test_image):
    input_tensor = preprocess_image(test_image)
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    return predicted.item(), confidence.item()

# Class names (97 classes)
class_name = [
    "Apple Blackrot", "Apple Cedar Apple Rust", "Apple Healthy", "Apple Scab", "Banana Healthy", "Banana Segatoka",
    "Banana Xamthomonas", "Basil Wilted", "Basil With Mildew", "Bean Angular Leaf Spot", "Bean Healthy", "Bean Rust",
    "Blueberry Healthy", "Brassica Black Rot", "Cassava Bacterial Blight", "Cassava Brown Streak Disease",
    "Cassava Green Mottle", "Cassava Healthy", "Cassava Mosaic Disease", "Cherry Healthy", "Cherry Powdery Mildew",
    "Chilli Healthy", "Chilli Leaf Curl", "Chilli Leaf Spot", "Chilli Whitefly", "Chilli Yellowish", "Citrus Black Spot",
    "Citrus Canker", "Citrus Greening", "Citrus Healthy", "Citrus Melanose", "Coffee Healthy", "Coffee Red Spider Mite",
    "Coffee Rust Level 1", "Coffee Rust Level 2", "Coffee Rust Level 3", "Corn Cercospora Leaf Spot", "Corn Common Rust",
    "Corn Healthy", "Corn Northern Leaf Blight", "Cotton Bacterial Blight", "Cotton Curl Virus", "Cotton Fussarium Wilt",
    "Cotton Healthy", "Grape Black Rot", "Grape Esca Black Measles", "Grape Healthy", "Grape Leaf Blight Isariopsis Leaf Spot",
    "Guava Canker", "Guava Dot", "Guava Healthy", "Guava Mummification", "Guava Rust", "Healthy Basil", "Healthy Coriander",
    "Kale With Spots", "Lettuce Anthracnose", "Lettuce Bacterial Spot", "Lettuce Downy Mildew", "Lettuce Soft Rot",
    "Mint Fusarium Wilt", "Mint Leaf Rust", "Orange Haunglongbing Citrus Greening", "Parsley Leaf Blight Disease",
    "Parsley Leaf Spot Disease", "Peach Bacterial Spot", "Peach Healthy", "Pepper Bell Bacterial Spot", "Pepper Bell Healthy",
    "Potato Early Blight", "Potato Healthy", "Potato Late Blight", "Powdery Mildew Mint Leaf", "Raspberry Healthy",
    "Rice Bacterial Leaf Blight", "Rice Brown Spot", "Rice Leaf Smut", "Soybean Healthy", "Squash Powdery Mildew",
    "Strawberry Healthy", "Strawberry Leaf Scorch", "Tea Leaf Blight", "Tea Red Leaf Spot", "Tea Red Scab",
    "Tomato Bacterial Spot", "Tomato Early Blight", "Tomato Healthy", "Tomato Late Blight", "Tomato Leaf Mold",
    "Tomato Mosaic Virus", "Tomato Septoria Leaf Spot", "Tomato Spider Mites Two Spotted Spider Mite", "Tomato Target Spot",
    "Tomato Yellow Leaf Curl Virus", "Wheat Healthy", "Wheat Septoria", "Wheat Stripe Rust"
]

# Cure / Precaution Suggestions (97 classes)

cures = {
    "Apple Blackrot": "Remove infected fruit and leaves, apply fungicides like captan or thiophanate-methyl.",
    "Apple Cedar Apple Rust": "Use resistant varieties, prune nearby junipers, and apply fungicide sprays.",
    "Apple Healthy": "No issues detected. Maintain regular watering, pruning, and pest control.",
    "Apple Scab": "Remove fallen leaves, apply sulfur-based fungicides.",
    "Banana Healthy": "No issues detected. Regular maintenance required.",
    "Banana Segatoka": "Apply fungicides, improve drainage and avoid waterlogging.",
    "Banana Xamthomonas": "Remove infected plants, disinfect tools, and apply copper-based sprays.",
    "Basil Wilted": "Remove infected plants, improve soil drainage.",
    "Basil With Mildew": "Apply fungicides and avoid overhead watering.",
    "Bean Angular Leaf Spot": "Remove infected leaves, use certified seeds.",
    "Bean Healthy": "No issues detected.",
    "Bean Rust": "Apply fungicides and rotate crops.",
    "Blueberry Healthy": "No issues detected.",
    "Brassica Black Rot": "Use resistant varieties, practice crop rotation.",
    "Cassava Bacterial Blight": "Remove infected stems, use clean cuttings.",
    "Cassava Brown Streak Disease": "Use virus-free planting material.",
    "Cassava Green Mottle": "Remove infected plants, use healthy cuttings.",
    "Cassava Healthy": "No issues detected.",
    "Cassava Mosaic Disease": "Use resistant varieties, remove infected plants.",
    "Cherry Healthy": "No issues detected.",
    "Cherry Powdery Mildew": "Apply fungicides and prune affected areas.",
    "Chilli Healthy": "No issues detected.",
    "Chilli Leaf Curl": "Control whitefly, use resistant varieties.",
    "Chilli Leaf Spot": "Remove infected leaves, apply copper fungicides.",
    "Chilli Whitefly": "Use yellow sticky traps, apply insecticides.",
    "Chilli Yellowish": "Check for nutrient deficiency and pests.",
    "Citrus Black Spot": "Remove infected fruits, apply fungicides.",
    "Citrus Canker": "Prune infected parts, use copper sprays.",
    "Citrus Greening": "Remove infected trees, control psyllid vectors.",
    "Citrus Healthy": "No issues detected.",
    "Citrus Melanose": "Apply fungicides and remove infected twigs.",
    "Coffee Healthy": "No issues detected.",
    "Coffee Red Spider Mite": "Apply miticides and improve humidity control.",
    "Coffee Rust Level 1": "Prune infected leaves, apply fungicides.",
    "Coffee Rust Level 2": "Apply copper-based fungicides regularly.",
    "Coffee Rust Level 3": "Consider removing heavily infected plants.",
    "Corn Cercospora Leaf Spot": "Rotate crops, apply fungicides.",
    "Corn Common Rust": "Use resistant varieties, apply fungicides.",
    "Corn Healthy": "No issues detected.",
    "Corn Northern Leaf Blight": "Use resistant hybrids, remove residue.",
    "Cotton Bacterial Blight": "Use certified seeds, apply copper sprays.",
    "Cotton Curl Virus": "Control whitefly, remove infected plants.",
    "Cotton Fussarium Wilt": "Use resistant varieties, rotate crops.",
    "Cotton Healthy": "No issues detected.",
    "Grape Black Rot": "Remove infected leaves and fruits, apply fungicides.",
    "Grape Esca Black Measles": "Remove diseased wood, improve air circulation.",
    "Grape Healthy": "No issues detected.",
    "Grape Leaf Blight Isariopsis Leaf Spot": "Apply fungicides and remove infected leaves.",
    "Guava Canker": "Remove infected parts, apply copper sprays.",
    "Guava Dot": "Apply fungicides and improve plant hygiene.",
    "Guava Healthy": "No issues detected.",
    "Guava Mummification": "Remove infected fruits, apply fungicides.",
    "Guava Rust": "Apply fungicides and prune infected leaves.",
    "Healthy Basil": "No issues detected.",
    "Healthy Coriander": "No issues detected.",
    "Kale With Spots": "Apply fungicides, remove infected leaves.",
    "Lettuce Anthracnose": "Use disease-free seeds, apply fungicides.",
    "Lettuce Bacterial Spot": "Remove infected plants, practice crop rotation.",
    "Lettuce Downy Mildew": "Apply fungicides and avoid overhead watering.",
    "Lettuce Soft Rot": "Improve drainage, avoid injury to plants.",
    "Mint Fusarium Wilt": "Remove infected plants, use clean cuttings.",
    "Mint Leaf Rust": "Apply fungicides, improve air circulation.",
    "Orange Haunglongbing Citrus Greening": "Remove infected trees, control psyllids.",
    "Parsley Leaf Blight Disease": "Apply fungicides, remove infected leaves.",
    "Parsley Leaf Spot Disease": "Remove infected leaves, improve plant hygiene.",
    "Peach Bacterial Spot": "Apply copper sprays, prune infected branches.",
    "Peach Healthy": "No issues detected.",
    "Pepper Bell Bacterial Spot": "Remove infected fruits, apply copper sprays.",
    "Pepper Bell Healthy": "No issues detected.",
    "Potato Early Blight": "Apply fungicides, rotate crops.",
    "Potato Healthy": "No issues detected.",
    "Potato Late Blight": "Apply fungicides, remove infected plants.",
    "Powdery Mildew Mint Leaf": "Apply fungicides and improve airflow.",
    "Raspberry Healthy": "No issues detected.",
    "Rice Bacterial Leaf Blight": "Use resistant varieties, apply copper sprays.",
    "Rice Brown Spot": "Apply fungicides and ensure balanced fertilization.",
    "Rice Leaf Smut": "Use disease-free seeds, apply fungicides.",
    "Soybean Healthy": "No issues detected.",
    "Squash Powdery Mildew": "Apply sulfur fungicides, ensure good airflow.",
    "Strawberry Healthy": "No issues detected.",
    "Strawberry Leaf Scorch": "Remove infected leaves, apply fungicides.",
    "Tea Leaf Blight": "Apply fungicides, prune infected parts.",
    "Tea Red Leaf Spot": "Apply fungicides, remove infected leaves.",
    "Tea Red Scab": "Apply fungicides and maintain proper spacing.",
    "Tomato Bacterial Spot": "Apply copper sprays, remove infected leaves.",
    "Tomato Early Blight": "Use resistant varieties, rotate crops, apply fungicides.",
    "Tomato Healthy": "No issues detected.",
    "Tomato Late Blight": "Apply fungicides immediately, remove infected plants.",
    "Tomato Leaf Mold": "Apply fungicides, avoid overhead watering.",
    "Tomato Mosaic Virus": "Use disease-free seeds, remove infected plants.",
    "Tomato Septoria Leaf Spot": "Remove infected leaves, apply fungicides.",
    "Tomato Spider Mites Two Spotted Spider Mite": "Use miticides, increase humidity.",
    "Tomato Target Spot": "Remove infected leaves, apply fungicides.",
    "Tomato Yellow Leaf Curl Virus": "Control whitefly, remove infected plants.",
    "Wheat Healthy": "No issues detected.",
    "Wheat Septoria": "Apply fungicides, rotate crops.",
    "Wheat Stripe Rust": "Use resistant wheat varieties, apply fungicides during early growth."
}


# ----------------- Pages -----------------

# Home Page
if app_mode == "Home":
    st.markdown("<h1 style='color:#228B22;'>🌿 Plant Disease Recognition System</h1>", unsafe_allow_html=True)
    st.image("home_page.jpeg", use_container_width=True)
    st.markdown("""
    <div style='font-size:18px;'>
    Welcome to the <b>Plant Disease Recognition System</b>!<br>
    <ul>
        <li>🌱 <b>Upload</b> a plant image on the <b>Disease Recognition</b> page.</li>
        <li>🔬 <b>Analyze</b> plant health using AI-powered detection.</li>
        <li>📊 <b>Get instant results</b> and recommendations.</li>
    </ul>
    <b>Why Choose Us?</b>
    <ul>
        <li>✅ <b>Accurate</b> deep learning model</li>
        <li>⚡ <b>Fast</b> and easy to use</li>
        <li>🖼️ <b>Modern</b> interface</li>
    </ul>
    <b>Get Started:</b> Go to <b>Disease Recognition</b> in the sidebar!
    </div>
    """, unsafe_allow_html=True)



# Disease Recognition Page
elif app_mode == "Disease Recognition":
    st.markdown("<h1 style='color:#228B22;'>🦠 Disease Recognition</h1>", unsafe_allow_html=True)
    st.markdown("Browse all possible disease classes, upload a clear image of a plant leaf, and detect possible diseases.")

    # Browse all classes
    st.markdown("#### 📚 Browse All Classes")
    st.selectbox("All possible disease classes", class_name, index=0, key="class_select")

    # Upload and preview section
    col1, col2 = st.columns([2, 1])
    with col1:
        test_image = st.file_uploader("📷 Upload a Plant Leaf Image", type=["jpg", "jpeg", "png"])
        predict_btn = st.button("🔍 Predict Disease", use_container_width=True)
    with col2:
        if test_image:
            st.markdown("##### Image Preview")
            st.image(test_image, caption="Uploaded Image", use_container_width=False, width=180)

# Prediction section
if 'test_image' in locals() and test_image and predict_btn:
    with st.spinner("Analyzing image..."):
        pred_idx, confidence = model_prediction(test_image)
    
    predicted_disease = class_name[pred_idx]
    
    st.success(f"🌱 **Prediction:** {predicted_disease}")
    st.progress(int(confidence * 100))
    st.info(f"Model confidence: **{confidence*100:.2f}%**")
    
    if predicted_disease in cures:
        st.warning(f"💡 Suggested Cure: {cures[predicted_disease]}\n\n"
                   "📝 Note: Even if the affected plant part (leaf, blossom, fruit, stem) differs, the disease and its cure are generally the same.")
    else:
        st.warning("💡 Cure information not available for this disease.")

elif 'test_image' in locals() and not test_image and predict_btn:
    st.warning("Please upload a plant leaf image to analyze.")

# Footer
st.markdown("""
    <hr>
    <center>
    <span style='font-size:15px;'>Made with 💚 by Rushikesh Kulkarni | 2025</span>
    </center>
    """, unsafe_allow_html=True)

