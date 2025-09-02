import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

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
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 97)  # 91 classes
    model.load_state_dict(torch.load("resnet50_plant_disease.pth", map_location=torch.device('cpu')))
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

# Class names
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
# Cure / Precaution Suggestions for all 97 classes
cures = {
    "Apple Blackrot": "Remove infected fruit and leaves, apply fungicides like captan or thiophanate-methyl.",
    "Apple Cedar Apple Rust": "Use resistant varieties, prune nearby junipers, and apply fungicide sprays.",
    "Apple Healthy": "No issues detected. Maintain regular watering, pruning, and pest control.",
    "Apple Scab": "Apply fungicides early in the season, prune infected leaves, ensure good air circulation.",
    "Banana Healthy": "No issues detected. Ensure balanced fertilization and pest management.",
    "Banana Segatoka": "Remove affected leaves and use fungicides like mancozeb or chlorothalonil.",
    "Banana Xamthomonas": "Destroy infected plants, use clean planting material, control insect vectors.",
    "Basil Wilted": "Improve soil drainage, avoid overwatering, rotate crops.",
    "Basil With Mildew": "Improve air circulation, apply neem oil or potassium bicarbonate sprays.",
    "Bean Angular Leaf Spot": "Remove infected leaves, use disease-free seeds, apply copper fungicides.",
    "Bean Healthy": "No issues detected. Practice crop rotation and balanced fertilization.",
    "Bean Rust": "Use resistant varieties, apply sulfur or copper-based fungicides.",
    "Blueberry Healthy": "No issues detected. Ensure proper pruning and pest management.",
    "Brassica Black Rot": "Remove infected leaves, avoid overhead irrigation, rotate crops.",
    "Cassava Bacterial Blight": "Use certified disease-free planting material, apply copper-based bactericides.",
    "Cassava Brown Streak Disease": "No cure. Remove infected plants and control whitefly vectors.",
    "Cassava Green Mottle": "Monitor crops, use resistant varieties, remove infected plants.",
    "Cassava Healthy": "No issues detected. Follow good agronomic practices.",
    "Cassava Mosaic Disease": "Use resistant varieties and virus-free cuttings, remove infected plants.",
    "Cherry Healthy": "No issues detected. Ensure proper pruning and fertilization.",
    "Cherry Powdery Mildew": "Apply sulfur or potassium bicarbonate sprays, prune infected branches.",
    "Chilli Healthy": "No issues detected. Practice crop rotation and pest control.",
    "Chilli Leaf Curl": "Control whiteflies, use resistant varieties, remove infected plants.",
    "Chilli Leaf Spot": "Apply fungicides like mancozeb, remove infected leaves.",
    "Chilli Whitefly": "Use yellow sticky traps, apply neem oil, and remove heavily infested plants.",
    "Chilli Yellowish": "Ensure proper nutrients, avoid water stress, monitor for pests.",
    "Citrus Black Spot": "Remove infected fruit, apply copper fungicides during wet season.",
    "Citrus Canker": "Remove infected branches, apply copper sprays, avoid moving infected plants.",
    "Citrus Greening": "No cure. Remove infected trees, control psyllid vectors.",
    "Citrus Healthy": "No issues detected. Ensure proper fertilization and irrigation.",
    "Citrus Melanose": "Apply fungicides, prune infected branches, avoid overhead irrigation.",
    "Coffee Healthy": "No issues detected. Ensure proper shade and fertilization.",
    "Coffee Red Spider Mite": "Spray miticides, maintain proper shade and humidity.",
    "Coffee Rust Level 1": "Remove infected leaves, apply fungicides like copper oxychloride.",
    "Coffee Rust Level 2": "Prune infected areas, use resistant varieties, apply fungicides.",
    "Coffee Rust Level 3": "Remove heavily infected plants, apply recommended fungicides.",
    "Corn Cercospora Leaf Spot": "Apply fungicides, rotate crops, remove infected residues.",
    "Corn Common Rust": "Plant resistant hybrids, apply fungicides if severe.",
    "Corn Healthy": "No issues detected. Maintain crop rotation and fertilization.",
    "Corn Northern Leaf Blight": "Plant resistant hybrids, apply fungicides if necessary.",
    "Cotton Bacterial Blight": "Use certified seeds, remove infected plants, apply copper sprays.",
    "Cotton Curl Virus": "Control whiteflies, remove infected plants, use resistant varieties.",
    "Cotton Fussarium Wilt": "Use resistant varieties, rotate crops, remove infected plants.",
    "Cotton Healthy": "No issues detected. Follow proper irrigation and nutrient management.",
    "Grape Black Rot": "Remove infected fruit, apply fungicides like mancozeb or captan.",
    "Grape Esca Black Measles": "Prune infected canes, disinfect pruning tools.",
    "Grape Healthy": "No issues detected. Practice good pruning and irrigation.",
    "Grape Leaf Blight Isariopsis Leaf Spot": "Apply fungicides, remove infected leaves.",
    "Guava Canker": "Remove infected tissue, apply copper-based fungicides.",
    "Guava Dot": "Remove infected fruit, maintain proper pruning and sanitation.",
    "Guava Healthy": "No issues detected. Ensure irrigation and nutrient management.",
    "Guava Mummification": "Remove infected fruits, prune and disinfect trees.",
    "Guava Rust": "Apply copper sprays, prune infected branches.",
    "Healthy Basil": "No issues detected. Maintain proper watering and nutrition.",
    "Healthy Coriander": "No issues detected. Avoid waterlogging and ensure good sunlight.",
    "Kale With Spots": "Apply fungicides, remove infected leaves, improve air circulation.",
    "Lettuce Anthracnose": "Apply fungicides, rotate crops, remove infected leaves.",
    "Lettuce Bacterial Spot": "Use disease-free seeds, remove infected leaves.",
    "Lettuce Downy Mildew": "Apply fungicides, avoid overhead irrigation, maintain spacing.",
    "Lettuce Soft Rot": "Ensure proper drainage, avoid mechanical damage.",
    "Mint Fusarium Wilt": "Use disease-free planting, rotate crops, remove infected plants.",
    "Mint Leaf Rust": "Apply sulfur or copper-based fungicides, remove infected leaves.",
    "Orange Haunglongbing Citrus Greening": "No cure. Remove infected trees, control psyllid vectors.",
    "Parsley Leaf Blight Disease": "Remove infected leaves, apply fungicides, rotate crops.",
    "Parsley Leaf Spot Disease": "Apply copper sprays, remove infected tissue.",
    "Peach Bacterial Spot": "Remove infected fruit and branches, apply copper fungicides.",
    "Peach Healthy": "No issues detected. Ensure proper irrigation and pruning.",
    "Pepper Bell Bacterial Spot": "Use certified seeds, apply copper fungicides.",
    "Pepper Bell Healthy": "No issues detected. Practice crop rotation and pest control.",
    "Potato Early Blight": "Rotate crops, use fungicides like chlorothalonil or copper-based sprays.",
    "Potato Healthy": "No issues detected. Maintain soil fertility and proper irrigation.",
    "Potato Late Blight": "Remove infected plants, apply fungicides (mancozeb, cymoxanil).",
    "Powdery Mildew Mint Leaf": "Apply neem oil or potassium bicarbonate, improve air circulation.",
    "Raspberry Healthy": "No issues detected. Maintain irrigation and pruning.",
    "Rice Bacterial Leaf Blight": "Use resistant varieties, apply copper-based sprays.",
    "Rice Brown Spot": "Apply fungicides, maintain balanced nutrients.",
    "Rice Leaf Smut": "Use disease-free seeds, remove infected leaves.",
    "Soybean Healthy": "No issues detected. Ensure proper crop rotation and nutrient management.",
    "Squash Powdery Mildew": "Apply sulfur sprays, improve air circulation, avoid overhead watering.",
    "Strawberry Healthy": "No issues detected. Maintain proper soil and irrigation.",
    "Strawberry Leaf Scorch": "Remove infected leaves, apply copper fungicides.",
    "Tea Leaf Blight": "Prune infected branches, apply fungicides.",
    "Tea Red Leaf Spot": "Remove affected leaves, apply copper sprays.",
    "Tea Red Scab": "Apply fungicides, remove infected leaves.",
    "Tomato Bacterial Spot": "Use certified seeds, remove infected leaves, apply copper sprays.",
    "Tomato Early Blight": "Apply copper fungicides, avoid overhead watering.",
    "Tomato Healthy": "No issues detected. Ensure proper irrigation and pruning.",
    "Tomato Late Blight": "Remove infected plants immediately, use fungicides.",
    "Tomato Leaf Mold": "Remove infected leaves, improve air circulation, apply fungicides.",
    "Tomato Mosaic Virus": "No cure. Remove infected plants, disinfect tools, control aphids.",
    "Tomato Septoria Leaf Spot": "Remove infected leaves, apply fungicides like chlorothalonil.",
    "Tomato Spider Mites Two Spotted Spider Mite": "Spray miticides, maintain proper humidity and shade.",
    "Tomato Target Spot": "Apply copper-based fungicides, remove infected leaves.",
    "Tomato Yellow Leaf Curl Virus": "Control whiteflies, remove infected plants, use resistant varieties.",
    "Wheat Healthy": "No issues detected. Ensure proper fertilization and irrigation.",
    "Wheat Septoria": "Rotate crops, remove infected residues, apply fungicides.",
    "Wheat Stripe Rust": "Use resistant wheat varieties, apply fungicides during early growth."
}

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

# About Page
elif app_mode == "About":
    st.markdown("<h1 style='color:#228B22;'>About</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:17px;'>
    <b>Dataset:</b><br>
    - 88,000+ RGB images of healthy and diseased crop leaves<br>
    - 97 different classes<br>
    - 80/20 train-validation split<br>
    - 33 test images for prediction<br>
    <br>
    <b>Model:</b><br>
    - Trained on 88,327 images<br>
    - Validated on 25,682 images<br>
    - Based on <b>ResNet50</b> deep learning architecture<br>
    <br>
    <b>Source:</b> <a href="https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset?resource=download" target="_blank">Kaggle Plant Disease Dataset</a>
    </div>
    """, unsafe_allow_html=True)

# Disease Recognition Page
elif app_mode == "Disease Recognition":
    st.markdown("<h1 style='color:#228B22;'>🦠 Disease Recognition</h1>", unsafe_allow_html=True)
    st.markdown("Browse all possible disease classes, upload a clear image of a plant leaf, and detect possible diseases.")

    # 1. Browse all classes (on top)
    st.markdown("#### 📚 Browse All Classes")
    st.selectbox("All possible disease classes", class_name, index=0, key="class_select")

    # 2. Upload and preview section (side by side)
    col1, col2 = st.columns([2, 1])
    with col1:
        test_image = st.file_uploader("📷 Upload a Plant Leaf Image", type=["jpg", "jpeg", "png"])
        predict_btn = st.button("🔍 Predict Disease", use_container_width=True)
    with col2:
        if test_image:
            st.markdown("##### Image Preview")
            st.image(test_image, caption="Uploaded Image", use_container_width=False, width=180)

# 3. Prediction section
if test_image and predict_btn:
    with st.spinner("Analyzing image..."):
        pred_idx, confidence = model_prediction(test_image)
    
    predicted_disease = class_name[pred_idx]
    
    # Show prediction
    st.success(f"🌱 **Prediction:** {predicted_disease}")
    st.progress(int(confidence * 100))
    st.info(f"Model confidence: **{confidence*100:.2f}%**")
    
    # Show Cure Suggestion with note
    if predicted_disease in cures:
        st.warning(f"💡 Suggested Cure: {cures[predicted_disease]}\n\n"
                   "📝 Note: Even if the affected plant part (leaf, blossom, fruit, stem) differs, the disease and its cure are generally the same.")
    else:
        st.warning("💡 Cure information not available for this disease.")

elif not test_image and predict_btn:
    st.warning("Please upload a plant leaf image to analyze.")
    st.warning("Please upload a plant leaf image to analyze.")


# Footer
st.markdown("""
    <hr>
    <center>
    <span style='font-size:15px;'>Made with 💚 by Rushikesh Kulkarni | 2025</span>
    </center>
    """, unsafe_allow_html=True)
