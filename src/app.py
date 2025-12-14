import streamlit as st
import torch
import torch.nn.functional as f
from torchvision import transforms
from PIL import Image
from model import CatDogCNN  # Import our architecture

# CONFIGURATION
MODEL_PATH = "../models/cat_dog_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = ["Cat", "Dog"]


# --- 1. Load the Model (Cached) ---
# We use @st.cache_resource so Streamlit loads the model once
# and keeps it in memory. This makes the app much faster.
@st.cache_resource
def load_model():
	model = CatDogCNN().to(DEVICE)
	
	# Load the weights we trained
	# map_location ensures it works even if you switch from GPU to CPU
	model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
	model.eval()  # Set to evaluation mode
	return model


# --- 2. Preprocessing (The Translation Layer) ---
def process_image(image):
	"""
	Takes a PIL image and makes it ready for the model.
	Must match the 'Validation' transforms from dataset.py!
	"""
	transform = transforms.Compose([
		transforms.Resize((128, 128)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
	])
	return transform(image).unsqueeze(0)  # Add batch dimension (1, 3, 128, 128)


# --- 3. The Streamlit Interface ---
st.title("🐱 Cat vs Dog Classifier 🐶")
st.write("Upload an image, and the AI will tell you what it sees!")

# File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
	# Display the image to the user
	image = Image.open(uploaded_file).convert('RGB')
	st.image(image, caption='Uploaded Image', use_column_width=True)
	
	# Load model
	model = load_model()
	
	# Predict button
	if st.button('Classify This Image'):
		with st.spinner('Thinking...'):
			# Process and Predict
			img_tensor = process_image(image).to(DEVICE)
			
			with torch.no_grad():
				output = model(img_tensor)
				# Convert raw scores to probabilities (percentages)
				probs = f.softmax(output, dim=1)
				
				# Get the winner
				confidence, predicted_class = torch.max(probs, 1)
			
			# Display Result
			class_name = CLASS_NAMES[predicted_class.item()]
			confidence_score = confidence.item() * 100
			
			if class_name == "Dog":
				st.success(f"🐶 It's a **DOG**! ({confidence_score:.1f}% sure)")
			else:
				st.success(f"🐱 It's a **CAT**! ({confidence_score:.1f}% sure)")