# run `python utils/store_face_emb.py` from the FaceRecognitionModel directory

from dotenv import load_dotenv
import os
import sys
# Import modules from base directory
sys.path.insert(0, os.getcwd())

from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from tasks.face_recognition import FaceRecognitionModule


# Define the name for the face embedding
NAME = "Matthew"

# Load environment variables from the .env file
load_dotenv()

# Get the model paths
torch_model_path = os.getenv("face_recog_model")
model = FaceRecognitionModule.load_from_checkpoint(torch_model_path)
model.eval()

# Get your image
img_path = os.getenv("face_path")
img = Image.open(img_path).convert("RGB")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((112, 112)),
])

img = transform(img)

# Apply model
face_emb = model(img.unsqueeze(0).to(model.device))[0]  # Add batch dimension

# Save to .npy file
if os.path.exists(os.getenv("stored_face_embs")):
    face_embs = torch.load(os.getenv("stored_face_embs"))
else:
    face_embs = {}
    
face_embs[NAME] = face_emb.cpu().detach().numpy()
torch.save(face_embs, os.getenv("stored_face_embs"))