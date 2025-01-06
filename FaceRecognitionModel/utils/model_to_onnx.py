# run `python utils/model_to_onnx.py` from the FaceRecognitionModel directory

from dotenv import load_dotenv
import os
import sys
# Import modules from base directory
sys.path.insert(0, os.getcwd())

import torch
from tasks.face_recognition import FaceRecognitionModule


# Load environment variables from the .env file
load_dotenv()

# Get the model paths
torch_model_path = os.getenv("face_recog_model")
onnx_model_path = os.getenv("face_recog_onnx_model")

model = FaceRecognitionModule.load_from_checkpoint(torch_model_path)

dummy_input = torch.randn(1, 3, 112, 112)
model.to_onnx(onnx_model_path, dummy_input, export_params=True)