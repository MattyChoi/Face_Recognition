from dotenv import load_dotenv
import os

import cv2
import numpy as np
import torch
import onnxruntime

# Load environment variables from the .env file
load_dotenv()

# Load the pre-trained model
model_file = os.getenv("face_bb_model")
config_file = os.getenv("face_prototxt")
net = cv2.dnn.readNetFromCaffe(config_file, model_file)

# Load the ONNX face recognition model
onnx_model_path = os.getenv("face_recog_onnx_model")
onnx_session = onnxruntime.InferenceSession(onnx_model_path)

# Load the stored face_embeddings and labels
face_dict = torch.load(os.getenv("stored_face_embs"))
assert face_dict is not None, "No face embeddings found"
face_labels, face_embs = zip(*face_dict.items())
face_embs = np.stack(face_embs, axis=0)

# Preprocessing helper for face recognition
def preprocess_face(face_img):
    # Change from BGR to RGB color format
    # rgb_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    # Resize the face image to the input size expected by the model (e.g., 112x112)
    resized_img = cv2.resize(face_img, (112, 112))
    # Normalize pixel values to [0, 1] range
    normalized_img = resized_img.astype(np.float32) / 255.0
    # Rearrange dimensions to match (batch_size, channels, height, width)
    input_tensor = np.transpose(normalized_img, (2, 0, 1))
    input_tensor = np.expand_dims(input_tensor, axis=0)  # Add batch dimension
    return input_tensor


# Face recognition function
def face_recognition(face_img):
    # run the onnx model
    input_tensor = preprocess_face(face_img)
    input_name = onnx_session.get_inputs()[0].name
    outputs = onnx_session.run(None, {input_name: input_tensor})
    emb = outputs[0][0]

    # calculate the distances between the stored embeddings with the current embedding
    distances = np.linalg.norm(face_embs - emb, axis=1)
    closest_idx = np.argmin(distances)
    print(f"Closest match: {face_labels[closest_idx]} with distance {distances[closest_idx]}")
    if distances[closest_idx] < 0.6:  # Example threshold
        return face_labels[closest_idx]
        
    return "Unknown"


def main():
    # Open the default camera (camera index 0)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open the camera.")
        exit()

    while True:
        # Read a frame from the video feed
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture video.")
            break

        # Get the frame dimensions
        h, w = frame.shape[:2]

        # Preprocess the frame
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
        net.setInput(blob)

        # Perform face detection
        detections = net.forward()

        # Loop through detections
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]  # Confidence score

            # Filter out weak detections
            if confidence > 0.5:
                # Get the bounding box coordinates
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Draw the bounding box with confidence
                text = f"{confidence:.2f}"
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)
                # cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                # Apply face reconigition model
                face = frame[startY:endY, startX:endX]
                person = face_recognition(face)
                
                if person:
                    cv2.putText(frame, person, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

        # Display the resulting frame
        cv2.imshow("Face Detection", frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture and close windows
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()