import cv2
import numpy as np

# Load the pre-trained model
model_file = "res10_300x300_ssd_iter_140000.caffemodel"
config_file = "deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(config_file, model_file)


# Load the face recognition model
def face_recognition(face_img):
    return None


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
                cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                # Apply face reconigition model
                face = frame[startY:endY, startX:endX]
                person = face_recognition(face)
                
                if person:
                    cv2.putText(frame, person, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

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