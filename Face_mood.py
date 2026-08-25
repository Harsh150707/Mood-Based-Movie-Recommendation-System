import cv2
import subprocess

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

print("Webcam Active. Face the camera and press 's' to analyze features, or 'q' to quit.")

# Default Feature Values: [Energy, Valence, Suspense]
extracted_features = [0.5, 0.5, 0.2]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Calculate bounding box aspect ratio as a raw feature metric
        aspect_ratio = float(w) / h
        
        # Simple feature vector mapping based on face bounding geometry
        # Energy, Valence (Happiness), Suspense
        if aspect_ratio > 1.0:
            extracted_features = [0.2, 0.9, 0.1] # High Valence (Smile expansion)
        else:
            extracted_features = [0.7, 0.2, 0.8] # High Energy / Suspense

    cv2.imshow("OpenCV Feature Capture", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        print(f"\nCaptured Feature Vector: {extracted_features}")
        break
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# --- TRIGGER C++ OOP MACHINE LEARNING MODEL ---
print("\n[Python] Passing raw feature vector to C++ ML engine...")
subprocess.run([
    "./ml_engine", 
    str(extracted_features[0]), 
    str(extracted_features[1]), 
    str(extracted_features[2])
])