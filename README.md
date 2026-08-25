1. **Facial Feature Extraction:** OpenCV processes the webcam stream to track bounding boxes and aspect ratios, generating a feature vector representing `[Energy, Valence, Suspense]`.
2. **Inter-Process Communication (IPC):** The Python GUI passes these numerical inputs as command-line arguments to the compiled C++ binary.
3. **C++ ML Inference:** The native C++ program loads the `movies.csv` dataset, maps movie attributes into continuous vector space, and calculates Euclidean distances using a custom KNN algorithm.
4. **Interactive GUI Output:** The resulting top recommendations are parsed and rendered dynamically in the Tkinter UI panel.

---

## 🛠 Features & Highlights

* **Pure C++ ML Implementation:** Built without third-party ML libraries (like scikit-learn or TensorFlow) to demonstrate fundamental data structures, Euclidean distance equations, and memory management.
* **Object-Oriented Design (OOP):** Modular architecture utilizing custom C++ classes (`FeatureVector`, `Movie`, `KNNClassifier`, `KaggleCSVLoader`).
* **Real-Time Vision:** Haar Cascade classification via OpenCV to extract mood parameters from live video frames.
* **Cross-Language Integration:** Seamless subprocess bridging between Python (front-end/vision) and C++ (high-performance analytics core).

---

## 📂 Repository Structure

```text
.
├── ml_engine.cpp        # C++ Machine Learning Core (KNN implementation)
├── app_gui.py           # Python Tkinter GUI & OpenCV video processing
├── movies.csv           # Movie inventory dataset
└── README.md            # Project documentation
