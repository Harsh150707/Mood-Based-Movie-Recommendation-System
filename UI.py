import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import subprocess
import os

class MoodMovieApp:
    def __init__(self, window):
        self.window = window
        self.window.title("AI Mood-Based Movie Recommender")
        self.window.geometry("900x600")

        # Layout Split: Left = Webcam Feed, Right = C++ ML Output
        self.left_frame = tk.Frame(self.window, width=450, height=600, bg="#1e1e1e")
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.right_frame = tk.Frame(self.window, width=450, height=600, bg="#2d2d2d")
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # 1. OpenCV Video Widget
        self.video_label = tk.Label(self.left_frame, bg="#1e1e1e")
        self.video_label.pack(pady=20)

        self.btn_capture = tk.Button(
            self.left_frame, 
            text="Analyze Mood & Recommend", 
            font=("Arial", 12, "bold"),
            bg="#007acc", 
            fg="white",
            command=self.process_frame_and_recommend
        )
        self.btn_capture.pack(pady=10)

        # 2. UI Recommendation Header
        self.results_header = tk.Label(
            self.right_frame, 
            text="C++ Recommended Movies", 
            font=("Arial", 16, "bold"), 
            bg="#2d2d2d", 
            fg="white"
        )
        self.results_header.pack(pady=20)

        self.results_box = tk.Text(self.right_frame, font=("Consolas", 11), bg="#1e1e1e", fg="#00ff00", width=45, height=20)
        self.results_box.pack(padx=10, pady=10)

        # 3. Start Camera Loop
        self.cap = cv2.VideoCapture(0)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.current_features = [0.5, 0.5, 0.2] # Default [Energy, Valence, Suspense]

        self.update_webcam()

    def update_webcam(self):
        ret, frame = self.cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                aspect_ratio = float(w) / h
                
                # Derive raw feature metrics from facial geometry
                if aspect_ratio > 1.0:
                    self.current_features = [0.2, 0.9, 0.1] # High Valence (Smile expansion)
                    mood_text = "Mood: Happy"
                else:
                    self.current_features = [0.8, 0.2, 0.7] # High Energy / Suspense
                    mood_text = "Mood: Focused / Intense"

                cv2.putText(frame, mood_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Convert frame for Tkinter Display
            cv2_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2_img)
            img = img.resize((400, 300))
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        self.window.after(10, self.update_webcam)

    def process_frame_and_recommend(self):
        # --- THE INTEGRATION BRIDGE ---
        # 1. Take current extracted feature vector from OpenCV
        e, v, s = self.current_features

        # 2. Trigger compiled C++ binary using subprocess
        executable = "./ml_engine.exe" if os.name == 'nt' else "./ml_engine"
        
        try:
            result = subprocess.run(
                [executable, str(e), str(v), str(s)], 
                capture_output=True, 
                text=True, 
                check=True
            )
            
            # 3. Parse output string from C++ engine and update GUI
            self.results_box.delete("1.0", tk.END)
            self.results_box.insert(tk.END, f"Input Features: E={e}, V={v}, S={s}\n")
            self.results_box.insert(tk.END, "="*40 + "\n\n")

            lines = result.stdout.strip().split("\n")
            for idx, line in enumerate(lines, 1):
                if "|" in line:
                    title, genre, rating = line.split("|")
                    self.results_box.insert(tk.END, f"{idx}. {title}\n   Genre: {genre}\n   Rating: {rating}/10\n\n")

        except Exception as err:
            self.results_box.delete("1.0", tk.END)
            self.results_box.insert(tk.END, f"Error triggering C++ ML Core:\n{err}")

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = MoodMovieApp(root)
    root.mainloop()