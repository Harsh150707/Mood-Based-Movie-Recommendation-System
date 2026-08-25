1.Organize Project Files:Place all assets in a single directory.Create a dedicated folder on your computer 
(e.g., MoodRecommender) and ensure the following three files are placed directly inside it:ml_engine.cpp 
(Your C++ Machine Learning source code)app_gui.py (Your Python OpenCV + GUI script)movies.csv (Your Kaggle dataset file)
2.Install Required Python Dependencies:Run in Terminal or Command Prompt.Open your terminal (macOS/Linux) or Command Prompt / PowerShell (Windows), 
navigate to your project folder, and run:Bashpip install opencv-python pillow
(Note: OpenCV handles webcam streaming, while Pillow manages image formatting inside Tkinter.)
3.Compile the C++ Machine Learning Engine:Creates the binary executable.Compile your C++ source code using g++
with the C++17 standard so it can be called by Python:Windows (Command Prompt / PowerShell):DOSg++ -std=c++17 ml_engine.cpp -o ml_engine.exe
macOS / Linux (Terminal):Bashg++ -std=c++17 ml_engine.cpp -o ml_engine
Verify that a new executable file (ml_engine.exe or ml_engine) appears in your folder.4.Execute the Application:Launches the unified UI dashboard.
Run the master Python script to open the desktop application interface:Bashpython app_gui.py
5.Test the System Pipeline:End-to-end execution check.A desktop window titled AI Mood-Based Movie Recommender will open with your live webcam stream on the left side.
Position your face in front of the camera.Click Analyze Mood & Recommend.The Python layer will capture your facial geometry, pass the extracted feature vector directly to
./ml_engine, process your movies.csv dataset, and render the top recommended movies on the right panel.
