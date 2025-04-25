# Face-Recognition
Face Recognition with attendence marking 
Real-time Face Recognition-based Attendance System built with Python. It uses your webcam to detect and recognize faces from a predefined image dataset and automatically marks attendance when a known face is detected. The system includes a cooldown timer to prevent duplicate entries and displays a popup when attendance is marked.

🛠️ Libraries Used:
OpenCV: For webcam access and face display.

face_recognition: For face detection and recognition using deep learning (built on dlib).

NumPy: For numerical operations.

Tkinter: For GUI popups and attendance table display.

CSV: For logging attendance records.

Threading: To handle non-blocking popups during real-time video processing.

Datetime: To handle timestamps and cooldown logic.

OS: For file handling and image loading.

⚙️ How It Works:
The system loads face images from the imgs/ directory and encodes them using the face_recognition library.

A webcam feed is opened using OpenCV and frames are processed in real-time.

Each detected face is encoded and compared to the known face encodings.

If a match is found:

The system checks if the person has already been marked present within the last 5 minutes (cooldown).

If not, a Tkinter popup confirms that attendance is marked and the name + time is stored in Attendance.csv.

You can press 'g' at any time to view the current attendance in a separate GUI table.

Press 'q' to exit the application.

⏱️ Cooldown Timer
To prevent spamming or repeated entries, the system uses a cooldown timer of 5 minutes (timedelta(minutes=5)). This ensures each person is marked present only once in that period.
