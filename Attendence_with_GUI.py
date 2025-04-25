import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import ttk
import csv
import threading

IMG_PATH = 'imgs'
ATTENDANCE_FILE = 'Attendance.csv'
prompted_names = {}
COOLDOWN_TIME = timedelta(minutes=5)

def load_images(path):
    images = []
    names = []
    for file in os.listdir(path):
        img = cv2.imread(os.path.join(path, file))
        if img is not None:
            images.append(img)
            names.append(os.path.splitext(file)[0])
    return images, names

def find_encodings(images):
    encodings = []
    for img in images:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        enc = face_recognition.face_encodings(rgb)
        if enc:
            encodings.append(enc[0])
    return encodings

def mark_attendance(name):
    if not os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Time"])
    with open(ATTENDANCE_FILE, 'r+', newline='') as f:
        reader = csv.reader(f)
        lines = list(reader)
        nameList = [line[0] for line in lines[1:] if len(line) > 0]
        if name not in nameList:
            now = datetime.now().strftime('%H:%M:%S')
            writer = csv.writer(f)
            writer.writerow([name, now])

def show_attendance_popup(name):
    def close_popup():
        popup.destroy()
    popup = tk.Tk()
    popup.overrideredirect(True)
    popup.geometry("300x100+500+300")
    popup.configure(bg='black')
    label = tk.Label(
        popup,
        text=f"Attendance marked for {name}",
        font=("Helvetica", 14),
        bg='black',
        fg='limegreen'
    )
    label.pack(expand=True)
    popup.after(2000, close_popup)
    popup.lift()
    popup.attributes("-topmost", True)
    popup.after(10, lambda: popup.focus_force())
    popup.mainloop()
    return True

def popup_and_mark(name):
    if show_attendance_popup(name):
        mark_attendance(name)

def display_attendance():
    def on_closing():
        cap.release()
        cv2.destroyAllWindows()
        window.quit()
    window = tk.Tk()
    window.title("Attendance Table")
    tree = ttk.Treeview(window, columns=("Name", "Time"), show="headings")
    tree.heading("Name", text="Name")
    tree.heading("Time", text="Time")
    tree.pack(fill="both", expand=True)
    if os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, 'r') as f:
            reader = csv.reader(f)
            lines = list(reader)
            if len(lines) > 1:
                for row in lines[1:]:
                    if len(row) >= 2:
                        tree.insert("", "end", values=row)
    def refresh_table():
        for item in tree.get_children():
            tree.delete(item)
        with open(ATTENDANCE_FILE, 'r') as f:
            reader = csv.reader(f)
            lines = list(reader)
            if len(lines) > 1:
                for row in lines[1:]:
                    if len(row) >= 2:
                        tree.insert("", "end", values=row)
    refresh_button = tk.Button(window, text="Refresh", command=refresh_table)
    refresh_button.pack(pady=10)
    window.protocol("WM_DELETE_WINDOW", on_closing)
    window.mainloop()

images, classNames = load_images(IMG_PATH)
known_encodings = find_encodings(images)

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break
    small_img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    rgb_small = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)
    faces = face_recognition.face_locations(rgb_small)
    encs = face_recognition.face_encodings(rgb_small, faces)
    for encodeFace, faceLoc in zip(encs, faces):
        matches = face_recognition.compare_faces(known_encodings, encodeFace)
        faceDist = face_recognition.face_distance(known_encodings, encodeFace)
        matchIndex = np.argmin(faceDist)
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            current_time = datetime.now()
            if name not in prompted_names or current_time - prompted_names[name] > COOLDOWN_TIME:
                prompted_names[name] = current_time
                threading.Thread(target=popup_and_mark, args=(name,), daemon=True).start()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = [v * 4 for v in (y1, x2, y2, x1)]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.imshow("Webcam - Face Attendance", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('g'):
        display_attendance()

cap.release()
cv2.destroyAllWindows()
